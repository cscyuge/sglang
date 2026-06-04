import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import WanS2VPipelineConfig
from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VStreamR1AttentionLayout,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VConditionBundle,
    WanS2VStreamR1AttentionRequest,
    WanS2VStreamR1CacheMetadata,
    WanS2VStreamR1CacheState,
    WanS2VStreamR1DenoisingStage,
    _has_negative_prompt_embeds,
)


class TestWanS2VSamplingParams(unittest.TestCase):
    def test_num_output_latent_frames_sets_pixel_frames(self):
        params = WanS2VSamplingParams(num_output_latent_frames=21)

        self.assertEqual(params.num_frames, 81)

    def test_stream_r1_block_values_are_validated(self):
        with self.assertRaisesRegex(ValueError, "num_frame_per_block"):
            WanS2VSamplingParams(num_frame_per_block=0)

        with self.assertRaisesRegex(ValueError, "sink_size"):
            WanS2VSamplingParams(sink_size=-1)

        with self.assertRaisesRegex(ValueError, "denoising_steps"):
            WanS2VSamplingParams(denoising_steps=[])

    def test_stream_r1_kv_cache_is_forwarded_in_extra(self):
        params = WanS2VSamplingParams(
            stream_r1_kv_cache=True,
            num_frame_per_block=7,
            local_attn_size=9,
            sink_size=3,
        )

        extra = params.build_request_extra()

        self.assertTrue(extra["stream_r1_kv_cache"])
        self.assertEqual(extra["local_attn_size"], 9)
        self.assertEqual(extra["sink_size"], 3)

    def test_stream_r1_attention_values_are_cross_validated(self):
        with self.assertRaisesRegex(ValueError, "local_attn_size"):
            WanS2VSamplingParams(
                stream_r1_kv_cache=True,
                num_frame_per_block=7,
                local_attn_size=3,
            )

        with self.assertRaisesRegex(ValueError, "sink_size"):
            WanS2VSamplingParams(local_attn_size=3, sink_size=3)

    def test_pipeline_config_validates_kv_attention_shape(self):
        with self.assertRaisesRegex(ValueError, "local_attn_size"):
            WanS2VPipelineConfig(
                stream_r1_kv_cache=True,
                num_frame_per_block=7,
                local_attn_size=3,
            )


class TestWanS2VConditionBundle(unittest.TestCase):
    def _bundle(self) -> WanS2VConditionBundle:
        cond_states = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3, 1, 1)
        return WanS2VConditionBundle(
            prompt_embeds=torch.zeros(1, 4, 8),
            ref_latents=torch.ones(1, 2, 1, 1, 1),
            motion_latents=torch.ones(1, 2, 2, 1, 1),
            cond_states=cond_states,
            audio_input=torch.ones(1, 25, 1024, 12),
            audio_emb={"cache_key": "audio"},
            motion_frames=(73, 19),
            control_policy="lookahead",
            audio_metadata={"source": "unit"},
        )

    def test_slice_pads_cond_states_and_preserves_audio(self):
        bundle = self._bundle()

        chunk = bundle.slice(start=2, frames=3, policy="offline")

        self.assertEqual(chunk.cond_states.shape, (1, 2, 3, 1, 1))
        torch.testing.assert_close(
            chunk.cond_states[:, :, :1], bundle.cond_states[:, :, 2:3]
        )
        torch.testing.assert_close(
            chunk.cond_states[:, :, 1:], torch.zeros(1, 2, 2, 1, 1)
        )
        self.assertIs(chunk.audio_input, bundle.audio_input)
        self.assertIs(chunk.audio_emb, bundle.audio_emb)
        self.assertEqual(chunk.chunk_start, 2)
        self.assertEqual(chunk.chunk_frames, 3)
        self.assertEqual(chunk.control_policy, "offline")
        self.assertEqual(chunk.audio_metadata["chunk_start"], 2)
        self.assertEqual(chunk.audio_metadata["source"], "unit")

    def test_slice_rejects_invalid_ranges(self):
        bundle = self._bundle()

        with self.assertRaisesRegex(ValueError, "start"):
            bundle.slice(start=-1, frames=1)

        with self.assertRaisesRegex(ValueError, "frames"):
            bundle.slice(start=0, frames=0)


class TestWanS2VStreamR1AttentionLayout(unittest.TestCase):
    def test_no_kv_mask_limits_noisy_tokens_to_sink_local_and_condition(self):
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=8,
            total_seq_len=10,
            frame_seq_length=2,
            num_frame_per_block=2,
            local_attn_size=2,
            sink_size=1,
        )

        mask = layout.build_no_kv_attention_mask(torch.device("cpu"))[0]

        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(
            torch.nonzero(mask[0], as_tuple=False).flatten().tolist(),
            [0, 1, 2, 3, 8, 9],
        )
        self.assertEqual(
            torch.nonzero(mask[4], as_tuple=False).flatten().tolist(),
            [0, 1, 4, 5, 6, 7, 8, 9],
        )
        self.assertTrue(mask[8].all())

    def test_no_kv_mask_uses_global_current_start_for_later_chunks(self):
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=4,
            total_seq_len=6,
            frame_seq_length=2,
            num_frame_per_block=2,
            local_attn_size=2,
            sink_size=1,
            current_start=4,
        )

        mask = layout.build_no_kv_attention_mask(torch.device("cpu"))[0]

        self.assertEqual(
            torch.nonzero(mask[0], as_tuple=False).flatten().tolist(),
            [0, 1, 2, 3, 4, 5],
        )

    def test_layout_rejects_unaligned_current_start(self):
        with self.assertRaisesRegex(ValueError, "frame-aligned"):
            WanS2VStreamR1AttentionLayout(
                noisy_seq_len=4,
                total_seq_len=4,
                frame_seq_length=2,
                num_frame_per_block=2,
                local_attn_size=2,
                sink_size=0,
                current_start=1,
            )


class TestWanS2VStreamR1DenoisingStage(unittest.TestCase):
    def _stage(self) -> WanS2VStreamR1DenoisingStage:
        stage = WanS2VStreamR1DenoisingStage.__new__(WanS2VStreamR1DenoisingStage)
        stage.transformer = SimpleNamespace(
            config=SimpleNamespace(arch_config=SimpleNamespace(num_layers=2)),
            blocks=[
                SimpleNamespace(local_num_heads=3, dim_head=8),
                SimpleNamespace(local_num_heads=3, dim_head=8),
            ],
            hidden_size=48,
            num_attention_heads=6,
            use_context_parallel=False,
            set_stream_r1_attention=lambda *args, **kwargs: None,
        )
        stage.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000)
        )
        stage.cache_state = WanS2VStreamR1CacheState.disabled()
        stage.log_info = lambda *args, **kwargs: None
        stage._s2v_kv_attention_kernel_supported = False
        return stage

    def test_negative_prompt_embeds_presence_avoids_tensor_truthiness(self):
        self.assertTrue(
            _has_negative_prompt_embeds(
                SimpleNamespace(negative_prompt_embeds=torch.ones(2, 2))
            )
        )
        self.assertTrue(
            _has_negative_prompt_embeds(
                SimpleNamespace(negative_prompt_embeds=[torch.ones(2, 2)])
            )
        )
        self.assertFalse(
            _has_negative_prompt_embeds(SimpleNamespace(negative_prompt_embeds=[]))
        )
        self.assertFalse(
            _has_negative_prompt_embeds(SimpleNamespace(negative_prompt_embeds=None))
        )

    def test_attention_request_validates_block_local_sink_and_context(self):
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=False,
            num_frame_per_block=4,
            local_attn_size=4,
            sink_size=1,
            context_noise=1001,
        )

        with self.assertRaisesRegex(ValueError, "context_noise"):
            request.validate(latent_frames=8, train_timesteps=1000)

        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=False,
            num_frame_per_block=4,
            local_attn_size=3,
            sink_size=1,
            context_noise=0,
        )
        with self.assertRaisesRegex(ValueError, "local_attn_size"):
            request.validate(latent_frames=8, train_timesteps=1000)

    def test_kv_cache_state_allocates_and_resets_typed_blocks(self):
        metadata = WanS2VStreamR1CacheMetadata(
            batch_size=2,
            num_layers=2,
            frame_seq_length=5,
            local_num_attention_heads=3,
            attention_head_dim=8,
            local_attn_size=4,
            sink_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        state = WanS2VStreamR1CacheState.allocate(metadata)
        state.kv_cache[0]["global_end_index"].fill_(7)
        state.kv_cache[1]["local_end_index"].fill_(9)
        state.reset()

        self.assertTrue(state.enabled)
        self.assertTrue(state.allocated)
        self.assertEqual(state.kv_cache[0]["k"].shape, (2, 20, 3, 8))
        self.assertEqual(state.kv_cache[0]["v"].shape, (2, 20, 3, 8))
        self.assertEqual(state.kv_cache[0]["global_end_index"].item(), 0)
        self.assertEqual(state.kv_cache[1]["local_end_index"].item(), 0)

    def test_stage_prepares_metadata_but_guards_unimplemented_kv(self):
        stage = self._stage()
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=True,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        state = stage._prepare_cache_state(
            request=request,
            batch_size=1,
            frame_seq_length=10,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )

        self.assertTrue(state.enabled)
        self.assertFalse(state.allocated)
        self.assertEqual(state.metadata.cache_tokens, 60)
        self.assertEqual(state.metadata.local_num_attention_heads, 3)
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet"):
            stage._guard_cache_runtime(state)

    def test_configure_transformer_attention_passes_block_adapter_fields(self):
        stage = self._stage()
        calls = []
        stage.transformer.set_stream_r1_attention = (
            lambda *args, **kwargs: calls.append((args, kwargs))
        )
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=False,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        stage._configure_transformer_attention(request)

        self.assertEqual(
            calls,
            [((6, 1), {"num_frame_per_block": 4, "kv_cache": False})],
        )

    def test_kv_cache_rejects_context_parallel(self):
        stage = self._stage()
        stage.transformer.use_context_parallel = True
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=True,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        with self.assertRaisesRegex(NotImplementedError, "sequence/context"):
            stage._validate_stream_r1_parallel_compatibility(
                request,
                SimpleNamespace(
                    did_sp_shard_latents=False,
                    enable_sequence_shard=False,
                ),
            )

    def test_kv_cache_rejects_sequence_parallel_world_size(self):
        stage = self._stage()
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=True,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.wan_s2v._safe_sp_world_size",
            return_value=2,
        ):
            with self.assertRaisesRegex(NotImplementedError, "sequence/context"):
                stage._validate_stream_r1_parallel_compatibility(
                    request,
                    SimpleNamespace(
                        did_sp_shard_latents=False,
                        enable_sequence_shard=False,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
