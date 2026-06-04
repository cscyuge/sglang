import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import WanS2VPipelineConfig
from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VStreamR1AttentionLayout,
    WanS2VStreamR1NoisyKVCacheView,
    WanS2VStreamR1NoisyKVCacheUpdate,
    compose_wan_s2v_stream_r1_mixed_kv_view,
    split_wan_s2v_stream_r1_projected_kv,
    update_wan_s2v_stream_r1_noisy_kv_cache,
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

    def test_layout_derives_noisy_kv_update_capacity(self):
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=4,
            total_seq_len=6,
            frame_seq_length=2,
            num_frame_per_block=2,
            local_attn_size=3,
            sink_size=1,
            current_start=4,
        )

        update = layout.to_noisy_kv_cache_update(cache_start=0)

        self.assertEqual(layout.noisy_cache_tokens, 6)
        self.assertEqual(update.required_cache_tokens, 6)
        self.assertEqual(update.rolling_tokens, 4)
        self.assertEqual(update.current_start, 4)
        self.assertEqual(update.current_end, 8)


class TestWanS2VStreamR1ProjectedKVAdapters(unittest.TestCase):
    def _kv(self, seq_len: int, heads: int = 2):
        key = torch.arange(seq_len * heads, dtype=torch.float32).view(
            1, seq_len, heads, 1
        )
        value = key + 100
        return key, value

    def test_split_projected_kv_separates_noisy_and_condition_ranges(self):
        key, value = self._kv(seq_len=5)

        split = split_wan_s2v_stream_r1_projected_kv(
            key, value, noisy_seq_len=3
        )

        self.assertEqual(split.noisy_seq_len, 3)
        self.assertEqual(split.condition_seq_len, 2)
        self.assertEqual(split.total_seq_len, 5)
        torch.testing.assert_close(split.noisy_key, key[:, :3])
        torch.testing.assert_close(split.noisy_value, value[:, :3])
        torch.testing.assert_close(split.condition_key, key[:, 3:])
        torch.testing.assert_close(split.condition_value, value[:, 3:])

    def test_split_projected_kv_validates_shapes_and_noisy_range(self):
        key, value = self._kv(seq_len=5)

        with self.assertRaisesRegex(ValueError, "same shape"):
            split_wan_s2v_stream_r1_projected_kv(
                key, value[:, :4], noisy_seq_len=3
            )

        with self.assertRaisesRegex(ValueError, "positive"):
            split_wan_s2v_stream_r1_projected_kv(
                key, value, noisy_seq_len=0
            )

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            split_wan_s2v_stream_r1_projected_kv(
                key, value, noisy_seq_len=6
            )

    def test_compose_mixed_kv_appends_current_condition_after_cached_noisy(self):
        key, value = self._kv(seq_len=5)
        split = split_wan_s2v_stream_r1_projected_kv(
            key, value, noisy_seq_len=3
        )
        cached_key = torch.full((1, 4, 2, 1), -1.0)
        cached_value = torch.full((1, 4, 2, 1), -2.0)
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=7,
            local_end_index=4,
            local_start=3,
            local_end=7,
        )

        mixed = compose_wan_s2v_stream_r1_mixed_kv_view(noisy_view, split)

        self.assertEqual(mixed.key.shape, (1, 6, 2, 1))
        self.assertEqual(mixed.value.shape, (1, 6, 2, 1))
        self.assertEqual(mixed.cached_noisy_seq_len, 4)
        self.assertEqual(mixed.condition_seq_len, 2)
        self.assertEqual(mixed.condition_start_index, 4)
        self.assertEqual(mixed.total_seq_len, 6)
        self.assertEqual(mixed.global_end_index, 7)
        torch.testing.assert_close(mixed.key[:, :4], cached_key)
        torch.testing.assert_close(mixed.value[:, :4], cached_value)
        torch.testing.assert_close(mixed.key[:, 4:], split.condition_key)
        torch.testing.assert_close(mixed.value[:, 4:], split.condition_value)

    def test_compose_mixed_kv_accepts_no_condition_tokens(self):
        key, value = self._kv(seq_len=3)
        split = split_wan_s2v_stream_r1_projected_kv(
            key, value, noisy_seq_len=3
        )
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=key,
            value=value,
            global_end_index=3,
            local_end_index=3,
            local_start=0,
            local_end=3,
        )

        mixed = compose_wan_s2v_stream_r1_mixed_kv_view(noisy_view, split)

        self.assertIs(mixed.key, noisy_view.key)
        self.assertIs(mixed.value, noisy_view.value)
        self.assertEqual(mixed.condition_seq_len, 0)

    def test_compose_mixed_kv_validates_cached_and_current_dimensions(self):
        key, value = self._kv(seq_len=4)
        split = split_wan_s2v_stream_r1_projected_kv(
            key, value, noisy_seq_len=2
        )
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=torch.zeros(1, 2, 1, 1),
            value=torch.zeros(1, 2, 1, 1),
            global_end_index=2,
            local_end_index=2,
            local_start=0,
            local_end=2,
        )

        with self.assertRaisesRegex(ValueError, "batch/head"):
            compose_wan_s2v_stream_r1_mixed_kv_view(noisy_view, split)


class TestWanS2VStreamR1NoisyKVCacheUpdate(unittest.TestCase):
    def _cache(self, tokens: int):
        return {
            "k": torch.zeros(1, tokens, 1, 1),
            "v": torch.zeros(1, tokens, 1, 1),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }

    def _kv(self, start: int, length: int):
        key = torch.arange(start, start + length, dtype=torch.float32).view(
            1, length, 1, 1
        )
        value = key + 100
        return key, value

    def test_update_appends_then_replaces_same_noisy_block(self):
        cache = self._cache(tokens=6)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=2,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        key, value = self._kv(0, 4)

        view = update_wan_s2v_stream_r1_noisy_kv_cache(
            cache, key, value, update
        )

        self.assertEqual(view.global_end_index, 4)
        self.assertEqual(view.local_end_index, 4)
        torch.testing.assert_close(
            cache["k"][:, :4, 0, 0], torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        )
        torch.testing.assert_close(
            cache["v"][:, :4, 0, 0], torch.tensor([[100.0, 101.0, 102.0, 103.0]])
        )

        replacement_key = -key - 1
        replacement_value = replacement_key - 100
        view = update_wan_s2v_stream_r1_noisy_kv_cache(
            cache, replacement_key, replacement_value, update
        )

        self.assertEqual(view.global_end_index, 4)
        self.assertEqual(view.local_end_index, 4)
        torch.testing.assert_close(
            view.key[:, :, 0, 0], torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        )
        torch.testing.assert_close(
            view.value[:, :, 0, 0],
            torch.tensor([[-101.0, -102.0, -103.0, -104.0]]),
        )

    def test_update_preserves_sink_and_rolls_local_window(self):
        cache = self._cache(tokens=4)

        for current_start in (0, 3, 6):
            key, value = self._kv(current_start, 3)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=3,
                frame_seq_length=1,
                local_attn_size=4,
                sink_size=2,
                current_start=current_start,
            )
            view = update_wan_s2v_stream_r1_noisy_kv_cache(
                cache, key, value, update
            )

        self.assertEqual(view.global_end_index, 9)
        self.assertEqual(view.local_end_index, 4)
        self.assertEqual(view.local_start, 7)
        torch.testing.assert_close(
            view.key[:, :, 0, 0],
            torch.tensor([[0.0, 1.0, 7.0, 8.0]]),
        )
        torch.testing.assert_close(
            view.value[:, :, 0, 0],
            torch.tensor([[100.0, 101.0, 107.0, 108.0]]),
        )

    def test_update_supports_cache_start_offset(self):
        cache = self._cache(tokens=4)
        key, value = self._kv(4, 4)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=2,
            local_attn_size=2,
            sink_size=1,
            current_start=4,
            cache_start=4,
        )

        view = update_wan_s2v_stream_r1_noisy_kv_cache(
            cache, key, value, update
        )

        self.assertEqual(view.global_end_index, 8)
        self.assertEqual(view.local_end_index, 4)
        torch.testing.assert_close(
            view.key[:, :, 0, 0], torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        )

        key, value = self._kv(8, 4)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=2,
            local_attn_size=2,
            sink_size=1,
            current_start=8,
            cache_start=4,
        )
        view = update_wan_s2v_stream_r1_noisy_kv_cache(
            cache, key, value, update
        )

        self.assertEqual(view.global_end_index, 12)
        self.assertEqual(view.local_end_index, 4)
        self.assertEqual(view.local_start, 10)
        torch.testing.assert_close(
            view.key[:, :, 0, 0],
            torch.tensor([[4.0, 5.0, 10.0, 11.0]]),
        )

    def test_update_rejects_gaps_backwards_and_small_cache(self):
        cache = self._cache(tokens=3)
        key, value = self._kv(0, 2)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        gap_key, gap_value = self._kv(4, 2)
        gap_update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=4,
        )
        with self.assertRaisesRegex(ValueError, "skip"):
            update_wan_s2v_stream_r1_noisy_kv_cache(
                cache, gap_key, gap_value, gap_update
            )

        backward_update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        with self.assertRaisesRegex(ValueError, "backwards"):
            update_wan_s2v_stream_r1_noisy_kv_cache(
                cache, key[:, :1], value[:, :1], backward_update
            )

        small_cache = self._cache(tokens=2)
        with self.assertRaisesRegex(ValueError, "capacity"):
            update_wan_s2v_stream_r1_noisy_kv_cache(
                small_cache, key, value, update
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
