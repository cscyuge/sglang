import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import WanS2VPipelineConfig
from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.wan_s2v import (
    WanS2VTransformer3DModel,
    _build_s2v_noisy_rope_grid_sizes,
    _pad_stream_r1_attention_mask_for_sp,
    _rope_apply_precomputed,
    _rope_precompute,
    _rope_precompute_s2v_stream_r1_tensor_current_start,
    _segment_gate_add,
    _segment_modulate,
    _slice_s2v_audio_embeddings_with_tensor_start,
    rope_params,
)
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VStreamR1AttentionLayout,
    WanS2VStreamR1AttentionPlan,
    WanS2VStreamR1KVState,
    WanS2VStreamR1MixedKVView,
    WanS2VStreamR1NoisyKVCacheUpdate,
    WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer,
    WanS2VStreamR1NoisyKVCacheView,
    WanS2VTimestepMetadataPlan,
    WanS2VTimestepStaticMetadataBuffers,
    build_wan_s2v_stream_r1_noisy_kv_cache_update_plan,
    build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan,
    build_wan_s2v_stream_r1_cached_noisy_kv_index,
    build_wan_s2v_stream_r1_mixed_kv_attention_mask,
    build_wan_s2v_stream_r1_mixed_kv_attention_plan,
    build_wan_s2v_stream_r1_packed_attention_workspace,
    build_wan_s2v_stream_r1_segmented_packed_attention_workspace,
    compose_wan_s2v_stream_r1_segmented_mixed_kv_view,
    compose_wan_s2v_stream_r1_mixed_kv_view,
    pad_wan_s2v_stream_r1_mixed_kv_query_mask_for_sp,
    run_wan_s2v_stream_r1_cached_self_attention,
    split_wan_s2v_stream_r1_projected_kv,
    stream_r1_segmented_packed_varlen_attention,
    stream_r1_packed_varlen_attention,
    _pad_stream_r1_sp_packed_attention_output,
    _StreamR1ProfileSpan,
    update_wan_s2v_stream_r1_cached_self_attention_kv_cache,
    update_wan_s2v_stream_r1_noisy_kv_cache,
    update_wan_s2v_stream_r1_noisy_kv_cache_with_plan_buffer,
    validate_wan_s2v_stream_r1_forward_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VConditionBundle,
    WanS2VStreamR1AttentionRequest,
    WanS2VStreamR1CacheMetadata,
    WanS2VStreamR1CacheState,
    WanS2VStreamR1DenoisingStage,
    WanS2VTimestepAblationConfig,
    _WanS2VTransformerTimestepCudaGraphRunner,
    _has_negative_prompt_embeds,
    _select_wan_s2v_adaptive_timesteps,
)


class TestWanS2VStreamR1Profile(unittest.TestCase):
    def test_profile_span_uses_cpu_timing_during_cuda_graph_capture(self):
        profile = SimpleNamespace(use_cuda_events=True, timings=[])

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits."
                "wan_s2v_stream_r1._cuda_graph_capture_active",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.dits."
                "wan_s2v_stream_r1.torch.cuda.Event",
                side_effect=AssertionError("CUDA event should not be created"),
            ),
        ):
            with _StreamR1ProfileSpan(profile, "capture"):
                pass

        self.assertEqual(len(profile.timings), 1)
        self.assertEqual(profile.timings[0][0], "capture")


class TestWanS2VNoisyRopeGridSizes(unittest.TestCase):
    def test_rope_params_uses_complex64(self):
        self.assertEqual(rope_params(8, 4).dtype, torch.complex64)

    def test_rope_apply_matches_double_reference(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 3, 8, dtype=torch.float32)
        phases = torch.randn(2, 5, 3, 4, dtype=torch.float64)
        freqs = torch.polar(torch.ones_like(phases), phases)

        expected = torch.view_as_complex(
            x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
        )
        expected = torch.view_as_real(expected * freqs).flatten(3).float()

        actual = _rope_apply_precomputed(x, freqs.to(torch.complex64))

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_rope_apply_preserves_input_dtype(self):
        x = torch.randn(1, 3, 2, 8, dtype=torch.float32).to(torch.bfloat16)
        phases = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        freqs = torch.polar(torch.ones_like(phases), phases).to(torch.complex64)

        actual = _rope_apply_precomputed(x, freqs)

        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, x.shape)


class TestWanS2VStreamR1Fp8CommKernels(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_fp8_qkv_dequant_unpack_matches_reference(self):
        from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
            blockwise_dequant_fp8,
            blockwise_quant_fp8,
            fused_dequant_unpack_qkv_fp8,
        )
        from sglang.jit_kernel.diffusion.triton.usp_permute import (
            fused_unpack_qkv_from_all_to_all,
        )

        torch.manual_seed(3)
        B = 1
        S_local = 3
        H_local = 2
        world_size = 2
        D = 128
        group_size = 128
        packed = torch.randn(
            3 * H_local * world_size,
            B,
            S_local,
            D,
            device="cuda",
            dtype=torch.bfloat16,
        )
        packed_q, scale = blockwise_quant_fp8(packed, group_size=group_size)
        dequant = blockwise_dequant_fp8(
            packed_q,
            scale,
            group_size=group_size,
            dtype=packed.dtype,
        )
        expected = fused_unpack_qkv_from_all_to_all(
            dequant,
            B,
            S_local,
            H_local,
            D,
            world_size,
        )

        actual = fused_dequant_unpack_qkv_fp8(
            packed_q,
            scale,
            B,
            S_local,
            H_local,
            D,
            world_size,
            group_size=group_size,
            dtype=packed.dtype,
        )

        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_fp8_output_dequant_postunpack_matches_reference(self):
        from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
            blockwise_dequant_fp8,
            blockwise_quant_fp8,
            fused_dequant_unpack_output_fp8,
        )

        torch.manual_seed(4)
        batch_size = 1
        s_local = 3
        world_size = 2
        seq_len = s_local * world_size
        h_local = 2
        D = 128
        group_size = 128
        packed = torch.randn(
            seq_len,
            batch_size,
            h_local,
            D,
            device="cuda",
            dtype=torch.bfloat16,
        )
        packed_q, scale = blockwise_quant_fp8(packed, group_size=group_size)
        dequant = blockwise_dequant_fp8(
            packed_q,
            scale,
            group_size=group_size,
            dtype=packed.dtype,
        )
        expected = (
            dequant.reshape(world_size, s_local, batch_size, h_local, D)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
            .reshape(batch_size, s_local, h_local * world_size, D)
        )

        actual = fused_dequant_unpack_output_fp8(
            packed_q,
            scale,
            batch_size=batch_size,
            seq_len=seq_len,
            world_size=world_size,
            group_size=group_size,
            dtype=packed.dtype,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestWanS2VStreamR1SegmentKernels(unittest.TestCase):
    def test_segment_gate_add_matches_reference(self):
        torch.manual_seed(1)
        residual = torch.randn(2, 5, 4, dtype=torch.float32)
        update = torch.randn(2, 5, 4, dtype=torch.float32)
        gate = torch.randn(2, 2, 4, dtype=torch.float32)
        seg_idx = 3

        expected = residual + torch.cat(
            [
                update[:, :seg_idx] * gate[:, 0:1],
                update[:, seg_idx:] * gate[:, 1:2],
            ],
            dim=1,
        )
        actual = _segment_gate_add(residual, update, gate, seg_idx)

        torch.testing.assert_close(actual, expected)

    def test_segment_gate_add_preserves_residual_dtype(self):
        residual = torch.randn(1, 4, 3, dtype=torch.float32).to(torch.bfloat16)
        update = torch.randn(1, 4, 3, dtype=torch.float32).to(torch.bfloat16)
        gate = torch.randn(1, 2, 3, dtype=torch.float32)

        actual = _segment_gate_add(residual, update, gate, 2)

        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, residual.shape)

    def test_segment_modulate_matches_reference(self):
        torch.manual_seed(2)
        x = torch.randn(2, 5, 4, dtype=torch.float32)
        shift = torch.randn(2, 2, 4, dtype=torch.float32)
        scale = torch.randn(2, 2, 4, dtype=torch.float32)
        seg_idx = 3

        expected = torch.cat(
            [
                x[:, :seg_idx] * (1 + scale[:, 0:1]) + shift[:, 0:1],
                x[:, seg_idx:] * (1 + scale[:, 1:2]) + shift[:, 1:2],
            ],
            dim=1,
        )
        actual = _segment_modulate(x, shift, scale, seg_idx)

        torch.testing.assert_close(actual, expected)

    def test_segment_modulate_uses_requested_output_dtype(self):
        x = torch.randn(1, 4, 3, dtype=torch.float32).to(torch.bfloat16)
        shift = torch.randn(1, 2, 3, dtype=torch.float32)
        scale = torch.randn(1, 2, 3, dtype=torch.float32)
        seg_idx = 2

        expected = torch.cat(
            [
                x[:, :seg_idx] * (1 + scale[:, 0:1]) + shift[:, 0:1],
                x[:, seg_idx:] * (1 + scale[:, 1:2]) + shift[:, 1:2],
            ],
            dim=1,
        ).to(torch.bfloat16)
        actual = _segment_modulate(x, shift, scale, seg_idx, out_dtype=torch.bfloat16)

        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, x.shape)
        torch.testing.assert_close(actual, expected)

    def test_legacy_grid_has_zero_start_and_current_values(self):
        grid_sizes = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)

        rope_grid = _build_s2v_noisy_rope_grid_sizes(
            grid_sizes,
            stream_r1_mode=False,
            current_start=11,
            frame_seq_length=0,
        )

        self.assertEqual(len(rope_grid), 1)
        self.assertEqual(len(rope_grid[0]), 3)
        start, end, span = rope_grid[0]
        self.assertEqual(start.shape, grid_sizes.shape)
        self.assertEqual(end.shape, grid_sizes.shape)
        self.assertEqual(span.shape, grid_sizes.shape)
        torch.testing.assert_close(start, torch.zeros_like(grid_sizes))
        torch.testing.assert_close(end, grid_sizes)
        torch.testing.assert_close(span, grid_sizes)

    def test_stream_r1_grid_offsets_temporal_start_and_end(self):
        grid_sizes = torch.tensor([[3, 4, 5]], dtype=torch.long)

        rope_grid = _build_s2v_noisy_rope_grid_sizes(
            grid_sizes,
            stream_r1_mode=True,
            current_start=24,
            frame_seq_length=12,
        )

        start, end, span = rope_grid[0]
        torch.testing.assert_close(start, torch.tensor([[2, 0, 0]], dtype=torch.long))
        torch.testing.assert_close(end, torch.tensor([[5, 4, 5]], dtype=torch.long))
        torch.testing.assert_close(span, grid_sizes)
        torch.testing.assert_close(end - start, span)

    def test_stream_r1_tensor_current_start_rope_matches_legacy_grid(self):
        noisy_grid_sizes = torch.tensor([[3, 2, 2]], dtype=torch.long)
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0]).view(1, 3),
                torch.tensor([31, 2, 2]).view(1, 3),
                torch.tensor([1, 2, 2]).view(1, 3),
            ]
        ]
        current_start = 8
        frame_seq_length = 4
        noisy_seq_len = int(noisy_grid_sizes.prod().item())
        ref_seq_len = 4
        x = torch.zeros(1, noisy_seq_len + ref_seq_len, 1, 12)
        freqs = rope_params(max_seq_len=64, dim=12)
        legacy_grid = (
            _build_s2v_noisy_rope_grid_sizes(
                noisy_grid_sizes,
                stream_r1_mode=True,
                current_start=current_start,
                frame_seq_length=frame_seq_length,
            )
            + ref_grid_sizes
        )

        legacy = _rope_precompute(x, legacy_grid, freqs)
        tensor_current = _rope_precompute_s2v_stream_r1_tensor_current_start(
            x,
            noisy_grid_sizes,
            ref_grid_sizes,
            freqs,
            current_start=torch.tensor(current_start, dtype=torch.long),
            frame_seq_length=frame_seq_length,
        )

        torch.testing.assert_close(tensor_current, legacy)

    def test_stream_r1_grid_rejects_invalid_current_start(self):
        grid_sizes = torch.tensor([[3, 4, 5]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            _build_s2v_noisy_rope_grid_sizes(
                grid_sizes,
                stream_r1_mode=True,
                current_start=-12,
                frame_seq_length=12,
            )

        with self.assertRaisesRegex(ValueError, "frame-aligned"):
            _build_s2v_noisy_rope_grid_sizes(
                grid_sizes,
                stream_r1_mode=True,
                current_start=5,
                frame_seq_length=12,
            )


class TestWanS2VAudioTensorSlice(unittest.TestCase):
    def test_tensor_audio_slice_matches_legacy_slice(self):
        audio_emb = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
        audio_start_frame = torch.tensor(3, dtype=torch.long)
        motion_prefix_frames = torch.tensor(2, dtype=torch.long)
        latent_frames = 4

        tensor_slice = _slice_s2v_audio_embeddings_with_tensor_start(
            audio_emb,
            audio_start_frame=audio_start_frame,
            motion_prefix_frames=motion_prefix_frames,
            latent_frames=latent_frames,
        )

        legacy = audio_emb[:, 5:9, :]
        torch.testing.assert_close(tensor_slice, legacy)

    def test_tensor_audio_slice_supports_global_audio_shape(self):
        audio_emb = torch.arange(2 * 12 * 4 * 5, dtype=torch.float32).reshape(
            2, 12, 4, 5
        )
        tensor_slice = _slice_s2v_audio_embeddings_with_tensor_start(
            audio_emb,
            audio_start_frame=torch.tensor(1, dtype=torch.long),
            motion_prefix_frames=torch.tensor(3, dtype=torch.long),
            latent_frames=6,
        )

        torch.testing.assert_close(tensor_slice, audio_emb[:, 4:10])


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

    def test_timestep_profile_and_ablation_are_forwarded_in_extra(self):
        params = WanS2VSamplingParams(
            timestep_profile=True,
            timestep_profile_sync=True,
            timestep_ablation_mode="zero_pred",
            timestep_ablation_indices=[1, 3],
            timestep_ablation_blocks=[2],
            timestep_ablation_values=[937.5],
            timestep_ablation_log=True,
        )

        extra = params.build_request_extra()

        self.assertTrue(extra["timestep_profile"])
        self.assertTrue(extra["timestep_profile_sync"])
        self.assertEqual(extra["timestep_ablation_mode"], "zero_pred")
        self.assertEqual(extra["timestep_ablation_indices"], [1, 3])
        self.assertEqual(extra["timestep_ablation_blocks"], [2])
        self.assertEqual(extra["timestep_ablation_values"], [937.5])
        self.assertTrue(extra["timestep_ablation_log"])

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


class TestWanS2VAudioInjectionSP(unittest.TestCase):
    class _PreNorm:
        def __call__(self, hidden_states):
            return hidden_states * 2

    class _Injector:
        def __call__(self, *, x, context, context_lens):
            return x + context[:, :1]

    class _AudioInjector:
        def __init__(self, outer):
            self.injected_block_id = {0: 0}
            self.injector_pre_norm_feat = [outer._PreNorm()]
            self.injector = [outer._Injector()]

    def _model(self):
        model = WanS2VTransformer3DModel.__new__(WanS2VTransformer3DModel)
        model.audio_injector = self._AudioInjector(self)
        model.enable_adain = False
        model.adain_mode = None
        model.original_seq_len = 8
        model.sequence_shard_start = 0
        model.merged_audio_emb = torch.arange(1 * 4 * 3 * 2, dtype=torch.float32).view(
            1, 4, 3, 2
        )
        return model

    def test_sequence_sharded_audio_injection_matches_full_path(self):
        hidden_states = torch.arange(1 * 10 * 2, dtype=torch.float32).view(1, 10, 2)

        full_model = self._model()
        full_model.use_context_parallel = False
        expected = full_model._after_transformer_block(0, hidden_states.clone())

        shard_model = self._model()
        shard_model.use_context_parallel = True
        outputs = []
        start = 0
        for shard in torch.split(hidden_states, [3, 3, 4], dim=1):
            shard_model.sequence_shard_start = start
            outputs.append(shard_model._after_transformer_block(0, shard.clone()))
            start += shard.shape[1]

        torch.testing.assert_close(torch.cat(outputs, dim=1), expected)


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
    def test_sp_mask_padding_blocks_padded_keys_for_real_queries(self):
        mask = torch.tensor(
            [
                [
                    [True, False, True],
                    [False, True, True],
                    [True, True, False],
                ]
            ],
            dtype=torch.bool,
        )

        padded = _pad_stream_r1_attention_mask_for_sp(mask, 3, 2)

        self.assertEqual(padded.shape, (1, 5, 5))
        torch.testing.assert_close(padded[:, :3, :3], mask)
        self.assertFalse(padded[:, :3, 3:].any().item())
        self.assertTrue(padded[:, 3:, :].all().item())

    def test_sp_mask_padding_rejects_mismatched_shape(self):
        mask = torch.ones((1, 3, 4), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "does not match"):
            _pad_stream_r1_attention_mask_for_sp(mask, 3, 1)

    def test_sp_packed_output_padding_appends_sequence_tokens(self):
        packed = torch.arange(3 * 2, dtype=torch.float32).view(3, 2, 1)

        padded = _pad_stream_r1_sp_packed_attention_output(
            packed,
            batch_size=1,
            total_seq_len=3,
            sp_pad_tokens=2,
        )

        self.assertEqual(padded.shape, (5, 2, 1))
        torch.testing.assert_close(padded[:3], packed)
        torch.testing.assert_close(padded[3:], torch.zeros(2, 2, 1))

    def test_sp_packed_output_padding_rejects_non_batch_one(self):
        packed = torch.zeros(3, 2, 1)

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            _pad_stream_r1_sp_packed_attention_output(
                packed,
                batch_size=2,
                total_seq_len=3,
                sp_pad_tokens=1,
            )

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
        plan_mask = layout.to_no_kv_attention_plan().to_dense_mask(torch.device("cpu"))[
            0
        ]

        self.assertEqual(mask.shape, (10, 10))
        torch.testing.assert_close(plan_mask, mask)
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
        plan = layout.to_no_kv_attention_plan()

        self.assertEqual(plan.query_seq_len, 6)
        self.assertEqual(plan.kv_seq_len, 6)
        self.assertEqual(plan.query_block_tokens, 4)
        torch.testing.assert_close(plan.to_dense_mask(torch.device("cpu"))[0], mask)
        self.assertEqual(
            torch.nonzero(mask[0], as_tuple=False).flatten().tolist(),
            [0, 1, 2, 3, 4, 5],
        )

    def test_attention_plan_rejects_mismatched_noisy_kv_index(self):
        with self.assertRaisesRegex(ValueError, "noisy_kv_absolute_index"):
            WanS2VStreamR1AttentionPlan(
                query_seq_len=2,
                kv_seq_len=3,
                noisy_query_seq_len=2,
                noisy_kv_seq_len=2,
                condition_kv_seq_len=1,
                frame_seq_length=1,
                query_block_tokens=2,
                local_attn_size=2,
                sink_size=0,
                noisy_kv_absolute_index=torch.tensor([0]),
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

    def _abs_kv(self, start: int, length: int, heads: int = 2):
        return self._indexed_kv(range(start, start + length), heads=heads)

    def _indexed_kv(self, positions, heads: int = 2):
        key = (
            torch.tensor(list(positions), dtype=torch.float32)
            .view(1, -1, 1, 1)
            .expand(1, -1, heads, 1)
            .clone()
        )
        value = key + 100
        return key, value

    def _cache(self, tokens: int, heads: int = 2):
        return {
            "k": torch.zeros(1, tokens, heads, 1),
            "v": torch.zeros(1, tokens, heads, 1),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }

    def test_split_projected_kv_separates_noisy_and_condition_ranges(self):
        key, value = self._kv(seq_len=5)

        split = split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=3)

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
            split_wan_s2v_stream_r1_projected_kv(key, value[:, :4], noisy_seq_len=3)

        with self.assertRaisesRegex(ValueError, "positive"):
            split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=0)

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=6)

    def test_compose_mixed_kv_appends_current_condition_after_cached_noisy(self):
        key, value = self._kv(seq_len=5)
        split = split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=3)
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
        split = split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=3)
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

    def test_segmented_mixed_kv_keeps_noisy_and_condition_sources(self):
        key, value = self._kv(seq_len=5)
        split = split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=3)
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

        segmented = compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
            noisy_view,
            split,
        )

        self.assertIs(segmented.noisy_key, cached_key)
        self.assertIs(segmented.noisy_value, cached_value)
        self.assertIs(segmented.condition_key, split.condition_key)
        self.assertIs(segmented.condition_value, split.condition_value)
        self.assertEqual(segmented.cached_noisy_seq_len, 4)
        self.assertEqual(segmented.condition_seq_len, 2)
        self.assertEqual(segmented.condition_start_index, 4)
        self.assertEqual(segmented.total_seq_len, 6)
        self.assertFalse(hasattr(segmented, "key"))

    def test_compose_mixed_kv_validates_cached_and_current_dimensions(self):
        key, value = self._kv(seq_len=4)
        split = split_wan_s2v_stream_r1_projected_kv(key, value, noisy_seq_len=2)
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

    def test_cached_noisy_kv_index_tracks_sink_and_rolling_ranges(self):
        cache = self._cache(tokens=3)

        for current_start in (0, 1, 2, 3):
            key, value = self._abs_kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        index = build_wan_s2v_stream_r1_cached_noisy_kv_index(view, update)

        self.assertEqual(index.tolist(), [0, 2, 3])
        torch.testing.assert_close(
            view.key[:, :, 0, 0],
            torch.tensor([[0.001, 2.0, 3.0]]),
        )

    def test_mixed_kv_attention_mask_uses_cached_index_and_condition_tokens(self):
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=1,
            local_attn_size=5,
            sink_size=1,
            current_start=2,
        )
        cached_key, cached_value = self._indexed_kv([0, 2, 3, 4, 5])
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=6,
            local_end_index=5,
            local_start=2,
            local_end=6,
        )
        condition_key, condition_value = self._abs_kv(100, 2)
        mixed = WanS2VStreamR1MixedKVView(
            key=torch.cat([cached_key, condition_key], dim=1),
            value=torch.cat([cached_value, condition_value], dim=1),
            cached_noisy_seq_len=5,
            condition_seq_len=2,
            global_end_index=6,
            local_end_index=5,
            local_start=2,
            local_end=6,
        )

        index = build_wan_s2v_stream_r1_cached_noisy_kv_index(noisy_view, update)
        plan = build_wan_s2v_stream_r1_mixed_kv_attention_plan(
            noisy_view, mixed, update
        )
        mask = build_wan_s2v_stream_r1_mixed_kv_attention_mask(
            noisy_view, mixed, update
        )[0]

        self.assertEqual(index.tolist(), [0, 2, 3, 4, 5])
        self.assertEqual(plan.query_seq_len, 6)
        self.assertEqual(plan.kv_seq_len, 7)
        self.assertEqual(plan.noisy_kv_absolute_index.tolist(), [0, 2, 3, 4, 5])
        self.assertEqual(
            [
                (
                    group.query_start,
                    group.query_end,
                    group.kv_indices.tolist(),
                )
                for group in plan.query_groups(torch.device("cpu"))
            ],
            [
                (0, 2, [0, 1, 2, 5, 6]),
                (2, 4, [0, 2, 3, 4, 5, 6]),
                (4, 6, [1, 2, 3, 4, 5, 6]),
            ],
        )
        torch.testing.assert_close(plan.to_dense_mask(torch.device("cpu"))[0], mask)
        self.assertEqual(mask.shape, (6, 7))
        self.assertEqual(
            torch.nonzero(mask[0], as_tuple=False).flatten().tolist(),
            [0, 1, 2, 5, 6],
        )
        self.assertEqual(
            torch.nonzero(mask[2], as_tuple=False).flatten().tolist(),
            [0, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            torch.nonzero(mask[4], as_tuple=False).flatten().tolist(),
            [1, 2, 3, 4, 5, 6],
        )

    def test_mixed_kv_attention_mask_validates_mixed_view_metadata(self):
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        cached_key, cached_value = self._abs_kv(0, 2)
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=2,
            local_end_index=2,
            local_start=1,
            local_end=2,
        )
        mixed = WanS2VStreamR1MixedKVView(
            key=cached_key,
            value=cached_value,
            cached_noisy_seq_len=1,
            condition_seq_len=1,
            global_end_index=2,
            local_end_index=2,
            local_start=1,
            local_end=2,
        )

        with self.assertRaisesRegex(ValueError, "cached_noisy_seq_len"):
            build_wan_s2v_stream_r1_mixed_kv_attention_mask(noisy_view, mixed, update)

    def test_sp_mixed_kv_query_mask_padding_keeps_padded_queries_valid(self):
        mask = torch.tensor(
            [[[True, False, True], [False, True, True]]],
            dtype=torch.bool,
        )

        padded = pad_wan_s2v_stream_r1_mixed_kv_query_mask_for_sp(mask, 2, 2)

        self.assertEqual(padded.shape, (1, 4, 3))
        torch.testing.assert_close(padded[:, :2], mask)
        self.assertTrue(padded[:, 2:].all().item())

    def test_packed_varlen_attention_matches_dense_sdpa_reference(self):
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=1,
            local_attn_size=5,
            sink_size=1,
            current_start=4,
        )
        cached_key, cached_value = self._indexed_kv([0, 4, 5, 6, 7])
        cached_key = cached_key.expand(1, -1, 2, 4).contiguous()
        cached_value = cached_value.expand(1, -1, 2, 4).contiguous()
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=8,
            local_end_index=5,
            local_start=4,
            local_end=8,
        )
        condition_key = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
        condition_value = condition_key + 100
        mixed = WanS2VStreamR1MixedKVView(
            key=torch.cat([cached_key, condition_key], dim=1),
            value=torch.cat([cached_value, condition_value], dim=1),
            cached_noisy_seq_len=5,
            condition_seq_len=2,
            global_end_index=8,
            local_end_index=5,
            local_start=4,
            local_end=8,
        )
        plan = build_wan_s2v_stream_r1_mixed_kv_attention_plan(
            noisy_view, mixed, update
        )
        groups = plan.query_groups(torch.device("cpu"))
        self.assertEqual([group.kv_ranges for group in groups], [((0, 7),), ((1, 7),)])
        query = torch.randn(1, 6, 2, 4)
        scale = 0.5

        packed = stream_r1_packed_varlen_attention(
            query,
            mixed.key,
            mixed.value,
            plan,
            softmax_scale=scale,
            force_torch=True,
        )
        mask = plan.to_dense_mask(query.device).to(dtype=query.dtype)
        mask = (mask - 1.0) * torch.finfo(query.dtype).max
        dense = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            mixed.key.transpose(1, 2),
            mixed.value.transpose(1, 2),
            attn_mask=mask[:, None, :, :],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ).transpose(1, 2)

        torch.testing.assert_close(packed, dense, rtol=1e-5, atol=1e-5)

    def test_single_noisy_block_plan_merges_full_kv_condition_queries(self):
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=1,
            local_attn_size=5,
            sink_size=1,
            current_start=0,
        )
        cached_key, cached_value = self._indexed_kv([0, 1, 2, 3])
        cached_key = cached_key.expand(1, -1, 2, 4).contiguous()
        cached_value = cached_value.expand(1, -1, 2, 4).contiguous()
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=4,
            local_end_index=4,
            local_start=1,
            local_end=4,
        )
        condition_key = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
        condition_value = condition_key + 100
        split = split_wan_s2v_stream_r1_projected_kv(
            torch.cat([cached_key, condition_key], dim=1),
            torch.cat([cached_value, condition_value], dim=1),
            noisy_seq_len=4,
        )
        segmented = compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
            noisy_view,
            split,
        )
        plan = build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan(
            noisy_view,
            segmented,
            update,
        )
        query = torch.randn(1, 6, 2, 4)

        groups = plan.query_groups(torch.device("cpu"))
        workspace = build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
            query,
            segmented,
            plan,
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].query_start, 0)
        self.assertEqual(groups[0].query_end, 6)
        self.assertEqual(groups[0].kv_ranges, ((0, 6),))
        self.assertEqual(workspace.cu_seqlens_q.tolist(), [0, 6])
        self.assertEqual(workspace.cu_seqlens_k.tolist(), [0, 6])
        self.assertEqual(len(workspace.segments), 1)
        self.assertEqual(workspace.max_seqlen_q, 6)
        self.assertEqual(workspace.max_seqlen_k, 6)
        self.assertTrue(workspace.query_matches_input_order)
        self.assertTrue(plan.to_dense_mask(torch.device("cpu")).all().item())

    def test_packed_attention_workspace_records_segments(self):
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=4,
            total_seq_len=6,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=2,
            sink_size=1,
        )
        plan = layout.to_no_kv_attention_plan()
        query = torch.randn(1, 6, 1, 2)
        key = torch.randn(1, 6, 1, 2)
        value = torch.randn(1, 6, 1, 2)

        workspace = build_wan_s2v_stream_r1_packed_attention_workspace(
            query, key, value, plan
        )

        self.assertEqual(workspace.cu_seqlens_q.tolist(), [0, 2, 4, 6])
        self.assertEqual(workspace.max_seqlen_q, 2)
        self.assertGreaterEqual(workspace.max_seqlen_k, 4)
        self.assertEqual(len(workspace.segments), 3)
        self.assertTrue(workspace.query_matches_input_order)
        self.assertEqual(workspace.query.data_ptr(), query.data_ptr())

    def test_packed_attention_workspace_flattens_full_input_order_kv(self):
        plan = WanS2VStreamR1AttentionPlan(
            query_seq_len=6,
            kv_seq_len=6,
            noisy_query_seq_len=4,
            noisy_kv_seq_len=4,
            condition_kv_seq_len=2,
            frame_seq_length=1,
            query_block_tokens=4,
            local_attn_size=5,
            sink_size=1,
            noisy_kv_absolute_index=torch.arange(4),
        )
        query = torch.randn(2, 6, 2, 4)
        key = torch.randn(2, 6, 2, 4)
        value = torch.randn(2, 6, 2, 4)

        workspace = build_wan_s2v_stream_r1_packed_attention_workspace(
            query,
            key,
            value,
            plan,
        )

        self.assertTrue(workspace.query_matches_input_order)
        self.assertEqual(workspace.query.data_ptr(), query.data_ptr())
        self.assertEqual(workspace.key.data_ptr(), key.data_ptr())
        self.assertEqual(workspace.value.data_ptr(), value.data_ptr())
        self.assertEqual(workspace.key.shape, (12, 2, 4))
        self.assertEqual(workspace.value.shape, (12, 2, 4))

    def test_segmented_packed_attention_matches_materialized_workspace(self):
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=1,
            local_attn_size=5,
            sink_size=1,
            current_start=4,
        )
        cached_key, cached_value = self._indexed_kv([0, 4, 5, 6, 7])
        cached_key = cached_key.expand(1, -1, 2, 4).contiguous()
        cached_value = cached_value.expand(1, -1, 2, 4).contiguous()
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=8,
            local_end_index=5,
            local_start=4,
            local_end=8,
        )
        condition_key = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
        condition_value = condition_key + 100
        split = split_wan_s2v_stream_r1_projected_kv(
            torch.cat([cached_key[:, -4:], condition_key], dim=1),
            torch.cat([cached_value[:, -4:], condition_value], dim=1),
            noisy_seq_len=4,
        )
        segmented = compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
            noisy_view,
            split,
        )
        mixed = compose_wan_s2v_stream_r1_mixed_kv_view(noisy_view, split)
        segmented_plan = build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan(
            noisy_view,
            segmented,
            update,
        )
        materialized_plan = build_wan_s2v_stream_r1_mixed_kv_attention_plan(
            noisy_view,
            mixed,
            update,
        )
        query = torch.randn(1, 6, 2, 4)

        segmented_workspace = (
            build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
                query,
                segmented,
                segmented_plan,
            )
        )
        materialized_workspace = build_wan_s2v_stream_r1_packed_attention_workspace(
            query,
            mixed.key,
            mixed.value,
            materialized_plan,
        )
        segmented_output = stream_r1_segmented_packed_varlen_attention(
            query,
            segmented,
            segmented_plan,
            softmax_scale=0.5,
            force_torch=True,
        )
        materialized_output = stream_r1_packed_varlen_attention(
            query,
            mixed.key,
            mixed.value,
            materialized_plan,
            softmax_scale=0.5,
            force_torch=True,
        )

        torch.testing.assert_close(
            segmented_plan.to_dense_mask(query.device),
            materialized_plan.to_dense_mask(query.device),
        )
        torch.testing.assert_close(
            segmented_workspace.query, materialized_workspace.query
        )
        torch.testing.assert_close(segmented_workspace.key, materialized_workspace.key)
        torch.testing.assert_close(
            segmented_workspace.value, materialized_workspace.value
        )
        self.assertEqual(
            segmented_workspace.cu_seqlens_q.tolist(),
            materialized_workspace.cu_seqlens_q.tolist(),
        )
        self.assertEqual(
            segmented_workspace.cu_seqlens_k.tolist(),
            materialized_workspace.cu_seqlens_k.tolist(),
        )
        torch.testing.assert_close(segmented_output, materialized_output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_segmented_kv_pack_matches_workspace(self):
        from sglang.jit_kernel.diffusion.triton.stream_r1_segmented_pack import (
            fused_pack_segmented_kv,
        )

        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=1,
            local_attn_size=5,
            sink_size=1,
            current_start=4,
        )
        cached_key, cached_value = self._indexed_kv([0, 4, 5, 6, 7])
        cached_key = cached_key.expand(1, -1, 2, 4).contiguous().cuda()
        cached_value = cached_value.expand(1, -1, 2, 4).contiguous().cuda()
        noisy_view = WanS2VStreamR1NoisyKVCacheView(
            key=cached_key,
            value=cached_value,
            global_end_index=8,
            local_end_index=5,
            local_start=4,
            local_end=8,
        )
        condition_key = torch.arange(16, dtype=torch.float32, device="cuda").view(
            1, 2, 2, 4
        )
        condition_value = condition_key + 100
        split = split_wan_s2v_stream_r1_projected_kv(
            torch.cat([cached_key[:, -4:], condition_key], dim=1),
            torch.cat([cached_value[:, -4:], condition_value], dim=1),
            noisy_seq_len=4,
        )
        segmented = compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
            noisy_view,
            split,
        )
        plan = build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan(
            noisy_view,
            segmented,
            update,
        )
        query = torch.randn(1, 6, 2, 4, device="cuda")
        groups = plan.query_groups(query.device)
        self.assertEqual([group.kv_ranges for group in groups], [((0, 7),), ((1, 7),)])
        expected_key = torch.cat(
            [
                segmented.noisy_key[0, :5],
                segmented.condition_key[0, :2],
                segmented.noisy_key[0, 1:5],
                segmented.condition_key[0, :2],
            ],
            dim=0,
        )
        expected_value = torch.cat(
            [
                segmented.noisy_value[0, :5],
                segmented.condition_value[0, :2],
                segmented.noisy_value[0, 1:5],
                segmented.condition_value[0, :2],
            ],
            dim=0,
        )
        batch_indices = torch.tensor([0, 0], dtype=torch.int64, device="cuda")
        packed_starts = torch.tensor([0, 7], dtype=torch.int64, device="cuda")
        source_starts = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        lengths = torch.tensor([7, 6], dtype=torch.int64, device="cuda")
        out_key = torch.empty_like(expected_key)
        out_value = torch.empty_like(expected_value)

        packed_key, packed_value = fused_pack_segmented_kv(
            segmented.noisy_key,
            segmented.noisy_value,
            segmented.condition_key,
            segmented.condition_value,
            batch_indices,
            packed_starts,
            source_starts,
            lengths,
            total_tokens=13,
            noisy_seq_len=segmented.cached_noisy_seq_len,
            max_length=7,
            out=(out_key, out_value),
        )

        self.assertEqual(packed_key.data_ptr(), out_key.data_ptr())
        self.assertEqual(packed_value.data_ptr(), out_value.data_ptr())
        torch.testing.assert_close(packed_key, expected_key)
        torch.testing.assert_close(packed_value, expected_value)

        with patch(
            "sglang.jit_kernel.diffusion.triton.stream_r1_segmented_pack."
            "fused_pack_segmented_kv",
            wraps=fused_pack_segmented_kv,
        ) as fused_pack_mock:
            workspace = build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
                query,
                segmented,
                plan,
            )

        self.assertGreater(fused_pack_mock.call_count, 0)
        torch.testing.assert_close(workspace.key, expected_key)
        torch.testing.assert_close(workspace.value, expected_value)


class TestWanS2VStreamR1CachedSelfAttentionBranch(unittest.TestCase):
    def _cache(self, tokens: int):
        return {
            "k": torch.zeros(1, tokens, 1, 1),
            "v": torch.zeros(1, tokens, 1, 1),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }

    def _indexed_kv(self, positions):
        key = torch.tensor(list(positions), dtype=torch.float32).view(1, -1, 1, 1)
        value = key + 100
        return key, value

    def test_cached_branch_updates_cache_and_calls_attention_with_mixed_kv(self):
        cache = self._cache(tokens=4)
        calls = []

        def recording_attention(query, key, value, attn_mask=None):
            calls.append(
                {
                    "query": query.clone(),
                    "key": key.clone(),
                    "value": value.clone(),
                    "attn_mask": attn_mask.clone(),
                }
            )
            return query + 10

        query = torch.zeros(1, 3, 1, 1)
        key, value = self._indexed_kv([0, 1, 100])
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        output = run_wan_s2v_stream_r1_cached_self_attention(
            recording_attention,
            query=query,
            key=key,
            value=value,
            kv_cache=cache,
            layout=layout,
            cache_start=None,
        )

        torch.testing.assert_close(output, query + 10)
        self.assertEqual(cache["global_end_index"].item(), 2)
        torch.testing.assert_close(cache["k"][:, :2, 0, 0], torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(
            calls[-1]["key"][:, :, 0, 0], torch.tensor([[0.0, 1.0, 100.0]])
        )

        query = torch.ones(1, 3, 1, 1)
        key, value = self._indexed_kv([2, 3, 200])
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=2,
        )

        run_wan_s2v_stream_r1_cached_self_attention(
            recording_attention,
            query=query,
            key=key,
            value=value,
            kv_cache=cache,
            layout=layout,
            cache_start=None,
        )

        self.assertEqual(cache["global_end_index"].item(), 4)
        self.assertEqual(cache["local_end_index"].item(), 4)
        torch.testing.assert_close(
            cache["k"][:, :, 0, 0], torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        )
        torch.testing.assert_close(
            calls[-1]["key"][:, :, 0, 0],
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 200.0]]),
        )
        torch.testing.assert_close(
            calls[-1]["value"][:, :, 0, 0],
            torch.tensor([[100.0, 101.0, 102.0, 103.0, 300.0]]),
        )
        self.assertEqual(calls[-1]["attn_mask"].shape, (1, 3, 5))

    def test_cache_update_only_updates_cache_without_attention_callable(self):
        cache = self._cache(tokens=4)
        query = torch.zeros(1, 3, 1, 1)
        key, value = self._indexed_kv([0, 1, 100])
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        update_wan_s2v_stream_r1_cached_self_attention_kv_cache(
            query=query,
            key=key,
            value=value,
            kv_cache=cache,
            layout=layout,
            cache_start=None,
        )

        self.assertEqual(cache["global_end_index"].item(), 2)
        self.assertEqual(cache["local_end_index"].item(), 2)
        torch.testing.assert_close(cache["k"][:, :2, 0, 0], torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(
            cache["v"][:, :2, 0, 0], torch.tensor([[100.0, 101.0]])
        )

    def test_cached_branch_gathers_kv_and_uses_replicated_kv_for_sp(self):
        cache = self._cache(tokens=4)
        calls = []

        def recording_attention(query, key, value, attn_mask=None, **kwargs):
            calls.append(
                {
                    "query": query.clone(),
                    "key": key.clone(),
                    "value": value.clone(),
                    "attn_mask": attn_mask.clone(),
                    "kwargs": dict(kwargs),
                }
            )
            return query + 10

        query = torch.zeros(1, 2, 1, 1)
        key_local = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
        value_local = key_local + 100
        key_padded_global = torch.tensor([0.0, 1.0, 100.0, 999.0]).view(1, 4, 1, 1)
        value_padded_global = key_padded_global + 100

        def fake_all_gather(tensor, dim):
            self.assertEqual(dim, 1)
            if torch.equal(tensor, key_local):
                return key_padded_global
            if torch.equal(tensor, value_local):
                return value_padded_global
            raise AssertionError("unexpected tensor gathered")

        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "get_sp_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "sequence_model_parallel_all_gather",
            side_effect=fake_all_gather,
        ):
            output = run_wan_s2v_stream_r1_cached_self_attention(
                recording_attention,
                query=query,
                key=key_local,
                value=value_local,
                kv_cache=cache,
                layout=layout,
                cache_start=None,
                sequence_shard_enabled=True,
                sp_pad_tokens=1,
            )

        torch.testing.assert_close(output, query + 10)
        self.assertEqual(calls[-1]["kwargs"], {"kv_is_replicated": True})
        torch.testing.assert_close(
            calls[-1]["key"][:, :, 0, 0], torch.tensor([[0.0, 1.0, 100.0]])
        )
        torch.testing.assert_close(
            calls[-1]["value"][:, :, 0, 0], torch.tensor([[100.0, 101.0, 200.0]])
        )
        self.assertEqual(calls[-1]["attn_mask"].shape, (1, 4, 3))
        self.assertTrue(calls[-1]["attn_mask"][:, 3:].all().item())

    def test_cached_branch_uses_packed_backend_without_calling_dense_attention(self):
        cache = {
            "k": torch.zeros(1, 4, 1, 2),
            "v": torch.zeros(1, 4, 1, 2),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }

        class FailingAttention:
            softmax_scale = 1.0

            def __call__(self, *args, **kwargs):
                raise AssertionError("dense attention should not be called")

        query = torch.randn(1, 3, 1, 2)
        key = torch.randn(1, 3, 1, 2)
        value = torch.randn(1, 3, 1, 2)
        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        with patch.dict(
            "os.environ",
            {"SGLANG_STREAM_R1_ATTENTION_BACKEND": "packed_varlen"},
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "compose_wan_s2v_stream_r1_mixed_kv_view",
            side_effect=AssertionError("packed backend should use segmented K/V"),
        ):
            output = run_wan_s2v_stream_r1_cached_self_attention(
                FailingAttention(),
                query=query,
                key=key,
                value=value,
                kv_cache=cache,
                layout=layout,
                cache_start=None,
            )

        self.assertEqual(output.shape, query.shape)

    def test_cached_branch_falls_back_to_dense_attention_for_sp_packed_backend(self):
        cache = self._cache(tokens=4)
        calls = []

        def recording_attention(query, key, value, attn_mask=None, **kwargs):
            calls.append(
                {
                    "attn_mask": attn_mask.clone(),
                    "kwargs": dict(kwargs),
                }
            )
            return query + 10

        query = torch.zeros(1, 2, 1, 1)
        key_local = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
        value_local = key_local + 100
        key_padded_global = torch.tensor([0.0, 1.0, 100.0, 999.0]).view(1, 4, 1, 1)
        value_padded_global = key_padded_global + 100

        def fake_all_gather(tensor, dim):
            if torch.equal(tensor, key_local):
                return key_padded_global
            if torch.equal(tensor, value_local):
                return value_padded_global
            raise AssertionError("unexpected tensor gathered")

        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        with patch.dict(
            "os.environ",
            {"SGLANG_STREAM_R1_ATTENTION_BACKEND": "packed_varlen"},
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "get_sp_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "sequence_model_parallel_all_gather",
            side_effect=fake_all_gather,
        ):
            output = run_wan_s2v_stream_r1_cached_self_attention(
                recording_attention,
                query=query,
                key=key_local,
                value=value_local,
                kv_cache=cache,
                layout=layout,
                cache_start=None,
                sequence_shard_enabled=True,
                sp_pad_tokens=1,
            )

        torch.testing.assert_close(output, query + 10)
        self.assertEqual(calls[-1]["kwargs"], {"kv_is_replicated": True})
        self.assertEqual(calls[-1]["attn_mask"].shape, (1, 4, 3))

    def test_cached_branch_uses_head_sharded_packed_backend_for_sp(self):
        cache = {
            "k": torch.zeros(1, 4, 1, 2),
            "v": torch.zeros(1, 4, 1, 2),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }

        class FailingAttention:
            softmax_scale = 1.0

            def __call__(self, *args, **kwargs):
                raise AssertionError("dense attention should not be called")

        query_local = torch.randn(1, 2, 2, 2)
        key_local = torch.randn(1, 2, 2, 2)
        value_local = torch.randn(1, 2, 2, 2)
        query_global = torch.randn(1, 4, 1, 2)
        key_global = torch.randn(1, 4, 1, 2)
        value_global = torch.randn(1, 4, 1, 2)
        output_local = torch.randn(1, 2, 2, 2)
        output_all_to_all_inputs = []

        def fake_qkv_all_to_all(query, key, value):
            torch.testing.assert_close(query, query_local)
            torch.testing.assert_close(key, key_local)
            torch.testing.assert_close(value, value_local)
            return query_global, key_global, value_global

        def fake_output_all_to_all(output, head_dim):
            self.assertEqual(head_dim, 2)
            output_all_to_all_inputs.append(output.detach().clone())
            return output_local

        def fail_all_gather(*args, **kwargs):
            raise AssertionError("SP packed backend should not all-gather K/V")

        layout = WanS2VStreamR1AttentionLayout(
            noisy_seq_len=2,
            total_seq_len=3,
            frame_seq_length=1,
            num_frame_per_block=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        with patch.dict(
            "os.environ",
            {"SGLANG_STREAM_R1_ATTENTION_BACKEND": "packed_varlen"},
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "get_sp_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "_usp_input_all_to_all_qkv",
            side_effect=fake_qkv_all_to_all,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "_usp_output_all_to_all",
            side_effect=fake_output_all_to_all,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "sequence_model_parallel_all_gather",
            side_effect=fail_all_gather,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "compose_wan_s2v_stream_r1_mixed_kv_view",
            side_effect=AssertionError("SP packed backend should use segmented K/V"),
        ):
            output = run_wan_s2v_stream_r1_cached_self_attention(
                FailingAttention(),
                query=query_local,
                key=key_local,
                value=value_local,
                kv_cache=cache,
                layout=layout,
                cache_start=None,
                sequence_shard_enabled=True,
                sp_pad_tokens=1,
            )

        torch.testing.assert_close(output, output_local)
        self.assertEqual(cache["global_end_index"].item(), 2)
        self.assertEqual(cache["k"].shape, (1, 4, 1, 2))
        self.assertEqual(len(output_all_to_all_inputs), 1)
        self.assertEqual(output_all_to_all_inputs[0].shape, (1, 4, 1, 2))
        torch.testing.assert_close(
            output_all_to_all_inputs[0][:, 3:],
            torch.zeros(1, 1, 1, 2),
        )

    def test_cached_branch_requires_cache_and_layout(self):
        query = torch.zeros(1, 2, 1, 1)
        key, value = self._indexed_kv([0, 1])

        with self.assertRaisesRegex(ValueError, "kv_cache"):
            run_wan_s2v_stream_r1_cached_self_attention(
                lambda q, k, v, attn_mask=None: q,
                query=query,
                key=key,
                value=value,
                kv_cache=None,
                layout=WanS2VStreamR1AttentionLayout(
                    noisy_seq_len=2,
                    total_seq_len=2,
                    frame_seq_length=1,
                    num_frame_per_block=2,
                    local_attn_size=2,
                    sink_size=0,
                ),
                cache_start=None,
            )

        with self.assertRaisesRegex(ValueError, "attention layout"):
            run_wan_s2v_stream_r1_cached_self_attention(
                lambda q, k, v, attn_mask=None: q,
                query=query,
                key=key,
                value=value,
                kv_cache=self._cache(tokens=2),
                layout=None,
                cache_start=None,
            )


class TestWanS2VStreamR1ForwardCacheValidation(unittest.TestCase):
    def test_forward_cache_accepts_matching_stream_r1_self_cache(self):
        validate_wan_s2v_stream_r1_forward_cache(
            kv_cache=[{}, {}],
            crossattn_cache=None,
            stream_r1_mode=True,
            num_transformer_blocks=2,
        )

    def test_forward_cache_allows_legacy_no_cache_arguments(self):
        validate_wan_s2v_stream_r1_forward_cache(
            kv_cache=None,
            crossattn_cache=None,
            stream_r1_mode=False,
            num_transformer_blocks=2,
        )

    def test_forward_cache_rejects_self_cache_outside_stream_r1_mode(self):
        with self.assertRaisesRegex(ValueError, "stream_r1_mode=True"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=[{}],
                crossattn_cache=None,
                stream_r1_mode=False,
                num_transformer_blocks=1,
            )

    def test_forward_cache_rejects_non_list_self_cache(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=({},),
                crossattn_cache=None,
                stream_r1_mode=True,
                num_transformer_blocks=1,
            )

    def test_forward_cache_rejects_self_cache_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "kv_cache length"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=[{}],
                crossattn_cache=None,
                stream_r1_mode=True,
                num_transformer_blocks=2,
            )

    def test_forward_cache_accepts_matching_crossattn_cache(self):
        validate_wan_s2v_stream_r1_forward_cache(
            kv_cache=None,
            crossattn_cache=[{}, {}],
            stream_r1_mode=True,
            num_transformer_blocks=2,
        )

    def test_forward_cache_rejects_crossattn_cache_outside_stream_r1_mode(self):
        with self.assertRaisesRegex(ValueError, "stream_r1_mode=True"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=None,
                crossattn_cache=[{}],
                stream_r1_mode=False,
                num_transformer_blocks=1,
            )

    def test_forward_cache_rejects_non_list_crossattn_cache(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=None,
                crossattn_cache=({},),
                stream_r1_mode=True,
                num_transformer_blocks=1,
            )

    def test_forward_cache_rejects_crossattn_cache_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "crossattn_cache length"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=None,
                crossattn_cache=[{}],
                stream_r1_mode=True,
                num_transformer_blocks=2,
            )

    def test_forward_cache_rejects_non_dict_crossattn_cache_entry(self):
        with self.assertRaisesRegex(ValueError, "entries must be dicts"):
            validate_wan_s2v_stream_r1_forward_cache(
                kv_cache=None,
                crossattn_cache=[None],
                stream_r1_mode=True,
                num_transformer_blocks=1,
            )


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

    def test_update_plan_describes_append_and_cache_start(self):
        cache = self._cache(tokens=6)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=2,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )

        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(cache, update)

        self.assertEqual(plan.input_global_end, 0)
        self.assertEqual(plan.effective_global_end, 0)
        self.assertEqual(plan.append_tokens, 4)
        self.assertFalse(plan.evict)
        self.assertEqual(plan.cache_local_end, 4)
        self.assertEqual(plan.local_write_start, 0)
        self.assertEqual(plan.local_write_end, 4)
        self.assertEqual(plan.view_kind, "prefix")
        self.assertEqual(plan.view_local_end_index, 4)
        self.assertEqual(plan.local_start, 2)

        cache = self._cache(tokens=4)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=4,
            sink_size=1,
            current_start=4,
            cache_start=4,
        )

        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(cache, update)

        self.assertEqual(plan.input_global_end, 0)
        self.assertEqual(plan.effective_global_end, 4)
        self.assertEqual(plan.append_tokens, 2)
        self.assertEqual(plan.cache_local_end, 2)
        self.assertEqual(plan.local_write_start, 0)
        self.assertEqual(plan.local_write_end, 2)

    def test_update_plan_describes_eviction_roll_and_sink_compression(self):
        cache = self._cache(tokens=3)
        for current_start in (0, 1, 2):
            key, value = self._kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=3,
        )

        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(cache, update)

        self.assertTrue(plan.evict)
        self.assertEqual(plan.num_evicted_tokens, 1)
        self.assertEqual(plan.num_rolled_tokens, 1)
        self.assertEqual(plan.evicted_start, 1)
        self.assertEqual(plan.evicted_end, 2)
        self.assertEqual(plan.roll_src_start, 2)
        self.assertEqual(plan.roll_src_end, 3)
        self.assertEqual(plan.roll_dst_start, 1)
        self.assertEqual(plan.roll_dst_end, 2)
        self.assertEqual(plan.cache_local_end, 3)
        self.assertEqual(plan.local_write_start, 2)
        self.assertEqual(plan.local_write_end, 3)
        self.assertEqual(plan.view_kind, "prefix")
        self.assertEqual(plan.view_local_end_index, 3)
        self.assertTrue(plan.sink_compress)

    def test_update_accepts_explicit_plan(self):
        cache = self._cache(tokens=3)
        key, value = self._kv(0, 1)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(cache, update)

        view = update_wan_s2v_stream_r1_noisy_kv_cache(
            cache, key, value, update, plan=plan
        )

        self.assertEqual(view.global_end_index, plan.new_global_end)
        self.assertEqual(view.local_end_index, plan.new_local_end_index)
        torch.testing.assert_close(view.key[:, :, 0, 0], torch.tensor([[0.0]]))

    def test_update_plan_buffer_copies_host_plan_and_scalar_metadata(self):
        cache = self._cache(tokens=3)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=0,
        )
        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(cache, update)

        buffer = WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer.from_plan(
            plan,
            device=torch.device("cpu"),
        )

        self.assertIs(buffer.host_plan, plan)
        snapshot = buffer.scalar_snapshot()
        self.assertEqual(snapshot["input_global_end"], 0)
        self.assertEqual(snapshot["input_local_end"], 0)
        self.assertEqual(snapshot["local_write_start"], plan.local_write_start)
        self.assertEqual(snapshot["local_write_end"], plan.local_write_end)
        self.assertEqual(snapshot["new_global_end"], plan.new_global_end)
        self.assertEqual(snapshot["new_local_end_index"], plan.new_local_end_index)
        self.assertEqual(buffer.scalar_tensor("kv_start").item(), plan.kv_start)

    def test_graph_plan_buffer_update_matches_eager_append_window(self):
        eager_cache = self._cache(tokens=3)
        for current_start in (0, 1):
            key, value = self._kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            update_wan_s2v_stream_r1_noisy_kv_cache(eager_cache, key, value, update)
        graph_cache = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in eager_cache.items()
        }
        key, value = self._kv(2, 1)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=2,
        )
        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(graph_cache, update)
        plan_buffer = WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer.from_plan(
            plan,
            device=torch.device("cpu"),
        )

        eager_view = update_wan_s2v_stream_r1_noisy_kv_cache(
            eager_cache,
            key,
            value,
            update,
        )
        graph_view = update_wan_s2v_stream_r1_noisy_kv_cache_with_plan_buffer(
            graph_cache,
            key,
            value,
            update,
            plan_buffer,
        )

        torch.testing.assert_close(graph_cache["k"], eager_cache["k"])
        torch.testing.assert_close(graph_cache["v"], eager_cache["v"])
        torch.testing.assert_close(graph_view.key, eager_view.key)
        torch.testing.assert_close(graph_view.value, eager_view.value)
        self.assertEqual(graph_cache["global_end_index"].item(), 3)
        self.assertEqual(graph_cache["local_end_index"].item(), 3)

    def test_graph_plan_buffer_update_matches_eager_eviction_window(self):
        eager_cache = self._cache(tokens=3)
        for current_start in (0, 1, 2):
            key, value = self._kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            update_wan_s2v_stream_r1_noisy_kv_cache(eager_cache, key, value, update)
        graph_cache = {
            name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in eager_cache.items()
        }
        key, value = self._kv(3, 1)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=1,
            frame_seq_length=1,
            local_attn_size=3,
            sink_size=1,
            current_start=3,
        )
        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(graph_cache, update)
        plan_buffer = WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer.from_plan(
            plan,
            device=torch.device("cpu"),
        )

        eager_view = update_wan_s2v_stream_r1_noisy_kv_cache(
            eager_cache,
            key,
            value,
            update,
        )
        graph_view = update_wan_s2v_stream_r1_noisy_kv_cache_with_plan_buffer(
            graph_cache,
            key,
            value,
            update,
            plan_buffer,
        )

        torch.testing.assert_close(graph_cache["k"], eager_cache["k"])
        torch.testing.assert_close(graph_cache["v"], eager_cache["v"])
        torch.testing.assert_close(graph_view.key, eager_view.key)
        torch.testing.assert_close(graph_view.value, eager_view.value)
        self.assertEqual(graph_cache["global_end_index"].item(), 4)
        self.assertEqual(graph_cache["local_end_index"].item(), 3)

    def test_update_plan_uses_external_state_owner(self):
        cache = self._cache(tokens=3)
        state = WanS2VStreamR1KVState()

        for current_start in (0, 1):
            key, value = self._kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(
                cache,
                update,
                state=state,
            )

            view = update_wan_s2v_stream_r1_noisy_kv_cache(
                cache,
                key,
                value,
                update,
                plan=plan,
                commit_host_state=False,
            )

            self.assertEqual(state.global_end_index, current_start)
            self.assertNotIn("global_end_index_host", cache)
            state.apply_noisy_kv_cache_update_plan(plan)
            self.assertEqual(view.global_end_index, state.global_end_index)
            self.assertEqual(view.local_end_index, state.local_end_index)

        self.assertEqual(state.global_end_index, 2)
        self.assertEqual(state.local_end_index, 2)
        self.assertEqual(cache["global_end_index"].item(), 2)
        self.assertEqual(cache["local_end_index"].item(), 2)

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

        view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

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

    def test_update_compresses_evicted_tokens_into_sink_and_rolls_local_window(self):
        cache = self._cache(tokens=3)

        for current_start in (0, 1, 2, 3):
            key, value = self._kv(current_start, 1)
            update = WanS2VStreamR1NoisyKVCacheUpdate(
                noisy_seq_len=1,
                frame_seq_length=1,
                local_attn_size=3,
                sink_size=1,
                current_start=current_start,
            )
            view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        self.assertEqual(view.global_end_index, 4)
        self.assertEqual(view.local_end_index, 3)
        self.assertEqual(view.local_start, 2)
        torch.testing.assert_close(
            view.key[:, :, 0, 0],
            torch.tensor([[0.001, 2.0, 3.0]]),
        )
        torch.testing.assert_close(
            view.value[:, :, 0, 0],
            torch.tensor([[100.001, 102.0, 103.0]]),
        )

    def test_update_supports_cache_start_offset(self):
        cache = self._cache(tokens=4)
        key, value = self._kv(4, 2)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=4,
            sink_size=1,
            current_start=4,
            cache_start=4,
        )

        view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        self.assertEqual(view.global_end_index, 6)
        self.assertEqual(view.local_end_index, 2)
        torch.testing.assert_close(view.key[:, :, 0, 0], torch.tensor([[4.0, 5.0]]))

        key, value = self._kv(6, 2)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=2,
            frame_seq_length=1,
            local_attn_size=4,
            sink_size=1,
            current_start=6,
            cache_start=4,
        )
        view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        self.assertEqual(view.global_end_index, 8)
        self.assertEqual(view.local_end_index, 4)
        self.assertEqual(view.local_start, 5)
        torch.testing.assert_close(
            view.key[:, :, 0, 0],
            torch.tensor([[4.0, 5.0, 6.0, 7.0]]),
        )

    def test_update_reuses_contiguous_sink_view_without_cat(self):
        cache = self._cache(tokens=8)
        key, value = self._kv(0, 4)
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=4,
            frame_seq_length=2,
            local_attn_size=4,
            sink_size=1,
            current_start=0,
        )

        view = update_wan_s2v_stream_r1_noisy_kv_cache(cache, key, value, update)

        self.assertEqual(view.local_end_index, 4)
        self.assertEqual(view.key.data_ptr(), cache["k"].data_ptr())
        self.assertEqual(view.value.data_ptr(), cache["v"].data_ptr())
        cache["k"][0, 0, 0, 0] = 123.0
        cache["v"][0, 0, 0, 0] = 456.0
        self.assertEqual(view.key[0, 0, 0, 0].item(), 123.0)
        self.assertEqual(view.value[0, 0, 0, 0].item(), 456.0)

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
            update_wan_s2v_stream_r1_noisy_kv_cache(small_cache, key, value, update)


class TestWanS2VStreamR1DenoisingStage(unittest.TestCase):
    class _RecordingTransformer:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["hidden_states"]

    class _ForwardContextRecordingTransformer:
        def __init__(self):
            self.calls = []
            self.forward_batch = None

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            self.forward_batch = get_forward_context().forward_batch
            return kwargs["hidden_states"]

    class _KVCapableBlock:
        local_num_heads = 3
        dim_head = 8

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cis,
            attn_mask=None,
            stream_r1_kv_cache=None,
            stream_r1_attention_layout=None,
            cache_start=None,
            stream_r1_sequence_shard_enabled=False,
            stream_r1_sp_pad_tokens=0,
            crossattn_kv_cache=None,
        ):
            return hidden_states

    class _KVCapableTransformer:
        def __init__(self):
            self.config = SimpleNamespace(arch_config=SimpleNamespace(num_layers=2))
            self.blocks = [
                TestWanS2VStreamR1DenoisingStage._KVCapableBlock(),
                TestWanS2VStreamR1DenoisingStage._KVCapableBlock(),
            ]
            self.hidden_size = 48
            self.num_attention_heads = 6
            self.use_context_parallel = False

        def set_stream_r1_attention(
            self,
            local_attn_size,
            sink_size,
            *,
            num_frame_per_block=None,
            kv_cache=False,
        ):
            self.stream_r1_local_attn_size = local_attn_size
            self.stream_r1_sink_size = sink_size
            self.stream_r1_num_frame_per_block = num_frame_per_block
            self.stream_r1_kv_cache_requested = kv_cache

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep=None,
            kv_cache=None,
            current_start=0,
            cache_start=None,
            stream_r1_mode=False,
            **kwargs,
        ):
            return hidden_states

    class _RecordingAudioEncoder:
        def __init__(self):
            self.calls = []

        def __call__(self, audio_input):
            self.calls.append(audio_input.detach().clone())
            return {"encoded_call": len(self.calls)}

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
        stage._adaptive_prev_audio_feature = None
        stage._adaptive_prev_request_id = None
        stage._adaptive_total_blocks = 0
        stage._adaptive_reduced_blocks = 0
        stage._last_adaptive_step_decision = None
        stage._last_timestep_profile_rows = []
        stage._timestep_cuda_graph_runner = _WanS2VTransformerTimestepCudaGraphRunner()
        stage.crossattn_cache = None
        return stage

    def _metadata(self) -> WanS2VStreamR1CacheMetadata:
        return WanS2VStreamR1CacheMetadata(
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

    def _attention_request(
        self,
        *,
        stream_r1_kv_cache: bool,
        context_noise: int = 0,
    ) -> WanS2VStreamR1AttentionRequest:
        return WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=stream_r1_kv_cache,
            num_frame_per_block=4,
            local_attn_size=4,
            sink_size=1,
            context_noise=context_noise,
        )

    def _block_bundle(self) -> WanS2VConditionBundle:
        return WanS2VConditionBundle(
            prompt_embeds=torch.zeros(2, 3, 4),
            ref_latents=torch.ones(2, 3, 1, 2, 2),
            motion_latents=torch.full((2, 3, 2, 2, 2), 2.0),
            cond_states=torch.full((2, 3, 4, 2, 2), 3.0),
            audio_input=torch.full((2, 4, 5, 16), 4.0),
            audio_emb={"audio": "cached"},
            motion_frames=(73, 19),
            add_last_motion=2,
            drop_motion_frames=False,
        )

    def _audio_cache_bundle(
        self,
        *,
        cache_audio_embeddings: bool = True,
        audio_emb=None,
    ) -> WanS2VConditionBundle:
        return WanS2VConditionBundle(
            prompt_embeds=torch.zeros(1, 3, 4),
            ref_latents=torch.ones(1, 3, 1, 2, 2),
            motion_latents=torch.full((1, 3, 2, 2, 2), 2.0),
            cond_states=torch.full((1, 3, 4, 2, 2), 3.0),
            audio_input=torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4),
            audio_emb=audio_emb,
            motion_frames=(3, 1),
            add_last_motion=2,
            drop_motion_frames=False,
            audio_metadata={"cache_audio_embeddings": cache_audio_embeddings},
        )

    def _graph_kwargs(self):
        return {
            "hidden_states": torch.zeros(1, 3, 1, 2, 2),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": torch.zeros(1, 2, 4),
            "ref_latents": torch.zeros(1, 3, 1, 2, 2),
            "motion_latents": torch.zeros(1, 3, 2, 2, 2),
            "cond_states": torch.zeros(1, 3, 4, 2, 2),
            "audio_input": torch.zeros(1, 4, 5, 16),
            "audio_emb": None,
            "motion_frames": (73, 19),
            "add_last_motion": 2,
            "drop_motion_frames": False,
            "kv_cache": None,
            "crossattn_cache": None,
            "current_start": 0,
            "cache_start": None,
            "audio_start_frame": 0,
            "stream_r1_mode": True,
        }

    def test_timestep_metadata_buffers_update_without_reallocation(self):
        kwargs = self._graph_kwargs()
        plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )
        buffers = WanS2VTimestepStaticMetadataBuffers.from_plan(
            plan,
            device=torch.device("cpu"),
        )
        scalar_ptr = buffers.scalar_values.data_ptr()
        motion_ptr = buffers.motion_frames.data_ptr()

        next_plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs={**kwargs, "cache_start": 4, "drop_motion_frames": True},
            step_index=3,
            current_start=12,
            audio_start_frame=1,
            sequence_shard_enabled=True,
        )
        buffers.copy_from_plan_(next_plan)

        self.assertEqual(buffers.scalar_values.data_ptr(), scalar_ptr)
        self.assertEqual(buffers.motion_frames.data_ptr(), motion_ptr)
        snapshot = buffers.scalar_snapshot()
        self.assertEqual(snapshot["step_index"], 3)
        self.assertEqual(snapshot["current_start"], 12)
        self.assertEqual(snapshot["cache_start"], 4)
        self.assertEqual(snapshot["audio_start_frame"], 1)
        self.assertEqual(snapshot["sequence_shard_enabled"], 1)
        self.assertEqual(snapshot["drop_motion_frames"], 1)
        torch.testing.assert_close(
            buffers.motion_frames.cpu(),
            torch.tensor([73, 19], dtype=torch.long),
        )

    def test_timestep_metadata_forward_kwargs_use_host_plan_mirror(self):
        kwargs = self._graph_kwargs()
        plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs={**kwargs, "cache_start": 4},
            step_index=0,
            current_start=12,
            audio_start_frame=3,
            sequence_shard_enabled=False,
        )
        buffers = WanS2VTimestepStaticMetadataBuffers.from_plan(
            plan,
            device=torch.device("cpu"),
        )
        buffers.scalar_values.fill_(-999)
        buffers.motion_frames.fill_(-999)

        forward_kwargs = buffers.to_forward_kwargs()

        self.assertEqual(forward_kwargs["current_start"], 12)
        self.assertEqual(forward_kwargs["cache_start"], 4)
        self.assertEqual(forward_kwargs["audio_start_frame"], 3)
        self.assertEqual(forward_kwargs["motion_frames"], (73, 19))

    def test_timestep_graph_key_uses_structure_not_dynamic_offsets(self):
        kwargs = self._graph_kwargs()
        device = torch.device("cpu")
        plan_step0 = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )
        plan_step1 = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=1,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )
        plan_next_block = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=0,
            current_start=12,
            audio_start_frame=1,
            sequence_shard_enabled=False,
        )
        plan_drop_motion = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs={**kwargs, "drop_motion_frames": True},
            step_index=0,
            current_start=12,
            audio_start_frame=1,
            sequence_shard_enabled=False,
        )

        key_step0 = _WanS2VTransformerTimestepCudaGraphRunner.make_key(
            kwargs=kwargs,
            metadata_plan=plan_step0,
            metadata_device=device,
        )
        key_step1 = _WanS2VTransformerTimestepCudaGraphRunner.make_key(
            kwargs=kwargs,
            metadata_plan=plan_step1,
            metadata_device=device,
        )
        key_next_block = _WanS2VTransformerTimestepCudaGraphRunner.make_key(
            kwargs=kwargs,
            metadata_plan=plan_next_block,
            metadata_device=device,
        )
        key_drop_motion = _WanS2VTransformerTimestepCudaGraphRunner.make_key(
            kwargs=kwargs,
            metadata_plan=plan_drop_motion,
            metadata_device=device,
        )

        self.assertEqual(key_step0, key_step1)
        self.assertEqual(key_step0, key_next_block)
        self.assertNotEqual(key_step0, key_drop_motion)

    def test_timestep_graph_runner_preslices_audio_embeddings(self):
        kwargs = {
            **self._graph_kwargs(),
            "hidden_states": torch.zeros(1, 3, 2, 2, 2),
            "audio_input": None,
            "audio_emb": (
                torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2),
                torch.arange(100, 100 + 1 * 8 * 3, dtype=torch.float32).reshape(
                    1, 8, 3
                ),
            ),
            "motion_frames": (3, 2),
            "audio_start_frame": 1,
        }
        plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=0,
            current_start=0,
            audio_start_frame=1,
            sequence_shard_enabled=False,
        )

        graph_kwargs = _WanS2VTransformerTimestepCudaGraphRunner.prepare_graph_kwargs(
            kwargs,
            plan,
        )

        self.assertTrue(graph_kwargs["stream_r1_audio_emb_pre_sliced"])
        audio_global, audio_local = graph_kwargs["audio_emb"]
        torch.testing.assert_close(audio_global, kwargs["audio_emb"][0][:, 3:5])
        torch.testing.assert_close(audio_local, kwargs["audio_emb"][1][:, 3:5])

    def test_timestep_graph_runner_binds_plan_buffer_forward(self):
        kwargs = self._graph_kwargs()
        plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs={**kwargs, "cache_start": 4},
            step_index=0,
            current_start=12,
            audio_start_frame=3,
            sequence_shard_enabled=False,
        )
        buffers = WanS2VTimestepStaticMetadataBuffers.from_plan(
            plan,
            device=torch.device("cpu"),
        )
        calls = []

        class _Forward:
            def __call__(self, **call_kwargs):
                calls.append(("direct", call_kwargs))
                return call_kwargs["hidden_states"]

            def forward_with_plan_buffers(
                self,
                *,
                timestep_metadata_buffers,
                **call_kwargs,
            ):
                call_kwargs.update(timestep_metadata_buffers.to_forward_kwargs())
                calls.append(("plan", call_kwargs))
                return call_kwargs["hidden_states"] + 1

        bound_forward = (
            _WanS2VTransformerTimestepCudaGraphRunner.bind_forward_with_metadata(
                _Forward(),
                buffers,
            )
        )
        out = bound_forward(**kwargs)

        torch.testing.assert_close(out, kwargs["hidden_states"] + 1)
        self.assertEqual(calls[0][0], "plan")
        self.assertEqual(calls[0][1]["current_start"], 12)
        self.assertEqual(calls[0][1]["cache_start"], 4)

    def test_timestep_graph_runner_captures_on_key_miss_when_allowed(self):
        kwargs = self._graph_kwargs()
        runner = _WanS2VTransformerTimestepCudaGraphRunner()
        captured = []
        replayed = []

        def _capture_one(
            *,
            forward_fn,
            static_kwargs,
        ):
            captured.append(static_kwargs)
            return object(), forward_fn(**static_kwargs)

        def _replay(entry):
            replayed.append(entry)
            return entry.output + 1

        runner.backend = SimpleNamespace(
            capture_one=_capture_one,
            replay=_replay,
        )

        class _Forward:
            def __call__(self, **call_kwargs):
                return call_kwargs["hidden_states"] + 2

        output, status = runner.run(
            kwargs=kwargs,
            forward_fn=_Forward(),
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
            allow_capture=True,
        )

        self.assertEqual(status, "capture")
        self.assertEqual(runner.cached_graph_count, 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(replayed), 1)
        torch.testing.assert_close(
            output,
            kwargs["hidden_states"] + 3,
        )

    def test_timestep_graph_runner_rejects_key_miss_when_capture_disabled(self):
        kwargs = self._graph_kwargs()
        runner = _WanS2VTransformerTimestepCudaGraphRunner()

        with self.assertRaisesRegex(
            RuntimeError,
            "capture is disabled",
        ):
            runner.run(
                kwargs=kwargs,
                forward_fn=lambda **call_kwargs: call_kwargs["hidden_states"],
                step_index=0,
                current_start=0,
                audio_start_frame=0,
                sequence_shard_enabled=False,
                allow_capture=False,
            )

    def test_timestep_graph_runner_falls_back_on_kv_plan_mismatch(self):
        metadata = WanS2VStreamR1CacheMetadata(
            batch_size=1,
            num_layers=1,
            frame_seq_length=1,
            local_num_attention_heads=1,
            attention_head_dim=1,
            local_attn_size=4,
            sink_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache_state = WanS2VStreamR1CacheState.allocate(metadata)
        cache_state.prepare_kv_update_plans(noisy_seq_len=1, current_start=0)
        kwargs = {**self._graph_kwargs(), "kv_cache": cache_state.kv_cache}
        runner = _WanS2VTransformerTimestepCudaGraphRunner()
        runner.backend = SimpleNamespace(replay=lambda entry: entry.output)
        runner.graphs[("key",)] = SimpleNamespace(
            static_metadata=WanS2VTimestepStaticMetadataBuffers.from_plan(
                WanS2VTimestepMetadataPlan.from_kwargs(
                    kwargs=kwargs,
                    step_index=0,
                    current_start=0,
                    audio_start_frame=0,
                    sequence_shard_enabled=False,
                ),
                device=torch.device("cpu"),
            ),
            static_inputs=SimpleNamespace(copy_from_live_kwargs_=lambda kwargs: None),
            output=kwargs["hidden_states"] + 1,
            kv_update_plan_signature=(("stale",),),
        )
        runner.make_key = lambda **_: ("key",)
        calls = []

        class _Forward:
            def forward_with_plan_buffers(self, **call_kwargs):
                calls.append(call_kwargs)
                return call_kwargs["hidden_states"] + 2

        output, status = runner.run(
            kwargs=kwargs,
            forward_fn=_Forward(),
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )

        self.assertEqual(status, "eager_kv_plan_mismatch")
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(output, kwargs["hidden_states"] + 2)

    def test_timestep_graph_runner_replay_commits_matching_kv_plan(self):
        metadata = WanS2VStreamR1CacheMetadata(
            batch_size=1,
            num_layers=1,
            frame_seq_length=1,
            local_num_attention_heads=1,
            attention_head_dim=1,
            local_attn_size=4,
            sink_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache_state = WanS2VStreamR1CacheState.allocate(metadata)
        cache_state.prepare_kv_update_plans(noisy_seq_len=1, current_start=0)
        kwargs = {**self._graph_kwargs(), "kv_cache": cache_state.kv_cache}
        runner = _WanS2VTransformerTimestepCudaGraphRunner()
        metadata_plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )
        kv_signature = runner.kv_update_plan_signature(cache_state.kv_cache)
        replayed = []

        def _replay(entry):
            replayed.append(entry)
            return entry.output

        runner.backend = SimpleNamespace(replay=_replay)
        runner.graphs[("key",)] = SimpleNamespace(
            static_metadata=WanS2VTimestepStaticMetadataBuffers.from_plan(
                metadata_plan,
                device=torch.device("cpu"),
            ),
            static_inputs=SimpleNamespace(copy_from_live_kwargs_=lambda kwargs: None),
            output=kwargs["hidden_states"] + 1,
            kv_update_plan_signature=kv_signature,
        )
        runner.make_key = lambda **_: ("key",)

        output, status = runner.run(
            kwargs=kwargs,
            forward_fn=object(),
            step_index=0,
            current_start=0,
            audio_start_frame=0,
            sequence_shard_enabled=False,
        )

        self.assertEqual(status, "replay")
        self.assertEqual(len(replayed), 1)
        self.assertEqual(cache_state.kv_states[0].global_end_index, 1)
        self.assertEqual(cache_state.kv_states[0].local_end_index, 1)
        self.assertEqual(cache_state.kv_cache[0]["global_end_index_host"], 1)
        self.assertEqual(cache_state.kv_cache[0]["local_end_index_host"], 1)
        torch.testing.assert_close(output, kwargs["hidden_states"] + 1)

    def test_transformer_forward_with_plan_buffers_passes_metadata_to_forward(self):
        kwargs = self._graph_kwargs()
        plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs={
                **kwargs,
                "cache_start": 4,
                "motion_frames": (3, 1),
                "add_last_motion": 5,
                "drop_motion_frames": True,
                "stream_r1_mode": False,
            },
            step_index=2,
            current_start=12,
            audio_start_frame=7,
            sequence_shard_enabled=False,
        )
        buffers = WanS2VTimestepStaticMetadataBuffers.from_plan(
            plan,
            device=torch.device("cpu"),
        )
        model = WanS2VTransformer3DModel.__new__(WanS2VTransformer3DModel)
        calls = []

        def fake_forward(**forward_kwargs):
            calls.append(forward_kwargs)
            return forward_kwargs["hidden_states"] + 1

        model.forward = fake_forward
        out = WanS2VTransformer3DModel.forward_with_plan_buffers(
            model,
            timestep_metadata_buffers=buffers,
            **{
                **kwargs,
                "current_start": 0,
                "cache_start": None,
                "audio_start_frame": 0,
                "motion_frames": (73, 19),
                "add_last_motion": 2,
                "drop_motion_frames": False,
                "stream_r1_mode": True,
            },
        )

        torch.testing.assert_close(out, kwargs["hidden_states"] + 1)
        self.assertIs(calls[0]["timestep_metadata_buffers"], buffers)
        self.assertEqual(calls[0]["current_start"], 0)
        self.assertIsNone(calls[0]["cache_start"])
        self.assertEqual(calls[0]["audio_start_frame"], 0)
        self.assertEqual(calls[0]["motion_frames"], (73, 19))
        self.assertEqual(calls[0]["add_last_motion"], 2)
        self.assertFalse(calls[0]["drop_motion_frames"])
        self.assertTrue(calls[0]["stream_r1_mode"])

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
        self.assertEqual(len(state.kv_states), 2)
        self.assertIs(state.kv_cache[0]["state"], state.kv_states[0])
        self.assertIs(state.kv_cache[1]["state"], state.kv_states[1])
        self.assertIsInstance(
            state.kv_cache[0]["update_plan_buffer"],
            WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer,
        )
        state.prepare_kv_update_plans(noisy_seq_len=5, current_start=0)
        plan_buffer = state.kv_cache[0]["update_plan_buffer"]
        self.assertIsNotNone(plan_buffer.host_plan)
        self.assertEqual(plan_buffer.host_plan.update.noisy_seq_len, 5)
        self.assertEqual(plan_buffer.scalar_snapshot()["new_global_end"], 5)
        state.kv_cache[0]["global_end_index"].fill_(7)
        state.kv_cache[1]["local_end_index"].fill_(9)
        state.kv_states[0].global_end_index = 7
        state.kv_states[1].local_end_index = 9
        state.reset()

        self.assertTrue(state.enabled)
        self.assertTrue(state.allocated)
        self.assertEqual(state.kv_cache[0]["k"].shape, (2, 20, 3, 8))
        self.assertEqual(state.kv_cache[0]["v"].shape, (2, 20, 3, 8))
        self.assertEqual(state.kv_cache[0]["global_end_index"].item(), 0)
        self.assertEqual(state.kv_cache[1]["local_end_index"].item(), 0)
        self.assertEqual(state.kv_states[0].global_end_index, 0)
        self.assertEqual(state.kv_states[1].local_end_index, 0)
        self.assertIsNone(state.kv_cache[0]["update_plan_buffer"].host_plan)

    def test_crossattn_cache_is_stage_owned_and_marked_for_refresh(self):
        stage = self._stage()

        first_cache = stage._prepare_request_crossattn_cache(True)
        self.assertIs(first_cache, stage.crossattn_cache)
        self.assertEqual(len(first_cache), 2)
        self.assertTrue(all(item["needs_update"] for item in first_cache))

        cached_k = torch.zeros(1, 4, 2, 3)
        cached_v = torch.ones(1, 4, 2, 3)
        first_cache[0]["k"] = cached_k
        first_cache[0]["v"] = cached_v
        first_cache[0]["needs_update"] = False

        second_cache = stage._prepare_request_crossattn_cache(True)
        self.assertIs(second_cache, first_cache)
        self.assertIs(second_cache[0]["k"], cached_k)
        self.assertIs(second_cache[0]["v"], cached_v)
        self.assertTrue(second_cache[0]["needs_update"])
        self.assertFalse(
            _WanS2VTransformerTimestepCudaGraphRunner.crossattn_cache_ready(
                second_cache
            )
        )

        second_cache[0]["needs_update"] = False
        second_cache[1]["k"] = torch.zeros(1, 4, 2, 3)
        second_cache[1]["v"] = torch.ones(1, 4, 2, 3)
        second_cache[1]["needs_update"] = False
        self.assertTrue(
            _WanS2VTransformerTimestepCudaGraphRunner.crossattn_cache_ready(
                second_cache
            )
        )

        self.assertIsNone(stage._prepare_crossattn_cache(False))
        self.assertIsNone(stage.crossattn_cache)

    def test_kv_cache_state_prebuild_clears_discontinuous_update_plan(self):
        metadata = WanS2VStreamR1CacheMetadata(
            batch_size=1,
            num_layers=1,
            frame_seq_length=5,
            local_num_attention_heads=1,
            attention_head_dim=4,
            local_attn_size=4,
            sink_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        state = WanS2VStreamR1CacheState.allocate(metadata)
        state.prepare_kv_update_plans(noisy_seq_len=5, current_start=0)
        self.assertIsNotNone(state.kv_cache[0]["update_plan_buffer"].host_plan)

        state.reset()
        state.prepare_kv_update_plans(noisy_seq_len=5, current_start=10)

        self.assertIsNone(state.kv_cache[0]["update_plan_buffer"].host_plan)

    def test_kv_cache_state_allocates_normal_tensors_in_inference_mode(self):
        metadata = WanS2VStreamR1CacheMetadata(
            batch_size=1,
            num_layers=1,
            frame_seq_length=2,
            local_num_attention_heads=1,
            attention_head_dim=4,
            local_attn_size=2,
            sink_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        with torch.inference_mode():
            state = WanS2VStreamR1CacheState.allocate(metadata)

        self.assertFalse(state.kv_cache[0]["global_end_index"].is_inference())
        self.assertFalse(state.kv_cache[0]["k"].is_inference())
        state.kv_cache[0]["global_end_index"].fill_(3)
        state.reset()
        self.assertEqual(state.kv_cache[0]["global_end_index"].item(), 0)

    def test_stage_prepares_metadata_but_guards_unsupported_kv(self):
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
        with self.assertRaisesRegex(NotImplementedError, "does not expose"):
            stage._guard_cache_runtime(state)

    def test_stage_auto_detects_and_allocates_kv_cache(self):
        self.assertIsNone(
            WanS2VStreamR1DenoisingStage._s2v_kv_attention_kernel_supported
        )
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = None
        stage.transformer = self._KVCapableTransformer()
        request = self._attention_request(stream_r1_kv_cache=True)

        state = stage._prepare_cache_state(
            request=request,
            batch_size=2,
            frame_seq_length=5,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )

        self.assertIs(stage.cache_state, state)
        self.assertTrue(state.enabled)
        self.assertTrue(state.allocated)
        self.assertEqual(state.metadata.cache_tokens, 20)
        self.assertEqual(state.metadata.local_num_attention_heads, 3)
        self.assertEqual(len(state.kv_cache), 2)
        for block_cache in state.kv_cache:
            self.assertEqual(block_cache["k"].shape, (2, 20, 3, 8))
            self.assertEqual(block_cache["v"].shape, (2, 20, 3, 8))
            self.assertEqual(block_cache["k"].dtype, torch.float16)
            self.assertEqual(block_cache["v"].dtype, torch.float16)
            self.assertEqual(block_cache["k"].device, torch.device("cpu"))
            self.assertEqual(block_cache["global_end_index"].dtype, torch.long)
            self.assertEqual(block_cache["local_end_index"].dtype, torch.long)

        stage._guard_cache_runtime(state)

    def test_stage_allocates_head_sharded_kv_cache_for_packed_sp_backend(self):
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = True
        request = self._attention_request(stream_r1_kv_cache=True)

        with patch.dict(
            "os.environ",
            {"SGLANG_STREAM_R1_ATTENTION_BACKEND": "packed_varlen"},
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1."
            "get_sp_world_size",
            return_value=3,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.wan_s2v.get_sp_world_size",
            return_value=3,
        ):
            state = stage._prepare_cache_state(
                request=request,
                batch_size=2,
                frame_seq_length=5,
                dtype=torch.float16,
                device=torch.device("cpu"),
            )

        self.assertTrue(state.allocated)
        self.assertEqual(state.metadata.local_num_attention_heads, 1)
        for block_cache in state.kv_cache:
            self.assertEqual(block_cache["k"].shape, (2, 20, 1, 8))
            self.assertEqual(block_cache["v"].shape, (2, 20, 1, 8))

    def test_stage_override_allocates_and_forwards_kv_cache(self):
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = True
        request = self._attention_request(stream_r1_kv_cache=True)

        state = stage._prepare_cache_state(
            request=request,
            batch_size=2,
            frame_seq_length=5,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )

        self.assertIs(stage.cache_state, state)
        self.assertTrue(state.enabled)
        self.assertTrue(state.allocated)
        stage._guard_cache_runtime(state)

        recorder = self._RecordingTransformer()
        stage.transformer = recorder
        stage._clean_context_refresh(
            block_latents=torch.ones(2, 3, 4, 2, 2),
            prompt_embeds=torch.zeros(2, 3, 4),
            block_bundle=self._block_bundle(),
            current_start=15,
            attention_request=request,
            cache_state=state,
        )

        self.assertEqual(len(recorder.calls), 1)
        self.assertIs(recorder.calls[0]["kv_cache"], state.kv_cache)

    def test_clean_context_refresh_noops_when_cache_disabled(self):
        stage = self._stage()
        recorder = self._RecordingTransformer()
        stage.transformer = recorder
        block_latents = torch.ones(2, 3, 4, 2, 2)

        stage._clean_context_refresh(
            block_latents=block_latents,
            prompt_embeds=torch.zeros(2, 3, 4),
            block_bundle=self._block_bundle(),
            current_start=10,
            attention_request=self._attention_request(stream_r1_kv_cache=False),
            cache_state=WanS2VStreamR1CacheState.disabled(),
        )

        self.assertEqual(recorder.calls, [])

    def test_clean_context_refresh_keeps_unsupported_guard(self):
        stage = self._stage()
        recorder = self._RecordingTransformer()
        stage.transformer = recorder
        state = WanS2VStreamR1CacheState.metadata_only(self._metadata())

        with self.assertRaisesRegex(NotImplementedError, "does not expose"):
            stage._clean_context_refresh(
                block_latents=torch.ones(2, 3, 4, 2, 2),
                prompt_embeds=torch.zeros(2, 3, 4),
                block_bundle=self._block_bundle(),
                current_start=10,
                attention_request=self._attention_request(stream_r1_kv_cache=True),
                cache_state=state,
            )

        self.assertEqual(recorder.calls, [])

    def test_clean_context_refresh_calls_transformer_with_clean_block_kwargs(self):
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = True
        recorder = self._RecordingTransformer()
        stage.transformer = recorder
        state = WanS2VStreamR1CacheState.allocate(self._metadata())
        block_latents = torch.ones(2, 3, 4, 2, 2)
        prompt_embeds = torch.full((2, 3, 4), 5.0)
        block_bundle = self._block_bundle()

        stage._clean_context_refresh(
            block_latents=block_latents,
            prompt_embeds=prompt_embeds,
            block_bundle=block_bundle,
            current_start=15,
            attention_request=self._attention_request(
                stream_r1_kv_cache=True,
                context_noise=123,
            ),
            cache_state=state,
        )

        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual(
            set(call),
            {
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
                "ref_latents",
                "motion_latents",
                "cond_states",
                "audio_input",
                "audio_emb",
                "motion_frames",
                "add_last_motion",
                "drop_motion_frames",
                "kv_cache",
                "crossattn_cache",
                "current_start",
                "cache_start",
                "audio_start_frame",
                "stream_r1_mode",
                "stream_r1_refresh_only",
            },
        )
        self.assertIs(call["hidden_states"], block_latents)
        self.assertEqual(call["timestep"].dtype, torch.long)
        self.assertEqual(call["timestep"].tolist(), [123, 123])
        self.assertIs(call["encoder_hidden_states"], prompt_embeds)
        self.assertIs(call["ref_latents"], block_bundle.ref_latents)
        self.assertIs(call["motion_latents"], block_bundle.motion_latents)
        self.assertIs(call["cond_states"], block_bundle.cond_states)
        self.assertIs(call["audio_input"], block_bundle.audio_input)
        self.assertIs(call["audio_emb"], block_bundle.audio_emb)
        self.assertEqual(call["motion_frames"], (73, 19))
        self.assertEqual(call["add_last_motion"], 2)
        self.assertFalse(call["drop_motion_frames"])
        self.assertIs(call["kv_cache"], state.kv_cache)
        self.assertIsNone(call["crossattn_cache"])
        self.assertEqual(call["current_start"], 15)
        self.assertIsNone(call["cache_start"])
        self.assertTrue(call["stream_r1_mode"])
        self.assertTrue(call["stream_r1_refresh_only"])

    def test_clean_context_refresh_forwards_crossattn_cache(self):
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = True
        recorder = self._RecordingTransformer()
        stage.transformer = recorder
        state = WanS2VStreamR1CacheState.allocate(self._metadata())
        crossattn_cache = [{"k": object()}, {}]

        stage._clean_context_refresh(
            block_latents=torch.ones(2, 3, 4, 2, 2),
            prompt_embeds=torch.zeros(2, 3, 4),
            block_bundle=self._block_bundle(),
            current_start=15,
            attention_request=self._attention_request(stream_r1_kv_cache=True),
            cache_state=state,
            crossattn_cache=crossattn_cache,
        )

        self.assertEqual(len(recorder.calls), 1)
        self.assertIs(recorder.calls[0]["crossattn_cache"], crossattn_cache)

    def test_clean_context_refresh_sets_forward_context(self):
        stage = self._stage()
        stage._s2v_kv_attention_kernel_supported = True
        recorder = self._ForwardContextRecordingTransformer()
        stage.transformer = recorder
        state = WanS2VStreamR1CacheState.allocate(self._metadata())
        forward_batch = SimpleNamespace(enable_sequence_shard=False)

        stage._clean_context_refresh(
            block_latents=torch.ones(2, 3, 4, 2, 2),
            prompt_embeds=torch.zeros(2, 3, 4),
            block_bundle=self._block_bundle(),
            current_start=15,
            attention_request=self._attention_request(stream_r1_kv_cache=True),
            cache_state=state,
            forward_batch=forward_batch,
        )

        self.assertEqual(len(recorder.calls), 1)
        self.assertIs(recorder.forward_batch, forward_batch)

    def test_adaptive_timestep_subset_preserves_first_and_last(self):
        timesteps = torch.tensor([1000, 750, 500, 250])

        torch.testing.assert_close(
            _select_wan_s2v_adaptive_timesteps(timesteps, 3),
            torch.tensor([1000, 500, 250]),
        )
        torch.testing.assert_close(
            _select_wan_s2v_adaptive_timesteps(timesteps, 2),
            torch.tensor([1000, 250]),
        )

    def test_adaptive_steps_reduce_for_similar_audio(self):
        stage = self._stage()
        server_args = SimpleNamespace(
            pipeline_config=WanS2VPipelineConfig(
                wan_s2v_adaptive_steps=True,
                wan_s2v_adaptive_steps_threshold=0.2,
                wan_s2v_adaptive_steps_reduced_step_count=2,
                wan_s2v_adaptive_steps_warmup_blocks=1,
            )
        )
        batch = SimpleNamespace(extra={}, request_id="request-a")
        timesteps = torch.tensor([1000, 750, 500, 250])
        first = self._block_bundle()
        second = self._block_bundle()
        second.audio_input = first.audio_input * 1.01

        first_decision = stage._select_adaptive_timesteps(
            batch=batch,
            server_args=server_args,
            block_bundle=first,
            timesteps=timesteps,
            block_index=0,
        )
        second_decision = stage._select_adaptive_timesteps(
            batch=batch,
            server_args=server_args,
            block_bundle=second,
            timesteps=timesteps,
            block_index=1,
        )

        self.assertFalse(first_decision.reduced)
        self.assertTrue(second_decision.reduced)
        self.assertEqual(second_decision.reason, "similar_audio")
        self.assertEqual(second_decision.step_count, 2)
        torch.testing.assert_close(
            second_decision.timesteps,
            torch.tensor([1000, 250]),
        )

    def test_adaptive_steps_log_only_keeps_base_timesteps(self):
        stage = self._stage()
        server_args = SimpleNamespace(
            pipeline_config=WanS2VPipelineConfig(
                wan_s2v_adaptive_steps=True,
                wan_s2v_adaptive_steps_log_only=True,
                wan_s2v_adaptive_steps_threshold=0.2,
                wan_s2v_adaptive_steps_reduced_step_count=2,
            )
        )
        batch = SimpleNamespace(extra={}, request_id="request-a")
        timesteps = torch.tensor([1000, 750, 500, 250])
        first = self._block_bundle()
        second = self._block_bundle()

        stage._select_adaptive_timesteps(
            batch=batch,
            server_args=server_args,
            block_bundle=first,
            timesteps=timesteps,
            block_index=0,
        )
        decision = stage._select_adaptive_timesteps(
            batch=batch,
            server_args=server_args,
            block_bundle=second,
            timesteps=timesteps,
            block_index=1,
        )

        self.assertFalse(decision.reduced)
        self.assertTrue(decision.log_only)
        self.assertEqual(decision.target_step_count, 2)
        self.assertEqual(decision.step_count, 4)
        self.assertEqual(decision.reason, "similar_audio_log_only")
        torch.testing.assert_close(decision.timesteps, timesteps)

    def test_timestep_profile_config_resolves_from_pipeline_config(self):
        stage = self._stage()
        server_args = SimpleNamespace(
            pipeline_config=WanS2VPipelineConfig(
                wan_s2v_timestep_profile=True,
                wan_s2v_timestep_profile_log=True,
                wan_s2v_timestep_profile_nvtx=False,
                wan_s2v_timestep_profile_sync=True,
            )
        )
        batch = SimpleNamespace(extra={})

        config = stage._resolve_timestep_profile_config(batch, server_args)

        self.assertTrue(config.enabled)
        self.assertTrue(config.log)
        self.assertFalse(config.nvtx)
        self.assertTrue(config.synchronize)

    def test_timestep_ablation_defaults_to_step1_reuse(self):
        stage = self._stage()
        server_args = SimpleNamespace(pipeline_config=WanS2VPipelineConfig())
        batch = SimpleNamespace(extra={})

        config = stage._resolve_timestep_ablation_config(batch, server_args)

        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "reuse_previous_pred")
        self.assertEqual(config.step_indices, (1,))
        self.assertEqual(config.warmup_blocks, 2)
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=1,
                step_index=1,
                timestep_value=937.5,
            ),
            "full",
        )
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=2,
                step_index=1,
                timestep_value=937.5,
            ),
            "reuse_previous_pred",
        )

    def test_timestep_ablation_config_selects_target_steps(self):
        stage = self._stage()
        server_args = SimpleNamespace(pipeline_config=WanS2VPipelineConfig())
        batch = SimpleNamespace(
            extra={
                "timestep_ablation_mode": "zero_pred",
                "timestep_ablation_indices": "1, 3",
                "timestep_ablation_values": "625",
                "timestep_ablation_blocks": [2],
                "timestep_ablation_warmup_blocks": 1,
            }
        )

        config = stage._resolve_timestep_ablation_config(batch, server_args)

        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "zero_pred")
        self.assertEqual(config.step_indices, (1, 3))
        self.assertEqual(config.timestep_values, (625.0,))
        self.assertEqual(config.block_indices, (2,))
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=0,
                step_index=1,
                timestep_value=937.5,
            ),
            "full",
        )
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=2,
                step_index=1,
                timestep_value=937.5,
            ),
            "zero_pred",
        )
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=2,
                step_index=0,
                timestep_value=625.0,
            ),
            "zero_pred",
        )
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=3,
                step_index=1,
                timestep_value=937.5,
            ),
            "full",
        )

    def test_timestep_ablation_disabled_without_targets(self):
        stage = self._stage()
        config = WanS2VTimestepAblationConfig(
            mode="skip_update",
            step_indices=(),
            timestep_values=(),
            block_indices=(),
            warmup_blocks=0,
            value_tolerance=1e-3,
            scale=1.0,
            log=False,
        )

        self.assertFalse(config.enabled)
        self.assertEqual(
            stage._select_timestep_ablation_action(
                config,
                block_index=0,
                step_index=0,
                timestep_value=1000.0,
            ),
            "full",
        )

    def test_audio_embedding_cache_precomputes_encoder_once_with_motion_prefix(self):
        stage = self._stage()
        encoder = self._RecordingAudioEncoder()
        stage.transformer = SimpleNamespace(casual_audio_encoder=encoder)
        bundle = self._audio_cache_bundle()
        audio_input = bundle.audio_input

        stage._maybe_cache_audio_embeddings(bundle)
        cached_emb = bundle.audio_emb
        stage._maybe_cache_audio_embeddings(bundle)

        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(cached_emb, {"encoded_call": 1})
        self.assertIs(bundle.audio_emb, cached_emb)
        self.assertIs(bundle.slice(0, 1).audio_emb, cached_emb)
        cached_audio = encoder.calls[0]
        self.assertEqual(cached_audio.shape, (1, 2, 3, 7))
        torch.testing.assert_close(
            cached_audio[..., :3],
            audio_input[..., 0:1].repeat(1, 1, 1, 3),
        )
        torch.testing.assert_close(cached_audio[..., 3:], audio_input)

    def test_audio_embedding_cache_skips_when_disabled(self):
        stage = self._stage()
        encoder = self._RecordingAudioEncoder()
        stage.transformer = SimpleNamespace(casual_audio_encoder=encoder)
        bundle = self._audio_cache_bundle(cache_audio_embeddings=False)

        stage._maybe_cache_audio_embeddings(bundle)

        self.assertEqual(encoder.calls, [])
        self.assertIsNone(bundle.audio_emb)

    def test_audio_embedding_cache_skips_when_prefilled(self):
        stage = self._stage()
        encoder = self._RecordingAudioEncoder()
        stage.transformer = SimpleNamespace(casual_audio_encoder=encoder)
        prefilled_audio_emb = {"audio": "prefilled"}
        bundle = self._audio_cache_bundle(audio_emb=prefilled_audio_emb)

        stage._maybe_cache_audio_embeddings(bundle)

        self.assertEqual(encoder.calls, [])
        self.assertIs(bundle.audio_emb, prefilled_audio_emb)

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

    def test_kv_cache_allows_context_parallel_runtime(self):
        stage = self._stage()
        stage.transformer.use_context_parallel = True
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=True,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        stage._validate_stream_r1_parallel_compatibility(
            request,
            SimpleNamespace(
                did_sp_shard_latents=False,
                enable_sequence_shard=False,
            ),
        )

    def test_kv_cache_allows_sequence_parallel_world_size(self):
        stage = self._stage()
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=True,
            num_frame_per_block=4,
            local_attn_size=6,
            sink_size=1,
            context_noise=0,
        )

        stage._validate_stream_r1_parallel_compatibility(
            request,
            SimpleNamespace(
                did_sp_shard_latents=True,
                enable_sequence_shard=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
