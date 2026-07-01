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
)
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VStreamR1AttentionLayout,
    WanS2VStreamR1AttentionPlan,
    WanS2VStreamR1MixedKVView,
    WanS2VStreamR1NoisyKVCacheUpdate,
    WanS2VStreamR1NoisyKVCacheView,
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
    update_wan_s2v_stream_r1_noisy_kv_cache,
    validate_wan_s2v_stream_r1_forward_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VConditionBundle,
    WanS2VStreamR1AttentionRequest,
    WanS2VStreamR1CacheMetadata,
    WanS2VStreamR1CacheState,
    WanS2VStreamR1DenoisingStage,
    _has_negative_prompt_embeds,
)


class TestWanS2VNoisyRopeGridSizes(unittest.TestCase):
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
        torch.testing.assert_close(segmented_workspace.query, materialized_workspace.query)
        torch.testing.assert_close(segmented_workspace.key, materialized_workspace.key)
        torch.testing.assert_close(segmented_workspace.value, materialized_workspace.value)
        self.assertEqual(
            segmented_workspace.cu_seqlens_q.tolist(),
            materialized_workspace.cu_seqlens_q.tolist(),
        )
        self.assertEqual(
            segmented_workspace.cu_seqlens_k.tolist(),
            materialized_workspace.cu_seqlens_k.tolist(),
        )
        torch.testing.assert_close(segmented_output, materialized_output)


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
