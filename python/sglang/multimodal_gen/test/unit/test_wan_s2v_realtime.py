# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import (
    WanS2VPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeChunkContext,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.wan_s2v_realtime_adapter import (
    WanS2VRealtimeAdapter,
    WanS2VRealtimeAdapterState,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.wan_s2v_scheduler import (
    build_wan_s2v_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime import (
    AudioRingBuffer,
    WanS2VRealtimeSessionRunner,
    _WanS2VPerChunkRealtimeState,
    _WanS2VStreamingVAECudaGraphRunner,
    _WanS2VStreamingVAEState,
    _audio_window_after_extend,
    _wait_for_session_audio_chunk,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError
from sglang.multimodal_gen.runtime.utils.chunk_timeline import (
    write_flashtalk_audio_chunk_meta,
)


class _RecordingStage:
    def __init__(
        self,
        name,
        calls,
        *,
        require_latents=False,
        set_image_latent=False,
    ):
        self.name = name
        self.calls = calls
        self.require_latents = require_latents
        self.set_image_latent = set_image_latent

    def __call__(self, batch, server_args):
        self.calls.append(self.name)
        if self.require_latents and batch.latents is None:
            raise AssertionError("stage requires preallocated latents")
        if self.set_image_latent:
            batch.image_latent = torch.zeros(
                1,
                1,
                1,
                1,
                1,
                dtype=batch.latents.dtype,
                device=batch.latents.device,
            )
        return batch


class _RecordingTextEncodingStage:
    def __init__(self):
        self.calls = []

    def __call__(self, batch, server_args):
        del server_args
        self.calls.append(batch.prompt)
        batch.prompt_embeds = [
            torch.full((1, 1, 1), float(len(self.calls)), dtype=torch.float32)
        ]
        batch.pooled_embeds = [
            torch.full((1, 1), float(len(self.calls)), dtype=torch.float32)
        ]
        batch.prompt_attention_mask = [torch.ones(1, 1, dtype=torch.int64)]
        batch.prompt_embeds_mask = [torch.ones(1, 1, 1, dtype=torch.bool)]
        batch.prompt_seq_lens = [[1]]
        batch.negative_prompt_embeds = []
        batch.neg_pooled_embeds = []
        batch.negative_attention_mask = None
        batch.negative_prompt_embeds_mask = None
        batch.negative_prompt_seq_lens = None
        batch.is_prompt_processed = True
        return batch


class _FakeRealtimeRunner(WanS2VRealtimeSessionRunner):
    def __init__(self, stages_by_type):
        self.pipeline = None
        self.stages = []
        self.stages_by_type = stages_by_type

    def _get_stage(self, stage_type):
        return self.stages_by_type[stage_type]


class _FakeAudioPrefetchRunner(_FakeRealtimeRunner):
    def __init__(self):
        super().__init__({})
        self.seen_audio_window = None
        self.calls = []

    def _next_audio_chunk(self, **kwargs):
        return (
            7,
            np.array([4, 5], dtype=np.float32),
            {"chunk_source": "mic"},
            False,
        )

    def _prepare_audio_feature_cpu(
        self,
        audio_stage,
        audio_window,
        *,
        ensure_loaded=True,
    ):
        self.calls.append("prepare_audio_feature_cpu")
        self.seen_audio_window = np.asarray(audio_window, dtype=np.float32).copy()
        return torch.ones(1, 4)

    def _encode_audio_feature(
        self,
        audio_stage,
        audio_feature,
        *,
        target_audio_frames,
        audio_window_video_frames,
        wav2vec_graph_runner=None,
    ):
        self.calls.append("encode_audio_feature")
        torch.testing.assert_close(audio_feature, torch.ones(1, 4))
        return torch.ones(1, 2, 3, 4)


class _FakeWanDecoder(torch.nn.Module):
    def forward(self, x):
        from sglang.multimodal_gen.runtime.models.vaes.wanvae import first_chunk

        frames = 1 if bool(first_chunk.get()) else 4
        return x.new_zeros(x.shape[0], 1, frames, x.shape[3], x.shape[4])


class _FakeWanVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_feature_cache = True
        self.post_quant_conv = torch.nn.Identity()
        self.decoder = _FakeWanDecoder()
        self.config = SimpleNamespace(patch_size=None)
        self.clear_cache_calls = 0
        self.clear_cache()

    def clear_cache(self):
        self.clear_cache_calls += 1
        self._feat_map = [None]
        self._conv_idx = 0


class _FakePipelineConfig:
    vae_precision = "fp32"
    vae_tiling = False
    vae_config = SimpleNamespace(arch_config=SimpleNamespace(scale_factor_temporal=4))

    def get_decode_scale_and_shift(self, device, dtype, vae):
        return 1.0, None

    def preprocess_decoding(self, latents, server_args, vae=None):
        return latents


class WanS2VRealtimeHelpersTest(unittest.TestCase):
    def _make_prompt_session(
        self,
        *,
        prompt="old prompt",
        negative_prompt=None,
    ):
        adapter = WanS2VRealtimeAdapter()
        session = GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        session.set_request(request)
        state = session.adapter_state
        assert isinstance(state, WanS2VRealtimeAdapterState)
        state.configure(fps=16, num_frame_per_block=3)
        state.configure_prompt(request.prompt, request.negative_prompt)
        return adapter, session, request, state

    def test_ws_audio_prefetch_reservation_preserves_window_order(self):
        state = WanS2VRealtimeAdapterState()
        state.configure(fps=16, num_frame_per_block=3)
        samples = np.linspace(-0.5, 0.5, 21000, dtype=np.float32)

        state.receive_audio_delta(
            {
                "seq": 0,
                "pts_ms": 0.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "f32le",
                "audio": samples.astype("<f4", copy=False).tobytes(),
                "sample_count": int(samples.size),
            },
            event_id=1,
        )

        first = state.pop_window()
        reserved = state.reserve_prefetch_window()

        self.assertEqual(first.chunk_idx, 0)
        self.assertEqual(len(first.samples), 9000)
        self.assertIsNotNone(reserved)
        assert reserved is not None
        self.assertEqual(reserved.chunk_idx, 1)
        self.assertEqual(len(reserved.samples), 12000)
        self.assertTrue(state.has_ready_window())

        second = state.pop_window()

        self.assertEqual(second.chunk_idx, 1)
        np.testing.assert_array_equal(second.samples, reserved.samples)
        self.assertFalse(state.has_ready_window())

    def test_prompt_update_applies_to_next_unstarted_chunk(self):
        adapter, session, request, _state = self._make_prompt_session()
        event = RealtimeEvent(
            type="event",
            kind="prompt.update",
            event_id=10,
            payload={
                "prompt": "new prompt",
                "negative_prompt": "no blur",
            },
        )

        event_log = adapter.ingest_event(session, event)
        ack = adapter.build_event_ack(session, event, event_log)
        chunk = RealtimeChunkContext(
            session_id=session.id,
            index=0,
            request_id="request-0",
        )
        inputs = adapter.sample_chunk_inputs(session, SimpleNamespace(), chunk, 3)

        self.assertIn("prompt_revision=1", event_log)
        self.assertIsNotNone(ack)
        assert ack is not None
        self.assertEqual(ack["prompt_update"]["revision"], 1)
        self.assertEqual(ack["prompt_update"]["effective_chunk_index"], 0)
        self.assertEqual(inputs.prompt, "new prompt")
        self.assertEqual(request.prompt, "new prompt")
        self.assertEqual(request.negative_prompt, "no blur")

    def test_prompt_update_waits_until_running_chunk_completes(self):
        adapter, session, request, _state = self._make_prompt_session()
        session.current_chunk = RealtimeChunkContext(
            session_id=session.id,
            index=0,
            request_id="running-request",
        )
        event = RealtimeEvent(
            type="event",
            kind="prompt",
            event_id=11,
            payload="next prompt",
        )

        adapter.ingest_event(session, event)
        chunk0_inputs = adapter.sample_chunk_inputs(
            session,
            SimpleNamespace(),
            session.current_chunk,
            3,
        )
        chunk1 = RealtimeChunkContext(
            session_id=session.id,
            index=1,
            request_id="request-1",
        )
        chunk1_inputs = adapter.sample_chunk_inputs(
            session,
            SimpleNamespace(),
            chunk1,
            3,
        )

        self.assertEqual(chunk0_inputs.prompt, "old prompt")
        self.assertEqual(chunk1_inputs.prompt, "next prompt")
        self.assertEqual(request.prompt, "next prompt")

    def test_prompt_update_rejects_expired_effective_chunk_index(self):
        adapter, session, _request, _state = self._make_prompt_session()
        session.current_chunk = RealtimeChunkContext(
            session_id=session.id,
            index=2,
            request_id="running-request",
        )
        event = RealtimeEvent(
            type="event",
            kind="prompt.update",
            event_id=12,
            payload={
                "prompt": "too late",
                "effective_chunk_index": 2,
            },
        )

        with self.assertRaises(RealtimeProtocolError) as cm:
            adapter.ingest_event(session, event)

        self.assertEqual(cm.exception.code, "prompt_update_too_late")
        self.assertEqual(cm.exception.details["effective_chunk_index"], 2)
        self.assertEqual(cm.exception.details["next_unstarted_chunk_index"], 3)

    def test_prompt_update_chunk_stats_include_revision_metadata(self):
        adapter, session, _request, state = self._make_prompt_session()
        event = RealtimeEvent(
            type="event",
            kind="prompt.update",
            event_id=13,
            payload={"prompt": "stats prompt"},
        )
        adapter.ingest_event(session, event)
        chunk = RealtimeChunkContext(
            session_id=session.id,
            index=0,
            request_id="request-0",
        )
        adapter.sample_chunk_inputs(session, SimpleNamespace(), chunk, 3)
        batch = SimpleNamespace(
            realtime_event_id=7,
            extra={
                "wan_s2v_audio_window_meta": {
                    "chunk_idx": 0,
                    "pts_start_ms": 0.0,
                    "pts_end_ms": 562.5,
                    "sample_count": 9000,
                    "is_final": False,
                },
                "wan_s2v_prompt": state.latest_prompt_chunk_debug,
            },
        )

        stats = adapter.build_chunk_stats_extra(session, batch, result=None)

        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats["prompt"]["revision"], 1)
        self.assertEqual(stats["prompt"]["event_id"], 13)
        self.assertEqual(stats["prompt"]["effective_chunk_index"], 0)
        self.assertEqual(stats["prompt"]["prompt_len"], len("stats prompt"))
        self.assertEqual(stats["audio_window"]["duration_ms"], 562.5)

    def test_model_side_prompt_condition_refresh_updates_cached_base_batch(self):
        text_stage = _RecordingTextEncodingStage()
        runner = _FakeRealtimeRunner({TextEncodingStage: text_stage})
        state = _WanS2VPerChunkRealtimeState()
        state.block_public_frames = 9
        base = Req(
            sampling_params=WanS2VSamplingParams(
                prompt="old prompt",
                negative_prompt=None,
                guidance_scale=1.0,
            ),
            extra={"wan_s2v_prompt": {"revision": 0}},
        )
        base.prompt_embeds = [torch.zeros(1, 1, 1)]
        base.pooled_embeds = [torch.zeros(1, 1)]
        base.prompt_attention_mask = [torch.ones(1, 1, dtype=torch.int64)]
        base.prompt_embeds_mask = [torch.ones(1, 1, 1, dtype=torch.bool)]
        base.prompt_seq_lens = [[1]]
        runner._store_per_chunk_base_batch(state, base)
        runner._record_prompt_condition_key(state, base)

        updated = Req(
            sampling_params=WanS2VSamplingParams(
                prompt="new prompt",
                negative_prompt=None,
                guidance_scale=1.0,
            ),
            extra={"wan_s2v_prompt": {"revision": 1}},
        )
        work_batch = runner._work_batch_for_realtime_chunk(state, updated)
        timings = runner._refresh_prompt_condition_if_needed(
            state,
            work_batch,
            SimpleNamespace(),
        )

        self.assertTrue(timings["prompt_condition_refresh"])
        self.assertEqual(text_stage.calls, ["new prompt"])
        self.assertEqual(state.prompt_condition_revision, 1)
        torch.testing.assert_close(work_batch.prompt_embeds[0], torch.ones(1, 1, 1))
        assert state.base_batch is not None
        torch.testing.assert_close(
            state.base_batch.prompt_embeds[0],
            torch.ones(1, 1, 1),
        )

        same_revision = Req(
            sampling_params=WanS2VSamplingParams(
                prompt="new prompt",
                negative_prompt=None,
                guidance_scale=1.0,
            ),
            extra={"wan_s2v_prompt": {"revision": 1}},
        )
        next_work_batch = runner._work_batch_for_realtime_chunk(state, same_revision)
        next_timings = runner._refresh_prompt_condition_if_needed(
            state,
            next_work_batch,
            SimpleNamespace(),
        )

        self.assertFalse(next_timings["prompt_condition_refresh"])
        self.assertEqual(text_stage.calls, ["new prompt"])
        torch.testing.assert_close(next_work_batch.prompt_embeds[0], torch.ones(1, 1, 1))

    def test_ws_audio_prefetch_requires_explicit_experiment_flag(self):
        adapter = WanS2VRealtimeAdapter()
        session = SimpleNamespace(request=SimpleNamespace(max_chunks=None))
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(wan_s2v_audio_overlap=True)
        )
        chunk = SimpleNamespace(index=0)
        window = SimpleNamespace(is_final=False)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WAN_S2V_WS_AUDIO_CPU_PREFETCH", None)
            self.assertFalse(
                adapter._should_prefetch_next_audio_window(
                    session,
                    server_args,
                    chunk,
                    window,
                )
            )

        with patch.dict(os.environ, {"WAN_S2V_WS_AUDIO_CPU_PREFETCH": "1"}):
            self.assertTrue(
                adapter._should_prefetch_next_audio_window(
                    session,
                    server_args,
                    chunk,
                    window,
                )
            )

    def test_audio_ring_buffer_wraps_in_chronological_order(self):
        ring = AudioRingBuffer(5)

        ring.extend(np.array([1, 2, 3], dtype=np.float32))
        ring.extend(np.array([4, 5, 6], dtype=np.float32))

        np.testing.assert_array_equal(
            ring.snapshot(), np.array([2, 3, 4, 5, 6], dtype=np.float32)
        )

    def test_audio_ring_buffer_keeps_last_capacity_samples(self):
        ring = AudioRingBuffer(4)

        ring.extend(np.arange(10, dtype=np.float32))

        np.testing.assert_array_equal(
            ring.snapshot(), np.array([6, 7, 8, 9], dtype=np.float32)
        )

    def test_audio_window_after_extend_matches_ring_order(self):
        window = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        samples = np.array([5, 6], dtype=np.float32)

        np.testing.assert_array_equal(
            _audio_window_after_extend(window, samples),
            np.array([2, 3, 4, 5, 6], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            _audio_window_after_extend(window, np.arange(10, dtype=np.float32)),
            np.array([5, 6, 7, 8, 9], dtype=np.float32),
        )

    def test_prepare_step_noises_matches_sequential_rng_draws(self):
        runner = _FakeRealtimeRunner({})
        latents = torch.zeros(1, 2, 3, 4, 5)
        timesteps = torch.arange(4)
        generator = torch.Generator(device="cpu").manual_seed(7)
        expected_generator = torch.Generator(device="cpu").manual_seed(7)
        expected_shape = latents.permute(0, 2, 1, 3, 4).shape

        noises = runner._prepare_step_noises(latents, timesteps, generator)
        expected = tuple(
            torch.randn(expected_shape, generator=expected_generator)
            for _ in range(3)
        )

        self.assertEqual(len(noises), 3)
        for actual, expected_noise in zip(noises, expected):
            torch.testing.assert_close(actual, expected_noise)
        torch.testing.assert_close(
            torch.randn((), generator=generator),
            torch.randn((), generator=expected_generator),
        )

    def test_wait_for_session_audio_chunk_loads_existing_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = os.path.join(tmp, "audio_chunks")
            os.makedirs(chunks_dir)
            expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            np.save(os.path.join(chunks_dir, "chunk_0000.npy"), expected)

            actual = _wait_for_session_audio_chunk(tmp, 0, timeout=0.1)

        np.testing.assert_array_equal(actual, expected)

    def test_wait_for_session_audio_chunk_returns_none_on_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "audio_chunks"))
            open(os.path.join(tmp, "end"), "w").close()

            actual = _wait_for_session_audio_chunk(tmp, 0, timeout=0.1)

        self.assertIsNone(actual)

    def test_realtime_reference_vae_is_deferred_until_block_latents_exist(self):
        calls = []
        image_stage = _RecordingStage(
            "image",
            calls,
            require_latents=True,
            set_image_latent=True,
        )
        runner = _FakeRealtimeRunner(
            {
                InputValidationStage: _RecordingStage("input", calls),
                TextEncodingStage: _RecordingStage("text", calls),
                ImageVAEEncodingStage: image_stage,
            }
        )
        batch = SimpleNamespace(
            num_frames=597,
            extra={},
            image_latent=None,
            latents=None,
        )
        server_args = SimpleNamespace()

        batch = runner._prepare_reference_and_prompt(batch, server_args, 25)

        self.assertEqual(calls, ["input", "text"])
        self.assertEqual(batch.num_frames, 25)
        self.assertEqual(batch.extra["wan_s2v_realtime_original_num_frames"], 597)
        with self.assertRaisesRegex(RuntimeError, "requires block latents"):
            runner._prepare_reference_latents_once(batch, server_args, image_stage)

        batch.latents = torch.zeros(1, 16, 7, 4, 4, dtype=torch.float16)
        batch = runner._prepare_reference_latents_once(batch, server_args, image_stage)

        self.assertEqual(calls, ["input", "text", "image"])
        self.assertIsNotNone(batch.image_latent)
        self.assertEqual(batch.image_latent.dtype, torch.float16)

        batch = runner._prepare_reference_latents_once(batch, server_args, image_stage)
        self.assertEqual(calls, ["input", "text", "image"])

    def test_next_audio_chunk_skips_filler_when_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = os.path.join(tmp, "audio_chunks")
            os.makedirs(chunks_dir)
            np.save(os.path.join(chunks_dir, "chunk_0000.npy"), np.zeros(3))
            np.save(os.path.join(chunks_dir, "chunk_0001.npy"), np.ones(4))
            write_flashtalk_audio_chunk_meta(
                tmp,
                0,
                {"is_filler": True, "chunk_source": "idle_silence"},
            )
            write_flashtalk_audio_chunk_meta(
                tmp,
                1,
                {"is_filler": False, "chunk_source": "mic", "turn_id": "t1"},
            )
            runner = _FakeRealtimeRunner({})

            chunk_idx, audio, meta, ended = runner._next_audio_chunk(
                session_dir=tmp,
                audio_chunk_idx=0,
                cancel_file=None,
                idle_policy="hold",
                timeline_path=None,
            )

        self.assertFalse(ended)
        self.assertEqual(chunk_idx, 1)
        np.testing.assert_array_equal(audio, np.ones(4, dtype=np.float32))
        self.assertEqual(meta["chunk_source"], "mic")
        self.assertEqual(meta["turn_id"], "t1")

    def test_prefetch_next_audio_chunk_uses_snapshot_plus_next_audio(self):
        runner = _FakeAudioPrefetchRunner()

        prefetched = runner._prefetch_next_audio_chunk(
            batch=SimpleNamespace(),
            server_args=SimpleNamespace(),
            audio_stage=SimpleNamespace(),
            session_dir="/tmp/session",
            audio_chunk_idx=7,
            cancel_file=None,
            idle_policy="hold",
            timeline_path=None,
            audio_window_snapshot=np.array([0, 1, 2, 3], dtype=np.float32),
            target_audio_frames=12,
            audio_window_video_frames=128,
            wav2vec_graph_runner=None,
            overlap_stream=None,
            after_denoise_event=None,
        )

        self.assertFalse(prefetched.end_requested)
        self.assertEqual(prefetched.audio_chunk_idx, 7)
        np.testing.assert_array_equal(
            prefetched.audio,
            np.array([4, 5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            runner.seen_audio_window,
            np.array([2, 3, 4, 5], dtype=np.float32),
        )
        self.assertEqual(
            runner.calls,
            ["prepare_audio_feature_cpu", "encode_audio_feature"],
        )
        torch.testing.assert_close(prefetched.audio_input, torch.ones(1, 2, 3, 4))
        self.assertGreaterEqual(prefetched.audio_cpu_s, 0.0)
        self.assertGreaterEqual(prefetched.audio_gpu_enqueue_s, 0.0)

    def test_realtime_config_validation(self):
        cfg = WanS2VPipelineConfig(
            wan_s2v_realtime=True,
            wan_s2v_realtime_audio_window_seconds=8.0,
            wan_s2v_idle_policy="hold",
        )

        self.assertTrue(cfg.wan_s2v_realtime)
        self.assertEqual(cfg.wan_s2v_idle_policy, "hold")

        with self.assertRaisesRegex(ValueError, "wan_s2v_idle_policy"):
            WanS2VPipelineConfig(wan_s2v_idle_policy="busy_loop")

    def test_tpp_config_accepts_three_fixed_steps_and_cancels_flush(self):
        cfg = WanS2VPipelineConfig(
            stream_r1_mode=True,
            stream_r1_kv_cache=True,
            num_frame_per_block=1,
            denoising_step_list=[1000, 666, 333],
            wan_s2v_clean_context_refresh_mode="never",
            wan_s2v_timestep_ablation_mode="off",
            wan_s2v_tpp=True,
            wan_s2v_tpp_dit_ranks=[1, 2, 3],
            wan_s2v_tpp_decode_rank=0,
        )

        self.assertTrue(cfg.wan_s2v_tpp)
        self.assertEqual(cfg.wan_s2v_tpp_dit_ranks, [1, 2, 3])
        self.assertEqual(cfg.wan_s2v_tpp_decode_rank, 0)
        self.assertEqual(cfg.wan_s2v_tpp_transport, "host_staged_gloo")
        self.assertEqual(cfg.wan_s2v_clean_context_refresh_mode, "never")

        nccl_cfg = WanS2VPipelineConfig(
            stream_r1_mode=True,
            stream_r1_kv_cache=True,
            num_frame_per_block=1,
            denoising_step_list=[1000, 666, 333],
            wan_s2v_clean_context_refresh_mode="never",
            wan_s2v_timestep_ablation_mode="off",
            wan_s2v_tpp=True,
            wan_s2v_tpp_dit_ranks=[1, 2, 3],
            wan_s2v_tpp_decode_rank=0,
            wan_s2v_tpp_transport="official_blocking_nccl",
        )
        self.assertEqual(
            nccl_cfg.wan_s2v_tpp_transport,
            "official_blocking_nccl",
        )

        with self.assertRaisesRegex(ValueError, "wan_s2v_tpp_transport"):
            WanS2VPipelineConfig(wan_s2v_tpp_transport="unknown")

    def test_tpp_config_accepts_uniform_sp2_stage_groups(self):
        cfg = WanS2VPipelineConfig(
            stream_r1_mode=True,
            stream_r1_kv_cache=True,
            num_frame_per_block=1,
            denoising_step_list=[1000, 666, 333],
            wan_s2v_clean_context_refresh_mode="never",
            wan_s2v_timestep_ablation_mode="off",
            wan_s2v_tpp=True,
            wan_s2v_tpp_dit_ranks=[2, 4, 6],
            wan_s2v_tpp_decode_rank=0,
            wan_s2v_tpp_stage_parallel_size=2,
        )

        self.assertEqual(cfg.wan_s2v_tpp_stage_parallel_size, 2)
        self.assertEqual(cfg.wan_s2v_tpp_dit_ranks, [2, 4, 6])
        self.assertFalse(cfg.vae_config.use_parallel_decode)

        with self.assertRaisesRegex(ValueError, "stage leaders must align"):
            WanS2VPipelineConfig(
                stream_r1_mode=True,
                stream_r1_kv_cache=True,
                num_frame_per_block=1,
                denoising_step_list=[1000, 666, 333],
                wan_s2v_clean_context_refresh_mode="never",
                wan_s2v_timestep_ablation_mode="off",
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 4, 6],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_stage_parallel_size=2,
            )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            WanS2VPipelineConfig(wan_s2v_tpp_stage_parallel_size=0)

    def test_tpp_flat_server_config_reapplies_sp2_derived_settings(self):
        cfg = WanS2VPipelineConfig()
        flat_config = {
            "stream_r1_mode": True,
            "stream_r1_kv_cache": True,
            "num_frame_per_block": 1,
            "denoising_step_list": [1000, 666, 333],
            "wan_s2v_clean_context_refresh_mode": "never",
            "wan_s2v_timestep_ablation_mode": "off",
            "wan_s2v_tpp": True,
            "wan_s2v_tpp_dit_ranks": [2, 4, 6],
            "wan_s2v_tpp_decode_rank": 0,
            "wan_s2v_tpp_stage_parallel_size": 2,
        }

        cfg.update_config_from_dict(flat_config)

        self.assertFalse(cfg.vae_config.use_parallel_decode)
        self.assertEqual(cfg.wan_s2v_tpp_dit_ranks, [2, 4, 6])

    def test_tpp_config_rejects_flush_and_incompatible_features(self):
        common = {
            "num_frame_per_block": 1,
            "denoising_step_list": [1000, 666, 333],
            "wan_s2v_timestep_ablation_mode": "off",
            "wan_s2v_tpp": True,
            "wan_s2v_tpp_dit_ranks": [1, 2, 3],
            "wan_s2v_tpp_decode_rank": 0,
        }

        with self.assertRaisesRegex(ValueError, "cancels the flush step"):
            WanS2VPipelineConfig(
                **common,
                wan_s2v_clean_context_refresh_mode="always",
            )

        with self.assertRaisesRegex(ValueError, "latent warm start"):
            WanS2VPipelineConfig(
                **common,
                wan_s2v_clean_context_refresh_mode="never",
                wan_s2v_latent_warm_start=True,
            )

    def test_tpp_config_requires_one_rank_per_timestep(self):
        with self.assertRaisesRegex(ValueError, "one denoising_step_list entry"):
            WanS2VPipelineConfig(
                num_frame_per_block=1,
                denoising_step_list=[1000, 666, 333],
                wan_s2v_clean_context_refresh_mode="never",
                wan_s2v_timestep_ablation_mode="off",
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 2],
                wan_s2v_tpp_decode_rank=0,
            )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_pp_group"
    )
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    def test_tpp_dit_rank_runs_only_its_fixed_timestep(
        self,
        distributed_send,
        distributed_recv,
        get_pp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def __init__(self):
                self.calls = []

            def denoise_stream_r1_block(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["block_latents"] + 1

        cpu_group = object()
        get_pp_group.return_value = SimpleNamespace(cpu_group=cpu_group)

        def recv_ack(tensor, **_kwargs):
            tensor.fill_(0)

        distributed_recv.side_effect = recv_ack
        runner = _FakeRealtimeRunner({})
        denoising_stage = _DenoisingStage()
        latents = torch.zeros(1, 1, 1, 1, 1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 2, 3],
                wan_s2v_tpp_decode_rank=0,
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=denoising_stage,
            batch=SimpleNamespace(),
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        self.assertEqual(denoising_stage.calls[0]["only_step_index"], 0)
        self.assertEqual(distributed_send.call_count, 2)
        self.assertEqual(distributed_send.call_args_list[0].kwargs["dst"], 2)
        self.assertIs(
            distributed_send.call_args_list[0].kwargs["group"], cpu_group
        )
        self.assertEqual(distributed_send.call_args_list[0].args[0].item(), 0)
        torch.testing.assert_close(
            distributed_send.call_args_list[1].args[0],
            torch.ones_like(latents),
        )
        self.assertEqual(distributed_send.call_args_list[1].kwargs["dst"], 2)
        self.assertIs(
            distributed_send.call_args_list[1].kwargs["group"], cpu_group
        )
        distributed_recv.assert_called_once()
        self.assertEqual(distributed_recv.call_args.kwargs["src"], 2)
        self.assertIs(distributed_recv.call_args.kwargs["group"], cpu_group)
        get_pp_group.assert_called_once_with()
        torch.testing.assert_close(output, torch.ones_like(latents))
        self.assertEqual(timing["tpp_role"], "dit")
        self.assertEqual(timing["tpp_stage_index"], 0)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=0,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_pp_group"
    )
    @patch("torch.distributed.send")
    @patch("torch.distributed.recv")
    def test_tpp_decode_rank_receives_last_stage_without_dit_forward(
        self,
        distributed_recv,
        distributed_send,
        get_pp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def denoise_stream_r1_block(self, **kwargs):
                raise AssertionError("decode rank must not run a DiT timestep")

        recv_call_index = 0

        def recv_into(tensor, **_kwargs):
            nonlocal recv_call_index
            tensor.fill_(0 if recv_call_index == 0 else 7)
            recv_call_index += 1

        distributed_recv.side_effect = recv_into
        cpu_group = object()
        get_pp_group.return_value = SimpleNamespace(cpu_group=cpu_group)
        runner = _FakeRealtimeRunner({})
        latents = torch.zeros(1, 1, 1, 1, 1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 2, 3],
                wan_s2v_tpp_decode_rank=0,
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=_DenoisingStage(),
            batch=SimpleNamespace(),
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        self.assertEqual(distributed_recv.call_count, 2)
        self.assertEqual(distributed_recv.call_args_list[0].kwargs["src"], 3)
        self.assertIs(
            distributed_recv.call_args_list[0].kwargs["group"], cpu_group
        )
        self.assertEqual(distributed_recv.call_args_list[1].kwargs["src"], 3)
        self.assertIs(
            distributed_recv.call_args_list[1].kwargs["group"], cpu_group
        )
        distributed_send.assert_called_once()
        self.assertEqual(distributed_send.call_args.kwargs["dst"], 3)
        self.assertIs(distributed_send.call_args.kwargs["group"], cpu_group)
        self.assertEqual(distributed_send.call_args.args[0].item(), 0)
        get_pp_group.assert_called_once_with()
        torch.testing.assert_close(output, torch.full_like(latents, 7))
        self.assertEqual(timing["tpp_role"], "decode")
        self.assertIsNone(timing["tpp_stage_index"])

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_pp_group"
    )
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    def test_tpp_official_nccl_dit_uses_default_group_without_header_or_ack(
        self,
        distributed_send,
        distributed_recv,
        get_pp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def denoise_stream_r1_block(self, **kwargs):
                return kwargs["block_latents"] + 1

        runner = _FakeRealtimeRunner({})
        latents = torch.zeros(1, 1, 1, 1, 1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 2, 3],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_transport="official_blocking_nccl",
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=_DenoisingStage(),
            batch=SimpleNamespace(),
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        distributed_send.assert_called_once()
        self.assertEqual(distributed_send.call_args.kwargs["dst"], 2)
        self.assertNotIn("group", distributed_send.call_args.kwargs)
        torch.testing.assert_close(
            distributed_send.call_args.args[0],
            torch.ones_like(latents),
        )
        distributed_recv.assert_not_called()
        get_pp_group.assert_not_called()
        torch.testing.assert_close(output, torch.ones_like(latents))
        self.assertEqual(timing["tpp_transport"], "official_blocking_nccl")

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=0,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_pp_group"
    )
    @patch("torch.distributed.send")
    @patch("torch.distributed.recv")
    def test_tpp_official_nccl_decode_receives_gpu_payload_only(
        self,
        distributed_recv,
        distributed_send,
        get_pp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def denoise_stream_r1_block(self, **_kwargs):
                raise AssertionError("decode rank must not run a DiT timestep")

        def recv_into(tensor, **_kwargs):
            tensor.fill_(7)

        distributed_recv.side_effect = recv_into
        runner = _FakeRealtimeRunner({})
        latents = torch.zeros(1, 1, 1, 1, 1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[1, 2, 3],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_transport="official_blocking_nccl",
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=_DenoisingStage(),
            batch=SimpleNamespace(),
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        distributed_recv.assert_called_once()
        self.assertEqual(distributed_recv.call_args.kwargs["src"], 3)
        self.assertNotIn("group", distributed_recv.call_args.kwargs)
        distributed_send.assert_not_called()
        get_pp_group.assert_not_called()
        torch.testing.assert_close(output, torch.full_like(latents, 7))
        self.assertEqual(timing["tpp_transport"], "official_blocking_nccl")

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=3,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_sp_group"
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_pp_group"
    )
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    def test_tpp_sp2_lane_one_runs_first_step_and_sends_to_next_group(
        self,
        distributed_send,
        distributed_recv,
        get_pp_group,
        get_sp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def __init__(self):
                self.calls = []

            def denoise_stream_r1_block(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["block_latents"] + 1

        get_sp_group.return_value = SimpleNamespace(
            world_size=2,
            ranks=[2, 3],
            first_rank=2,
            rank_in_group=1,
        )
        runner = _FakeRealtimeRunner({})
        denoising_stage = _DenoisingStage()
        latents = torch.zeros(1, 1, 1, 1, 1)
        batch = SimpleNamespace(enable_sequence_shard=False)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[2, 4, 6],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_stage_parallel_size=2,
                wan_s2v_tpp_transport="official_blocking_nccl",
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=denoising_stage,
            batch=batch,
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        self.assertTrue(batch.enable_sequence_shard)
        self.assertEqual(denoising_stage.calls[0]["only_step_index"], 0)
        distributed_recv.assert_not_called()
        distributed_send.assert_called_once()
        self.assertEqual(distributed_send.call_args.kwargs["dst"], 5)
        self.assertNotIn("group", distributed_send.call_args.kwargs)
        get_pp_group.assert_not_called()
        torch.testing.assert_close(output, torch.ones_like(latents))
        self.assertEqual(timing["tpp_stage_parallel_size"], 2)
        self.assertEqual(timing["tpp_stage_leader"], 2)
        self.assertEqual(timing["tpp_stage_ranks"], [2, 3])
        self.assertEqual(timing["tpp_lane_index"], 1)
        self.assertFalse(timing["tpp_is_stage_leader"])

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=5,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_sp_group"
    )
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    def test_tpp_sp2_middle_stage_receives_and_sends_on_same_lane(
        self,
        distributed_send,
        distributed_recv,
        get_sp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def __init__(self):
                self.calls = []

            def denoise_stream_r1_block(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["block_latents"] + 1

        def recv_into(tensor, **_kwargs):
            tensor.fill_(4)

        distributed_recv.side_effect = recv_into
        get_sp_group.return_value = SimpleNamespace(
            world_size=2,
            ranks=[4, 5],
            first_rank=4,
            rank_in_group=1,
        )
        runner = _FakeRealtimeRunner({})
        denoising_stage = _DenoisingStage()
        latents = torch.zeros(1, 1, 1, 1, 1)
        batch = SimpleNamespace(enable_sequence_shard=False)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[2, 4, 6],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_stage_parallel_size=2,
                wan_s2v_tpp_transport="official_blocking_nccl",
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=denoising_stage,
            batch=batch,
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        distributed_recv.assert_called_once()
        self.assertEqual(distributed_recv.call_args.kwargs["src"], 3)
        distributed_send.assert_called_once()
        self.assertEqual(distributed_send.call_args.kwargs["dst"], 7)
        self.assertEqual(denoising_stage.calls[0]["only_step_index"], 1)
        torch.testing.assert_close(output, torch.full_like(latents, 5))
        self.assertEqual(timing["tpp_lane_index"], 1)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_world_rank",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_sp_group"
    )
    @patch("torch.distributed.send")
    @patch("torch.distributed.recv")
    def test_tpp_sp2_decode_follower_receives_last_lane_without_dit(
        self,
        distributed_recv,
        distributed_send,
        get_sp_group,
        _get_world_rank,
    ):
        class _DenoisingStage:
            def denoise_stream_r1_block(self, **_kwargs):
                raise AssertionError("decode follower must not run DiT")

        def recv_into(tensor, **_kwargs):
            tensor.fill_(7)

        distributed_recv.side_effect = recv_into
        get_sp_group.return_value = SimpleNamespace(
            world_size=2,
            ranks=[0, 1],
            first_rank=0,
            rank_in_group=1,
        )
        runner = _FakeRealtimeRunner({})
        latents = torch.zeros(1, 1, 1, 1, 1)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_tpp=True,
                wan_s2v_tpp_dit_ranks=[2, 4, 6],
                wan_s2v_tpp_decode_rank=0,
                wan_s2v_tpp_stage_parallel_size=2,
                wan_s2v_tpp_transport="official_blocking_nccl",
            )
        )

        output, timing = runner._denoise_tpp_block(
            denoising_stage=_DenoisingStage(),
            batch=SimpleNamespace(),
            server_args=server_args,
            block_latents=latents,
            block_bundle=SimpleNamespace(),
            block_start=0,
            frame_seq_length=1,
            timesteps=torch.tensor([1000.0, 666.0, 333.0]),
            prompt_embeds=torch.zeros(1),
            cache_state=SimpleNamespace(),
            crossattn_cache=None,
            generator=None,
            dit_dtype=torch.float32,
            autocast_enabled=False,
            step_noises_btchw=(torch.zeros_like(latents),) * 2,
            block_index=0,
            allow_timestep_cuda_graph_capture=False,
            timestep_values=(1000.0, 666.0, 333.0),
        )

        distributed_recv.assert_called_once()
        self.assertEqual(distributed_recv.call_args.kwargs["src"], 7)
        distributed_send.assert_not_called()
        torch.testing.assert_close(output, torch.full_like(latents, 7))
        self.assertEqual(timing["tpp_role"], "decode")
        self.assertFalse(timing["tpp_is_stage_leader"])

    def test_first_frame_controls_are_configurable(self):
        cfg = WanS2VPipelineConfig(
            s2v_init_first_frame=True,
            s2v_anchor_first_frame=True,
        )

        self.assertTrue(cfg.s2v_init_first_frame)
        self.assertTrue(cfg.s2v_anchor_first_frame)

    def test_first_frame_controls_are_forwarded_in_request_extra(self):
        params = WanS2VSamplingParams(
            init_first_frame=True,
            anchor_first_frame=True,
        )

        extra = params.build_request_extra()

        self.assertTrue(extra["init_first_frame"])
        self.assertTrue(extra["anchor_first_frame"])

    def test_sampling_defaults_match_stream_r1_benchmark(self):
        params = WanS2VSamplingParams()

        self.assertEqual(params.guidance_scale, 1.0)
        self.assertIsNone(params.negative_prompt)

    def test_streaming_vae_output_frame_cadence(self):
        runner = _FakeRealtimeRunner({})
        server_args = SimpleNamespace(pipeline_config=_FakePipelineConfig())

        self.assertEqual(
            runner._block_output_frames(
                server_args,
                3,
                0,
                use_streaming_vae_cache=True,
            ),
            9,
        )
        self.assertEqual(
            runner._block_output_frames(
                server_args,
                3,
                1,
                use_streaming_vae_cache=True,
            ),
            12,
        )
        self.assertEqual(
            runner._block_output_frames(
                server_args,
                3,
                1,
                use_streaming_vae_cache=False,
            ),
            9,
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_streaming_vae_decode_keeps_temporal_cache(self, _device):
        runner = _FakeRealtimeRunner({})
        vae = _FakeWanVAE()
        decoding_stage = SimpleNamespace(
            vae=vae,
            scale_and_shift=lambda latents, server_args: latents,
            decode=lambda latents, server_args: torch.empty(0),
        )
        server_args = SimpleNamespace(
            pipeline_config=_FakePipelineConfig(),
            disable_autocast=True,
        )
        state = _WanS2VStreamingVAEState(enabled=True)

        first = runner._decode_block_frames(
            decoding_stage,
            torch.zeros(1, 1, 2, 1, 1),
            server_args,
            state,
        )
        second = runner._decode_block_frames(
            decoding_stage,
            torch.zeros(1, 1, 2, 1, 1),
            server_args,
            state,
        )

        self.assertTrue(state.initialized)
        self.assertEqual(state.decoded_latent_frames, 4)
        self.assertEqual(state.last_decode_mode, "eager")
        self.assertEqual(vae.clear_cache_calls, 2)
        self.assertEqual(first.shape, (1, 1, 5, 1, 1))
        self.assertEqual(second.shape, (1, 1, 8, 1, 1))

    def test_vae_graph_replay_refreshes_static_cache_from_live_cache(self):
        runner = _WanS2VStreamingVAECudaGraphRunner()
        runner.static_input = torch.zeros(1)
        runner.static_output = torch.zeros(1)
        runner.cache_input_map = [torch.zeros(2)]
        runner.cache_output_map = [torch.zeros(2)]

        def _replay():
            runner.cache_output_map[0].copy_(runner.cache_input_map[0] + 3)
            runner.static_output.copy_(runner.static_input + 5)

        runner.graph = SimpleNamespace(replay=_replay)
        live_cache = [torch.tensor([7.0, 11.0])]

        output = runner.replay(torch.tensor([2.0]), live_cache)

        torch.testing.assert_close(output, torch.tensor([7.0]))
        torch.testing.assert_close(runner.cache_input_map[0], torch.tensor([10.0, 14.0]))
        torch.testing.assert_close(live_cache[0], torch.tensor([10.0, 14.0]))
        self.assertIsNot(live_cache[0], runner.cache_input_map[0])

    def test_wav2vec_graph_prewarm_video_frames_include_realtime_defaults(self):
        runner = _FakeRealtimeRunner({})
        batch = SimpleNamespace(fps=25)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps=[16, 24]
            )
        )

        self.assertEqual(
            runner._wav2vec_cuda_graph_warmup_video_frames(
                batch,
                server_args,
                audio_window_seconds=8.0,
            ),
            (128, 192, 200),
        )

    def test_vae_cuda_graph_config_flag_is_available(self):
        cfg = WanS2VPipelineConfig(wan_s2v_vae_cuda_graph=True)

        self.assertTrue(cfg.wan_s2v_vae_cuda_graph)

    def test_wav2vec_graph_warmup_fps_config_is_normalized(self):
        cfg = WanS2VPipelineConfig(
            wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps=[16, 24]
        )

        self.assertEqual(
            cfg.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps,
            (16, 24),
        )

        with self.assertRaisesRegex(ValueError, "warmup_fps"):
            WanS2VPipelineConfig(
                wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps=[0]
            )

    def test_adaptive_steps_config_is_available(self):
        cfg = WanS2VPipelineConfig(
            wan_s2v_adaptive_steps=True,
            wan_s2v_adaptive_steps_threshold=0.12,
            wan_s2v_adaptive_steps_reduced_step_count=2,
            wan_s2v_adaptive_steps_log_only=True,
        )

        self.assertTrue(cfg.wan_s2v_adaptive_steps)
        self.assertEqual(cfg.wan_s2v_adaptive_steps_threshold, 0.12)
        self.assertEqual(cfg.wan_s2v_adaptive_steps_reduced_step_count, 2)
        self.assertTrue(cfg.wan_s2v_adaptive_steps_log_only)

        with self.assertRaisesRegex(ValueError, "adaptive_steps_threshold"):
            WanS2VPipelineConfig(wan_s2v_adaptive_steps_threshold=-0.1)

    def test_adaptive_steps_controls_are_forwarded_in_request_extra(self):
        params = WanS2VSamplingParams(
            adaptive_steps=True,
            adaptive_steps_threshold=0.11,
            adaptive_steps_reduced_step_count=2,
            adaptive_steps_log_only=True,
        )

        extra = params.build_request_extra()

        self.assertTrue(extra["adaptive_steps"])
        self.assertEqual(extra["adaptive_steps_threshold"], 0.11)
        self.assertEqual(extra["adaptive_steps_reduced_step_count"], 2)
        self.assertTrue(extra["adaptive_steps_log_only"])

    @patch(
        "sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime.torch.cuda.is_available",
        return_value=True,
    )
    def test_timestep_graph_preoutput_requirement(self, _cuda_available):
        runner = _FakeRealtimeRunner({})
        denoising_stage = SimpleNamespace(
            _resolve_timestep_cuda_graph_config=lambda batch, server_args: SimpleNamespace(
                enabled=True,
                step_indices=(0,),
            )
        )

        self.assertTrue(
            runner._requires_preoutput_timestep_cuda_graph(
                denoising_stage=denoising_stage,
                batch=SimpleNamespace(),
                server_args=SimpleNamespace(),
                step_count=4,
            )
        )
        self.assertFalse(
            runner._requires_preoutput_timestep_cuda_graph(
                denoising_stage=denoising_stage,
                batch=SimpleNamespace(),
                server_args=SimpleNamespace(),
                step_count=0,
            )
        )

    def test_timestep_graph_capture_remains_allowed_after_output_starts(self):
        runner = _FakeRealtimeRunner({})

        self.assertTrue(
            runner._should_allow_timestep_cuda_graph_capture(
                timestep_graph_output_started=False
            )
        )
        self.assertTrue(
            runner._should_allow_timestep_cuda_graph_capture(
                timestep_graph_output_started=True
            )
        )

    def test_timestep_graph_ready_for_output_requires_graph_hit(self):
        runner = _FakeRealtimeRunner({})
        denoising_stage = SimpleNamespace(
            _last_timestep_cuda_graph_statuses=[
                {"status": "warmup"},
                {"status": "step_filtered"},
            ]
        )

        self.assertFalse(runner._timestep_cuda_graph_ready_for_output(denoising_stage))

        denoising_stage._last_timestep_cuda_graph_statuses = [
            {"status": "capture"},
            {"status": "step_filtered"},
        ]
        self.assertTrue(runner._timestep_cuda_graph_ready_for_output(denoising_stage))
        self.assertIsNone(runner._timestep_cuda_graph_preoutput_error(denoising_stage))

        denoising_stage._last_timestep_cuda_graph_statuses = [
            {"status": "eager_kv_plan_mismatch"}
        ]
        self.assertFalse(runner._timestep_cuda_graph_ready_for_output(denoising_stage))
        self.assertEqual(
            runner._timestep_cuda_graph_preoutput_error(denoising_stage),
            "eager_kv_plan_mismatch",
        )

    def test_stream_r1_config_defaults_to_baseline_flow_shift(self):
        self.assertEqual(WanS2VPipelineConfig().flow_shift, 3.0)
        self.assertEqual(WanS2VPipelineConfig(stream_r1_mode=True).flow_shift, 5.0)

    def test_pipeline_config_defaults_enable_timestep_cuda_graph_step0(self):
        config = WanS2VPipelineConfig()
        self.assertTrue(config.wan_s2v_timestep_cuda_graph)
        self.assertEqual(config.wan_s2v_timestep_cuda_graph_indices, [0])

    def test_stream_r1_scheduler_matches_baseline_timesteps(self):
        scheduler = build_wan_s2v_scheduler(stream_r1_mode=True, flow_shift=5.0)

        self.assertIsInstance(scheduler, SelfForcingFlowMatchScheduler)
        scheduler.set_timesteps(1000, shift=5.0)
        timesteps = scheduler.timesteps[[0, 250, 500, 750]]
        torch.testing.assert_close(
            timesteps,
            torch.tensor([1000.0, 937.5, 833.3333, 625.0]),
            rtol=1e-4,
            atol=1e-3,
        )

    def test_non_stream_r1_scheduler_keeps_default_scheduler(self):
        self.assertIsInstance(
            build_wan_s2v_scheduler(stream_r1_mode=False, flow_shift=3.0),
            FlowUniPCMultistepScheduler,
        )

    def test_init_first_frame_builds_motion_pixels(self):
        stage = object.__new__(ImageVAEEncodingStage)
        batch = SimpleNamespace(extra={"init_first_frame": [True, False]})
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(s2v_init_first_frame=False)
        )
        ref_pixels = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(
            2, 3, 1, 2, 2
        )

        flags = stage._resolve_s2v_init_first_frame_flags(batch, server_args, 2)
        motion_pixels = stage._build_s2v_motion_pixels(ref_pixels, flags, 8)

        self.assertEqual(flags, [True, False])
        self.assertEqual(motion_pixels.shape, (2, 3, 8, 2, 2))
        torch.testing.assert_close(
            motion_pixels[0:1, :, -6:],
            ref_pixels[0:1].expand(-1, -1, 6, -1, -1),
        )
        torch.testing.assert_close(
            motion_pixels[0:1, :, :2], torch.zeros(1, 3, 2, 2, 2)
        )
        torch.testing.assert_close(motion_pixels[1], torch.zeros(3, 8, 2, 2))


if __name__ == "__main__":
    unittest.main()
