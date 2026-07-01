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
    _WanS2VStreamingVAEState,
    _audio_window_after_extend,
    _wait_for_session_audio_chunk,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
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

    def test_vae_cuda_graph_config_flag_is_available(self):
        cfg = WanS2VPipelineConfig(wan_s2v_vae_cuda_graph=True)

        self.assertTrue(cfg.wan_s2v_vae_cuda_graph)

    def test_stream_r1_config_defaults_to_baseline_flow_shift(self):
        self.assertEqual(WanS2VPipelineConfig().flow_shift, 3.0)
        self.assertEqual(WanS2VPipelineConfig(stream_r1_mode=True).flow_shift, 5.0)

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
