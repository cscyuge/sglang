# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import (
    WanS2VPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime import (
    AudioRingBuffer,
    _wait_for_session_audio_chunk,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)


class WanS2VRealtimeHelpersTest(unittest.TestCase):
    def test_audio_ring_buffer_wraps_in_chronological_order(self):
        ring = AudioRingBuffer(5)

        ring.extend(np.array([1, 2, 3], dtype=np.float32))
        ring.extend(np.array([4, 5, 6], dtype=np.float32))

        np.testing.assert_array_equal(
            ring.snapshot(), np.array([2, 3, 4, 5, 6], dtype=np.float64)
        )

    def test_audio_ring_buffer_keeps_last_capacity_samples(self):
        ring = AudioRingBuffer(4)

        ring.extend(np.arange(10, dtype=np.float32))

        np.testing.assert_array_equal(
            ring.snapshot(), np.array([6, 7, 8, 9], dtype=np.float64)
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
