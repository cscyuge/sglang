# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

from sglang.multimodal_gen.configs.pipeline_configs.wan_s2v import (
    WanS2VPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime import (
    AudioRingBuffer,
    _wait_for_session_audio_chunk,
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


if __name__ == "__main__":
    unittest.main()
