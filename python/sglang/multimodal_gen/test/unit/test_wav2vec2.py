# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.encoders.wav2vec2 import (
    resample_wan_s2v_hidden_states,
)


class WanS2VWav2VecResampleTest(unittest.TestCase):
    def test_matches_original_two_stage_resampling(self):
        hidden = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)

        actual = resample_wan_s2v_hidden_states(
            hidden,
            audio_num_samples=20,
            sample_rate=10,
            num_video_frames=4,
            native_fps=5,
            intermediate_fps=3,
        )

        at_intermediate_fps = F.interpolate(
            hidden[:, :, 0, 0].unsqueeze(1),
            size=6,
            mode="linear",
            align_corners=True,
        ).squeeze(1)
        indices = torch.tensor([0, 2, 3, 4])
        expected = at_intermediate_fps.index_select(1, indices).reshape(1, 4, 1, 1)
        torch.testing.assert_close(actual, expected)

    def test_zero_pads_indices_beyond_available_features(self):
        hidden = torch.ones(1, 5, 2, 3)

        actual = resample_wan_s2v_hidden_states(
            hidden,
            audio_num_samples=40,
            sample_rate=10,
            num_video_frames=8,
            native_fps=5,
            intermediate_fps=3,
        )

        torch.testing.assert_close(actual[:, :2], torch.ones(1, 2, 2, 3))
        torch.testing.assert_close(actual[:, 2:], torch.zeros(1, 6, 2, 3))


if __name__ == "__main__":
    unittest.main()
