import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.wan_s2v import WanS2VSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VConditionBundle,
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


class TestWanS2VStreamR1DenoisingStage(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
