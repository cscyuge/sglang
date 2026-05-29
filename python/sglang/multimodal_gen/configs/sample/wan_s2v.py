# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Wan2.2-S2V."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class WanS2VSamplingParams(SamplingParams):
    num_frames: int = 81
    num_inference_steps: int = 40
    guidance_scale: float = 4.5
    flow_shift: float = 3.0
    audio_path: str | None = None
    disable_sp_frame_padding: bool = True
