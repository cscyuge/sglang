# SPDX-License-Identifier: Apache-2.0
"""FlashTalk sampling parameters."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class FlashTalkSamplingParams(SamplingParams):
    """Sampling parameters for FlashTalk audio-driven talking face generation."""

    # Video parameters
    height: int = 768
    width: int = 448
    num_frames: int = 33
    fps: int = 25

    # Denoising
    num_inference_steps: int = 2  # FlashTalk default: 2 steps
    guidance_scale: float = 1.0  # No CFG

    # Default seed
    seed: int = 1024

    # Audio input
    audio_path: str | None = None
    audio_encode_mode: str = "stream"  # "stream" or "batch"

    # Color correction
    color_correction_strength: float = 0.8

    # FlashTalk does not use negative prompts
    negative_prompt: str | None = None

    # Supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (768, 448),  # portrait
            (448, 768),  # landscape
        ]
    )
