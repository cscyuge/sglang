# SPDX-License-Identifier: Apache-2.0
"""FlashTalk sampling parameters."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class FlashTalkSamplingParams(SamplingParams):
    """Sampling parameters for FlashTalk audio-driven talking face generation."""

    # Video parameters
    height: int = 768
    width: int = 448
    num_frames: int = 33
    fps: int = 25

    # Disable SP frame padding — FlashTalk was trained on 33-frame chunks
    # (9 latent temporal steps). The flat patch sequence (9×48×28 = 12096) is
    # already divisible by 8, so no adjustment is needed. Padding to 16 latent
    # steps pollutes self-attention and breaks audio-visual alignment.
    adjust_frames: bool = False

    # Denoising
    num_inference_steps: int = 4  # FlashTalk default: 4 steps
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

    # TeaCache: cross-chunk caching for FlashTalk.
    # With only 4 denoising steps per chunk, within-chunk caching is
    # ineffective (large timestep jumps). Instead, FlashTalk caches
    # per-timestep residuals and compares the same timestep across
    # consecutive chunks where content changes gradually.
    # Threshold controls sensitivity: lower = more caching (faster, lower quality).
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            teacache_thresh_aggressive=0.0,  # disabled; set >0 to enable 1-step mode
            coefficients=[1.0, 0.0],  # identity polynomial (no rescaling)
        )
    )

    # Supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (768, 448),  # portrait (height > width)
            (448, 768),  # landscape (width > height)
        ]
    )
