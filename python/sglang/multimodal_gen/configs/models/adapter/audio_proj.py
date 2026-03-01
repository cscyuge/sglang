# SPDX-License-Identifier: Apache-2.0
"""AudioProjModel adapter configuration for FlashTalk."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.adapter.base import (
    AdapterArchConfig,
    AdapterConfig,
)


@dataclass
class AudioProjArchConfig(AdapterArchConfig):
    """Architecture config for AudioProjModel.

    The audio projection has two paths:
    - First frame: 5-frame audio window -> proj1 -> proj2 -> proj3
    - Subsequent frames: 12-frame audio window (VAE temporal stride=4) -> proj1_vf -> proj2 -> proj3
    """

    audio_window_first: int = 5  # audio frame window for first video frame
    audio_window_vf: int = (
        12  # audio frame window for subsequent frames (vae temporal factor * 3)
    )
    context_tokens: int = 32  # number of output context tokens per timestep
    output_dim: int = 768  # output feature dimension
    hidden_dim: int = 512  # intermediate projection dimension
    num_audio_layers: int = 12  # number of Wav2Vec2 layers used
    audio_feat_dim: int = 768  # per-layer audio feature dimension


@dataclass
class AudioProjConfig(AdapterConfig):
    arch_config: AdapterArchConfig = field(default_factory=AudioProjArchConfig)

    prefix: str = "audio_proj"
