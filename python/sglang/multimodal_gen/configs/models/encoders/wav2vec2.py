# SPDX-License-Identifier: Apache-2.0
"""Wav2Vec2 audio encoder configuration for FlashTalk."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    EncoderArchConfig,
    EncoderConfig,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class Wav2Vec2ArchConfig(EncoderArchConfig):
    hidden_size: int = 768
    num_hidden_layers: int = 12
    sample_rate: int = 16000
    # Wav2Vec2 does not need custom attention backends
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,
        }
    )


@dataclass
class Wav2Vec2Config(EncoderConfig):
    arch_config: EncoderArchConfig = field(default_factory=Wav2Vec2ArchConfig)

    prefix: str = "wav2vec2"
