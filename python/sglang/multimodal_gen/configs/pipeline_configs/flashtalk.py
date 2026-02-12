# SPDX-License-Identifier: Apache-2.0
"""FlashTalk pipeline configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    FlashTalkWanVideoConfig,
)
from sglang.multimodal_gen.configs.models.encoders import CLIPVisionConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    WanI2V720PConfig,
)


@dataclass
class FlashTalkPipelineConfig(WanI2V720PConfig):
    """Pipeline configuration for FlashTalk audio-driven talking face generation.

    Extends WanI2V720PConfig with:
    - Audio encoder configuration
    - No classifier-free guidance (single pass denoising)
    - Motion frame caching for streaming generation
    - Lab color correction
    """

    task_type: ModelTaskType = ModelTaskType.I2V

    # DiT uses FlashTalk variant with audio cross-attention
    dit_config: DiTConfig = field(default_factory=FlashTalkWanVideoConfig)

    # FlashTalk-specific: no CFG
    use_cfg: bool = False

    # Flow shift for FlashTalk
    flow_shift: float | None = 5.0

    # Audio encoder precision
    audio_encoder_precision: str = "fp32"

    # Motion frame caching for streaming
    motion_frames_num: int = 5

    # Lab color correction
    enable_lab_color_correction: bool = True

    # Default FlashTalk resolution
    max_area: int = 768 * 448

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
