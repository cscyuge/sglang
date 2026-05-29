# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V pipeline configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanS2VConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.flashtalk import (
    _flashtalk_t5_config,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanI2V720PConfig


@dataclass
class WanS2VPipelineConfig(WanI2V720PConfig):
    """Pipeline config for flat Wan2.2-S2V checkpoints."""

    task_type: ModelTaskType = ModelTaskType.I2V
    dit_config: DiTConfig = field(default_factory=WanS2VConfig)
    text_encoder_configs: tuple = field(default_factory=lambda: (_flashtalk_t5_config(),))
    use_cfg: bool = True
    flow_shift: float | None = 3.0
    audio_encoder_precision: str = "fp32"
    audio_encoder_path: str | None = None
    max_area: int = 1024 * 704

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def postprocess_image_latent(self, latent_condition, batch):
        return latent_condition
