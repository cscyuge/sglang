# SPDX-License-Identifier: Apache-2.0
"""FlashTalk pipeline configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    FlashTalkWanVideoConfig,
)
from sglang.multimodal_gen.configs.models.encoders import CLIPVisionConfig, T5Config
from sglang.multimodal_gen.configs.models.encoders.clip import CLIPVisionArchConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    WanI2V720PConfig,
)


def _flashtalk_t5_config() -> T5Config:
    """T5 config matching the FlashTalk UMT5-XXL encoder checkpoint."""
    return T5Config(
        arch_config=T5ArchConfig(
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            feed_forward_proj="gated-gelu",
            is_gated_act=True,
            dense_act_fn="gelu_new",
            layer_norm_epsilon=1e-6,
            text_len=512,
        )
    )


def _flashtalk_clip_config() -> CLIPVisionConfig:
    """CLIP config matching FlashTalk's ViT-Huge-14 checkpoint."""
    return CLIPVisionConfig(
        arch_config=CLIPVisionArchConfig(
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=1024,
        )
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

    # T5 encoder config matching FlashTalk's UMT5-XXL checkpoint
    text_encoder_configs: tuple = field(
        default_factory=lambda: (_flashtalk_t5_config(),)
    )

    # CLIP ViT-Huge-14 config
    image_encoder_config: EncoderConfig = field(default_factory=_flashtalk_clip_config)

    # FlashTalk-specific: no CFG
    use_cfg: bool = False

    # Flow shift for FlashTalk
    flow_shift: float | None = 5.0

    # Audio encoder precision
    audio_encoder_precision: str = "fp32"

    # Path to Wav2Vec2 model directory for audio-driven generation
    audio_encoder_path: str | None = None

    # Motion frame caching for streaming
    motion_frames_num: int = 5

    # Per-chunk frame count for multi-chunk generation (model training value).
    # SP may pad batch.num_frames for GPU divisibility (e.g. 33→61 with 8 GPUs),
    # but the model was trained on 33-frame chunks. This value overrides
    # batch.num_frames inside the multi-chunk loop.
    chunk_frame_num: int = 33

    # Lab color correction
    enable_lab_color_correction: bool = True

    # Cached audio duration for per-chunk sliding window (seconds).
    # Matches original FlashTalk's ``cached_audio_duration`` in infer_params.
    # Each chunk's audio is processed with an 8-second sliding deque padded
    # with silence at the beginning, matching the original streaming behavior.
    cached_audio_duration: int = 8

    # Default FlashTalk resolution
    max_area: int = 768 * 448

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
