# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPVisionConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PAINTER_I2V_ADVANCED = "painter_i2v_advanced"
_WORKFLOW_IMAGE_LATENT_VARIANT_PARTS = "_workflow_image_latent_variant_parts"
_WORKFLOW_IMAGE_LATENTS = "workflow_image_latents"


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


def _get_painter_i2v_advanced_options(batch) -> dict | None:
    workflow = batch.extra.get("workflow") if isinstance(batch.extra, dict) else None
    if not isinstance(workflow, dict):
        return None
    effective_parameters = workflow.get("effective_parameters")
    if not isinstance(effective_parameters, dict):
        return None

    image_conditioning = effective_parameters.get("image_conditioning")
    if not isinstance(image_conditioning, dict):
        return None
    if str(image_conditioning.get("type", "")).lower() != _PAINTER_I2V_ADVANCED:
        return None

    enhanced_experts = _string_tuple(
        image_conditioning.get("enhanced_experts", ("high_noise",))
    )
    original_experts = _string_tuple(
        image_conditioning.get("original_experts", ("low_noise",))
    )
    return {
        "motion_amplitude": float(
            effective_parameters.get(
                "motion_amplitude", image_conditioning.get("motion_amplitude", 1.3)
            )
        ),
        "color_protect": bool(
            effective_parameters.get(
                "color_protect", image_conditioning.get("color_protect", True)
            )
        ),
        "correct_strength": float(
            effective_parameters.get(
                "correct_strength", image_conditioning.get("correct_strength", 0.05)
            )
        ),
        "enhanced_experts": enhanced_experts,
        "original_experts": original_experts,
    }


def _string_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _apply_painter_i2v_advanced_conditioning(
    latent_condition: torch.Tensor,
    *,
    motion_amplitude: float,
    color_protect: bool,
    correct_strength: float,
) -> torch.Tensor:
    if latent_condition.dim() != 5:
        return latent_condition

    concat_latent_image = latent_condition
    original_latent = concat_latent_image.clone()
    enhanced_latent = concat_latent_image

    if motion_amplitude > 1.0 and concat_latent_image.shape[2] > 1:
        base_latent = concat_latent_image[:, :, 0:1]
        gray_latent = concat_latent_image[:, :, 1:]
        diff = gray_latent - base_latent
        diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
        diff_centered = diff - diff_mean
        scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
        scaled_latent = torch.clamp(scaled_latent, -6, 6)
        enhanced_latent = torch.cat([base_latent, scaled_latent], dim=2)

    if color_protect and correct_strength > 0:
        corrected = enhanced_latent.clone()
        orig_mean = original_latent.mean(dim=(2, 3, 4))
        enhanced_mean = corrected.mean(dim=(2, 3, 4))
        mean_drift = torch.abs(enhanced_mean - orig_mean) / (
            torch.abs(orig_mean) + 1e-6
        )
        problem_channels = mean_drift > 0.18
        drift_amount = enhanced_mean - orig_mean
        correction = (
            drift_amount
            * problem_channels.to(dtype=corrected.dtype)
            * float(correct_strength)
            * 0.03
        )
        corrected = torch.where(
            corrected > 0,
            corrected - correction[:, :, None, None, None],
            corrected,
        )

        orig_brightness = original_latent.mean()
        enhanced_brightness = corrected.mean()
        if enhanced_brightness < orig_brightness * 0.92:
            max_boost = torch.as_tensor(
                1.05, device=corrected.device, dtype=corrected.dtype
            )
            brightness_boost = torch.minimum(
                orig_brightness / (enhanced_brightness + 1e-6), max_boost
            )
            corrected = torch.where(
                corrected < 0.5, corrected * brightness_boost, corrected
            )

        enhanced_latent = torch.clamp(corrected, -6, 6)

    return enhanced_latent


@dataclass
class WanI2VCommonConfig(PipelineConfig):
    # for all wan i2v pipelines
    def adjust_num_frames(self, num_frames):
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        if num_frames % vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
            )
            return num_frames
        return num_frames

    def postprocess_image_latent(self, latent_condition, batch):
        image_latents = super().postprocess_image_latent(latent_condition, batch)
        painter_options = _get_painter_i2v_advanced_options(batch)
        if painter_options is None:
            return image_latents

        enhanced_condition = _apply_painter_i2v_advanced_conditioning(
            latent_condition,
            motion_amplitude=painter_options["motion_amplitude"],
            color_protect=painter_options["color_protect"],
            correct_strength=painter_options["correct_strength"],
        )
        enhanced_image_latents = super().postprocess_image_latent(
            enhanced_condition, batch
        )

        variant_parts = batch.extra.setdefault(_WORKFLOW_IMAGE_LATENT_VARIANT_PARTS, {})
        for expert_name in painter_options["enhanced_experts"]:
            variant_parts.setdefault(expert_name, []).append(enhanced_image_latents)
        for expert_name in painter_options["original_experts"]:
            variant_parts.setdefault(expert_name, []).append(image_latents)

        return enhanced_image_latents

    def finalize_image_latent_variants(self, batch) -> None:
        variant_parts = batch.extra.pop(_WORKFLOW_IMAGE_LATENT_VARIANT_PARTS, None)
        if not isinstance(variant_parts, dict):
            return

        batch.extra[_WORKFLOW_IMAGE_LATENTS] = {
            str(expert_name): torch.cat(parts, dim=1)
            for expert_name, parts in variant_parts.items()
            if parts
        }


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            auto_dit_layerwise_offload_high_memory_disable_gb=130,
        )


@dataclass
class TurboWanT2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [988, 932, 852, 608]
    )


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig, WanI2VCommonConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    max_area: int = 480 * 832
    # WanConfig-specific parameters with defaults
    task_type: ModelTaskType = ModelTaskType.I2V
    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            auto_dit_layerwise_offload_high_memory_disable_gb=130,
        )


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    max_area: int = 720 * 1280
    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class TurboWanI2V720Config(WanI2V720PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [996, 932, 852, 608]
    )
    boundary_ratio: float | None = 0.9

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
