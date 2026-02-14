# SPDX-License-Identifier: Apache-2.0
"""
FlashTalk-specific pipeline stages.

Contains:
- FlashTalkDenoisingStage: Denoising loop without CFG, with audio context
- FlashTalkColorCorrectionStage: Lab color space correction
"""

import os
import time

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class FlashTalkDenoisingStage(PipelineStage):
    """Denoising stage for FlashTalk.

    Key differences from standard DenoisingStage:
    - No classifier-free guidance (single forward pass per step)
    - Passes audio_context to transformer at each step
    - Uses flow-matching denoising schedule
    - Supports motion frame caching for streaming generation
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self._maybe_enable_torch_compile(self.transformer)

    def _maybe_enable_torch_compile(self, module: object) -> None:
        """Compile a module with torch.compile if enabled.

        Mirrors the pattern from DenoisingStage. No-op if torch compile is
        disabled or the object is not an nn.Module.
        """
        if not self.server_args.enable_torch_compile or not isinstance(
            module, nn.Module
        ):
            return
        try:
            import torch._inductor.config as _inductor_cfg

            _inductor_cfg.reorder_for_compute_comm_overlap = True
        except ImportError:
            pass
        mode = os.environ.get(
            "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
        )
        logger.info("Compiling FlashTalk transformer with mode: %s", mode)
        module.compile(mode=mode, fullgraph=False, dynamic=None)

    def _timestep_transform(
        self, t: torch.Tensor, shift: float = 5.0, num_timesteps: int = 1000
    ) -> torch.Tensor:
        """Apply timestep shift transformation for flow matching."""
        t_normalized = t / num_timesteps
        new_t = shift * t_normalized / (1 + (shift - 1) * t_normalized)
        return new_t * num_timesteps

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """Run the FlashTalk denoising loop (no CFG)."""

        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        sp_size = get_sp_world_size()

        # Enable transformer-internal sequence sharding for multi-GPU SP.
        # This lets the transformer handle sequence split/gather internally,
        # so FlashTalk-specific tensors (image_latent, motion_latent,
        # audio_context) don't need external sharding.
        use_sp = sp_size > 1
        if use_sp:
            batch.enable_sequence_shard = True

        # Skip per-step CPU offload when using SP: moving the full 14B model
        # between CPU and GPU at every denoising step is prohibitively slow.
        use_cpu_offload = (
            server_args.dit_cpu_offload
            and not server_args.use_fsdp_inference
            and not use_sp
        )

        # Move transformer to GPU if CPU-offloaded
        if use_cpu_offload and next(self.transformer.parameters()).device.type == "cpu":
            self.transformer.to(device)
        elif use_sp and next(self.transformer.parameters()).device.type == "cpu":
            # For multi-GPU SP, move to GPU once and keep it there
            logger.info("Moving transformer to GPU (SP mode, skipping per-step offload)")
            self.transformer.to(device)

        latents = batch.latents.to(device=device, dtype=dit_dtype)
        timesteps = batch.timesteps
        audio_context = batch.extra.get("audio_context")
        human_num = batch.extra.get("human_num", 1)

        # Get conditioning
        encoder_hidden_states = batch.prompt_embeds
        if isinstance(encoder_hidden_states, list) and len(encoder_hidden_states) > 0:
            encoder_hidden_states = encoder_hidden_states[0]

        image_embeds = batch.image_embeds
        encoder_hidden_states_image = None
        if image_embeds and len(image_embeds) > 0:
            encoder_hidden_states_image = image_embeds[0]

        # Prepare image latent conditioning (y)
        if batch.image_latent is not None:
            latent_model_input_extra = batch.image_latent.to(
                device=device, dtype=dit_dtype
            )
        else:
            latent_model_input_extra = None

        # Flow shift
        flow_shift = server_args.pipeline_config.flow_shift or 5.0
        num_timesteps = 1000

        # Build timesteps with shift — match original FlashTalk schedule exactly.
        # Original uses DESCENDING timesteps [1000, 750, 500, 250] + [0],
        # then applies timestep_transform(shift=5) to each.
        if timesteps is None:
            num_steps = batch.num_inference_steps or 4
            if num_steps == 2:
                ts_list = [1000, 500]
            elif num_steps == 4:
                ts_list = [1000, 750, 500, 250]
            else:
                import numpy as np

                ts_list = list(
                    np.linspace(num_timesteps, 1, num_steps, dtype=np.float32)
                )
            ts_list.append(0.0)
            timesteps = [
                self._timestep_transform(
                    torch.tensor([t], device=device),
                    shift=flow_shift,
                    num_timesteps=num_timesteps,
                )
                for t in ts_list
            ]
        else:
            # Ensure timesteps ends with 0
            timesteps = list(timesteps)
            if not isinstance(timesteps[-1], torch.Tensor):
                timesteps = [
                    torch.tensor([t], device=device) for t in timesteps
                ]
            timesteps.append(torch.tensor([0.0], device=device))

        # Apply motion frame latent if available
        motion_latent = batch.extra.get("motion_latent")
        if motion_latent is not None:
            n_motion = motion_latent.shape[2] if motion_latent.dim() == 5 else motion_latent.shape[1]
            if latents.dim() == 5:
                latents[:, :, :n_motion] = motion_latent
            else:
                latents[:, :n_motion] = motion_latent

        generator = batch.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None

        autocast_enabled = dit_dtype != torch.float32 and not getattr(
            server_args, "disable_autocast", False
        )

        denoising_start_time = time.time()
        num_steps = len(timesteps) - 1

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=dit_dtype,
            enabled=autocast_enabled,
        ):
            for i in range(num_steps):
                with StageProfiler(
                    f"flashtalk_denoising_step_{i}",
                    logger=logger,
                    timings=batch.timings,
                    perf_dump_path_provided=(
                        batch.perf_dump_path is not None
                        if hasattr(batch, "perf_dump_path")
                        else False
                    ),
                ):
                    t_i = timesteps[i]
                    if t_i.dim() == 0:
                        t_i = t_i.unsqueeze(0)

                    # Prepare model input
                    latent_model_input = latents.to(dit_dtype)
                    if latent_model_input_extra is not None:
                        latent_model_input = torch.cat(
                            [latent_model_input, latent_model_input_extra], dim=1
                        ).to(dit_dtype)

                    # Single forward pass (no CFG)
                    with set_forward_context(
                        current_timestep=i,
                        attn_metadata=None,
                        forward_batch=batch,
                    ):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=t_i,
                            encoder_hidden_states_image=encoder_hidden_states_image,
                            audio_context=audio_context,
                            human_num=human_num,
                        )

                    # FlashTalk uses negative noise prediction
                    noise_pred = -noise_pred

                    # Flow matching update step
                    t_cur = t_i[:, None, None, None] / num_timesteps
                    t_next_val = timesteps[i + 1]
                    if t_next_val.dim() == 0:
                        t_next_val = t_next_val.unsqueeze(0)
                    t_next = t_next_val[:, None, None, None] / num_timesteps
                    x_0 = latents + noise_pred * t_cur
                    latents = (1 - t_next) * x_0 + t_next * torch.randn(
                        x_0.size(),
                        dtype=x_0.dtype,
                        device=device,
                        generator=generator,
                    )

                    # Re-apply motion frame latent
                    if motion_latent is not None:
                        if latents.dim() == 5:
                            latents[:, :, :n_motion] = motion_latent
                        else:
                            latents[:, :n_motion] = motion_latent

        denoising_end_time = time.time()
        if num_steps > 0:
            logger.info(
                "FlashTalk denoising: avg %.4f s/step",
                (denoising_end_time - denoising_start_time) / num_steps,
            )

        batch.latents = latents

        # Cache motion frames for streaming
        motion_frames_num = server_args.pipeline_config.motion_frames_num
        if motion_frames_num > 0:
            batch.extra["motion_latent_output"] = latents[
                :, :, -motion_frames_num:
            ].clone()

        # Offload transformer back to CPU if needed (skipped for SP mode)
        if use_cpu_offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

        return batch

    def verify_input(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def verify_output(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result


# ============================================
# Lab Color Correction Utilities
# ============================================


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to Lab color space. Last dim must be 3."""
    # sRGB gamma correction inverse
    linear_rgb = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    # Linear RGB to XYZ (D65 white point)
    xyz_from_rgb = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    shape = linear_rgb.shape
    xyz = linear_rgb.reshape(-1, 3) @ xyz_from_rgb.T
    xyz = xyz.reshape(shape)

    # XYZ to Lab
    xyz_ref = torch.tensor(
        [0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device
    )
    xyz_normalized = xyz / xyz_ref
    xyz_normalized = torch.clamp(xyz_normalized, 1e-8, None)

    epsilon = 0.008856
    kappa = 903.3
    f_xyz = torch.where(
        xyz_normalized > epsilon,
        xyz_normalized ** (1 / 3),
        (kappa * xyz_normalized + 16) / 116,
    )

    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    return torch.stack([L, a, b], dim=-1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert Lab to RGB [0,1]. Last dim must be 3."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    f_y = (L + 16) / 116
    f_x = (a / 500) + f_y
    f_z = f_y - (b / 200)

    epsilon = 0.008856
    kappa = 903.3

    x = torch.where(f_x**3 > epsilon, f_x**3, (116 * f_x - 16) / kappa)
    y = torch.where(
        L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa
    )
    z = torch.where(f_z**3 > epsilon, f_z**3, (116 * f_z - 16) / kappa)

    xyz_ref = torch.tensor(
        [0.95047, 1.0, 1.08883], dtype=lab.dtype, device=lab.device
    )
    xyz = torch.stack([x, y, z], dim=-1) * xyz_ref

    rgb_from_xyz = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=lab.dtype,
        device=lab.device,
    )
    shape = xyz.shape
    rgb = xyz.reshape(-1, 3) @ rgb_from_xyz.T
    rgb = rgb.reshape(shape)

    # Gamma correction
    rgb = torch.where(
        rgb > 0.0031308,
        1.055 * (rgb ** (1 / 2.4)) - 0.055,
        12.92 * rgb,
    )
    return torch.clamp(rgb, 0.0, 1.0)


def match_and_blend_colors(
    source: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Match source video colors to reference image using Lab color transfer.

    Args:
        source: (B, C, T, H, W) video in [-1, 1]
        reference: (B, C, 1, H, W) reference image in [-1, 1]
        strength: blending strength [0, 1]

    Returns:
        Color-corrected video (B, C, T, H, W) in [-1, 1]
    """
    if strength <= 0.0:
        return source.clone()

    B, C, T, H, W = source.shape

    # [-1,1] -> [0,1]
    source_01 = (source + 1.0) / 2.0
    ref_01 = (reference + 1.0) / 2.0

    # (B, C, T, H, W) -> (B, T, H, W, C)
    source_p = source_01.permute(0, 2, 3, 4, 1)
    ref_p = ref_01.permute(0, 2, 3, 4, 1)

    source_lab = _rgb_to_lab(source_p)
    ref_lab = _rgb_to_lab(ref_p)

    ref_mean = ref_lab.mean(dim=[2, 3], keepdim=True)
    ref_std = ref_lab.std(dim=[2, 3], keepdim=True, unbiased=False)
    source_mean = source_lab.mean(dim=[2, 3], keepdim=True)
    source_std = source_lab.std(dim=[2, 3], keepdim=True, unbiased=False)
    source_std_safe = torch.where(
        source_std < 1e-8, torch.ones_like(source_std), source_std
    )

    corrected_lab = (
        (source_lab - source_mean) * (ref_std / source_std_safe) + ref_mean
    )
    corrected_rgb = _lab_to_rgb(corrected_lab)
    blended = (1 - strength) * source_p + strength * corrected_rgb

    # (B, T, H, W, C) -> (B, C, T, H, W), [0,1] -> [-1,1]
    result = blended.permute(0, 4, 1, 2, 3) * 2.0 - 1.0
    return result.contiguous()


class FlashTalkColorCorrectionStage(PipelineStage):
    """Lab color correction stage for FlashTalk.

    Matches the color statistics of generated frames to the reference image
    in Lab color space. Controlled by `enable_lab_color_correction` and
    `color_correction_strength` in the pipeline config.
    """

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """Apply Lab color correction to decoded output."""
        enable = getattr(
            server_args.pipeline_config, "enable_lab_color_correction", False
        )
        extra = getattr(batch, "extra", {}) or {}
        strength = extra.get("color_correction_strength", 1.0)

        if not enable or strength <= 0.0 or batch.output is None:
            return batch

        # Get reference image (the condition image)
        reference = extra.get("color_reference")
        if reference is None:
            return batch

        # output is (B, C, T, H, W) in [-1, 1]
        batch.output = match_and_blend_colors(
            batch.output, reference, strength
        )

        return batch

    def verify_input(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        return VerificationResult()

    def verify_output(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        return VerificationResult()
