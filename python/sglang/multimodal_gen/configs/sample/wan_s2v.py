# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for Wan2.2-S2V."""

from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class WanS2VSamplingParams(SamplingParams):
    num_frames: int = 81
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    negative_prompt: str | None = None
    flow_shift: float = 3.0
    audio_path: str | None = None
    audio_tensor: Any | None = None
    pose_video_path: str | None = None
    pose_video_tensor: Any | None = None
    stream_r1_mode: bool | None = None
    stream_r1_kv_cache: bool | None = None
    num_output_latent_frames: int | None = None
    num_frame_per_block: int | None = None
    local_attn_size: int | None = None
    sink_size: int | None = None
    context_noise: int | None = None
    denoising_steps: list[int] | None = None
    warp_denoising_step: bool | None = None
    control_policy: str | None = None
    audio_lookahead_frames: int | None = None
    init_first_frame: bool | list[bool] | None = None
    anchor_first_frame: bool | None = None
    cache_audio_embeddings: bool | None = None
    use_stream_r1_ema: bool | None = None
    adaptive_steps: bool | None = None
    adaptive_steps_threshold: float | None = None
    adaptive_steps_aggressive_threshold: float | None = None
    adaptive_steps_reduced_step_count: int | None = None
    adaptive_steps_aggressive_step_count: int | None = None
    adaptive_steps_warmup_blocks: int | None = None
    adaptive_steps_log_only: bool | None = None
    disable_sp_frame_padding: bool = True

    def __post_init__(self) -> None:
        if self.num_output_latent_frames is not None:
            if self.num_output_latent_frames <= 0:
                raise ValueError("num_output_latent_frames must be positive")
            # Wan S2V uses the Wan VAE temporal ratio of 4. Keep the public
            # request compatible with the existing latent preparation stage.
            self.num_frames = (self.num_output_latent_frames - 1) * 4 + 1

        if self.num_frame_per_block is not None and self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if self.local_attn_size is not None and self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size is not None and self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if (
            self.local_attn_size is not None
            and self.sink_size is not None
            and self.sink_size >= self.local_attn_size
        ):
            raise ValueError("sink_size must be smaller than local_attn_size")
        if (
            self.stream_r1_kv_cache
            and self.local_attn_size is not None
            and self.num_frame_per_block is not None
            and self.local_attn_size < self.num_frame_per_block
        ):
            raise ValueError(
                "local_attn_size must be at least num_frame_per_block when "
                "stream_r1_kv_cache is enabled"
            )
        if self.context_noise is not None and self.context_noise < 0:
            raise ValueError("context_noise must be non-negative")
        if self.audio_lookahead_frames is not None and self.audio_lookahead_frames < 0:
            raise ValueError("audio_lookahead_frames must be non-negative")
        if self.denoising_steps is not None and len(self.denoising_steps) == 0:
            raise ValueError("denoising_steps must not be empty")
        if self.adaptive_steps_threshold is not None and self.adaptive_steps_threshold < 0:
            raise ValueError("adaptive_steps_threshold must be non-negative")
        if (
            self.adaptive_steps_aggressive_threshold is not None
            and self.adaptive_steps_aggressive_threshold < 0
        ):
            raise ValueError("adaptive_steps_aggressive_threshold must be non-negative")
        if (
            self.adaptive_steps_reduced_step_count is not None
            and self.adaptive_steps_reduced_step_count <= 0
        ):
            raise ValueError("adaptive_steps_reduced_step_count must be positive")
        if (
            self.adaptive_steps_aggressive_step_count is not None
            and self.adaptive_steps_aggressive_step_count <= 0
        ):
            raise ValueError("adaptive_steps_aggressive_step_count must be positive")
        if (
            self.adaptive_steps_warmup_blocks is not None
            and self.adaptive_steps_warmup_blocks < 0
        ):
            raise ValueError("adaptive_steps_warmup_blocks must be non-negative")

        super().__post_init__()

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        for field_name in (
            "audio_path",
            "audio_tensor",
            "pose_video_path",
            "pose_video_tensor",
            "stream_r1_mode",
            "stream_r1_kv_cache",
            "num_output_latent_frames",
            "num_frame_per_block",
            "local_attn_size",
            "sink_size",
            "context_noise",
            "denoising_steps",
            "warp_denoising_step",
            "control_policy",
            "audio_lookahead_frames",
            "init_first_frame",
            "anchor_first_frame",
            "cache_audio_embeddings",
            "use_stream_r1_ema",
            "adaptive_steps",
            "adaptive_steps_threshold",
            "adaptive_steps_aggressive_threshold",
            "adaptive_steps_reduced_step_count",
            "adaptive_steps_aggressive_step_count",
            "adaptive_steps_warmup_blocks",
            "adaptive_steps_log_only",
        ):
            value = getattr(self, field_name)
            if value is not None:
                extra[field_name] = value
        return extra
