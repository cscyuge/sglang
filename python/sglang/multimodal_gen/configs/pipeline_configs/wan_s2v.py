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
    text_encoder_configs: tuple = field(
        default_factory=lambda: (_flashtalk_t5_config(),)
    )
    use_cfg: bool = True
    flow_shift: float | None = None
    audio_encoder_precision: str = "fp32"
    audio_encoder_path: str | None = None
    max_area: int = 1024 * 704
    force_condition_image_to_requested_size: bool = True
    stream_r1_mode: bool = False
    stream_r1_kv_cache: bool = False
    stream_r1_crossattn_cache: bool = True
    num_frame_per_block: int = 7
    local_attn_size: int = 9
    sink_size: int = 3
    context_noise: int = 0
    denoising_step_list: list[int] | None = None
    warp_denoising_step: bool = True
    s2v_control_policy: str = "lookahead"
    s2v_audio_lookahead_frames: int = 2
    s2v_init_first_frame: bool = False
    s2v_anchor_first_frame: bool = False
    cache_audio_embeddings: bool = True
    stream_r1_generator_checkpoint_path: str | None = None
    use_stream_r1_ema: bool = False
    wan_s2v_realtime: bool = True
    wan_s2v_realtime_audio_window_seconds: float = 8.0
    wan_s2v_idle_policy: str = "hold"
    wan_s2v_max_silence_blocks: int = 1
    wan_s2v_audio_overlap: bool = False
    wan_s2v_latent_condition_overlap: bool = False
    wan_s2v_wav2vec_cuda_graph: bool = False
    wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps: tuple[int, ...] | list[int] | None = (
        16,
        24,
        25,
    )
    wan_s2v_streaming_vae_cache: bool = True
    wan_s2v_vae_cuda_graph: bool = False
    wan_s2v_adaptive_steps: bool = False
    wan_s2v_adaptive_steps_threshold: float = 0.08
    wan_s2v_adaptive_steps_aggressive_threshold: float = 0.0
    wan_s2v_adaptive_steps_reduced_step_count: int = 2
    wan_s2v_adaptive_steps_aggressive_step_count: int = 1
    wan_s2v_adaptive_steps_warmup_blocks: int = 1
    wan_s2v_adaptive_steps_log_only: bool = False
    wan_s2v_latent_warm_start: bool = False
    wan_s2v_latent_warm_start_alpha: float = 0.25
    wan_s2v_latent_warm_start_mode: str = "repeat_tail"
    wan_s2v_latent_warm_start_warmup_blocks: int = 1
    wan_s2v_latent_warm_start_timestep_index: int = 1
    wan_s2v_latent_warm_start_effective_sigma: float | None = None
    wan_s2v_latent_warm_start_log: bool = False
    wan_s2v_clean_context_refresh_mode: str = "interval"
    wan_s2v_clean_context_refresh_interval: int = 2
    wan_s2v_clean_context_refresh_warmup_blocks: int = 1
    wan_s2v_clean_context_refresh_log: bool = False
    wan_s2v_timestep_profile: bool = False
    wan_s2v_timestep_profile_log: bool = False
    wan_s2v_timestep_profile_nvtx: bool = True
    wan_s2v_timestep_profile_sync: bool = False
    wan_s2v_timestep_ablation_mode: str = "reuse_previous_pred"
    wan_s2v_timestep_ablation_indices: list[int] | None = field(
        default_factory=lambda: [1]
    )
    wan_s2v_timestep_ablation_values: list[float] | None = None
    wan_s2v_timestep_ablation_blocks: list[int] | None = None
    wan_s2v_timestep_ablation_warmup_blocks: int = 2
    wan_s2v_timestep_ablation_value_tolerance: float = 1e-3
    wan_s2v_timestep_ablation_scale: float = 1.0
    wan_s2v_timestep_ablation_log: bool = False
    wan_s2v_timestep_cuda_graph: bool = True
    wan_s2v_timestep_cuda_graph_indices: list[int] | None = field(
        default_factory=lambda: [0]
    )
    wan_s2v_timestep_cuda_graph_warmup_blocks: int = 2
    wan_s2v_timestep_cuda_graph_max_graphs: int = 16
    wan_s2v_timestep_cuda_graph_log: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        if self.flow_shift is None:
            self.flow_shift = 5.0 if self.stream_r1_mode else 3.0
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.stream_r1_kv_cache and self.local_attn_size < self.num_frame_per_block:
            raise ValueError(
                "local_attn_size must be at least num_frame_per_block when "
                "stream_r1_kv_cache is enabled"
            )
        if self.context_noise < 0:
            raise ValueError("context_noise must be non-negative")
        if self.s2v_audio_lookahead_frames < 0:
            raise ValueError("s2v_audio_lookahead_frames must be non-negative")
        if self.denoising_step_list is not None and len(self.denoising_step_list) == 0:
            raise ValueError("denoising_step_list must not be empty")
        if self.wan_s2v_realtime_audio_window_seconds <= 0:
            raise ValueError("wan_s2v_realtime_audio_window_seconds must be positive")
        if self.wan_s2v_idle_policy not in ("hold", "silence"):
            raise ValueError("wan_s2v_idle_policy must be either 'hold' or 'silence'")
        if self.wan_s2v_max_silence_blocks < 0:
            raise ValueError("wan_s2v_max_silence_blocks must be non-negative")
        if self.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps is None:
            self.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps = ()
        else:
            self.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps = tuple(
                int(fps)
                for fps in self.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps
            )
            if any(
                fps <= 0
                for fps in self.wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps
            ):
                raise ValueError(
                    "wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps must contain "
                    "positive FPS values"
                )
        if self.wan_s2v_adaptive_steps_threshold < 0:
            raise ValueError("wan_s2v_adaptive_steps_threshold must be non-negative")
        if self.wan_s2v_adaptive_steps_aggressive_threshold < 0:
            raise ValueError(
                "wan_s2v_adaptive_steps_aggressive_threshold must be non-negative"
            )
        if self.wan_s2v_adaptive_steps_reduced_step_count <= 0:
            raise ValueError(
                "wan_s2v_adaptive_steps_reduced_step_count must be positive"
            )
        if self.wan_s2v_adaptive_steps_aggressive_step_count <= 0:
            raise ValueError(
                "wan_s2v_adaptive_steps_aggressive_step_count must be positive"
            )
        if self.wan_s2v_adaptive_steps_warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_adaptive_steps_warmup_blocks must be non-negative"
            )
        if not (0.0 <= self.wan_s2v_latent_warm_start_alpha <= 1.0):
            raise ValueError("wan_s2v_latent_warm_start_alpha must be in [0, 1]")
        self.wan_s2v_latent_warm_start_mode = str(
            self.wan_s2v_latent_warm_start_mode
        ).lower()
        if self.wan_s2v_latent_warm_start_mode not in ("repeat_tail", "linear"):
            raise ValueError(
                "wan_s2v_latent_warm_start_mode must be 'repeat_tail' or 'linear'"
            )
        if self.wan_s2v_latent_warm_start_warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_latent_warm_start_warmup_blocks must be non-negative"
            )
        if self.wan_s2v_latent_warm_start_timestep_index < 0:
            raise ValueError(
                "wan_s2v_latent_warm_start_timestep_index must be non-negative"
            )
        if self.wan_s2v_latent_warm_start_effective_sigma is not None and not (
            0.0 <= self.wan_s2v_latent_warm_start_effective_sigma <= 1.0
        ):
            raise ValueError(
                "wan_s2v_latent_warm_start_effective_sigma must be in [0, 1]"
            )
        self.wan_s2v_clean_context_refresh_mode = str(
            self.wan_s2v_clean_context_refresh_mode
        ).lower()
        if self.wan_s2v_clean_context_refresh_mode not in (
            "always",
            "interval",
            "never",
        ):
            raise ValueError(
                "wan_s2v_clean_context_refresh_mode must be one of "
                "'always', 'interval', or 'never'"
            )
        if self.wan_s2v_clean_context_refresh_interval <= 0:
            raise ValueError("wan_s2v_clean_context_refresh_interval must be positive")
        if self.wan_s2v_clean_context_refresh_warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_clean_context_refresh_warmup_blocks must be non-negative"
            )
        self.wan_s2v_timestep_ablation_mode = str(
            self.wan_s2v_timestep_ablation_mode
        ).lower()
        if self.wan_s2v_timestep_ablation_mode not in (
            "off",
            "skip_update",
            "reuse_previous_pred",
            "zero_pred",
            "scale_pred",
        ):
            raise ValueError(
                "wan_s2v_timestep_ablation_mode must be one of 'off', "
                "'skip_update', 'reuse_previous_pred', 'zero_pred', or "
                "'scale_pred'"
            )
        if self.wan_s2v_timestep_ablation_warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_timestep_ablation_warmup_blocks must be non-negative"
            )
        if self.wan_s2v_timestep_ablation_value_tolerance < 0:
            raise ValueError(
                "wan_s2v_timestep_ablation_value_tolerance must be non-negative"
            )
        if self.wan_s2v_timestep_cuda_graph_warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_timestep_cuda_graph_warmup_blocks must be non-negative"
            )
        if self.wan_s2v_timestep_cuda_graph_max_graphs <= 0:
            raise ValueError("wan_s2v_timestep_cuda_graph_max_graphs must be positive")
        for field_name in (
            "wan_s2v_timestep_ablation_indices",
            "wan_s2v_timestep_ablation_blocks",
            "wan_s2v_timestep_cuda_graph_indices",
        ):
            values = getattr(self, field_name)
            if values is not None and any(int(value) < 0 for value in values):
                raise ValueError(f"{field_name} values must be non-negative")

    def postprocess_image_latent(self, latent_condition, batch):
        return latent_condition
