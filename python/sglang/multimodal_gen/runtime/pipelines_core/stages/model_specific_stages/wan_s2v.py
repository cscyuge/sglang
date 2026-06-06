# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V specific pipeline stages."""

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VKVCacheBlock,
)
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.audio_encoding import (
    AudioEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


@dataclass(frozen=True)
class WanS2VStreamR1AttentionRequest:
    stream_r1_kv_cache: bool
    num_frame_per_block: int
    local_attn_size: int
    sink_size: int
    context_noise: int

    def validate(self, *, latent_frames: int, train_timesteps: int) -> None:
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if latent_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "Stream-R1 S2V requires latent frames to be divisible by "
                f"num_frame_per_block, got latent_frames={latent_frames}, "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.local_attn_size < self.num_frame_per_block:
            raise ValueError(
                "local_attn_size must be at least num_frame_per_block for "
                "Stream-R1 S2V local/KV attention"
            )
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.context_noise < 0:
            raise ValueError("context_noise must be non-negative")
        if self.context_noise > train_timesteps:
            raise ValueError(
                f"context_noise must be in [0, {train_timesteps}], "
                f"got {self.context_noise}"
            )


@dataclass(frozen=True)
class WanS2VStreamR1CacheMetadata:
    batch_size: int
    num_layers: int
    frame_seq_length: int
    local_num_attention_heads: int
    attention_head_dim: int
    local_attn_size: int
    sink_size: int
    dtype: torch.dtype
    device: torch.device

    @property
    def cache_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    @property
    def bytes_per_kv_cache(self) -> int:
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.num_layers
            * self.batch_size
            * self.cache_tokens
            * self.local_num_attention_heads
            * self.attention_head_dim
            * 2
            * itemsize
        )


@dataclass
class WanS2VStreamR1CacheState:
    metadata: WanS2VStreamR1CacheMetadata | None = None
    kv_cache: list[WanS2VKVCacheBlock] | None = None

    @property
    def enabled(self) -> bool:
        return self.metadata is not None

    @property
    def allocated(self) -> bool:
        return self.kv_cache is not None

    @classmethod
    def disabled(cls) -> "WanS2VStreamR1CacheState":
        return cls()

    @classmethod
    def metadata_only(
        cls, metadata: WanS2VStreamR1CacheMetadata
    ) -> "WanS2VStreamR1CacheState":
        return cls(metadata=metadata)

    @classmethod
    def allocate(
        cls, metadata: WanS2VStreamR1CacheMetadata
    ) -> "WanS2VStreamR1CacheState":
        kv_cache: list[WanS2VKVCacheBlock] = []
        for _ in range(metadata.num_layers):
            kv_cache.append(
                {
                    "k": torch.zeros(
                        (
                            metadata.batch_size,
                            metadata.cache_tokens,
                            metadata.local_num_attention_heads,
                            metadata.attention_head_dim,
                        ),
                        dtype=metadata.dtype,
                        device=metadata.device,
                    ),
                    "v": torch.zeros(
                        (
                            metadata.batch_size,
                            metadata.cache_tokens,
                            metadata.local_num_attention_heads,
                            metadata.attention_head_dim,
                        ),
                        dtype=metadata.dtype,
                        device=metadata.device,
                    ),
                    "global_end_index": torch.zeros(
                        (1,), dtype=torch.long, device=metadata.device
                    ),
                    "local_end_index": torch.zeros(
                        (1,), dtype=torch.long, device=metadata.device
                    ),
                }
            )
        return cls(metadata=metadata, kv_cache=kv_cache)

    def reset(self) -> None:
        if self.kv_cache is None:
            return
        for block_cache in self.kv_cache:
            block_cache["global_end_index"].zero_()
            block_cache["local_end_index"].zero_()


@dataclass
class WanS2VConditionBundle:
    """Request-scoped Wan S2V conditions stored in BCTHW latent layout."""

    prompt_embeds: torch.Tensor | list[torch.Tensor]
    ref_latents: torch.Tensor
    motion_latents: torch.Tensor
    cond_states: torch.Tensor
    audio_input: torch.Tensor | None
    audio_emb: Any | None = None
    motion_frames: tuple[int, int] = (73, 19)
    add_last_motion: int = 2
    drop_motion_frames: bool = False
    control_policy: str = "lookahead"
    chunk_start: int = 0
    chunk_frames: int | None = None
    audio_lookahead_frames: int = 2
    audio_metadata: dict[str, Any] = field(default_factory=dict)

    def slice(
        self,
        start: int,
        frames: int,
        policy: str | None = None,
    ) -> "WanS2VConditionBundle":
        if start < 0:
            raise ValueError("start must be non-negative")
        if frames <= 0:
            raise ValueError("frames must be positive")

        return WanS2VConditionBundle(
            prompt_embeds=self.prompt_embeds,
            ref_latents=self.ref_latents,
            motion_latents=self.motion_latents,
            cond_states=_slice_or_pad_bcthw(self.cond_states, start, frames),
            audio_input=self.audio_input,
            audio_emb=self.audio_emb,
            motion_frames=self.motion_frames,
            add_last_motion=self.add_last_motion,
            drop_motion_frames=self.drop_motion_frames,
            control_policy=policy or self.control_policy,
            chunk_start=start,
            chunk_frames=frames,
            audio_lookahead_frames=self.audio_lookahead_frames,
            audio_metadata={
                **self.audio_metadata,
                "chunk_start": start,
                "chunk_frames": frames,
                "control_policy": policy or self.control_policy,
            },
        )


def _slice_or_pad_bcthw(tensor: torch.Tensor, start: int, frames: int) -> torch.Tensor:
    if tensor.dim() != 5:
        raise ValueError(f"Expected BCTHW tensor, got shape {tuple(tensor.shape)}")
    end = start + frames
    sliced = tensor[:, :, start : min(end, tensor.shape[2])]
    if sliced.shape[2] == frames:
        return sliced
    pad_shape = list(sliced.shape)
    pad_shape[2] = frames - sliced.shape[2]
    pad = torch.zeros(
        pad_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([sliced, pad], dim=2)


def _resolve_motion_frames(raw_motion_frames: Any) -> tuple[int, int]:
    if isinstance(raw_motion_frames, (list, tuple)):
        if len(raw_motion_frames) != 2:
            raise ValueError("motion_frames must be an int or a pair of ints")
        motion_frames = int(raw_motion_frames[0])
        latent_motion_frames = int(raw_motion_frames[1])
    else:
        motion_frames = int(raw_motion_frames)
        latent_motion_frames = (motion_frames + 3) // 4
    if motion_frames <= 0 or latent_motion_frames <= 0:
        raise ValueError("motion_frames values must be positive")
    return motion_frames, latent_motion_frames


def _to_bcthw(
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value.dim() != 5:
        raise ValueError(f"{name} must be a BCTHW tensor, got {tuple(value.shape)}")
    return value.to(device=device, dtype=dtype)


def _prepare_s2v_audio_input(
    audio_input: torch.Tensor | None,
    *,
    target_audio_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if audio_input is None:
        return None
    audio_input = audio_input.to(device=device, dtype=dtype)
    if audio_input.shape[-1] < target_audio_frames:
        pad_frames = target_audio_frames - audio_input.shape[-1]
        audio_input = torch.nn.functional.pad(audio_input, (0, pad_frames))
    elif audio_input.shape[-1] > target_audio_frames:
        audio_input = audio_input[..., :target_audio_frames]
    return audio_input


def build_wan_s2v_condition_bundle(
    batch: Req,
    server_args: ServerArgs,
    *,
    latents: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> WanS2VConditionBundle:
    ref_latents = batch.image_latent
    if ref_latents is None:
        raise ValueError("Wan S2V requires a reference image")
    if isinstance(ref_latents, list):
        raise ValueError("Wan S2V requires tensor image latents in BCTHW layout")
    ref_latents = ref_latents[:, :, :1].to(device=device, dtype=dtype)

    audio_input = batch.extra.get("audio_input")
    if audio_input is None:
        raise ValueError("Wan S2V requires audio_path/audio_tensor")
    audio_input = _prepare_s2v_audio_input(
        audio_input,
        target_audio_frames=latents.shape[2] * 4,
        dtype=dtype,
        device=device,
    )

    arch_config = server_args.pipeline_config.dit_config.arch_config
    motion_frames = _resolve_motion_frames(getattr(arch_config, "motion_frames", 73))
    batch_size, _, _, latent_h, latent_w = latents.shape

    motion_latents = batch.extra.get("motion_latents")
    if motion_latents is None:
        motion_latents = torch.zeros(
            batch_size,
            16,
            motion_frames[1],
            latent_h,
            latent_w,
            dtype=dtype,
            device=device,
        )
    else:
        motion_latents = _to_bcthw(
            motion_latents, dtype=dtype, device=device, name="motion_latents"
        )

    cond_states = batch.extra.get("cond_states")
    if cond_states is None:
        cond_states = batch.extra.get("pose_latents")
    if cond_states is None:
        cond_states = torch.zeros_like(latents)
    else:
        cond_states = _to_bcthw(
            cond_states, dtype=dtype, device=device, name="cond_states"
        )
        cond_states = _slice_or_pad_bcthw(cond_states, 0, latents.shape[2])

    prompt_embeds = batch.prompt_embeds
    if isinstance(prompt_embeds, list):
        prompt_embeds = prompt_embeds[0]

    return WanS2VConditionBundle(
        prompt_embeds=prompt_embeds,
        ref_latents=ref_latents,
        motion_latents=motion_latents,
        cond_states=cond_states,
        audio_input=audio_input,
        audio_emb=batch.extra.get("audio_emb"),
        motion_frames=motion_frames,
        add_last_motion=int(batch.extra.get("add_last_motion", 2)),
        drop_motion_frames=bool(batch.extra.get("drop_motion_frames", False)),
        control_policy=_resolve_request_value(
            batch,
            server_args,
            "control_policy",
            "s2v_control_policy",
            "lookahead",
        ),
        chunk_start=0,
        chunk_frames=latents.shape[2],
        audio_lookahead_frames=int(
            _resolve_request_value(
                batch,
                server_args,
                "audio_lookahead_frames",
                "s2v_audio_lookahead_frames",
                2,
            )
        ),
        audio_metadata={
            "audio_path": batch.extra.get("audio_path"),
            "cache_audio_embeddings": _resolve_request_value(
                batch,
                server_args,
                "cache_audio_embeddings",
                "cache_audio_embeddings",
                True,
            ),
        },
    )


def _resolve_request_value(
    batch: Req,
    server_args: ServerArgs,
    request_key: str,
    config_key: str | None = None,
    default: Any = None,
) -> Any:
    if request_key in batch.extra and batch.extra[request_key] is not None:
        return batch.extra[request_key]
    try:
        value = getattr(batch, request_key)
    except AttributeError:
        value = None
    if value is not None:
        return value
    if config_key is None:
        config_key = request_key
    return getattr(server_args.pipeline_config, config_key, default)


def _safe_sp_world_size() -> int:
    try:
        return get_sp_world_size()
    except AssertionError:
        return 1


def _coerce_timestep_list(value: Any, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, str):
        value = [part for part in value.replace(",", " ").split(" ") if part]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list of integer timesteps")
    timesteps = [int(item) for item in value]
    if not timesteps:
        raise ValueError(f"{field_name} must not be empty")
    return timesteps


def _has_negative_prompt_embeds(batch: Req) -> bool:
    negative_prompt_embeds = getattr(batch, "negative_prompt_embeds", None)
    if negative_prompt_embeds is None:
        return False
    if isinstance(negative_prompt_embeds, (list, tuple)):
        return any(item is not None for item in negative_prompt_embeds)
    return True


class WanS2VAudioEncodingStage(AudioEncodingStage):
    """Encode audio into raw Wav2Vec hidden states consumed by WanModel_S2V."""

    def __init__(self, audio_encoder=None, wav2vec_feature_extractor=None) -> None:
        super().__init__(
            audio_encoder=audio_encoder,
            audio_proj=None,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        audio_path = batch.extra.get("audio_path")
        audio_tensor = batch.extra.get("audio_tensor")
        if audio_path is None and audio_tensor is None:
            return batch

        device = get_local_torch_device()
        self.load_model()
        sample_rate = 16000
        fps = batch.fps if hasattr(batch, "fps") and batch.fps else 16

        if audio_tensor is not None:
            speech_array = audio_tensor
        else:
            speech_array = self._load_audio(audio_path, sample_rate)
        if isinstance(speech_array, np.ndarray):
            speech_array = self._loudness_norm(speech_array, sample_rate)
            audio_duration = len(speech_array) / sample_rate
        else:
            audio_duration = speech_array.shape[-1] / sample_rate

        num_video_frames = max(1, int(audio_duration * fps))
        if self.wav2vec_feature_extractor is not None:
            audio_feature = np.squeeze(
                self.wav2vec_feature_extractor(
                    speech_array, sampling_rate=sample_rate
                ).input_values
            )
            audio_feature = torch.from_numpy(audio_feature).float().to(device).unsqueeze(0)
        else:
            audio_feature = (
                torch.from_numpy(speech_array).float().to(device)
                if isinstance(speech_array, np.ndarray)
                else speech_array.float().to(device)
            )
            if audio_feature.dim() == 1:
                audio_feature = audio_feature.unsqueeze(0)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            audio_features = self.audio_encoder(
                audio_feature, num_video_frames=num_video_frames
            )
        # Wav2Vec wrapper returns [B, T, num_layers, hidden]. S2V expects
        # [B, num_layers, hidden, T].
        batch.extra["audio_input"] = audio_features.permute(0, 2, 3, 1).contiguous()
        batch.extra["audio_features_all"] = audio_features
        self.offload_model()
        return batch


class WanS2VDenoisingStage(PipelineStage):
    """Denoising loop for Wan2.2-S2V.

    This covers the single-clip path: reference image latent, zero pose
    condition, zero initial motion latent, Wav2Vec hidden states, and CFG.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    def load_model(self):
        if self.server_args.dit_cpu_offload:
            self.transformer.to(get_local_torch_device())

    def offload_model(self):
        if self.server_args.dit_cpu_offload:
            self.transformer.to("cpu")

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        latents = batch.latents.to(device=device, dtype=dit_dtype)
        bundle = build_wan_s2v_condition_bundle(
            batch,
            server_args,
            latents=latents,
            dtype=dit_dtype,
            device=device,
        )
        ref_latents = bundle.ref_latents
        audio_input = bundle.audio_input
        motion_latents = bundle.motion_latents
        cond_states = bundle.cond_states
        motion_frames, lat_motion_frames = bundle.motion_frames

        prompt_embeds = batch.prompt_embeds
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        negative_prompt_embeds = None
        if getattr(batch, "negative_prompt_embeds", None) is not None:
            negative_prompt_embeds = batch.negative_prompt_embeds
            if isinstance(negative_prompt_embeds, list):
                negative_prompt_embeds = (
                    negative_prompt_embeds[0] if negative_prompt_embeds else None
                )

        timesteps = batch.timesteps
        if timesteps is None:
            num_steps = batch.num_inference_steps or 40
            self.scheduler.set_timesteps(
                num_steps,
                device=device,
                shift=server_args.pipeline_config.flow_shift or 3.0,
            )
            timesteps = self.scheduler.timesteps

        guidance_scale = batch.guidance_scale or 4.5
        generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        autocast_enabled = dit_dtype != torch.float32 and not server_args.disable_autocast

        self.load_model()
        try:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=dit_dtype,
                enabled=autocast_enabled,
            ):
                for i, t in enumerate(timesteps):
                    t_i = t.reshape(1).to(device)
                    with set_forward_context(
                        current_timestep=i, attn_metadata=None, forward_batch=batch
                    ):
                        noise_pred_cond = self.transformer(
                            hidden_states=latents,
                            timestep=t_i,
                            encoder_hidden_states=prompt_embeds,
                            ref_latents=ref_latents,
                            motion_latents=motion_latents,
                            cond_states=cond_states,
                            audio_input=audio_input,
                            motion_frames=[motion_frames, lat_motion_frames],
                        )
                        if guidance_scale > 1 and negative_prompt_embeds is not None:
                            noise_pred_uncond = self.transformer(
                                hidden_states=latents,
                                timestep=t_i,
                                encoder_hidden_states=negative_prompt_embeds,
                                ref_latents=ref_latents,
                                motion_latents=motion_latents,
                                cond_states=cond_states,
                                audio_input=0.0 * audio_input,
                                motion_frames=[motion_frames, lat_motion_frames],
                            )
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_cond - noise_pred_uncond
                            )
                        else:
                            noise_pred = noise_pred_cond

                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        return_dict=False,
                        generator=generator,
                    )[0]
        finally:
            self.offload_model()

        batch.latents = latents
        return batch


class WanS2VStreamR1DenoisingStage(WanS2VDenoisingStage):
    """Stream-R1 S2V block-wise denoising.

    The KV path is enabled only when the loaded transformer exposes the S2V
    cached-attention interfaces. Tests and emergency rollbacks may still set
    ``_s2v_kv_attention_kernel_supported`` to a bool override.
    """

    _s2v_kv_attention_kernel_supported: bool | None = None

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.cache_state = WanS2VStreamR1CacheState.disabled()

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ) -> torch.Tensor:
        request_steps = _coerce_timestep_list(
            _resolve_request_value(
                batch,
                server_args,
                "denoising_steps",
                "denoising_step_list",
                None,
            ),
            "denoising_steps",
        )
        if request_steps is None:
            request_steps = _coerce_timestep_list(
                getattr(server_args.pipeline_config, "denoising_step_list", None),
                "denoising_step_list",
            )

        flow_shift = getattr(server_args.pipeline_config, "flow_shift", 3.0)
        if "flow_shift" in batch.extra and batch.extra["flow_shift"] is not None:
            flow_shift = batch.extra["flow_shift"]
        if request_steps is None:
            self.scheduler.set_timesteps(
                batch.num_inference_steps or 40,
                device=device,
                shift=flow_shift,
            )
            return self.scheduler.timesteps

        train_steps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        self.scheduler.set_timesteps(train_steps, device=device, shift=flow_shift)
        timesteps = torch.tensor(request_steps, dtype=torch.long)
        if torch.any(timesteps < 0) or torch.any(timesteps > train_steps):
            raise ValueError(
                f"denoising_steps must be in [0, {train_steps}], got {request_steps}"
            )

        warp_denoising_step = bool(
            _resolve_request_value(
                batch,
                server_args,
                "warp_denoising_step",
                "warp_denoising_step",
                True,
            )
        )
        if warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (
                    self.scheduler.timesteps.detach().cpu(),
                    torch.tensor([0], dtype=self.scheduler.timesteps.dtype),
                )
            )
            indices = train_steps - timesteps
            if torch.any(indices < 0) or torch.any(
                indices >= scheduler_timesteps.numel()
            ):
                raise ValueError(
                    "warped denoising_steps produced invalid scheduler indices"
                )
            timesteps = scheduler_timesteps[indices]

        timesteps = timesteps.to(device=device)
        self.log_info("Using Stream-R1 S2V timesteps: %s", timesteps)
        return timesteps

    def _train_timesteps(self) -> int:
        scheduler_config = getattr(self.scheduler, "config", None)
        return int(getattr(scheduler_config, "num_train_timesteps", 1000))

    def _resolve_attention_request(
        self,
        batch: Req,
        server_args: ServerArgs,
        latent_frames: int,
    ) -> WanS2VStreamR1AttentionRequest:
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "stream_r1_kv_cache",
                    "stream_r1_kv_cache",
                    False,
                )
            ),
            num_frame_per_block=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "num_frame_per_block",
                    "num_frame_per_block",
                    7,
                )
            ),
            local_attn_size=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "local_attn_size",
                    "local_attn_size",
                    9,
                )
            ),
            sink_size=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "sink_size",
                    "sink_size",
                    3,
                )
            ),
            context_noise=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "context_noise",
                    "context_noise",
                    0,
                )
            ),
        )
        request.validate(
            latent_frames=latent_frames, train_timesteps=self._train_timesteps()
        )
        self._validate_stream_r1_parallel_compatibility(request, batch)
        return request

    def _validate_stream_r1_parallel_compatibility(
        self, request: WanS2VStreamR1AttentionRequest, batch: Req
    ) -> None:
        if not request.stream_r1_kv_cache:
            return
        sp_world_size = _safe_sp_world_size()
        context_parallel_enabled = bool(
            getattr(self.transformer, "use_context_parallel", False)
        )
        sequence_parallel_enabled = bool(
            getattr(batch, "did_sp_shard_latents", False)
            or (sp_world_size > 1 and getattr(batch, "enable_sequence_shard", False))
        )
        if context_parallel_enabled or sequence_parallel_enabled or sp_world_size > 1:
            raise NotImplementedError(
                "Stream-R1 S2V KV cache is incompatible with sequence/context "
                "parallelism in this phase; disable SP/CP or set "
                "stream_r1_kv_cache=false."
            )

    def _configure_transformer_attention(
        self, request: WanS2VStreamR1AttentionRequest
    ) -> None:
        setter = getattr(self.transformer, "set_stream_r1_attention", None)
        if callable(setter):
            setter(
                request.local_attn_size,
                request.sink_size,
                num_frame_per_block=request.num_frame_per_block,
                kv_cache=request.stream_r1_kv_cache,
            )
            return
        setattr(self.transformer, "stream_r1_local_attn_size", request.local_attn_size)
        setattr(self.transformer, "stream_r1_sink_size", request.sink_size)
        setattr(
            self.transformer,
            "stream_r1_num_frame_per_block",
            request.num_frame_per_block,
        )
        setattr(
            self.transformer,
            "stream_r1_kv_cache_requested",
            request.stream_r1_kv_cache,
        )

    @staticmethod
    def _callable_accepts_parameters(
        fn: Any,
        required_parameters: set[str],
    ) -> bool:
        if not callable(fn):
            return False
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return required_parameters.issubset(signature.parameters)

    def _supports_s2v_kv_attention_kernel(self) -> bool:
        override = self._s2v_kv_attention_kernel_supported
        if override is not None:
            return bool(override)

        if not callable(getattr(self.transformer, "set_stream_r1_attention", None)):
            return False
        if not self._callable_accepts_parameters(
            getattr(self.transformer, "forward", None),
            {"kv_cache", "current_start", "cache_start", "stream_r1_mode"},
        ):
            return False

        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return False
        return all(
            self._callable_accepts_parameters(
                getattr(block, "forward", None),
                {
                    "stream_r1_kv_cache",
                    "stream_r1_attention_layout",
                    "cache_start",
                },
            )
            for block in blocks
        )

    def _build_cache_metadata(
        self,
        *,
        request: WanS2VStreamR1AttentionRequest,
        batch_size: int,
        frame_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> WanS2VStreamR1CacheMetadata:
        blocks = getattr(self.transformer, "blocks", None)
        first_block = blocks[0] if blocks is not None and len(blocks) > 0 else None
        arch_config = getattr(
            getattr(self.transformer, "config", None), "arch_config", None
        )
        num_layers = int(
            getattr(
                arch_config,
                "num_layers",
                len(blocks) if blocks is not None else 0,
            )
        )
        if num_layers <= 0:
            raise ValueError("Wan S2V Stream-R1 cache metadata requires num_layers")
        global_heads = int(getattr(self.transformer, "num_attention_heads", 1))
        hidden_size = int(getattr(self.transformer, "hidden_size", global_heads))
        local_num_heads = int(
            getattr(
                first_block,
                "local_num_heads",
                getattr(self.transformer, "local_num_heads", global_heads),
            )
        )
        attention_head_dim = int(
            getattr(
                first_block,
                "dim_head",
                getattr(
                    self.transformer,
                    "attention_head_dim",
                    hidden_size // global_heads,
                ),
            )
        )
        return WanS2VStreamR1CacheMetadata(
            batch_size=batch_size,
            num_layers=num_layers,
            frame_seq_length=frame_seq_length,
            local_num_attention_heads=local_num_heads,
            attention_head_dim=attention_head_dim,
            local_attn_size=request.local_attn_size,
            sink_size=request.sink_size,
            dtype=dtype,
            device=torch.device(device),
        )

    def _prepare_cache_state(
        self,
        *,
        request: WanS2VStreamR1AttentionRequest,
        batch_size: int,
        frame_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> WanS2VStreamR1CacheState:
        if not request.stream_r1_kv_cache:
            self.cache_state = WanS2VStreamR1CacheState.disabled()
            return self.cache_state

        metadata = self._build_cache_metadata(
            request=request,
            batch_size=batch_size,
            frame_seq_length=frame_seq_length,
            dtype=dtype,
            device=device,
        )
        if not self._supports_s2v_kv_attention_kernel():
            self.cache_state = WanS2VStreamR1CacheState.metadata_only(metadata)
        elif not self.cache_state.allocated or self.cache_state.metadata != metadata:
            self.cache_state = WanS2VStreamR1CacheState.allocate(metadata)
        else:
            self.cache_state.reset()

        self.log_info(
            "Prepared Stream-R1 S2V KV cache metadata: layers=%s, tokens=%s, "
            "local_heads=%s, head_dim=%s, sink_tokens=%s, estimated_kv_cache=%.2f MiB",
            metadata.num_layers,
            metadata.cache_tokens,
            metadata.local_num_attention_heads,
            metadata.attention_head_dim,
            metadata.sink_tokens,
            metadata.bytes_per_kv_cache / (1024**2),
        )
        return self.cache_state

    def _guard_cache_runtime(self, cache_state: WanS2VStreamR1CacheState) -> None:
        if not cache_state.enabled:
            return
        if not self._supports_s2v_kv_attention_kernel():
            assert cache_state.metadata is not None
            metadata = cache_state.metadata
            raise NotImplementedError(
                "Stream-R1 S2V KV cache was requested and validated, but the "
                "loaded transformer does not expose the required S2V KV "
                "attention runtime. "
                "Set stream_r1_kv_cache=false to run the current block-wise "
                "Stream-R1 path without KV cache. "
                f"local_attn_size={metadata.local_attn_size}, "
                f"sink_size={metadata.sink_size}, "
                f"cache_tokens={metadata.cache_tokens}."
            )
        if not cache_state.allocated:
            raise RuntimeError("Stream-R1 S2V KV cache state was not allocated")

    @staticmethod
    def _prepend_motion_audio_frames(
        audio_input: torch.Tensor,
        motion_frames: list[int] | tuple[int, int],
    ) -> torch.Tensor:
        return torch.cat(
            [
                audio_input[..., 0:1].repeat(1, 1, 1, int(motion_frames[0])),
                audio_input,
            ],
            dim=-1,
        )

    def _maybe_cache_audio_embeddings(
        self,
        bundle: WanS2VConditionBundle,
        *,
        dtype: torch.dtype | None = None,
        autocast_enabled: bool = False,
    ) -> None:
        if not bool(bundle.audio_metadata.get("cache_audio_embeddings", False)):
            return
        if bundle.audio_emb is not None or bundle.audio_input is None:
            return

        audio_input = self._prepend_motion_audio_frames(
            bundle.audio_input, bundle.motion_frames
        )
        if dtype is None:
            bundle.audio_emb = self.transformer.casual_audio_encoder(audio_input)
            return

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            bundle.audio_emb = self.transformer.casual_audio_encoder(audio_input)

    def _clean_context_refresh(
        self,
        *,
        block_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        block_bundle: WanS2VConditionBundle,
        current_start: int,
        attention_request: WanS2VStreamR1AttentionRequest,
        cache_state: WanS2VStreamR1CacheState,
        dtype: torch.dtype | None = None,
        autocast_enabled: bool = False,
        forward_batch: Req | None = None,
    ) -> None:
        if not cache_state.enabled:
            return None

        self._guard_cache_runtime(cache_state)
        timestep = torch.full(
            (block_latents.shape[0],),
            int(attention_request.context_noise),
            dtype=torch.long,
            device=block_latents.device,
        )

        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            if dtype is None:
                self.transformer(
                    hidden_states=block_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    ref_latents=block_bundle.ref_latents,
                    motion_latents=block_bundle.motion_latents,
                    cond_states=block_bundle.cond_states,
                    audio_input=block_bundle.audio_input,
                    audio_emb=block_bundle.audio_emb,
                    motion_frames=block_bundle.motion_frames,
                    add_last_motion=block_bundle.add_last_motion,
                    drop_motion_frames=block_bundle.drop_motion_frames,
                    kv_cache=cache_state.kv_cache,
                    crossattn_cache=None,
                    current_start=current_start,
                    cache_start=None,
                    stream_r1_mode=True,
                )
                return None

            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=dtype,
                enabled=autocast_enabled,
            ):
                self.transformer(
                    hidden_states=block_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    ref_latents=block_bundle.ref_latents,
                    motion_latents=block_bundle.motion_latents,
                    cond_states=block_bundle.cond_states,
                    audio_input=block_bundle.audio_input,
                    audio_emb=block_bundle.audio_emb,
                    motion_frames=block_bundle.motion_frames,
                    add_last_motion=block_bundle.add_last_motion,
                    drop_motion_frames=block_bundle.drop_motion_frames,
                    kv_cache=cache_state.kv_cache,
                    crossattn_cache=None,
                    current_start=current_start,
                    cache_start=None,
                    stream_r1_mode=True,
                )
        return None

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        sp_world_size = _safe_sp_world_size()
        if sp_world_size > 1:
            # Stream-R1 S2V owns its block-wise loop, so it must opt into the
            # transformer-internal SP path instead of relying on the generic
            # denoising stage to shard latents before this stage runs.
            batch.enable_sequence_shard = True
        latents = batch.latents.to(device=device, dtype=dit_dtype)
        latent_frames = latents.shape[2]
        attention_request = self._resolve_attention_request(
            batch, server_args, latent_frames
        )
        num_frame_per_block = attention_request.num_frame_per_block
        timesteps = self._prepare_timesteps(batch, server_args, device)
        if timesteps.numel() == 0:
            raise ValueError("Stream-R1 S2V requires at least one timestep")

        bundle = build_wan_s2v_condition_bundle(
            batch,
            server_args,
            latents=latents,
            dtype=dit_dtype,
            device=device,
        )
        prompt_embeds = bundle.prompt_embeds
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]

        if (batch.guidance_scale or 1.0) > 1 and _has_negative_prompt_embeds(batch):
            self.log_warning(
                "Stream-R1 S2V no-KV mode does not run CFG in this phase; "
                "negative_prompt_embeds will be ignored."
            )

        patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
        _, _, _, latent_h, latent_w = latents.shape
        frame_seq_length = (latent_h // patch_size[1]) * (latent_w // patch_size[2])
        self._configure_transformer_attention(attention_request)
        cache_state = self._prepare_cache_state(
            request=attention_request,
            batch_size=latents.shape[0],
            frame_seq_length=frame_seq_length,
            dtype=dit_dtype,
            device=device,
        )
        self._guard_cache_runtime(cache_state)
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )

        self.load_model()
        try:
            self._maybe_cache_audio_embeddings(
                bundle,
                dtype=dit_dtype,
                autocast_enabled=autocast_enabled,
            )
            for block_start in range(0, latent_frames, num_frame_per_block):
                block_end = block_start + num_frame_per_block
                block_bundle = bundle.slice(
                    block_start,
                    num_frame_per_block,
                    policy=bundle.control_policy,
                )
                current_latents = latents[:, :, block_start:block_end, :, :]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    t_expand = t_cur.reshape(1).repeat(current_latents.shape[0])
                    with (
                        torch.autocast(
                            device_type=current_platform.device_type,
                            dtype=dit_dtype,
                            enabled=autocast_enabled,
                        ),
                        set_forward_context(
                            current_timestep=i,
                            attn_metadata=None,
                            forward_batch=batch,
                        ),
                    ):
                        noise_pred_bcthw = self.transformer(
                            hidden_states=current_latents,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            ref_latents=block_bundle.ref_latents,
                            motion_latents=block_bundle.motion_latents,
                            cond_states=block_bundle.cond_states,
                            audio_input=block_bundle.audio_input,
                            audio_emb=block_bundle.audio_emb,
                            motion_frames=block_bundle.motion_frames,
                            add_last_motion=block_bundle.add_last_motion,
                            drop_motion_frames=block_bundle.drop_motion_frames,
                            kv_cache=cache_state.kv_cache,
                            crossattn_cache=None,
                            current_start=block_start * frame_seq_length,
                            cache_start=None,
                            stream_r1_mode=True,
                        )
                    noise_pred_btchw = noise_pred_bcthw.permute(0, 2, 1, 3, 4)
                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=noise_pred_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents_btchw.flatten(0, 1),
                        timestep=t_cur.reshape(1),
                        scheduler=self.scheduler,
                    ).unflatten(0, noise_pred_btchw.shape[:2])

                    if i < timesteps.numel() - 1:
                        next_timestep = timesteps[i + 1].reshape(1).to(device=device)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=generator,
                            device=device,
                        )
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep,
                        ).unflatten(0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

                latents[:, :, block_start:block_end, :, :] = current_latents
                self._clean_context_refresh(
                    block_latents=current_latents,
                    prompt_embeds=prompt_embeds,
                    block_bundle=block_bundle,
                    current_start=block_start * frame_seq_length,
                    attention_request=attention_request,
                    cache_state=cache_state,
                    dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                    forward_batch=batch,
                )
        finally:
            self.offload_model()

        batch.latents = latents
        return batch


class WanS2VDenoisingDispatchStage(PipelineStage):
    """Select legacy or Stream-R1 S2V denoising per config/request."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.default_stage = WanS2VDenoisingStage(transformer, scheduler)
        self.stream_r1_stage = WanS2VStreamR1DenoisingStage(transformer, scheduler)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        stream_r1_mode = bool(
            _resolve_request_value(
                batch,
                server_args,
                "stream_r1_mode",
                "stream_r1_mode",
                False,
            )
        )
        if stream_r1_mode:
            return self.stream_r1_stage.forward(batch, server_args)
        return self.default_stage.forward(batch, server_args)
