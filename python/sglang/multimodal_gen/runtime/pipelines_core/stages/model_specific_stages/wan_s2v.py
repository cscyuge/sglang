# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V specific pipeline stages."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
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
    """No-KV Stream-R1 S2V block-wise denoising.

    This implements the Phase 2 block loop and fixed timestep handling. S2V KV
    attention and clean-context cache mutation are intentionally left disabled.
    """

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

    def _validate_block_options(
        self,
        batch: Req,
        server_args: ServerArgs,
        latent_frames: int,
    ) -> int:
        num_frame_per_block = int(
            _resolve_request_value(
                batch,
                server_args,
                "num_frame_per_block",
                "num_frame_per_block",
                7,
            )
        )
        if num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if latent_frames % num_frame_per_block != 0:
            raise ValueError(
                "Stream-R1 S2V requires latent frames to be divisible by "
                f"num_frame_per_block, got latent_frames={latent_frames}, "
                f"num_frame_per_block={num_frame_per_block}"
            )

        use_kv_cache = bool(
            _resolve_request_value(
                batch,
                server_args,
                "stream_r1_kv_cache",
                "stream_r1_kv_cache",
                False,
            )
        )
        if use_kv_cache:
            raise NotImplementedError(
                "Stream-R1 S2V KV attention is not implemented in this phase"
            )
        return num_frame_per_block

    def _clean_context_refresh(
        self,
        *,
        block_latents: torch.Tensor,
        context_noise: int,
    ) -> None:
        # Phase 2 keeps kv_cache=None, so there is no cache state to refresh.
        # The method remains as the explicit lifecycle hook for Phase 3.
        del block_latents, context_noise
        return None

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        latents = batch.latents.to(device=device, dtype=dit_dtype)
        latent_frames = latents.shape[2]
        num_frame_per_block = self._validate_block_options(
            batch, server_args, latent_frames
        )
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
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )
        context_noise = int(
            _resolve_request_value(
                batch,
                server_args,
                "context_noise",
                "context_noise",
                0,
            )
        )

        self.load_model()
        try:
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
                            kv_cache=None,
                            crossattn_cache=None,
                            current_start=block_start * frame_seq_length,
                            cache_start=None,
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
                    context_noise=context_noise,
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
