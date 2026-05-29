# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V specific pipeline stages."""

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
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
        ref_latents = batch.image_latent
        if ref_latents is None:
            raise ValueError("Wan S2V requires a reference image")
        ref_latents = ref_latents[:, :, :1].to(device=device, dtype=dit_dtype)

        audio_input = batch.extra.get("audio_input")
        if audio_input is None:
            raise ValueError("Wan S2V requires audio_path/audio_tensor")
        audio_input = audio_input.to(device=device, dtype=dit_dtype)
        target_audio_frames = latents.shape[2] * 4
        if audio_input.shape[-1] < target_audio_frames:
            pad_frames = target_audio_frames - audio_input.shape[-1]
            audio_input = torch.nn.functional.pad(audio_input, (0, pad_frames))
        elif audio_input.shape[-1] > target_audio_frames:
            audio_input = audio_input[..., :target_audio_frames]

        batch_size, _, _, latent_h, latent_w = latents.shape
        motion_frames = getattr(server_args.pipeline_config.dit_config.arch_config, "motion_frames", 73)
        lat_motion_frames = (motion_frames + 3) // 4
        motion_latents = torch.zeros(
            batch_size,
            16,
            lat_motion_frames,
            latent_h,
            latent_w,
            dtype=dit_dtype,
            device=device,
        )
        cond_states = torch.zeros_like(latents)

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
