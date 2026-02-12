# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding stage for FlashTalk pipeline.

Handles loading audio from WAV files, encoding via Wav2Vec2,
and projecting through AudioProjModel into context tokens for the DiT.
"""

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AudioEncodingStage(PipelineStage):
    """Stage for encoding audio input into context tokens for FlashTalk.

    Supports two input modes:
    - audio_path: path to a WAV file
    - audio_tensor: pre-loaded audio waveform tensor

    Output is stored in batch.extra["audio_context"] with shape
    (B, N_t, context_tokens, output_dim).
    """

    def __init__(
        self,
        audio_encoder=None,
        audio_proj=None,
        wav2vec_feature_extractor=None,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_proj = audio_proj
        self.wav2vec_feature_extractor = wav2vec_feature_extractor

    def load_model(self):
        if self.server_args.audio_encoder_cpu_offload:
            device = get_local_torch_device()
            self._move_to_device(device)

    def offload_model(self):
        if self.server_args.audio_encoder_cpu_offload:
            self._move_to_device("cpu")

    def _move_to_device(self, device):
        for component in [self.audio_encoder, self.audio_proj]:
            if component is not None and hasattr(component, "to"):
                component.to(device)

    def _load_audio(self, audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """Load audio from file path."""
        import soundfile as sf

        speech_array, sr = sf.read(audio_path)
        if sr != sample_rate:
            import librosa

            speech_array = librosa.resample(
                speech_array, orig_sr=sr, target_sr=sample_rate
            )
        return speech_array

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """Encode audio into context tokens.

        Reads audio from batch.extra["audio_path"] or batch.extra["audio_tensor"],
        processes through Wav2Vec2 and AudioProjModel, and stores result in
        batch.extra["audio_context"].
        """
        audio_path = batch.extra.get("audio_path")
        audio_tensor = batch.extra.get("audio_tensor")

        if audio_path is None and audio_tensor is None:
            # No audio input, skip
            return batch

        cuda_device = get_local_torch_device()
        self.load_model()

        sample_rate = 16000
        fps = batch.fps if hasattr(batch, "fps") and batch.fps else 25

        # Load audio waveform
        if audio_tensor is not None:
            speech_array = audio_tensor
        else:
            speech_array = self._load_audio(audio_path, sample_rate)

        # Compute expected video frames from audio duration
        if isinstance(speech_array, np.ndarray):
            audio_duration = len(speech_array) / sample_rate
        else:
            audio_duration = speech_array.shape[-1] / sample_rate
        num_video_frames = int(audio_duration * fps)

        # Process through Wav2Vec2 feature extractor
        if self.wav2vec_feature_extractor is not None:
            audio_feature = np.squeeze(
                self.wav2vec_feature_extractor(
                    speech_array, sampling_rate=sample_rate
                ).input_values
            )
            audio_feature = (
                torch.from_numpy(audio_feature).float().to(device=cuda_device)
            )
            audio_feature = audio_feature.unsqueeze(0)
        else:
            if isinstance(speech_array, np.ndarray):
                audio_feature = (
                    torch.from_numpy(speech_array).float().to(device=cuda_device)
                )
            else:
                audio_feature = speech_array.float().to(device=cuda_device)
            if audio_feature.dim() == 1:
                audio_feature = audio_feature.unsqueeze(0)

        # Encode with Wav2Vec2
        with set_forward_context(current_timestep=0, attn_metadata=None):
            audio_features = self.audio_encoder(
                audio_feature, num_video_frames=num_video_frames
            )
        # audio_features: (B, num_video_frames, num_layers, feat_dim)

        # Project through AudioProjModel
        vae_temporal_factor = (
            server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
            if hasattr(server_args.pipeline_config, "vae_config")
            else 4
        )
        audio_context = self.audio_proj(
            audio_features, vae_temporal_factor=vae_temporal_factor
        )
        # audio_context: (B, N_t, context_tokens, output_dim)

        batch.extra["audio_context"] = audio_context

        self.offload_model()
        return batch

    def verify_input(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        """Verify audio encoding stage inputs."""
        result = VerificationResult()
        has_audio = (
            batch.extra.get("audio_path") is not None
            or batch.extra.get("audio_tensor") is not None
        )
        result.add_check("audio_input", has_audio, V.is_true)
        return result

    def verify_output(
        self, batch: Req, server_args: ServerArgs
    ) -> VerificationResult:
        """Verify audio encoding stage outputs."""
        result = VerificationResult()
        return result
