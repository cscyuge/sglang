# SPDX-License-Identifier: Apache-2.0
"""
Wav2Vec2 audio encoder for FlashTalk.

Wraps HuggingFace Wav2Vec2Model to extract multi-layer hidden states
and aligns them to video frame rate via linear interpolation.

Replicates the original FlashTalk Wav2Vec2 forward pass:
1. CNN feature extraction at native audio framerate (~50fps for 16kHz)
2. Linear interpolation of CNN features to video framerate (e.g. 25fps)
3. Feature projection + encoder transformer at video framerate
4. Collect all encoder hidden states at video framerate

This ordering is critical: the encoder's self-attention operates at
video temporal resolution, matching the original FlashTalk behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_wan_s2v_hidden_states(
    hidden_states: torch.Tensor,
    *,
    audio_num_samples: int,
    sample_rate: int,
    num_video_frames: int,
    native_fps: float = 50.0,
    intermediate_fps: float = 30.0,
) -> torch.Tensor:
    """Match Wan S2V's post-Wav2Vec 50 Hz -> 30 Hz -> video FPS path."""
    if hidden_states.ndim != 4:
        raise ValueError(
            "hidden_states must have shape [B, T, L, D], got "
            f"{tuple(hidden_states.shape)}"
        )
    if audio_num_samples <= 0 or sample_rate <= 0 or num_video_frames <= 0:
        raise ValueError("audio length, sample rate, and video frames must be positive")

    batch, native_frames, layers, dim = hidden_states.shape
    intermediate_frames = max(
        1, int(native_frames / float(native_fps) * float(intermediate_fps))
    )
    features = hidden_states.permute(0, 2, 3, 1).reshape(
        batch, layers * dim, native_frames
    )
    features = F.interpolate(
        features,
        size=intermediate_frames,
        mode="linear",
        align_corners=True,
    )
    features = features.reshape(
        batch, layers, dim, intermediate_frames
    ).permute(0, 3, 1, 2)

    duration = audio_num_samples / float(sample_rate)
    video_fps = num_video_frames / duration
    indices = torch.round(
        torch.arange(num_video_frames, device=features.device, dtype=torch.float32)
        * (float(intermediate_fps) / video_fps)
    ).to(dtype=torch.long)
    valid = indices < intermediate_frames
    gathered = features.index_select(1, indices.clamp(max=intermediate_frames - 1))
    return gathered * valid.view(1, -1, 1, 1).to(dtype=gathered.dtype)


class Wav2Vec2AudioEncoder(nn.Module):
    """Wav2Vec2-based audio encoder that extracts all hidden layer features.

    Input: raw audio waveform (B, L) at 16kHz sample rate.
    Output: (B, seq_len, num_layers, hidden_size) multi-layer audio features.
    """

    def __init__(
        self,
        model_path: str,
        num_hidden_layers: int = 12,
        target_fps: int = 25,
        sample_rate: int = 16000,
        freeze_feature_extractor: bool = True,
        include_embedding_layer: bool = False,
        wan_s2v_post_encoder_resample: bool = False,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path)
        self.wav2vec2.config.output_hidden_states = True
        self.num_hidden_layers = num_hidden_layers
        self.target_fps = target_fps
        self.sample_rate = sample_rate
        self.include_embedding_layer = include_embedding_layer
        self.wan_s2v_post_encoder_resample = wan_s2v_post_encoder_resample

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

    def linear_interpolation(
        self, features: torch.Tensor, output_len: int
    ) -> torch.Tensor:
        """Interpolate audio features to match target frame count.

        Args:
            features: (B, T_audio, D) audio features
            output_len: target temporal length (number of video frames)

        Returns:
            Interpolated features of shape (B, output_len, D)
        """
        if features.shape[1] == output_len:
            return features
        # (B, T, D) -> (B, D, T) for F.interpolate
        features = features.permute(0, 2, 1)
        features = F.interpolate(
            features, size=output_len, mode="linear", align_corners=True
        )
        # (B, D, T) -> (B, T, D)
        return features.permute(0, 2, 1)

    @torch.no_grad()
    def forward(
        self, audio_waveform: torch.Tensor, num_video_frames: int | None = None
    ) -> torch.Tensor:
        """Encode audio waveform into multi-layer features.

        Replicates the original FlashTalk Wav2Vec2 forward pass:
        1. CNN feature_extractor at native framerate
        2. Interpolate CNN features to num_video_frames (video fps)
        3. feature_projection + encoder at video framerate
        4. Collect hidden_states[1:num_hidden_layers+1]

        Args:
            audio_waveform: (B, L) raw audio at 16kHz
            num_video_frames: if provided, interpolate CNN features to this
                many frames BEFORE the encoder (matching original FlashTalk)

        Returns:
            (B, seq_len, num_layers, hidden_size) audio features
        """
        wav2vec = self.wav2vec2

        if self.wan_s2v_post_encoder_resample:
            outputs = wav2vec(
                audio_waveform,
                output_hidden_states=True,
                return_dict=True,
            )
            if self.include_embedding_layer:
                selected = outputs.hidden_states[: self.num_hidden_layers]
            else:
                selected = outputs.hidden_states[1 : self.num_hidden_layers + 1]
            all_layer_features = torch.stack(selected, dim=2)
            if num_video_frames is None:
                return all_layer_features
            return resample_wan_s2v_hidden_states(
                all_layer_features,
                audio_num_samples=int(audio_waveform.shape[-1]),
                sample_rate=self.sample_rate,
                num_video_frames=num_video_frames,
            )

        # Step 1: CNN feature extraction at native audio framerate
        extract_features = wav2vec.feature_extractor(audio_waveform)
        extract_features = extract_features.transpose(1, 2)  # (B, T_native, D)

        # Step 2: Interpolate to video framerate BEFORE encoder
        if num_video_frames is not None:
            extract_features = self.linear_interpolation(
                extract_features, num_video_frames
            )

        # Step 3: Feature projection (layer norm + projection + dropout)
        hidden_states, extract_features = wav2vec.feature_projection(extract_features)
        hidden_states = wav2vec._mask_hidden_states(hidden_states)

        # Step 4: Encoder transformer at video framerate
        encoder_outputs = wav2vec.encoder(
            hidden_states,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.include_embedding_layer:
            all_hidden = encoder_outputs.hidden_states[: self.num_hidden_layers]
        else:
            # Skip embedding layer (index 0), take encoder layers 1..num_hidden_layers.
            all_hidden = encoder_outputs.hidden_states[1 : self.num_hidden_layers + 1]

        # Stack to (B, T, num_layers, D)
        all_layer_features = torch.stack(all_hidden, dim=2)

        return all_layer_features
