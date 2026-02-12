# SPDX-License-Identifier: Apache-2.0
"""
Wav2Vec2 audio encoder for FlashTalk.

Wraps HuggingFace Wav2Vec2Model to extract multi-layer hidden states
and aligns them to video frame rate via linear interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path)
        self.wav2vec2.config.output_hidden_states = True
        self.num_hidden_layers = num_hidden_layers
        self.target_fps = target_fps
        self.sample_rate = sample_rate

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

        Args:
            audio_waveform: (B, L) raw audio at 16kHz
            num_video_frames: if provided, interpolate to this many frames

        Returns:
            (B, seq_len, num_layers, hidden_size) audio features
        """
        outputs = self.wav2vec2(audio_waveform, output_hidden_states=True)
        # hidden_states is a tuple of (num_layers + 1) tensors, each (B, T, D)
        # Skip the embedding layer output (index 0), take layers 1..num_hidden_layers
        hidden_states = outputs.hidden_states[1 : self.num_hidden_layers + 1]

        # Stack to (B, T, num_layers, D)
        all_layer_features = torch.stack(hidden_states, dim=2)

        if num_video_frames is not None:
            B, T, L, D = all_layer_features.shape
            # Reshape to (B, T, L*D) for interpolation, then reshape back
            features_flat = all_layer_features.reshape(B, T, L * D)
            features_flat = self.linear_interpolation(features_flat, num_video_frames)
            all_layer_features = features_flat.reshape(
                B, num_video_frames, L, D
            )

        return all_layer_features
