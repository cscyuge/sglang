# SPDX-License-Identifier: Apache-2.0
"""
AudioProjModel adapter for FlashTalk.

Projects multi-layer Wav2Vec2 features into context tokens for the audio
cross-attention layers in the DiT.

Two projection paths handle different temporal windows:
- First frame: 5-frame audio window (proj1)
- Subsequent frames: 12-frame audio window (proj1_vf)

Both share proj2 and proj3 layers to produce (B, N_t, context_tokens, output_dim).
"""

import torch
import torch.nn as nn


class AudioProjModel(nn.Module):
    """Projects audio features into context tokens for cross-attention.

    Input: (B, num_video_frames, num_audio_layers, audio_feat_dim) from Wav2Vec2
    Output: (B, N_t, context_tokens, output_dim) context for DiT cross-attention

    where N_t is the number of temporal steps after VAE temporal compression.
    """

    def __init__(
        self,
        audio_window_first: int = 5,
        audio_window_vf: int = 12,
        context_tokens: int = 32,
        output_dim: int = 768,
        hidden_dim: int = 512,
        num_audio_layers: int = 12,
        audio_feat_dim: int = 768,
    ):
        super().__init__()
        self.audio_window_first = audio_window_first
        self.audio_window_vf = audio_window_vf
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_audio_layers = num_audio_layers
        self.audio_feat_dim = audio_feat_dim

        # Input dims: window_size * num_layers * feat_dim
        first_input_dim = audio_window_first * num_audio_layers * audio_feat_dim
        vf_input_dim = audio_window_vf * num_audio_layers * audio_feat_dim

        # First-frame projection
        self.proj1 = nn.Linear(first_input_dim, hidden_dim)
        # Subsequent-frame projection
        self.proj1_vf = nn.Linear(vf_input_dim, hidden_dim)

        # Shared layers
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj3 = nn.Linear(hidden_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

    def _project_window(
        self, window_features: torch.Tensor, is_first: bool
    ) -> torch.Tensor:
        """Project a single window of audio features.

        Args:
            window_features: (B, window_size, num_layers, feat_dim)
            is_first: whether this is the first-frame window

        Returns:
            (B, context_tokens, output_dim)
        """
        B = window_features.shape[0]
        # Flatten window: (B, window_size * num_layers * feat_dim)
        x = window_features.reshape(B, -1)

        proj1 = self.proj1 if is_first else self.proj1_vf
        x = self.act(proj1(x))
        x = self.act(self.proj2(x))
        x = self.proj3(x)
        # (B, context_tokens, output_dim)
        x = x.reshape(B, self.context_tokens, self.output_dim)
        x = self.norm(x)
        return x

    def forward(
        self,
        audio_features: torch.Tensor,
        vae_temporal_factor: int = 4,
    ) -> torch.Tensor:
        """Project audio features into context tokens for all temporal steps.

        Args:
            audio_features: (B, num_video_frames, num_layers, feat_dim)
            vae_temporal_factor: VAE temporal compression factor

        Returns:
            (B, N_t, context_tokens, output_dim) audio context for DiT
        """
        B, num_video_frames, num_layers, feat_dim = audio_features.shape

        # Number of latent temporal steps
        # For Wan VAE: first frame + (num_frames - 1) / vae_temporal_factor
        N_t = 1 + (num_video_frames - 1) // vae_temporal_factor

        all_context = []

        # Process first temporal step
        # Window: frames [0, audio_window_first)
        end_first = min(self.audio_window_first, num_video_frames)
        first_window = audio_features[:, :end_first]
        if first_window.shape[1] < self.audio_window_first:
            # Pad if not enough frames
            pad = torch.zeros(
                B,
                self.audio_window_first - first_window.shape[1],
                num_layers,
                feat_dim,
                device=audio_features.device,
                dtype=audio_features.dtype,
            )
            first_window = torch.cat([first_window, pad], dim=1)
        ctx = self._project_window(first_window, is_first=True)
        all_context.append(ctx)

        # Process subsequent temporal steps
        for t in range(1, N_t):
            # Center of the window in video frame space
            center_frame = t * vae_temporal_factor
            half_window = self.audio_window_vf // 2
            start = max(0, center_frame - half_window)
            end = start + self.audio_window_vf
            if end > num_video_frames:
                end = num_video_frames
                start = max(0, end - self.audio_window_vf)

            window = audio_features[:, start:end]
            if window.shape[1] < self.audio_window_vf:
                pad = torch.zeros(
                    B,
                    self.audio_window_vf - window.shape[1],
                    num_layers,
                    feat_dim,
                    device=audio_features.device,
                    dtype=audio_features.dtype,
                )
                window = torch.cat([window, pad], dim=1)
            ctx = self._project_window(window, is_first=False)
            all_context.append(ctx)

        # (B, N_t, context_tokens, output_dim)
        audio_context = torch.stack(all_context, dim=1)
        return audio_context
