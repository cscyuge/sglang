# SPDX-License-Identifier: Apache-2.0
"""
AudioProjModel adapter for FlashTalk.

Projects multi-layer Wav2Vec2 features into context tokens for the audio
cross-attention layers in the DiT.

Replicates the original FlashTalk audio windowing strategy:
1. Pre-window at VIDEO frame resolution: each frame gets a centered window of
   `audio_window_first` (default 5) neighbouring audio features.
2. First latent temporal step uses the first frame's full 5-feature window.
3. Subsequent latent temporal steps combine 4 video sub-frames' windowed
   features into `audio_window_vf` (default 8 = audio_window + vae_scale - 1)
   features using the original's specific sub-sampling pattern.

Both paths share proj2 and proj3 layers to produce (B, N_t, context_tokens, output_dim).
"""

import torch
import torch.nn as nn
from torch.amp import autocast


class AudioProjModel(nn.Module):
    """Projects audio features into context tokens for cross-attention.

    Input: (B, num_video_frames, num_audio_layers, audio_feat_dim) from Wav2Vec2
    Output: (B, N_t, context_tokens, output_dim) context for DiT cross-attention

    where N_t is the number of temporal steps after VAE temporal compression.
    """

    def __init__(
        self,
        audio_window_first: int = 5,
        audio_window_vf: int = 8,
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

    def forward(
        self,
        audio_features: torch.Tensor,
        vae_temporal_factor: int = 4,
    ) -> torch.Tensor:
        """Project audio features into context tokens for all temporal steps.

        Replicates the original FlashTalk audio windowing strategy:
        1. Pre-window each video frame with a centered 5-frame window
        2. First latent step: use first frame's 5-feature window via proj1
        3. Subsequent latent steps: sub-sample 4 video sub-frames' windows into
           8 features (3 from first sub-frame + 2 from middle + 3 from last)
           via proj1_vf
        4. Shared proj2 + proj3 + LayerNorm to produce context tokens

        Args:
            audio_features: (B, num_video_frames, num_layers, feat_dim)
            vae_temporal_factor: VAE temporal compression factor (default 4)

        Returns:
            (B, N_t, context_tokens, output_dim) audio context for DiT
        """
        B, num_video_frames, num_layers, feat_dim = audio_features.shape
        device = audio_features.device

        # Number of latent temporal steps
        N_t = 1 + (num_video_frames - 1) // vae_temporal_factor

        # -----------------------------------------------------------
        # Step 1: Pre-window at VIDEO frame resolution
        # Original: indices = [-2, -1, 0, 1, 2] for audio_window=5
        # -----------------------------------------------------------
        half_w = self.audio_window_first // 2  # 2
        window_offsets = torch.arange(
            -half_w, half_w + 1, device=device
        )  # [-2, -1, 0, 1, 2]
        frame_indices = torch.arange(num_video_frames, device=device)
        # (num_frames, window_size): each frame's window indices, clamped
        windowed_indices = (
            frame_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
        )
        windowed_indices = windowed_indices.clamp(0, num_video_frames - 1)
        # (B, num_frames, window_size, num_layers, feat_dim)
        windowed = audio_features[:, windowed_indices]

        # -----------------------------------------------------------
        # Step 2: First latent temporal step (t=0)
        # Use first frame's full 5-feature window
        # -----------------------------------------------------------
        # (B, 1, 5, num_layers, feat_dim) → (B*1, 5*num_layers*feat_dim)
        first_frame = windowed[:, :1]
        first_flat = first_frame.reshape(B, -1)
        first_out = torch.relu(self.proj1(first_flat))  # (B, hidden_dim)
        first_out = first_out.unsqueeze(1)  # (B, 1, hidden_dim)

        # -----------------------------------------------------------
        # Step 3: Subsequent latent temporal steps (t=1..N_t-1)
        # Group the remaining video frames by vae_temporal_factor (4),
        # then sub-sample the pre-windowed features.
        # -----------------------------------------------------------
        subsequent_windowed = windowed[:, 1:]  # (B, num_frames-1, 5, L, D)
        n_subsequent_steps = N_t - 1

        if n_subsequent_steps > 0:
            # Reshape to (B, N_t-1, vae_scale, window, layers, dim)
            grouped = subsequent_windowed.reshape(
                B, n_subsequent_steps, vae_temporal_factor,
                self.audio_window_first, num_layers, feat_dim,
            )
            mid = self.audio_window_first // 2  # 2

            # First sub-frame (index 0): take window[:mid+1] = [0,1,2] → 3
            first_sub = grouped[:, :, :1, : mid + 1]
            first_sub = first_sub.reshape(
                B, n_subsequent_steps, 1 * (mid + 1), num_layers, feat_dim
            )

            # Middle sub-frames (indices 1..vae_scale-2): take center only
            middle_sub = grouped[:, :, 1:-1, mid : mid + 1]
            n_middle = vae_temporal_factor - 2
            middle_sub = middle_sub.reshape(
                B, n_subsequent_steps, n_middle, num_layers, feat_dim
            )

            # Last sub-frame (index vae_scale-1): take window[mid:] = [2,3,4] → 3
            last_sub = grouped[:, :, -1:, mid:]
            last_sub = last_sub.reshape(
                B, n_subsequent_steps,
                1 * (self.audio_window_first - mid), num_layers, feat_dim,
            )

            # Concatenate: (B, N_t-1, 3+n_middle+3, layers, dim)
            combined = torch.cat([first_sub, middle_sub, last_sub], dim=2)
            # (B*(N_t-1), audio_window_vf * layers * dim)
            combined_flat = combined.reshape(B * n_subsequent_steps, -1)
            subsequent_out = torch.relu(self.proj1_vf(combined_flat))
            subsequent_out = subsequent_out.reshape(
                B, n_subsequent_steps, -1
            )  # (B, N_t-1, hidden_dim)
        else:
            subsequent_out = first_out[:, :0]  # empty tensor with right shape

        # -----------------------------------------------------------
        # Step 4: Shared projection
        # -----------------------------------------------------------
        # (B, N_t, hidden_dim)
        all_out = torch.cat([first_out, subsequent_out], dim=1)
        BN = B * N_t
        all_out = all_out.reshape(BN, -1)

        all_out = torch.relu(self.proj2(all_out))
        context_tokens = self.proj3(all_out)
        context_tokens = context_tokens.reshape(
            BN, self.context_tokens, self.output_dim
        )

        # Normalize in fp32 (matching original's amp.autocast(dtype=float32))
        with autocast("cuda", enabled=False):
            context_tokens = self.norm(context_tokens.float())

        context_tokens = context_tokens.reshape(
            B, N_t, self.context_tokens, self.output_dim
        )
        return context_tokens

    def forward_prewindowed(
        self,
        windowed_features: torch.Tensor,
        vae_temporal_factor: int = 4,
    ) -> torch.Tensor:
        """Project pre-windowed audio features into context tokens.

        Accepts features that have already been windowed externally (matching
        the original FlashTalk's ``get_audio_embedding`` which windows on the
        full wav2vec2 output before slicing per chunk).

        Args:
            windowed_features: (B, num_video_frames, window_size, num_layers, feat_dim)
                Pre-windowed audio features for this chunk.
            vae_temporal_factor: VAE temporal compression factor (default 4).

        Returns:
            (B, N_t, context_tokens, output_dim) audio context for DiT.
        """
        B, num_video_frames, window_size, num_layers, feat_dim = (
            windowed_features.shape
        )
        N_t = 1 + (num_video_frames - 1) // vae_temporal_factor

        # --- First latent temporal step (t=0) ---
        # (B, 1, window, layers, dim) → flatten → proj1
        first_frame = windowed_features[:, :1]  # (B, 1, 5, L, D)
        first_flat = first_frame.reshape(B, -1)  # (B, 5*L*D)
        first_out = torch.relu(self.proj1(first_flat))  # (B, hidden)
        first_out = first_out.unsqueeze(1)  # (B, 1, hidden)

        # --- Subsequent latent temporal steps (t=1..N_t-1) ---
        n_subsequent_steps = N_t - 1

        if n_subsequent_steps > 0:
            subsequent = windowed_features[:, 1:]  # (B, 32, 5, L, D)
            # Group by vae_temporal_factor: (B, N_t-1, vae_scale, window, L, D)
            grouped = subsequent.reshape(
                B,
                n_subsequent_steps,
                vae_temporal_factor,
                window_size,
                num_layers,
                feat_dim,
            )
            mid = window_size // 2  # 2

            # Sub-sample matching the original FlashTalk pattern:
            # first sub-frame: window[:mid+1], middle sub-frames: center,
            # last sub-frame: window[mid:]
            first_sub = grouped[:, :, :1, : mid + 1].reshape(
                B, n_subsequent_steps, mid + 1, num_layers, feat_dim
            )
            middle_sub = grouped[:, :, 1:-1, mid : mid + 1].reshape(
                B, n_subsequent_steps, vae_temporal_factor - 2, num_layers, feat_dim
            )
            last_sub = grouped[:, :, -1:, mid:].reshape(
                B, n_subsequent_steps, window_size - mid, num_layers, feat_dim
            )

            combined = torch.cat([first_sub, middle_sub, last_sub], dim=2)
            combined_flat = combined.reshape(B * n_subsequent_steps, -1)
            subsequent_out = torch.relu(self.proj1_vf(combined_flat))
            subsequent_out = subsequent_out.reshape(B, n_subsequent_steps, -1)
        else:
            subsequent_out = first_out[:, :0]

        # --- Shared projection ---
        all_out = torch.cat([first_out, subsequent_out], dim=1)  # (B, N_t, hidden)
        BN = B * N_t
        all_out = all_out.reshape(BN, -1)

        all_out = torch.relu(self.proj2(all_out))
        context_tokens = self.proj3(all_out)
        context_tokens = context_tokens.reshape(BN, self.context_tokens, self.output_dim)

        with autocast("cuda", enabled=False):
            context_tokens = self.norm(context_tokens.float())

        return context_tokens.reshape(B, N_t, self.context_tokens, self.output_dim)
