# SPDX-License-Identifier: Apache-2.0
"""
Audio cross-attention layer for FlashTalk.

Implements SingleStreamMultiAttention that performs cross-attention between
video latent tokens (Q) and audio context tokens (KV), with optional 1D RoPE
for multi-person spatial separation.
"""

import torch
import torch.nn as nn
from einops import rearrange

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention


class RotaryPositionalEmbedding1D(nn.Module):
    """1D Rotary Positional Embedding for multi-person spatial separation."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply 1D RoPE to input tensor.

        Args:
            x: (B, H, S, D) tensor
            positions: (S,) position indices (can be fractional)

        Returns:
            x with RoPE applied, same shape as input
        """
        # positions: (S,) -> (S, dim//2) via outer product with inv_freq
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        # (S, dim//2) -> (S, dim) via [cos, sin] interleave
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()

        # Reshape for broadcasting: (1, 1, S, dim//2)
        cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(0)
        sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(0)

        # Split x into pairs for rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # Apply rotation
        out1 = x1 * cos_freqs - x2 * sin_freqs
        out2 = x1 * sin_freqs + x2 * cos_freqs
        # Interleave back
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out


def normalize_and_scale(
    values: torch.Tensor,
    source_range: tuple[float, float],
    target_range: tuple[float, float],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize values from source_range to target_range."""
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (values - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


class FlashTalkAudioCrossAttention(nn.Module):
    """Audio cross-attention layer for FlashTalk.

    Q comes from video latent tokens (dim), KV from audio context tokens (audio_dim).
    Supports multi-person spatial separation via 1D RoPE when human_num > 1.
    """

    def __init__(
        self,
        dim: int,
        audio_dim: int,
        num_heads: int,
        qk_norm: bool = False,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)

        # Q projection from video latent
        self.q_linear = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        # KV projection from audio context
        self.kv_linear = ColumnParallelLinear(
            audio_dim, dim * 2, bias=True, gather_output=False
        )
        # Output projection
        self.proj = RowParallelLinear(dim, dim, input_is_parallel=True)

        # QK norm (optional)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=eps)
            self.k_norm = RMSNorm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Multi-person RoPE settings
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, class_interval)
        self.rope_h2 = (class_range - class_interval, class_range)
        self.rope_bak = class_range // 2

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

        # Attention
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
        )

    def _single_person_forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        N_t: int,
    ) -> torch.Tensor:
        """Simple cross-attention without multi-person RoPE."""
        # Reshape to per-temporal-step
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape

        # Q from video
        q, _ = self.q_linear(x)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))
        if self.qk_norm:
            q_flat = q.flatten(2)
            q_flat = self.q_norm(q_flat)
            q = q_flat.unflatten(2, (self.local_num_heads, self.head_dim))

        # KV from audio — use "split" layout (all K first, all V second)
        # matching the original checkpoint's view(B, N_a, 2, H, D) convention.
        _, N_a, _ = encoder_hidden_states.shape
        kv, _ = self.kv_linear(encoder_hidden_states)
        kv = kv.unflatten(2, (2, self.local_num_heads, self.head_dim))
        k, v = kv[:, :, 0], kv[:, :, 1]
        if self.qk_norm:
            k_flat = k.flatten(2)
            k_flat = self.k_norm(k_flat)
            k = k_flat.unflatten(2, (self.local_num_heads, self.head_dim))

        # Attention
        out = self.attn(q, k, v)
        out = out.flatten(2)
        out, _ = self.proj(out)

        # Reshape back
        out = rearrange(out, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        return out

    def _multi_person_forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        N_t: int,
        x_ref_attn_map: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention with multi-person 1D RoPE spatial separation."""
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        B, N, C = x.shape

        # Q projection
        q, _ = self.q_linear(x)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))
        if self.qk_norm:
            q_flat = q.flatten(2)
            q_flat = self.q_norm(q_flat)
            q = q_flat.unflatten(2, (self.local_num_heads, self.head_dim))

        # Compute position encoding from attention map
        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min_values = torch.cat([max_values, min_values], dim=2)

        h1_max, h1_min = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        h2_max, h2_min = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(
            x_ref_attn_map[0], (h1_min.item(), h1_max.item()), self.rope_h1
        )
        human2 = normalize_and_scale(
            x_ref_attn_map[1], (h2_min.item(), h2_max.item()), self.rope_h2
        )
        back = torch.full(
            (x_ref_attn_map.size(1),),
            self.rope_bak,
            dtype=human1.dtype,
            device=human1.device,
        )
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[
            range(x_ref_attn_map.size(1)), max_indices
        ]

        # Apply 1D RoPE to Q
        q = rearrange(q, "(B N_t) S H D -> B H (N_t S) D", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) D -> (B N_t) S H D", N_t=N_t)

        # KV from audio — use "split" layout (all K first, all V second)
        # matching the original checkpoint's view(B, N_a, 2, H, D) convention.
        _, N_a, _ = encoder_hidden_states.shape
        kv, _ = self.kv_linear(encoder_hidden_states)
        kv = kv.unflatten(2, (2, self.local_num_heads, self.head_dim))
        k, v = kv[:, :, 0], kv[:, :, 1]
        if self.qk_norm:
            k_flat = k.flatten(2)
            k_flat = self.k_norm(k_flat)
            k = k_flat.unflatten(2, (self.local_num_heads, self.head_dim))

        # Apply 1D RoPE to K
        per_frame = torch.zeros(N_a, dtype=k.dtype, device=k.device)
        half = N_a // 2
        per_frame[:half] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[half:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = per_frame.repeat(N_t)
        k = rearrange(k, "(B N_t) S H D -> B H (N_t S) D", N_t=N_t)
        k = self.rope_1d(k, encoder_pos)
        k = rearrange(k, "B H (N_t S) D -> (B N_t) S H D", N_t=N_t)

        # Attention
        out = self.attn(q, k, v)
        out = out.flatten(2)
        out, _ = self.proj(out)

        out = rearrange(out, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        return out

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape: tuple[int, int, int],
        x_ref_attn_map: torch.Tensor | None = None,
        human_num: int = 1,
    ) -> torch.Tensor:
        """Forward pass for audio cross-attention.

        Args:
            x: (B, N_t * S, C) or (B, local_seq, C) when SP-sharded
            encoder_hidden_states: (B_t, context_tokens, audio_dim) audio context
            shape: (N_t, N_h, N_w) temporal and spatial dimensions (full, pre-SP)
            x_ref_attn_map: optional attention map for multi-person separation
            human_num: number of people in the video

        Returns:
            (B, seq, C) output after audio cross-attention, same shape as input
        """
        N_t = shape[0]
        encoder_hidden_states = encoder_hidden_states.squeeze(0)

        # Align audio temporal dimension with latent temporal dimension
        audio_N_t = encoder_hidden_states.shape[0]
        if audio_N_t != N_t:
            if audio_N_t < N_t:
                # Pad by repeating the last frame
                pad = encoder_hidden_states[-1:].expand(
                    N_t - audio_N_t, -1, -1
                )
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, pad], dim=0
                )
            else:
                # Truncate
                encoder_hidden_states = encoder_hidden_states[:N_t]

        sp_size = get_sp_world_size()

        if sp_size > 1:
            # When SP is active, x is a shard of the flattened sequence. The
            # audio cross-attention groups tokens by temporal step (each step
            # attends to its own audio context). With SP sharding by flat index,
            # each GPU's shard spans partial temporal steps, so the naive
            # rearrange(x, "B (N_t S) C -> ...") would assign the wrong audio
            # context to tokens.
            #
            # Fix: all-gather the full sequence, perform audio cross-attention
            # on the full (correctly-grouped) sequence, then take back our shard.
            sp_rank = get_sp_group().rank_in_group
            local_seq = x.shape[1]
            x_full = sequence_model_parallel_all_gather(x, dim=1)

            if human_num == 1 or x_ref_attn_map is None:
                out_full = self._single_person_forward(
                    x_full, encoder_hidden_states, N_t
                )
            else:
                out_full = self._multi_person_forward(
                    x_full, encoder_hidden_states, N_t, x_ref_attn_map
                )

            # Take this rank's shard back
            start = sp_rank * local_seq
            return out_full[:, start : start + local_seq, :].contiguous()

        if human_num == 1 or x_ref_attn_map is None:
            return self._single_person_forward(x, encoder_hidden_states, N_t)
        else:
            return self._multi_person_forward(
                x, encoder_hidden_states, N_t, x_ref_attn_map
            )
