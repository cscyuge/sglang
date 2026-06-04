# SPDX-License-Identifier: Apache-2.0
"""Lightweight Stream-R1 helpers for Wan S2V attention."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WanS2VStreamR1AttentionLayout:
    noisy_seq_len: int
    total_seq_len: int
    frame_seq_length: int
    num_frame_per_block: int
    local_attn_size: int
    sink_size: int
    current_start: int = 0

    def __post_init__(self) -> None:
        if self.noisy_seq_len <= 0:
            raise ValueError("noisy_seq_len must be positive")
        if self.total_seq_len < self.noisy_seq_len:
            raise ValueError("total_seq_len must be at least noisy_seq_len")
        if self.frame_seq_length <= 0:
            raise ValueError("frame_seq_length must be positive")
        if self.noisy_seq_len % self.frame_seq_length != 0:
            raise ValueError("noisy_seq_len must be divisible by frame_seq_length")
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.current_start < 0:
            raise ValueError("current_start must be non-negative")
        if self.current_start % self.frame_seq_length != 0:
            raise ValueError("current_start must be frame-aligned")

    @property
    def condition_seq_len(self) -> int:
        return self.total_seq_len - self.noisy_seq_len

    @property
    def block_tokens(self) -> int:
        return self.num_frame_per_block * self.frame_seq_length

    @property
    def local_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    def build_no_kv_attention_mask(self, device: torch.device) -> torch.Tensor:
        """Build the Stream-R1 no-KV local/sink mask for mixed S2V tokens.

        Noisy latent queries are restricted to sink noisy tokens, their local
        block window, and all current condition tokens. Condition queries keep
        dense attention so reference/motion tokens retain legacy semantics.
        """

        q_idx = torch.arange(self.total_seq_len, device=device).view(-1, 1)
        kv_idx = torch.arange(self.total_seq_len, device=device).view(1, -1)
        noisy_q = q_idx < self.noisy_seq_len
        noisy_kv = kv_idx < self.noisy_seq_len
        condition_kv = kv_idx >= self.noisy_seq_len

        q_abs = self.current_start + q_idx
        kv_abs = self.current_start + kv_idx
        block_end = (
            torch.div(q_abs, self.block_tokens, rounding_mode="floor") + 1
        ) * self.block_tokens
        local_start = block_end - self.local_tokens

        sink_visible = noisy_kv & (kv_abs < self.sink_tokens)
        local_visible = noisy_kv & (kv_abs >= local_start) & (kv_abs < block_end)
        noisy_query_visible = condition_kv | sink_visible | local_visible

        return torch.where(
            noisy_q,
            noisy_query_visible,
            torch.ones_like(noisy_query_visible),
        ).unsqueeze(0)
