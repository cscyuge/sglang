"""Triton helpers for Wan S2V rotary embeddings."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _wan_s2v_apply_rope_kernel(
    x_ptr,
    freqs_ptr,
    out_ptr,
    total_pairs: tl.constexpr,
    half_dim: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_s: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_d: tl.constexpr,
    f_stride_b: tl.constexpr,
    f_stride_s: tl.constexpr,
    f_stride_h: tl.constexpr,
    f_stride_p: tl.constexpr,
    f_stride_c: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pair_offsets < total_pairs

    pair_idx = pair_offsets % half_dim
    head_idx = (pair_offsets // half_dim) % heads
    seq_idx = (pair_offsets // (half_dim * heads)) % seq_len
    batch_idx = pair_offsets // (half_dim * heads * seq_len)

    x_base = (
        batch_idx * x_stride_b
        + seq_idx * x_stride_s
        + head_idx * x_stride_h
        + (pair_idx * 2) * x_stride_d
    )
    f_base = (
        batch_idx * f_stride_b
        + seq_idx * f_stride_s
        + head_idx * f_stride_h
        + pair_idx * f_stride_p
    )
    out_base = (
        batch_idx * out_stride_b
        + seq_idx * out_stride_s
        + head_idx * out_stride_h
        + (pair_idx * 2) * out_stride_d
    )

    x0 = tl.load(x_ptr + x_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + x_stride_d, mask=mask, other=0.0).to(tl.float32)
    freq_r = tl.load(freqs_ptr + f_base, mask=mask, other=1.0).to(tl.float32)
    freq_i = tl.load(freqs_ptr + f_base + f_stride_c, mask=mask, other=0.0).to(
        tl.float32
    )

    out0 = x0 * freq_r - x1 * freq_i
    out1 = x1 * freq_r + x0 * freq_i

    tl.store(out_ptr + out_base, out0, mask=mask)
    tl.store(out_ptr + out_base + out_stride_d, out1, mask=mask)


def apply_wan_s2v_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply precomputed complex RoPE frequencies to ``x``.

    Args:
        x: Tensor with shape [B, S, H, D].
        freqs: Complex tensor with shape [B, S, H, D / 2].
    """

    if x.dim() != 4:
        raise ValueError(f"Wan S2V RoPE expects [B, S, H, D], got {tuple(x.shape)}")
    if freqs.dim() != 4:
        raise ValueError(
            f"Wan S2V RoPE freqs expect [B, S, H, D/2], got {tuple(freqs.shape)}"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Wan S2V RoPE head dimension must be even, got {x.shape[-1]}")
    if freqs.shape[0] != x.shape[0] or freqs.shape[2] != x.shape[2]:
        raise ValueError(
            "Wan S2V RoPE freqs must match batch/head dimensions: "
            f"x={tuple(x.shape)} freqs={tuple(freqs.shape)}"
        )
    if freqs.shape[1] < x.shape[1] or freqs.shape[3] != x.shape[-1] // 2:
        raise ValueError(
            "Wan S2V RoPE freqs must cover sequence and half head dimensions: "
            f"x={tuple(x.shape)} freqs={tuple(freqs.shape)}"
        )
    if not freqs.is_complex():
        raise ValueError(f"Wan S2V RoPE freqs must be complex, got {freqs.dtype}")

    freqs = freqs[:, : x.shape[1]]
    freqs_real = torch.view_as_real(freqs)
    out = torch.empty_like(x)
    batch, seq_len, heads, dim = x.shape
    half_dim = dim // 2
    total_pairs = batch * seq_len * heads * half_dim
    grid = (triton.cdiv(total_pairs, 256),)
    _wan_s2v_apply_rope_kernel[grid](
        x,
        freqs_real,
        out,
        total_pairs,
        half_dim,
        heads,
        seq_len,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        freqs_real.stride(0),
        freqs_real.stride(1),
        freqs_real.stride(2),
        freqs_real.stride(3),
        freqs_real.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out
