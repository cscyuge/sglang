"""Blockwise FP8 helpers for USP activation communication."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = torch.finfo(_FP8_DTYPE).min


@triton.jit
def _blockwise_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group_id = tl.program_id(1)
    col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    mask = row_offsets[:, None] < M
    x_offsets = row_offsets[:, None] * N + col_offsets[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(absmax / FP8_MAX, 1.0e-10)
    q = tl.clamp(x / scale[:, None], FP8_MIN, FP8_MAX).to(tl.float8e4nv)

    tl.store(q_ptr + x_offsets, q, mask=mask)
    tl.store(
        scale_ptr + row_offsets * (N // GROUP_SIZE) + group_id,
        scale,
        mask=row_offsets < M,
    )


@triton.jit
def _blockwise_dequant_fp8_kernel(
    q_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group_id = tl.program_id(1)
    col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    mask = row_offsets[:, None] < M
    offsets = row_offsets[:, None] * N + col_offsets[None, :]
    scale = tl.load(
        scale_ptr + row_offsets * (N // GROUP_SIZE) + group_id,
        mask=row_offsets < M,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = q * scale[:, None]
    tl.store(out_ptr + offsets, out, mask=mask)


def _flatten_contiguous_rows(x: torch.Tensor) -> tuple[int, int]:
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous tensor, got stride={tuple(x.stride())}")
    if x.ndim == 0:
        raise ValueError("expected at least one tensor dimension")
    return x.numel() // x.shape[-1], x.shape[-1]


def _block_m_for_rows(rows: int) -> int:
    if rows <= 2048:
        return 4
    if rows <= 16384:
        return 8
    return 16


def blockwise_quant_fp8(
    x: torch.Tensor,
    *,
    group_size: int,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"FP8 comm quant expects bf16/fp16 input, got {x.dtype}")
    rows, cols = _flatten_contiguous_rows(x)
    if cols % group_size != 0:
        raise ValueError(
            f"last dimension {cols} must be divisible by group_size={group_size}"
        )

    q_shape = tuple(x.shape)
    scale_shape = tuple(x.shape[:-1]) + (cols // group_size,)
    if out_q is None:
        out_q = torch.empty(q_shape, dtype=_FP8_DTYPE, device=x.device)
    elif (
        out_q.shape != q_shape
        or out_q.dtype != _FP8_DTYPE
        or out_q.device != x.device
    ):
        raise ValueError("out_q must match input shape/device and use float8_e4m3fn")
    if out_scale is None:
        out_scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    elif (
        out_scale.shape != scale_shape
        or out_scale.dtype != torch.float32
        or out_scale.device != x.device
    ):
        raise ValueError("out_scale must match scale shape/device and use float32")

    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m), cols // group_size)
    _blockwise_quant_fp8_kernel[grid](
        x,
        out_q,
        out_scale,
        rows,
        cols,
        group_size,
        block_m,
        _FP8_MAX,
        _FP8_MIN,
        num_warps=4,
        num_stages=2,
    )
    return out_q, out_scale


def blockwise_dequant_fp8(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.dtype != _FP8_DTYPE:
        raise ValueError(f"FP8 comm dequant expects float8_e4m3fn input, got {q.dtype}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"FP8 comm dequant output must be bf16/fp16, got {dtype}")
    rows, cols = _flatten_contiguous_rows(q)
    scale_shape = tuple(q.shape[:-1]) + (cols // group_size,)
    if scale.shape != scale_shape or scale.dtype != torch.float32:
        raise ValueError(
            f"scale must have shape={scale_shape} dtype=float32, "
            f"got shape={tuple(scale.shape)} dtype={scale.dtype}"
        )
    if out is None:
        out = torch.empty(q.shape, dtype=dtype, device=q.device)
    elif out.shape != q.shape or out.dtype != dtype or out.device != q.device:
        raise ValueError("out must match q shape/device and requested dtype")

    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m), cols // group_size)
    _blockwise_dequant_fp8_kernel[grid](
        q,
        scale,
        out,
        rows,
        cols,
        group_size,
        block_m,
        num_warps=4,
        num_stages=2,
    )
    return out
