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
        out_q.shape != q_shape or out_q.dtype != _FP8_DTYPE or out_q.device != x.device
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


@triton.jit
def _fp8_dequant_unpack_qkv_kernel(
    packed_q_ptr,
    scale_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    N: tl.constexpr,
    S_LOCAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_p_hh,
    stride_p_b,
    stride_p_s,
    stride_scale_hh,
    stride_scale_b,
    stride_scale_s,
    stride_scale_g,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    BLOCK_M: tl.constexpr,
):
    m_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < N

    h = m_offsets % H_LOCAL
    tmp = m_offsets // H_LOCAL
    s = tmp % S_LOCAL
    tmp = tmp // S_LOCAL
    ws = tmp % WORLD_SIZE
    b = tmp // WORLD_SIZE

    d_offsets = tl.arange(0, D)
    group_offsets = d_offsets // GROUP_SIZE
    chunk_base = ws * 3 * H_LOCAL

    packed_base = b[:, None] * stride_p_b + s[:, None] * stride_p_s + d_offsets[None, :]
    scale_base = (
        b[:, None] * stride_scale_b
        + s[:, None] * stride_scale_s
        + group_offsets[None, :] * stride_scale_g
    )

    q_hh = chunk_base[:, None] + 3 * h[:, None]
    k_hh = q_hh + 1
    v_hh = q_hh + 2

    q_scale = tl.load(
        scale_ptr + scale_base + q_hh * stride_scale_hh,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(
        scale_ptr + scale_base + k_hh * stride_scale_hh,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    v_scale = tl.load(
        scale_ptr + scale_base + v_hh * stride_scale_hh,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    q_val = (
        tl.load(
            packed_q_ptr + packed_base + q_hh * stride_p_hh,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        * q_scale
    )
    k_val = (
        tl.load(
            packed_q_ptr + packed_base + k_hh * stride_p_hh,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        * k_scale
    )
    v_val = (
        tl.load(
            packed_q_ptr + packed_base + v_hh * stride_p_hh,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        * v_scale
    )

    s_global = ws * S_LOCAL + s
    out_base = (
        b[:, None] * stride_o_b
        + s_global[:, None] * stride_o_s
        + h[:, None] * stride_o_h
        + d_offsets[None, :]
    )
    tl.store(q_ptr + out_base, q_val, mask=m_mask[:, None])
    tl.store(k_ptr + out_base, k_val, mask=m_mask[:, None])
    tl.store(v_ptr + out_base, v_val, mask=m_mask[:, None])


@triton.jit
def _fp8_dequant_unpack_output_kernel(
    packed_q_ptr,
    scale_ptr,
    out_ptr,
    N: tl.constexpr,
    S_LOCAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_p_s,
    stride_p_b,
    stride_p_h,
    stride_scale_s,
    stride_scale_b,
    stride_scale_h,
    stride_scale_g,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    BLOCK_M: tl.constexpr,
):
    m_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < N

    h = m_offsets % H_LOCAL
    tmp = m_offsets // H_LOCAL
    ws = tmp % WORLD_SIZE
    tmp = tmp // WORLD_SIZE
    s_local = tmp % S_LOCAL
    b = tmp // S_LOCAL

    d_offsets = tl.arange(0, D)
    group_offsets = d_offsets // GROUP_SIZE
    s_global = ws * S_LOCAL + s_local

    packed_offsets = (
        s_global[:, None] * stride_p_s
        + b[:, None] * stride_p_b
        + h[:, None] * stride_p_h
        + d_offsets[None, :]
    )
    scale_offsets = (
        s_global[:, None] * stride_scale_s
        + b[:, None] * stride_scale_b
        + h[:, None] * stride_scale_h
        + group_offsets[None, :] * stride_scale_g
    )
    scale = tl.load(scale_ptr + scale_offsets, mask=m_mask[:, None], other=0.0).to(
        tl.float32
    )
    value = (
        tl.load(packed_q_ptr + packed_offsets, mask=m_mask[:, None], other=0.0).to(
            tl.float32
        )
        * scale
    )

    h_global = ws * H_LOCAL + h
    out_offsets = (
        b[:, None] * stride_o_b
        + s_local[:, None] * stride_o_s
        + h_global[:, None] * stride_o_h
        + d_offsets[None, :]
    )
    tl.store(out_ptr + out_offsets, value, mask=m_mask[:, None])


def fused_dequant_unpack_qkv_fp8(
    packed_q: torch.Tensor,
    scale: torch.Tensor,
    B: int,
    S_local: int,
    H_local: int,
    D: int,
    world_size: int,
    *,
    group_size: int,
    dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_shape = (3 * H_local * world_size, B, S_local, D)
    if packed_q.shape != packed_shape:
        raise ValueError(
            f"packed_q must have shape {packed_shape}, got {tuple(packed_q.shape)}"
        )
    if packed_q.dtype != _FP8_DTYPE:
        raise ValueError(f"packed_q must use float8_e4m3fn, got {packed_q.dtype}")
    if not packed_q.is_contiguous():
        raise ValueError("packed_q must be contiguous")
    scale_shape = packed_shape[:-1] + (D // group_size,)
    if scale.shape != scale_shape or scale.dtype != torch.float32:
        raise ValueError(
            f"scale must have shape={scale_shape} dtype=float32, "
            f"got shape={tuple(scale.shape)} dtype={scale.dtype}"
        )
    if D % group_size != 0:
        raise ValueError(f"D={D} must be divisible by group_size={group_size}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"output dtype must be bf16/fp16, got {dtype}")

    out_shape = (B, S_local * world_size, H_local, D)
    if out is None:
        q = torch.empty(out_shape, dtype=dtype, device=packed_q.device)
        k = torch.empty(out_shape, dtype=dtype, device=packed_q.device)
        v = torch.empty(out_shape, dtype=dtype, device=packed_q.device)
    else:
        q, k, v = out
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.shape != out_shape:
                raise ValueError(
                    f"{name} output must have shape {out_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != dtype or tensor.device != packed_q.device:
                raise ValueError(f"{name} output dtype/device must match request")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} output must be contiguous")

    rows = B * world_size * S_local * H_local
    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m),)
    _fp8_dequant_unpack_qkv_kernel[grid](
        packed_q,
        scale,
        q,
        k,
        v,
        rows,
        S_local,
        H_local,
        D,
        world_size,
        group_size,
        packed_q.stride(0),
        packed_q.stride(1),
        packed_q.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        scale.stride(3),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_m,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v


def fused_dequant_unpack_output_fp8(
    packed_q: torch.Tensor,
    scale: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    world_size: int,
    group_size: int,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if packed_q.ndim != 4:
        raise ValueError(
            f"packed_q must have shape [S, B, H, D], got {tuple(packed_q.shape)}"
        )
    if packed_q.dtype != _FP8_DTYPE:
        raise ValueError(f"packed_q must use float8_e4m3fn, got {packed_q.dtype}")
    if not packed_q.is_contiguous():
        raise ValueError("packed_q must be contiguous")
    if packed_q.shape[0] != seq_len or packed_q.shape[1] != batch_size:
        raise ValueError(
            "packed_q leading dimensions must match seq_len/batch_size: "
            f"shape={tuple(packed_q.shape)} seq_len={seq_len} batch_size={batch_size}"
        )
    if seq_len % world_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by world_size")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"output dtype must be bf16/fp16, got {dtype}")

    h_local = packed_q.shape[2]
    D = packed_q.shape[3]
    if D % group_size != 0:
        raise ValueError(f"D={D} must be divisible by group_size={group_size}")
    scale_shape = tuple(packed_q.shape[:-1]) + (D // group_size,)
    if scale.shape != scale_shape or scale.dtype != torch.float32:
        raise ValueError(
            f"scale must have shape={scale_shape} dtype=float32, "
            f"got shape={tuple(scale.shape)} dtype={scale.dtype}"
        )

    s_local = seq_len // world_size
    h_global = h_local * world_size
    out_shape = (batch_size, s_local, h_global, D)
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=packed_q.device)
    elif out.shape != out_shape or out.dtype != dtype or out.device != packed_q.device:
        raise ValueError("out must match fused output shape/device/dtype")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    rows = batch_size * s_local * world_size * h_local
    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m),)
    _fp8_dequant_unpack_output_kernel[grid](
        packed_q,
        scale,
        out,
        rows,
        s_local,
        h_local,
        D,
        world_size,
        group_size,
        packed_q.stride(0),
        packed_q.stride(1),
        packed_q.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        scale.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_m,
        num_warps=4,
        num_stages=2,
    )
    return out
