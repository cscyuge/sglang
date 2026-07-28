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
def _blockwise_quant_fp8_rowpack_kernel(
    x_ptr,
    q_out_ptr,
    scale_out_ptr,
    M: tl.constexpr,
    REST: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    MERGED_ROWS_PER_PEER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
):
    flat_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = flat_rows < M
    group_id = tl.program_id(1)
    col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    x_offsets = flat_rows[:, None] * D + col_offsets[None, :]
    x = tl.load(x_ptr + x_offsets, mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )
    absmax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(absmax / FP8_MAX, 1.0e-10)
    q = tl.clamp(x / scale[:, None], FP8_MIN, FP8_MAX).to(tl.float8e4nv)

    first_row = flat_rows // REST
    rest_idx = flat_rows - first_row * REST
    peer = first_row // CHUNK_ROWS
    inner = first_row - peer * CHUNK_ROWS
    payload_row = peer * MERGED_ROWS_PER_PEER + inner
    payload_offsets = (
        payload_row[:, None] * REST * D
        + rest_idx[:, None] * D
        + col_offsets[None, :]
    )
    tl.store(q_out_ptr + payload_offsets, q, mask=row_mask[:, None])

    scale_cols = D // GROUP_SIZE
    scale_byte_linear = (inner * scale_cols + group_id) * 4
    scale_row = (
        peer * MERGED_ROWS_PER_PEER
        + CHUNK_ROWS
        + scale_byte_linear // D
    )
    scale_col = scale_byte_linear % D
    scale_byte_offset = (scale_row * REST + rest_idx) * D + scale_col
    tl.store(scale_out_ptr + scale_byte_offset // 4, scale, mask=row_mask)


@triton.jit
def _blockwise_quant_qkv_fp8_rowpack_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    scale_out_ptr,
    M: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    MERGED_ROWS_PER_PEER: tl.constexpr,
    stride_q_b,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
):
    packed_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = packed_rows < M
    qkv_id = packed_rows % 3
    tmp = packed_rows // 3
    h = tmp % H
    tmp = tmp // H
    s = tmp % S
    b = tmp // S

    group_id = tl.program_id(1)
    col_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    q_offsets = (
        b[:, None] * stride_q_b
        + s[:, None] * stride_q_s
        + h[:, None] * stride_q_h
        + col_offsets[None, :] * stride_q_d
    )
    k_offsets = (
        b[:, None] * stride_k_b
        + s[:, None] * stride_k_s
        + h[:, None] * stride_k_h
        + col_offsets[None, :] * stride_k_d
    )
    v_offsets = (
        b[:, None] * stride_v_b
        + s[:, None] * stride_v_s
        + h[:, None] * stride_v_h
        + col_offsets[None, :] * stride_v_d
    )
    x = tl.load(
        q_ptr + q_offsets,
        mask=row_mask[:, None] & (qkv_id[:, None] == 0),
        other=0.0,
    ).to(tl.float32)
    x += tl.load(
        k_ptr + k_offsets,
        mask=row_mask[:, None] & (qkv_id[:, None] == 1),
        other=0.0,
    ).to(tl.float32)
    x += tl.load(
        v_ptr + v_offsets,
        mask=row_mask[:, None] & (qkv_id[:, None] == 2),
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(absmax / FP8_MAX, 1.0e-10)
    q = tl.clamp(x / scale[:, None], FP8_MIN, FP8_MAX).to(tl.float8e4nv)

    first_row = 3 * h + qkv_id
    rest_idx = b * S + s
    rest = M // (3 * H)
    peer = first_row // CHUNK_ROWS
    inner = first_row - peer * CHUNK_ROWS
    payload_row = peer * MERGED_ROWS_PER_PEER + inner
    payload_offsets = (
        payload_row[:, None] * rest * D
        + rest_idx[:, None] * D
        + col_offsets[None, :]
    )
    tl.store(q_out_ptr + payload_offsets, q, mask=row_mask[:, None])

    scale_cols = D // GROUP_SIZE
    scale_byte_linear = (inner * scale_cols + group_id) * 4
    scale_row = (
        peer * MERGED_ROWS_PER_PEER
        + CHUNK_ROWS
        + scale_byte_linear // D
    )
    scale_col = scale_byte_linear % D
    scale_byte_offset = (scale_row * rest + rest_idx) * D + scale_col
    tl.store(scale_out_ptr + scale_byte_offset // 4, scale, mask=row_mask)


@triton.jit
def _pack_fp8_payload_scale_aligned_kernel(
    payload_ptr,
    scale_u8_ptr,
    combined_ptr,
    TOTAL: tl.constexpr,
    REST: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    SCALE_BYTE_COLS: tl.constexpr,
    MERGED_ROWS_PER_PEER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    col = offsets % D
    tmp = offsets // D
    rest_idx = tmp % REST
    out_row = tmp // REST

    peer = out_row // MERGED_ROWS_PER_PEER
    inner = out_row - peer * MERGED_ROWS_PER_PEER
    is_payload = inner < CHUNK_ROWS

    src_row = peer * CHUNK_ROWS + inner
    payload_offsets = src_row * REST * D + rest_idx * D + col

    scale_linear = (inner - CHUNK_ROWS) * D + col
    scale_row = peer * CHUNK_ROWS + scale_linear // SCALE_BYTE_COLS
    scale_byte = scale_linear % SCALE_BYTE_COLS
    scale_offsets = scale_row * REST * SCALE_BYTE_COLS + rest_idx * SCALE_BYTE_COLS + scale_byte
    valid_scale = (inner >= CHUNK_ROWS) & (scale_linear < CHUNK_ROWS * SCALE_BYTE_COLS)

    payload_val = tl.load(payload_ptr + payload_offsets, mask=mask & is_payload, other=0)
    scale_val = tl.load(scale_u8_ptr + scale_offsets, mask=mask & valid_scale, other=0)
    out = tl.where(is_payload, payload_val, scale_val)
    tl.store(combined_ptr + offsets, out, mask=mask)


def _rowpack_scale_rows(
    *,
    chunk_rows: int,
    d: int,
    scale_cols: int,
) -> int:
    scale_byte_cols = scale_cols * 4
    return triton.cdiv(chunk_rows * scale_byte_cols, d)


def _flatten_rowpack_shape(payload: torch.Tensor) -> tuple[int, int, int]:
    if payload.ndim < 2:
        raise ValueError("payload must have at least row and feature dimensions")
    rows = payload.shape[0]
    d = payload.shape[-1]
    rest = payload.numel() // (rows * d)
    return rows, rest, d


def aligned_rowpack_shape(
    payload_shape: tuple[int, ...],
    *,
    scale_cols: int,
    world_size: int,
) -> tuple[int, ...]:
    rows = payload_shape[0]
    d = payload_shape[-1]
    if rows % world_size != 0:
        raise ValueError(f"rows={rows} must be divisible by world_size={world_size}")
    chunk_rows = rows // world_size
    scale_rows = _rowpack_scale_rows(
        chunk_rows=chunk_rows,
        d=d,
        scale_cols=scale_cols,
    )
    return (world_size * (chunk_rows + scale_rows),) + payload_shape[1:]


def _validate_quant_rowpack_output(
    x: torch.Tensor,
    *,
    group_size: int,
    world_size: int,
    payload_shape: tuple[int, ...],
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, int, int, int, int]:
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"FP8 rowpack quant expects bf16/fp16 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")
    rows, rest, d = _flatten_rowpack_shape(x)
    if d % group_size != 0 or d % 4 != 0:
        raise ValueError(
            f"last dimension d={d} must be divisible by group_size={group_size} and 4"
        )
    if rows % world_size != 0:
        raise ValueError(f"rows={rows} must be divisible by world_size={world_size}")
    combined_shape = aligned_rowpack_shape(
        payload_shape,
        scale_cols=d // group_size,
        world_size=world_size,
    )
    if out is None:
        out = torch.empty(combined_shape, dtype=torch.uint8, device=x.device)
    elif (
        out.shape != combined_shape
        or out.dtype != torch.uint8
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must match aligned row-pack shape/device, be contiguous, and use uint8"
        )
    chunk_rows = rows // world_size
    scale_rows = _rowpack_scale_rows(
        chunk_rows=chunk_rows,
        d=d,
        scale_cols=d // group_size,
    )
    return out, rows, rest, d, chunk_rows + scale_rows


def blockwise_quant_fp8_rowpack(
    x: torch.Tensor,
    *,
    group_size: int,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    out, rows, rest, d, merged_rows_per_peer = _validate_quant_rowpack_output(
        x,
        group_size=group_size,
        world_size=world_size,
        payload_shape=tuple(x.shape),
        out=out,
    )
    flat_rows = rows * rest
    block_m = _block_m_for_rows(flat_rows)
    grid = (triton.cdiv(flat_rows, block_m), d // group_size)
    _blockwise_quant_fp8_rowpack_kernel[grid](
        x,
        out.view(torch.float8_e4m3fn),
        out.view(torch.float32),
        flat_rows,
        rest,
        d,
        group_size,
        rows // world_size,
        merged_rows_per_peer,
        block_m,
        _FP8_MAX,
        _FP8_MIN,
        num_warps=4,
        num_stages=2,
    )
    return out


def blockwise_quant_qkv_fp8_rowpack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError(
            "q, k, and v must have the same [B, S, H, D] shape, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    batch, seq_len, heads, d = q.shape
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError(
            "FP8 rowpack quant expects q, k, and v to share bf16/fp16 dtype, "
            f"got q={q.dtype} k={k.dtype} v={v.dtype}"
        )
    if k.device != q.device or v.device != q.device:
        raise ValueError(
            "q, k, and v must be on the same device, "
            f"got q={q.device} k={k.device} v={v.device}"
        )
    if d % group_size != 0 or d % 4 != 0:
        raise ValueError(
            f"last dimension d={d} must be divisible by group_size={group_size} and 4"
        )
    payload_shape = (3 * heads, batch, seq_len, d)
    rows = payload_shape[0]
    if rows % world_size != 0:
        raise ValueError(f"rows={rows} must be divisible by world_size={world_size}")
    combined_shape = aligned_rowpack_shape(
        payload_shape,
        scale_cols=d // group_size,
        world_size=world_size,
    )
    if out is None:
        out = torch.empty(combined_shape, dtype=torch.uint8, device=q.device)
    elif (
        out.shape != combined_shape
        or out.dtype != torch.uint8
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must match aligned row-pack shape/device, be contiguous, and use uint8"
        )
    chunk_rows = rows // world_size
    scale_rows = _rowpack_scale_rows(
        chunk_rows=chunk_rows,
        d=d,
        scale_cols=d // group_size,
    )
    merged_rows_per_peer = chunk_rows + scale_rows
    packed_input_rows = batch * seq_len * heads * 3
    block_m = _block_m_for_rows(packed_input_rows)
    grid = (triton.cdiv(packed_input_rows, block_m), d // group_size)
    _blockwise_quant_qkv_fp8_rowpack_kernel[grid](
        q,
        k,
        v,
        out.view(torch.float8_e4m3fn),
        out.view(torch.float32),
        packed_input_rows,
        seq_len,
        heads,
        d,
        group_size,
        chunk_rows,
        merged_rows_per_peer,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        block_m,
        _FP8_MAX,
        _FP8_MIN,
        num_warps=4,
        num_stages=2,
    )
    return out


def pack_fp8_payload_scale_aligned(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    world_size: int,
    out: torch.Tensor | None = None,
    block: int = 1024,
) -> torch.Tensor:
    if payload.dtype != _FP8_DTYPE:
        raise ValueError(f"payload must use float8_e4m3fn, got {payload.dtype}")
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must use float32, got {scale.dtype}")
    if not payload.is_contiguous() or not scale.is_contiguous():
        raise ValueError("payload and scale must be contiguous")
    rows, rest, d = _flatten_rowpack_shape(payload)
    if scale.ndim != payload.ndim or scale.shape[:-1] != payload.shape[:-1]:
        raise ValueError(
            "scale leading dimensions must match payload: "
            f"payload={tuple(payload.shape)} scale={tuple(scale.shape)}"
        )
    if d % 4 != 0:
        raise ValueError(f"payload last dimension d={d} must be divisible by 4")
    if rows % world_size != 0:
        raise ValueError(f"rows={rows} must be divisible by world_size={world_size}")

    combined_shape = aligned_rowpack_shape(
        tuple(payload.shape),
        scale_cols=scale.shape[-1],
        world_size=world_size,
    )
    if out is None:
        out = torch.empty(combined_shape, dtype=torch.uint8, device=payload.device)
    elif out.shape != combined_shape or out.dtype != torch.uint8 or out.device != payload.device:
        raise ValueError("out must match aligned row-pack shape/device and use uint8")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    chunk_rows = rows // world_size
    scale_byte_cols = scale.shape[-1] * 4
    scale_rows = _rowpack_scale_rows(
        chunk_rows=chunk_rows,
        d=d,
        scale_cols=scale.shape[-1],
    )
    merged_rows_per_peer = chunk_rows + scale_rows
    total = out.numel()
    _pack_fp8_payload_scale_aligned_kernel[(triton.cdiv(total, block),)](
        payload.view(torch.uint8),
        scale.view(torch.uint8),
        out,
        total,
        rest,
        d,
        world_size,
        chunk_rows,
        scale_byte_cols,
        merged_rows_per_peer,
        block,
        num_warps=8,
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


@triton.jit
def _fp8_dequant_unpack_qkv_rowpack_kernel(
    packed_q_ptr,
    scale_f32_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    N: tl.constexpr,
    S_LOCAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_FLOATS_PER_ROW: tl.constexpr,
    MERGED_ROWS_PER_PEER: tl.constexpr,
    stride_p_row,
    stride_p_b,
    stride_p_s,
    stride_scale_row,
    stride_scale_b,
    stride_scale_s,
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
    scale_cols = D // GROUP_SIZE
    chunk_base = ws * MERGED_ROWS_PER_PEER

    q_inner = 3 * h
    k_inner = q_inner + 1
    v_inner = q_inner + 2

    q_row = chunk_base + q_inner
    k_row = chunk_base + k_inner
    v_row = chunk_base + v_inner

    q_scale_linear = q_inner[:, None] * scale_cols + group_offsets[None, :]
    k_scale_linear = k_inner[:, None] * scale_cols + group_offsets[None, :]
    v_scale_linear = v_inner[:, None] * scale_cols + group_offsets[None, :]

    q_scale_row = chunk_base[:, None] + 3 * H_LOCAL + q_scale_linear // SCALE_FLOATS_PER_ROW
    k_scale_row = chunk_base[:, None] + 3 * H_LOCAL + k_scale_linear // SCALE_FLOATS_PER_ROW
    v_scale_row = chunk_base[:, None] + 3 * H_LOCAL + v_scale_linear // SCALE_FLOATS_PER_ROW
    q_scale_col = q_scale_linear % SCALE_FLOATS_PER_ROW
    k_scale_col = k_scale_linear % SCALE_FLOATS_PER_ROW
    v_scale_col = v_scale_linear % SCALE_FLOATS_PER_ROW

    scale_base = b[:, None] * stride_scale_b + s[:, None] * stride_scale_s
    q_scale = tl.load(
        scale_f32_ptr + q_scale_row * stride_scale_row + scale_base + q_scale_col,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(
        scale_f32_ptr + k_scale_row * stride_scale_row + scale_base + k_scale_col,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    v_scale = tl.load(
        scale_f32_ptr + v_scale_row * stride_scale_row + scale_base + v_scale_col,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    payload_base = b[:, None] * stride_p_b + s[:, None] * stride_p_s + d_offsets[None, :]
    q_val = (
        tl.load(
            packed_q_ptr + q_row[:, None] * stride_p_row + payload_base,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        * q_scale
    )
    k_val = (
        tl.load(
            packed_q_ptr + k_row[:, None] * stride_p_row + payload_base,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        * k_scale
    )
    v_val = (
        tl.load(
            packed_q_ptr + v_row[:, None] * stride_p_row + payload_base,
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
def _fp8_dequant_unpack_output_rowpack_kernel(
    packed_q_ptr,
    scale_f32_ptr,
    out_ptr,
    N: tl.constexpr,
    S_LOCAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_FLOATS_PER_ROW: tl.constexpr,
    MERGED_ROWS_PER_PEER: tl.constexpr,
    stride_p_row,
    stride_p_b,
    stride_p_h,
    stride_scale_row,
    stride_scale_b,
    stride_scale_h,
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
    scale_cols = D // GROUP_SIZE
    chunk_base = ws * MERGED_ROWS_PER_PEER
    payload_row = chunk_base + s_local

    scale_linear = s_local[:, None] * scale_cols + group_offsets[None, :]
    scale_row = chunk_base[:, None] + S_LOCAL + scale_linear // SCALE_FLOATS_PER_ROW
    scale_col = scale_linear % SCALE_FLOATS_PER_ROW
    scale_base = b[:, None] * stride_scale_b + h[:, None] * stride_scale_h
    scale = tl.load(
        scale_f32_ptr + scale_row * stride_scale_row + scale_base + scale_col,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    payload_offsets = (
        payload_row[:, None] * stride_p_row
        + b[:, None] * stride_p_b
        + h[:, None] * stride_p_h
        + d_offsets[None, :]
    )
    value = (
        tl.load(packed_q_ptr + payload_offsets, mask=m_mask[:, None], other=0.0).to(
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
    if packed_q.stride(-1) != 1:
        raise ValueError("packed_q last dimension must be contiguous")
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
    if packed_q.stride(-1) != 1:
        raise ValueError("packed_q last dimension must be contiguous")
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


def fused_dequant_unpack_qkv_fp8_rowpack(
    packed: torch.Tensor,
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
    if packed.dtype != torch.uint8:
        raise ValueError(f"row-packed tensor must use uint8, got {packed.dtype}")
    if D % group_size != 0:
        raise ValueError(f"D={D} must be divisible by group_size={group_size}")
    if D % 4 != 0:
        raise ValueError(f"D={D} must be divisible by 4")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"output dtype must be bf16/fp16, got {dtype}")

    chunk_rows = 3 * H_local
    scale_cols = D // group_size
    scale_rows = _rowpack_scale_rows(
        chunk_rows=chunk_rows,
        d=D,
        scale_cols=scale_cols,
    )
    merged_rows_per_peer = chunk_rows + scale_rows
    packed_shape = (world_size * merged_rows_per_peer, B, S_local, D)
    if packed.shape != packed_shape:
        raise ValueError(
            f"packed rowpack must have shape {packed_shape}, got {tuple(packed.shape)}"
        )
    if not packed.is_contiguous():
        raise ValueError("packed rowpack tensor must be contiguous")

    out_shape = (B, S_local * world_size, H_local, D)
    if out is None:
        q = torch.empty(out_shape, dtype=dtype, device=packed.device)
        k = torch.empty(out_shape, dtype=dtype, device=packed.device)
        v = torch.empty(out_shape, dtype=dtype, device=packed.device)
    else:
        q, k, v = out
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.shape != out_shape:
                raise ValueError(
                    f"{name} output must have shape {out_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != dtype or tensor.device != packed.device:
                raise ValueError(f"{name} output dtype/device must match request")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} output must be contiguous")

    packed_q = packed.view(torch.float8_e4m3fn)
    scale_f32 = packed.view(torch.float32)
    rows = B * world_size * S_local * H_local
    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m),)
    _fp8_dequant_unpack_qkv_rowpack_kernel[grid](
        packed_q,
        scale_f32,
        q,
        k,
        v,
        rows,
        S_local,
        H_local,
        D,
        world_size,
        group_size,
        D // 4,
        merged_rows_per_peer,
        packed_q.stride(0),
        packed_q.stride(1),
        packed_q.stride(2),
        scale_f32.stride(0),
        scale_f32.stride(1),
        scale_f32.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_m,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v


def fused_dequant_unpack_output_fp8_rowpack(
    packed: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    world_size: int,
    group_size: int,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if packed.ndim != 4:
        raise ValueError(
            f"packed rowpack must have shape [S_merged, B, H, D], got {tuple(packed.shape)}"
        )
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed rowpack must use uint8, got {packed.dtype}")
    if not packed.is_contiguous():
        raise ValueError("packed rowpack tensor must be contiguous")
    if packed.shape[1] != batch_size:
        raise ValueError(
            f"packed batch dimension must match batch_size={batch_size}, got {packed.shape[1]}"
        )
    if seq_len % world_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by world_size")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"output dtype must be bf16/fp16, got {dtype}")

    h_local = packed.shape[2]
    D = packed.shape[3]
    if D % group_size != 0:
        raise ValueError(f"D={D} must be divisible by group_size={group_size}")
    if D % 4 != 0:
        raise ValueError(f"D={D} must be divisible by 4")
    s_local = seq_len // world_size
    scale_cols = D // group_size
    scale_rows = _rowpack_scale_rows(
        chunk_rows=s_local,
        d=D,
        scale_cols=scale_cols,
    )
    merged_rows_per_peer = s_local + scale_rows
    expected_rows = world_size * merged_rows_per_peer
    if packed.shape[0] != expected_rows:
        raise ValueError(
            f"packed row count must be {expected_rows}, got {packed.shape[0]}"
        )

    h_global = h_local * world_size
    out_shape = (batch_size, s_local, h_global, D)
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=packed.device)
    elif out.shape != out_shape or out.dtype != dtype or out.device != packed.device:
        raise ValueError("out must match fused output shape/device/dtype")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    packed_q = packed.view(torch.float8_e4m3fn)
    scale_f32 = packed.view(torch.float32)
    rows = batch_size * s_local * world_size * h_local
    block_m = _block_m_for_rows(rows)
    grid = (triton.cdiv(rows, block_m),)
    _fp8_dequant_unpack_output_rowpack_kernel[grid](
        packed_q,
        scale_f32,
        out,
        rows,
        s_local,
        h_local,
        D,
        world_size,
        group_size,
        D // 4,
        merged_rows_per_peer,
        packed_q.stride(0),
        packed_q.stride(1),
        packed_q.stride(2),
        scale_f32.stride(0),
        scale_f32.stride(1),
        scale_f32.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_m,
        num_warps=4,
        num_stages=2,
    )
    return out
