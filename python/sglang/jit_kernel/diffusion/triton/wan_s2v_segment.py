"""Triton helpers for Wan S2V segment-wise elementwise ops."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _segment_modulate_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    seg_idx: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_s: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_shift_b: tl.constexpr,
    stride_shift_seg: tl.constexpr,
    stride_shift_c: tl.constexpr,
    stride_scale_b: tl.constexpr,
    stride_scale_seg: tl.constexpr,
    stride_scale_c: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_s: tl.constexpr,
    stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    col = offsets % hidden_dim
    seq = (offsets // hidden_dim) % seq_len
    batch = offsets // (hidden_dim * seq_len)
    segment = seq >= seg_idx

    x_offsets = batch * stride_x_b + seq * stride_x_s + col * stride_x_c
    shift_offsets = (
        batch * stride_shift_b
        + segment.to(tl.int64) * stride_shift_seg
        + col * stride_shift_c
    )
    scale_offsets = (
        batch * stride_scale_b
        + segment.to(tl.int64) * stride_scale_seg
        + col * stride_scale_c
    )
    out_offsets = batch * stride_out_b + seq * stride_out_s + col * stride_out_c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * (1.0 + scale) + shift
    tl.store(out_ptr + out_offsets, out, mask=mask)


@triton.jit
def _segment_gate_add_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    seg_idx: tl.constexpr,
    stride_res_b: tl.constexpr,
    stride_res_s: tl.constexpr,
    stride_res_c: tl.constexpr,
    stride_upd_b: tl.constexpr,
    stride_upd_s: tl.constexpr,
    stride_upd_c: tl.constexpr,
    stride_gate_b: tl.constexpr,
    stride_gate_seg: tl.constexpr,
    stride_gate_c: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_s: tl.constexpr,
    stride_out_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    col = offsets % hidden_dim
    seq = (offsets // hidden_dim) % seq_len
    batch = offsets // (hidden_dim * seq_len)
    segment = seq >= seg_idx

    residual_offsets = (
        batch * stride_res_b + seq * stride_res_s + col * stride_res_c
    )
    update_offsets = batch * stride_upd_b + seq * stride_upd_s + col * stride_upd_c
    gate_offsets = (
        batch * stride_gate_b + segment.to(tl.int64) * stride_gate_seg + col * stride_gate_c
    )
    out_offsets = batch * stride_out_b + seq * stride_out_s + col * stride_out_c

    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    update = tl.load(update_ptr + update_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    out = residual + update * gate
    tl.store(out_ptr + out_offsets, out, mask=mask)


def segment_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    seg_idx: int,
) -> torch.Tensor:
    """Compute ``residual + update * gate[:, segment]`` for Wan S2V segments."""

    if residual.dim() != 3:
        raise ValueError(
            f"Wan S2V segment gate add expects residual [B, S, C], got {tuple(residual.shape)}"
        )
    if update.shape != residual.shape:
        raise ValueError(
            "Wan S2V segment gate add update must match residual shape: "
            f"residual={tuple(residual.shape)} update={tuple(update.shape)}"
        )
    if gate.dim() != 3 or gate.shape[0] != residual.shape[0] or gate.shape[1] < 2:
        raise ValueError(
            "Wan S2V segment gate must have shape [B, 2, C]: "
            f"residual={tuple(residual.shape)} gate={tuple(gate.shape)}"
        )
    if gate.shape[2] != residual.shape[2]:
        raise ValueError(
            "Wan S2V segment gate hidden dimension mismatch: "
            f"residual={tuple(residual.shape)} gate={tuple(gate.shape)}"
        )
    if residual.dtype != update.dtype:
        raise ValueError(
            f"residual/update dtype mismatch: {residual.dtype} vs {update.dtype}"
        )
    if not (residual.is_cuda and update.is_cuda and gate.is_cuda):
        raise ValueError("Wan S2V segment gate add expects CUDA tensors")
    if residual.device != update.device or residual.device != gate.device:
        raise ValueError("Wan S2V segment gate add tensors must share a device")
    if residual.stride(-1) != 1 or update.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("Wan S2V segment gate add expects contiguous hidden dimension")

    batch, seq_len, hidden_dim = residual.shape
    out = torch.empty_like(residual)
    seg_idx = min(max(0, int(seg_idx)), seq_len)
    total = batch * seq_len * hidden_dim
    grid = (triton.cdiv(total, 256),)
    _segment_gate_add_kernel[grid](
        residual,
        update,
        gate,
        out,
        total,
        seq_len,
        hidden_dim,
        seg_idx,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        update.stride(0),
        update.stride(1),
        update.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out


def segment_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    seg_idx: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``x * (1 + scale[:, segment]) + shift[:, segment]``."""

    if x.dim() != 3:
        raise ValueError(
            f"Wan S2V segment modulate expects x [B, S, C], got {tuple(x.shape)}"
        )
    if shift.dim() != 3 or shift.shape[0] != x.shape[0] or shift.shape[1] < 2:
        raise ValueError(
            "Wan S2V segment shift must have shape [B, 2, C]: "
            f"x={tuple(x.shape)} shift={tuple(shift.shape)}"
        )
    if scale.dim() != 3 or scale.shape[0] != x.shape[0] or scale.shape[1] < 2:
        raise ValueError(
            "Wan S2V segment scale must have shape [B, 2, C]: "
            f"x={tuple(x.shape)} scale={tuple(scale.shape)}"
        )
    if shift.shape[2] != x.shape[2] or scale.shape[2] != x.shape[2]:
        raise ValueError(
            "Wan S2V segment modulate hidden dimension mismatch: "
            f"x={tuple(x.shape)} shift={tuple(shift.shape)} scale={tuple(scale.shape)}"
        )
    if not (x.is_cuda and shift.is_cuda and scale.is_cuda):
        raise ValueError("Wan S2V segment modulate expects CUDA tensors")
    if x.device != shift.device or x.device != scale.device:
        raise ValueError("Wan S2V segment modulate tensors must share a device")
    if x.stride(-1) != 1 or shift.stride(-1) != 1 or scale.stride(-1) != 1:
        raise ValueError("Wan S2V segment modulate expects contiguous hidden dimension")

    batch, seq_len, hidden_dim = x.shape
    out_dtype = out_dtype or torch.promote_types(
        torch.promote_types(x.dtype, shift.dtype), scale.dtype
    )
    out = torch.empty_like(x, dtype=out_dtype)
    seg_idx = min(max(0, int(seg_idx)), seq_len)
    total = batch * seq_len * hidden_dim
    grid = (triton.cdiv(total, 256),)
    _segment_modulate_kernel[grid](
        x,
        shift,
        scale,
        out,
        total,
        seq_len,
        hidden_dim,
        seg_idx,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        shift.stride(0),
        shift.stride(1),
        shift.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out
