"""Triton helpers for Wan S2V segment-wise elementwise ops."""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from triton.language.extra import libdevice  # type: ignore


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
def _segment_modulate_quant_fp8_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    q_ptr,
    q_scale_ptr,
    rows: tl.constexpr,
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
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    GROUP_BLOCKS_PER_ROW: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    EPS: tl.constexpr,
):
    group_block = tl.program_id(0)
    row = group_block // GROUP_BLOCKS_PER_ROW
    group_block = group_block - row * GROUP_BLOCKS_PER_ROW
    group = group_block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    group_mask = group < GROUPS_PER_ROW
    col = group[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    seq = row % seq_len
    batch = row // seq_len
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
    x = tl.load(x_ptr + x_offsets, mask=group_mask[:, None]).to(tl.float32)
    shift = tl.load(
        shift_ptr + shift_offsets,
        mask=group_mask[:, None],
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + scale_offsets,
        mask=group_mask[:, None],
    ).to(tl.float32)
    modulated = x * (1.0 + scale) + shift
    # Match the standalone segment-modulate kernel's BF16 store/reload before
    # applying per-token, per-128 FP8 activation quantization.
    modulated = modulated.to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(modulated), axis=1), EPS)
    q_scale = absmax / FP8_MAX
    q_scale_inv = FP8_MAX / absmax
    q = tl.clamp(
        modulated * q_scale_inv[:, None],
        FP8_MIN,
        FP8_MAX,
    ).to(tl.float8e4nv)

    q_offsets = row * hidden_dim + col
    tl.store(q_ptr + q_offsets, q, mask=group_mask[:, None])
    # FlashInfer CUTLASS scale_major_mode="MN" consumes [K/128, M].
    tl.store(
        q_scale_ptr + group * rows + row,
        q_scale,
        mask=group_mask,
    )


@triton.jit
def _gelu_tanh_quant_fp8_kernel(
    x_ptr,
    bias_ptr,
    q_ptr,
    q_scale_ptr,
    rows: tl.constexpr,
    hidden_dim: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    GROUP_BLOCKS_PER_ROW: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    group_block = tl.program_id(0)
    row = group_block // GROUP_BLOCKS_PER_ROW
    group_block = group_block - row * GROUP_BLOCKS_PER_ROW
    group = group_block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    group_mask = group < GROUPS_PER_ROW
    col = group[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    offsets = row * hidden_dim + col

    x = tl.load(x_ptr + offsets, mask=group_mask[:, None]).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + col, mask=group_mask[:, None]).to(tl.float32)
        # The unfused FP8 linear materializes ``output += bias`` in BF16
        # before GELU consumes it. Preserve that intermediate rounding.
        x = (x + bias).to(tl.bfloat16).to(tl.float32)
    x_cubed = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)
    activated = 0.5 * x * (1.0 + libdevice.tanh(inner))
    # nn.GELU receives BF16 and materializes BF16 before activation quant.
    activated = activated.to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(activated), axis=1), EPS)
    q_scale = absmax / FP8_MAX
    q_scale_inv = FP8_MAX / absmax
    q = tl.clamp(
        activated * q_scale_inv[:, None],
        FP8_MIN,
        FP8_MAX,
    ).to(tl.float8e4nv)
    tl.store(q_ptr + offsets, q, mask=group_mask[:, None])
    tl.store(
        q_scale_ptr + group * rows + row,
        q_scale,
        mask=group_mask,
    )


@triton.jit
def _segment_gate_add_kernel(
    residual_ptr,
    update_ptr,
    gate_ptr,
    bias_ptr,
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
    HAS_BIAS: tl.constexpr,
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
    if HAS_BIAS:
        bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
        # Match the separate BF16 linear-bias kernel before applying the gate.
        update = (update + bias).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    out = residual + update * gate
    tl.store(out_ptr + out_offsets, out, mask=mask)


def segment_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    seg_idx: int,
    *,
    bias: torch.Tensor | None = None,
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
    if bias is not None:
        if (
            bias.dim() != 1
            or bias.shape[0] != residual.shape[2]
            or bias.dtype != update.dtype
            or bias.device != update.device
        ):
            raise ValueError(
                "Wan S2V segment gate add bias must match update hidden dimension, "
                f"dtype, and device: update={tuple(update.shape)}/{update.dtype}/"
                f"{update.device} bias={tuple(bias.shape)}/{bias.dtype}/{bias.device}"
            )
        if bias.stride(0) != 1:
            raise ValueError("Wan S2V segment gate add expects contiguous bias")

    batch, seq_len, hidden_dim = residual.shape
    out = torch.empty_like(residual)
    seg_idx = min(max(0, int(seg_idx)), seq_len)
    total = batch * seq_len * hidden_dim
    grid = (triton.cdiv(total, 256),)
    _segment_gate_add_kernel[grid](
        residual,
        update,
        gate,
        bias if bias is not None else update,
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
        HAS_BIAS=bias is not None,
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


def segment_modulate_quant_fp8(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    seg_idx: int,
    *,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Wan segment modulation with CUTLASS-layout FP8 prequantization."""

    if x.dim() != 3:
        raise ValueError(
            f"Wan S2V segment prequant expects x [B, S, C], got {tuple(x.shape)}"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError(f"Wan S2V segment prequant expects BF16 x, got {x.dtype}")
    if shift.dim() != 3 or scale.dim() != 3:
        raise ValueError("shift and scale must have shape [B, 2, C]")
    if (
        shift.shape[0] != x.shape[0]
        or scale.shape[0] != x.shape[0]
        or shift.shape[1] < 2
        or scale.shape[1] < 2
        or shift.shape[2] != x.shape[2]
        or scale.shape[2] != x.shape[2]
    ):
        raise ValueError(
            "shift and scale must match x batch/hidden dimensions and have "
            f"two segments, got x={tuple(x.shape)} shift={tuple(shift.shape)} "
            f"scale={tuple(scale.shape)}"
        )
    if not (x.is_cuda and shift.is_cuda and scale.is_cuda):
        raise ValueError("Wan S2V segment prequant expects CUDA tensors")
    if x.device != shift.device or x.device != scale.device:
        raise ValueError("Wan S2V segment prequant tensors must share a device")
    if x.stride(-1) != 1 or shift.stride(-1) != 1 or scale.stride(-1) != 1:
        raise ValueError("Wan S2V segment prequant expects contiguous hidden dimension")

    batch, seq_len, hidden_dim = x.shape
    if hidden_dim % group_size != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by group_size={group_size}"
        )
    rows = batch * seq_len
    groups_per_row = hidden_dim // group_size
    block_groups = 4
    group_blocks_per_row = triton.cdiv(groups_per_row, block_groups)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(
        (groups_per_row, rows),
        device=x.device,
        dtype=torch.float32,
    )
    seg_idx = min(max(0, int(seg_idx)), seq_len)
    _segment_modulate_quant_fp8_kernel[(rows * group_blocks_per_row,)](
        x,
        shift,
        scale,
        q,
        q_scale,
        rows,
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
        GROUP_SIZE=group_size,
        GROUPS_PER_ROW=groups_per_row,
        BLOCK_GROUPS=block_groups,
        GROUP_BLOCKS_PER_ROW=group_blocks_per_row,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        EPS=1.0e-10,
        num_warps=8,
        num_stages=1,
    )
    return q, q_scale


def gelu_tanh_quant_fp8(
    x: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse BF16 tanh-GELU with CUTLASS-layout FP8 prequantization."""

    if x.ndim < 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError(
            "Wan S2V GELU prequant expects a CUDA BF16 tensor with at least 2 dims"
        )
    if not x.is_contiguous():
        raise ValueError("Wan S2V GELU prequant expects contiguous input")
    hidden_dim = x.shape[-1]
    if bias is not None:
        if (
            bias.dim() != 1
            or bias.shape[0] != hidden_dim
            or bias.dtype != x.dtype
            or bias.device != x.device
            or bias.stride(0) != 1
        ):
            raise ValueError(
                "Wan S2V GELU prequant bias must be contiguous and match the "
                f"input hidden dimension, dtype, and device: x={tuple(x.shape)}/"
                f"{x.dtype}/{x.device} bias={tuple(bias.shape)}/{bias.dtype}/"
                f"{bias.device}"
            )
    if hidden_dim % group_size != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by group_size={group_size}"
        )
    rows = x.numel() // hidden_dim
    groups_per_row = hidden_dim // group_size
    block_groups = 4
    group_blocks_per_row = triton.cdiv(groups_per_row, block_groups)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(
        (groups_per_row, rows),
        device=x.device,
        dtype=torch.float32,
    )
    _gelu_tanh_quant_fp8_kernel[(rows * group_blocks_per_row,)](
        x,
        bias if bias is not None else x,
        q,
        q_scale,
        rows,
        hidden_dim,
        GROUP_SIZE=group_size,
        GROUPS_PER_ROW=groups_per_row,
        BLOCK_GROUPS=block_groups,
        GROUP_BLOCKS_PER_ROW=group_blocks_per_row,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        EPS=1.0e-10,
        HAS_BIAS=bias is not None,
        num_warps=8,
        num_stages=1,
    )
    return q, q_scale
