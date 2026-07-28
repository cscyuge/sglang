"""Triton kernels for Stream-R1 segmented mixed-KV packing.

The segmented Stream-R1 path keeps noisy-cache K/V and current-condition K/V as
separate tensors. FA4 varlen attention still wants packed [total_tokens, H, D]
K/V. This kernel packs K and V together from the two segmented sources in a
single launch.
"""

from __future__ import annotations

import os

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


_SEGMENTED_KV_PACK_BLOCK_ROWS = int(
    os.environ.get("SGLANG_STREAM_R1_SEGMENTED_KV_PACK_BLOCK_ROWS", "4")
)
_SEGMENTED_KV_PACK_BLOCK_HD = int(
    os.environ.get("SGLANG_STREAM_R1_SEGMENTED_KV_PACK_BLOCK_HD", "0")
)
_SEGMENTED_KV_PACK_NUM_WARPS = int(
    os.environ.get("SGLANG_STREAM_R1_SEGMENTED_KV_PACK_NUM_WARPS", "4")
)


@triton.jit
def _fused_pack_segmented_kv_kernel(
    noisy_k_ptr,
    noisy_v_ptr,
    cond_k_ptr,
    cond_v_ptr,
    out_k_ptr,
    out_v_ptr,
    batch_indices_ptr,
    packed_starts_ptr,
    source_starts_ptr,
    lengths_ptr,
    noisy_seq_len,
    HD,
    D: tl.constexpr,
    stride_nk_b,
    stride_nk_s,
    stride_nk_h,
    stride_nk_d,
    stride_nv_b,
    stride_nv_s,
    stride_nv_h,
    stride_nv_d,
    stride_ck_b,
    stride_ck_s,
    stride_ck_h,
    stride_ck_d,
    stride_cv_b,
    stride_cv_s,
    stride_cv_h,
    stride_cv_d,
    stride_ok_t,
    stride_ok_h,
    stride_ok_d,
    stride_ov_t,
    stride_ov_h,
    stride_ov_d,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    range_id = tl.program_id(0)
    row_block = tl.program_id(1)

    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.program_id(2) * BLOCK_HD + tl.arange(0, BLOCK_HD)
    col_mask = cols < HD

    length = tl.load(lengths_ptr + range_id)
    row_mask = rows < length
    batch_index = tl.load(batch_indices_ptr + range_id).to(tl.int64)
    packed_start = tl.load(packed_starts_ptr + range_id).to(tl.int64)
    source_start = tl.load(source_starts_ptr + range_id).to(tl.int64)

    virtual_source = source_start + rows
    noisy_mask_rows = virtual_source < noisy_seq_len
    cond_source = virtual_source - noisy_seq_len

    heads = cols // D
    dims = cols - heads * D
    mask = row_mask[:, None] & col_mask[None, :]

    noisy_k_offsets = (
        batch_index * stride_nk_b
        + virtual_source[:, None] * stride_nk_s
        + heads[None, :] * stride_nk_h
        + dims[None, :] * stride_nk_d
    )
    noisy_v_offsets = (
        batch_index * stride_nv_b
        + virtual_source[:, None] * stride_nv_s
        + heads[None, :] * stride_nv_h
        + dims[None, :] * stride_nv_d
    )
    cond_k_offsets = (
        batch_index * stride_ck_b
        + cond_source[:, None] * stride_ck_s
        + heads[None, :] * stride_ck_h
        + dims[None, :] * stride_ck_d
    )
    cond_v_offsets = (
        batch_index * stride_cv_b
        + cond_source[:, None] * stride_cv_s
        + heads[None, :] * stride_cv_h
        + dims[None, :] * stride_cv_d
    )

    noisy_mask = mask & noisy_mask_rows[:, None]
    cond_mask = mask & (~noisy_mask_rows[:, None])
    noisy_k = tl.load(noisy_k_ptr + noisy_k_offsets, mask=noisy_mask, other=0.0)
    noisy_v = tl.load(noisy_v_ptr + noisy_v_offsets, mask=noisy_mask, other=0.0)
    cond_k = tl.load(cond_k_ptr + cond_k_offsets, mask=cond_mask, other=0.0)
    cond_v = tl.load(cond_v_ptr + cond_v_offsets, mask=cond_mask, other=0.0)
    packed_k = tl.where(noisy_mask_rows[:, None], noisy_k, cond_k)
    packed_v = tl.where(noisy_mask_rows[:, None], noisy_v, cond_v)

    out_rows = packed_start + rows
    out_k_offsets = (
        out_rows[:, None] * stride_ok_t
        + heads[None, :] * stride_ok_h
        + dims[None, :] * stride_ok_d
    )
    out_v_offsets = (
        out_rows[:, None] * stride_ov_t
        + heads[None, :] * stride_ov_h
        + dims[None, :] * stride_ov_d
    )
    tl.store(out_k_ptr + out_k_offsets, packed_k, mask=mask)
    tl.store(out_v_ptr + out_v_offsets, packed_v, mask=mask)


def _validate_copy_plan_tensor(name: str, tensor: torch.Tensor, device: torch.device):
    if tensor.device != device:
        raise ValueError(f"{name} device must match source tensors")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32 or int64")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D")


def fused_pack_segmented_kv(
    noisy_key: torch.Tensor,
    noisy_value: torch.Tensor,
    condition_key: torch.Tensor,
    condition_value: torch.Tensor,
    batch_indices: torch.Tensor,
    packed_starts: torch.Tensor,
    source_starts: torch.Tensor,
    lengths: torch.Tensor,
    *,
    total_tokens: int,
    noisy_seq_len: int,
    max_length: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    block_rows: int | None = None,
    block_hd: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack segmented K/V ranges into FA varlen layout in one Triton launch."""

    if not noisy_key.is_cuda:
        raise ValueError("fused segmented KV pack requires CUDA tensors")
    if noisy_key.shape != noisy_value.shape:
        raise ValueError("noisy key/value shapes must match")
    if condition_key.shape != condition_value.shape:
        raise ValueError("condition key/value shapes must match")
    if noisy_key.dim() != 4 or condition_key.dim() != 4:
        raise ValueError("segmented K/V tensors must be [B, S, H, D]")
    if noisy_key.shape[0] != condition_key.shape[0] or noisy_key.shape[2:] != condition_key.shape[2:]:
        raise ValueError("noisy and condition K/V batch/head dimensions must match")
    if noisy_key.dtype != noisy_value.dtype or noisy_key.dtype != condition_key.dtype or noisy_key.dtype != condition_value.dtype:
        raise ValueError("segmented K/V dtypes must match")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    device = noisy_key.device
    for name, tensor in (
        ("batch_indices", batch_indices),
        ("packed_starts", packed_starts),
        ("source_starts", source_starts),
        ("lengths", lengths),
    ):
        _validate_copy_plan_tensor(name, tensor, device)
    if not (
        batch_indices.shape == packed_starts.shape == source_starts.shape == lengths.shape
    ):
        raise ValueError("copy plan tensors must have matching shape")
    if batch_indices.numel() == 0:
        raise ValueError("copy plan must be non-empty")

    _, _, num_heads, head_dim = noisy_key.shape
    hd = num_heads * head_dim
    block_rows = (
        _SEGMENTED_KV_PACK_BLOCK_ROWS if block_rows is None else block_rows
    )
    block_hd = _SEGMENTED_KV_PACK_BLOCK_HD if block_hd is None else block_hd
    num_warps = (
        _SEGMENTED_KV_PACK_NUM_WARPS if num_warps is None else num_warps
    )
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    if block_hd == 0:
        block_hd = triton.next_power_of_2(hd)
    if block_hd <= 0 or block_hd & (block_hd - 1):
        raise ValueError("block_hd must be zero or a positive power of two")
    if block_hd > 131072:
        raise ValueError(f"unsupported segmented KV feature size: {hd}")
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported Triton warp count")

    output_shape = (total_tokens, num_heads, head_dim)
    if out is None:
        packed_key = torch.empty(output_shape, dtype=noisy_key.dtype, device=device)
        packed_value = torch.empty(output_shape, dtype=noisy_key.dtype, device=device)
    else:
        packed_key, packed_value = out
        for name, tensor in (("packed_key", packed_key), ("packed_value", packed_value)):
            if tensor.shape != output_shape:
                raise ValueError(
                    f"{name} must have shape {output_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != noisy_key.dtype or tensor.device != device:
                raise ValueError(f"{name} dtype/device must match source tensors")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    grid = (
        batch_indices.numel(),
        triton.cdiv(max_length, block_rows),
        triton.cdiv(hd, block_hd),
    )
    with torch.get_device_module().device(device):
        _fused_pack_segmented_kv_kernel[grid](
            noisy_key,
            noisy_value,
            condition_key,
            condition_value,
            packed_key,
            packed_value,
            batch_indices,
            packed_starts,
            source_starts,
            lengths,
            noisy_seq_len,
            hd,
            head_dim,
            noisy_key.stride(0),
            noisy_key.stride(1),
            noisy_key.stride(2),
            noisy_key.stride(3),
            noisy_value.stride(0),
            noisy_value.stride(1),
            noisy_value.stride(2),
            noisy_value.stride(3),
            condition_key.stride(0),
            condition_key.stride(1),
            condition_key.stride(2),
            condition_key.stride(3),
            condition_value.stride(0),
            condition_value.stride(1),
            condition_value.stride(2),
            condition_value.stride(3),
            packed_key.stride(0),
            packed_key.stride(1),
            packed_key.stride(2),
            packed_value.stride(0),
            packed_value.stride(1),
            packed_value.stride(2),
            BLOCK_ROWS=block_rows,
            BLOCK_HD=block_hd,
            num_warps=num_warps,
        )
    return packed_key, packed_value
