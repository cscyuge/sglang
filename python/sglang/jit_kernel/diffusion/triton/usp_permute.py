"""Triton kernels for fused QKV pack/unpack in Ulysses SP all-to-all.

Batches 3 separate input all-to-all calls (for q, k, v) into 1 by
interleaving QKV heads in a single packed buffer:

  Pack:   q,k,v [B, S, H, D] → packed [3*H, B, S, D]
  Unpack: packed [3*H, B, S, D] → q,k,v [B, S_global, H_local, D]

Interleaving: packed[3*h + t, b, s, :] = src_t[b, s, h, :]
This ensures each all-to-all chunk of size 3*H_local cleanly contains
H_local heads from each of q, k, v.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _pack_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    B,
    S,
    H,
    D: tl.constexpr,
    stride_q_b,
    stride_q_s,
    stride_q_h,
    stride_k_b,
    stride_k_s,
    stride_k_h,
    stride_v_b,
    stride_v_s,
    stride_v_h,
    # out is contiguous [3*H, B, S, D]
    stride_o_hh,  # stride along dim0 (3*H)
    stride_o_b,
    stride_o_s,
):
    """Pack q, k, v [B, S, H, D] into packed [3*H, B, S, D].

    Grid: (B * S * H,)
    Each program handles one (b, s, h) tuple and copies D elements for all 3 tensors.
    """
    pid = tl.program_id(0)
    # Decode (b, s, h) from linear index
    h = pid % H
    tmp = pid // H
    s = tmp % S
    b = tmp // S

    d_offs = tl.arange(0, D)

    # Load q[b, s, h, :], k[b, s, h, :], v[b, s, h, :]
    src_base_q = q_ptr + b * stride_q_b + s * stride_q_s + h * stride_q_h
    src_base_k = k_ptr + b * stride_k_b + s * stride_k_s + h * stride_k_h
    src_base_v = v_ptr + b * stride_v_b + s * stride_v_s + h * stride_v_h

    q_val = tl.load(src_base_q + d_offs)
    k_val = tl.load(src_base_k + d_offs)
    v_val = tl.load(src_base_v + d_offs)

    # Store to packed[3*h + t, b, s, :] for t in {0,1,2}
    out_base = out_ptr + b * stride_o_b + s * stride_o_s
    tl.store(out_base + (3 * h + 0) * stride_o_hh + d_offs, q_val)
    tl.store(out_base + (3 * h + 1) * stride_o_hh + d_offs, k_val)
    tl.store(out_base + (3 * h + 2) * stride_o_hh + d_offs, v_val)


@triton.jit
def _unpack_qkv_kernel(
    packed_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    B,
    S_local,
    H_local,
    D: tl.constexpr,
    W,  # world_size
    # packed is contiguous [3*H_global, B, S_local, D] where H_global = H_local * W
    stride_p_hh,  # stride along dim0
    stride_p_b,
    stride_p_s,
    # output q,k,v are contiguous [B, S_global, H_local, D] where S_global = S_local * W
    stride_o_b,
    stride_o_s,
    stride_o_h,
):
    """Unpack packed [3*H_global, B, S_local, D] → q,k,v [B, S_global, H_local, D].

    After all-to-all, the packed buffer has shape [3*H_global, B, S_local, D].
    It can be viewed as [W, 3*H_local, B, S_local, D] where chunk ws contains
    the heads from rank ws.

    Mapping: dst_t[b, ws*S_local + s, h, :] = packed[ws * 3*H_local + 3*h + t, b, s, :]

    Grid: (B * S_local * H_local * W,)
    Each program handles one (b, ws, s, h) and copies D elements for all 3 tensors.
    """
    pid = tl.program_id(0)
    # Decode (b, ws, s, h)
    h = pid % H_local
    tmp = pid // H_local
    s = tmp % S_local
    tmp2 = tmp // S_local
    ws = tmp2 % W
    b = tmp2 // W

    d_offs = tl.arange(0, D)

    chunk_base = ws * 3 * H_local

    # Load packed[chunk_base + 3*h + t, b, s, :]
    p_base = packed_ptr + b * stride_p_b + s * stride_p_s
    q_val = tl.load(p_base + (chunk_base + 3 * h + 0) * stride_p_hh + d_offs)
    k_val = tl.load(p_base + (chunk_base + 3 * h + 1) * stride_p_hh + d_offs)
    v_val = tl.load(p_base + (chunk_base + 3 * h + 2) * stride_p_hh + d_offs)

    # Store to dst[b, ws*S_local + s, h, :]
    s_global = ws * S_local + s
    dst_off = b * stride_o_b + s_global * stride_o_s + h * stride_o_h
    tl.store(q_ptr + dst_off + d_offs, q_val)
    tl.store(k_ptr + dst_off + d_offs, k_val)
    tl.store(v_ptr + dst_off + d_offs, v_val)


def fused_pack_qkv_for_all_to_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pack q, k, v [B, S, H, D] into interleaved [3*H, B, S, D] for batched all-to-all.

    Returns a contiguous tensor ready for _usp_all_to_all_single.
    """
    B, S, H, D = q.shape
    packed = torch.empty(3 * H, B, S, D, dtype=q.dtype, device=q.device)

    grid = (B * S * H,)
    _pack_qkv_kernel[grid](
        q, k, v, packed,
        B, S, H, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        packed.stride(0), packed.stride(1), packed.stride(2),
        num_warps=4,
    )
    return packed


def fused_unpack_qkv_from_all_to_all(
    packed: torch.Tensor,
    B: int,
    S_local: int,
    H_local: int,
    D: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack packed [3*H_global, B, S_local, D] → q,k,v [B, S_global, H_local, D].

    Called after _usp_all_to_all_single on the packed buffer.
    """
    S_global = S_local * world_size
    q = torch.empty(B, S_global, H_local, D, dtype=packed.dtype, device=packed.device)
    k = torch.empty(B, S_global, H_local, D, dtype=packed.dtype, device=packed.device)
    v = torch.empty(B, S_global, H_local, D, dtype=packed.dtype, device=packed.device)

    grid = (B * world_size * S_local * H_local,)
    _unpack_qkv_kernel[grid](
        packed, q, k, v,
        B, S_local, H_local, D, world_size,
        packed.stride(0), packed.stride(1), packed.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        num_warps=4,
    )
    return q, k, v
