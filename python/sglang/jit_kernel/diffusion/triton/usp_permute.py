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

_QKV_PERMUTE_BLOCK_M = 4


@triton.jit
def _pack_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
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
    # out is contiguous [3*H, B, S, D]
    stride_o_hh,  # stride along dim0 (3*H)
    stride_o_b,
    stride_o_s,
    BLOCK_M: tl.constexpr,
):
    """Pack q, k, v [B, S, H, D] into packed [3*H, B, S, D].

    Each program handles a small block of flattened (b, s, h) tuples and
    copies D elements for all 3 tensors. Blocking reduces CTA count on the
    Wan S2V hot shape without changing the interleaved all-to-all layout.
    """
    m_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < N

    # Decode (b, s, h) from linear index
    h = m_offsets % H
    tmp = m_offsets // H
    s = tmp % S
    b = tmp // S

    d_offs = tl.arange(0, D)

    # Load q[b, s, h, :], k[b, s, h, :], v[b, s, h, :]
    src_base_q = (
        q_ptr
        + b[:, None] * stride_q_b
        + s[:, None] * stride_q_s
        + h[:, None] * stride_q_h
    )
    src_base_k = (
        k_ptr
        + b[:, None] * stride_k_b
        + s[:, None] * stride_k_s
        + h[:, None] * stride_k_h
    )
    src_base_v = (
        v_ptr
        + b[:, None] * stride_v_b
        + s[:, None] * stride_v_s
        + h[:, None] * stride_v_h
    )

    q_val = tl.load(src_base_q + d_offs[None, :] * stride_q_d, mask=m_mask[:, None])
    k_val = tl.load(src_base_k + d_offs[None, :] * stride_k_d, mask=m_mask[:, None])
    v_val = tl.load(src_base_v + d_offs[None, :] * stride_v_d, mask=m_mask[:, None])

    # Store to packed[3*h + t, b, s, :] for t in {0,1,2}
    out_base = out_ptr + b[:, None] * stride_o_b + s[:, None] * stride_o_s
    tl.store(
        out_base + (3 * h[:, None] + 0) * stride_o_hh + d_offs[None, :],
        q_val,
        mask=m_mask[:, None],
    )
    tl.store(
        out_base + (3 * h[:, None] + 1) * stride_o_hh + d_offs[None, :],
        k_val,
        mask=m_mask[:, None],
    )
    tl.store(
        out_base + (3 * h[:, None] + 2) * stride_o_hh + d_offs[None, :],
        v_val,
        mask=m_mask[:, None],
    )


@triton.jit
def _unpack_qkv_kernel(
    packed_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    N: tl.constexpr,
    S_local: tl.constexpr,
    H_local: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,  # world_size
    # packed is contiguous [3*H_global, B, S_local, D] where H_global = H_local * W
    stride_p_hh,  # stride along dim0
    stride_p_b,
    stride_p_s,
    # output q,k,v are contiguous [B, S_global, H_local, D] where S_global = S_local * W
    stride_o_b,
    stride_o_s,
    stride_o_h,
    BLOCK_M: tl.constexpr,
):
    """Unpack packed [3*H_global, B, S_local, D] → q,k,v [B, S_global, H_local, D].

    After all-to-all, the packed buffer has shape [3*H_global, B, S_local, D].
    It can be viewed as [W, 3*H_local, B, S_local, D] where chunk ws contains
    the heads from rank ws.

    Mapping: dst_t[b, ws*S_local + s, h, :] = packed[ws * 3*H_local + 3*h + t, b, s, :]

    Each program handles a small block of flattened (b, ws, s, h) tuples and
    copies D elements for all 3 tensors.
    """
    m_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < N

    # Decode (b, ws, s, h)
    h = m_offsets % H_local
    tmp = m_offsets // H_local
    s = tmp % S_local
    tmp2 = tmp // S_local
    ws = tmp2 % W
    b = tmp2 // W

    d_offs = tl.arange(0, D)

    chunk_base = ws * 3 * H_local

    # Load packed[chunk_base + 3*h + t, b, s, :]
    p_base = packed_ptr + b[:, None] * stride_p_b + s[:, None] * stride_p_s
    q_val = tl.load(
        p_base + (chunk_base[:, None] + 3 * h[:, None] + 0) * stride_p_hh + d_offs[None, :],
        mask=m_mask[:, None],
    )
    k_val = tl.load(
        p_base + (chunk_base[:, None] + 3 * h[:, None] + 1) * stride_p_hh + d_offs[None, :],
        mask=m_mask[:, None],
    )
    v_val = tl.load(
        p_base + (chunk_base[:, None] + 3 * h[:, None] + 2) * stride_p_hh + d_offs[None, :],
        mask=m_mask[:, None],
    )

    # Store to dst[b, ws*S_local + s, h, :]
    s_global = ws * S_local + s
    dst_off = (
        b[:, None] * stride_o_b
        + s_global[:, None] * stride_o_s
        + h[:, None] * stride_o_h
    )
    tl.store(q_ptr + dst_off + d_offs[None, :], q_val, mask=m_mask[:, None])
    tl.store(k_ptr + dst_off + d_offs[None, :], k_val, mask=m_mask[:, None])
    tl.store(v_ptr + dst_off + d_offs[None, :], v_val, mask=m_mask[:, None])


def fused_pack_qkv_for_all_to_all(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack q, k, v [B, S, H, D] into interleaved [3*H, B, S, D] for batched all-to-all.

    Returns a contiguous tensor ready for _usp_all_to_all_single.
    """
    B, S, H, D = q.shape
    packed_shape = (3 * H, B, S, D)
    if out is None:
        packed = torch.empty(packed_shape, dtype=q.dtype, device=q.device)
    else:
        if out.shape != packed_shape:
            raise ValueError(
                f"packed QKV output must have shape {packed_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != q.dtype or out.device != q.device:
            raise ValueError("packed QKV output dtype/device must match q")
        if not out.is_contiguous():
            raise ValueError("packed QKV output must be contiguous")
        packed = out

    block_m = _QKV_PERMUTE_BLOCK_M
    grid = (triton.cdiv(B * S * H, block_m),)
    _pack_qkv_kernel[grid](
        q, k, v, packed,
        B * S * H, S, H, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        packed.stride(0), packed.stride(1), packed.stride(2),
        block_m,
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
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack packed [3*H_global, B, S_local, D] → q,k,v [B, S_global, H_local, D].

    Called after _usp_all_to_all_single on the packed buffer.
    """
    S_global = S_local * world_size
    out_shape = (B, S_global, H_local, D)
    if out is None:
        q = torch.empty(out_shape, dtype=packed.dtype, device=packed.device)
        k = torch.empty(out_shape, dtype=packed.dtype, device=packed.device)
        v = torch.empty(out_shape, dtype=packed.dtype, device=packed.device)
    else:
        q, k, v = out
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.shape != out_shape:
                raise ValueError(
                    f"unpacked {name} output must have shape {out_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != packed.dtype or tensor.device != packed.device:
                raise ValueError(
                    f"unpacked {name} output dtype/device must match packed"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"unpacked {name} output must be contiguous")

    block_m = _QKV_PERMUTE_BLOCK_M
    grid = (triton.cdiv(B * world_size * S_local * H_local, block_m),)
    _unpack_qkv_kernel[grid](
        packed, q, k, v,
        B * world_size * S_local * H_local, S_local, H_local, D, world_size,
        packed.stride(0), packed.stride(1), packed.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        block_m,
        num_warps=4,
    )
    return q, k, v
