"""
TileLang Conv3D implicit GEMM kernel for VAE decoder.

Reformulates 3D convolution as GEMM with on-the-fly im2col,
replacing cuDNN's SM80 fallback kernels on H100.

Only targets stride=1 3x3x3 convolutions (already padded input).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import tilelang
    import tilelang.language as T

    _HAS_TILELANG = True
except ImportError:
    _HAS_TILELANG = False


def conv3d_implicit_gemm(
    N_batch,
    C_in,
    D_in,
    H_in,
    W_in,
    C_out,
    KD,
    KH,
    KW,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=2,
    threads=128,
    dtype="bfloat16",
    accum_dtype="float32",
):
    """Build TileLang Conv3D program with flat 2D output."""
    OD = D_in - KD + 1
    OH = H_in - KH + 1
    OW = W_in - KW + 1
    M_total = N_batch * OD * OH * OW
    K_total = KD * KH * KW * C_in
    ODHW = OD * OH * OW
    OHW = OH * OW

    @T.prim_func
    def tl_conv3d(
        data: T.Tensor((N_batch, D_in, H_in, W_in, C_in), dtype),
        weight: T.Tensor((K_total, C_out), dtype),
        output: T.Tensor((M_total, C_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M_total, block_M),
            T.ceildiv(C_out, block_N),
            threads=threads,
        ) as (bm, bn):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            weight_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            T.clear(out_local)
            for k_iter in T.Pipelined(
                T.ceildiv(K_total, block_K), num_stages=num_stages
            ):
                for i, j in T.Parallel(block_M, block_K):
                    m = bm * block_M + i
                    k = k_iter * block_K + j
                    n_idx = m // ODHW
                    m_rem = m % ODHW
                    od = m_rem // OHW
                    oh = (m_rem % OHW) // OW
                    ow = m_rem % OW
                    c = k % C_in
                    k_rem = k // C_in
                    kw_idx = k_rem % KW
                    kh_idx = (k_rem // KW) % KH
                    kd_idx = k_rem // (KH * KW)
                    id_idx = od + kd_idx
                    ih_idx = oh + kh_idx
                    iw_idx = ow + kw_idx
                    in_bound = (
                        (m < M_total)
                        and (k < K_total)
                        and (id_idx < D_in)
                        and (ih_idx < H_in)
                        and (iw_idx < W_in)
                    )
                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[n_idx, id_idx, ih_idx, iw_idx, c],
                        T.cast(0, dtype),
                    )

                T.copy(
                    weight[k_iter * block_K, bn * block_N], weight_shared
                )
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, output[bm * block_M, bn * block_N])

    return tl_conv3d


def prepare_weight(weight_pytorch: torch.Tensor) -> torch.Tensor:
    """Convert Conv3d weight (C_out, C_in, KD, KH, KW) -> GEMM format (K_total, C_out)."""
    C_out, C_in, KD, KH, KW = weight_pytorch.shape
    w = weight_pytorch.permute(2, 3, 4, 1, 0).contiguous()
    return w.reshape(KD * KH * KW * C_in, C_out)


# Optimal tile configs from sweep on H100
_BEST_TILES = {
    (16, 384): (64, 64, 64),
    (96, 96): (64, 96, 32),
    (192, 192): (64, 64, 32),
    (384, 384): (64, 96, 64),
}

# Channel pairs we accelerate (3x3x3, stride=1 only)
TARGET_CHANNEL_PAIRS = frozenset(_BEST_TILES.keys())


def _pick_tiles(C_in, C_out):
    """Return (block_M, block_N, block_K) for given channel pair."""
    return _BEST_TILES.get((C_in, C_out), (128, 128, 64))


_kernel_cache: dict = {}
_weight_cache: dict = {}  # Maps (id, shape) -> (gemm_weight, original_weight_ref)


def compile_kernel(N_batch, C_in, D_in, H_in, W_in, C_out, KD, KH, KW):
    """Compile and cache a TileLang Conv3D kernel for the given shape."""
    OD = D_in - KD + 1
    OH = H_in - KH + 1
    OW = W_in - KW + 1
    M_total = N_batch * OD * OH * OW

    block_M, block_N, block_K = _pick_tiles(C_in, C_out)
    dtype_str = "bfloat16"
    cache_key = (N_batch, C_in, D_in, H_in, W_in, C_out, KD, KH, KW)

    if cache_key not in _kernel_cache:
        kernel_fn = conv3d_implicit_gemm(
            N_batch=N_batch,
            C_in=C_in,
            D_in=D_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            KD=KD,
            KH=KH,
            KW=KW,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=2,
            threads=128,
            dtype=dtype_str,
            accum_dtype="float32",
        )
        compiled = tilelang.compile(kernel_fn, out_idx=[2])
        compiled._out_shape = (N_batch, OD, OH, OW, C_out)
        _kernel_cache[cache_key] = compiled

    return _kernel_cache[cache_key]


def get_or_prepare_weight(weight: torch.Tensor) -> torch.Tensor:
    """Get or create GEMM-format weight. Uses data_ptr + shape as key for robustness."""
    key = (weight.data_ptr(), weight.shape)
    entry = _weight_cache.get(key)
    if entry is None:
        gemm_w = prepare_weight(weight.data.contiguous())
        # Store reference to original weight to prevent GC / data_ptr reuse
        _weight_cache[key] = (gemm_w, weight)
        return gemm_w
    return entry[0]


def tilelang_conv3d_forward(
    x_padded: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Run TileLang Conv3D on already-padded NCDHW input.

    Args:
        x_padded: (B, C_in, D, H, W) already padded, NCDHW format
        weight: Conv3d weight parameter (C_out, C_in, KD, KH, KW)
        bias: (C_out,) or None

    Returns:
        output: (B, C_out, OD, OH, OW) in NCDHW format, contiguous
    """
    B, C_in, D, H, W = x_padded.shape
    C_out = weight.shape[0]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]

    kernel = compile_kernel(B, C_in, D, H, W, C_out, KD, KH, KW)
    w_gemm = get_or_prepare_weight(weight)

    # NCDHW -> NDHWC
    x_ndhwc = x_padded.permute(0, 2, 3, 4, 1).contiguous()
    out_flat = kernel(x_ndhwc, w_gemm)

    # Reshape flat output
    out_ndhwc = out_flat.view(kernel._out_shape)

    if bias is not None:
        out_ndhwc = out_ndhwc + bias.view(1, 1, 1, 1, -1)

    # NDHWC -> NCDHW (channels_last_3d view, no copy)
    out = out_ndhwc.permute(0, 4, 1, 2, 3)
    return out


def is_available() -> bool:
    """Check if TileLang Conv3D is available."""
    return _HAS_TILELANG
