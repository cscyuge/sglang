"""
TileLang Conv2D implicit GEMM kernel for VAE decoder upsampler.

Reformulates 2D convolution as GEMM with on-the-fly im2col,
replacing cuDNN Conv2D kernels (and their nchwToNhwc/nhwcToNchw overhead)
in the VAE decoder's upsampling path.

Only targets stride=1 3x3 convolutions (already padded input).
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


def conv2d_implicit_gemm(
    N_batch,
    C_in,
    H_in,
    W_in,
    C_out,
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
    """Build TileLang Conv2D program with flat 2D output."""
    OH = H_in - KH + 1
    OW = W_in - KW + 1
    M_total = N_batch * OH * OW
    K_total = KH * KW * C_in
    OHW = OH * OW

    @T.prim_func
    def tl_conv2d(
        data: T.Tensor((N_batch, H_in, W_in, C_in), dtype),
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
                    n_idx = m // OHW
                    m_rem = m % OHW
                    oh = m_rem // OW
                    ow = m_rem % OW
                    c = k % C_in
                    k_rem = k // C_in
                    kw_idx = k_rem % KW
                    kh_idx = k_rem // KW
                    ih_idx = oh + kh_idx
                    iw_idx = ow + kw_idx
                    in_bound = (
                        (m < M_total)
                        and (k < K_total)
                        and (ih_idx < H_in)
                        and (iw_idx < W_in)
                    )
                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[n_idx, ih_idx, iw_idx, c],
                        T.cast(0, dtype),
                    )

                T.copy(
                    weight[k_iter * block_K, bn * block_N], weight_shared
                )
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, output[bm * block_M, bn * block_N])

    return tl_conv2d


def prepare_weight_2d(weight_pytorch: torch.Tensor) -> torch.Tensor:
    """Convert Conv2d weight (C_out, C_in, KH, KW) -> GEMM format (K_total, C_out)."""
    C_out, C_in, KH, KW = weight_pytorch.shape
    w = weight_pytorch.permute(2, 3, 1, 0).contiguous()
    return w.reshape(KH * KW * C_in, C_out)


# Optimal tile configs from sweep on H100
_BEST_TILES_2D = {
    (384, 192): (128, 64, 64),
    (192, 96): (128, 96, 32),
}

# Channel pairs we accelerate (3x3, stride=1 only)
TARGET_CHANNEL_PAIRS_2D = frozenset(_BEST_TILES_2D.keys())


def _pick_tiles_2d(C_in, C_out):
    """Return (block_M, block_N, block_K) for given channel pair."""
    return _BEST_TILES_2D.get((C_in, C_out), (128, 128, 64))


_kernel_cache_2d: dict = {}
_weight_cache_2d: dict = {}


def compile_kernel_2d(N_batch, C_in, H_in, W_in, C_out, KH, KW):
    """Compile and cache a TileLang Conv2D kernel for the given shape."""
    OH = H_in - KH + 1
    OW = W_in - KW + 1

    block_M, block_N, block_K = _pick_tiles_2d(C_in, C_out)
    dtype_str = "bfloat16"
    cache_key = (N_batch, C_in, H_in, W_in, C_out, KH, KW)

    if cache_key not in _kernel_cache_2d:
        kernel_fn = conv2d_implicit_gemm(
            N_batch=N_batch,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
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
        compiled._out_shape = (N_batch, OH, OW, C_out)
        _kernel_cache_2d[cache_key] = compiled

    return _kernel_cache_2d[cache_key]


def get_or_prepare_weight_2d(weight: torch.Tensor) -> torch.Tensor:
    """Get or create GEMM-format weight. Uses data_ptr + shape as key for robustness."""
    key = (weight.data_ptr(), weight.shape)
    entry = _weight_cache_2d.get(key)
    if entry is None:
        gemm_w = prepare_weight_2d(weight.data.contiguous())
        _weight_cache_2d[key] = (gemm_w, weight)
        return gemm_w
    return entry[0]


def tilelang_conv2d_forward(
    x_padded: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Run TileLang Conv2D on already-padded NCHW input.

    Args:
        x_padded: (B, C_in, H, W) already padded, NCHW format
        weight: Conv2d weight parameter (C_out, C_in, KH, KW)
        bias: (C_out,) or None

    Returns:
        output: (B, C_out, OH, OW) in channels_last format
    """
    B, C_in, H, W = x_padded.shape
    C_out = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]

    kernel = compile_kernel_2d(B, C_in, H, W, C_out, KH, KW)
    w_gemm = get_or_prepare_weight_2d(weight)

    # NCHW -> NHWC (free if input is already channels_last)
    x_nhwc = x_padded.permute(0, 2, 3, 1).contiguous()
    out_flat = kernel(x_nhwc, w_gemm)

    # Reshape flat output
    out_nhwc = out_flat.view(kernel._out_shape)

    if bias is not None:
        out_nhwc = out_nhwc + bias.view(1, 1, 1, -1)

    # NHWC -> NCHW as a channels_last view (no data copy).
    # The permuted view has shape (B, C, OH, OW) with NHWC strides,
    # which IS channels_last format. Downstream ops work correctly on
    # channels_last tensors, and the next tl_conv2d call's input
    # permute(0,2,3,1).contiguous() becomes a no-op because the memory
    # is already NHWC-contiguous.
    out = out_nhwc.permute(0, 3, 1, 2)
    return out


def is_available() -> bool:
    """Check if TileLang Conv2D is available."""
    return _HAS_TILELANG
