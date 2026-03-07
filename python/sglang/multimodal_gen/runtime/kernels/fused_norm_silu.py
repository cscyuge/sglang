"""
Triton fused L2Norm + SiLU kernel for VAE decoder.

Fuses WanRMS_norm (F.normalize * scale * gamma) + SiLU into a single kernel,
eliminating the intermediate global memory round-trip between the reduce and
pointwise phases that torch.compile generates.

Operation per spatial element:
    y[c] = silu(x[c] / ||x||_2 * sqrt(C) * gamma[c])

where silu(v) = v * sigmoid(v).

Supports both NCTHW (contiguous) and CL3D (channels_last_3d) memory formats.
CL3D is ~2x faster due to contiguous channel access (stride_c=1).
"""

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _fused_l2norm_silu_kernel(
        X_ptr,
        Gamma_ptr,
        Out_ptr,
        stride_c,
        stride_s,
        BLOCK_C: tl.constexpr,
        C: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """Fused L2Norm + SiLU, one spatial element per program.

        Args:
            stride_c: stride between channels = x.stride(1)
            stride_s: stride between spatial elements = x.stride(4)
        """
        m = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C

        # offset = m * stride_s + c * stride_c
        base = m * stride_s
        offsets = base + cols * stride_c
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(Gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # L2 norm + normalize + scale
        sq_sum = tl.sum(x * x, axis=0)
        inv_norm = tl.rsqrt(sq_sum + EPS)
        val = x * inv_norm * gamma

        # Optional SiLU
        if APPLY_SILU:
            val = val * tl.sigmoid(val)

        tl.store(Out_ptr + offsets, val.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _fused_l2norm_silu_kernel_fp16(
        X_ptr,
        Gamma_ptr,
        Out_ptr,
        stride_c,
        stride_s,
        BLOCK_C: tl.constexpr,
        C: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        EPS: tl.constexpr,
    ):
        m = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C

        base = m * stride_s
        offsets = base + cols * stride_c
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(Gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        sq_sum = tl.sum(x * x, axis=0)
        inv_norm = tl.rsqrt(sq_sum + EPS)
        val = x * inv_norm * gamma

        if APPLY_SILU:
            val = val * tl.sigmoid(val)

        tl.store(Out_ptr + offsets, val.to(tl.float16), mask=mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# Pre-computed gamma cache: maps id(norm_module) -> (gamma_scaled, module_ref)
_gamma_cache: dict = {}


def _get_gamma_scaled(norm_module):
    """Get or compute pre-scaled gamma (gamma * sqrt(C))."""
    key = id(norm_module)
    entry = _gamma_cache.get(key)
    if entry is not None and entry[1] is norm_module:
        return entry[0]
    gamma = norm_module.gamma.data.squeeze()  # (C,)
    gamma_scaled = (gamma * norm_module.scale).contiguous()
    _gamma_cache[key] = (gamma_scaled, norm_module)
    return gamma_scaled


def fused_rms_norm_silu(
    x: torch.Tensor,
    norm_module,
    apply_silu: bool = True,
) -> torch.Tensor:
    """
    Fused WanRMS_norm + optional SiLU, in-place on any 5D memory format.

    Supports both NCTHW (contiguous) and CL3D (channels_last_3d) inputs.
    CL3D is ~2x faster due to contiguous channel access.

    Args:
        x: (B, C, T, H, W) input tensor, any memory format
        norm_module: WanRMS_norm module (provides gamma and scale)
        apply_silu: whether to apply SiLU after normalization

    Returns:
        output: same shape and memory format as x
    """
    B, C, T_dim, H, W = x.shape

    # Kernel assumes spatial elements are compactly packed (m * stride_s
    # linearly indexes all M elements). This breaks on sliced views where
    # e.g. stride(2) = C*H_full*W > C*H_local*W. Ensure contiguity first.
    # Accept both NCTHW (contiguous) and CL3D (channels_last_3d) — only
    # force a copy for irregular strides (sliced views, etc.).
    if not x.is_contiguous() and not x.is_contiguous(
        memory_format=torch.channels_last_3d
    ):
        x = x.contiguous()

    THW = T_dim * H * W
    M = B * THW

    gamma_scaled = _get_gamma_scaled(norm_module)

    BLOCK_C = _next_pow2(C)
    EPS = 1e-12

    # Strides for accessing channels and spatial elements
    stride_c = x.stride(1)  # channel stride (T*H*W for NCTHW, 1 for CL3D)
    stride_s = x.stride(4)  # spatial stride (1 for NCTHW, C for CL3D)

    out = torch.empty_like(x)

    kernel = (
        _fused_l2norm_silu_kernel
        if x.dtype == torch.bfloat16
        else _fused_l2norm_silu_kernel_fp16
    )

    if B == 1:
        kernel[(M,)](
            x,
            gamma_scaled,
            out,
            stride_c,
            stride_s,
            BLOCK_C=BLOCK_C,
            C=C,
            APPLY_SILU=apply_silu,
            EPS=EPS,
        )
    else:
        # Multi-batch: launch per-batch to handle stride(0) correctly
        stride_b = x.stride(0)
        elem_size = x.element_size()
        for b in range(B):
            offset = b * stride_b * elem_size
            kernel[(THW,)](
                x.data_ptr() + offset,
                gamma_scaled,
                out.data_ptr() + offset,
                stride_c,
                stride_s,
                BLOCK_C=BLOCK_C,
                C=C,
                APPLY_SILU=apply_silu,
                EPS=EPS,
            )

    return out


def is_available() -> bool:
    """Check if Triton fused norm+SiLU is available."""
    return _HAS_TRITON
