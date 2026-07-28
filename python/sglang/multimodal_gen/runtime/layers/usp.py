# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import inspect
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.distributed.tensor.experimental._attention import _cp_options

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.srt.utils.common import torch_release

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)
_USP_BUFFER_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _env_enabled(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    return value.lower() not in ("0", "false", "no", "off")


def _usp_reuse_buffers_enabled() -> bool:
    return _env_enabled(
        "SGLANG_USP_REUSE_BUFFERS",
        os.getenv("SGLANG_STREAM_R1_REUSE_PACKED_BUFFERS", "1"),
    )


def _usp_explicit_all_to_all_enabled() -> bool:
    return _env_enabled("SGLANG_USP_EXPLICIT_ALL_TO_ALL", "1")


def _usp_fp8_comm_scope() -> str:
    raw = os.getenv("SGLANG_STREAM_R1_SP_COMM_FP8")
    if raw is None:
        raw = os.getenv("SGLANG_STREAM_R1_SP_COMM_FP8_SCOPE", "both")
    value = raw.strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return ""
    if value in ("output", "v_only", "qkv", "both"):
        return value
    if value in ("1", "true", "yes", "y", "on"):
        return os.getenv("SGLANG_STREAM_R1_SP_COMM_FP8_SCOPE", "both").lower()
    return os.getenv("SGLANG_STREAM_R1_SP_COMM_FP8_SCOPE", "both").lower()


def _usp_fp8_comm_enabled(target: str) -> bool:
    scope = _usp_fp8_comm_scope()
    if target == "output":
        return scope in ("output", "both")
    if target == "qkv":
        return scope in ("qkv", "both")
    if target == "v_only":
        return scope == "v_only"
    return False


def _usp_fp8_comm_block_size() -> int:
    value = os.getenv("SGLANG_STREAM_R1_SP_COMM_FP8_BLOCK_SIZE", "128")
    try:
        block_size = int(value)
    except ValueError:
        block_size = 128
    return block_size if block_size in (16, 32, 64, 128) else 128


def _usp_fp8_comm_rowpack_enabled() -> bool:
    return _env_enabled("SGLANG_STREAM_R1_SP_COMM_FP8_ROWPACK", "1")


def _usp_fp8_comm_fused_rowpack_enabled() -> bool:
    return _env_enabled("SGLANG_STREAM_R1_SP_COMM_FP8_FUSED_ROWPACK", "0")


def _usp_device_cache_key(device: torch.device) -> tuple[str, int]:
    index = device.index
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return device.type, -1 if index is None else int(index)


def _usp_get_buffer(
    name: str,
    like: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if like.device.type != "cuda":
        return None
    if not _usp_reuse_buffers_enabled():
        return None
    buffer_dtype = like.dtype if dtype is None else dtype
    stream_key = int(torch.cuda.current_stream(like.device).cuda_stream)
    cache_key = (
        name,
        *_usp_device_cache_key(like.device),
        stream_key,
        buffer_dtype,
        shape,
    )
    buffer = _USP_BUFFER_CACHE.get(cache_key)
    if buffer is None:
        # Reused USP buffers are updated with copy_/out= writes across calls.
        # Allocate them as normal tensors even when the caller is under
        # torch.inference_mode(), otherwise PyTorch rejects later inplace writes.
        with torch.inference_mode(False):
            buffer = torch.empty(shape, dtype=buffer_dtype, device=like.device)
        _USP_BUFFER_CACHE[cache_key] = buffer
    return buffer


def _usp_permute_contiguous(
    x: torch.Tensor,
    order: tuple[int, ...],
    *,
    cache_name: str,
) -> torch.Tensor:
    permuted = x.permute(order)
    out = _usp_get_buffer(cache_name, x, tuple(permuted.shape))
    if out is None:
        return permuted.contiguous()
    out.copy_(permuted)
    return out


def _comm_nvtx_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_COMM_NVTX", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _comm_nvtx_caller() -> str:
    frame = inspect.currentframe()
    if frame is None:
        return "caller=unknown"
    frame = frame.f_back
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.endswith("usp.py"):
            return (
                f"caller={os.path.basename(filename)}:"
                f"{frame.f_lineno}:{frame.f_code.co_name}"
            )
        frame = frame.f_back
    return "caller=unknown"


@contextmanager
def _comm_nvtx_range(message: str):
    if not _comm_nvtx_enabled() or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _tensor_desc(name: str, tensor: torch.Tensor) -> str:
    return f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype}"


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _usp_all_to_all_single(
    x: torch.Tensor,
    *,
    cache_name: str = "usp_all_to_all",
) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    x_shape = x.shape
    with _comm_nvtx_range(
        "sgl_mm_usp_all_to_all_single "
        f"shape={tuple(x_shape)} dtype={x.dtype} {_comm_nvtx_caller()}"
    ):
        if not x.is_contiguous():
            x = x.contiguous()
        if _usp_explicit_all_to_all_enabled() and x.device.type == "cuda":
            out = _usp_get_buffer(f"{cache_name}.out", x, tuple(x_shape))
            if out is None:
                out = torch.empty_like(x)
            dist.all_to_all_single(out, x, group=ulysses_pg)
            x = out
        else:
            flat = x.flatten()
            x = ft_c.all_to_all_single(
                flat, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
            )
            x = _maybe_wait(x)
        x = x.reshape(x_shape)
    return x


def _usp_blockwise_fp8_all_to_all_single(
    x: torch.Tensor,
    *,
    cache_name: str,
) -> torch.Tensor:
    fp8_result = _usp_blockwise_fp8_all_to_all_payload_scale(
        x,
        cache_name=cache_name,
    )
    if fp8_result is None:
        return _usp_all_to_all_single(x, cache_name=cache_name)

    x_q, x_scale, group_size = fp8_result
    from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import blockwise_dequant_fp8

    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_dequant "
        f"group_size={group_size} shape={tuple(x.shape)} {_comm_nvtx_caller()}"
    ):
        out = _usp_get_buffer(
            f"{cache_name}.fp8_dequant",
            x,
            tuple(x.shape),
            dtype=x.dtype,
        )
        return blockwise_dequant_fp8(
            x_q,
            x_scale,
            group_size=group_size,
            dtype=x.dtype,
            out=out,
        )


def _usp_blockwise_fp8_all_to_all_payload_scale(
    x: torch.Tensor,
    *,
    cache_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous():
        return None
    group_size = _usp_fp8_comm_block_size()
    if x.shape[-1] % group_size != 0:
        return None

    from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
        blockwise_quant_fp8,
    )

    scale_shape = tuple(x.shape[:-1]) + (x.shape[-1] // group_size,)
    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_quant "
        f"group_size={group_size} {_tensor_desc('x', x)} {_comm_nvtx_caller()}"
    ):
        q_out = _usp_get_buffer(
            f"{cache_name}.fp8_quant",
            x,
            tuple(x.shape),
            dtype=torch.float8_e4m3fn,
        )
        scale_out = _usp_get_buffer(
            f"{cache_name}.fp8_scale_quant",
            x,
            scale_shape,
            dtype=torch.float32,
        )
        x_q, x_scale = blockwise_quant_fp8(
            x,
            group_size=group_size,
            out_q=q_out,
            out_scale=scale_out,
        )

    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_all_to_all "
        f"group_size={group_size} shape={tuple(x.shape)} {_comm_nvtx_caller()}"
    ):
        x_q_bytes = _usp_all_to_all_single(
            x_q.view(torch.uint8),
            cache_name=f"{cache_name}.fp8_payload",
        )
        x_scale = _usp_all_to_all_single(
            x_scale,
            cache_name=f"{cache_name}.fp8_scale",
        )
        x_q = x_q_bytes.view(torch.float8_e4m3fn)

    return x_q, x_scale, group_size


def _usp_blockwise_fp8_all_to_all_rowpack_payload_scale(
    x: torch.Tensor,
    *,
    cache_name: str,
) -> tuple[torch.Tensor, int] | None:
    if not _usp_fp8_comm_rowpack_enabled():
        return None
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous():
        return None
    group_size = _usp_fp8_comm_block_size()
    if x.shape[-1] % group_size != 0 or x.shape[-1] % 4 != 0:
        return None
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1 or x.shape[0] % world_size != 0:
        return None

    from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
        aligned_rowpack_shape,
        blockwise_quant_fp8_rowpack,
        blockwise_quant_fp8,
        pack_fp8_payload_scale_aligned,
    )

    scale_shape = tuple(x.shape[:-1]) + (x.shape[-1] // group_size,)
    rowpack_shape = aligned_rowpack_shape(
        tuple(x.shape),
        scale_cols=scale_shape[-1],
        world_size=world_size,
    )
    if _usp_fp8_comm_fused_rowpack_enabled():
        with _comm_nvtx_range(
            "sgl_mm_usp_fp8_comm_fused_quant_pack_rowpack "
            f"group_size={group_size} payload_shape={tuple(x.shape)} "
            f"rowpack_shape={rowpack_shape}"
        ):
            rowpack = _usp_get_buffer(
                f"{cache_name}.fp8_rowpack_pack",
                x,
                rowpack_shape,
                dtype=torch.uint8,
            )
            rowpack = blockwise_quant_fp8_rowpack(
                x,
                group_size=group_size,
                world_size=world_size,
                out=rowpack,
            )
        with _comm_nvtx_range(
            "sgl_mm_usp_fp8_comm_all_to_all_rowpack "
            f"group_size={group_size} shape={tuple(rowpack.shape)} "
            f"{_comm_nvtx_caller()}"
        ):
            rowpack = _usp_all_to_all_single(
                rowpack,
                cache_name=f"{cache_name}.fp8_rowpack",
            )
        return rowpack, group_size

    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_quant "
        f"group_size={group_size} {_tensor_desc('x', x)} {_comm_nvtx_caller()}"
    ):
        q_out = _usp_get_buffer(
            f"{cache_name}.fp8_quant",
            x,
            tuple(x.shape),
            dtype=torch.float8_e4m3fn,
        )
        scale_out = _usp_get_buffer(
            f"{cache_name}.fp8_scale_quant",
            x,
            scale_shape,
            dtype=torch.float32,
        )
        x_q, x_scale = blockwise_quant_fp8(
            x,
            group_size=group_size,
            out_q=q_out,
            out_scale=scale_out,
        )

    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_pack_payload_scale_rowpack "
        f"group_size={group_size} payload_shape={tuple(x_q.shape)} "
        f"rowpack_shape={rowpack_shape}"
    ):
        rowpack = _usp_get_buffer(
            f"{cache_name}.fp8_rowpack_pack",
            x,
            rowpack_shape,
            dtype=torch.uint8,
        )
        rowpack = pack_fp8_payload_scale_aligned(
            x_q,
            x_scale,
            world_size=world_size,
            out=rowpack,
        )

    with _comm_nvtx_range(
        "sgl_mm_usp_fp8_comm_all_to_all_rowpack "
        f"group_size={group_size} shape={tuple(rowpack.shape)} {_comm_nvtx_caller()}"
    ):
        rowpack = _usp_all_to_all_single(
            rowpack,
            cache_name=f"{cache_name}.fp8_rowpack",
        )
    return rowpack, group_size


def _usp_input_all_to_all(
    x: torch.Tensor,
    head_dim: int = 1,
    *,
    fp8_comm: bool = False,
) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h_local, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h_local, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (h_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_global, s_local, d = x.shape
        # Shape transition: [b, h_global, s_local, d] -> [h_global, b, s_local, d]
        permute_order = (1, 0, 2, 3)
    else:  # head_dim == 2
        b, s_local, h_global, d = x.shape
        # Shape transition: [b, s_local, h_global, d] -> [h_global, b, s_local, d]
        permute_order = (2, 0, 1, 3)

    assert (
        h_global % world_size == 0
    ), f"h_global ({h_global}) must be divisible by world_size ({world_size})"

    h_local, s_global = h_global // world_size, s_local * world_size

    with _comm_nvtx_range(
        "sgl_mm_usp_input_prepack "
        f"head_dim={head_dim} {_tensor_desc('x', x)} {_comm_nvtx_caller()}"
    ):
        x = _usp_permute_contiguous(
            x,
            permute_order,
            cache_name=f"usp_input_prepack.h{head_dim}",
        )
    if fp8_comm:
        x = _usp_blockwise_fp8_all_to_all_single(x, cache_name="usp_input_fp8")
    else:
        x = _usp_all_to_all_single(x, cache_name="usp_input")
    x = x.reshape(world_size, h_local, b, s_local, d)

    # Reorder dims to place 'world_size' adjacent to 's_local' to merge them into 's_global'
    if head_dim == 1:
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, h_local, world_size, s_local, d]
        with _comm_nvtx_range(
            "sgl_mm_usp_input_postunpack "
            f"head_dim={head_dim} world_size={world_size}"
        ):
            x = _usp_permute_contiguous(
                x,
                (2, 1, 0, 3, 4),
                cache_name=f"usp_input_postunpack.h{head_dim}",
            ).reshape(b, h_local, s_global, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, h_local, b, s_local, d] -> [b, world_size, s_local, h_local, d]
        with _comm_nvtx_range(
            "sgl_mm_usp_input_postunpack "
            f"head_dim={head_dim} world_size={world_size}"
        ):
            x = _usp_permute_contiguous(
                x,
                (2, 0, 3, 1, 4),
                cache_name=f"usp_input_postunpack.h{head_dim}",
            ).reshape(b, s_global, h_local, d)

    return x


def _usp_input_all_to_all_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Ulysses input all-to-all for QKV (head_dim=2 layout).

    Fuses 3 separate all-to-all calls into 1 by interleaving QKV heads
    in a single packed buffer. Reduces 6 copy kernels + 3 NCCL calls
    to 2 Triton kernels + 1 NCCL call.

    Only supports the head_dim=2 layout: q, k, v are [B, S_local, H, D].
    Falls back to 3 separate calls for GQA (num_kv_heads < num_heads).
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    # GQA guard: q and k/v must have the same number of heads
    if q.shape != k.shape or q.shape != v.shape:
        q = _usp_input_all_to_all(q, head_dim=2)
        k = _usp_input_all_to_all(k, head_dim=2)
        v = _usp_input_all_to_all(v, head_dim=2)
        return q, k, v

    if _usp_fp8_comm_enabled("v_only"):
        q = _usp_input_all_to_all(q, head_dim=2)
        k = _usp_input_all_to_all(k, head_dim=2)
        v = _usp_input_all_to_all(v, head_dim=2, fp8_comm=True)
        return q, k, v

    B, S_local, H_global, D = q.shape
    assert (
        H_global % world_size == 0
    ), f"H_global ({H_global}) must be divisible by world_size ({world_size})"
    H_local = H_global // world_size

    if (
        _usp_fp8_comm_enabled("qkv")
        and _usp_fp8_comm_rowpack_enabled()
        and _usp_fp8_comm_fused_rowpack_enabled()
    ):
        from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
            aligned_rowpack_shape,
            blockwise_quant_qkv_fp8_rowpack,
            fused_dequant_unpack_qkv_fp8_rowpack,
        )

        group_size = _usp_fp8_comm_block_size()
        payload_shape = (3 * H_global, B, S_local, D)
        rowpack_shape = aligned_rowpack_shape(
            payload_shape,
            scale_cols=D // group_size,
            world_size=world_size,
        )
        with _comm_nvtx_range(
            "sgl_mm_usp_qkv_fused_quant_pack_rowpack "
            f"world_size={world_size} group_size={group_size} "
            f"q={tuple(q.shape)} rowpack={rowpack_shape}"
        ):
            packed_rowpack = _usp_get_buffer(
                "usp_qkv_fp8_rowpack.fp8_rowpack_pack",
                q,
                rowpack_shape,
                dtype=torch.uint8,
            )
            packed_rowpack = blockwise_quant_qkv_fp8_rowpack(
                q,
                k,
                v,
                group_size=group_size,
                world_size=world_size,
                out=packed_rowpack,
            )
        packed_rowpack = _usp_all_to_all_single(
            packed_rowpack,
            cache_name="usp_qkv_fp8_rowpack.fp8_rowpack",
        )
        qkv_out_buffer = _usp_get_buffer(
            "usp_qkv_unpack",
            q,
            (3, B, S_local * world_size, H_local, D),
        )
        qkv_out = None
        if qkv_out_buffer is not None:
            qkv_out = (
                qkv_out_buffer[0],
                qkv_out_buffer[1],
                qkv_out_buffer[2],
            )
        return fused_dequant_unpack_qkv_fp8_rowpack(
            packed_rowpack,
            B,
            S_local,
            H_local,
            D,
            world_size,
            group_size=group_size,
            dtype=q.dtype,
            out=qkv_out,
        )

    from sglang.jit_kernel.diffusion.triton.usp_permute import (
        fused_pack_qkv_for_all_to_all,
        fused_unpack_qkv_from_all_to_all,
    )

    # 1. Fused pack: q,k,v [B,S,H,D] → packed [3*H, B, S, D]
    with _comm_nvtx_range(
        "sgl_mm_usp_qkv_pack "
        f"world_size={world_size} {_tensor_desc('q', q)} {_comm_nvtx_caller()}"
    ):
        packed_out = _usp_get_buffer(
            "usp_qkv_pack",
            q,
            (3 * H_global, B, S_local, D),
        )
        packed = fused_pack_qkv_for_all_to_all(q, k, v, out=packed_out)

    # 2. Single NCCL all-to-all. When enabled, send blockwise FP8 payload
    # plus FP32 scales and fuse dequantization with the QKV unpack.
    if _usp_fp8_comm_enabled("qkv"):
        rowpack_result = _usp_blockwise_fp8_all_to_all_rowpack_payload_scale(
            packed,
            cache_name="usp_qkv_fp8_rowpack",
        )
        if rowpack_result is not None:
            packed_rowpack, group_size = rowpack_result
            from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
                fused_dequant_unpack_qkv_fp8_rowpack,
            )

            with _comm_nvtx_range(
                "sgl_mm_usp_qkv_rowpack_fused_dequant_unpack "
                f"world_size={world_size} group_size={group_size} "
                f"packed={tuple(packed_rowpack.shape)}"
            ):
                qkv_out_buffer = _usp_get_buffer(
                    "usp_qkv_unpack",
                    q,
                    (3, B, S_local * world_size, H_local, D),
                )
                qkv_out = None
                if qkv_out_buffer is not None:
                    qkv_out = (
                        qkv_out_buffer[0],
                        qkv_out_buffer[1],
                        qkv_out_buffer[2],
                    )
                return fused_dequant_unpack_qkv_fp8_rowpack(
                    packed_rowpack,
                    B,
                    S_local,
                    H_local,
                    D,
                    world_size,
                    group_size=group_size,
                    dtype=q.dtype,
                    out=qkv_out,
                )

        fp8_result = _usp_blockwise_fp8_all_to_all_payload_scale(
            packed,
            cache_name="usp_qkv_fp8",
        )
        if fp8_result is not None:
            packed_q, packed_scale, group_size = fp8_result
            from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
                fused_dequant_unpack_qkv_fp8,
            )

            with _comm_nvtx_range(
                "sgl_mm_usp_qkv_fused_dequant_unpack "
                f"world_size={world_size} group_size={group_size} "
                f"packed={tuple(packed_q.shape)}"
            ):
                qkv_out_buffer = _usp_get_buffer(
                    "usp_qkv_unpack",
                    q,
                    (3, B, S_local * world_size, H_local, D),
                )
                qkv_out = None
                if qkv_out_buffer is not None:
                    qkv_out = (
                        qkv_out_buffer[0],
                        qkv_out_buffer[1],
                        qkv_out_buffer[2],
                    )
                return fused_dequant_unpack_qkv_fp8(
                    packed_q,
                    packed_scale,
                    B,
                    S_local,
                    H_local,
                    D,
                    world_size,
                    group_size=group_size,
                    dtype=q.dtype,
                    out=qkv_out,
                )
        packed = _usp_all_to_all_single(packed, cache_name="usp_qkv")
    else:
        packed = _usp_all_to_all_single(packed, cache_name="usp_qkv")

    # 3. Fused unpack: packed [3*H, B, S_local, D] → q,k,v [B, S_global, H_local, D]
    with _comm_nvtx_range(
        "sgl_mm_usp_qkv_unpack "
        f"world_size={world_size} {_tensor_desc('packed', packed)}"
    ):
        qkv_out_buffer = _usp_get_buffer(
            "usp_qkv_unpack",
            packed,
            (3, B, S_local * world_size, H_local, D),
        )
        qkv_out = None
        if qkv_out_buffer is not None:
            qkv_out = (qkv_out_buffer[0], qkv_out_buffer[1], qkv_out_buffer[2])
        q, k, v = fused_unpack_qkv_from_all_to_all(
            packed, B, S_local, H_local, D, world_size, out=qkv_out
        )

    return q, k, v


def _usp_output_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input).

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h_local, s, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"

    # Move the dimension to be split (s_global) to dim 0 for all_to_all_single
    if head_dim == 1:
        b, h_local, s_global, d = x.shape
        # Shape transition: [b, h_local, s_global, d] -> [s_global, b, h_local, d]
        permute_order = (2, 0, 1, 3)
    else:  # head_dim == 2
        b, s_global, h_local, d = x.shape
        # Shape transition: [b, s_global, h_local, d] -> [s_global, b, h_local, d]
        permute_order = (1, 0, 2, 3)

    assert (
        s_global % world_size == 0
    ), f"s_global ({s_global}) must be divisible by world_size ({world_size})"

    s_local, h_global = s_global // world_size, h_local * world_size

    with _comm_nvtx_range(
        "sgl_mm_usp_output_prepack "
        f"head_dim={head_dim} {_tensor_desc('x', x)} {_comm_nvtx_caller()}"
    ):
        x = _usp_permute_contiguous(
            x,
            permute_order,
            cache_name=f"usp_output_prepack.h{head_dim}",
        )
    x = _usp_all_to_all_single(x, cache_name="usp_output")
    x = x.reshape(world_size, s_local, b, h_local, d)

    # Reorder dims to place 'world_size' adjacent to 'h_local' to merge them into 'h_global'
    if head_dim == 1:
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, world_size, h_local, s_local, d]
        with _comm_nvtx_range(
            "sgl_mm_usp_output_postunpack "
            f"head_dim={head_dim} world_size={world_size}"
        ):
            x = _usp_permute_contiguous(
                x,
                (2, 0, 3, 1, 4),
                cache_name=f"usp_output_postunpack.h{head_dim}",
            ).reshape(b, h_global, s_local, d)
    else:  # head_dim == 2
        # Shape transition: [world_size, s_local, b, h_local, d] -> [b, s_local, world_size, h_local, d]
        with _comm_nvtx_range(
            "sgl_mm_usp_output_postunpack "
            f"head_dim={head_dim} world_size={world_size}"
        ):
            x = _usp_permute_contiguous(
                x,
                (2, 1, 0, 3, 4),
                cache_name=f"usp_output_postunpack.h{head_dim}",
            ).reshape(b, s_local, h_global, d)

    return x


def _usp_output_all_to_all_packed_bshd(
    packed: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Output all-to-all from packed BSHD order for the common batch=1 path.

    ``packed`` is the varlen attention output in flattened ``[B*S, H_local, D]``
    order. For Wan S2V realtime serving B is 1, so this tensor can be viewed as
    ``[S, B, H_local, D]`` and sent directly to NCCL, skipping the generic
    output prepack copy from ``[B, S, H, D]``.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return packed.reshape(batch_size, seq_len, packed.shape[-2], packed.shape[-1])

    if packed.ndim != 3:
        raise ValueError(
            f"packed output must have shape [B*S, H, D], got {tuple(packed.shape)}"
        )
    if batch_size != 1:
        return _usp_output_all_to_all(
            packed.reshape(batch_size, seq_len, packed.shape[-2], packed.shape[-1]),
            head_dim=2,
        )
    if packed.shape[0] != seq_len:
        raise ValueError(
            "packed output token count must match batch_size * seq_len: "
            f"packed={packed.shape[0]} batch_size={batch_size} seq_len={seq_len}"
        )
    if seq_len % world_size != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by world_size ({world_size})"
        )

    if not packed.is_contiguous():
        packed = packed.contiguous()

    h_local = packed.shape[1]
    d = packed.shape[2]
    s_local = seq_len // world_size
    h_global = h_local * world_size

    with _comm_nvtx_range(
        "sgl_mm_usp_output_prepack_packed_view "
        f"seq_len={seq_len} h_local={h_local} dtype={packed.dtype} "
        f"{_comm_nvtx_caller()}"
    ):
        x = packed.reshape(seq_len, batch_size, h_local, d)

    if _usp_fp8_comm_enabled("output"):
        rowpack_result = _usp_blockwise_fp8_all_to_all_rowpack_payload_scale(
            x,
            cache_name="usp_output_packed_fp8_rowpack",
        )
        if rowpack_result is not None:
            packed_rowpack, group_size = rowpack_result
            from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
                fused_dequant_unpack_output_fp8_rowpack,
            )

            with _comm_nvtx_range(
                "sgl_mm_usp_output_rowpack_fused_dequant_postunpack "
                f"world_size={world_size} group_size={group_size} "
                f"shape={tuple(packed_rowpack.shape)}"
            ):
                out = _usp_get_buffer(
                    "usp_output_postunpack.packed_bshd",
                    x,
                    (batch_size, s_local, h_global, d),
                    dtype=packed.dtype,
                )
                return fused_dequant_unpack_output_fp8_rowpack(
                    packed_rowpack,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    world_size=world_size,
                    group_size=group_size,
                    dtype=packed.dtype,
                    out=out,
                )

        fp8_result = _usp_blockwise_fp8_all_to_all_payload_scale(
            x,
            cache_name="usp_output_packed_fp8",
        )
        if fp8_result is not None:
            packed_q, packed_scale, group_size = fp8_result
            from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
                fused_dequant_unpack_output_fp8,
            )

            with _comm_nvtx_range(
                "sgl_mm_usp_output_fused_dequant_postunpack "
                f"world_size={world_size} group_size={group_size} "
                f"shape={tuple(packed_q.shape)}"
            ):
                out = _usp_get_buffer(
                    "usp_output_postunpack.packed_bshd",
                    x,
                    (batch_size, s_local, h_global, d),
                    dtype=packed.dtype,
                )
                return fused_dequant_unpack_output_fp8(
                    packed_q,
                    packed_scale,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    world_size=world_size,
                    group_size=group_size,
                    dtype=packed.dtype,
                    out=out,
                )
        x = _usp_all_to_all_single(x, cache_name="usp_output_packed")
    else:
        x = _usp_all_to_all_single(x, cache_name="usp_output_packed")
    x = x.reshape(world_size, s_local, batch_size, h_local, d)

    with _comm_nvtx_range(
        "sgl_mm_usp_output_postunpack "
        f"head_dim=2 world_size={world_size} packed_bshd=True"
    ):
        x = _usp_permute_contiguous(
            x,
            (2, 1, 0, 3, 4),
            cache_name="usp_output_postunpack.packed_bshd",
        ).reshape(batch_size, s_local, h_global, d)

    return x


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """
    Ring Attention implementation.

    This function implements Ring Attention, a strategy for distributed attention
    computation that reduces peak memory usage. It accepts a generic attention
    implementation (`attn_impl`) which is called by the underlying PyTorch
    distributed attention primitive.

    Args:
        query, key, value: The input tensors for attention.
        attn_impl: An instance of an attention implementation backend
                   (e.g., FlashAttentionImpl) whose `forward` method will be
                   used as the computational kernel.
        is_causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
    """
    # torch.distributed.tensor.experimental._attention is not a public API,
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.

    # Note: Please be aware that Attention Backend and Ring Attention may require different QKV tensor shapes.
    # For example, FlashAttention expects the format to be BSHD.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        # logger.warning(f"Warning: return_softmax_lse is only supported for FlashAttentionImpl")
        output, softmax_lse, *rest = attn_impl.forward(
            q,
            k,
            v,
            attn_metadata=None,
            return_softmax_lse=True,
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = torch_release >= (2, 6)

    attn_kwargs = dict(
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
        group=ring_pg,  # https://github.com/pytorch/pytorch/blob/c907c778f42ba2fdaf25b733dd25baf9779c6a12/torch/distributed/tensor/experimental/_context_parallel/_attention.py#L309
    )

    if use_segment_id:
        # For torch >= 2.6, segment_id is required. The value '1' is a placeholder
        # as we are not using complex segmentation features.
        out, *_ = _templated_ring_attention(
            seq_dim=1,  # segment_id
            **attn_kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            **attn_kwargs,
        )

    # Permute the output back to [B, S, H, D] layout.
    output = torch.permute(out, [0, 2, 1, 3])
    return output
