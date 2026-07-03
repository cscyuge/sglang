# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/forward_context.py
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention import AttentionMetadata
    from sglang.multimodal_gen.runtime.pipelines_core import Req

logger = init_logger(__name__)

# TODO(will): check if this is needed
# track_batchsize: bool = envs.SGLANG_DIFFUSION_LOG_BATCHSIZE_INTERVAL >= 0
track_batchsize: bool = False
last_logging_time: float = 0
forward_start_time: float = 0
# batchsize_logging_interval: float = envs.SGLANG_DIFFUSION_LOG_BATCHSIZE_INTERVAL
batchsize_logging_interval: float = 1000
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class ForwardContext:
    current_timestep: int
    # TODO(will): check this arg
    # copy from vllm_config.compilation_config.static_forward_context
    # attn_layers: Dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    forward_batch: Optional["Req"] = None
    attention_backend_cls: Optional[Type] = None

    def set_attn_backend_cls(self, attention_backend_cls: Type):
        if self.attention_backend_cls:
            if self.attention_backend_cls != attention_backend_cls:
                raise RuntimeError(
                    f"Different types of attention backend in a same context detected, previous: {self.attention_backend_cls}, new: {attention_backend_cls}"
                )
        else:
            self.attention_backend_cls = attention_backend_cls


_forward_context_local = threading.local()

# Compatibility fallback for code paths that still seed the context directly.
# Runtime callers should use set_forward_context(), which is thread-local.
_forward_context: Optional["ForwardContext"] = None


def _get_thread_forward_context() -> Optional["ForwardContext"]:
    return getattr(_forward_context_local, "value", None)


def _set_thread_forward_context(context: Optional["ForwardContext"]) -> None:
    if context is None:
        if hasattr(_forward_context_local, "value"):
            delattr(_forward_context_local, "value")
        return
    _forward_context_local.value = context


def get_forward_context() -> "ForwardContext":
    """Get the current forward context."""
    context = _get_thread_forward_context() or _forward_context
    assert context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return context


# TODO(will): finalize the interface
@contextmanager
def set_forward_context(
    current_timestep, attn_metadata, forward_batch: Optional["Req"] = None
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    prev_context = _get_thread_forward_context()
    _set_thread_forward_context(
        ForwardContext(
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
            forward_batch=forward_batch,
        )
    )

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = (
                    attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
                )
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
        _set_thread_forward_context(prev_context)
