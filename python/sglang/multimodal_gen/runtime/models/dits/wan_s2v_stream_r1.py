# SPDX-License-Identifier: Apache-2.0
"""Lightweight Stream-R1 helpers for Wan S2V attention."""

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, TypedDict

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_qkv,
    _usp_output_all_to_all,
)


class WanS2VKVCacheBlock(TypedDict):
    k: torch.Tensor
    v: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor


def _stream_r1_profile_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_PROFILE", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stream_r1_attention_backend() -> str:
    value = os.getenv("SGLANG_STREAM_R1_ATTENTION_BACKEND", "dense_sdpa").lower()
    value = value.replace("-", "_")
    if value in ("", "dense", "dense_sdpa", "sdpa"):
        return "dense_sdpa"
    if value in ("packed", "packed_varlen", "varlen"):
        return "packed_varlen"
    raise ValueError(
        "Unsupported SGLANG_STREAM_R1_ATTENTION_BACKEND="
        f"{value!r}; expected dense_sdpa or packed_varlen"
    )


def wan_s2v_stream_r1_attention_backend() -> str:
    return _stream_r1_attention_backend()


def wan_s2v_stream_r1_uses_head_sharded_sp_kv_cache() -> bool:
    return _stream_r1_attention_backend() == "packed_varlen" and get_sp_world_size() > 1


class _StreamR1ProfileSpan:
    def __init__(self, profile: "_StreamR1Profile", name: str):
        self.profile = profile
        self.name = name
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._start_time: float | None = None

    def __enter__(self):
        if self.profile.use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.profile.use_cuda_events:
            assert self._start_event is not None
            assert self._end_event is not None
            self._end_event.record()
            self._end_event.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            assert self._start_time is not None
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.profile.timings.append((self.name, elapsed_ms))
        return False


class _StreamR1Profile:
    def __init__(self, *, tag: str, device: torch.device | None):
        self.enabled = _stream_r1_profile_enabled()
        self.tag = tag
        self.timings: list[tuple[str, float]] = []
        self.use_cuda_events = (
            self.enabled
            and device is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        )

    def span(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _StreamR1ProfileSpan(self, name)

    def log(self, **metadata) -> None:
        if not self.enabled or not self.timings:
            return
        fields = [f"tag={self.tag}"]
        fields.extend(f"{key}={value}" for key, value in metadata.items())
        fields.extend(f"{name}={elapsed_ms:.3f}ms" for name, elapsed_ms in self.timings)
        print("[stream-r1-profile] " + " ".join(fields), flush=True)


@dataclass(frozen=True)
class WanS2VStreamR1AttentionPlan:
    query_seq_len: int
    kv_seq_len: int
    noisy_query_seq_len: int
    noisy_kv_seq_len: int
    condition_kv_seq_len: int
    frame_seq_length: int
    query_block_tokens: int
    local_attn_size: int
    sink_size: int
    current_start: int = 0
    cache_start: int = 0
    noisy_kv_absolute_index: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.query_seq_len <= 0:
            raise ValueError("query_seq_len must be positive")
        if self.kv_seq_len <= 0:
            raise ValueError("kv_seq_len must be positive")
        if self.noisy_query_seq_len < 0:
            raise ValueError("noisy_query_seq_len must be non-negative")
        if self.noisy_query_seq_len > self.query_seq_len:
            raise ValueError("noisy_query_seq_len must not exceed query_seq_len")
        if self.noisy_kv_seq_len < 0:
            raise ValueError("noisy_kv_seq_len must be non-negative")
        if self.condition_kv_seq_len < 0:
            raise ValueError("condition_kv_seq_len must be non-negative")
        if self.noisy_kv_seq_len + self.condition_kv_seq_len != self.kv_seq_len:
            raise ValueError("noisy and condition KV lengths must add up to kv_seq_len")
        if self.frame_seq_length <= 0:
            raise ValueError("frame_seq_length must be positive")
        if self.query_block_tokens <= 0:
            raise ValueError("query_block_tokens must be positive")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.current_start < 0:
            raise ValueError("current_start must be non-negative")
        if self.cache_start < 0:
            raise ValueError("cache_start must be non-negative")
        if self.current_start % self.frame_seq_length != 0:
            raise ValueError("current_start must be frame-aligned")
        if self.cache_start % self.frame_seq_length != 0:
            raise ValueError("cache_start must be frame-aligned")
        if self.noisy_kv_absolute_index is not None:
            if self.noisy_kv_absolute_index.dim() != 1:
                raise ValueError("noisy_kv_absolute_index must be 1D")
            if self.noisy_kv_absolute_index.shape[0] != self.noisy_kv_seq_len:
                raise ValueError(
                    "noisy_kv_absolute_index length must match noisy_kv_seq_len"
                )

    @property
    def condition_query_seq_len(self) -> int:
        return self.query_seq_len - self.noisy_query_seq_len

    @property
    def local_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    @property
    def sink_end(self) -> int:
        return self.cache_start + self.sink_tokens

    def noisy_kv_index(self, device: torch.device) -> torch.Tensor:
        if self.noisy_kv_absolute_index is not None:
            return self.noisy_kv_absolute_index.to(device=device)
        return self.current_start + torch.arange(
            self.noisy_kv_seq_len,
            dtype=torch.long,
            device=device,
        )

    def query_groups(self, device: torch.device) -> list["WanS2VStreamR1QueryGroup"]:
        groups: list[WanS2VStreamR1QueryGroup] = []
        noisy_kv_abs = self.noisy_kv_index(device)
        condition_indices = torch.arange(
            self.noisy_kv_seq_len,
            self.kv_seq_len,
            dtype=torch.long,
            device=device,
        )

        query_start = 0
        while query_start < self.noisy_query_seq_len:
            query_abs = self.current_start + query_start
            block_end = (
                query_abs // self.query_block_tokens + 1
            ) * self.query_block_tokens
            query_end = min(
                self.noisy_query_seq_len,
                max(query_start + 1, block_end - self.current_start),
            )
            local_start = block_end - self.local_tokens
            visible_noisy = (noisy_kv_abs < self.sink_end) | (
                (noisy_kv_abs >= local_start) & (noisy_kv_abs < block_end)
            )
            noisy_indices = torch.nonzero(visible_noisy, as_tuple=False).flatten()
            if condition_indices.numel() > 0:
                kv_indices = torch.cat([noisy_indices, condition_indices])
            else:
                kv_indices = noisy_indices
            groups.append(
                WanS2VStreamR1QueryGroup(
                    query_start=query_start,
                    query_end=query_end,
                    kv_indices=kv_indices,
                )
            )
            query_start = query_end

        if self.condition_query_seq_len > 0:
            groups.append(
                WanS2VStreamR1QueryGroup(
                    query_start=self.noisy_query_seq_len,
                    query_end=self.query_seq_len,
                    kv_indices=torch.arange(
                        self.kv_seq_len,
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )
        return groups

    def to_dense_mask(self, device: torch.device) -> torch.Tensor:
        """Export the compact plan as the current dense bool mask reference."""

        mask = torch.zeros(
            (self.query_seq_len, self.kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
        for group in self.query_groups(device):
            mask[group.query_start : group.query_end, group.kv_indices] = True
        return mask.unsqueeze(0)


@dataclass(frozen=True)
class WanS2VStreamR1QueryGroup:
    query_start: int
    query_end: int
    kv_indices: torch.Tensor

    @property
    def query_len(self) -> int:
        return self.query_end - self.query_start


@dataclass(frozen=True)
class WanS2VStreamR1PackedAttentionSegment:
    batch_index: int
    query_start: int
    query_end: int
    packed_query_start: int
    packed_query_end: int
    packed_kv_start: int
    packed_kv_end: int


@dataclass(frozen=True)
class WanS2VStreamR1PackedAttentionWorkspace:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    segments: tuple[WanS2VStreamR1PackedAttentionSegment, ...]


def build_wan_s2v_stream_r1_packed_attention_workspace(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
) -> WanS2VStreamR1PackedAttentionWorkspace:
    """Pack Stream-R1 plan-visible K/V ranges as varlen attention segments."""

    _validate_key_value_pair(key, value, name="packed attention key/value")
    if query.dim() != 4:
        raise ValueError("packed attention query must have shape [B, S, H, D]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("packed attention query/key batch/head dimensions must match")
    if query.shape[1] != plan.query_seq_len:
        raise ValueError("packed attention query length must match attention plan")
    if key.shape[1] != plan.kv_seq_len:
        raise ValueError("packed attention key length must match attention plan")

    query_parts = []
    key_parts = []
    value_parts = []
    cu_q = [0]
    cu_k = [0]
    segments: list[WanS2VStreamR1PackedAttentionSegment] = []
    max_q = 0
    max_k = 0
    query_groups = plan.query_groups(query.device)
    for batch_index in range(query.shape[0]):
        for group in query_groups:
            if group.query_len <= 0:
                raise ValueError("packed attention query groups must be non-empty")
            if group.kv_indices.numel() <= 0:
                raise ValueError("packed attention KV groups must be non-empty")
            q_part = query[batch_index, group.query_start : group.query_end]
            k_part = key[batch_index].index_select(0, group.kv_indices)
            v_part = value[batch_index].index_select(0, group.kv_indices)
            query_parts.append(q_part)
            key_parts.append(k_part)
            value_parts.append(v_part)

            q_start = cu_q[-1]
            k_start = cu_k[-1]
            q_end = q_start + q_part.shape[0]
            k_end = k_start + k_part.shape[0]
            cu_q.append(q_end)
            cu_k.append(k_end)
            max_q = max(max_q, q_part.shape[0])
            max_k = max(max_k, k_part.shape[0])
            segments.append(
                WanS2VStreamR1PackedAttentionSegment(
                    batch_index=batch_index,
                    query_start=group.query_start,
                    query_end=group.query_end,
                    packed_query_start=q_start,
                    packed_query_end=q_end,
                    packed_kv_start=k_start,
                    packed_kv_end=k_end,
                )
            )

    return WanS2VStreamR1PackedAttentionWorkspace(
        query=torch.cat(query_parts, dim=0).contiguous(),
        key=torch.cat(key_parts, dim=0).contiguous(),
        value=torch.cat(value_parts, dim=0).contiguous(),
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=query.device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=query.device),
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        segments=tuple(segments),
    )


def _unpack_wan_s2v_stream_r1_packed_attention(
    packed_output: torch.Tensor,
    workspace: WanS2VStreamR1PackedAttentionWorkspace,
    output_shape: torch.Size,
) -> torch.Tensor:
    output = packed_output.new_empty(output_shape)
    for segment in workspace.segments:
        output[
            segment.batch_index,
            segment.query_start : segment.query_end,
        ] = packed_output[segment.packed_query_start : segment.packed_query_end]
    return output


def _run_wan_s2v_stream_r1_packed_torch_attention(
    workspace: WanS2VStreamR1PackedAttentionWorkspace,
    output_shape: torch.Size,
    *,
    softmax_scale: float | None,
) -> torch.Tensor:
    output = workspace.query.new_empty(output_shape)
    for segment in workspace.segments:
        q = workspace.query[
            segment.packed_query_start : segment.packed_query_end
        ].transpose(0, 1).unsqueeze(0)
        k = workspace.key[segment.packed_kv_start : segment.packed_kv_end].transpose(
            0, 1
        ).unsqueeze(0)
        v = workspace.value[segment.packed_kv_start : segment.packed_kv_end].transpose(
            0, 1
        ).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        ).squeeze(0).transpose(0, 1)
        output[
            segment.batch_index,
            segment.query_start : segment.query_end,
        ] = out
    return output


def stream_r1_packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
    *,
    softmax_scale: float | None,
    force_torch: bool = False,
) -> torch.Tensor:
    """Run Stream-R1 packed varlen attention for one non-SP mixed-KV call."""

    workspace = build_wan_s2v_stream_r1_packed_attention_workspace(
        query,
        key,
        value,
        plan,
    )
    if force_torch or query.device.type != "cuda":
        return _run_wan_s2v_stream_r1_packed_torch_attention(
            workspace,
            query.shape,
            softmax_scale=softmax_scale,
        )

    try:
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
    except Exception as exc:  # pragma: no cover - depends on optional kernels
        raise RuntimeError(
            "SGLANG_STREAM_R1_ATTENTION_BACKEND=packed_varlen requires "
            "flash_attn_varlen_func on CUDA"
        ) from exc

    result = flash_attn_varlen_func(
        workspace.query,
        workspace.key,
        workspace.value,
        workspace.cu_seqlens_q,
        workspace.cu_seqlens_k,
        max_seqlen_q=workspace.max_seqlen_q,
        max_seqlen_k=workspace.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
    )
    packed_output = result[0] if isinstance(result, tuple) else result
    return _unpack_wan_s2v_stream_r1_packed_attention(
        packed_output,
        workspace,
        query.shape,
    )


def _pad_stream_r1_sp_attention_output(
    output: torch.Tensor,
    *,
    total_seq_len: int,
    sp_pad_tokens: int,
) -> torch.Tensor:
    if sp_pad_tokens <= 0:
        return output
    if output.shape[1] != total_seq_len:
        raise ValueError("Stream-R1 SP attention output length must be unpadded")
    pad = output.new_zeros(
        output.shape[0],
        sp_pad_tokens,
        output.shape[2],
        output.shape[3],
    )
    return torch.cat([output, pad], dim=1)


def _can_use_stream_r1_sp_head_sharded_packed_attention(
    query: torch.Tensor,
    kv_cache: WanS2VKVCacheBlock,
    *,
    sp_world_size: int,
) -> bool:
    if sp_world_size <= 1:
        return False
    if query.shape[2] % sp_world_size != 0:
        return False
    expected_local_heads = query.shape[2] // sp_world_size
    return kv_cache["k"].shape[2] == expected_local_heads


@dataclass(frozen=True)
class WanS2VStreamR1AttentionLayout:
    noisy_seq_len: int
    total_seq_len: int
    frame_seq_length: int
    num_frame_per_block: int
    local_attn_size: int
    sink_size: int
    current_start: int = 0

    def __post_init__(self) -> None:
        if self.noisy_seq_len <= 0:
            raise ValueError("noisy_seq_len must be positive")
        if self.total_seq_len < self.noisy_seq_len:
            raise ValueError("total_seq_len must be at least noisy_seq_len")
        if self.frame_seq_length <= 0:
            raise ValueError("frame_seq_length must be positive")
        if self.noisy_seq_len % self.frame_seq_length != 0:
            raise ValueError("noisy_seq_len must be divisible by frame_seq_length")
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.current_start < 0:
            raise ValueError("current_start must be non-negative")
        if self.current_start % self.frame_seq_length != 0:
            raise ValueError("current_start must be frame-aligned")

    @property
    def condition_seq_len(self) -> int:
        return self.total_seq_len - self.noisy_seq_len

    @property
    def block_tokens(self) -> int:
        return self.num_frame_per_block * self.frame_seq_length

    @property
    def local_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    @property
    def noisy_cache_tokens(self) -> int:
        return self.local_tokens

    def to_noisy_kv_cache_update(
        self,
        *,
        cache_start: int = 0,
    ) -> "WanS2VStreamR1NoisyKVCacheUpdate":
        return WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=self.noisy_seq_len,
            frame_seq_length=self.frame_seq_length,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            current_start=self.current_start,
            cache_start=cache_start,
        )

    def to_no_kv_attention_plan(self) -> WanS2VStreamR1AttentionPlan:
        return WanS2VStreamR1AttentionPlan(
            query_seq_len=self.total_seq_len,
            kv_seq_len=self.total_seq_len,
            noisy_query_seq_len=self.noisy_seq_len,
            noisy_kv_seq_len=self.noisy_seq_len,
            condition_kv_seq_len=self.condition_seq_len,
            frame_seq_length=self.frame_seq_length,
            query_block_tokens=self.block_tokens,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
            current_start=self.current_start,
            cache_start=0,
        )

    def build_no_kv_attention_mask(self, device: torch.device) -> torch.Tensor:
        """Build the Stream-R1 no-KV local/sink mask for mixed S2V tokens.

        Noisy latent queries are restricted to sink noisy tokens, their local
        block window, and all current condition tokens. Condition queries keep
        dense attention so reference/motion tokens retain legacy semantics.
        """

        return self.to_no_kv_attention_plan().to_dense_mask(device)


@dataclass(frozen=True)
class WanS2VStreamR1NoisyKVCacheUpdate:
    noisy_seq_len: int
    frame_seq_length: int
    local_attn_size: int
    sink_size: int
    current_start: int
    cache_start: int = 0

    def __post_init__(self) -> None:
        if self.noisy_seq_len <= 0:
            raise ValueError("noisy_seq_len must be positive")
        if self.frame_seq_length <= 0:
            raise ValueError("frame_seq_length must be positive")
        if self.noisy_seq_len % self.frame_seq_length != 0:
            raise ValueError("noisy_seq_len must be divisible by frame_seq_length")
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.current_start < 0:
            raise ValueError("current_start must be non-negative")
        if self.cache_start < 0:
            raise ValueError("cache_start must be non-negative")
        if self.current_start < self.cache_start:
            raise ValueError(
                "current_start must be greater than or equal to cache_start"
            )
        if self.current_start % self.frame_seq_length != 0:
            raise ValueError("current_start must be frame-aligned")
        if self.cache_start % self.frame_seq_length != 0:
            raise ValueError("cache_start must be frame-aligned")

    @property
    def current_end(self) -> int:
        return self.current_start + self.noisy_seq_len

    @property
    def local_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    @property
    def required_cache_tokens(self) -> int:
        return self.local_tokens

    @property
    def rolling_tokens(self) -> int:
        return self.local_tokens - self.sink_tokens

    @property
    def sink_end(self) -> int:
        return self.cache_start + self.sink_tokens


@dataclass(frozen=True)
class WanS2VStreamR1NoisyKVCacheView:
    key: torch.Tensor
    value: torch.Tensor
    global_end_index: int
    local_end_index: int
    local_start: int
    local_end: int


@dataclass(frozen=True)
class WanS2VStreamR1ProjectedKVSplit:
    noisy_key: torch.Tensor
    noisy_value: torch.Tensor
    condition_key: torch.Tensor
    condition_value: torch.Tensor
    noisy_seq_len: int

    @property
    def condition_seq_len(self) -> int:
        return self.condition_key.shape[1]

    @property
    def total_seq_len(self) -> int:
        return self.noisy_seq_len + self.condition_seq_len


@dataclass(frozen=True)
class WanS2VStreamR1MixedKVView:
    key: torch.Tensor
    value: torch.Tensor
    cached_noisy_seq_len: int
    condition_seq_len: int
    global_end_index: int
    local_end_index: int
    local_start: int
    local_end: int

    @property
    def condition_start_index(self) -> int:
        return self.cached_noisy_seq_len

    @property
    def total_seq_len(self) -> int:
        return self.cached_noisy_seq_len + self.condition_seq_len


@dataclass(frozen=True)
class _KVSegment:
    start: int
    end: int
    key: torch.Tensor
    value: torch.Tensor


def update_wan_s2v_stream_r1_noisy_kv_cache(
    kv_cache: WanS2VKVCacheBlock,
    key: torch.Tensor,
    value: torch.Tensor,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
) -> WanS2VStreamR1NoisyKVCacheView:
    """Mutate one layer's noisy-token KV cache using Stream-R1 S2V windows.

    The cache layout is a fixed local-attention budget. When sink tokens are
    enabled, they occupy the cache prefix and the remaining suffix rolls.
    Condition-token K/V are intentionally out of scope; mixed-token attention
    must still be wired separately before runtime KV support is enabled.
    """

    _validate_noisy_kv_update_inputs(kv_cache, key, value, update)

    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]
    old_global_end = int(kv_cache["global_end_index"].item())
    old_local_end = int(kv_cache["local_end_index"].item())
    if old_global_end == 0 and old_local_end == 0 and update.cache_start > 0:
        old_global_end = update.cache_start
    if old_global_end < update.cache_start:
        raise ValueError("KV cache global_end_index is before cache_start")
    if old_local_end < 0 or old_local_end > cache_k.shape[1]:
        raise ValueError("KV cache local_end_index is outside cache capacity")
    if old_local_end > update.required_cache_tokens:
        raise ValueError(
            "KV cache local_end_index exceeds the Stream-R1 local window"
        )
    if update.current_start > old_global_end:
        raise ValueError("KV cache update cannot skip noisy-token ranges")
    if update.current_end < old_global_end:
        raise ValueError("KV cache update cannot move global_end_index backwards")

    old_segments = _collect_old_kv_segments(
        kv_cache,
        update=update,
        old_global_end=old_global_end,
        old_local_end=old_local_end,
    )
    new_segment = _KVSegment(update.current_start, update.current_end, key, value)
    source_segments = [new_segment, *old_segments]

    final_global_end = update.current_end
    sink_len = min(update.sink_tokens, final_global_end - update.cache_start)
    local_start = max(update.sink_end, final_global_end - update.rolling_tokens)
    local_len = max(0, final_global_end - local_start)

    cache_k.zero_()
    cache_v.zero_()
    if sink_len > 0:
        _copy_abs_range(
            cache_k,
            cache_v,
            0,
            update.cache_start,
            update.cache_start + sink_len,
            source_segments,
        )
    if local_len > 0:
        _copy_abs_range(
            cache_k,
            cache_v,
            update.sink_tokens,
            local_start,
            final_global_end,
            source_segments,
        )

    local_end_index = update.sink_tokens + local_len if local_len > 0 else sink_len
    kv_cache["global_end_index"].fill_(final_global_end)
    kv_cache["local_end_index"].fill_(local_end_index)
    return WanS2VStreamR1NoisyKVCacheView(
        key=cache_k[:, :local_end_index],
        value=cache_v[:, :local_end_index],
        global_end_index=final_global_end,
        local_end_index=local_end_index,
        local_start=local_start,
        local_end=final_global_end,
    )


def split_wan_s2v_stream_r1_projected_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    noisy_seq_len: int,
) -> WanS2VStreamR1ProjectedKVSplit:
    """Split projected mixed S2V self-attention K/V into noisy and condition spans."""

    _validate_key_value_pair(key, value, name="projected key/value")
    if noisy_seq_len <= 0:
        raise ValueError("noisy_seq_len must be positive")
    if noisy_seq_len > key.shape[1]:
        raise ValueError(
            "noisy_seq_len must not exceed projected sequence length: "
            f"noisy_seq_len={noisy_seq_len}, seq_len={key.shape[1]}"
        )
    return WanS2VStreamR1ProjectedKVSplit(
        noisy_key=key[:, :noisy_seq_len],
        noisy_value=value[:, :noisy_seq_len],
        condition_key=key[:, noisy_seq_len:],
        condition_value=value[:, noisy_seq_len:],
        noisy_seq_len=noisy_seq_len,
    )


def compose_wan_s2v_stream_r1_mixed_kv_view(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    current_kv: WanS2VStreamR1ProjectedKVSplit,
) -> WanS2VStreamR1MixedKVView:
    """Compose cached noisy K/V with current condition K/V for S2V attention."""

    _validate_key_value_pair(noisy_view.key, noisy_view.value, name="cached noisy K/V")
    _validate_key_value_pair(
        current_kv.noisy_key,
        current_kv.noisy_value,
        name="current noisy K/V",
    )
    _validate_key_value_pair(
        current_kv.condition_key,
        current_kv.condition_value,
        name="current condition K/V",
    )
    _validate_compatible_kv_prefix(noisy_view.key, current_kv.noisy_key)
    _validate_compatible_kv_prefix(noisy_view.key, current_kv.condition_key)

    if current_kv.condition_seq_len > 0:
        key = torch.cat([noisy_view.key, current_kv.condition_key], dim=1)
        value = torch.cat([noisy_view.value, current_kv.condition_value], dim=1)
    else:
        key = noisy_view.key
        value = noisy_view.value

    return WanS2VStreamR1MixedKVView(
        key=key,
        value=value,
        cached_noisy_seq_len=noisy_view.key.shape[1],
        condition_seq_len=current_kv.condition_seq_len,
        global_end_index=noisy_view.global_end_index,
        local_end_index=noisy_view.local_end_index,
        local_start=noisy_view.local_start,
        local_end=noisy_view.local_end,
    )


def build_wan_s2v_stream_r1_cached_noisy_kv_index(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return absolute noisy-token positions for a cached noisy-KV view.

    The view order mirrors the fixed-budget cache layout: an optional sink
    prefix followed by the rolling local suffix. The returned index is intended
    for attention-mask construction only; it is not wired into runtime KV
    mutation in this phase.
    """

    _validate_noisy_view_for_update(noisy_view, update)

    device = device or noisy_view.key.device
    local_len = _noisy_view_local_len(noisy_view)
    sink_len = noisy_view.local_end_index - local_len
    parts = []
    if sink_len > 0:
        parts.append(
            torch.arange(
                update.cache_start,
                update.cache_start + sink_len,
                dtype=torch.long,
                device=device,
            )
        )
    if local_len > 0:
        parts.append(
            torch.arange(
                noisy_view.local_start,
                noisy_view.local_end,
                dtype=torch.long,
                device=device,
            )
        )
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(parts)


def build_wan_s2v_stream_r1_mixed_kv_attention_mask(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    mixed_view: WanS2VStreamR1MixedKVView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build cached mixed-KV attention mask for Stream-R1 S2V.

    Query order is ``[current noisy tokens + current condition tokens]``.
    KV order is ``[cached noisy tokens + current condition tokens]``. Noisy
    queries may attend to cached sink noisy tokens, cached local noisy tokens
    for their block, and all current condition tokens. Condition queries keep
    dense attention over the composed cached-mixed KV view.
    """

    plan = build_wan_s2v_stream_r1_mixed_kv_attention_plan(
        noisy_view,
        mixed_view,
        update,
        device=device,
    )
    return plan.to_dense_mask(device or mixed_view.key.device)


def build_wan_s2v_stream_r1_mixed_kv_attention_plan(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    mixed_view: WanS2VStreamR1MixedKVView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    device: torch.device | None = None,
) -> WanS2VStreamR1AttentionPlan:
    """Build a compact mixed-KV attention plan for Stream-R1 S2V."""

    _validate_mixed_view_for_noisy_view(noisy_view, mixed_view)
    cached_noisy_index = build_wan_s2v_stream_r1_cached_noisy_kv_index(
        noisy_view,
        update,
        device=device or mixed_view.key.device,
    )
    return WanS2VStreamR1AttentionPlan(
        query_seq_len=update.noisy_seq_len + mixed_view.condition_seq_len,
        kv_seq_len=mixed_view.total_seq_len,
        noisy_query_seq_len=update.noisy_seq_len,
        noisy_kv_seq_len=mixed_view.cached_noisy_seq_len,
        condition_kv_seq_len=mixed_view.condition_seq_len,
        frame_seq_length=update.frame_seq_length,
        query_block_tokens=update.noisy_seq_len,
        local_attn_size=update.local_attn_size,
        sink_size=update.sink_size,
        current_start=update.current_start,
        cache_start=update.cache_start,
        noisy_kv_absolute_index=cached_noisy_index,
    )


def pad_wan_s2v_stream_r1_mixed_kv_query_mask_for_sp(
    attn_mask: torch.Tensor,
    original_query_seq_len: int,
    pad_tokens: int,
) -> torch.Tensor:
    if pad_tokens <= 0:
        return attn_mask
    if attn_mask.dim() != 3:
        raise ValueError("Stream-R1 SP mixed-KV attention mask must be [B, S, K]")
    if attn_mask.shape[-2] != original_query_seq_len:
        raise ValueError(
            "Stream-R1 SP mixed-KV attention mask query length does not match "
            "the unpadded sequence"
        )

    padded_query_seq_len = original_query_seq_len + pad_tokens
    padded_mask = torch.ones(
        (attn_mask.shape[0], padded_query_seq_len, attn_mask.shape[-1]),
        dtype=attn_mask.dtype,
        device=attn_mask.device,
    )
    padded_mask[:, :original_query_seq_len, :] = attn_mask
    return padded_mask


def run_wan_s2v_stream_r1_cached_self_attention(
    attention: Callable[..., torch.Tensor],
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: WanS2VKVCacheBlock | None,
    layout: WanS2VStreamR1AttentionLayout | None,
    cache_start: int | None,
    sequence_shard_enabled: bool = False,
    sp_pad_tokens: int = 0,
) -> torch.Tensor:
    """Run one guarded Stream-R1 S2V cached self-attention step."""

    if kv_cache is None:
        raise ValueError("Stream-R1 S2V cached attention requires kv_cache")
    if layout is None:
        raise ValueError("Stream-R1 S2V cached attention requires attention layout")
    if sp_pad_tokens < 0:
        raise ValueError("sp_pad_tokens must be non-negative")
    if query.dim() != 4:
        raise ValueError("query tensor must have shape [B, S, H, D]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query and key/value batch/head dimensions must match")
    if value.shape[0] != key.shape[0] or value.shape[2:] != key.shape[2:]:
        raise ValueError("key and value batch/head dimensions must match")
    profile = _StreamR1Profile(
        tag="stream_r1_cached_self_attention",
        device=query.device,
    )
    attention_backend = _stream_r1_attention_backend()
    selected_backend = attention_backend
    use_sp_head_sharded_packed_attention = False
    if sequence_shard_enabled:
        sp_world_size = get_sp_world_size()
        padded_seq_len = layout.total_seq_len + sp_pad_tokens
        if query.shape[1] * sp_world_size != padded_seq_len:
            raise ValueError(
                "local query sequence length does not match the padded "
                "Stream-R1 SP attention layout"
            )
        if key.shape[1] * sp_world_size != padded_seq_len:
            raise ValueError(
                "local key/value sequence length does not match the padded "
                "Stream-R1 SP attention layout"
            )
        use_sp_head_sharded_packed_attention = (
            attention_backend == "packed_varlen"
            and _can_use_stream_r1_sp_head_sharded_packed_attention(
                query,
                kv_cache,
                sp_world_size=sp_world_size,
            )
        )
        if use_sp_head_sharded_packed_attention:
            selected_backend = "packed_varlen_sp_head_sharded"
            with profile.span("usp_qkv_all_to_all"):
                query_for_attention, key, value = _usp_input_all_to_all_qkv(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                )
            query_for_attention = query_for_attention[
                :, : layout.total_seq_len
            ].contiguous()
            key = key[:, : layout.total_seq_len].contiguous()
            value = value[:, : layout.total_seq_len].contiguous()
        else:
            if attention_backend == "packed_varlen":
                selected_backend = "dense_sdpa_sp_fallback"
            query_for_attention = query
            with profile.span("kv_all_gather"):
                key = sequence_model_parallel_all_gather(key.contiguous(), dim=1)
                value = sequence_model_parallel_all_gather(value.contiguous(), dim=1)
            key = key[:, : layout.total_seq_len].contiguous()
            value = value[:, : layout.total_seq_len].contiguous()
    else:
        if query.shape[1] != layout.total_seq_len:
            raise ValueError(
                "query sequence length must match the Stream-R1 attention layout"
            )
        if key.shape[1] != layout.total_seq_len:
            raise ValueError(
                "key/value sequence length must match the Stream-R1 attention layout"
            )
        query_for_attention = query

    with profile.span("split_current_kv"):
        update = layout.to_noisy_kv_cache_update(cache_start=cache_start or 0)
        current_kv = split_wan_s2v_stream_r1_projected_kv(
            key,
            value,
            noisy_seq_len=layout.noisy_seq_len,
        )
    with profile.span("cache_update"):
        noisy_view = update_wan_s2v_stream_r1_noisy_kv_cache(
            kv_cache,
            current_kv.noisy_key,
            current_kv.noisy_value,
            update,
        )
    with profile.span("mixed_kv_compose"):
        mixed_view = compose_wan_s2v_stream_r1_mixed_kv_view(noisy_view, current_kv)
    with profile.span("attention_plan"):
        mixed_plan = build_wan_s2v_stream_r1_mixed_kv_attention_plan(
            noisy_view,
            mixed_view,
            update,
            device=query_for_attention.device,
        )
    if use_sp_head_sharded_packed_attention:
        with profile.span("packed_varlen_attention"):
            output = stream_r1_packed_varlen_attention(
                query_for_attention,
                mixed_view.key,
                mixed_view.value,
                mixed_plan,
                softmax_scale=getattr(attention, "softmax_scale", None),
            )
        with profile.span("sp_output_pad"):
            output = _pad_stream_r1_sp_attention_output(
                output,
                total_seq_len=layout.total_seq_len,
                sp_pad_tokens=sp_pad_tokens,
            )
        with profile.span("usp_output_all_to_all"):
            output = _usp_output_all_to_all(output.contiguous(), head_dim=2)
    elif attention_backend == "packed_varlen" and not sequence_shard_enabled:
        with profile.span("packed_varlen_attention"):
            output = stream_r1_packed_varlen_attention(
                query_for_attention,
                mixed_view.key,
                mixed_view.value,
                mixed_plan,
                softmax_scale=getattr(attention, "softmax_scale", None),
            )
    else:
        if attention_backend == "packed_varlen" and sequence_shard_enabled:
            selected_backend = "dense_sdpa_sp_fallback"
        with profile.span("attention_plan_dense_mask"):
            mixed_mask = mixed_plan.to_dense_mask(query_for_attention.device)
        if sequence_shard_enabled:
            with profile.span("sp_mask_pad"):
                mixed_mask = pad_wan_s2v_stream_r1_mixed_kv_query_mask_for_sp(
                    mixed_mask,
                    layout.total_seq_len,
                    sp_pad_tokens,
                )
            with profile.span("attention"):
                output = attention(
                    query_for_attention,
                    mixed_view.key,
                    mixed_view.value,
                    attn_mask=mixed_mask,
                    kv_is_replicated=True,
                )
        else:
            with profile.span("attention"):
                output = attention(
                    query_for_attention,
                    mixed_view.key,
                    mixed_view.value,
                    attn_mask=mixed_mask,
                )
    profile.log(
        attention_backend=selected_backend,
        query_seq_len=query.shape[1],
        mixed_kv_seq_len=mixed_view.total_seq_len,
        cached_noisy_seq_len=mixed_view.cached_noisy_seq_len,
        sequence_shard_enabled=sequence_shard_enabled,
        sp_pad_tokens=sp_pad_tokens,
    )
    return output


def validate_wan_s2v_stream_r1_forward_cache(
    *,
    kv_cache: list[WanS2VKVCacheBlock] | None,
    crossattn_cache: list | None,
    stream_r1_mode: bool,
    num_transformer_blocks: int,
) -> None:
    """Validate cache arguments accepted by Wan S2V transformer forward."""

    if crossattn_cache is not None:
        if not stream_r1_mode:
            raise ValueError(
                "Wan S2V crossattn_cache is only supported when "
                "stream_r1_mode=True"
            )
        if not isinstance(crossattn_cache, list):
            raise ValueError(
                "Wan S2V crossattn_cache must be a list of per-block cache entries"
            )
        if len(crossattn_cache) != num_transformer_blocks:
            raise ValueError(
                "Wan S2V crossattn_cache length must match the number of "
                f"transformer blocks: crossattn_cache length={len(crossattn_cache)}, "
                f"blocks={num_transformer_blocks}"
            )
        for idx, entry in enumerate(crossattn_cache):
            if not isinstance(entry, dict):
                raise ValueError(
                    "Wan S2V crossattn_cache entries must be dicts: "
                    f"index={idx}, type={type(entry).__name__}"
                )

    if kv_cache is None:
        return
    if not stream_r1_mode:
        raise ValueError(
            "Wan S2V kv_cache is only supported when stream_r1_mode=True"
        )
    if not isinstance(kv_cache, list):
        raise ValueError(
            "Wan S2V kv_cache must be a list of per-block cache entries"
        )
    if len(kv_cache) != num_transformer_blocks:
        raise ValueError(
            "Wan S2V kv_cache length must match the number of transformer "
            f"blocks: kv_cache length={len(kv_cache)}, "
            f"blocks={num_transformer_blocks}"
        )


def _validate_noisy_kv_update_inputs(
    kv_cache: WanS2VKVCacheBlock,
    key: torch.Tensor,
    value: torch.Tensor,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
) -> None:
    _validate_key_value_pair(key, value, name="key/value")
    if key.shape[1] != update.noisy_seq_len:
        raise ValueError(
            "key/value sequence length must match update.noisy_seq_len"
        )
    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]
    if cache_k.shape != cache_v.shape:
        raise ValueError("KV cache k and v tensors must have the same shape")
    if cache_k.dim() != 4:
        raise ValueError("KV cache tensors must have shape [B, S, H, D]")
    if cache_k.shape[0] != key.shape[0] or cache_k.shape[2:] != key.shape[2:]:
        raise ValueError("KV cache and key/value batch/head dimensions must match")
    if cache_k.device != key.device or cache_v.device != value.device:
        raise ValueError("KV cache and key/value tensors must be on the same device")
    if cache_k.dtype != key.dtype or cache_v.dtype != value.dtype:
        raise ValueError("KV cache and key/value tensors must use the same dtype")
    if cache_k.shape[1] < update.required_cache_tokens:
        raise ValueError(
            "KV cache capacity is smaller than the Stream-R1 local window: "
            f"capacity={cache_k.shape[1]}, required={update.required_cache_tokens}"
        )


def _validate_key_value_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    name: str,
) -> None:
    if key.shape != value.shape:
        raise ValueError(f"{name} tensors must have the same shape")
    if key.dim() != 4:
        raise ValueError(f"{name} tensors must have shape [B, S, H, D]")
    if key.device != value.device:
        raise ValueError(f"{name} tensors must be on the same device")
    if key.dtype != value.dtype:
        raise ValueError(f"{name} tensors must use the same dtype")


def _validate_compatible_kv_prefix(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    if expected.shape[0] != actual.shape[0] or expected.shape[2:] != actual.shape[2:]:
        raise ValueError(
            "cached noisy K/V and current K/V batch/head dimensions must match"
        )
    if expected.device != actual.device:
        raise ValueError("cached noisy K/V and current K/V must be on the same device")
    if expected.dtype != actual.dtype:
        raise ValueError("cached noisy K/V and current K/V must use the same dtype")


def _validate_noisy_view_for_update(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
) -> None:
    _validate_key_value_pair(noisy_view.key, noisy_view.value, name="cached noisy K/V")
    if noisy_view.local_end_index != noisy_view.key.shape[1]:
        raise ValueError("cached noisy K/V view length must match local_end_index")
    if noisy_view.local_end_index < 0:
        raise ValueError("cached noisy K/V local_end_index must be non-negative")
    if noisy_view.global_end_index != update.current_end:
        raise ValueError(
            "cached noisy K/V global_end_index must match update.current_end"
        )
    if noisy_view.local_end != noisy_view.global_end_index:
        raise ValueError("cached noisy K/V local_end must match global_end_index")

    local_len = _noisy_view_local_len(noisy_view)
    if local_len > update.rolling_tokens:
        raise ValueError("cached noisy K/V local suffix exceeds rolling budget")
    if local_len > noisy_view.local_end_index:
        raise ValueError("cached noisy K/V local suffix exceeds view length")
    sink_len = noisy_view.local_end_index - local_len
    if sink_len > update.sink_tokens:
        raise ValueError("cached noisy K/V sink prefix exceeds sink budget")
    if sink_len > max(0, update.current_end - update.cache_start):
        raise ValueError("cached noisy K/V sink prefix exceeds available history")
    if local_len > 0 and noisy_view.local_start < update.sink_end:
        raise ValueError("cached noisy K/V local suffix overlaps sink prefix")


def _validate_mixed_view_for_noisy_view(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    mixed_view: WanS2VStreamR1MixedKVView,
) -> None:
    _validate_key_value_pair(mixed_view.key, mixed_view.value, name="mixed K/V")
    if mixed_view.cached_noisy_seq_len != noisy_view.key.shape[1]:
        raise ValueError(
            "mixed K/V cached_noisy_seq_len must match the noisy view length"
        )
    if mixed_view.condition_seq_len < 0:
        raise ValueError("mixed K/V condition_seq_len must be non-negative")
    if mixed_view.total_seq_len != mixed_view.key.shape[1]:
        raise ValueError("mixed K/V metadata must match key/value sequence length")
    if mixed_view.global_end_index != noisy_view.global_end_index:
        raise ValueError("mixed K/V global_end_index must match the noisy view")
    if mixed_view.local_end_index != noisy_view.local_end_index:
        raise ValueError("mixed K/V local_end_index must match the noisy view")
    if mixed_view.local_start != noisy_view.local_start:
        raise ValueError("mixed K/V local_start must match the noisy view")
    if mixed_view.local_end != noisy_view.local_end:
        raise ValueError("mixed K/V local_end must match the noisy view")
    _validate_compatible_kv_prefix(
        mixed_view.key[:, : mixed_view.cached_noisy_seq_len],
        noisy_view.key,
    )


def _noisy_view_local_len(noisy_view: WanS2VStreamR1NoisyKVCacheView) -> int:
    return max(0, noisy_view.local_end - noisy_view.local_start)


def _collect_old_kv_segments(
    kv_cache: WanS2VKVCacheBlock,
    *,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    old_global_end: int,
    old_local_end: int,
) -> list[_KVSegment]:
    if old_global_end <= update.cache_start or old_local_end == 0:
        return []

    segments: list[_KVSegment] = []
    old_sink_len = min(
        update.sink_tokens,
        old_global_end - update.cache_start,
        old_local_end,
    )
    if old_sink_len > 0:
        segments.append(
            _KVSegment(
                update.cache_start,
                update.cache_start + old_sink_len,
                kv_cache["k"][:, :old_sink_len].clone(),
                kv_cache["v"][:, :old_sink_len].clone(),
            )
        )

    old_local_len = max(0, old_local_end - update.sink_tokens)
    if old_local_len > 0:
        old_local_start = old_global_end - old_local_len
        if old_local_start < update.sink_end:
            raise ValueError("KV cache local window overlaps the sink prefix")
        segments.append(
            _KVSegment(
                old_local_start,
                old_global_end,
                kv_cache["k"][
                    :, update.sink_tokens : update.sink_tokens + old_local_len
                ].clone(),
                kv_cache["v"][
                    :, update.sink_tokens : update.sink_tokens + old_local_len
                ].clone(),
            )
        )
    return segments


def _copy_abs_range(
    dest_k: torch.Tensor,
    dest_v: torch.Tensor,
    dest_start: int,
    abs_start: int,
    abs_end: int,
    source_segments: list[_KVSegment],
) -> None:
    cursor = abs_start
    while cursor < abs_end:
        segment = _find_segment_containing(source_segments, cursor)
        if segment is None:
            raise ValueError(
                "KV cache update source data is missing for absolute token "
                f"range starting at {cursor}"
            )
        copy_end = min(abs_end, segment.end)
        src_start = cursor - segment.start
        src_end = copy_end - segment.start
        dst_start = dest_start + cursor - abs_start
        dst_end = dst_start + copy_end - cursor
        dest_k[:, dst_start:dst_end] = segment.key[:, src_start:src_end]
        dest_v[:, dst_start:dst_end] = segment.value[:, src_start:src_end]
        cursor = copy_end


def _find_segment_containing(
    segments: list[_KVSegment],
    position: int,
) -> _KVSegment | None:
    for segment in segments:
        if segment.start <= position < segment.end:
            return segment
    return None
