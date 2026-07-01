# SPDX-License-Identifier: Apache-2.0
"""Lightweight Stream-R1 helpers for Wan S2V attention."""

import os
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
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

_STREAM_R1_SINK_COMPRESSION_ALPHA = 0.999


class WanS2VKVCacheBlock(TypedDict):
    k: torch.Tensor
    v: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor


def _stream_r1_profile_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_PROFILE", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stream_r1_comm_nvtx_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_COMM_NVTX", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stream_r1_reuse_packed_metadata_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_REUSE_PACKED_METADATA", "1")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stream_r1_reuse_packed_buffers_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_REUSE_PACKED_BUFFERS", "1")
    return value.lower() not in ("", "0", "false", "no", "off")


def _stream_r1_fused_segmented_pack_enabled() -> bool:
    value = os.getenv("SGLANG_STREAM_R1_FUSED_SEGMENTED_PACK", "1")
    return value.lower() not in ("", "0", "false", "no", "off")


@contextmanager
def _stream_r1_comm_nvtx_range(message: str):
    if not _stream_r1_comm_nvtx_enabled() or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


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


def _stream_r1_flash_attention_version() -> str:
    value = (
        os.getenv("SGLANG_STREAM_R1_FLASH_ATTENTION_VERSION", "sm120").strip().lower()
    )
    value = value.replace("-", "_")
    if value in ("3", "fa3"):
        return "3"
    if value in ("4", "fa4"):
        return "4"
    if value in ("sm120", "fa4_sm120", "hf_sm120", "flash_attn_4_sm120"):
        return "sm120"
    raise ValueError(
        "Unsupported SGLANG_STREAM_R1_FLASH_ATTENTION_VERSION="
        f"{value!r}; expected 3, 4, or sm120"
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
    condition_queries_use_current_noisy_only: bool = False

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

    def _uses_single_noisy_block_query_groups(self) -> bool:
        return (
            self.noisy_kv_absolute_index is not None
            and self.noisy_query_seq_len > 0
            and self.query_block_tokens == self.noisy_query_seq_len
            and self.local_tokens >= self.noisy_kv_seq_len
            and self.current_start % self.noisy_query_seq_len == 0
        )

    def noisy_kv_index(self, device: torch.device) -> torch.Tensor:
        if self.noisy_kv_absolute_index is not None:
            return self.noisy_kv_absolute_index.to(device=device)
        return self.current_start + torch.arange(
            self.noisy_kv_seq_len,
            dtype=torch.long,
            device=device,
        )

    def query_groups(self, device: torch.device) -> list["WanS2VStreamR1QueryGroup"]:
        if self._uses_single_noisy_block_query_groups():
            return self._single_noisy_block_query_groups(device)

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
            if self.condition_queries_use_current_noisy_only:
                current_noisy_end = self.current_start + self.query_block_tokens
                visible_noisy = (noisy_kv_abs >= self.current_start) & (
                    noisy_kv_abs < current_noisy_end
                )
                noisy_indices = torch.nonzero(visible_noisy, as_tuple=False).flatten()
                if condition_indices.numel() > 0:
                    kv_indices = torch.cat([noisy_indices, condition_indices])
                else:
                    kv_indices = noisy_indices
            else:
                kv_indices = torch.arange(
                    self.kv_seq_len,
                    dtype=torch.long,
                    device=device,
                )
            groups.append(
                WanS2VStreamR1QueryGroup(
                    query_start=self.noisy_query_seq_len,
                    query_end=self.query_seq_len,
                    kv_indices=kv_indices,
                )
            )
        return groups

    def _single_noisy_block_query_groups(
        self,
        device: torch.device,
    ) -> list["WanS2VStreamR1QueryGroup"]:
        full_kv_indices = torch.arange(
            self.kv_seq_len,
            dtype=torch.long,
            device=device,
        )
        condition_kv_start = 0
        if (
            self.condition_query_seq_len > 0
            and self.condition_queries_use_current_noisy_only
        ):
            condition_kv_start = max(
                0,
                self.noisy_kv_seq_len - self.noisy_query_seq_len,
            )

        if self.condition_query_seq_len > 0 and condition_kv_start == 0:
            return [
                WanS2VStreamR1QueryGroup(
                    query_start=0,
                    query_end=self.query_seq_len,
                    kv_indices=full_kv_indices,
                    kv_ranges=((0, self.kv_seq_len),),
                )
            ]

        groups = [
            WanS2VStreamR1QueryGroup(
                query_start=0,
                query_end=self.noisy_query_seq_len,
                kv_indices=full_kv_indices,
                kv_ranges=((0, self.kv_seq_len),),
            )
        ]

        if self.condition_query_seq_len > 0:
            condition_kv_indices = torch.arange(
                condition_kv_start,
                self.kv_seq_len,
                dtype=torch.long,
                device=device,
            )
            groups.append(
                WanS2VStreamR1QueryGroup(
                    query_start=self.noisy_query_seq_len,
                    query_end=self.query_seq_len,
                    kv_indices=condition_kv_indices,
                    kv_ranges=((condition_kv_start, self.kv_seq_len),),
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
    kv_ranges: tuple[tuple[int, int], ...] | None = None

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
class WanS2VStreamR1PackedKVCopyRange:
    batch_index: int
    packed_start: int
    packed_end: int
    source_start: int
    source_end: int

    @property
    def length(self) -> int:
        return self.packed_end - self.packed_start


@dataclass(frozen=True)
class WanS2VStreamR1PackedKVCopyPlan:
    batch_indices: torch.Tensor
    packed_starts: torch.Tensor
    source_starts: torch.Tensor
    lengths: torch.Tensor
    max_length: int


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
    query_matches_input_order: bool = False


@dataclass(frozen=True)
class WanS2VStreamR1PackedAttentionMetadata:
    query_groups: tuple[WanS2VStreamR1QueryGroup, ...]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    total_query_tokens: int
    total_kv_tokens: int
    segments: tuple[WanS2VStreamR1PackedAttentionSegment, ...]
    query_matches_input_order: bool = False
    kv_copy_ranges: tuple[WanS2VStreamR1PackedKVCopyRange, ...] | None = None
    kv_copy_plan: WanS2VStreamR1PackedKVCopyPlan | None = None


_STREAM_R1_PACKED_METADATA_CACHE_LIMIT = 128
_STREAM_R1_PACKED_METADATA_CACHE: OrderedDict[
    tuple[object, ...],
    WanS2VStreamR1PackedAttentionMetadata,
] = OrderedDict()
_STREAM_R1_PACKED_BUFFER_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _stream_r1_device_cache_key(device: torch.device) -> tuple[str, int]:
    index = device.index
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return device.type, -1 if index is None else int(index)


def _stream_r1_packed_metadata_cache_key(
    plan: WanS2VStreamR1AttentionPlan,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[object, ...] | None:
    if device.type != "cuda":
        return None
    if not _stream_r1_reuse_packed_metadata_enabled():
        return None
    if not plan._uses_single_noisy_block_query_groups():
        return None
    return (
        *_stream_r1_device_cache_key(device),
        batch_size,
        plan.query_seq_len,
        plan.kv_seq_len,
        plan.noisy_query_seq_len,
        plan.noisy_kv_seq_len,
        plan.condition_kv_seq_len,
        plan.frame_seq_length,
        plan.query_block_tokens,
        plan.local_attn_size,
        plan.sink_size,
        plan.current_start,
        plan.cache_start,
        plan.condition_queries_use_current_noisy_only,
    )


def _stream_r1_get_packed_buffer(
    name: str,
    like: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor | None:
    if like.device.type != "cuda":
        return None
    if not _stream_r1_reuse_packed_buffers_enabled():
        return None
    if total_tokens <= 0:
        raise ValueError("packed attention buffer token count must be positive")
    stream_key = int(torch.cuda.current_stream(like.device).cuda_stream)
    cache_key = (
        name,
        *_stream_r1_device_cache_key(like.device),
        stream_key,
        like.dtype,
        like.shape[-2],
        like.shape[-1],
    )
    wanted_shape = (total_tokens, like.shape[-2], like.shape[-1])
    buffer = _STREAM_R1_PACKED_BUFFER_CACHE.get(cache_key)
    if (
        buffer is None
        or buffer.shape[0] < total_tokens
        or buffer.shape[1:] != wanted_shape[1:]
    ):
        # Packed buffers are mutated across denoise calls. Create a normal tensor
        # even when the request runs under torch.inference_mode(); otherwise a
        # later out= write from eager code can trip PyTorch's inference tensor
        # inplace guard.
        with torch.inference_mode(False):
            buffer = torch.empty(wanted_shape, dtype=like.dtype, device=like.device)
        _STREAM_R1_PACKED_BUFFER_CACHE[cache_key] = buffer
    return buffer[:total_tokens]


def _query_groups_match_input_order(
    query_groups: list[WanS2VStreamR1QueryGroup],
    query_seq_len: int,
) -> bool:
    query_start = 0
    for group in query_groups:
        if group.query_start != query_start:
            return False
        if group.query_end <= group.query_start:
            return False
        query_start = group.query_end
    return query_start == query_seq_len


def _flatten_query_for_packed_attention(query: torch.Tensor) -> torch.Tensor:
    if query.is_contiguous():
        return query.view(
            query.shape[0] * query.shape[1], query.shape[2], query.shape[3]
        )
    return query.reshape(
        query.shape[0] * query.shape[1],
        query.shape[2],
        query.shape[3],
    ).contiguous()


def _flatten_bshd_for_packed_attention(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
        )
    return tensor.reshape(
        tensor.shape[0] * tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
    ).contiguous()


def _packed_kv_copy_ranges_match_input_order(
    copy_ranges: tuple[WanS2VStreamR1PackedKVCopyRange, ...],
    *,
    batch_size: int,
    seq_len: int,
    total_tokens: int,
) -> bool:
    if total_tokens != batch_size * seq_len:
        return False
    if len(copy_ranges) != batch_size:
        return False
    for batch_index, copy_range in enumerate(copy_ranges):
        packed_start = batch_index * seq_len
        if copy_range.batch_index != batch_index:
            return False
        if copy_range.packed_start != packed_start:
            return False
        if copy_range.packed_end != packed_start + seq_len:
            return False
        if copy_range.source_start != 0 or copy_range.source_end != seq_len:
            return False
    return True


def _build_packed_kv_copy_plan(
    copy_ranges: list[WanS2VStreamR1PackedKVCopyRange],
    *,
    device: torch.device,
) -> WanS2VStreamR1PackedKVCopyPlan | None:
    if not copy_ranges:
        return None
    batch_indices = torch.tensor(
        [copy_range.batch_index for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    packed_starts = torch.tensor(
        [copy_range.packed_start for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    source_starts = torch.tensor(
        [copy_range.source_start for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    lengths = torch.tensor(
        [copy_range.length for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    return WanS2VStreamR1PackedKVCopyPlan(
        batch_indices=batch_indices,
        packed_starts=packed_starts,
        source_starts=source_starts,
        lengths=lengths,
        max_length=max(copy_range.length for copy_range in copy_ranges),
    )


def _select_packed_kv_parts(
    tensor: torch.Tensor,
    batch_index: int,
    group: WanS2VStreamR1QueryGroup,
) -> list[torch.Tensor]:
    if group.kv_ranges:
        parts = [
            tensor[batch_index, range_start:range_end]
            for range_start, range_end in group.kv_ranges
            if range_end > range_start
        ]
        if not parts:
            raise ValueError("packed attention KV ranges must be non-empty")
        return parts
    return [tensor[batch_index].index_select(0, group.kv_indices)]


def _split_virtual_range_for_segmented_kv(
    range_start: int,
    range_end: int,
    *,
    noisy_seq_len: int,
) -> tuple[tuple[str, int, int], ...]:
    if range_end <= range_start:
        return ()
    parts = []
    noisy_end = min(range_end, noisy_seq_len)
    if range_start < noisy_end:
        parts.append(("noisy", range_start, noisy_end))
    condition_start = max(range_start, noisy_seq_len)
    if range_end > condition_start:
        parts.append(("condition", condition_start - noisy_seq_len, range_end - noisy_seq_len))
    return tuple(parts)


def _select_segmented_packed_kv_parts(
    noisy_tensor: torch.Tensor,
    condition_tensor: torch.Tensor,
    batch_index: int,
    group: WanS2VStreamR1QueryGroup,
    *,
    noisy_seq_len: int,
) -> list[torch.Tensor]:
    if group.kv_ranges:
        parts = []
        for range_start, range_end in group.kv_ranges:
            for segment, segment_start, segment_end in _split_virtual_range_for_segmented_kv(
                range_start,
                range_end,
                noisy_seq_len=noisy_seq_len,
            ):
                source = noisy_tensor if segment == "noisy" else condition_tensor
                parts.append(source[batch_index, segment_start:segment_end])
        if not parts:
            raise ValueError("packed attention KV ranges must be non-empty")
        return parts

    kv_indices = group.kv_indices
    if kv_indices.numel() <= 0:
        raise ValueError("packed attention KV groups must be non-empty")
    if kv_indices.numel() > 1 and not torch.all(kv_indices[1:] >= kv_indices[:-1]).item():
        raise ValueError("packed attention segmented KV indices must be sorted")

    split = int(torch.searchsorted(kv_indices, noisy_seq_len).item())
    parts = []
    if split > 0:
        parts.append(noisy_tensor[batch_index].index_select(0, kv_indices[:split]))
    if split < kv_indices.numel():
        parts.append(
            condition_tensor[batch_index].index_select(
                0,
                kv_indices[split:] - noisy_seq_len,
            )
        )
    if not parts:
        raise ValueError("packed attention KV groups must be non-empty")
    return parts


def _build_wan_s2v_stream_r1_packed_attention_metadata(
    query: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
) -> WanS2VStreamR1PackedAttentionMetadata:
    query_groups = plan.query_groups(query.device)
    query_matches_input_order = _query_groups_match_input_order(
        query_groups,
        plan.query_seq_len,
    )
    cu_q = [0]
    cu_k = [0]
    segments: list[WanS2VStreamR1PackedAttentionSegment] = []
    kv_copy_ranges: list[WanS2VStreamR1PackedKVCopyRange] = []
    can_use_kv_copy_ranges = True
    max_q = 0
    max_k = 0
    for batch_index in range(query.shape[0]):
        for group in query_groups:
            if group.query_len <= 0:
                raise ValueError("packed attention query groups must be non-empty")
            if group.kv_indices.numel() <= 0:
                raise ValueError("packed attention KV groups must be non-empty")
            if group.kv_ranges:
                k_part_len = sum(
                    max(0, range_end - range_start)
                    for range_start, range_end in group.kv_ranges
                )
            else:
                can_use_kv_copy_ranges = False
                k_part_len = int(group.kv_indices.numel())
            if k_part_len <= 0:
                raise ValueError("packed attention KV groups must be non-empty")

            q_start = cu_q[-1]
            k_start = cu_k[-1]
            q_end = q_start + group.query_len
            k_end = k_start + k_part_len
            cu_q.append(q_end)
            cu_k.append(k_end)
            max_q = max(max_q, group.query_len)
            max_k = max(max_k, k_part_len)
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
            if group.kv_ranges:
                packed_offset = k_start
                for range_start, range_end in group.kv_ranges:
                    if range_end <= range_start:
                        continue
                    range_len = range_end - range_start
                    kv_copy_ranges.append(
                        WanS2VStreamR1PackedKVCopyRange(
                            batch_index=batch_index,
                            packed_start=packed_offset,
                            packed_end=packed_offset + range_len,
                            source_start=range_start,
                            source_end=range_end,
                        )
                    )
                    packed_offset += range_len
                if packed_offset != k_end:
                    raise ValueError("packed attention KV range length mismatch")

    copy_ranges_tuple = tuple(kv_copy_ranges) if can_use_kv_copy_ranges else None
    copy_plan = (
        _build_packed_kv_copy_plan(kv_copy_ranges, device=query.device)
        if can_use_kv_copy_ranges
        else None
    )
    return WanS2VStreamR1PackedAttentionMetadata(
        query_groups=tuple(query_groups),
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=query.device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=query.device),
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        total_query_tokens=cu_q[-1],
        total_kv_tokens=cu_k[-1],
        segments=tuple(segments),
        query_matches_input_order=query_matches_input_order,
        kv_copy_ranges=copy_ranges_tuple,
        kv_copy_plan=copy_plan,
    )


def _get_wan_s2v_stream_r1_packed_attention_metadata(
    query: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
) -> WanS2VStreamR1PackedAttentionMetadata:
    cache_key = _stream_r1_packed_metadata_cache_key(
        plan,
        batch_size=query.shape[0],
        device=query.device,
    )
    if cache_key is None:
        return _build_wan_s2v_stream_r1_packed_attention_metadata(query, plan)

    metadata = _STREAM_R1_PACKED_METADATA_CACHE.get(cache_key)
    if metadata is not None:
        _STREAM_R1_PACKED_METADATA_CACHE.move_to_end(cache_key)
        return metadata

    metadata = _build_wan_s2v_stream_r1_packed_attention_metadata(query, plan)
    _STREAM_R1_PACKED_METADATA_CACHE[cache_key] = metadata
    if len(_STREAM_R1_PACKED_METADATA_CACHE) > _STREAM_R1_PACKED_METADATA_CACHE_LIMIT:
        _STREAM_R1_PACKED_METADATA_CACHE.popitem(last=False)
    return metadata


def _cat_packed_attention_parts(
    parts: list[torch.Tensor],
    *,
    name: str,
    total_tokens: int,
) -> torch.Tensor:
    if not parts:
        raise ValueError("packed attention parts must be non-empty")
    buffer = _stream_r1_get_packed_buffer(name, parts[0], total_tokens)
    if buffer is None:
        return torch.cat(parts, dim=0).contiguous()
    torch.cat(parts, dim=0, out=buffer)
    return buffer


def _pack_query_for_packed_attention(
    query: torch.Tensor,
    metadata: WanS2VStreamR1PackedAttentionMetadata,
) -> torch.Tensor:
    if metadata.query_matches_input_order:
        return _flatten_query_for_packed_attention(query)
    query_parts = [
        query[segment.batch_index, segment.query_start : segment.query_end]
        for segment in metadata.segments
    ]
    return torch.cat(query_parts, dim=0).contiguous()


def _pack_packed_kv_ranges(
    tensor: torch.Tensor,
    copy_ranges: tuple[WanS2VStreamR1PackedKVCopyRange, ...],
    *,
    name: str,
    total_tokens: int,
) -> torch.Tensor:
    if _packed_kv_copy_ranges_match_input_order(
        copy_ranges,
        batch_size=tensor.shape[0],
        seq_len=tensor.shape[1],
        total_tokens=total_tokens,
    ):
        return _flatten_bshd_for_packed_attention(tensor)

    parts = []
    for copy_range in copy_ranges:
        if copy_range.length <= 0:
            raise ValueError("packed attention KV copy range must be non-empty")
        if copy_range.source_end > tensor.shape[1]:
            raise ValueError("packed attention KV copy range exceeds source length")
        parts.append(
            tensor[
                copy_range.batch_index,
                copy_range.source_start : copy_range.source_end,
            ]
        )
    return _cat_packed_attention_parts(parts, name=name, total_tokens=total_tokens)


def _pack_segmented_packed_kv_ranges(
    noisy_tensor: torch.Tensor,
    condition_tensor: torch.Tensor,
    copy_ranges: tuple[WanS2VStreamR1PackedKVCopyRange, ...],
    *,
    name: str,
    total_tokens: int,
    noisy_seq_len: int,
) -> torch.Tensor:
    if condition_tensor.shape[1] == 0 and _packed_kv_copy_ranges_match_input_order(
        copy_ranges,
        batch_size=noisy_tensor.shape[0],
        seq_len=noisy_tensor.shape[1],
        total_tokens=total_tokens,
    ):
        return _flatten_bshd_for_packed_attention(noisy_tensor)

    parts = []
    for copy_range in copy_ranges:
        dst_offset = copy_range.packed_start
        for segment, segment_start, segment_end in _split_virtual_range_for_segmented_kv(
            copy_range.source_start,
            copy_range.source_end,
            noisy_seq_len=noisy_seq_len,
        ):
            source = noisy_tensor if segment == "noisy" else condition_tensor
            length = segment_end - segment_start
            if length <= 0:
                continue
            if segment_end > source.shape[1]:
                raise ValueError(
                    "packed attention segmented KV copy range exceeds source length"
                )
            parts.append(source[copy_range.batch_index, segment_start:segment_end])
            dst_offset += length
        if dst_offset != copy_range.packed_end:
            raise ValueError("packed attention segmented KV copy length mismatch")
    return _cat_packed_attention_parts(parts, name=name, total_tokens=total_tokens)


def _try_fused_pack_segmented_kv_ranges(
    noisy_key: torch.Tensor,
    noisy_value: torch.Tensor,
    condition_key: torch.Tensor,
    condition_value: torch.Tensor,
    metadata: WanS2VStreamR1PackedAttentionMetadata,
    *,
    noisy_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    copy_ranges = metadata.kv_copy_ranges
    copy_plan = metadata.kv_copy_plan
    if copy_ranges is None or copy_plan is None:
        return None
    if not _stream_r1_fused_segmented_pack_enabled():
        return None
    if noisy_key.device.type != "cuda":
        return None
    if condition_key.shape[1] == 0 and _packed_kv_copy_ranges_match_input_order(
        copy_ranges,
        batch_size=noisy_key.shape[0],
        seq_len=noisy_key.shape[1],
        total_tokens=metadata.total_kv_tokens,
    ):
        return (
            _flatten_bshd_for_packed_attention(noisy_key),
            _flatten_bshd_for_packed_attention(noisy_value),
        )

    try:
        from sglang.jit_kernel.diffusion.triton.stream_r1_segmented_pack import (
            fused_pack_segmented_kv,
        )
    except Exception:
        return None

    packed_key = _stream_r1_get_packed_buffer(
        "stream_r1_packed_key",
        noisy_key,
        metadata.total_kv_tokens,
    )
    packed_value = _stream_r1_get_packed_buffer(
        "stream_r1_packed_value",
        noisy_value,
        metadata.total_kv_tokens,
    )
    out = None
    if packed_key is not None and packed_value is not None:
        out = (packed_key, packed_value)
    return fused_pack_segmented_kv(
        noisy_key,
        noisy_value,
        condition_key,
        condition_value,
        copy_plan.batch_indices,
        copy_plan.packed_starts,
        copy_plan.source_starts,
        copy_plan.lengths,
        total_tokens=metadata.total_kv_tokens,
        noisy_seq_len=noisy_seq_len,
        max_length=copy_plan.max_length,
        out=out,
    )


def _build_wan_s2v_stream_r1_packed_attention_workspace_from_packed_kv(
    query: torch.Tensor,
    metadata: WanS2VStreamR1PackedAttentionMetadata,
    *,
    packed_key: torch.Tensor,
    packed_value: torch.Tensor,
) -> WanS2VStreamR1PackedAttentionWorkspace:
    return WanS2VStreamR1PackedAttentionWorkspace(
        query=_pack_query_for_packed_attention(query, metadata),
        key=packed_key,
        value=packed_value,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        segments=metadata.segments,
        query_matches_input_order=metadata.query_matches_input_order,
    )


def _build_wan_s2v_stream_r1_packed_attention_workspace_from_parts(
    query: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
    *,
    select_kv_parts: Callable[
        [int, WanS2VStreamR1QueryGroup],
        tuple[list[torch.Tensor], list[torch.Tensor]],
    ],
    kv_source_shape: object,
) -> WanS2VStreamR1PackedAttentionWorkspace:
    """Pack Stream-R1 plan-visible K/V ranges as varlen attention segments."""

    if query.dim() != 4:
        raise ValueError("packed attention query must have shape [B, S, H, D]")
    if query.shape[1] != plan.query_seq_len:
        raise ValueError("packed attention query length must match attention plan")

    key_parts = []
    value_parts = []
    metadata = _get_wan_s2v_stream_r1_packed_attention_metadata(query, plan)
    segment_index = 0
    for batch_index in range(query.shape[0]):
        for group in metadata.query_groups:
            segment = metadata.segments[segment_index]
            segment_index += 1
            k_group_parts, v_group_parts = select_kv_parts(batch_index, group)
            k_part_len = sum(part.shape[0] for part in k_group_parts)
            v_part_len = sum(part.shape[0] for part in v_group_parts)
            if k_part_len <= 0 or v_part_len <= 0:
                raise ValueError("packed attention KV groups must be non-empty")
            if k_part_len != v_part_len:
                raise ValueError("packed attention key/value part lengths must match")
            if k_part_len != segment.packed_kv_end - segment.packed_kv_start:
                raise ValueError("packed attention metadata KV length mismatch")
            key_parts.extend(k_group_parts)
            value_parts.extend(v_group_parts)
    if segment_index != len(metadata.segments):
        raise ValueError("packed attention metadata segment count mismatch")

    with _stream_r1_comm_nvtx_range(
        "stream_r1_packed_workspace.query "
        f"query_matches_input_order={metadata.query_matches_input_order} "
        f"groups={len(metadata.segments)} q_shape={tuple(query.shape)}"
    ):
        packed_query = _pack_query_for_packed_attention(query, metadata)
    with _stream_r1_comm_nvtx_range(
        "stream_r1_packed_workspace.key_cat "
        f"parts={len(key_parts)} kv_seq_len={plan.kv_seq_len} shape={kv_source_shape}"
    ):
        packed_key = _cat_packed_attention_parts(
            key_parts,
            name="stream_r1_packed_key",
            total_tokens=metadata.total_kv_tokens,
        )
    with _stream_r1_comm_nvtx_range(
        "stream_r1_packed_workspace.value_cat "
        f"parts={len(value_parts)} kv_seq_len={plan.kv_seq_len} shape={kv_source_shape}"
    ):
        packed_value = _cat_packed_attention_parts(
            value_parts,
            name="stream_r1_packed_value",
            total_tokens=metadata.total_kv_tokens,
        )

    return WanS2VStreamR1PackedAttentionWorkspace(
        query=packed_query,
        key=packed_key,
        value=packed_value,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q,
        max_seqlen_k=metadata.max_seqlen_k,
        segments=metadata.segments,
        query_matches_input_order=metadata.query_matches_input_order,
    )


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
    if key.shape[1] != plan.kv_seq_len:
        raise ValueError("packed attention key length must match attention plan")

    metadata = _get_wan_s2v_stream_r1_packed_attention_metadata(query, plan)
    if metadata.kv_copy_ranges is not None:
        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_workspace.key_copy_ranges "
            f"ranges={len(metadata.kv_copy_ranges)} kv_seq_len={plan.kv_seq_len} "
            f"shape={tuple(key.shape)}"
        ):
            packed_key = _pack_packed_kv_ranges(
                key,
                metadata.kv_copy_ranges,
                name="stream_r1_packed_key",
                total_tokens=metadata.total_kv_tokens,
            )
        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_workspace.value_copy_ranges "
            f"ranges={len(metadata.kv_copy_ranges)} kv_seq_len={plan.kv_seq_len} "
            f"shape={tuple(value.shape)}"
        ):
            packed_value = _pack_packed_kv_ranges(
                value,
                metadata.kv_copy_ranges,
                name="stream_r1_packed_value",
                total_tokens=metadata.total_kv_tokens,
            )
        return _build_wan_s2v_stream_r1_packed_attention_workspace_from_packed_kv(
            query,
            metadata,
            packed_key=packed_key,
            packed_value=packed_value,
        )

    def select_kv_parts(
        batch_index: int,
        group: WanS2VStreamR1QueryGroup,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            _select_packed_kv_parts(key, batch_index, group),
            _select_packed_kv_parts(value, batch_index, group),
        )

    return _build_wan_s2v_stream_r1_packed_attention_workspace_from_parts(
        query,
        plan,
        select_kv_parts=select_kv_parts,
        kv_source_shape=tuple(key.shape),
    )


def build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
    query: torch.Tensor,
    segmented_view: "WanS2VStreamR1SegmentedMixedKVView",
    plan: WanS2VStreamR1AttentionPlan,
) -> WanS2VStreamR1PackedAttentionWorkspace:
    """Pack Stream-R1 K/V directly from segmented noisy and condition sources."""

    _validate_key_value_pair(
        segmented_view.noisy_key,
        segmented_view.noisy_value,
        name="segmented packed noisy K/V",
    )
    _validate_key_value_pair(
        segmented_view.condition_key,
        segmented_view.condition_value,
        name="segmented packed condition K/V",
    )
    _validate_compatible_kv_prefix(
        segmented_view.noisy_key,
        segmented_view.condition_key,
    )
    if query.dim() != 4:
        raise ValueError("packed attention query must have shape [B, S, H, D]")
    if (
        query.shape[0] != segmented_view.noisy_key.shape[0]
        or query.shape[2:] != segmented_view.noisy_key.shape[2:]
    ):
        raise ValueError("packed attention query/key batch/head dimensions must match")
    if segmented_view.total_seq_len != plan.kv_seq_len:
        raise ValueError("segmented packed K/V length must match attention plan")

    metadata = _get_wan_s2v_stream_r1_packed_attention_metadata(query, plan)
    if metadata.kv_copy_ranges is not None:
        with _stream_r1_comm_nvtx_range(
            "stream_r1_segmented_packed_workspace.fused_kv_pack "
            f"ranges={len(metadata.kv_copy_ranges)} kv_seq_len={plan.kv_seq_len} "
            f"noisy={tuple(segmented_view.noisy_key.shape)} "
            f"condition={tuple(segmented_view.condition_key.shape)}"
        ):
            fused_packed_kv = _try_fused_pack_segmented_kv_ranges(
                segmented_view.noisy_key,
                segmented_view.noisy_value,
                segmented_view.condition_key,
                segmented_view.condition_value,
                metadata,
                noisy_seq_len=segmented_view.cached_noisy_seq_len,
            )
        if fused_packed_kv is not None:
            packed_key, packed_value = fused_packed_kv
            return _build_wan_s2v_stream_r1_packed_attention_workspace_from_packed_kv(
                query,
                metadata,
                packed_key=packed_key,
                packed_value=packed_value,
            )

        with _stream_r1_comm_nvtx_range(
            "stream_r1_segmented_packed_workspace.key_copy_ranges "
            f"ranges={len(metadata.kv_copy_ranges)} kv_seq_len={plan.kv_seq_len} "
            f"noisy={tuple(segmented_view.noisy_key.shape)} "
            f"condition={tuple(segmented_view.condition_key.shape)}"
        ):
            packed_key = _pack_segmented_packed_kv_ranges(
                segmented_view.noisy_key,
                segmented_view.condition_key,
                metadata.kv_copy_ranges,
                name="stream_r1_packed_key",
                total_tokens=metadata.total_kv_tokens,
                noisy_seq_len=segmented_view.cached_noisy_seq_len,
            )
        with _stream_r1_comm_nvtx_range(
            "stream_r1_segmented_packed_workspace.value_copy_ranges "
            f"ranges={len(metadata.kv_copy_ranges)} kv_seq_len={plan.kv_seq_len} "
            f"noisy={tuple(segmented_view.noisy_value.shape)} "
            f"condition={tuple(segmented_view.condition_value.shape)}"
        ):
            packed_value = _pack_segmented_packed_kv_ranges(
                segmented_view.noisy_value,
                segmented_view.condition_value,
                metadata.kv_copy_ranges,
                name="stream_r1_packed_value",
                total_tokens=metadata.total_kv_tokens,
                noisy_seq_len=segmented_view.cached_noisy_seq_len,
            )
        return _build_wan_s2v_stream_r1_packed_attention_workspace_from_packed_kv(
            query,
            metadata,
            packed_key=packed_key,
            packed_value=packed_value,
        )

    def select_kv_parts(
        batch_index: int,
        group: WanS2VStreamR1QueryGroup,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            _select_segmented_packed_kv_parts(
                segmented_view.noisy_key,
                segmented_view.condition_key,
                batch_index,
                group,
                noisy_seq_len=segmented_view.cached_noisy_seq_len,
            ),
            _select_segmented_packed_kv_parts(
                segmented_view.noisy_value,
                segmented_view.condition_value,
                batch_index,
                group,
                noisy_seq_len=segmented_view.cached_noisy_seq_len,
            ),
        )

    return _build_wan_s2v_stream_r1_packed_attention_workspace_from_parts(
        query,
        plan,
        select_kv_parts=select_kv_parts,
        kv_source_shape=(
            tuple(segmented_view.noisy_key.shape),
            tuple(segmented_view.condition_key.shape),
        ),
    )


def _unpack_wan_s2v_stream_r1_packed_attention(
    packed_output: torch.Tensor,
    workspace: WanS2VStreamR1PackedAttentionWorkspace,
    output_shape: torch.Size,
) -> torch.Tensor:
    if workspace.query_matches_input_order:
        return packed_output.reshape(output_shape)
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
        q = (
            workspace.query[segment.packed_query_start : segment.packed_query_end]
            .transpose(0, 1)
            .unsqueeze(0)
        )
        k = (
            workspace.key[segment.packed_kv_start : segment.packed_kv_end]
            .transpose(0, 1)
            .unsqueeze(0)
        )
        v = (
            workspace.value[segment.packed_kv_start : segment.packed_kv_end]
            .transpose(0, 1)
            .unsqueeze(0)
        )
        out = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        output[
            segment.batch_index,
            segment.query_start : segment.query_end,
        ] = out
    return output


def _run_stream_r1_packed_varlen_attention_workspace(
    query: torch.Tensor,
    workspace: WanS2VStreamR1PackedAttentionWorkspace,
    *,
    softmax_scale: float | None,
    force_torch: bool = False,
) -> torch.Tensor:
    if force_torch or query.device.type != "cuda":
        return _run_wan_s2v_stream_r1_packed_torch_attention(
            workspace,
            query.shape,
            softmax_scale=softmax_scale,
        )

    flash_attention_version = _stream_r1_flash_attention_version()
    if flash_attention_version == "sm120":
        try:
            from flash_attn_4_sm120 import flash_attn_varlen_func
        except Exception as exc:  # pragma: no cover - depends on optional kernels
            raise RuntimeError(
                "SGLANG_STREAM_R1_FLASH_ATTENTION_VERSION=sm120 requires "
                "flash_attn_4_sm120 on PYTHONPATH"
            ) from exc

        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_attention.fa4_sm120 "
            f"max_q={workspace.max_seqlen_q} max_k={workspace.max_seqlen_k}"
        ):
            result = flash_attn_varlen_func(
                workspace.query,
                workspace.key,
                workspace.value,
                cu_seqlens_q=workspace.cu_seqlens_q,
                cu_seqlens_k=workspace.cu_seqlens_k,
                max_seqlen_q=workspace.max_seqlen_q,
                max_seqlen_k=workspace.max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=False,
            )
    else:
        try:
            from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
        except Exception as exc:  # pragma: no cover - depends on optional kernels
            raise RuntimeError(
                "SGLANG_STREAM_R1_ATTENTION_BACKEND=packed_varlen requires "
                "flash_attn_varlen_func on CUDA"
            ) from exc

        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_attention.fa "
            f"version={flash_attention_version} "
            f"max_q={workspace.max_seqlen_q} max_k={workspace.max_seqlen_k}"
        ):
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
                ver=int(flash_attention_version),
            )
    packed_output = result[0] if isinstance(result, tuple) else result
    with _stream_r1_comm_nvtx_range(
        "stream_r1_packed_attention.output_unpack "
        f"packed={tuple(packed_output.shape)} out={tuple(query.shape)}"
    ):
        return _unpack_wan_s2v_stream_r1_packed_attention(
            packed_output,
            workspace,
            query.shape,
        )


def stream_r1_packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WanS2VStreamR1AttentionPlan,
    *,
    softmax_scale: float | None,
    force_torch: bool = False,
) -> torch.Tensor:
    """Run Stream-R1 packed varlen attention for one materialized mixed-KV call."""

    with _stream_r1_comm_nvtx_range(
        "stream_r1_packed_attention.build_workspace "
        f"q={tuple(query.shape)} k={tuple(key.shape)}"
    ):
        workspace = build_wan_s2v_stream_r1_packed_attention_workspace(
            query,
            key,
            value,
            plan,
        )
    return _run_stream_r1_packed_varlen_attention_workspace(
        query,
        workspace,
        softmax_scale=softmax_scale,
        force_torch=force_torch,
    )


def stream_r1_segmented_packed_varlen_attention(
    query: torch.Tensor,
    segmented_view: "WanS2VStreamR1SegmentedMixedKVView",
    plan: WanS2VStreamR1AttentionPlan,
    *,
    softmax_scale: float | None,
    force_torch: bool = False,
) -> torch.Tensor:
    """Run Stream-R1 packed varlen attention without materializing mixed K/V."""

    with _stream_r1_comm_nvtx_range(
        "stream_r1_segmented_packed_attention.build_workspace "
        f"q={tuple(query.shape)} noisy={tuple(segmented_view.noisy_key.shape)} "
        f"condition={tuple(segmented_view.condition_key.shape)}"
    ):
        workspace = build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
            query,
            segmented_view,
            plan,
        )
    return _run_stream_r1_packed_varlen_attention_workspace(
        query,
        workspace,
        softmax_scale=softmax_scale,
        force_torch=force_torch,
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
class WanS2VStreamR1SegmentedMixedKVView:
    noisy_key: torch.Tensor
    noisy_value: torch.Tensor
    condition_key: torch.Tensor
    condition_value: torch.Tensor
    global_end_index: int
    local_end_index: int
    local_start: int
    local_end: int

    @property
    def cached_noisy_seq_len(self) -> int:
        return self.noisy_key.shape[1]

    @property
    def condition_seq_len(self) -> int:
        return self.condition_key.shape[1]

    @property
    def condition_start_index(self) -> int:
        return self.cached_noisy_seq_len

    @property
    def total_seq_len(self) -> int:
        return self.cached_noisy_seq_len + self.condition_seq_len


def update_wan_s2v_stream_r1_noisy_kv_cache(
    kv_cache: WanS2VKVCacheBlock,
    key: torch.Tensor,
    value: torch.Tensor,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
) -> WanS2VStreamR1NoisyKVCacheView:
    """Mutate one layer's noisy-token KV cache using Stream-R1 S2V windows."""

    _validate_noisy_kv_update_inputs(kv_cache, key, value, update)

    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]
    global_end = int(kv_cache["global_end_index"].item())
    local_end = int(kv_cache["local_end_index"].item())
    if global_end == 0 and local_end == 0 and update.cache_start > 0:
        global_end = update.cache_start
    if global_end < update.cache_start:
        raise ValueError("KV cache global_end_index is before cache_start")
    if local_end < 0 or local_end > cache_k.shape[1]:
        raise ValueError("KV cache local_end_index is outside cache capacity")
    if local_end > update.required_cache_tokens:
        raise ValueError("KV cache local_end_index exceeds the Stream-R1 local window")
    if update.current_start > global_end:
        raise ValueError("KV cache update cannot skip noisy-token ranges")
    if update.current_end < global_end:
        raise ValueError("KV cache update cannot move global_end_index backwards")

    if (
        update.current_end > global_end
        and update.noisy_seq_len + local_end > cache_k.shape[1]
    ):
        num_evicted_tokens = update.noisy_seq_len + local_end - cache_k.shape[1]
        num_rolled_tokens = max(0, local_end - num_evicted_tokens - update.sink_tokens)
        evicted_start = update.sink_tokens
        evicted_end = update.sink_tokens + num_evicted_tokens
        evicted_k = cache_k[:, evicted_start:evicted_end].clone()
        evicted_v = cache_v[:, evicted_start:evicted_end].clone()

        if num_rolled_tokens > 0:
            cache_k[:, update.sink_tokens : update.sink_tokens + num_rolled_tokens] = (
                cache_k[
                    :,
                    update.sink_tokens
                    + num_evicted_tokens : update.sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
            )
            cache_v[:, update.sink_tokens : update.sink_tokens + num_rolled_tokens] = (
                cache_v[
                    :,
                    update.sink_tokens
                    + num_evicted_tokens : update.sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
            )

        local_end = local_end + update.current_end - global_end - num_evicted_tokens
        if update.sink_tokens > 0 and evicted_k.numel() > 0:
            if evicted_k.shape[1] == update.sink_tokens:
                cache_k[:, : update.sink_tokens] = (
                    _STREAM_R1_SINK_COMPRESSION_ALPHA * cache_k[:, : update.sink_tokens]
                    + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_k
                )
                cache_v[:, : update.sink_tokens] = (
                    _STREAM_R1_SINK_COMPRESSION_ALPHA * cache_v[:, : update.sink_tokens]
                    + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_v
                )
            else:
                copy_len = min(evicted_k.shape[1], update.sink_tokens)
                cache_k[:, :copy_len] = evicted_k[:, :copy_len]
                cache_v[:, :copy_len] = evicted_v[:, :copy_len]
    else:
        local_end = local_end + update.current_end - global_end

    local_write_start = local_end - update.noisy_seq_len
    cache_k[:, local_write_start:local_end] = key[:, : update.noisy_seq_len]
    cache_v[:, local_write_start:local_end] = value[:, : update.noisy_seq_len]

    kv_start = max(
        update.sink_tokens,
        local_end - update.local_tokens + update.sink_tokens,
    )
    if update.sink_tokens > 0:
        if kv_start == update.sink_tokens:
            view_key = cache_k[:, :local_end]
            view_value = cache_v[:, :local_end]
        elif kv_start >= local_end:
            view_key = cache_k[:, : update.sink_tokens]
            view_value = cache_v[:, : update.sink_tokens]
        else:
            with _stream_r1_comm_nvtx_range(
                "stream_r1_noisy_kv_view.sink_cat "
                f"sink_tokens={update.sink_tokens} kv_start={kv_start} local_end={local_end}"
            ):
                view_key = torch.cat(
                    [cache_k[:, : update.sink_tokens], cache_k[:, kv_start:local_end]],
                    dim=1,
                )
                view_value = torch.cat(
                    [cache_v[:, : update.sink_tokens], cache_v[:, kv_start:local_end]],
                    dim=1,
                )
    else:
        view_key = cache_k[:, kv_start:local_end]
        view_value = cache_v[:, kv_start:local_end]

    local_suffix_len = max(0, local_end - kv_start)
    local_start = update.current_end - local_suffix_len
    local_end_index = int(view_key.shape[1])
    kv_cache["global_end_index"].fill_(update.current_end)
    kv_cache["local_end_index"].fill_(local_end_index)
    return WanS2VStreamR1NoisyKVCacheView(
        key=view_key,
        value=view_value,
        global_end_index=update.current_end,
        local_end_index=local_end_index,
        local_start=local_start,
        local_end=update.current_end,
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
        with _stream_r1_comm_nvtx_range(
            "stream_r1_mixed_kv.compose_cat "
            f"cached_noisy={noisy_view.key.shape[1]} "
            f"condition={current_kv.condition_seq_len}"
        ):
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


def compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    current_kv: WanS2VStreamR1ProjectedKVSplit,
) -> WanS2VStreamR1SegmentedMixedKVView:
    """Compose cached noisy K/V and condition K/V without materializing mixed K/V."""

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

    return WanS2VStreamR1SegmentedMixedKVView(
        noisy_key=noisy_view.key,
        noisy_value=noisy_view.value,
        condition_key=current_kv.condition_key,
        condition_value=current_kv.condition_value,
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


def _build_wan_s2v_stream_r1_mixed_kv_attention_plan_from_lengths(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    *,
    cached_noisy_seq_len: int,
    condition_seq_len: int,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    device: torch.device,
) -> WanS2VStreamR1AttentionPlan:
    cached_noisy_index = build_wan_s2v_stream_r1_cached_noisy_kv_index(
        noisy_view,
        update,
        device=device,
    )
    return WanS2VStreamR1AttentionPlan(
        query_seq_len=update.noisy_seq_len + condition_seq_len,
        kv_seq_len=cached_noisy_seq_len + condition_seq_len,
        noisy_query_seq_len=update.noisy_seq_len,
        noisy_kv_seq_len=cached_noisy_seq_len,
        condition_kv_seq_len=condition_seq_len,
        frame_seq_length=update.frame_seq_length,
        query_block_tokens=update.noisy_seq_len,
        local_attn_size=update.local_attn_size,
        sink_size=update.sink_size,
        current_start=update.current_start,
        cache_start=update.cache_start,
        noisy_kv_absolute_index=cached_noisy_index,
        condition_queries_use_current_noisy_only=True,
    )


def build_wan_s2v_stream_r1_mixed_kv_attention_plan(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    mixed_view: WanS2VStreamR1MixedKVView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    device: torch.device | None = None,
) -> WanS2VStreamR1AttentionPlan:
    """Build a compact mixed-KV attention plan for Stream-R1 S2V."""

    _validate_mixed_view_for_noisy_view(noisy_view, mixed_view)
    return _build_wan_s2v_stream_r1_mixed_kv_attention_plan_from_lengths(
        noisy_view,
        cached_noisy_seq_len=mixed_view.cached_noisy_seq_len,
        condition_seq_len=mixed_view.condition_seq_len,
        update=update,
        device=device or mixed_view.key.device,
    )


def build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    segmented_view: WanS2VStreamR1SegmentedMixedKVView,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    device: torch.device | None = None,
) -> WanS2VStreamR1AttentionPlan:
    """Build a compact mixed-KV attention plan for segmented Stream-R1 S2V K/V."""

    _validate_segmented_mixed_view_for_noisy_view(noisy_view, segmented_view)
    return _build_wan_s2v_stream_r1_mixed_kv_attention_plan_from_lengths(
        noisy_view,
        cached_noisy_seq_len=segmented_view.cached_noisy_seq_len,
        condition_seq_len=segmented_view.condition_seq_len,
        update=update,
        device=device or segmented_view.noisy_key.device,
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
                    query,
                    key,
                    value,
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
    use_segmented_packed_attention = use_sp_head_sharded_packed_attention or (
        attention_backend == "packed_varlen" and not sequence_shard_enabled
    )
    if use_segmented_packed_attention:
        with profile.span("segmented_mixed_kv_compose"):
            segmented_mixed_view = compose_wan_s2v_stream_r1_segmented_mixed_kv_view(
                noisy_view,
                current_kv,
            )
        with profile.span("attention_plan"):
            mixed_plan = build_wan_s2v_stream_r1_segmented_mixed_kv_attention_plan(
                noisy_view,
                segmented_mixed_view,
                update,
                device=query_for_attention.device,
            )
    else:
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
            output = stream_r1_segmented_packed_varlen_attention(
                query_for_attention,
                segmented_mixed_view,
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
            output = _usp_output_all_to_all(output, head_dim=2)
    elif attention_backend == "packed_varlen" and not sequence_shard_enabled:
        with profile.span("packed_varlen_attention"):
            output = stream_r1_segmented_packed_varlen_attention(
                query_for_attention,
                segmented_mixed_view,
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
        mixed_kv_seq_len=mixed_plan.kv_seq_len,
        cached_noisy_seq_len=mixed_plan.noisy_kv_seq_len,
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
                "Wan S2V crossattn_cache is only supported when " "stream_r1_mode=True"
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
        raise ValueError("Wan S2V kv_cache is only supported when stream_r1_mode=True")
    if not isinstance(kv_cache, list):
        raise ValueError("Wan S2V kv_cache must be a list of per-block cache entries")
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
        raise ValueError("key/value sequence length must match update.noisy_seq_len")
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


def _validate_segmented_mixed_view_for_noisy_view(
    noisy_view: WanS2VStreamR1NoisyKVCacheView,
    segmented_view: WanS2VStreamR1SegmentedMixedKVView,
) -> None:
    _validate_key_value_pair(
        segmented_view.noisy_key,
        segmented_view.noisy_value,
        name="segmented noisy K/V",
    )
    _validate_key_value_pair(
        segmented_view.condition_key,
        segmented_view.condition_value,
        name="segmented condition K/V",
    )
    if segmented_view.cached_noisy_seq_len != noisy_view.key.shape[1]:
        raise ValueError(
            "segmented mixed K/V cached_noisy_seq_len must match the noisy view length"
        )
    if segmented_view.global_end_index != noisy_view.global_end_index:
        raise ValueError("segmented mixed K/V global_end_index must match the noisy view")
    if segmented_view.local_end_index != noisy_view.local_end_index:
        raise ValueError("segmented mixed K/V local_end_index must match the noisy view")
    if segmented_view.local_start != noisy_view.local_start:
        raise ValueError("segmented mixed K/V local_start must match the noisy view")
    if segmented_view.local_end != noisy_view.local_end:
        raise ValueError("segmented mixed K/V local_end must match the noisy view")
    _validate_compatible_kv_prefix(segmented_view.noisy_key, noisy_view.key)
    _validate_compatible_kv_prefix(
        segmented_view.noisy_key,
        segmented_view.condition_key,
    )


def _noisy_view_local_len(noisy_view: WanS2VStreamR1NoisyKVCacheView) -> int:
    return max(0, noisy_view.local_end - noisy_view.local_start)
