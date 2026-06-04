# SPDX-License-Identifier: Apache-2.0
"""Lightweight Stream-R1 helpers for Wan S2V attention."""

from dataclasses import dataclass
from typing import TypedDict

import torch


class WanS2VKVCacheBlock(TypedDict):
    k: torch.Tensor
    v: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor


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

    def build_no_kv_attention_mask(self, device: torch.device) -> torch.Tensor:
        """Build the Stream-R1 no-KV local/sink mask for mixed S2V tokens.

        Noisy latent queries are restricted to sink noisy tokens, their local
        block window, and all current condition tokens. Condition queries keep
        dense attention so reference/motion tokens retain legacy semantics.
        """

        q_idx = torch.arange(self.total_seq_len, device=device).view(-1, 1)
        kv_idx = torch.arange(self.total_seq_len, device=device).view(1, -1)
        noisy_q = q_idx < self.noisy_seq_len
        noisy_kv = kv_idx < self.noisy_seq_len
        condition_kv = kv_idx >= self.noisy_seq_len

        q_abs = self.current_start + q_idx
        kv_abs = self.current_start + kv_idx
        block_end = (
            torch.div(q_abs, self.block_tokens, rounding_mode="floor") + 1
        ) * self.block_tokens
        local_start = block_end - self.local_tokens

        sink_visible = noisy_kv & (kv_abs < self.sink_tokens)
        local_visible = noisy_kv & (kv_abs >= local_start) & (kv_abs < block_end)
        noisy_query_visible = condition_kv | sink_visible | local_visible

        return torch.where(
            noisy_q,
            noisy_query_visible,
            torch.ones_like(noisy_query_visible),
        ).unsqueeze(0)


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


def _validate_noisy_kv_update_inputs(
    kv_cache: WanS2VKVCacheBlock,
    key: torch.Tensor,
    value: torch.Tensor,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
) -> None:
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.dim() != 4:
        raise ValueError("key/value must have shape [B, S, H, D]")
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
