# SPDX-License-Identifier: Apache-2.0
"""Lightweight Stream-R1 helpers for Wan S2V attention."""

import os
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, NotRequired, TypedDict

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_qkv,
    _usp_output_all_to_all,
    _usp_output_all_to_all_packed_bshd,
)

_STREAM_R1_SINK_COMPRESSION_ALPHA = 0.999


@dataclass
class WanS2VStreamR1KVState:
    """Host-owned Stream-R1 noisy KV cache state for one transformer layer."""

    global_end_index: int = 0
    local_end_index: int = 0

    @classmethod
    def from_cache_block(
        cls, kv_cache: "WanS2VKVCacheBlock"
    ) -> "WanS2VStreamR1KVState":
        return cls(
            global_end_index=_kv_cache_host_index(
                kv_cache,
                tensor_key="global_end_index",
                host_key="global_end_index_host",
            ),
            local_end_index=_kv_cache_host_index(
                kv_cache,
                tensor_key="local_end_index",
                host_key="local_end_index_host",
            ),
        )

    def reset(self) -> None:
        self.global_end_index = 0
        self.local_end_index = 0

    def apply_noisy_kv_cache_update_plan(
        self, plan: "WanS2VStreamR1NoisyKVCacheUpdatePlan"
    ) -> None:
        if self.global_end_index != plan.input_global_end:
            raise ValueError(
                "KV state global_end_index does not match update plan: "
                f"state={self.global_end_index}, plan={plan.input_global_end}"
            )
        if self.local_end_index != plan.input_local_end:
            raise ValueError(
                "KV state local_end_index does not match update plan: "
                f"state={self.local_end_index}, plan={plan.input_local_end}"
            )
        self.global_end_index = plan.new_global_end
        self.local_end_index = plan.new_local_end_index


class WanS2VKVCacheBlock(TypedDict):
    k: torch.Tensor
    v: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor
    global_end_index_host: NotRequired[int]
    local_end_index_host: NotRequired[int]
    state: NotRequired[WanS2VStreamR1KVState]
    update_plan_buffer: NotRequired["WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer"]


def get_wan_s2v_stream_r1_kv_cache_state(
    kv_cache: WanS2VKVCacheBlock,
) -> WanS2VStreamR1KVState:
    state = kv_cache.get("state")
    if state is not None:
        if not isinstance(state, WanS2VStreamR1KVState):
            raise TypeError(
                "Wan S2V Stream-R1 KV cache state must be "
                "WanS2VStreamR1KVState"
            )
        return state
    return WanS2VStreamR1KVState.from_cache_block(kv_cache)


def sync_wan_s2v_stream_r1_kv_cache_host_shadow(
    kv_cache: WanS2VKVCacheBlock,
    state: WanS2VStreamR1KVState,
) -> None:
    kv_cache["global_end_index_host"] = int(state.global_end_index)
    kv_cache["local_end_index_host"] = int(state.local_end_index)


@dataclass(frozen=True)
class WanS2VTimestepMetadataPlan:
    """Dynamic scalar metadata for one Wan transformer timestep."""

    step_index: int
    current_start: int
    cache_start: int | None
    audio_start_frame: int | None
    sequence_shard_enabled: bool
    motion_frames: tuple[int, ...]
    add_last_motion: int
    drop_motion_frames: bool
    stream_r1_mode: bool

    @classmethod
    def from_kwargs(
        cls,
        *,
        kwargs: dict[str, Any],
        step_index: int,
        current_start: int,
        audio_start_frame: int | None,
        sequence_shard_enabled: bool,
    ) -> "WanS2VTimestepMetadataPlan":
        motion_frames = kwargs.get("motion_frames", ())
        return cls(
            step_index=int(step_index),
            current_start=int(current_start),
            cache_start=(
                None if kwargs.get("cache_start") is None else int(kwargs["cache_start"])
            ),
            audio_start_frame=(
                None if audio_start_frame is None else int(audio_start_frame)
            ),
            sequence_shard_enabled=bool(sequence_shard_enabled),
            motion_frames=tuple(int(item) for item in motion_frames),
            add_last_motion=int(kwargs.get("add_last_motion", 0)),
            drop_motion_frames=bool(kwargs.get("drop_motion_frames", False)),
            stream_r1_mode=bool(kwargs.get("stream_r1_mode", False)),
        )

    @property
    def dynamic_key_signature(self) -> tuple[Any, ...]:
        return (
            int(self.current_start),
            -1 if self.cache_start is None else int(self.cache_start),
            -1 if self.audio_start_frame is None else int(self.audio_start_frame),
            bool(self.sequence_shard_enabled),
            tuple(self.motion_frames),
            int(self.add_last_motion),
            bool(self.drop_motion_frames),
            bool(self.stream_r1_mode),
        )

    @property
    def structure_key_signature(self) -> tuple[Any, ...]:
        return (
            bool(self.sequence_shard_enabled),
            tuple(self.motion_frames),
            int(self.add_last_motion),
            bool(self.drop_motion_frames),
            bool(self.stream_r1_mode),
        )


class WanS2VTimestepStaticMetadataBuffers:
    """Stable-address scalar metadata buffers for one Wan timestep graph entry."""

    SCALAR_KEYS = (
        "step_index",
        "current_start",
        "cache_start",
        "audio_start_frame",
        "sequence_shard_enabled",
        "add_last_motion",
        "drop_motion_frames",
        "stream_r1_mode",
    )
    SCALAR_INDEX = {key: index for index, key in enumerate(SCALAR_KEYS)}
    NONE_SENTINEL = -1

    def __init__(
        self,
        *,
        scalar_values: torch.Tensor,
        motion_frames: torch.Tensor,
        host_plan: WanS2VTimestepMetadataPlan | None = None,
    ) -> None:
        self.scalar_values = scalar_values
        self.motion_frames = motion_frames
        self._host_plan = host_plan

    @staticmethod
    def device_from_kwargs(kwargs: dict[str, Any]) -> torch.device:
        hidden_states = kwargs.get("hidden_states")
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states.device
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @classmethod
    def _scalar_values_from_plan(
        cls, plan: WanS2VTimestepMetadataPlan
    ) -> tuple[int, ...]:
        return (
            int(plan.step_index),
            int(plan.current_start),
            cls.NONE_SENTINEL if plan.cache_start is None else int(plan.cache_start),
            cls.NONE_SENTINEL
            if plan.audio_start_frame is None
            else int(plan.audio_start_frame),
            int(plan.sequence_shard_enabled),
            int(plan.add_last_motion),
            int(plan.drop_motion_frames),
            int(plan.stream_r1_mode),
        )

    @classmethod
    def from_plan(
        cls,
        plan: WanS2VTimestepMetadataPlan,
        *,
        device: torch.device,
    ) -> "WanS2VTimestepStaticMetadataBuffers":
        with torch.inference_mode(False):
            buffers = cls(
                scalar_values=torch.empty(
                    (len(cls.SCALAR_KEYS),),
                    dtype=torch.long,
                    device=device,
                ),
                motion_frames=torch.empty(
                    (len(plan.motion_frames),),
                    dtype=torch.long,
                    device=device,
                ),
            )
        buffers.copy_from_plan_(plan)
        return buffers

    @classmethod
    def signature_from_plan(
        cls,
        plan: WanS2VTimestepMetadataPlan,
        *,
        device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            (len(cls.SCALAR_KEYS), str(torch.long), str(device)),
            (len(plan.motion_frames), str(torch.long), str(device)),
        )

    def copy_from_plan_(self, plan: WanS2VTimestepMetadataPlan) -> None:
        if self.motion_frames.numel() != len(plan.motion_frames):
            raise ValueError("Wan timestep metadata motion_frames length mismatch")
        self.scalar_values.copy_(
            self.scalar_values.new_tensor(self._scalar_values_from_plan(plan))
        )
        self.motion_frames.copy_(self.motion_frames.new_tensor(plan.motion_frames))
        self._host_plan = plan

    def scalar_snapshot(self) -> dict[str, int]:
        values = self.scalar_values.detach().cpu().tolist()
        return dict(zip(self.SCALAR_KEYS, values))

    def scalar_tensor(self, key: str) -> torch.Tensor:
        try:
            index = self.SCALAR_INDEX[key]
        except KeyError as exc:
            raise KeyError(f"unknown Wan timestep metadata scalar {key!r}") from exc
        return self.scalar_values[index]

    def to_forward_kwargs(self) -> dict[str, Any]:
        """Materialize Python kwargs for the legacy Wan forward path.

        This adapter is intentionally not the final graph body: while the
        current transformer still needs Python scalars for slicing/control flow,
        CUDA buffers cannot be materialized during graph capture.
        """

        plan = self._host_plan
        if plan is not None:
            return {
                "current_start": int(plan.current_start),
                "cache_start": plan.cache_start,
                "audio_start_frame": plan.audio_start_frame,
                "motion_frames": tuple(int(item) for item in plan.motion_frames),
                "add_last_motion": int(plan.add_last_motion),
                "drop_motion_frames": bool(plan.drop_motion_frames),
                "stream_r1_mode": bool(plan.stream_r1_mode),
            }

        if self.scalar_values.device.type == "cuda" and _cuda_graph_capture_active():
            raise RuntimeError(
                "Wan timestep metadata buffers cannot be materialized as Python "
                "scalars during CUDA graph capture"
            )
        scalars = self.scalar_snapshot()
        motion_frames = tuple(int(item) for item in self.motion_frames.cpu().tolist())
        cache_start = int(scalars["cache_start"])
        audio_start_frame = int(scalars["audio_start_frame"])
        return {
            "current_start": int(scalars["current_start"]),
            "cache_start": None
            if cache_start == self.NONE_SENTINEL
            else cache_start,
            "audio_start_frame": None
            if audio_start_frame == self.NONE_SENTINEL
            else audio_start_frame,
            "motion_frames": motion_frames,
            "add_last_motion": int(scalars["add_last_motion"]),
            "drop_motion_frames": bool(scalars["drop_motion_frames"]),
            "stream_r1_mode": bool(scalars["stream_r1_mode"]),
        }


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


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _kv_cache_host_index(
    kv_cache: WanS2VKVCacheBlock,
    *,
    tensor_key: str,
    host_key: str,
) -> int:
    if host_key in kv_cache:
        return int(kv_cache[host_key])
    if _cuda_graph_capture_active():
        raise RuntimeError(
            f"KV cache host shadow {host_key!r} is required during CUDA graph capture"
        )
    return int(kv_cache[tensor_key].item())


def _graph_safe_int_tensor(
    values: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if torch.device(device).type == "cuda" and _cuda_graph_capture_active():
        if not values:
            return torch.empty((0,), dtype=dtype, device=device)
        return torch.stack(
            [
                torch.full((), int(value), dtype=dtype, device=device)
                for value in values
            ]
        )
    return torch.tensor(values, dtype=dtype, device=device)


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


def _stream_r1_fa4_sm120_tile_mn() -> tuple[int, int] | None:
    value = os.getenv("SGLANG_STREAM_R1_FA4_SM120_TILE_MN", "").strip().lower()
    if not value:
        return None
    normalized = value.replace("x", ",").replace(" ", ",")
    parts = [item for item in normalized.split(",") if item]
    if len(parts) != 2:
        raise ValueError(
            "SGLANG_STREAM_R1_FA4_SM120_TILE_MN must use MxN syntax, "
            f"got {value!r}"
        )
    try:
        tile_m, tile_n = (int(item) for item in parts)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_STREAM_R1_FA4_SM120_TILE_MN must contain integers, "
            f"got {value!r}"
        ) from exc
    if tile_m <= 0 or tile_n <= 0 or tile_m % 16 or tile_n % 16:
        raise ValueError(
            "SGLANG_STREAM_R1_FA4_SM120_TILE_MN dimensions must be positive "
            f"multiples of 16, got {(tile_m, tile_n)}"
        )
    return tile_m, tile_n


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
        self._use_cuda_events = False

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._use_cuda_events = bool(
            self.profile.use_cuda_events and not _cuda_graph_capture_active()
        )
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._use_cuda_events and not _cuda_graph_capture_active():
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
    batch_indices = _graph_safe_int_tensor(
        [copy_range.batch_index for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    packed_starts = _graph_safe_int_tensor(
        [copy_range.packed_start for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    source_starts = _graph_safe_int_tensor(
        [copy_range.source_start for copy_range in copy_ranges],
        dtype=torch.int64,
        device=device,
    )
    lengths = _graph_safe_int_tensor(
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
        cu_seqlens_q=_graph_safe_int_tensor(
            cu_q, dtype=torch.int32, device=query.device
        ),
        cu_seqlens_k=_graph_safe_int_tensor(
            cu_k, dtype=torch.int32, device=query.device
        ),
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
    return_packed_output: bool = False,
) -> torch.Tensor:
    if force_torch or query.device.type != "cuda":
        if return_packed_output:
            raise ValueError("packed output fast path requires CUDA varlen attention")
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

        tile_mn = _stream_r1_fa4_sm120_tile_mn()
        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_attention.fa4_sm120 "
            f"max_q={workspace.max_seqlen_q} max_k={workspace.max_seqlen_k} "
            f"tile_mn={tile_mn or 'default'}"
        ):
            if tile_mn is None:
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
                from flash_attn_4_sm120.interface import _flash_attn_fwd

                result = _flash_attn_fwd(
                    workspace.query,
                    workspace.key,
                    workspace.value,
                    cu_seqlens_q=workspace.cu_seqlens_q,
                    cu_seqlens_k=workspace.cu_seqlens_k,
                    max_seqlen_q=workspace.max_seqlen_q,
                    max_seqlen_k=workspace.max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=False,
                    tile_mn=tile_mn,
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
    if return_packed_output:
        if not workspace.query_matches_input_order:
            raise ValueError("packed output fast path requires input-order query")
        return packed_output
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


def _pad_stream_r1_sp_packed_attention_output(
    packed_output: torch.Tensor,
    *,
    batch_size: int,
    total_seq_len: int,
    sp_pad_tokens: int,
) -> torch.Tensor:
    if sp_pad_tokens <= 0:
        return packed_output
    if batch_size != 1:
        raise ValueError("packed SP output padding fast path requires batch_size=1")
    if packed_output.shape[0] != total_seq_len:
        raise ValueError("Stream-R1 SP packed output length must be unpadded")
    pad = packed_output.new_zeros(
        sp_pad_tokens,
        packed_output.shape[1],
        packed_output.shape[2],
    )
    return torch.cat([packed_output, pad], dim=0)


def stream_r1_segmented_packed_varlen_attention_sp_output_all_to_all(
    query: torch.Tensor,
    segmented_view: "WanS2VStreamR1SegmentedMixedKVView",
    plan: WanS2VStreamR1AttentionPlan,
    *,
    softmax_scale: float | None,
    total_seq_len: int,
    sp_pad_tokens: int,
    force_torch: bool = False,
) -> torch.Tensor:
    """Run SP packed attention and avoid the output prepack copy when possible."""

    with _stream_r1_comm_nvtx_range(
        "stream_r1_segmented_packed_attention.build_workspace "
        f"q={tuple(query.shape)} noisy={tuple(segmented_view.noisy_key.shape)} "
        f"condition={tuple(segmented_view.condition_key.shape)} sp_output=True"
    ):
        workspace = build_wan_s2v_stream_r1_segmented_packed_attention_workspace(
            query,
            segmented_view,
            plan,
        )

    use_packed_output_fast_path = (
        not force_torch
        and query.device.type == "cuda"
        and query.shape[0] == 1
        and workspace.query_matches_input_order
    )
    if use_packed_output_fast_path:
        with _stream_r1_comm_nvtx_range(
            "stream_r1_packed_attention.output_all_to_all_packed "
            f"total_seq_len={total_seq_len} sp_pad_tokens={sp_pad_tokens}"
        ):
            packed_output = _run_stream_r1_packed_varlen_attention_workspace(
                query,
                workspace,
                softmax_scale=softmax_scale,
                force_torch=force_torch,
                return_packed_output=True,
            )
            packed_output = _pad_stream_r1_sp_packed_attention_output(
                packed_output,
                batch_size=query.shape[0],
                total_seq_len=total_seq_len,
                sp_pad_tokens=sp_pad_tokens,
            )
            return _usp_output_all_to_all_packed_bshd(
                packed_output,
                batch_size=query.shape[0],
                seq_len=total_seq_len + sp_pad_tokens,
            )

    output = _run_stream_r1_packed_varlen_attention_workspace(
        query,
        workspace,
        softmax_scale=softmax_scale,
        force_torch=force_torch,
    )
    output = _pad_stream_r1_sp_attention_output(
        output,
        total_seq_len=total_seq_len,
        sp_pad_tokens=sp_pad_tokens,
    )
    return _usp_output_all_to_all(output, head_dim=2)


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
class WanS2VStreamR1NoisyKVCacheUpdatePlan:
    """Host-side plan for mutating one layer's noisy-token KV cache.

    This is the first step toward making Stream-R1 KV cache updates graph
    replay friendly: Python computes ranges once, then eager and future graph
    paths apply the same explicit plan instead of re-deriving state inside the
    transformer forward.
    """

    update: WanS2VStreamR1NoisyKVCacheUpdate
    input_global_end: int
    input_local_end: int
    effective_global_end: int
    previous_local_end: int
    cache_capacity: int
    append_tokens: int
    evict: bool
    num_evicted_tokens: int
    num_rolled_tokens: int
    evicted_start: int
    evicted_end: int
    roll_src_start: int
    roll_src_end: int
    roll_dst_start: int
    roll_dst_end: int
    cache_local_end: int
    local_write_start: int
    local_write_end: int
    kv_start: int
    view_kind: str
    view_local_end_index: int
    local_suffix_len: int
    local_start: int
    sink_copy_len: int
    sink_compress: bool

    @property
    def new_global_end(self) -> int:
        return self.update.current_end

    @property
    def new_local_end_index(self) -> int:
        return self.view_local_end_index


class WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer:
    """Stable-address scalar metadata for one layer's noisy KV update plan."""

    VIEW_KIND_TO_INDEX = {
        "prefix": 0,
        "sink_only": 1,
        "sink_suffix_cat": 2,
        "suffix": 3,
    }
    SCALAR_KEYS = (
        "input_global_end",
        "input_local_end",
        "effective_global_end",
        "previous_local_end",
        "cache_capacity",
        "append_tokens",
        "evict",
        "num_evicted_tokens",
        "num_rolled_tokens",
        "evicted_start",
        "evicted_end",
        "roll_src_start",
        "roll_src_end",
        "roll_dst_start",
        "roll_dst_end",
        "cache_local_end",
        "local_write_start",
        "local_write_end",
        "kv_start",
        "view_kind",
        "view_local_end_index",
        "local_suffix_len",
        "local_start",
        "sink_copy_len",
        "sink_compress",
        "new_global_end",
        "new_local_end_index",
    )
    SCALAR_INDEX = {key: index for index, key in enumerate(SCALAR_KEYS)}

    def __init__(
        self,
        *,
        scalar_values: torch.Tensor,
        rolling_cache_indices: torch.Tensor | None = None,
        rolling_cache_mask: torch.Tensor | None = None,
        rolling_key_indices: torch.Tensor | None = None,
        rolling_key_mask: torch.Tensor | None = None,
        rolling_target_mask: torch.Tensor | None = None,
        sink_evict_indices: torch.Tensor | None = None,
        sink_copy_mask: torch.Tensor | None = None,
        host_plan: WanS2VStreamR1NoisyKVCacheUpdatePlan | None = None,
    ) -> None:
        self.scalar_values = scalar_values
        device = scalar_values.device
        self.rolling_cache_indices = (
            rolling_cache_indices
            if rolling_cache_indices is not None
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        self.rolling_cache_mask = (
            rolling_cache_mask
            if rolling_cache_mask is not None
            else torch.empty((0,), dtype=torch.bool, device=device)
        )
        self.rolling_key_indices = (
            rolling_key_indices
            if rolling_key_indices is not None
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        self.rolling_key_mask = (
            rolling_key_mask
            if rolling_key_mask is not None
            else torch.empty((0,), dtype=torch.bool, device=device)
        )
        self.rolling_target_mask = (
            rolling_target_mask
            if rolling_target_mask is not None
            else torch.empty((0,), dtype=torch.bool, device=device)
        )
        self.sink_evict_indices = (
            sink_evict_indices
            if sink_evict_indices is not None
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        self.sink_copy_mask = (
            sink_copy_mask
            if sink_copy_mask is not None
            else torch.empty((0,), dtype=torch.bool, device=device)
        )
        self._host_plan = host_plan

    @property
    def host_plan(self) -> WanS2VStreamR1NoisyKVCacheUpdatePlan | None:
        return self._host_plan

    @classmethod
    def allocate(
        cls,
        *,
        device: torch.device,
        cache_capacity: int = 0,
        sink_tokens: int = 0,
    ) -> "WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer":
        cache_capacity = int(cache_capacity)
        sink_tokens = int(sink_tokens)
        if cache_capacity < 0:
            raise ValueError("cache_capacity must be non-negative")
        if sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        rolling_tokens = max(0, cache_capacity - sink_tokens)
        return cls(
            scalar_values=torch.zeros(
                (len(cls.SCALAR_KEYS),),
                dtype=torch.long,
                device=device,
            ),
            rolling_cache_indices=torch.zeros(
                (rolling_tokens,),
                dtype=torch.long,
                device=device,
            ),
            rolling_cache_mask=torch.zeros(
                (rolling_tokens,),
                dtype=torch.bool,
                device=device,
            ),
            rolling_key_indices=torch.zeros(
                (rolling_tokens,),
                dtype=torch.long,
                device=device,
            ),
            rolling_key_mask=torch.zeros(
                (rolling_tokens,),
                dtype=torch.bool,
                device=device,
            ),
            rolling_target_mask=torch.zeros(
                (rolling_tokens,),
                dtype=torch.bool,
                device=device,
            ),
            sink_evict_indices=torch.zeros(
                (sink_tokens,),
                dtype=torch.long,
                device=device,
            ),
            sink_copy_mask=torch.zeros(
                (sink_tokens,),
                dtype=torch.bool,
                device=device,
            ),
        )

    @classmethod
    def from_plan(
        cls,
        plan: WanS2VStreamR1NoisyKVCacheUpdatePlan,
        *,
        device: torch.device,
    ) -> "WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer":
        buffer = cls.allocate(
            device=device,
            cache_capacity=plan.cache_capacity,
            sink_tokens=plan.update.sink_tokens,
        )
        buffer.copy_from_plan_(plan)
        return buffer

    @classmethod
    def _scalar_values_from_plan(
        cls, plan: WanS2VStreamR1NoisyKVCacheUpdatePlan
    ) -> tuple[int, ...]:
        try:
            view_kind = cls.VIEW_KIND_TO_INDEX[plan.view_kind]
        except KeyError as exc:
            raise ValueError(f"unknown noisy KV cache view kind {plan.view_kind!r}") from exc
        return (
            int(plan.input_global_end),
            int(plan.input_local_end),
            int(plan.effective_global_end),
            int(plan.previous_local_end),
            int(plan.cache_capacity),
            int(plan.append_tokens),
            int(plan.evict),
            int(plan.num_evicted_tokens),
            int(plan.num_rolled_tokens),
            int(plan.evicted_start),
            int(plan.evicted_end),
            int(plan.roll_src_start),
            int(plan.roll_src_end),
            int(plan.roll_dst_start),
            int(plan.roll_dst_end),
            int(plan.cache_local_end),
            int(plan.local_write_start),
            int(plan.local_write_end),
            int(plan.kv_start),
            int(view_kind),
            int(plan.view_local_end_index),
            int(plan.local_suffix_len),
            int(plan.local_start),
            int(plan.sink_copy_len),
            int(plan.sink_compress),
            int(plan.new_global_end),
            int(plan.new_local_end_index),
        )

    def copy_from_plan_(self, plan: WanS2VStreamR1NoisyKVCacheUpdatePlan) -> None:
        self.scalar_values.copy_(
            self.scalar_values.new_tensor(self._scalar_values_from_plan(plan))
        )
        self._copy_index_metadata_from_plan_(plan)
        self._host_plan = plan

    def copy_from_buffer_(
        self, other: "WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer"
    ) -> None:
        if self.scalar_values.shape != other.scalar_values.shape:
            raise ValueError("KV update scalar buffer shape mismatch")
        if self.rolling_cache_indices.shape != other.rolling_cache_indices.shape:
            raise ValueError("KV update rolling cache index shape mismatch")
        if self.rolling_cache_mask.shape != other.rolling_cache_mask.shape:
            raise ValueError("KV update rolling cache mask shape mismatch")
        if self.rolling_key_indices.shape != other.rolling_key_indices.shape:
            raise ValueError("KV update rolling key index shape mismatch")
        if self.rolling_key_mask.shape != other.rolling_key_mask.shape:
            raise ValueError("KV update rolling key mask shape mismatch")
        if self.rolling_target_mask.shape != other.rolling_target_mask.shape:
            raise ValueError("KV update rolling target mask shape mismatch")
        if self.sink_evict_indices.shape != other.sink_evict_indices.shape:
            raise ValueError("KV update sink index shape mismatch")
        if self.sink_copy_mask.shape != other.sink_copy_mask.shape:
            raise ValueError("KV update sink mask shape mismatch")

        self.scalar_values.copy_(other.scalar_values)
        self.rolling_cache_indices.copy_(other.rolling_cache_indices)
        self.rolling_cache_mask.copy_(other.rolling_cache_mask)
        self.rolling_key_indices.copy_(other.rolling_key_indices)
        self.rolling_key_mask.copy_(other.rolling_key_mask)
        self.rolling_target_mask.copy_(other.rolling_target_mask)
        self.sink_evict_indices.copy_(other.sink_evict_indices)
        self.sink_copy_mask.copy_(other.sink_copy_mask)
        self._host_plan = other.host_plan

    def clear(self) -> None:
        self.scalar_values.zero_()
        self.rolling_cache_indices.zero_()
        self.rolling_cache_mask.zero_()
        self.rolling_key_indices.zero_()
        self.rolling_key_mask.zero_()
        self.rolling_target_mask.zero_()
        self.sink_evict_indices.zero_()
        self.sink_copy_mask.zero_()
        self._host_plan = None

    def _copy_index_metadata_from_plan_(
        self, plan: WanS2VStreamR1NoisyKVCacheUpdatePlan
    ) -> None:
        rolling_tokens = max(0, plan.cache_capacity - plan.update.sink_tokens)
        sink_tokens = plan.update.sink_tokens
        if self.rolling_cache_indices.numel() != rolling_tokens:
            if self.rolling_cache_indices.numel() != 0:
                raise ValueError("rolling KV plan buffer length mismatch")
            return
        if self.sink_evict_indices.numel() != sink_tokens:
            if self.sink_evict_indices.numel() != 0:
                raise ValueError("sink KV plan buffer length mismatch")
            return

        device = self.scalar_values.device
        self.rolling_cache_indices.zero_()
        self.rolling_cache_mask.zero_()
        self.rolling_key_indices.zero_()
        self.rolling_key_mask.zero_()
        self.rolling_target_mask.zero_()

        if rolling_tokens > 0:
            if plan.evict:
                keep_src_start = plan.roll_src_start
                keep_len = plan.num_rolled_tokens
            else:
                keep_src_start = plan.update.sink_tokens
                keep_len = max(0, plan.local_write_start - plan.update.sink_tokens)
            keep_len = max(0, min(int(keep_len), rolling_tokens))
            if keep_len > 0:
                torch.arange(
                    keep_src_start,
                    keep_src_start + keep_len,
                    dtype=torch.long,
                    device=device,
                    out=self.rolling_cache_indices[:keep_len],
                )
                self.rolling_cache_mask[:keep_len].fill_(True)

            key_start = plan.local_write_start - plan.update.sink_tokens
            key_len = min(
                plan.update.noisy_seq_len,
                max(0, rolling_tokens - max(0, key_start)),
            )
            if key_start >= 0 and key_len > 0:
                torch.arange(
                    key_len,
                    dtype=torch.long,
                    device=device,
                    out=self.rolling_key_indices[key_start : key_start + key_len],
                )
                self.rolling_key_mask[key_start : key_start + key_len].fill_(True)
            torch.logical_or(
                self.rolling_cache_mask,
                self.rolling_key_mask,
                out=self.rolling_target_mask,
            )

        self.sink_evict_indices.zero_()
        self.sink_copy_mask.zero_()
        if sink_tokens > 0:
            torch.arange(
                plan.evicted_start,
                plan.evicted_start + sink_tokens,
                dtype=torch.long,
                device=device,
                out=self.sink_evict_indices,
            )
            self.sink_evict_indices.clamp_(0, max(plan.cache_capacity - 1, 0))
            sink_copy_len = max(0, min(plan.sink_copy_len, sink_tokens))
            if sink_copy_len > 0:
                self.sink_copy_mask[:sink_copy_len].fill_(True)

    def scalar_tensor(self, key: str) -> torch.Tensor:
        try:
            index = self.SCALAR_INDEX[key]
        except KeyError as exc:
            raise KeyError(f"unknown noisy KV update plan scalar {key!r}") from exc
        return self.scalar_values[index]

    def scalar_snapshot(self) -> dict[str, int]:
        values = self.scalar_values.detach().cpu().tolist()
        return dict(zip(self.SCALAR_KEYS, values))


def _noisy_kv_update_plan_matches_state(
    plan: WanS2VStreamR1NoisyKVCacheUpdatePlan,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    state: WanS2VStreamR1KVState,
) -> bool:
    return (
        plan.update == update
        and plan.input_global_end == state.global_end_index
        and plan.input_local_end == state.local_end_index
    )


def build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(
    kv_cache: WanS2VKVCacheBlock,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    *,
    state: WanS2VStreamR1KVState | None = None,
) -> WanS2VStreamR1NoisyKVCacheUpdatePlan:
    cache_k = kv_cache["k"]
    if state is None:
        state = get_wan_s2v_stream_r1_kv_cache_state(kv_cache)
    global_end = int(state.global_end_index)
    local_end = int(state.local_end_index)
    input_global_end = global_end
    input_local_end = local_end

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

    append_tokens = update.current_end - global_end
    evict = append_tokens > 0 and update.noisy_seq_len + local_end > cache_k.shape[1]
    num_evicted_tokens = (
        update.noisy_seq_len + local_end - cache_k.shape[1] if evict else 0
    )
    num_rolled_tokens = (
        max(0, local_end - num_evicted_tokens - update.sink_tokens) if evict else 0
    )
    evicted_start = update.sink_tokens
    evicted_end = update.sink_tokens + num_evicted_tokens
    roll_src_start = update.sink_tokens + num_evicted_tokens
    roll_src_end = roll_src_start + num_rolled_tokens
    roll_dst_start = update.sink_tokens
    roll_dst_end = update.sink_tokens + num_rolled_tokens

    if evict:
        cache_local_end = local_end + append_tokens - num_evicted_tokens
    else:
        cache_local_end = local_end + append_tokens

    local_write_start = cache_local_end - update.noisy_seq_len
    local_write_end = cache_local_end
    kv_start = max(
        update.sink_tokens,
        cache_local_end - update.local_tokens + update.sink_tokens,
    )

    if update.sink_tokens > 0:
        if kv_start == update.sink_tokens:
            view_kind = "prefix"
            view_local_end_index = cache_local_end
        elif kv_start >= cache_local_end:
            view_kind = "sink_only"
            view_local_end_index = update.sink_tokens
        else:
            view_kind = "sink_suffix_cat"
            view_local_end_index = update.sink_tokens + cache_local_end - kv_start
    else:
        view_kind = "suffix"
        view_local_end_index = cache_local_end - kv_start

    local_suffix_len = max(0, cache_local_end - kv_start)
    local_start = update.current_end - local_suffix_len
    sink_copy_len = min(num_evicted_tokens, update.sink_tokens) if evict else 0
    sink_compress = (
        evict
        and update.sink_tokens > 0
        and num_evicted_tokens == update.sink_tokens
    )

    return WanS2VStreamR1NoisyKVCacheUpdatePlan(
        update=update,
        input_global_end=input_global_end,
        input_local_end=input_local_end,
        effective_global_end=global_end,
        previous_local_end=local_end,
        cache_capacity=cache_k.shape[1],
        append_tokens=append_tokens,
        evict=evict,
        num_evicted_tokens=num_evicted_tokens,
        num_rolled_tokens=num_rolled_tokens,
        evicted_start=evicted_start,
        evicted_end=evicted_end,
        roll_src_start=roll_src_start,
        roll_src_end=roll_src_end,
        roll_dst_start=roll_dst_start,
        roll_dst_end=roll_dst_end,
        cache_local_end=cache_local_end,
        local_write_start=local_write_start,
        local_write_end=local_write_end,
        kv_start=kv_start,
        view_kind=view_kind,
        view_local_end_index=view_local_end_index,
        local_suffix_len=local_suffix_len,
        local_start=local_start,
        sink_copy_len=sink_copy_len,
        sink_compress=sink_compress,
    )


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
    plan: WanS2VStreamR1NoisyKVCacheUpdatePlan | None = None,
    *,
    commit_host_state: bool = True,
) -> WanS2VStreamR1NoisyKVCacheView:
    """Mutate one layer's noisy-token KV cache using Stream-R1 S2V windows."""

    _validate_noisy_kv_update_inputs(kv_cache, key, value, update)

    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]
    host_state = (
        get_wan_s2v_stream_r1_kv_cache_state(kv_cache) if commit_host_state else None
    )
    if plan is None:
        plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(
            kv_cache,
            update,
            state=host_state,
        )
    elif plan.update != update:
        raise ValueError("KV cache update plan does not match update")

    if plan.evict:
        evicted_k = cache_k[:, plan.evicted_start : plan.evicted_end].clone()
        evicted_v = cache_v[:, plan.evicted_start : plan.evicted_end].clone()

        if plan.num_rolled_tokens > 0:
            cache_k[:, plan.roll_dst_start : plan.roll_dst_end] = cache_k[
                :, plan.roll_src_start : plan.roll_src_end
            ].clone()
            cache_v[:, plan.roll_dst_start : plan.roll_dst_end] = cache_v[
                :, plan.roll_src_start : plan.roll_src_end
            ].clone()

        if update.sink_tokens > 0 and evicted_k.numel() > 0:
            if plan.sink_compress:
                cache_k[:, : update.sink_tokens] = (
                    _STREAM_R1_SINK_COMPRESSION_ALPHA * cache_k[:, : update.sink_tokens]
                    + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_k
                )
                cache_v[:, : update.sink_tokens] = (
                    _STREAM_R1_SINK_COMPRESSION_ALPHA * cache_v[:, : update.sink_tokens]
                    + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_v
                )
            elif plan.sink_copy_len > 0:
                cache_k[:, : plan.sink_copy_len] = evicted_k[:, : plan.sink_copy_len]
                cache_v[:, : plan.sink_copy_len] = evicted_v[:, : plan.sink_copy_len]

    cache_k[:, plan.local_write_start : plan.local_write_end] = key[
        :, : update.noisy_seq_len
    ]
    cache_v[:, plan.local_write_start : plan.local_write_end] = value[
        :, : update.noisy_seq_len
    ]

    if update.sink_tokens > 0:
        if plan.view_kind == "prefix":
            view_key = cache_k[:, : plan.cache_local_end]
            view_value = cache_v[:, : plan.cache_local_end]
        elif plan.view_kind == "sink_only":
            view_key = cache_k[:, : update.sink_tokens]
            view_value = cache_v[:, : update.sink_tokens]
        else:
            with _stream_r1_comm_nvtx_range(
                "stream_r1_noisy_kv_view.sink_cat "
                f"sink_tokens={update.sink_tokens} "
                f"kv_start={plan.kv_start} local_end={plan.cache_local_end}"
            ):
                view_key = torch.cat(
                    [
                        cache_k[:, : update.sink_tokens],
                        cache_k[:, plan.kv_start : plan.cache_local_end],
                    ],
                    dim=1,
                )
                view_value = torch.cat(
                    [
                        cache_v[:, : update.sink_tokens],
                        cache_v[:, plan.kv_start : plan.cache_local_end],
                    ],
                    dim=1,
                )
    else:
        view_key = cache_k[:, plan.kv_start : plan.cache_local_end]
        view_value = cache_v[:, plan.kv_start : plan.cache_local_end]

    local_end_index = plan.view_local_end_index
    kv_cache["global_end_index"].fill_(update.current_end)
    kv_cache["local_end_index"].fill_(local_end_index)
    if commit_host_state:
        host_state.apply_noisy_kv_cache_update_plan(plan)
        sync_wan_s2v_stream_r1_kv_cache_host_shadow(kv_cache, host_state)
    return WanS2VStreamR1NoisyKVCacheView(
        key=view_key,
        value=view_value,
        global_end_index=update.current_end,
        local_end_index=local_end_index,
        local_start=plan.local_start,
        local_end=update.current_end,
    )


def _expand_kv_gather_index(index: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return index.view(1, -1, 1, 1).expand(
        target.shape[0],
        -1,
        target.shape[2],
        target.shape[3],
    )


def _mask4(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask.view(1, -1, 1, 1).expand(
        target.shape[0],
        -1,
        target.shape[2],
        target.shape[3],
    )


def update_wan_s2v_stream_r1_noisy_kv_cache_with_plan_buffer(
    kv_cache: WanS2VKVCacheBlock,
    key: torch.Tensor,
    value: torch.Tensor,
    update: WanS2VStreamR1NoisyKVCacheUpdate,
    plan_buffer: WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer,
) -> WanS2VStreamR1NoisyKVCacheView:
    """Graph-friendly fixed-shape noisy KV update using prepared plan buffers."""

    _validate_noisy_kv_update_inputs(kv_cache, key, value, update)
    plan = plan_buffer.host_plan
    if plan is None:
        raise ValueError("graph KV update requires a prepared host plan")
    if plan.update != update:
        raise ValueError("KV cache update plan does not match update")

    cache_k = kv_cache["k"]
    cache_v = kv_cache["v"]
    cache_capacity = cache_k.shape[1]
    sink_tokens = update.sink_tokens
    rolling_tokens = cache_capacity - sink_tokens

    if plan_buffer.rolling_cache_indices.numel() != rolling_tokens:
        raise ValueError("graph KV update rolling index buffer length mismatch")
    if plan_buffer.sink_evict_indices.numel() != sink_tokens:
        raise ValueError("graph KV update sink index buffer length mismatch")

    rolling_k_old = cache_k[:, sink_tokens:].clone() if rolling_tokens > 0 else None
    rolling_v_old = cache_v[:, sink_tokens:].clone() if rolling_tokens > 0 else None
    if rolling_tokens > 0:
        cache_index = _expand_kv_gather_index(
            plan_buffer.rolling_cache_indices,
            rolling_k_old,
        )
        key_index = _expand_kv_gather_index(
            plan_buffer.rolling_key_indices,
            rolling_k_old,
        )
        rolling_k_from_cache = torch.gather(cache_k, dim=1, index=cache_index)
        rolling_v_from_cache = torch.gather(cache_v, dim=1, index=cache_index)
        noisy_key = key[:, : update.noisy_seq_len]
        noisy_value = value[:, : update.noisy_seq_len]
        rolling_k_from_key = torch.gather(noisy_key, dim=1, index=key_index)
        rolling_v_from_key = torch.gather(noisy_value, dim=1, index=key_index)
        cache_mask = _mask4(plan_buffer.rolling_cache_mask, rolling_k_old)
        key_mask = _mask4(plan_buffer.rolling_key_mask, rolling_k_old)
        target_mask = _mask4(plan_buffer.rolling_target_mask, rolling_k_old)
        rolling_k_new = torch.where(cache_mask, rolling_k_from_cache, rolling_k_old)
        rolling_v_new = torch.where(cache_mask, rolling_v_from_cache, rolling_v_old)
        rolling_k_new = torch.where(key_mask, rolling_k_from_key, rolling_k_new)
        rolling_v_new = torch.where(key_mask, rolling_v_from_key, rolling_v_new)
        rolling_k_new = torch.where(target_mask, rolling_k_new, rolling_k_old)
        rolling_v_new = torch.where(target_mask, rolling_v_new, rolling_v_old)
    else:
        rolling_k_new = None
        rolling_v_new = None

    if sink_tokens > 0:
        old_sink_k = cache_k[:, :sink_tokens].clone()
        old_sink_v = cache_v[:, :sink_tokens].clone()
        sink_index = _expand_kv_gather_index(plan_buffer.sink_evict_indices, old_sink_k)
        evicted_k = torch.gather(cache_k, dim=1, index=sink_index)
        evicted_v = torch.gather(cache_v, dim=1, index=sink_index)
        sink_compress = plan_buffer.scalar_tensor("sink_compress").to(
            dtype=torch.bool
        ).view(1, 1, 1, 1)
        evict = plan_buffer.scalar_tensor("evict").to(dtype=torch.bool).view(
            1, 1, 1, 1
        )
        copy_mask = _mask4(plan_buffer.sink_copy_mask, old_sink_k)
        compressed_k = (
            _STREAM_R1_SINK_COMPRESSION_ALPHA * old_sink_k
            + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_k
        )
        compressed_v = (
            _STREAM_R1_SINK_COMPRESSION_ALPHA * old_sink_v
            + (1 - _STREAM_R1_SINK_COMPRESSION_ALPHA) * evicted_v
        )
        copied_k = torch.where(copy_mask, evicted_k, old_sink_k)
        copied_v = torch.where(copy_mask, evicted_v, old_sink_v)
        sink_k_new = torch.where(sink_compress, compressed_k, copied_k)
        sink_v_new = torch.where(sink_compress, compressed_v, copied_v)
        cache_k[:, :sink_tokens] = torch.where(evict, sink_k_new, old_sink_k)
        cache_v[:, :sink_tokens] = torch.where(evict, sink_v_new, old_sink_v)

    if rolling_tokens > 0:
        cache_k[:, sink_tokens:] = rolling_k_new
        cache_v[:, sink_tokens:] = rolling_v_new

    kv_cache["global_end_index"].copy_(
        plan_buffer.scalar_tensor("new_global_end").view(1)
    )
    kv_cache["local_end_index"].copy_(
        plan_buffer.scalar_tensor("new_local_end_index").view(1)
    )

    if update.sink_tokens > 0:
        if plan.view_kind == "prefix":
            view_key = cache_k[:, : plan.cache_local_end]
            view_value = cache_v[:, : plan.cache_local_end]
        elif plan.view_kind == "sink_only":
            view_key = cache_k[:, : update.sink_tokens]
            view_value = cache_v[:, : update.sink_tokens]
        else:
            view_key = torch.cat(
                [
                    cache_k[:, : update.sink_tokens],
                    cache_k[:, plan.kv_start : plan.cache_local_end],
                ],
                dim=1,
            )
            view_value = torch.cat(
                [
                    cache_v[:, : update.sink_tokens],
                    cache_v[:, plan.kv_start : plan.cache_local_end],
                ],
                dim=1,
            )
    else:
        view_key = cache_k[:, plan.kv_start : plan.cache_local_end]
        view_value = cache_v[:, plan.kv_start : plan.cache_local_end]

    return WanS2VStreamR1NoisyKVCacheView(
        key=view_key,
        value=view_value,
        global_end_index=plan.new_global_end,
        local_end_index=plan.new_local_end_index,
        local_start=plan.local_start,
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


@dataclass(frozen=True)
class _WanS2VCachedAttentionInputs:
    query_for_attention: torch.Tensor
    current_kv: WanS2VStreamR1ProjectedKVSplit
    noisy_view: WanS2VStreamR1NoisyKVCacheView
    update: WanS2VStreamR1NoisyKVCacheUpdate
    attention_backend: str
    selected_backend: str
    use_sp_head_sharded_packed_attention: bool


def _prepare_wan_s2v_stream_r1_cached_attention_inputs(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: WanS2VKVCacheBlock | None,
    layout: WanS2VStreamR1AttentionLayout | None,
    cache_start: int | None,
    sequence_shard_enabled: bool,
    sp_pad_tokens: int,
    profile: _StreamR1Profile,
    graph_kv_update: bool = False,
) -> _WanS2VCachedAttentionInputs:
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
        kv_state = get_wan_s2v_stream_r1_kv_cache_state(kv_cache)
        update_plan_buffer = kv_cache.get("update_plan_buffer")
        update_plan = (
            update_plan_buffer.host_plan
            if update_plan_buffer is not None
            else None
        )
        if update_plan is None or not _noisy_kv_update_plan_matches_state(
            update_plan,
            update,
            kv_state,
        ):
            update_plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(
                kv_cache,
                update,
                state=kv_state,
            )
            if update_plan_buffer is not None:
                update_plan_buffer.copy_from_plan_(update_plan)
        if graph_kv_update:
            if not isinstance(
                update_plan_buffer, WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer
            ):
                raise ValueError("graph KV update requires a plan buffer")
            noisy_view = update_wan_s2v_stream_r1_noisy_kv_cache_with_plan_buffer(
                kv_cache,
                current_kv.noisy_key,
                current_kv.noisy_value,
                update,
                update_plan_buffer,
            )
        else:
            noisy_view = update_wan_s2v_stream_r1_noisy_kv_cache(
                kv_cache,
                current_kv.noisy_key,
                current_kv.noisy_value,
                update,
                plan=update_plan,
                commit_host_state=False,
            )
            kv_state.apply_noisy_kv_cache_update_plan(update_plan)
            sync_wan_s2v_stream_r1_kv_cache_host_shadow(kv_cache, kv_state)
    return _WanS2VCachedAttentionInputs(
        query_for_attention=query_for_attention,
        current_kv=current_kv,
        noisy_view=noisy_view,
        update=update,
        attention_backend=attention_backend,
        selected_backend=selected_backend,
        use_sp_head_sharded_packed_attention=use_sp_head_sharded_packed_attention,
    )


def update_wan_s2v_stream_r1_cached_self_attention_kv_cache(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: WanS2VKVCacheBlock | None,
    layout: WanS2VStreamR1AttentionLayout | None,
    cache_start: int | None,
    sequence_shard_enabled: bool = False,
    sp_pad_tokens: int = 0,
    graph_kv_update: bool = False,
) -> None:
    """Update cached Stream-R1 noisy K/V without computing attention output."""

    profile = _StreamR1Profile(
        tag="stream_r1_cached_self_attention_cache_update",
        device=query.device,
    )
    prepared = _prepare_wan_s2v_stream_r1_cached_attention_inputs(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        layout=layout,
        cache_start=cache_start,
        sequence_shard_enabled=sequence_shard_enabled,
        sp_pad_tokens=sp_pad_tokens,
        profile=profile,
        graph_kv_update=graph_kv_update,
    )
    profile.log(
        attention_backend=prepared.selected_backend,
        query_seq_len=query.shape[1],
        cached_noisy_seq_len=prepared.noisy_view.key.shape[1],
        sequence_shard_enabled=sequence_shard_enabled,
        sp_pad_tokens=sp_pad_tokens,
        cache_update_only=True,
    )


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
    graph_kv_update: bool = False,
) -> torch.Tensor:
    """Run one guarded Stream-R1 S2V cached self-attention step."""

    profile = _StreamR1Profile(
        tag="stream_r1_cached_self_attention",
        device=query.device,
    )
    prepared = _prepare_wan_s2v_stream_r1_cached_attention_inputs(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        layout=layout,
        cache_start=cache_start,
        sequence_shard_enabled=sequence_shard_enabled,
        sp_pad_tokens=sp_pad_tokens,
        profile=profile,
        graph_kv_update=graph_kv_update,
    )
    query_for_attention = prepared.query_for_attention
    current_kv = prepared.current_kv
    noisy_view = prepared.noisy_view
    attention_backend = prepared.attention_backend
    selected_backend = prepared.selected_backend
    use_sp_head_sharded_packed_attention = (
        prepared.use_sp_head_sharded_packed_attention
    )
    update = prepared.update
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
        with profile.span("packed_varlen_attention_sp_output_all_to_all"):
            output = stream_r1_segmented_packed_varlen_attention_sp_output_all_to_all(
                query_for_attention,
                segmented_mixed_view,
                mixed_plan,
                softmax_scale=getattr(attention, "softmax_scale", None),
                total_seq_len=layout.total_seq_len,
                sp_pad_tokens=sp_pad_tokens,
            )
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
