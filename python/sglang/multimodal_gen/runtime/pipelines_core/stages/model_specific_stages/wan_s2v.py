# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V specific pipeline stages."""

import inspect
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VKVCacheBlock,
    WanS2VStreamR1NoisyKVCacheUpdate,
    WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer,
    WanS2VStreamR1KVState,
    WanS2VTimestepMetadataPlan,
    WanS2VTimestepStaticMetadataBuffers,
    build_wan_s2v_stream_r1_noisy_kv_cache_update_plan,
    wan_s2v_stream_r1_uses_head_sharded_sp_kv_cache,
)
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.audio_encoding import (
    AudioEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


@dataclass(frozen=True)
class WanS2VStreamR1AttentionRequest:
    stream_r1_kv_cache: bool
    num_frame_per_block: int
    local_attn_size: int
    sink_size: int
    context_noise: int

    def validate(self, *, latent_frames: int, train_timesteps: int) -> None:
        if self.num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if latent_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "Stream-R1 S2V requires latent frames to be divisible by "
                f"num_frame_per_block, got latent_frames={latent_frames}, "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        if self.local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if self.local_attn_size < self.num_frame_per_block:
            raise ValueError(
                "local_attn_size must be at least num_frame_per_block for "
                "Stream-R1 S2V local/KV attention"
            )
        if self.sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if self.sink_size >= self.local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if self.context_noise < 0:
            raise ValueError("context_noise must be non-negative")
        if self.context_noise > train_timesteps:
            raise ValueError(
                f"context_noise must be in [0, {train_timesteps}], "
                f"got {self.context_noise}"
            )


@dataclass(frozen=True)
class WanS2VStreamR1CacheMetadata:
    batch_size: int
    num_layers: int
    frame_seq_length: int
    local_num_attention_heads: int
    attention_head_dim: int
    local_attn_size: int
    sink_size: int
    dtype: torch.dtype
    device: torch.device

    @property
    def cache_tokens(self) -> int:
        return self.local_attn_size * self.frame_seq_length

    @property
    def sink_tokens(self) -> int:
        return self.sink_size * self.frame_seq_length

    @property
    def bytes_per_kv_cache(self) -> int:
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.num_layers
            * self.batch_size
            * self.cache_tokens
            * self.local_num_attention_heads
            * self.attention_head_dim
            * 2
            * itemsize
        )


@dataclass
class WanS2VStreamR1CacheState:
    metadata: WanS2VStreamR1CacheMetadata | None = None
    kv_cache: list[WanS2VKVCacheBlock] | None = None
    kv_states: list[WanS2VStreamR1KVState] | None = None

    @property
    def enabled(self) -> bool:
        return self.metadata is not None

    @property
    def allocated(self) -> bool:
        return self.kv_cache is not None

    @classmethod
    def disabled(cls) -> "WanS2VStreamR1CacheState":
        return cls()

    @classmethod
    def metadata_only(
        cls, metadata: WanS2VStreamR1CacheMetadata
    ) -> "WanS2VStreamR1CacheState":
        return cls(metadata=metadata)

    @classmethod
    def allocate(
        cls, metadata: WanS2VStreamR1CacheMetadata
    ) -> "WanS2VStreamR1CacheState":
        kv_cache: list[WanS2VKVCacheBlock] = []
        kv_states: list[WanS2VStreamR1KVState] = []
        with torch.inference_mode(False):
            # Stream-R1 updates every layer with the same noisy-token window.
            # Sharing the read-only plan metadata avoids rewriting 40 identical
            # plan buffers before each clean-refresh/timestep graph replay.
            update_plan_buffer = WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer.allocate(
                device=metadata.device,
                cache_capacity=metadata.cache_tokens,
                sink_tokens=metadata.sink_tokens,
            )
            for _ in range(metadata.num_layers):
                state = WanS2VStreamR1KVState()
                kv_states.append(state)
                kv_cache.append(
                    {
                        "k": torch.zeros(
                            (
                                metadata.batch_size,
                                metadata.cache_tokens,
                                metadata.local_num_attention_heads,
                                metadata.attention_head_dim,
                            ),
                            dtype=metadata.dtype,
                            device=metadata.device,
                        ),
                        "v": torch.zeros(
                            (
                                metadata.batch_size,
                                metadata.cache_tokens,
                                metadata.local_num_attention_heads,
                                metadata.attention_head_dim,
                            ),
                            dtype=metadata.dtype,
                            device=metadata.device,
                        ),
                        "global_end_index": torch.zeros(
                            (1,), dtype=torch.long, device=metadata.device
                        ),
                        "local_end_index": torch.zeros(
                            (1,), dtype=torch.long, device=metadata.device
                        ),
                        "global_end_index_host": 0,
                        "local_end_index_host": 0,
                        "state": state,
                        "update_plan_buffer": update_plan_buffer,
                    }
                )
        return cls(metadata=metadata, kv_cache=kv_cache, kv_states=kv_states)

    def reset(self) -> None:
        if self.kv_cache is None:
            return
        if self.kv_states is not None:
            for state in self.kv_states:
                state.reset()
        cleared_plan_buffers: set[int] = set()
        for block_cache in self.kv_cache:
            block_cache["global_end_index"].zero_()
            block_cache["local_end_index"].zero_()
            block_state = block_cache.get("state")
            if isinstance(block_state, WanS2VStreamR1KVState):
                block_state.reset()
                block_cache["global_end_index_host"] = block_state.global_end_index
                block_cache["local_end_index_host"] = block_state.local_end_index
            else:
                block_cache["global_end_index_host"] = 0
                block_cache["local_end_index_host"] = 0
            update_plan_buffer = block_cache.get("update_plan_buffer")
            if isinstance(
                update_plan_buffer, WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer
            ) and id(update_plan_buffer) not in cleared_plan_buffers:
                update_plan_buffer.clear()
                cleared_plan_buffers.add(id(update_plan_buffer))

    def prepare_kv_update_plans(
        self,
        *,
        noisy_seq_len: int,
        current_start: int,
        cache_start: int = 0,
    ) -> dict[str, Any]:
        stats = {
            "kv_plan_builds": 0,
            "kv_plan_shared_copies": 0,
            "kv_plan_shared_reuses": 0,
            "kv_plan_existing_reuses": 0,
            "kv_plan_clears": 0,
            "kv_plan_build_ms": 0.0,
            "kv_plan_copy_from_plan_ms": 0.0,
            "kv_plan_shared_copy_ms": 0.0,
            "kv_plan_clear_ms": 0.0,
        }
        if self.metadata is None or self.kv_cache is None or self.kv_states is None:
            return stats
        update = WanS2VStreamR1NoisyKVCacheUpdate(
            noisy_seq_len=int(noisy_seq_len),
            frame_seq_length=self.metadata.frame_seq_length,
            local_attn_size=self.metadata.local_attn_size,
            sink_size=self.metadata.sink_size,
            current_start=int(current_start),
            cache_start=int(cache_start),
        )
        template_buffer: WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer | None = None
        template_plan = None
        invalid_state_signatures: set[tuple[int, int, int]] = set()
        for block_cache, state in zip(self.kv_cache, self.kv_states):
            update_plan_buffer = block_cache.get("update_plan_buffer")
            if not isinstance(
                update_plan_buffer, WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer
            ):
                continue
            state_signature = (
                int(state.global_end_index),
                int(state.local_end_index),
                int(block_cache["k"].shape[1]),
            )
            if state_signature in invalid_state_signatures:
                stage_started = time.perf_counter()
                update_plan_buffer.clear()
                stats["kv_plan_clear_ms"] += (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stats["kv_plan_clears"] += 1
                continue
            host_plan = update_plan_buffer.host_plan
            if (
                host_plan is not None
                and host_plan.update == update
                and host_plan.input_global_end == state.global_end_index
                and host_plan.input_local_end == state.local_end_index
            ):
                if template_buffer is None:
                    template_buffer = update_plan_buffer
                    template_plan = host_plan
                stats["kv_plan_existing_reuses"] += 1
                continue
            if (
                template_buffer is not None
                and template_plan is not None
                and state.global_end_index == template_plan.input_global_end
                and state.local_end_index == template_plan.input_local_end
                and int(block_cache["k"].shape[1]) == int(template_plan.cache_capacity)
            ):
                if update_plan_buffer is template_buffer:
                    stats["kv_plan_shared_reuses"] += 1
                else:
                    stage_started = time.perf_counter()
                    update_plan_buffer.copy_from_buffer_(template_buffer)
                    stats["kv_plan_shared_copy_ms"] += (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    stats["kv_plan_shared_copies"] += 1
                continue
            if template_buffer is not None and update_plan_buffer is template_buffer:
                raise RuntimeError(
                    "Shared Wan S2V KV update plan buffer requires layer KV "
                    "states to remain aligned"
                )
            try:
                stage_started = time.perf_counter()
                plan = build_wan_s2v_stream_r1_noisy_kv_cache_update_plan(
                    block_cache,
                    update,
                    state=state,
                )
                stats["kv_plan_build_ms"] += (
                    time.perf_counter() - stage_started
                ) * 1000.0
            except ValueError:
                invalid_state_signatures.add(state_signature)
                stage_started = time.perf_counter()
                update_plan_buffer.clear()
                stats["kv_plan_clear_ms"] += (
                    time.perf_counter() - stage_started
                ) * 1000.0
                stats["kv_plan_clears"] += 1
                continue
            stage_started = time.perf_counter()
            update_plan_buffer.copy_from_plan_(plan)
            stats["kv_plan_copy_from_plan_ms"] += (
                time.perf_counter() - stage_started
            ) * 1000.0
            stats["kv_plan_builds"] += 1
            if template_buffer is None:
                template_buffer = update_plan_buffer
                template_plan = plan
        for key in (
            "kv_plan_build_ms",
            "kv_plan_copy_from_plan_ms",
            "kv_plan_shared_copy_ms",
            "kv_plan_clear_ms",
        ):
            stats[key] = round(float(stats[key]), 3)
        return stats


@dataclass(frozen=True)
class WanS2VAdaptiveStepConfig:
    enabled: bool
    threshold: float
    aggressive_threshold: float
    reduced_step_count: int
    aggressive_step_count: int
    warmup_blocks: int
    log_only: bool


@dataclass(frozen=True)
class WanS2VLatentWarmStartConfig:
    enabled: bool
    alpha: float
    mode: str
    warmup_blocks: int
    timestep_index: int
    effective_sigma: float | None
    log: bool


@dataclass(frozen=True)
class WanS2VCleanContextRefreshConfig:
    mode: str
    interval: int
    warmup_blocks: int
    log: bool


@dataclass(frozen=True)
class WanS2VTimestepProfileConfig:
    enabled: bool
    log: bool
    nvtx: bool
    synchronize: bool


@dataclass(frozen=True)
class WanS2VTimestepAblationConfig:
    mode: str
    step_indices: tuple[int, ...]
    timestep_values: tuple[float, ...]
    block_indices: tuple[int, ...]
    warmup_blocks: int
    value_tolerance: float
    scale: float
    log: bool

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and bool(self.step_indices or self.timestep_values)


@dataclass(frozen=True)
class WanS2VTimestepCudaGraphConfig:
    enabled: bool
    step_indices: tuple[int, ...]
    warmup_blocks: int
    max_graphs: int
    log: bool


class _WanS2VTimestepTensorTree:
    """Small tensor-tree helpers used by the timestep graph path."""

    _NON_TENSOR_ADDRESS_SIGNATURE = ("__non_tensor__",)

    @staticmethod
    def tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )

    @classmethod
    def nested_signature(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return cls.tensor_signature(value)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(cls.nested_signature(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                (key, cls.nested_signature(value[key])) for key in sorted(value)
            )
        return (type(value).__name__, value)

    @staticmethod
    def tensor_address_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    @classmethod
    def nested_address_signature(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return cls.tensor_address_signature(value)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return tuple(cls.nested_address_signature(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                (key, signature)
                for key in sorted(value)
                for signature in (cls.nested_address_signature(value[key]),)
                if signature != cls._NON_TENSOR_ADDRESS_SIGNATURE
            )
        return cls._NON_TENSOR_ADDRESS_SIGNATURE

    @classmethod
    def clone_static(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            with torch.inference_mode(False):
                static = torch.empty_strided(
                    tuple(value.shape),
                    tuple(value.stride()),
                    dtype=value.dtype,
                    device=value.device,
                )
                static.copy_(value)
            return static
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(cls.clone_static(item) for item in value)
        if isinstance(value, list):
            return [cls.clone_static(item) for item in value]
        if isinstance(value, dict):
            return {key: cls.clone_static(item) for key, item in value.items()}
        return value

    @classmethod
    def copy_static_(cls, static: Any, live: Any) -> None:
        if isinstance(static, torch.Tensor):
            if not isinstance(live, torch.Tensor):
                raise TypeError("CUDA graph static tensor input received non-tensor")
            if static.shape != live.shape or static.dtype != live.dtype:
                raise ValueError(
                    "CUDA graph static input shape/dtype mismatch: "
                    f"static={tuple(static.shape)}/{static.dtype}, "
                    f"live={tuple(live.shape)}/{live.dtype}"
                )
            static.copy_(live)
            return
        if static is None:
            if live is not None:
                raise ValueError("CUDA graph static None input received a value")
            return
        if isinstance(static, tuple):
            if not isinstance(live, tuple) or len(static) != len(live):
                raise ValueError("CUDA graph tuple input structure mismatch")
            for static_item, live_item in zip(static, live):
                cls.copy_static_(static_item, live_item)
            return
        if isinstance(static, list):
            if not isinstance(live, list) or len(static) != len(live):
                raise ValueError("CUDA graph list input structure mismatch")
            for static_item, live_item in zip(static, live):
                cls.copy_static_(static_item, live_item)
            return
        if isinstance(static, dict):
            if not isinstance(live, dict) or set(static) != set(live):
                raise ValueError("CUDA graph dict input structure mismatch")
            for key in static:
                cls.copy_static_(static[key], live[key])


class _WanS2VTimestepStaticInputBuffers:
    """Static-address transformer inputs for one captured timestep shape."""

    _STATIC_INPUT_KEYS = (
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "ref_latents",
        "motion_latents",
        "cond_states",
        "audio_input",
        "audio_emb",
        "audio_emb_global",
    )

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    @classmethod
    def from_live_kwargs(
        cls, kwargs: dict[str, Any]
    ) -> "_WanS2VTimestepStaticInputBuffers":
        return cls(
            {
                key: _WanS2VTimestepTensorTree.clone_static(kwargs.get(key))
                for key in cls._STATIC_INPUT_KEYS
            }
        )

    @classmethod
    def signature_from_kwargs(cls, kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            (key, _WanS2VTimestepTensorTree.nested_signature(kwargs.get(key)))
            for key in cls._STATIC_INPUT_KEYS
        )

    def copy_from_live_kwargs_(self, kwargs: dict[str, Any]) -> None:
        for key in self._STATIC_INPUT_KEYS:
            _WanS2VTimestepTensorTree.copy_static_(self.values[key], kwargs.get(key))

    def bind_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        static_kwargs = dict(kwargs)
        static_kwargs.update(self.values)
        return static_kwargs


@dataclass
class _WanS2VTimestepCudaGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_inputs: _WanS2VTimestepStaticInputBuffers
    static_metadata: WanS2VTimestepStaticMetadataBuffers
    static_kwargs: dict[str, Any]
    output: Any
    kv_update_plan_signature: tuple[Any, ...] | None = None


class _WanS2VTimestepFullCudaGraphBackend:
    """Full-graph capture/replay for one Wan transformer timestep shape."""

    def capture_one(
        self,
        *,
        forward_fn: Any,
        static_kwargs: dict[str, Any],
    ) -> tuple[torch.cuda.CUDAGraph, Any]:
        graph = torch.cuda.CUDAGraph()
        try:
            # Do not run backend-local warmups here: the Wan timestep forward
            # mutates KV cache state. The public warmup_blocks gate keeps early
            # one-time setup out of capture without replaying this step eagerly.
            with torch.cuda.graph(graph):
                output = forward_fn(**static_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Wan S2V timestep CUDA graph capture failed. Disable "
                "wan_s2v_timestep_cuda_graph to fall back to eager execution."
            ) from exc
        return graph, output

    @staticmethod
    def replay(entry: _WanS2VTimestepCudaGraphEntry) -> Any:
        entry.graph.replay()
        return entry.output


class _WanS2VTransformerTimestepCudaGraphRunner:
    """CUDA graph runner for one Stream-R1 Wan S2V transformer timestep.

    This mirrors the LLM decode full-graph split at a smaller scope: the
    runner owns graph keys, LRU state, and fallbacks; static input buffers own
    stable tensor addresses; the backend only captures and replays the full
    transformer forward.
    """

    def __init__(self) -> None:
        self.max_graphs = 16
        self.graphs: OrderedDict[tuple[Any, ...], _WanS2VTimestepCudaGraphEntry] = (
            OrderedDict()
        )
        self.backend = _WanS2VTimestepFullCudaGraphBackend()

    @property
    def cached_graph_count(self) -> int:
        return len(self.graphs)

    def configure(self, *, max_graphs: int) -> None:
        self.max_graphs = max(1, int(max_graphs))
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)

    def clear(self) -> None:
        self.graphs.clear()

    def _evict_lru_graph(self, *, device: torch.device) -> None:
        if device.type == "cuda":
            # A graph replay is asynchronous. Ensure its executable is no longer
            # in flight before dropping the final references held by the entry.
            torch.cuda.synchronize(device)
        self.graphs.popitem(last=False)

    @staticmethod
    def crossattn_cache_ready(crossattn_cache: list[dict] | None) -> bool:
        if crossattn_cache is None:
            return True
        return all(
            isinstance(item.get("k"), torch.Tensor)
            and isinstance(item.get("v"), torch.Tensor)
            and not bool(item.get("needs_update", False))
            for item in crossattn_cache
        )

    @classmethod
    def _cache_signature(
        cls,
        kv_cache: list[WanS2VKVCacheBlock] | None,
        crossattn_cache: list[dict] | None,
    ) -> tuple[Any, ...]:
        return (
            _WanS2VTimestepTensorTree.nested_address_signature(kv_cache),
            _WanS2VTimestepTensorTree.nested_address_signature(crossattn_cache),
        )

    @staticmethod
    def kv_update_plan_signature(
        kv_cache: list[WanS2VKVCacheBlock] | None,
    ) -> tuple[Any, ...] | None:
        if kv_cache is None:
            return None
        signature = []
        for block_cache in kv_cache:
            update_plan_buffer = block_cache.get("update_plan_buffer")
            if not isinstance(
                update_plan_buffer, WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer
            ):
                return None
            host_plan = update_plan_buffer.host_plan
            if host_plan is None:
                return None
            signature.append(
                (
                    int(host_plan.cache_capacity),
                    int(host_plan.update.noisy_seq_len),
                    int(host_plan.update.sink_tokens),
                    int(host_plan.update.local_tokens),
                    int(host_plan.cache_local_end),
                    int(host_plan.local_write_start),
                    int(host_plan.local_write_end),
                    int(host_plan.kv_start),
                    int(
                        WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer.VIEW_KIND_TO_INDEX[
                            host_plan.view_kind
                        ]
                    ),
                    int(host_plan.view_local_end_index),
                    int(host_plan.local_suffix_len),
                )
            )
        return tuple(signature)

    @staticmethod
    def commit_kv_update_plans(
        kv_cache: list[WanS2VKVCacheBlock] | None,
    ) -> None:
        if kv_cache is None:
            return
        for block_cache in kv_cache:
            update_plan_buffer = block_cache.get("update_plan_buffer")
            if not isinstance(
                update_plan_buffer, WanS2VStreamR1NoisyKVCacheUpdatePlanBuffer
            ):
                raise ValueError("Wan timestep graph KV replay requires plan buffers")
            host_plan = update_plan_buffer.host_plan
            if host_plan is None:
                raise ValueError("Wan timestep graph KV replay requires prepared plans")
            kv_state = block_cache.get("state")
            if not isinstance(kv_state, WanS2VStreamR1KVState):
                raise ValueError("Wan timestep graph KV replay requires KV state owner")
            kv_state.apply_noisy_kv_cache_update_plan(host_plan)
            block_cache["global_end_index_host"] = int(kv_state.global_end_index)
            block_cache["local_end_index_host"] = int(kv_state.local_end_index)

    @classmethod
    def make_key(
        cls,
        *,
        kwargs: dict[str, Any],
        metadata_plan: WanS2VTimestepMetadataPlan,
        metadata_device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            "wan_s2v_transformer_timestep_v5",
            metadata_plan.structure_key_signature,
            bool(kwargs.get("stream_r1_audio_emb_pre_sliced", False)),
            bool(kwargs.get("stream_r1_graph_kv_update", False)),
            bool(kwargs.get("stream_r1_refresh_only", False)),
            WanS2VTimestepStaticMetadataBuffers.signature_from_plan(
                metadata_plan,
                device=metadata_device,
            ),
            _WanS2VTimestepStaticInputBuffers.signature_from_kwargs(kwargs),
            cls._cache_signature(
                kwargs.get("kv_cache"), kwargs.get("crossattn_cache")
            ),
        )

    @staticmethod
    def _static_input_key_diff(existing: Any, current: Any) -> list[str]:
        if not isinstance(existing, tuple) or not isinstance(current, tuple):
            return ["<non_tuple>"]
        changed: list[str] = []
        for existing_item, current_item in zip(existing, current):
            if (
                not isinstance(existing_item, tuple)
                or not isinstance(current_item, tuple)
                or len(existing_item) != 2
                or len(current_item) != 2
            ):
                if existing_item != current_item:
                    changed.append("<unknown>")
                continue
            existing_name, existing_sig = existing_item
            current_name, current_sig = current_item
            name = str(existing_name)
            if existing_name != current_name:
                changed.append(f"{name}->{current_name}")
            elif existing_sig != current_sig:
                changed.append(name)
        if len(existing) != len(current):
            changed.append("<length>")
        return changed

    @classmethod
    def _key_diff_summary(
        cls,
        existing_key: tuple[Any, ...],
        current_key: tuple[Any, ...],
    ) -> dict[str, Any]:
        component_names = (
            "version",
            "metadata_structure",
            "audio_pre_sliced",
            "graph_kv_update",
            "refresh_only",
            "metadata_buffers",
            "static_inputs",
            "cache_addresses",
        )
        changed = []
        for idx, (existing_item, current_item) in enumerate(
            zip(existing_key, current_key)
        ):
            if existing_item == current_item:
                continue
            name = component_names[idx] if idx < len(component_names) else str(idx)
            if name == "static_inputs":
                changed.append(
                    {
                        "component": name,
                        "inputs": cls._static_input_key_diff(
                            existing_item, current_item
                        ),
                    }
                )
            elif name == "cache_addresses":
                cache_changed = []
                if (
                    isinstance(existing_item, tuple)
                    and isinstance(current_item, tuple)
                    and len(existing_item) == 2
                    and len(current_item) == 2
                ):
                    if existing_item[0] != current_item[0]:
                        cache_changed.append("kv_cache")
                    if existing_item[1] != current_item[1]:
                        cache_changed.append("crossattn_cache")
                else:
                    cache_changed.append("<unknown>")
                changed.append({"component": name, "caches": cache_changed})
            else:
                changed.append({"component": name})
        if len(existing_key) != len(current_key):
            changed.append({"component": "<length>"})
        return {"changed": changed}

    def _log_key_miss_debug(self, key: tuple[Any, ...]) -> None:
        if not os.getenv("WAN_S2V_TIMESTEP_CUDA_GRAPH_KEY_DEBUG"):
            return
        summaries = [
            self._key_diff_summary(existing_key, key)
            for existing_key in reversed(self.graphs.keys())
        ]
        logger.info(
            "Wan S2V timestep CUDA graph key miss: cached_graphs=%d diffs=%s",
            len(self.graphs),
            summaries,
        )

    @staticmethod
    def _latent_frames_from_hidden_states(hidden_states: Any) -> int | None:
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.dim() >= 5:
                return int(hidden_states.shape[2])
            if hidden_states.dim() >= 4:
                return int(hidden_states.shape[1])
            return None
        if isinstance(hidden_states, (list, tuple)) and hidden_states:
            first = hidden_states[0]
            if isinstance(first, torch.Tensor) and first.dim() >= 4:
                return int(first.shape[1])
        return None

    @staticmethod
    def _slice_audio_embedding_for_graph(
        audio_embedding: torch.Tensor,
        *,
        audio_slice_start: int,
        audio_slice_end: int,
    ) -> torch.Tensor:
        if audio_embedding.dim() < 2:
            raise ValueError(
                "Wan timestep graph audio embedding must have batch/time dimensions"
            )
        if audio_slice_end > int(audio_embedding.shape[1]):
            raise ValueError(
                "Wan timestep graph audio embeddings do not cover the requested "
                f"slice: end={audio_slice_end}, available={audio_embedding.shape[1]}"
            )
        return audio_embedding[:, audio_slice_start:audio_slice_end].contiguous()

    @classmethod
    def prepare_graph_kwargs(
        cls,
        kwargs: dict[str, Any],
        metadata_plan: WanS2VTimestepMetadataPlan,
    ) -> dict[str, Any]:
        graph_kwargs = dict(kwargs)
        audio_emb = graph_kwargs.get("audio_emb")
        if audio_emb is None:
            return graph_kwargs
        latent_frames = cls._latent_frames_from_hidden_states(
            graph_kwargs.get("hidden_states")
        )
        if latent_frames is None:
            return graph_kwargs
        if len(metadata_plan.motion_frames) < 2:
            return graph_kwargs

        audio_start_frame = (
            0
            if metadata_plan.audio_start_frame is None
            else int(metadata_plan.audio_start_frame)
        )
        audio_slice_start = int(metadata_plan.motion_frames[1]) + audio_start_frame
        audio_slice_end = audio_slice_start + int(latent_frames)

        if isinstance(audio_emb, tuple):
            if len(audio_emb) != 2:
                return graph_kwargs
            audio_emb_global, local_audio_emb = audio_emb
            if not isinstance(audio_emb_global, torch.Tensor) or not isinstance(
                local_audio_emb, torch.Tensor
            ):
                return graph_kwargs
            graph_kwargs["audio_emb"] = (
                cls._slice_audio_embedding_for_graph(
                    audio_emb_global,
                    audio_slice_start=audio_slice_start,
                    audio_slice_end=audio_slice_end,
                ),
                cls._slice_audio_embedding_for_graph(
                    local_audio_emb,
                    audio_slice_start=audio_slice_start,
                    audio_slice_end=audio_slice_end,
                ),
            )
        elif isinstance(audio_emb, torch.Tensor):
            graph_kwargs["audio_emb"] = cls._slice_audio_embedding_for_graph(
                audio_emb,
                audio_slice_start=audio_slice_start,
                audio_slice_end=audio_slice_end,
            )
            audio_emb_global = graph_kwargs.get("audio_emb_global")
            if isinstance(audio_emb_global, torch.Tensor):
                graph_kwargs["audio_emb_global"] = cls._slice_audio_embedding_for_graph(
                    audio_emb_global,
                    audio_slice_start=audio_slice_start,
                    audio_slice_end=audio_slice_end,
                )
        else:
            return graph_kwargs

        graph_kwargs["stream_r1_audio_emb_pre_sliced"] = True
        graph_kwargs["audio_input"] = None
        return graph_kwargs

    @staticmethod
    def _disable_graph_kv_update(kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs.get("stream_r1_graph_kv_update", False):
            return kwargs
        eager_kwargs = dict(kwargs)
        eager_kwargs["stream_r1_graph_kv_update"] = False
        return eager_kwargs

    @staticmethod
    def bind_forward_with_metadata(
        forward_fn: Any,
        metadata_buffers: WanS2VTimestepStaticMetadataBuffers,
    ) -> Any:
        forward_with_plan_buffers = getattr(
            forward_fn, "forward_with_plan_buffers", None
        )
        if not callable(forward_with_plan_buffers):
            return forward_fn

        def _forward_with_plan_buffers(**call_kwargs: Any) -> Any:
            return forward_with_plan_buffers(
                timestep_metadata_buffers=metadata_buffers,
                **call_kwargs,
            )

        return _forward_with_plan_buffers

    def run(
        self,
        *,
        kwargs: dict[str, Any],
        forward_fn: Any,
        step_index: int,
        current_start: int,
        audio_start_frame: int | None,
        sequence_shard_enabled: bool,
        allow_capture: bool = True,
    ) -> tuple[Any, str]:
        metadata_plan = WanS2VTimestepMetadataPlan.from_kwargs(
            kwargs=kwargs,
            step_index=step_index,
            current_start=current_start,
            audio_start_frame=audio_start_frame,
            sequence_shard_enabled=sequence_shard_enabled,
        )
        graph_kwargs = self.prepare_graph_kwargs(kwargs, metadata_plan)
        if graph_kwargs.get("kv_cache") is not None:
            graph_kwargs = {
                **graph_kwargs,
                "stream_r1_graph_kv_update": True,
            }
        metadata_device = WanS2VTimestepStaticMetadataBuffers.device_from_kwargs(
            graph_kwargs
        )
        key = self.make_key(
            kwargs=graph_kwargs,
            metadata_plan=metadata_plan,
            metadata_device=metadata_device,
        )
        entry = self.graphs.get(key)
        kv_plan_signature = self.kv_update_plan_signature(graph_kwargs.get("kv_cache"))
        if entry is not None:
            entry.static_metadata.copy_from_plan_(metadata_plan)
            if graph_kwargs.get("kv_cache") is not None:
                if (
                    kv_plan_signature is None
                    or entry.kv_update_plan_signature != kv_plan_signature
                ):
                    bound_forward = self.bind_forward_with_metadata(
                        forward_fn,
                        entry.static_metadata,
                    )
                    return (
                        bound_forward(**self._disable_graph_kv_update(graph_kwargs)),
                        "eager_kv_plan_mismatch",
                    )
            entry.static_inputs.copy_from_live_kwargs_(graph_kwargs)
            output = self.backend.replay(entry)
            self.commit_kv_update_plans(graph_kwargs.get("kv_cache"))
            self.graphs.move_to_end(key)
            return output, "replay"

        if not allow_capture:
            raise RuntimeError(
                "Wan S2V timestep CUDA graph attempted capture while capture is disabled."
            )

        if len(self.graphs) >= self.max_graphs:
            self._evict_lru_graph(device=metadata_device)

        self._log_key_miss_debug(key)

        static_inputs = _WanS2VTimestepStaticInputBuffers.from_live_kwargs(
            graph_kwargs
        )
        static_metadata = WanS2VTimestepStaticMetadataBuffers.from_plan(
            metadata_plan,
            device=metadata_device,
        )
        static_kwargs = static_inputs.bind_kwargs(graph_kwargs)
        bound_forward = self.bind_forward_with_metadata(
            forward_fn,
            static_metadata,
        )
        graph, output = self.backend.capture_one(
            forward_fn=bound_forward,
            static_kwargs=static_kwargs,
        )
        entry = _WanS2VTimestepCudaGraphEntry(
            graph=graph,
            static_inputs=static_inputs,
            static_metadata=static_metadata,
            static_kwargs=static_kwargs,
            output=output,
            kv_update_plan_signature=kv_plan_signature,
        )
        output = self.backend.replay(entry)
        self.commit_kv_update_plans(graph_kwargs.get("kv_cache"))
        self.graphs[key] = entry
        return output, "capture"


@dataclass(frozen=True)
class WanS2VAdaptiveStepDecision:
    timesteps: torch.Tensor
    base_step_count: int
    step_count: int
    target_step_count: int
    reduced: bool
    enabled: bool
    log_only: bool
    rel_l1: float | None = None
    reason: str = "disabled"


@dataclass(frozen=True)
class WanS2VCleanContextRefreshDecision:
    refresh: bool
    mode: str
    reason: str
    interval: int


@dataclass
class WanS2VConditionBundle:
    """Request-scoped Wan S2V conditions stored in BCTHW latent layout."""

    prompt_embeds: torch.Tensor | list[torch.Tensor]
    ref_latents: torch.Tensor
    motion_latents: torch.Tensor
    cond_states: torch.Tensor
    audio_input: torch.Tensor | None
    audio_emb: Any | None = None
    motion_frames: tuple[int, int] = (73, 19)
    add_last_motion: int = 2
    drop_motion_frames: bool = False
    control_policy: str = "lookahead"
    chunk_start: int = 0
    chunk_frames: int | None = None
    audio_lookahead_frames: int = 2
    audio_metadata: dict[str, Any] = field(default_factory=dict)

    def slice(
        self,
        start: int,
        frames: int,
        policy: str | None = None,
    ) -> "WanS2VConditionBundle":
        if start < 0:
            raise ValueError("start must be non-negative")
        if frames <= 0:
            raise ValueError("frames must be positive")

        return WanS2VConditionBundle(
            prompt_embeds=self.prompt_embeds,
            ref_latents=self.ref_latents,
            motion_latents=self.motion_latents,
            cond_states=_slice_or_pad_bcthw(self.cond_states, start, frames),
            audio_input=self.audio_input,
            audio_emb=self.audio_emb,
            motion_frames=self.motion_frames,
            add_last_motion=self.add_last_motion,
            drop_motion_frames=self.drop_motion_frames,
            control_policy=policy or self.control_policy,
            chunk_start=start,
            chunk_frames=frames,
            audio_lookahead_frames=self.audio_lookahead_frames,
            audio_metadata={
                **self.audio_metadata,
                "chunk_start": start,
                "chunk_frames": frames,
                "control_policy": policy or self.control_policy,
            },
        )


def _slice_or_pad_bcthw(tensor: torch.Tensor, start: int, frames: int) -> torch.Tensor:
    if tensor.dim() != 5:
        raise ValueError(f"Expected BCTHW tensor, got shape {tuple(tensor.shape)}")
    end = start + frames
    sliced = tensor[:, :, start : min(end, tensor.shape[2])]
    if sliced.shape[2] == frames:
        return sliced
    pad_shape = list(sliced.shape)
    pad_shape[2] = frames - sliced.shape[2]
    pad = torch.zeros(
        pad_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([sliced, pad], dim=2)


def _resolve_motion_frames(raw_motion_frames: Any) -> tuple[int, int]:
    if isinstance(raw_motion_frames, (list, tuple)):
        if len(raw_motion_frames) != 2:
            raise ValueError("motion_frames must be an int or a pair of ints")
        motion_frames = int(raw_motion_frames[0])
        latent_motion_frames = int(raw_motion_frames[1])
    else:
        motion_frames = int(raw_motion_frames)
        latent_motion_frames = (motion_frames + 3) // 4
    if motion_frames <= 0 or latent_motion_frames <= 0:
        raise ValueError("motion_frames values must be positive")
    return motion_frames, latent_motion_frames


def _to_bcthw(
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value.dim() != 5:
        raise ValueError(f"{name} must be a BCTHW tensor, got {tuple(value.shape)}")
    return value.to(device=device, dtype=dtype)


def _prepare_s2v_audio_input(
    audio_input: torch.Tensor | None,
    *,
    target_audio_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if audio_input is None:
        return None
    audio_input = audio_input.to(device=device, dtype=dtype)
    if audio_input.shape[-1] < target_audio_frames:
        pad_frames = target_audio_frames - audio_input.shape[-1]
        audio_input = torch.nn.functional.pad(audio_input, (0, pad_frames))
    elif audio_input.shape[-1] > target_audio_frames:
        audio_input = audio_input[..., :target_audio_frames]
    return audio_input


def build_wan_s2v_condition_bundle(
    batch: Req,
    server_args: ServerArgs,
    *,
    latents: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> WanS2VConditionBundle:
    ref_latents = batch.image_latent
    if ref_latents is None:
        raise ValueError("Wan S2V requires a reference image")
    if isinstance(ref_latents, list):
        raise ValueError("Wan S2V requires tensor image latents in BCTHW layout")
    ref_latents = ref_latents[:, :, :1].to(device=device, dtype=dtype)

    audio_input = batch.extra.get("audio_input")
    if audio_input is None:
        raise ValueError("Wan S2V requires audio_path/audio_tensor")
    audio_input = _prepare_s2v_audio_input(
        audio_input,
        target_audio_frames=latents.shape[2] * 4,
        dtype=dtype,
        device=device,
    )

    arch_config = server_args.pipeline_config.dit_config.arch_config
    motion_frames = _resolve_motion_frames(getattr(arch_config, "motion_frames", 73))
    batch_size, _, _, latent_h, latent_w = latents.shape

    motion_latents = batch.extra.get("motion_latents")
    if motion_latents is None:
        motion_latents = torch.zeros(
            batch_size,
            16,
            motion_frames[1],
            latent_h,
            latent_w,
            dtype=dtype,
            device=device,
        )
    else:
        motion_latents = _to_bcthw(
            motion_latents, dtype=dtype, device=device, name="motion_latents"
        )

    cond_states = batch.extra.get("cond_states")
    if cond_states is None:
        cond_states = batch.extra.get("pose_latents")
    if cond_states is None:
        cond_states = torch.zeros_like(latents)
    else:
        cond_states = _to_bcthw(
            cond_states, dtype=dtype, device=device, name="cond_states"
        )
        cond_states = _slice_or_pad_bcthw(cond_states, 0, latents.shape[2])

    prompt_embeds = batch.prompt_embeds
    if isinstance(prompt_embeds, list):
        prompt_embeds = prompt_embeds[0]

    return WanS2VConditionBundle(
        prompt_embeds=prompt_embeds,
        ref_latents=ref_latents,
        motion_latents=motion_latents,
        cond_states=cond_states,
        audio_input=audio_input,
        audio_emb=batch.extra.get("audio_emb"),
        motion_frames=motion_frames,
        add_last_motion=int(batch.extra.get("add_last_motion", 2)),
        drop_motion_frames=bool(batch.extra.get("drop_motion_frames", False)),
        control_policy=_resolve_request_value(
            batch,
            server_args,
            "control_policy",
            "s2v_control_policy",
            "lookahead",
        ),
        chunk_start=0,
        chunk_frames=latents.shape[2],
        audio_lookahead_frames=int(
            _resolve_request_value(
                batch,
                server_args,
                "audio_lookahead_frames",
                "s2v_audio_lookahead_frames",
                2,
            )
        ),
        audio_metadata={
            "audio_path": batch.extra.get("audio_path"),
            "cache_audio_embeddings": _resolve_request_value(
                batch,
                server_args,
                "cache_audio_embeddings",
                "cache_audio_embeddings",
                True,
            ),
        },
    )


def _resolve_request_value(
    batch: Req,
    server_args: ServerArgs,
    request_key: str,
    config_key: str | None = None,
    default: Any = None,
) -> Any:
    if request_key in batch.extra and batch.extra[request_key] is not None:
        return batch.extra[request_key]
    try:
        value = getattr(batch, request_key)
    except AttributeError:
        value = None
    if value is not None:
        return value
    if config_key is None:
        config_key = request_key
    return getattr(server_args.pipeline_config, config_key, default)


def _safe_sp_world_size() -> int:
    try:
        return get_sp_world_size()
    except AssertionError:
        return 1


def _safe_distributed_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _coerce_timestep_list(value: Any, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, str):
        value = [part for part in value.replace(",", " ").split(" ") if part]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list of integer timesteps")
    timesteps = [int(item) for item in value]
    if not timesteps:
        raise ValueError(f"{field_name} must not be empty")
    return timesteps


def _coerce_optional_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, str):
        value = [part for part in value.replace(",", " ").split(" ") if part]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list of integer values")
    return tuple(int(item) for item in value)


def _coerce_optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, str):
        value = [part for part in value.replace(",", " ").split(" ") if part]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list of numeric values")
    return tuple(float(item) for item in value)


def _has_negative_prompt_embeds(batch: Req) -> bool:
    negative_prompt_embeds = getattr(batch, "negative_prompt_embeds", None)
    if negative_prompt_embeds is None:
        return False
    if isinstance(negative_prompt_embeds, (list, tuple)):
        return any(item is not None for item in negative_prompt_embeds)
    return True


def _collect_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.numel() > 0 else []
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for key in sorted(value):
            tensors.extend(_collect_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_collect_tensors(item))
        return tensors
    return []


def _select_wan_s2v_adaptive_timesteps(
    timesteps: torch.Tensor,
    step_count: int,
) -> torch.Tensor:
    base_step_count = int(timesteps.numel())
    if step_count <= 0:
        raise ValueError("adaptive step_count must be positive")
    if step_count >= base_step_count:
        return timesteps
    if step_count == 1:
        return timesteps[:1]

    positions = torch.linspace(
        0,
        base_step_count - 1,
        steps=step_count,
        device=timesteps.device,
    ).round()
    indices = positions.to(dtype=torch.long)
    indices[0] = 0
    indices[-1] = base_step_count - 1
    return timesteps.index_select(0, indices)


class WanS2VAudioEncodingStage(AudioEncodingStage):
    """Encode audio into raw Wav2Vec hidden states consumed by WanModel_S2V."""

    def __init__(self, audio_encoder=None, wav2vec_feature_extractor=None) -> None:
        super().__init__(
            audio_encoder=audio_encoder,
            audio_proj=None,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        audio_path = batch.extra.get("audio_path")
        audio_tensor = batch.extra.get("audio_tensor")
        if audio_path is None and audio_tensor is None:
            return batch

        device = get_local_torch_device()
        self.load_model()
        sample_rate = 16000
        fps = batch.fps if hasattr(batch, "fps") and batch.fps else 16

        if audio_tensor is not None:
            speech_array = audio_tensor
        else:
            speech_array = self._load_audio(audio_path, sample_rate)
        if isinstance(speech_array, np.ndarray):
            speech_array = self._loudness_norm(speech_array, sample_rate)
            audio_duration = len(speech_array) / sample_rate
        else:
            audio_duration = speech_array.shape[-1] / sample_rate

        num_video_frames = max(1, int(audio_duration * fps))
        if self.wav2vec_feature_extractor is not None:
            audio_feature = np.squeeze(
                self.wav2vec_feature_extractor(
                    speech_array, sampling_rate=sample_rate
                ).input_values
            )
            audio_feature = (
                torch.from_numpy(audio_feature).float().to(device).unsqueeze(0)
            )
        else:
            audio_feature = (
                torch.from_numpy(speech_array).float().to(device)
                if isinstance(speech_array, np.ndarray)
                else speech_array.float().to(device)
            )
            if audio_feature.dim() == 1:
                audio_feature = audio_feature.unsqueeze(0)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            audio_features = self.audio_encoder(
                audio_feature, num_video_frames=num_video_frames
            )
        # Wav2Vec wrapper returns [B, T, num_layers, hidden]. S2V expects
        # [B, num_layers, hidden, T].
        batch.extra["audio_input"] = audio_features.permute(0, 2, 3, 1).contiguous()
        batch.extra["audio_features_all"] = audio_features
        self.offload_model()
        return batch


class WanS2VDenoisingStage(PipelineStage):
    """Denoising loop for Wan2.2-S2V.

    This covers the single-clip path: reference image latent, zero pose
    condition, zero initial motion latent, Wav2Vec hidden states, and CFG.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self._deep_gemm_m_list: tuple[int, ...] = ()

    def load_model(self):
        if self.server_args.dit_cpu_offload:
            self.transformer.to(get_local_torch_device())

    def offload_model(self):
        if self.server_args.dit_cpu_offload:
            self.transformer.to("cpu")

    def _configure_deep_gemm_for_latents(
        self,
        latents: torch.Tensor,
        *,
        patch_size: int | Sequence[int],
        sp_size: int,
        latent_frames: int | None = None,
    ) -> None:
        """Restrict DeepGEMM warmup to Wan S2V's fixed local token count."""
        try:
            from sglang.srt.layers.deep_gemm_wrapper import (
                ENABLE_JIT_DEEPGEMM,
                set_deep_gemm_m_list,
            )
        except ImportError:
            return

        if not ENABLE_JIT_DEEPGEMM or latents.dim() != 5:
            return

        if isinstance(patch_size, int):
            patch = (1, patch_size, patch_size)
        else:
            patch = tuple(int(item) for item in patch_size)
            if len(patch) == 2:
                patch = (1, patch[0], patch[1])
        if len(patch) < 3 or patch[1] <= 0 or patch[2] <= 0:
            return

        batch_size, _channels, total_frames, latent_h, latent_w = latents.shape
        frames = int(total_frames if latent_frames is None else latent_frames)
        if frames <= 0:
            return
        frame_seq_length = (int(latent_h) // patch[1]) * (int(latent_w) // patch[2])
        if frame_seq_length <= 0:
            return
        sp = max(int(sp_size), 1)
        local_seq_len = (frames * frame_seq_length + sp - 1) // sp
        m_value = int(batch_size) * local_seq_len
        if m_value <= 0:
            return

        m_list = tuple(sorted(set(self._deep_gemm_m_list + (m_value,))))
        if m_list == self._deep_gemm_m_list:
            return

        gpu_id = latents.device.index if latents.device.index is not None else 0
        set_deep_gemm_m_list(list(m_list), gpu_id=gpu_id)
        self._deep_gemm_m_list = m_list
        self.log_info(
            "DeepGEMM configured for Wan S2V: M=%s "
            "(B=%d, frames=%d, frame_seq=%d, SP=%d)",
            list(m_list),
            int(batch_size),
            frames,
            frame_seq_length,
            sp,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        latents = batch.latents.to(device=device, dtype=dit_dtype)
        self._configure_deep_gemm_for_latents(
            latents,
            patch_size=server_args.pipeline_config.dit_config.arch_config.patch_size,
            sp_size=_safe_sp_world_size(),
        )
        bundle = build_wan_s2v_condition_bundle(
            batch,
            server_args,
            latents=latents,
            dtype=dit_dtype,
            device=device,
        )
        ref_latents = bundle.ref_latents
        audio_input = bundle.audio_input
        motion_latents = bundle.motion_latents
        cond_states = bundle.cond_states
        motion_frames, lat_motion_frames = bundle.motion_frames

        prompt_embeds = batch.prompt_embeds
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        negative_prompt_embeds = None
        if getattr(batch, "negative_prompt_embeds", None) is not None:
            negative_prompt_embeds = batch.negative_prompt_embeds
            if isinstance(negative_prompt_embeds, list):
                negative_prompt_embeds = (
                    negative_prompt_embeds[0] if negative_prompt_embeds else None
                )

        timesteps = batch.timesteps
        if timesteps is None:
            num_steps = batch.num_inference_steps or 40
            self.scheduler.set_timesteps(
                num_steps,
                device=device,
                shift=server_args.pipeline_config.flow_shift or 3.0,
            )
            timesteps = self.scheduler.timesteps

        guidance_scale = batch.guidance_scale or 4.5
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )

        self.load_model()
        try:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=dit_dtype,
                enabled=autocast_enabled,
            ):
                for i, t in enumerate(timesteps):
                    t_i = t.reshape(1).to(device)
                    with set_forward_context(
                        current_timestep=i, attn_metadata=None, forward_batch=batch
                    ):
                        noise_pred_cond = self.transformer(
                            hidden_states=latents,
                            timestep=t_i,
                            encoder_hidden_states=prompt_embeds,
                            ref_latents=ref_latents,
                            motion_latents=motion_latents,
                            cond_states=cond_states,
                            audio_input=audio_input,
                            motion_frames=[motion_frames, lat_motion_frames],
                        )
                        if guidance_scale > 1 and negative_prompt_embeds is not None:
                            noise_pred_uncond = self.transformer(
                                hidden_states=latents,
                                timestep=t_i,
                                encoder_hidden_states=negative_prompt_embeds,
                                ref_latents=ref_latents,
                                motion_latents=motion_latents,
                                cond_states=cond_states,
                                audio_input=0.0 * audio_input,
                                motion_frames=[motion_frames, lat_motion_frames],
                            )
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_cond - noise_pred_uncond
                            )
                        else:
                            noise_pred = noise_pred_cond

                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        return_dict=False,
                        generator=generator,
                    )[0]
        finally:
            self.offload_model()

        batch.latents = latents
        return batch


class WanS2VStreamR1DenoisingStage(WanS2VDenoisingStage):
    """Stream-R1 S2V block-wise denoising.

    The KV path is enabled only when the loaded transformer exposes the S2V
    cached-attention interfaces. Tests and emergency rollbacks may still set
    ``_s2v_kv_attention_kernel_supported`` to a bool override.
    """

    _s2v_kv_attention_kernel_supported: bool | None = None

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.cache_state = WanS2VStreamR1CacheState.disabled()
        self._adaptive_prev_audio_feature: tuple[torch.Tensor, ...] | None = None
        self._adaptive_prev_request_id: str | None = None
        self._adaptive_total_blocks: int = 0
        self._adaptive_reduced_blocks: int = 0
        self._last_adaptive_step_decision: WanS2VAdaptiveStepDecision | None = None
        self._last_timestep_profile_rows: list[dict[str, Any]] = []
        self._timestep_cuda_graph_runner = (
            _WanS2VTransformerTimestepCudaGraphRunner()
        )
        self.crossattn_cache: list[dict[str, Any]] | None = None

    def _prepare_crossattn_cache(self, enabled: bool) -> list[dict[str, Any]] | None:
        if not enabled:
            self.crossattn_cache = None
            return None

        num_blocks = len(self.transformer.blocks)
        if (
            self.crossattn_cache is None
            or len(self.crossattn_cache) != num_blocks
            or any(not isinstance(item, dict) for item in self.crossattn_cache)
        ):
            self.crossattn_cache = [{} for _ in range(num_blocks)]
        return self.crossattn_cache

    @staticmethod
    def _mark_crossattn_cache_needs_update(
        crossattn_cache: list[dict[str, Any]] | None,
    ) -> None:
        if crossattn_cache is None:
            return
        for item in crossattn_cache:
            item["needs_update"] = True

    def _prepare_request_crossattn_cache(
        self,
        enabled: bool,
    ) -> list[dict[str, Any]] | None:
        crossattn_cache = self._prepare_crossattn_cache(enabled)
        self._mark_crossattn_cache_needs_update(crossattn_cache)
        return crossattn_cache

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ) -> torch.Tensor:
        request_steps = _coerce_timestep_list(
            _resolve_request_value(
                batch,
                server_args,
                "denoising_steps",
                "denoising_step_list",
                None,
            ),
            "denoising_steps",
        )
        if request_steps is None:
            request_steps = _coerce_timestep_list(
                getattr(server_args.pipeline_config, "denoising_step_list", None),
                "denoising_step_list",
            )

        flow_shift = getattr(server_args.pipeline_config, "flow_shift", 3.0)
        if "flow_shift" in batch.extra and batch.extra["flow_shift"] is not None:
            flow_shift = batch.extra["flow_shift"]
        if request_steps is None:
            self.scheduler.set_timesteps(
                batch.num_inference_steps or 40,
                device=device,
                shift=flow_shift,
            )
            return self.scheduler.timesteps

        train_steps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        self.scheduler.set_timesteps(train_steps, device=device, shift=flow_shift)
        timesteps = torch.tensor(request_steps, dtype=torch.long)
        if torch.any(timesteps < 0) or torch.any(timesteps > train_steps):
            raise ValueError(
                f"denoising_steps must be in [0, {train_steps}], got {request_steps}"
            )

        warp_denoising_step = bool(
            _resolve_request_value(
                batch,
                server_args,
                "warp_denoising_step",
                "warp_denoising_step",
                True,
            )
        )
        if warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (
                    self.scheduler.timesteps.detach().cpu(),
                    torch.tensor([0], dtype=self.scheduler.timesteps.dtype),
                )
            )
            indices = train_steps - timesteps
            if torch.any(indices < 0) or torch.any(
                indices >= scheduler_timesteps.numel()
            ):
                raise ValueError(
                    "warped denoising_steps produced invalid scheduler indices"
                )
            timesteps = scheduler_timesteps[indices]

        timesteps = timesteps.to(device=device)
        self.log_info("Using Stream-R1 S2V timesteps: %s", timesteps)
        return timesteps

    def _reset_adaptive_step_state(self, request_id: str | None) -> None:
        self._adaptive_prev_audio_feature = None
        self._adaptive_prev_request_id = request_id
        self._adaptive_total_blocks = 0
        self._adaptive_reduced_blocks = 0
        self._last_adaptive_step_decision = None

    def _resolve_adaptive_step_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VAdaptiveStepConfig:
        return WanS2VAdaptiveStepConfig(
            enabled=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps",
                    "wan_s2v_adaptive_steps",
                    False,
                )
            ),
            threshold=float(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_threshold",
                    "wan_s2v_adaptive_steps_threshold",
                    0.08,
                )
            ),
            aggressive_threshold=float(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_aggressive_threshold",
                    "wan_s2v_adaptive_steps_aggressive_threshold",
                    0.0,
                )
            ),
            reduced_step_count=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_reduced_step_count",
                    "wan_s2v_adaptive_steps_reduced_step_count",
                    2,
                )
            ),
            aggressive_step_count=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_aggressive_step_count",
                    "wan_s2v_adaptive_steps_aggressive_step_count",
                    1,
                )
            ),
            warmup_blocks=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_warmup_blocks",
                    "wan_s2v_adaptive_steps_warmup_blocks",
                    1,
                )
            ),
            log_only=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "adaptive_steps_log_only",
                    "wan_s2v_adaptive_steps_log_only",
                    False,
                )
            ),
        )

    def _resolve_timestep_profile_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VTimestepProfileConfig:
        return WanS2VTimestepProfileConfig(
            enabled=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_profile",
                    "wan_s2v_timestep_profile",
                    False,
                )
            ),
            log=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_profile_log",
                    "wan_s2v_timestep_profile_log",
                    False,
                )
            ),
            nvtx=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_profile_nvtx",
                    "wan_s2v_timestep_profile_nvtx",
                    True,
                )
            ),
            synchronize=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_profile_sync",
                    "wan_s2v_timestep_profile_sync",
                    False,
                )
            ),
        )

    def _resolve_timestep_ablation_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VTimestepAblationConfig:
        mode = str(
            _resolve_request_value(
                batch,
                server_args,
                "timestep_ablation_mode",
                "wan_s2v_timestep_ablation_mode",
                "off",
            )
        ).lower()
        valid_modes = {
            "off",
            "skip_update",
            "reuse_previous_pred",
            "zero_pred",
            "scale_pred",
        }
        if mode not in valid_modes:
            raise ValueError(
                "wan_s2v_timestep_ablation_mode must be one of "
                f"{sorted(valid_modes)}, got {mode!r}"
            )
        value_tolerance = float(
            _resolve_request_value(
                batch,
                server_args,
                "timestep_ablation_value_tolerance",
                "wan_s2v_timestep_ablation_value_tolerance",
                1e-3,
            )
        )
        if value_tolerance < 0:
            raise ValueError(
                "wan_s2v_timestep_ablation_value_tolerance must be non-negative"
            )
        warmup_blocks = int(
            _resolve_request_value(
                batch,
                server_args,
                "timestep_ablation_warmup_blocks",
                "wan_s2v_timestep_ablation_warmup_blocks",
                0,
            )
        )
        if warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_timestep_ablation_warmup_blocks must be non-negative"
            )
        return WanS2VTimestepAblationConfig(
            mode=mode,
            step_indices=_coerce_optional_int_tuple(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_ablation_indices",
                    "wan_s2v_timestep_ablation_indices",
                    (),
                ),
                "wan_s2v_timestep_ablation_indices",
            ),
            timestep_values=_coerce_optional_float_tuple(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_ablation_values",
                    "wan_s2v_timestep_ablation_values",
                    (),
                ),
                "wan_s2v_timestep_ablation_values",
            ),
            block_indices=_coerce_optional_int_tuple(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_ablation_blocks",
                    "wan_s2v_timestep_ablation_blocks",
                    (),
                ),
                "wan_s2v_timestep_ablation_blocks",
            ),
            warmup_blocks=warmup_blocks,
            value_tolerance=value_tolerance,
            scale=float(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_ablation_scale",
                    "wan_s2v_timestep_ablation_scale",
                    1.0,
                )
            ),
            log=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_ablation_log",
                    "wan_s2v_timestep_ablation_log",
                    False,
                )
            ),
        )

    def _resolve_timestep_cuda_graph_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VTimestepCudaGraphConfig:
        max_graphs = int(
            _resolve_request_value(
                batch,
                server_args,
                "timestep_cuda_graph_max_graphs",
                "wan_s2v_timestep_cuda_graph_max_graphs",
                16,
            )
        )
        if max_graphs <= 0:
            raise ValueError("wan_s2v_timestep_cuda_graph_max_graphs must be positive")
        warmup_blocks = int(
            _resolve_request_value(
                batch,
                server_args,
                "timestep_cuda_graph_warmup_blocks",
                "wan_s2v_timestep_cuda_graph_warmup_blocks",
                2,
            )
        )
        if warmup_blocks < 0:
            raise ValueError(
                "wan_s2v_timestep_cuda_graph_warmup_blocks must be non-negative"
            )
        return WanS2VTimestepCudaGraphConfig(
            enabled=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_cuda_graph",
                    "wan_s2v_timestep_cuda_graph",
                    False,
                )
            ),
            step_indices=_coerce_optional_int_tuple(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_cuda_graph_indices",
                    "wan_s2v_timestep_cuda_graph_indices",
                    (),
                ),
                "wan_s2v_timestep_cuda_graph_indices",
            ),
            warmup_blocks=warmup_blocks,
            max_graphs=max_graphs,
            log=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "timestep_cuda_graph_log",
                    "wan_s2v_timestep_cuda_graph_log",
                    False,
                )
            ),
        )

    def _can_use_timestep_cuda_graph(
        self,
        config: WanS2VTimestepCudaGraphConfig,
        *,
        block_index: int,
        step_index: int,
        device: torch.device,
        crossattn_cache: list[dict] | None,
    ) -> tuple[bool, str]:
        if not config.enabled:
            return False, "disabled"
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            return False, "non_cuda"
        if block_index < config.warmup_blocks:
            return False, "warmup"
        if config.step_indices and step_index not in config.step_indices:
            return False, "step_filtered"
        if not _WanS2VTransformerTimestepCudaGraphRunner.crossattn_cache_ready(
            crossattn_cache
        ):
            return False, "crossattn_cache_not_ready"
        return True, "enabled"

    def _select_timestep_ablation_action(
        self,
        config: WanS2VTimestepAblationConfig,
        *,
        block_index: int,
        step_index: int,
        timestep_value: float,
    ) -> str:
        if not config.enabled:
            return "full"
        if block_index < config.warmup_blocks:
            return "full"
        if config.block_indices and block_index not in config.block_indices:
            return "full"

        selected = step_index in config.step_indices
        if not selected and config.timestep_values:
            selected = any(
                abs(timestep_value - target) <= config.value_tolerance
                for target in config.timestep_values
            )
        return config.mode if selected else "full"

    @staticmethod
    def _maybe_sync_timestep_profile(
        config: WanS2VTimestepProfileConfig,
        device: torch.device,
    ) -> None:
        if (
            config.synchronize
            and torch.cuda.is_available()
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.synchronize(device)

    def _timestep_profile_timestamp(
        self,
        config: WanS2VTimestepProfileConfig,
        device: torch.device,
    ) -> float:
        self._maybe_sync_timestep_profile(config, device)
        return time.perf_counter()

    def _extract_adaptive_audio_feature(
        self,
        bundle: WanS2VConditionBundle,
    ) -> tuple[torch.Tensor, ...] | None:
        tensors = _collect_tensors(bundle.audio_emb)
        if tensors:
            tensors = [
                self._slice_adaptive_audio_emb_tensor(tensor, bundle)
                for tensor in tensors
            ]
        else:
            tensors = _collect_tensors(bundle.audio_input)
            tensors = [
                self._slice_adaptive_audio_input_tensor(tensor, bundle)
                for tensor in tensors
            ]
        if not tensors:
            return None
        return tuple(t.detach().to(dtype=torch.float32).clone() for t in tensors)

    def _slice_adaptive_audio_emb_tensor(
        self,
        tensor: torch.Tensor,
        bundle: WanS2VConditionBundle,
    ) -> torch.Tensor:
        if tensor.dim() < 2 or bundle.chunk_frames is None:
            return tensor
        start = int(bundle.motion_frames[1]) + int(bundle.chunk_start)
        end = start + int(bundle.chunk_frames)
        if start < 0 or end > tensor.shape[1]:
            return tensor
        return tensor[:, start:end]

    def _slice_adaptive_audio_input_tensor(
        self,
        tensor: torch.Tensor,
        bundle: WanS2VConditionBundle,
    ) -> torch.Tensor:
        if tensor.dim() == 0 or bundle.chunk_frames is None:
            return tensor
        start = int(bundle.chunk_start) * 4
        end = start + int(bundle.chunk_frames) * 4
        if start < 0 or end > tensor.shape[-1]:
            return tensor
        return tensor[..., start:end]

    def _relative_audio_feature_l1(
        self,
        current: tuple[torch.Tensor, ...],
        previous: tuple[torch.Tensor, ...],
    ) -> float | None:
        if len(current) != len(previous):
            return None
        diff_sum = None
        prev_sum = None
        for cur, prev in zip(current, previous):
            if cur.shape != prev.shape:
                return None
            cur = cur.to(device=prev.device)
            diff = (cur - prev).abs().sum(dtype=torch.float32)
            denom = prev.abs().sum(dtype=torch.float32)
            diff_sum = diff if diff_sum is None else diff_sum + diff
            prev_sum = denom if prev_sum is None else prev_sum + denom
        if diff_sum is None or prev_sum is None:
            return None
        if float(prev_sum.item()) <= 0:
            return float("inf")
        return float((diff_sum / prev_sum.clamp_min(1e-6)).item())

    def _broadcast_adaptive_step_count(self, step_count: int) -> int:
        if _safe_sp_world_size() <= 1:
            return step_count
        step_count_tensor = torch.tensor([step_count], dtype=torch.int32)
        torch.distributed.broadcast(
            step_count_tensor,
            src=0,
            group=get_sp_group().cpu_group,
        )
        return int(step_count_tensor.item())

    def _select_adaptive_timesteps(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        block_bundle: WanS2VConditionBundle,
        timesteps: torch.Tensor,
        block_index: int,
    ) -> WanS2VAdaptiveStepDecision:
        base_step_count = int(timesteps.numel())
        config = self._resolve_adaptive_step_config(batch, server_args)
        request_id = getattr(batch, "request_id", None)
        if block_index == 0 or request_id != self._adaptive_prev_request_id:
            self._reset_adaptive_step_state(request_id)

        if not config.enabled or base_step_count <= 1:
            decision = WanS2VAdaptiveStepDecision(
                timesteps=timesteps,
                base_step_count=base_step_count,
                step_count=base_step_count,
                target_step_count=base_step_count,
                reduced=False,
                enabled=config.enabled,
                log_only=config.log_only,
                reason="disabled" if not config.enabled else "single_step",
            )
            self._last_adaptive_step_decision = decision
            return decision

        self._adaptive_total_blocks += 1
        feature = self._extract_adaptive_audio_feature(block_bundle)
        rel_l1 = None
        target_step_count = base_step_count
        reason = "missing_audio_feature"
        if feature is not None:
            if self._adaptive_prev_audio_feature is None:
                reason = "first_block"
            elif block_index < config.warmup_blocks:
                reason = "warmup"
            else:
                rel_l1 = self._relative_audio_feature_l1(
                    feature,
                    self._adaptive_prev_audio_feature,
                )
                if rel_l1 is None:
                    reason = "feature_shape_changed"
                elif (
                    config.aggressive_threshold > 0
                    and rel_l1 < config.aggressive_threshold
                ):
                    target_step_count = min(
                        config.aggressive_step_count,
                        base_step_count,
                    )
                    reason = "aggressive_audio_match"
                elif rel_l1 < config.threshold:
                    target_step_count = min(config.reduced_step_count, base_step_count)
                    reason = "similar_audio"
                else:
                    reason = "audio_changed"
            self._adaptive_prev_audio_feature = feature

        target_step_count = max(1, min(target_step_count, base_step_count))
        effective_step_count = base_step_count if config.log_only else target_step_count
        effective_step_count = self._broadcast_adaptive_step_count(effective_step_count)
        effective_step_count = max(1, min(effective_step_count, base_step_count))
        reduced = effective_step_count < base_step_count
        if reduced:
            self._adaptive_reduced_blocks += 1
        if config.log_only and target_step_count < base_step_count:
            reason = f"{reason}_log_only"
        selected_timesteps = _select_wan_s2v_adaptive_timesteps(
            timesteps,
            effective_step_count,
        )
        decision = WanS2VAdaptiveStepDecision(
            timesteps=selected_timesteps,
            base_step_count=base_step_count,
            step_count=effective_step_count,
            target_step_count=target_step_count,
            reduced=reduced,
            enabled=True,
            log_only=config.log_only,
            rel_l1=rel_l1,
            reason=reason,
        )
        self._last_adaptive_step_decision = decision
        self.log_info(
            "Wan S2V adaptive steps block %d: steps=%d/%d target=%d "
            "rel_l1=%s reason=%s",
            block_index,
            effective_step_count,
            base_step_count,
            target_step_count,
            "n/a" if rel_l1 is None else f"{rel_l1:.6f}",
            reason,
        )
        return decision

    def _resolve_clean_context_refresh_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VCleanContextRefreshConfig:
        mode = str(
            _resolve_request_value(
                batch,
                server_args,
                "clean_context_refresh_mode",
                "wan_s2v_clean_context_refresh_mode",
                "interval",
            )
        ).lower()
        if mode not in {"always", "interval", "never"}:
            mode = "interval"
        return WanS2VCleanContextRefreshConfig(
            mode=mode,
            interval=max(
                1,
                int(
                    _resolve_request_value(
                        batch,
                        server_args,
                        "clean_context_refresh_interval",
                        "wan_s2v_clean_context_refresh_interval",
                        2,
                    )
                ),
            ),
            warmup_blocks=max(
                0,
                int(
                    _resolve_request_value(
                        batch,
                        server_args,
                        "clean_context_refresh_warmup_blocks",
                        "wan_s2v_clean_context_refresh_warmup_blocks",
                        1,
                    )
                ),
            ),
            log=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "clean_context_refresh_log",
                    "wan_s2v_clean_context_refresh_log",
                    False,
                )
            ),
        )

    def select_clean_context_refresh(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        block_index: int,
        config: WanS2VCleanContextRefreshConfig | None = None,
    ) -> WanS2VCleanContextRefreshDecision:
        if config is None:
            config = self._resolve_clean_context_refresh_config(batch, server_args)

        refresh = True
        reason = "always"
        if config.mode == "never":
            refresh = False
            reason = "never"
        elif block_index < config.warmup_blocks:
            refresh = True
            reason = "warmup"
        elif config.mode == "always":
            refresh = True
            reason = "always"
        elif config.mode == "interval":
            refresh = (block_index % config.interval) == 0
            reason = "interval" if refresh else "interval_skip"

        decision_reason = reason
        if not refresh and not decision_reason.endswith("_skip"):
            decision_reason = f"{decision_reason}_skip"
        decision = WanS2VCleanContextRefreshDecision(
            refresh=refresh,
            mode=config.mode,
            reason=decision_reason,
            interval=config.interval,
        )
        if config.log:
            self.log_info(
                "Wan S2V clean refresh block %d: refresh=%s mode=%s reason=%s "
                "interval=%d",
                block_index,
                decision.refresh,
                decision.mode,
                decision.reason,
                decision.interval,
            )
        return decision

    def _resolve_latent_warm_start_config(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> WanS2VLatentWarmStartConfig:
        alpha = float(
            _resolve_request_value(
                batch,
                server_args,
                "latent_warm_start_alpha",
                "wan_s2v_latent_warm_start_alpha",
                0.25,
            )
        )
        mode = str(
            _resolve_request_value(
                batch,
                server_args,
                "latent_warm_start_mode",
                "wan_s2v_latent_warm_start_mode",
                "repeat_tail",
            )
        ).lower()
        if mode not in {"repeat_tail", "linear"}:
            mode = "repeat_tail"
        return WanS2VLatentWarmStartConfig(
            enabled=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "latent_warm_start",
                    "wan_s2v_latent_warm_start",
                    False,
                )
            ),
            alpha=max(0.0, min(1.0, alpha)),
            mode=mode,
            warmup_blocks=max(
                0,
                int(
                    _resolve_request_value(
                        batch,
                        server_args,
                        "latent_warm_start_warmup_blocks",
                        "wan_s2v_latent_warm_start_warmup_blocks",
                        1,
                    )
                ),
            ),
            timestep_index=max(
                0,
                int(
                    _resolve_request_value(
                        batch,
                        server_args,
                        "latent_warm_start_timestep_index",
                        "wan_s2v_latent_warm_start_timestep_index",
                        1,
                    )
                ),
            ),
            effective_sigma=self._resolve_latent_warm_start_effective_sigma(
                batch,
                server_args,
            ),
            log=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "latent_warm_start_log",
                    "wan_s2v_latent_warm_start_log",
                    False,
                )
            ),
        )

    def _resolve_latent_warm_start_effective_sigma(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> float | None:
        value = _resolve_request_value(
            batch,
            server_args,
            "latent_warm_start_effective_sigma",
            "wan_s2v_latent_warm_start_effective_sigma",
            None,
        )
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    def _predict_latent_warm_start(
        self,
        previous_clean_latents: torch.Tensor,
        target_latents: torch.Tensor,
        config: WanS2VLatentWarmStartConfig,
    ) -> torch.Tensor | None:
        if previous_clean_latents.dim() != 5 or target_latents.dim() != 5:
            return None
        if previous_clean_latents.shape[:2] != target_latents.shape[:2]:
            return None
        if previous_clean_latents.shape[3:] != target_latents.shape[3:]:
            return None
        if previous_clean_latents.shape[2] <= 0 or target_latents.shape[2] <= 0:
            return None

        prev = previous_clean_latents.to(
            device=target_latents.device,
            dtype=target_latents.dtype,
            non_blocking=True,
        )
        target_frames = int(target_latents.shape[2])
        last = prev[:, :, -1:, :, :]
        if config.mode == "linear" and prev.shape[2] >= 2:
            velocity = last - prev[:, :, -2:-1, :, :]
            frames = [last + velocity * float(i + 1) for i in range(target_frames)]
            return torch.cat(frames, dim=2).contiguous()
        return last.expand(-1, -1, target_frames, -1, -1).contiguous()

    def _latent_warm_start_sigma_for_timestep(
        self,
        timestep: torch.Tensor,
    ) -> float | None:
        scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
        scheduler_sigmas = getattr(self.scheduler, "sigmas", None)
        if scheduler_timesteps is None or scheduler_sigmas is None:
            return None
        target_timestep = timestep.detach().flatten()[0]
        timesteps = scheduler_timesteps.detach().to(device=target_timestep.device)
        sigmas = scheduler_sigmas.detach().to(device=target_timestep.device)
        index = torch.argmin((timesteps - target_timestep).abs())
        return float(sigmas[index].item())

    def apply_stream_r1_latent_warm_start(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        block_latents: torch.Tensor,
        previous_clean_latents: torch.Tensor | None,
        timesteps: torch.Tensor,
        block_index: int,
        config: WanS2VLatentWarmStartConfig | None = None,
    ) -> tuple[torch.Tensor, bool]:
        if config is None:
            config = self._resolve_latent_warm_start_config(batch, server_args)
        if (
            not config.enabled
            or config.alpha <= 0.0
            or block_index < config.warmup_blocks
            or previous_clean_latents is None
            or timesteps.numel() == 0
        ):
            return block_latents, False

        warm_clean = self._predict_latent_warm_start(
            previous_clean_latents,
            block_latents,
            config,
        )
        if warm_clean is None:
            return block_latents, False

        base_noise_btchw = block_latents.permute(0, 2, 1, 3, 4).contiguous()
        warm_clean_btchw = warm_clean.permute(0, 2, 1, 3, 4).contiguous()
        selected_timestep = None
        sigma_value = config.effective_sigma
        if sigma_value is None:
            timestep_index = min(config.timestep_index, int(timesteps.numel()) - 1)
            selected_timestep = timesteps[timestep_index : timestep_index + 1].to(
                device=block_latents.device
            )
            if config.log:
                sigma_value = self._latent_warm_start_sigma_for_timestep(
                    selected_timestep
                )
            noised_warm_btchw = self.scheduler.add_noise(
                warm_clean_btchw.flatten(0, 1),
                base_noise_btchw.flatten(0, 1),
                selected_timestep,
            ).unflatten(0, warm_clean_btchw.shape[:2])
        else:
            sigma = torch.tensor(
                sigma_value,
                dtype=base_noise_btchw.dtype,
                device=base_noise_btchw.device,
            )
            noised_warm_btchw = (1.0 - sigma) * warm_clean_btchw + (
                sigma * base_noise_btchw
            )
        if config.alpha < 1.0:
            noised_warm_btchw = base_noise_btchw + config.alpha * (
                noised_warm_btchw - base_noise_btchw
            )
        warm_latents = noised_warm_btchw.permute(0, 2, 1, 3, 4).contiguous()
        if config.log:
            clean_weight = None
            if sigma_value is not None:
                clean_weight = config.alpha * (1.0 - sigma_value)
            self.log_info(
                "Wan S2V latent warm start block %d: mode=%s alpha=%.3f "
                "warmup_blocks=%d timestep_index=%d timestep=%s sigma=%s "
                "clean_weight=%s",
                block_index,
                config.mode,
                config.alpha,
                config.warmup_blocks,
                min(config.timestep_index, int(timesteps.numel()) - 1),
                (
                    "explicit"
                    if selected_timestep is None
                    else f"{float(selected_timestep.flatten()[0].item()):.6f}"
                ),
                "n/a" if sigma_value is None else f"{sigma_value:.6f}",
                "n/a" if clean_weight is None else f"{clean_weight:.6f}",
            )
        return warm_latents.to(dtype=block_latents.dtype), True

    def _train_timesteps(self) -> int:
        scheduler_config = getattr(self.scheduler, "config", None)
        return int(getattr(scheduler_config, "num_train_timesteps", 1000))

    def _resolve_attention_request(
        self,
        batch: Req,
        server_args: ServerArgs,
        latent_frames: int,
    ) -> WanS2VStreamR1AttentionRequest:
        request = WanS2VStreamR1AttentionRequest(
            stream_r1_kv_cache=bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "stream_r1_kv_cache",
                    "stream_r1_kv_cache",
                    False,
                )
            ),
            num_frame_per_block=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "num_frame_per_block",
                    "num_frame_per_block",
                    7,
                )
            ),
            local_attn_size=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "local_attn_size",
                    "local_attn_size",
                    9,
                )
            ),
            sink_size=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "sink_size",
                    "sink_size",
                    3,
                )
            ),
            context_noise=int(
                _resolve_request_value(
                    batch,
                    server_args,
                    "context_noise",
                    "context_noise",
                    0,
                )
            ),
        )
        request.validate(
            latent_frames=latent_frames, train_timesteps=self._train_timesteps()
        )
        self._validate_stream_r1_parallel_compatibility(request, batch)
        return request

    def _validate_stream_r1_parallel_compatibility(
        self, request: WanS2VStreamR1AttentionRequest, batch: Req
    ) -> None:
        return

    def _configure_transformer_attention(
        self, request: WanS2VStreamR1AttentionRequest
    ) -> None:
        setter = getattr(self.transformer, "set_stream_r1_attention", None)
        if callable(setter):
            setter(
                request.local_attn_size,
                request.sink_size,
                num_frame_per_block=request.num_frame_per_block,
                kv_cache=request.stream_r1_kv_cache,
            )
            return
        setattr(self.transformer, "stream_r1_local_attn_size", request.local_attn_size)
        setattr(self.transformer, "stream_r1_sink_size", request.sink_size)
        setattr(
            self.transformer,
            "stream_r1_num_frame_per_block",
            request.num_frame_per_block,
        )
        setattr(
            self.transformer,
            "stream_r1_kv_cache_requested",
            request.stream_r1_kv_cache,
        )

    @staticmethod
    def _callable_accepts_parameters(
        fn: Any,
        required_parameters: set[str],
    ) -> bool:
        if not callable(fn):
            return False
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return required_parameters.issubset(signature.parameters)

    def _supports_s2v_kv_attention_kernel(self) -> bool:
        override = self._s2v_kv_attention_kernel_supported
        if override is not None:
            return bool(override)

        if not callable(getattr(self.transformer, "set_stream_r1_attention", None)):
            return False
        if not self._callable_accepts_parameters(
            getattr(self.transformer, "forward", None),
            {"kv_cache", "current_start", "cache_start", "stream_r1_mode"},
        ):
            return False

        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return False
        return all(
            self._callable_accepts_parameters(
                getattr(block, "forward", None),
                {
                    "stream_r1_kv_cache",
                    "stream_r1_attention_layout",
                    "cache_start",
                    "stream_r1_sequence_shard_enabled",
                    "stream_r1_sp_pad_tokens",
                },
            )
            for block in blocks
        )

    def _build_cache_metadata(
        self,
        *,
        request: WanS2VStreamR1AttentionRequest,
        batch_size: int,
        frame_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> WanS2VStreamR1CacheMetadata:
        blocks = getattr(self.transformer, "blocks", None)
        first_block = blocks[0] if blocks is not None and len(blocks) > 0 else None
        arch_config = getattr(
            getattr(self.transformer, "config", None), "arch_config", None
        )
        num_layers = int(
            getattr(
                arch_config,
                "num_layers",
                len(blocks) if blocks is not None else 0,
            )
        )
        if num_layers <= 0:
            raise ValueError("Wan S2V Stream-R1 cache metadata requires num_layers")
        global_heads = int(getattr(self.transformer, "num_attention_heads", 1))
        hidden_size = int(getattr(self.transformer, "hidden_size", global_heads))
        local_num_heads = int(
            getattr(
                first_block,
                "local_num_heads",
                getattr(self.transformer, "local_num_heads", global_heads),
            )
        )
        if wan_s2v_stream_r1_uses_head_sharded_sp_kv_cache():
            sp_world_size = get_sp_world_size()
            if local_num_heads % sp_world_size != 0:
                raise ValueError(
                    "Stream-R1 S2V packed SP KV cache requires local attention "
                    f"heads ({local_num_heads}) to be divisible by SP world size "
                    f"({sp_world_size})"
                )
            local_num_heads = local_num_heads // sp_world_size
        attention_head_dim = int(
            getattr(
                first_block,
                "dim_head",
                getattr(
                    self.transformer,
                    "attention_head_dim",
                    hidden_size // global_heads,
                ),
            )
        )
        return WanS2VStreamR1CacheMetadata(
            batch_size=batch_size,
            num_layers=num_layers,
            frame_seq_length=frame_seq_length,
            local_num_attention_heads=local_num_heads,
            attention_head_dim=attention_head_dim,
            local_attn_size=request.local_attn_size,
            sink_size=request.sink_size,
            dtype=dtype,
            device=torch.device(device),
        )

    def _prepare_cache_state(
        self,
        *,
        request: WanS2VStreamR1AttentionRequest,
        batch_size: int,
        frame_seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> WanS2VStreamR1CacheState:
        if not request.stream_r1_kv_cache:
            self.cache_state = WanS2VStreamR1CacheState.disabled()
            return self.cache_state

        metadata = self._build_cache_metadata(
            request=request,
            batch_size=batch_size,
            frame_seq_length=frame_seq_length,
            dtype=dtype,
            device=device,
        )
        if not self._supports_s2v_kv_attention_kernel():
            self.cache_state = WanS2VStreamR1CacheState.metadata_only(metadata)
        elif not self.cache_state.allocated or self.cache_state.metadata != metadata:
            self.cache_state = WanS2VStreamR1CacheState.allocate(metadata)
        else:
            self.cache_state.reset()

        self.log_info(
            "Prepared Stream-R1 S2V KV cache metadata: layers=%s, tokens=%s, "
            "local_heads=%s, head_dim=%s, sink_tokens=%s, estimated_kv_cache=%.2f MiB",
            metadata.num_layers,
            metadata.cache_tokens,
            metadata.local_num_attention_heads,
            metadata.attention_head_dim,
            metadata.sink_tokens,
            metadata.bytes_per_kv_cache / (1024**2),
        )
        return self.cache_state

    def _guard_cache_runtime(self, cache_state: WanS2VStreamR1CacheState) -> None:
        if not cache_state.enabled:
            return
        if not self._supports_s2v_kv_attention_kernel():
            assert cache_state.metadata is not None
            metadata = cache_state.metadata
            raise NotImplementedError(
                "Stream-R1 S2V KV cache was requested and validated, but the "
                "loaded transformer does not expose the required S2V KV "
                "attention runtime. "
                "Set stream_r1_kv_cache=false to run the current block-wise "
                "Stream-R1 path without KV cache. "
                f"local_attn_size={metadata.local_attn_size}, "
                f"sink_size={metadata.sink_size}, "
                f"cache_tokens={metadata.cache_tokens}."
            )
        if not cache_state.allocated:
            raise RuntimeError("Stream-R1 S2V KV cache state was not allocated")

    @staticmethod
    def _prepend_motion_audio_frames(
        audio_input: torch.Tensor,
        motion_frames: list[int] | tuple[int, int],
    ) -> torch.Tensor:
        return torch.cat(
            [
                audio_input[..., 0:1].repeat(1, 1, 1, int(motion_frames[0])),
                audio_input,
            ],
            dim=-1,
        )

    def _maybe_cache_audio_embeddings(
        self,
        bundle: WanS2VConditionBundle,
        *,
        dtype: torch.dtype | None = None,
        autocast_enabled: bool = False,
    ) -> None:
        if not bool(bundle.audio_metadata.get("cache_audio_embeddings", False)):
            return
        if bundle.audio_emb is not None or bundle.audio_input is None:
            return

        audio_input = self._prepend_motion_audio_frames(
            bundle.audio_input, bundle.motion_frames
        )
        if dtype is None:
            bundle.audio_emb = self.transformer.casual_audio_encoder(audio_input)
            return

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            bundle.audio_emb = self.transformer.casual_audio_encoder(audio_input)

    def _clean_context_refresh(
        self,
        *,
        block_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        block_bundle: WanS2VConditionBundle,
        current_start: int,
        attention_request: WanS2VStreamR1AttentionRequest,
        cache_state: WanS2VStreamR1CacheState,
        dtype: torch.dtype | None = None,
        autocast_enabled: bool = False,
        forward_batch: Req | None = None,
        server_args: ServerArgs | None = None,
        block_index: int = 0,
        crossattn_cache: list | None = None,
        audio_start_frame: int | None = None,
        allow_timestep_cuda_graph_capture: bool = True,
    ) -> dict[str, Any]:
        timings: dict[str, Any] = {}
        if not cache_state.enabled:
            timings["clean_refresh_enabled"] = False
            return timings

        prepare_started = time.perf_counter()
        self._guard_cache_runtime(cache_state)
        timestep = torch.full(
            (block_latents.shape[0],),
            int(attention_request.context_noise),
            dtype=torch.long,
            device=block_latents.device,
        )
        timings["clean_refresh_timestep_ms"] = round(
            (time.perf_counter() - prepare_started) * 1000.0, 3
        )

        plan_started = time.perf_counter()
        plan_stats: dict[str, Any] = {}
        if cache_state.metadata is not None:
            plan_stats = cache_state.prepare_kv_update_plans(
                noisy_seq_len=(
                    int(block_latents.shape[2])
                    * cache_state.metadata.frame_seq_length
                ),
                current_start=current_start,
                cache_start=0,
            )
        timings["clean_refresh_plan_ms"] = round(
            (time.perf_counter() - plan_started) * 1000.0, 3
        )
        for key, value in plan_stats.items():
            timings[f"clean_refresh_{key}"] = value
        timings["clean_refresh_prepare_ms"] = round(
            (time.perf_counter() - prepare_started) * 1000.0, 3
        )

        transformer_kwargs = {
            "hidden_states": block_latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "ref_latents": block_bundle.ref_latents,
            "motion_latents": block_bundle.motion_latents,
            "cond_states": block_bundle.cond_states,
            "audio_input": block_bundle.audio_input,
            "audio_emb": block_bundle.audio_emb,
            "motion_frames": block_bundle.motion_frames,
            "add_last_motion": block_bundle.add_last_motion,
            "drop_motion_frames": block_bundle.drop_motion_frames,
            "kv_cache": cache_state.kv_cache,
            "crossattn_cache": crossattn_cache,
            "current_start": current_start,
            "cache_start": None,
            "audio_start_frame": audio_start_frame,
            "stream_r1_mode": True,
            "stream_r1_refresh_only": True,
        }
        cuda_graph_status: str | None = None
        cuda_graph_config = None
        if server_args is not None and forward_batch is not None:
            cuda_graph_config = self._resolve_timestep_cuda_graph_config(
                forward_batch,
                server_args,
            )
            if cuda_graph_config.enabled:
                self._timestep_cuda_graph_runner.configure(
                    max_graphs=cuda_graph_config.max_graphs
                )

        def _run_refresh_forward() -> None:
            nonlocal cuda_graph_status
            use_cuda_graph = False
            if cuda_graph_config is not None:
                use_cuda_graph, cuda_graph_status = self._can_use_timestep_cuda_graph(
                    cuda_graph_config,
                    block_index=block_index,
                    step_index=0,
                    device=block_latents.device,
                    crossattn_cache=crossattn_cache,
                )
            if use_cuda_graph:
                context_batch = getattr(get_forward_context(), "forward_batch", None)
                sequence_shard_enabled = bool(
                    context_batch is not None
                    and getattr(context_batch, "enable_sequence_shard", False)
                    and _safe_sp_world_size() > 1
                )
                try:
                    _, cuda_graph_status = self._timestep_cuda_graph_runner.run(
                        kwargs=transformer_kwargs,
                        forward_fn=self.transformer,
                        step_index=0,
                        current_start=current_start,
                        audio_start_frame=audio_start_frame,
                        sequence_shard_enabled=sequence_shard_enabled,
                        allow_capture=allow_timestep_cuda_graph_capture,
                    )
                except RuntimeError as exc:
                    capture_disabled = "capture is disabled" in str(exc)
                    if allow_timestep_cuda_graph_capture or not capture_disabled:
                        raise
                    cuda_graph_status = "eager_capture_disabled"
                    self.transformer(**transformer_kwargs)
            else:
                self.transformer(**transformer_kwargs)

        forward_started = time.perf_counter()
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            if dtype is None:
                _run_refresh_forward()
            else:
                with torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=dtype,
                    enabled=autocast_enabled,
                ):
                    _run_refresh_forward()
        timings["clean_refresh_forward_body_ms"] = round(
            (time.perf_counter() - forward_started) * 1000.0, 3
        )
        timings["clean_refresh_cuda_graph"] = cuda_graph_status or "not_run"
        timings["clean_refresh_enabled"] = True
        return timings

    def denoise_stream_r1_block(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        block_latents: torch.Tensor,
        block_bundle: WanS2VConditionBundle,
        block_start: int,
        frame_seq_length: int,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache_state: WanS2VStreamR1CacheState,
        crossattn_cache: list[dict] | None,
        generator: torch.Generator | None,
        dit_dtype: torch.dtype,
        autocast_enabled: bool,
        audio_start_frame: int | None = None,
        step_noises_btchw: Sequence[torch.Tensor] | None = None,
        block_index: int | None = None,
        allow_timestep_cuda_graph_capture: bool = True,
        only_step_index: int | None = None,
    ) -> torch.Tensor:
        """Denoise one Stream-R1 S2V latent block.

        The full-clip path calls this once per block. Realtime session
        generation also calls this helper so it can preserve the same
        timestep and attention behavior without rerunning the whole pipeline.
        """
        current_latents = block_latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        video_raw_latent_shape = noise_latents_btchw.shape
        current_start = block_start * frame_seq_length
        if block_index is None:
            chunk_frames = block_bundle.chunk_frames or max(
                int(current_latents.shape[2]), 1
            )
            block_index = int(block_start) // int(chunk_frames)
        profile_config = self._resolve_timestep_profile_config(batch, server_args)
        ablation_config = self._resolve_timestep_ablation_config(batch, server_args)
        cuda_graph_config = self._resolve_timestep_cuda_graph_config(
            batch, server_args
        )
        if cuda_graph_config.enabled:
            self._timestep_cuda_graph_runner.configure(
                max_graphs=cuda_graph_config.max_graphs
            )
        else:
            self._timestep_cuda_graph_runner.clear()
        use_nvtx = bool(
            profile_config.nvtx and (profile_config.enabled or ablation_config.enabled)
        )
        timestep_values = [
            float(item) for item in timesteps.detach().cpu().reshape(-1).tolist()
        ]
        timestep_profile_rows: list[dict[str, Any]] = []
        timestep_cuda_graph_statuses: list[dict[str, Any]] = []
        self._last_timestep_profile_rows = timestep_profile_rows
        self._last_timestep_cuda_graph_statuses = timestep_cuda_graph_statuses
        if step_noises_btchw is not None and len(step_noises_btchw) != max(
            int(timesteps.numel()) - 1, 0
        ):
            raise ValueError(
                "Stream-R1 S2V precomputed step noise count does not match "
                f"timesteps: got {len(step_noises_btchw)}, "
                f"expected {max(int(timesteps.numel()) - 1, 0)}"
            )

        previous_noise_pred_btchw: torch.Tensor | None = None
        for i, t_cur in enumerate(timesteps):
            timestep_value = timestep_values[i]
            if only_step_index is not None and i != only_step_index:
                action = "skip_update"
            else:
                action = self._select_timestep_ablation_action(
                    ablation_config,
                    block_index=block_index,
                    step_index=i,
                    timestep_value=timestep_value,
                )
            effective_action = action
            t_expand = t_cur.reshape(1).repeat(current_latents.shape[0])
            step_marker = (
                "wan_s2v_timestep "
                f"block={block_index} step={i} timestep={timestep_value:.6g} "
                f"action={action}"
            )
            transformer_ms = None
            scheduler_ms = None
            cuda_graph_status = None
            step_started = self._timestep_profile_timestamp(
                profile_config, current_latents.device
            )
            with maybe_nvtx_range(step_marker, use_nvtx):
                if action == "skip_update":
                    noise_pred_btchw = None
                elif (
                    action == "reuse_previous_pred"
                    and previous_noise_pred_btchw is not None
                ):
                    noise_pred_btchw = previous_noise_pred_btchw
                elif action == "zero_pred":
                    noise_pred_btchw = torch.zeros_like(noise_latents_btchw)
                else:
                    if action == "reuse_previous_pred":
                        effective_action = "full_fallback_no_previous_pred"
                    transformer_started = self._timestep_profile_timestamp(
                        profile_config, current_latents.device
                    )
                    with maybe_nvtx_range(f"{step_marker} transformer", use_nvtx):
                        with (
                            torch.autocast(
                                device_type=current_platform.device_type,
                                dtype=dit_dtype,
                                enabled=autocast_enabled,
                            ),
                            set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ),
                        ):
                            if cache_state.metadata is not None:
                                cache_state.prepare_kv_update_plans(
                                    noisy_seq_len=(
                                        int(current_latents.shape[2])
                                        * cache_state.metadata.frame_seq_length
                                    ),
                                    current_start=current_start,
                                    cache_start=0,
                                )
                            transformer_kwargs = {
                                "hidden_states": current_latents,
                                "timestep": t_expand,
                                "encoder_hidden_states": prompt_embeds,
                                "ref_latents": block_bundle.ref_latents,
                                "motion_latents": block_bundle.motion_latents,
                                "cond_states": block_bundle.cond_states,
                                "audio_input": block_bundle.audio_input,
                                "audio_emb": block_bundle.audio_emb,
                                "motion_frames": block_bundle.motion_frames,
                                "add_last_motion": block_bundle.add_last_motion,
                                "drop_motion_frames": block_bundle.drop_motion_frames,
                                "kv_cache": cache_state.kv_cache,
                                "crossattn_cache": crossattn_cache,
                                "current_start": current_start,
                                "cache_start": None,
                                "audio_start_frame": audio_start_frame,
                                "stream_r1_mode": True,
                            }
                            use_cuda_graph, cuda_graph_status = (
                                self._can_use_timestep_cuda_graph(
                                    cuda_graph_config,
                                    block_index=block_index,
                                    step_index=i,
                                    device=current_latents.device,
                                    crossattn_cache=crossattn_cache,
                                )
                            )
                            if use_cuda_graph:
                                forward_batch = getattr(
                                    get_forward_context(), "forward_batch", None
                                )
                                sequence_shard_enabled = bool(
                                    forward_batch is not None
                                    and getattr(
                                        forward_batch,
                                        "enable_sequence_shard",
                                        False,
                                    )
                                    and _safe_sp_world_size() > 1
                                )
                                noise_pred_bcthw, cuda_graph_status = (
                                    self._timestep_cuda_graph_runner.run(
                                        kwargs=transformer_kwargs,
                                        forward_fn=self.transformer,
                                        step_index=i,
                                        current_start=current_start,
                                        audio_start_frame=audio_start_frame,
                                        sequence_shard_enabled=sequence_shard_enabled,
                                        allow_capture=allow_timestep_cuda_graph_capture,
                                    )
                                )
                            else:
                                noise_pred_bcthw = self.transformer(
                                    **transformer_kwargs
                                )
                    transformer_ms = (
                        self._timestep_profile_timestamp(
                            profile_config, current_latents.device
                        )
                        - transformer_started
                    ) * 1000.0
                    if cuda_graph_config.log and cuda_graph_status is not None:
                        self.log_info(
                            "Wan S2V timestep CUDA graph block=%d step=%d "
                            "status=%s cached_graphs=%d",
                            block_index,
                            i,
                            cuda_graph_status,
                            self._timestep_cuda_graph_runner.cached_graph_count,
                        )
                    if action == "scale_pred":
                        noise_pred_bcthw = noise_pred_bcthw * ablation_config.scale
                    noise_pred_btchw = noise_pred_bcthw.permute(0, 2, 1, 3, 4)

                if cuda_graph_config.enabled and cuda_graph_status is not None:
                    timestep_cuda_graph_statuses.append(
                        {
                            "block_idx": int(block_index),
                            "step_idx": int(i),
                            "status": cuda_graph_status,
                        }
                    )

                if noise_pred_btchw is not None:
                    previous_noise_pred_btchw = noise_pred_btchw
                    scheduler_started = self._timestep_profile_timestamp(
                        profile_config, current_latents.device
                    )
                    with maybe_nvtx_range(f"{step_marker} scheduler", use_nvtx):
                        pred_video_btchw = pred_noise_to_pred_video(
                            pred_noise=noise_pred_btchw.flatten(0, 1),
                            noise_input_latent=noise_latents_btchw.flatten(0, 1),
                            timestep=t_cur.reshape(1),
                            scheduler=self.scheduler,
                        ).unflatten(0, noise_pred_btchw.shape[:2])

                        if i < timesteps.numel() - 1:
                            next_timestep = (
                                timesteps[i + 1]
                                .reshape(1)
                                .to(device=current_latents.device)
                            )
                            if step_noises_btchw is None:
                                noise = torch.randn(
                                    video_raw_latent_shape,
                                    dtype=pred_video_btchw.dtype,
                                    generator=generator,
                                    device=current_latents.device,
                                )
                            else:
                                noise = step_noises_btchw[i]
                                if noise.shape != video_raw_latent_shape:
                                    raise ValueError(
                                        "Stream-R1 S2V precomputed step noise shape "
                                        f"mismatch: got {tuple(noise.shape)}, "
                                        f"expected {tuple(video_raw_latent_shape)}"
                                    )
                                if noise.device != current_latents.device:
                                    noise = noise.to(
                                        current_latents.device, non_blocking=True
                                    )
                                if noise.dtype != pred_video_btchw.dtype:
                                    noise = noise.to(dtype=pred_video_btchw.dtype)
                            noise_latents_btchw = self.scheduler.add_noise(
                                pred_video_btchw.flatten(0, 1),
                                noise.flatten(0, 1),
                                next_timestep,
                            ).unflatten(0, pred_video_btchw.shape[:2])
                            current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                        else:
                            current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)
                    scheduler_ms = (
                        self._timestep_profile_timestamp(
                            profile_config, current_latents.device
                        )
                        - scheduler_started
                    ) * 1000.0

            step_ms = (
                self._timestep_profile_timestamp(profile_config, current_latents.device)
                - step_started
            ) * 1000.0
            if profile_config.enabled or effective_action != "full":
                row = {
                    "rank": _safe_distributed_rank(),
                    "block_idx": int(block_index),
                    "step_idx": int(i),
                    "timestep": round(timestep_value, 6),
                    "action": effective_action,
                    "transformer_ms": (
                        None if transformer_ms is None else round(transformer_ms, 3)
                    ),
                    "scheduler_ms": (
                        None if scheduler_ms is None else round(scheduler_ms, 3)
                    ),
                    "total_ms": round(step_ms, 3),
                }
                if cuda_graph_config.enabled:
                    row["cuda_graph"] = cuda_graph_status or "not_run"
                timestep_profile_rows.append(row)
                if profile_config.log or (
                    ablation_config.log and effective_action != "full"
                ):
                    self.log_info(
                        "Wan S2V timestep block=%d step=%d timestep=%.6f "
                        "action=%s transformer_ms=%s scheduler_ms=%s total_ms=%.3f",
                        block_index,
                        i,
                        timestep_value,
                        effective_action,
                        "n/a" if transformer_ms is None else f"{transformer_ms:.3f}",
                        "n/a" if scheduler_ms is None else f"{scheduler_ms:.3f}",
                        step_ms,
                    )

        anchor_first_frame = bool(
            _resolve_request_value(
                batch,
                self.server_args,
                "anchor_first_frame",
                "s2v_anchor_first_frame",
                False,
            )
        )
        if anchor_first_frame and block_start == 0:
            current_latents = current_latents.clone()
            current_latents[:, :, :1] = block_bundle.ref_latents.to(
                device=current_latents.device,
                dtype=current_latents.dtype,
            )

        return current_latents

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        sp_world_size = _safe_sp_world_size()
        if sp_world_size > 1:
            # Stream-R1 S2V owns its block-wise loop, so it must opt into the
            # transformer-internal SP path instead of relying on the generic
            # denoising stage to shard latents before this stage runs.
            batch.enable_sequence_shard = True
        latents = batch.latents.to(device=device, dtype=dit_dtype)
        latent_frames = latents.shape[2]
        attention_request = self._resolve_attention_request(
            batch, server_args, latent_frames
        )
        num_frame_per_block = attention_request.num_frame_per_block
        timesteps = self._prepare_timesteps(batch, server_args, device)
        if timesteps.numel() == 0:
            raise ValueError("Stream-R1 S2V requires at least one timestep")

        bundle = build_wan_s2v_condition_bundle(
            batch,
            server_args,
            latents=latents,
            dtype=dit_dtype,
            device=device,
        )
        prompt_embeds = bundle.prompt_embeds
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]

        if (batch.guidance_scale or 1.0) > 1 and _has_negative_prompt_embeds(batch):
            self.log_warning(
                "Stream-R1 S2V no-KV mode does not run CFG in this phase; "
                "negative_prompt_embeds will be ignored."
            )

        patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
        _, _, _, latent_h, latent_w = latents.shape
        frame_seq_length = (latent_h // patch_size[1]) * (latent_w // patch_size[2])
        self._configure_deep_gemm_for_latents(
            latents,
            patch_size=patch_size,
            sp_size=sp_world_size,
            latent_frames=num_frame_per_block,
        )
        self._configure_transformer_attention(attention_request)
        cache_state = self._prepare_cache_state(
            request=attention_request,
            batch_size=latents.shape[0],
            frame_seq_length=frame_seq_length,
            dtype=dit_dtype,
            device=device,
        )
        self._guard_cache_runtime(cache_state)
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )

        self.load_model()
        try:
            self._maybe_cache_audio_embeddings(
                bundle,
                dtype=dit_dtype,
                autocast_enabled=autocast_enabled,
            )
            use_crossattn_cache = bool(
                _resolve_request_value(
                    batch,
                    server_args,
                    "stream_r1_crossattn_cache",
                    "stream_r1_crossattn_cache",
                    False,
                )
            )
            # Text context is constant across Stream-R1 chunks and refreshes.
            # The cache object is stage-owned so startup pre-captured graphs
            # can keep using the same external K/V tensor addresses.
            crossattn_cache = self._prepare_request_crossattn_cache(
                use_crossattn_cache
            )
            latent_warm_start_config = self._resolve_latent_warm_start_config(
                batch,
                server_args,
            )
            clean_refresh_config = self._resolve_clean_context_refresh_config(
                batch,
                server_args,
            )
            previous_clean_latents: torch.Tensor | None = None
            for block_start in range(0, latent_frames, num_frame_per_block):
                block_end = block_start + num_frame_per_block
                block_index = block_start // num_frame_per_block
                block_bundle = bundle.slice(
                    block_start,
                    num_frame_per_block,
                    policy=bundle.control_policy,
                )
                current_latents = latents[:, :, block_start:block_end, :, :]
                step_decision = self._select_adaptive_timesteps(
                    batch=batch,
                    server_args=server_args,
                    block_bundle=block_bundle,
                    timesteps=timesteps,
                    block_index=block_index,
                )
                current_latents, _ = self.apply_stream_r1_latent_warm_start(
                    batch=batch,
                    server_args=server_args,
                    block_latents=current_latents,
                    previous_clean_latents=previous_clean_latents,
                    timesteps=step_decision.timesteps,
                    block_index=block_index,
                    config=latent_warm_start_config,
                )
                current_latents = self.denoise_stream_r1_block(
                    batch=batch,
                    server_args=server_args,
                    block_latents=current_latents,
                    block_bundle=block_bundle,
                    block_start=block_start,
                    frame_seq_length=frame_seq_length,
                    timesteps=step_decision.timesteps,
                    prompt_embeds=prompt_embeds,
                    cache_state=cache_state,
                    crossattn_cache=crossattn_cache,
                    generator=generator,
                    dit_dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                    block_index=block_index,
                )

                latents[:, :, block_start:block_end, :, :] = current_latents
                previous_clean_latents = current_latents.detach()
                clean_refresh_decision = self.select_clean_context_refresh(
                    batch=batch,
                    server_args=server_args,
                    block_index=block_index,
                    config=clean_refresh_config,
                )
                if clean_refresh_decision.refresh:
                    self._clean_context_refresh(
                        block_latents=current_latents,
                        prompt_embeds=prompt_embeds,
                        block_bundle=block_bundle,
                        current_start=block_start * frame_seq_length,
                        attention_request=attention_request,
                        cache_state=cache_state,
                        dtype=dit_dtype,
                        autocast_enabled=autocast_enabled,
                        forward_batch=batch,
                        server_args=server_args,
                        block_index=block_index,
                        crossattn_cache=crossattn_cache,
                    )
        finally:
            self.offload_model()

        batch.latents = latents
        return batch


class WanS2VDenoisingDispatchStage(PipelineStage):
    """Select legacy or Stream-R1 S2V denoising per config/request."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.default_stage = WanS2VDenoisingStage(transformer, scheduler)
        self.stream_r1_stage = WanS2VStreamR1DenoisingStage(transformer, scheduler)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        stream_r1_mode = bool(
            _resolve_request_value(
                batch,
                server_args,
                "stream_r1_mode",
                "stream_r1_mode",
                False,
            )
        )
        if stream_r1_mode:
            return self.stream_r1_stage.forward(batch, server_args)
        return self.default_stage.forward(batch, server_args)
