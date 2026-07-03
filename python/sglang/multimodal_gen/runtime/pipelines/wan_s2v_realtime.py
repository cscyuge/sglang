# SPDX-License-Identifier: Apache-2.0
"""Realtime session runner for Wan2.2-S2V Stream-R1 inference."""

from __future__ import annotations

import gc
import json
import os
import time
from collections import OrderedDict
from copy import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VAudioEncodingStage,
    WanS2VConditionBundle,
    WanS2VDenoisingDispatchStage,
    build_wan_s2v_condition_bundle,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.chunk_timeline import (
    emit_chunk_timeline,
    flashtalk_chunk_timeline_path,
    is_flashtalk_filler_audio_meta,
    read_flashtalk_audio_chunk_meta,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class AudioRingBuffer:
    """Fixed-size numpy ring buffer for 16 kHz session audio."""

    __slots__ = ("_buf", "_pos", "_cap")

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._cap = capacity

    def extend(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        n = len(samples)
        if n == 0:
            return
        if n >= self._cap:
            np.copyto(self._buf, samples[-self._cap :])
            self._pos = 0
            return
        end = self._pos + n
        if end <= self._cap:
            self._buf[self._pos : end] = samples
        else:
            first = self._cap - self._pos
            self._buf[self._pos :] = samples[:first]
            self._buf[: n - first] = samples[first:]
        self._pos = end % self._cap

    def snapshot(self) -> np.ndarray:
        if self._pos == 0:
            return self._buf.copy()
        return np.concatenate((self._buf[self._pos :], self._buf[: self._pos]))


def _session_audio_chunk_path(session_dir: str, chunk_idx: int) -> str:
    return os.path.join(session_dir, "audio_chunks", f"chunk_{chunk_idx:04d}.npy")


def _session_audio_chunk_meta(session_dir: str, chunk_idx: int) -> dict[str, Any]:
    return read_flashtalk_audio_chunk_meta(session_dir, chunk_idx)


def _wait_for_session_audio_chunk(
    session_dir: str,
    chunk_idx: int,
    cancel_file: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 0.05,
) -> np.ndarray | None:
    chunk_path = _session_audio_chunk_path(session_dir, chunk_idx)
    end_path = os.path.join(session_dir, "end")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(end_path):
            return None
        if cancel_file and os.path.exists(cancel_file):
            return None
        if os.path.exists(chunk_path):
            try:
                return np.load(chunk_path)
            except Exception:
                time.sleep(0.01)
                try:
                    return np.load(chunk_path)
                except Exception:
                    return None
        time.sleep(poll_interval)
    return None


def _safe_world_rank() -> int:
    try:
        return get_world_rank()
    except Exception:
        return 0


def _pipeline_config_value(
    server_args: ServerArgs,
    key: str,
    default: Any,
) -> Any:
    return getattr(server_args.pipeline_config, key, default)


def _audio_window_after_extend(
    audio_window: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    audio_window = np.asarray(audio_window, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) == 0:
        return audio_window.copy()
    if len(samples) >= len(audio_window):
        return samples[-len(audio_window) :].copy()

    output = audio_window.copy()
    n = len(samples)
    output[:-n] = output[n:]
    output[-n:] = samples
    return output


class _WanS2VWav2VecCudaGraphRunner:
    def __init__(self, num_warmups: int = 2):
        self.graph = None
        self.num_warmups = num_warmups
        self.static_input = None
        self.static_output = None
        self._captured_shape: tuple[int, ...] | None = None
        self._captured_video_frames: int | None = None

    @property
    def is_captured(self) -> bool:
        return self.graph is not None

    def can_replay(self, audio_feature: torch.Tensor, num_video_frames: int) -> bool:
        return (
            self.is_captured
            and self._captured_shape == tuple(audio_feature.shape)
            and self._captured_video_frames == int(num_video_frames)
        )

    def capture(
        self,
        forward_fn,
        sample_input: torch.Tensor,
        num_video_frames: int,
    ) -> None:
        self.static_input = sample_input.clone()

        def run_once():
            return forward_fn(self.static_input)

        compiled_output = run_once()
        torch.cuda.synchronize(sample_input.device)
        del compiled_output

        stream = torch.cuda.Stream(device=sample_input.device)
        stream.wait_stream(torch.cuda.current_stream(sample_input.device))
        with torch.cuda.stream(stream):
            for _ in range(self.num_warmups):
                run_once()
        torch.cuda.current_stream(sample_input.device).wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = run_once()
        self._captured_shape = tuple(sample_input.shape)
        self._captured_video_frames = int(num_video_frames)

    def replay(self, audio_feature: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(audio_feature)
        self.graph.replay()
        return self.static_output


class _WanS2VWav2VecCudaGraphCache:
    def __init__(self, max_graphs: int = 16):
        self.max_graphs = max(1, int(max_graphs))
        self.runners: OrderedDict[tuple[Any, ...], _WanS2VWav2VecCudaGraphRunner] = (
            OrderedDict()
        )

    @staticmethod
    def key_from_sample(
        sample_feature: torch.Tensor,
        audio_window_video_frames: int,
    ) -> tuple[Any, ...]:
        return (
            tuple(sample_feature.shape),
            str(sample_feature.dtype),
            str(sample_feature.device),
            int(audio_window_video_frames),
        )

    def get(self, key: tuple[Any, ...]) -> _WanS2VWav2VecCudaGraphRunner | None:
        runner = self.runners.get(key)
        if runner is not None:
            self.runners.move_to_end(key)
        return runner

    def put(self, key: tuple[Any, ...], runner: _WanS2VWav2VecCudaGraphRunner) -> None:
        self.runners[key] = runner
        self.runners.move_to_end(key)
        while len(self.runners) > self.max_graphs:
            self.runners.popitem(last=False)


class _WanS2VStreamingVAECudaGraphRunner:
    """CUDA graph runner for steady-state Wan VAE streaming decode.

    The first realtime block must run eager to initialize Wan VAE temporal
    feature caches. The graph is captured only for later blocks where
    ``first_chunk`` is false for every latent frame and every active cache
    entry has stable initialized state and shape.
    """

    def __init__(self, num_warmups: int = 1):
        self.graph = None
        self.num_warmups = num_warmups
        self.static_input = None
        self.static_output = None
        self.cache_input_map: list[torch.Tensor] | None = None
        self.cache_output_map: list[torch.Tensor] | None = None
        self._captured_shape: tuple[int, ...] | None = None
        self._disabled = False

    @property
    def is_captured(self) -> bool:
        return self.graph is not None

    @property
    def disabled(self) -> bool:
        return self._disabled

    def disable(self) -> None:
        self._disabled = True

    @staticmethod
    def _clone_cache_map(cache_map: list[Any]) -> list[torch.Tensor | None]:
        cloned = []
        for item in cache_map:
            cloned.append(
                item.detach().clone() if isinstance(item, torch.Tensor) else None
            )
        return cloned

    @staticmethod
    def cache_ready(cache_map: list[Any]) -> bool:
        return bool(cache_map) and any(
            isinstance(item, torch.Tensor) for item in cache_map
        )

    @staticmethod
    def _copy_cache_map_(
        dst: list[torch.Tensor | None] | None,
        src: list[torch.Tensor | None] | None,
    ) -> None:
        if dst is None or src is None:
            raise RuntimeError("Wan VAE graph cache maps are not initialized")
        if len(dst) != len(src):
            raise RuntimeError(
                f"Wan VAE graph cache length changed: {len(dst)} != {len(src)}"
            )
        for idx, (dst_tensor, src_tensor) in enumerate(zip(dst, src)):
            if dst_tensor is None and src_tensor is None:
                continue
            if dst_tensor is None or src_tensor is None:
                raise RuntimeError(
                    "Wan VAE graph cache entry changed initialized state at "
                    f"entry {idx}"
                )
            if dst_tensor.shape != src_tensor.shape:
                raise RuntimeError(
                    "Wan VAE graph cache shape changed at entry "
                    f"{idx}: {tuple(dst_tensor.shape)} != {tuple(src_tensor.shape)}"
                )
            dst_tensor.copy_(src_tensor)

    def can_replay(self, latents: torch.Tensor) -> bool:
        return (
            not self.disabled
            and self.is_captured
            and self._captured_shape == tuple(latents.shape)
        )

    def _run_warmups(
        self,
        decode_fn,
        sample_input: torch.Tensor,
        live_cache_map: list[Any],
    ) -> None:
        if self.num_warmups <= 0:
            return
        stream = torch.cuda.Stream(device=sample_input.device)
        stream.wait_stream(torch.cuda.current_stream(sample_input.device))
        with torch.cuda.stream(stream):
            for _ in range(self.num_warmups):
                warm_cache = self._clone_cache_map(live_cache_map)
                warm_output = decode_fn(sample_input, warm_cache, False)
                del warm_output, warm_cache
        torch.cuda.current_stream(sample_input.device).wait_stream(stream)

    def capture(self, decode_fn, sample_input: torch.Tensor, live_cache_map: list[Any]):
        self._run_warmups(decode_fn, sample_input, live_cache_map)
        torch.cuda.synchronize(sample_input.device)

        self.static_input = sample_input.detach().clone()
        self.cache_input_map = self._clone_cache_map(live_cache_map)
        capture_cache_map = list(self.cache_input_map)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = decode_fn(self.static_input, capture_cache_map, False)

        self.cache_output_map = capture_cache_map
        # Capturing records the VAE decode but does not produce a usable result
        # for this realtime block. Replay once immediately so the capture block
        # has valid frames and the steady-state cache advances exactly once.
        self.graph.replay()
        self._copy_cache_map_(self.cache_input_map, self.cache_output_map)
        self._copy_cache_map_(live_cache_map, self.cache_output_map)
        self._captured_shape = tuple(sample_input.shape)
        return self.static_output

    def replay(self, latents: torch.Tensor, live_cache_map: list[Any] | None = None):
        self.static_input.copy_(latents)
        if live_cache_map is not None:
            self._copy_cache_map_(self.cache_input_map, live_cache_map)
        self.graph.replay()
        if live_cache_map is not None:
            self._copy_cache_map_(live_cache_map, self.cache_output_map)
        self._copy_cache_map_(self.cache_input_map, self.cache_output_map)
        return self.static_output


class _WanS2VStreamingVAECudaGraphCache:
    def __init__(self, max_graphs: int = 16):
        self.max_graphs = max(1, int(max_graphs))
        self.runners: OrderedDict[
            tuple[Any, ...], _WanS2VStreamingVAECudaGraphRunner
        ] = OrderedDict()

    @staticmethod
    def key_from_latents(latents: torch.Tensor) -> tuple[Any, ...]:
        return (tuple(latents.shape), str(latents.dtype), str(latents.device))

    def get(
        self, latents: torch.Tensor
    ) -> _WanS2VStreamingVAECudaGraphRunner | None:
        key = self.key_from_latents(latents)
        runner = self.runners.get(key)
        if runner is not None:
            self.runners.move_to_end(key)
        return runner

    def put(
        self,
        latents: torch.Tensor,
        runner: _WanS2VStreamingVAECudaGraphRunner,
    ) -> None:
        key = self.key_from_latents(latents)
        self.runners[key] = runner
        self.runners.move_to_end(key)
        while len(self.runners) > self.max_graphs:
            self.runners.popitem(last=False)

    @property
    def cached_graph_count(self) -> int:
        return len(self.runners)


@dataclass
class _PrefetchedAudioChunk:
    audio_chunk_idx: int
    audio: np.ndarray | None
    meta: dict[str, Any]
    end_requested: bool
    audio_input: torch.Tensor | None = None
    event: Any = None
    audio_s: float = 0.0
    audio_cpu_s: float = 0.0
    audio_gpu_enqueue_s: float = 0.0
    prepared_block: "_PreparedWanS2VBlock | None" = None


@dataclass
class _PreparedWanS2VBlock:
    block_idx: int
    latents: torch.Tensor
    step_noises_btchw: tuple[torch.Tensor, ...]
    bundle: WanS2VConditionBundle | None = None
    prompt_embeds: torch.Tensor | None = None
    event: Any = None
    prepare_s: float = 0.0
    latent_s: float = 0.0
    step_noise_s: float = 0.0
    condition_s: float = 0.0


@dataclass
class _BufferedWanS2VFrameBlock:
    block_idx: int
    frames: torch.Tensor
    frame_count: int
    frame_start_idx: int
    audio_chunk: np.ndarray
    audio_chunk_idx: int
    audio_meta: dict[str, Any]
    audio_prefetched: bool


@dataclass
class _WanS2VStreamingVAEState:
    enabled: bool
    initialized: bool = False
    decoded_latent_frames: int = 0
    last_decode_mode: str = "eager"


class WanS2VRealtimeSessionRunner:
    """Consume session audio chunks and emit Wan S2V frames block-by-block.

    This runner is intentionally separate from the normal batch pipeline. It
    reuses the same model stages, but keeps prompt/reference state resident and
    drives Stream-R1 one latent block at a time.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.stages = pipeline.stages

    def _wav2vec_cuda_graph_cache(self) -> _WanS2VWav2VecCudaGraphCache:
        cache = getattr(self.pipeline, "_wan_s2v_wav2vec_cuda_graph_cache", None)
        if cache is None:
            cache = _WanS2VWav2VecCudaGraphCache()
            setattr(self.pipeline, "_wan_s2v_wav2vec_cuda_graph_cache", cache)
        return cache

    def _vae_cuda_graph_cache(self) -> _WanS2VStreamingVAECudaGraphCache:
        cache = getattr(self.pipeline, "_wan_s2v_vae_cuda_graph_cache", None)
        if cache is None:
            cache = _WanS2VStreamingVAECudaGraphCache()
            setattr(self.pipeline, "_wan_s2v_vae_cuda_graph_cache", cache)
        return cache

    def _get_stage(self, stage_type: type) -> Any:
        for stage in self.stages:
            if isinstance(stage, stage_type):
                return stage
        raise RuntimeError(f"{stage_type.__name__} not found in Wan S2V pipeline")

    def _block_public_frames(
        self,
        server_args: ServerArgs,
        num_frame_per_block: int,
    ) -> int:
        temporal = int(
            getattr(
                server_args.pipeline_config.vae_config.arch_config,
                "scale_factor_temporal",
                4,
            )
        )
        return max(1, (num_frame_per_block - 1) * temporal + 1)

    def _block_output_frames(
        self,
        server_args: ServerArgs,
        num_frame_per_block: int,
        block_idx: int,
        *,
        use_streaming_vae_cache: bool,
    ) -> int:
        temporal = int(
            getattr(
                server_args.pipeline_config.vae_config.arch_config,
                "scale_factor_temporal",
                4,
            )
        )
        if use_streaming_vae_cache and block_idx > 0:
            return max(1, num_frame_per_block * temporal)
        return max(1, (num_frame_per_block - 1) * temporal + 1)

    def _prepare_reference_and_prompt(
        self,
        batch: Req,
        server_args: ServerArgs,
        block_public_frames: int,
    ) -> Req:
        original_num_frames = batch.num_frames
        batch.extra["wan_s2v_realtime_original_num_frames"] = original_num_frames
        batch.extra["wan_s2v_realtime_block_num_frames"] = block_public_frames
        batch.num_frames = block_public_frames

        for stage_type in (
            InputValidationStage,
            TextEncodingStage,
        ):
            stage = self._get_stage(stage_type)
            batch = stage(batch, server_args)
        return batch

    def _prepare_reference_latents_once(
        self,
        batch: Req,
        server_args: ServerArgs,
        image_stage: ImageVAEEncodingStage,
    ) -> Req:
        if getattr(batch, "image_latent", None) is not None:
            return batch
        if batch.latents is None:
            raise RuntimeError(
                "Wan S2V realtime reference VAE encoding requires block latents "
                "to establish the latent dtype."
            )
        return image_stage(batch, server_args)

    def _prepare_block_latents(
        self,
        batch: Req,
        server_args: ServerArgs,
        latent_stage: LatentPreparationStage,
        block_public_frames: int,
    ) -> torch.Tensor:
        batch.num_frames = block_public_frames
        batch.latents = None
        batch = latent_stage(batch, server_args)
        return batch.latents

    def _copy_batch_for_block_prepare(self, batch: Req) -> Req:
        block_batch = copy(batch)
        block_batch.extra = dict(batch.extra)
        block_batch.latents = None
        block_batch.latent_ids = None
        block_batch.raw_latent_shape = None
        return block_batch

    def _prepare_step_noises(
        self,
        block_latents: torch.Tensor,
        timesteps: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, ...]:
        num_noises = max(int(timesteps.numel()) - 1, 0)
        if num_noises == 0:
            return ()

        noise_shape = tuple(block_latents.permute(0, 2, 1, 3, 4).shape)
        return tuple(
            torch.randn(
                noise_shape,
                dtype=block_latents.dtype,
                generator=generator,
                device=block_latents.device,
            )
            for _ in range(num_noises)
        )

    def _prepare_block_inputs(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        latent_stage: LatentPreparationStage,
        block_idx: int,
        block_public_frames: int,
        timesteps: torch.Tensor | None,
        generator: torch.Generator | None,
        dit_dtype: torch.dtype,
        device: torch.device | str,
        audio_input: torch.Tensor | None,
        build_condition: bool,
        overlap_stream: torch.cuda.Stream | None = None,
    ) -> _PreparedWanS2VBlock:
        prepared_batch = self._copy_batch_for_block_prepare(batch)
        if audio_input is not None:
            prepared_batch.extra["audio_input"] = audio_input

        started = time.perf_counter()
        event = None

        def prepare_on_current_stream() -> tuple[
            torch.Tensor,
            tuple[torch.Tensor, ...],
            WanS2VConditionBundle | None,
            torch.Tensor | None,
            float,
            float,
            float,
        ]:
            latent_started = time.perf_counter()
            block_latents = self._prepare_block_latents(
                prepared_batch,
                server_args,
                latent_stage,
                block_public_frames,
            ).to(device=device, dtype=dit_dtype)
            prepared_batch.latents = block_latents
            latent_s = time.perf_counter() - latent_started

            if timesteps is None:
                step_noises = ()
                step_noise_s = 0.0
            else:
                step_noise_started = time.perf_counter()
                step_noises = self._prepare_step_noises(
                    block_latents,
                    timesteps,
                    generator,
                )
                step_noise_s = time.perf_counter() - step_noise_started

            condition_s = 0.0
            bundle = None
            prompt_embeds = None
            if build_condition:
                condition_started = time.perf_counter()
                bundle = build_wan_s2v_condition_bundle(
                    prepared_batch,
                    server_args,
                    latents=block_latents,
                    dtype=dit_dtype,
                    device=device,
                )
                prompt_embeds = bundle.prompt_embeds
                if isinstance(prompt_embeds, list):
                    prompt_embeds = prompt_embeds[0]
                condition_s = time.perf_counter() - condition_started
            return (
                block_latents,
                step_noises,
                bundle,
                prompt_embeds,
                latent_s,
                step_noise_s,
                condition_s,
            )

        if overlap_stream is not None and torch.cuda.is_available():
            with torch.cuda.device(device), torch.cuda.stream(overlap_stream):
                (
                    block_latents,
                    step_noises,
                    bundle,
                    prompt_embeds,
                    latent_s,
                    step_noise_s,
                    condition_s,
                ) = prepare_on_current_stream()
                event = overlap_stream.record_event()
        else:
            (
                block_latents,
                step_noises,
                bundle,
                prompt_embeds,
                latent_s,
                step_noise_s,
                condition_s,
            ) = prepare_on_current_stream()

        return _PreparedWanS2VBlock(
            block_idx=block_idx,
            latents=block_latents,
            step_noises_btchw=step_noises,
            bundle=bundle,
            prompt_embeds=prompt_embeds,
            event=event,
            prepare_s=time.perf_counter() - started,
            latent_s=latent_s,
            step_noise_s=step_noise_s,
            condition_s=condition_s,
        )

    def _should_gate_prefetch_for_timestep_cuda_graph(
        self,
        *,
        denoising_stage: Any,
        batch: Req,
        server_args: ServerArgs,
        block_idx: int,
        step_count: int,
    ) -> bool:
        timestep_graph_config = denoising_stage._resolve_timestep_cuda_graph_config(
            batch,
            server_args,
        )
        if (
            not timestep_graph_config.enabled
            or block_idx < timestep_graph_config.warmup_blocks
            or step_count <= 0
        ):
            return False
        if timestep_graph_config.step_indices:
            return any(
                0 <= int(step_index) < step_count
                for step_index in timestep_graph_config.step_indices
            )
        return True

    def _requires_preoutput_timestep_cuda_graph(
        self,
        *,
        denoising_stage: Any,
        batch: Req,
        server_args: ServerArgs,
        step_count: int,
    ) -> bool:
        timestep_graph_config = denoising_stage._resolve_timestep_cuda_graph_config(
            batch,
            server_args,
        )
        if (
            not timestep_graph_config.enabled
            or step_count <= 0
            or not torch.cuda.is_available()
        ):
            return False
        if timestep_graph_config.step_indices:
            return any(
                0 <= int(step_index) < step_count
                for step_index in timestep_graph_config.step_indices
            )
        return True

    @staticmethod
    def _should_allow_timestep_cuda_graph_capture(
        *,
        timestep_graph_output_started: bool,
    ) -> bool:
        # Startup/session prewarm should cover the common graph keys, but an
        # online request may still introduce a new supported shape. Let the
        # runner backfill that graph on miss instead of permanently falling
        # back or failing after streaming output has started.
        return True

    @staticmethod
    def _timestep_cuda_graph_ready_for_output(denoising_stage: Any) -> bool:
        statuses = getattr(denoising_stage, "_last_timestep_cuda_graph_statuses", [])
        relevant = [
            str(item.get("status"))
            for item in statuses
            if str(item.get("status"))
            not in {"disabled", "non_cuda", "warmup", "step_filtered"}
        ]
        return bool(relevant) and all(
            status in {"capture", "replay"} for status in relevant
        )

    @staticmethod
    def _timestep_cuda_graph_preoutput_error(denoising_stage: Any) -> str | None:
        statuses = getattr(denoising_stage, "_last_timestep_cuda_graph_statuses", [])
        relevant = [
            str(item.get("status"))
            for item in statuses
            if str(item.get("status"))
            not in {"disabled", "non_cuda", "warmup", "step_filtered"}
        ]
        bad_statuses = [
            status for status in relevant if status not in {"capture", "replay"}
        ]
        if not bad_statuses:
            return None
        return ", ".join(bad_statuses)

    @staticmethod
    def _write_progress(progress_file: str, block_idx: int) -> None:
        if _safe_world_rank() != 0:
            return
        try:
            with open(progress_file, "w", encoding="utf-8") as fp:
                fp.write(f"{block_idx + 1} -1")
        except Exception:
            pass

    def _save_streaming_frame_block(
        self,
        *,
        frame_block: _BufferedWanS2VFrameBlock,
        frame_dir: str,
        frame_executor: Any,
        frame_futures: list[Any],
        timeline_path: str | None,
        output_stream: torch.cuda.Stream | None,
    ) -> None:
        audio_meta = frame_block.audio_meta
        self.pipeline._save_streaming_frames(
            frame_block.frames,
            frame_block.block_idx,
            frame_dir,
            frame_executor,
            frame_futures,
            frame_block.frame_count,
            chunk_audio_data=frame_block.audio_chunk,
            timeline_path=timeline_path,
            audio_chunk_idx=frame_block.audio_chunk_idx,
            used_silence=False,
            audio_loaded=True,
            audio_prefetched=frame_block.audio_prefetched,
            chunk_source=audio_meta.get("chunk_source") or "audio",
            is_filler=is_flashtalk_filler_audio_meta(audio_meta),
            turn_id=audio_meta.get("turn_id"),
            frame_start_idx=frame_block.frame_start_idx,
            output_stream=output_stream,
        )

    def _decode_block_frames(
        self,
        decoding_stage: DecodingStage,
        latents: torch.Tensor,
        server_args: ServerArgs,
        stream_vae_state: _WanS2VStreamingVAEState,
        vae_graph_cache: _WanS2VStreamingVAECudaGraphCache | None = None,
    ) -> torch.Tensor:
        if not stream_vae_state.enabled:
            stream_vae_state.last_decode_mode = "eager"
            return decoding_stage.decode(latents, server_args)

        original_latents = latents
        vae = decoding_stage.vae
        required_attrs = (
            "use_feature_cache",
            "clear_cache",
            "post_quant_conv",
            "decoder",
            "config",
        )
        if not all(hasattr(vae, attr) for attr in required_attrs) or not bool(
            getattr(vae, "use_feature_cache", False)
        ):
            stream_vae_state.enabled = False
            stream_vae_state.last_decode_mode = "eager"
            return decoding_stage.decode(original_latents, server_args)

        try:
            from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
                feat_idx as wan_vae_feat_idx,
                first_chunk as wan_vae_first_chunk,
                forward_context as wan_vae_forward_context,
                unpatchify as wan_vae_unpatchify,
            )

            def decode_with_cache(
                prepared_latents: torch.Tensor,
                cache_map: list[Any],
                first_block: bool,
            ) -> torch.Tensor:
                x = vae.post_quant_conv(prepared_latents)
                outputs = []
                with wan_vae_forward_context(
                    feat_cache_arg=cache_map,
                    feat_idx_arg=0,
                ):
                    for idx in range(x.shape[2]):
                        wan_vae_feat_idx.set(0)
                        wan_vae_first_chunk.set(bool(first_block and idx == 0))
                        outputs.append(vae.decoder(x[:, :, idx : idx + 1, :, :]))

                image = torch.cat(outputs, dim=2)
                if getattr(vae.config, "patch_size", None) is not None:
                    image = wan_vae_unpatchify(
                        image, patch_size=getattr(vae.config, "patch_size")
                    )
                image = image.float().clamp(-1.0, 1.0)
                return (image / 2 + 0.5).clamp(0, 1)

            device = get_local_torch_device()
            vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            vae = vae.to(device=device, dtype=vae_dtype)
            latents = latents.to(device=device)
            latents = decoding_stage.scale_and_shift(latents, server_args)
            latents = server_args.pipeline_config.preprocess_decoding(
                latents, server_args, vae=vae
            )
            vae_autocast_enabled = (
                vae_dtype != torch.float32
            ) and not server_args.disable_autocast

            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                if not vae_autocast_enabled:
                    latents = latents.to(vae_dtype)
                else:
                    latents = latents.to(dtype=vae_dtype)

                if not stream_vae_state.initialized:
                    vae.clear_cache()

                if (
                    vae_graph_cache is not None
                    and stream_vae_state.initialized
                    and torch.cuda.is_available()
                    and latents.is_cuda
                ):
                    vae_graph_runner = vae_graph_cache.get(latents)
                    if (
                        vae_graph_runner is not None
                        and not vae_graph_runner.disabled
                        and vae_graph_runner.can_replay(latents)
                    ):
                        image = vae_graph_runner.replay(latents, vae._feat_map)
                        stream_vae_state.last_decode_mode = "graph_replay"
                    elif not _WanS2VStreamingVAECudaGraphRunner.cache_ready(
                        vae._feat_map
                    ):
                        image = decode_with_cache(
                            latents,
                            vae._feat_map,
                            not stream_vae_state.initialized,
                        )
                        stream_vae_state.last_decode_mode = "eager_cache_warmup"
                    else:
                        vae_graph_runner = _WanS2VStreamingVAECudaGraphRunner()
                        try:
                            image = vae_graph_runner.capture(
                                decode_with_cache,
                                latents,
                                vae._feat_map,
                            )
                            vae_graph_cache.put(latents, vae_graph_runner)
                            stream_vae_state.last_decode_mode = "graph_capture"
                            logger.info(
                                "Wan S2V VAE decode CUDA graph captured: "
                                "input_shape=%s output_shape=%s cache_entries=%d cached_graphs=%d",
                                tuple(latents.shape),
                                tuple(image.shape),
                                len(vae_graph_runner.cache_input_map or []),
                                vae_graph_cache.cached_graph_count,
                            )
                        except Exception as graph_exc:
                            vae_graph_runner.disable()
                            logger.warning(
                                "Wan S2V VAE decode CUDA graph disabled: %s",
                                graph_exc,
                            )
                            image = decode_with_cache(
                                latents,
                                vae._feat_map,
                                not stream_vae_state.initialized,
                            )
                            stream_vae_state.last_decode_mode = "eager"
                else:
                    image = decode_with_cache(
                        latents,
                        vae._feat_map,
                        not stream_vae_state.initialized,
                    )
                    stream_vae_state.last_decode_mode = "eager"

            stream_vae_state.initialized = True
            stream_vae_state.decoded_latent_frames += int(latents.shape[2])
            return image
        except Exception as exc:
            logger.warning(
                "Wan S2V streaming VAE cache disabled after decode failure: %s",
                exc,
            )
            try:
                vae.clear_cache()
            except Exception:
                pass
            stream_vae_state.enabled = False
            stream_vae_state.initialized = False
            stream_vae_state.last_decode_mode = "eager"
            return decoding_stage.decode(original_latents, server_args)

    def _prepare_audio_feature_cpu(
        self,
        audio_stage: WanS2VAudioEncodingStage,
        audio_window: np.ndarray,
        *,
        ensure_loaded: bool = True,
    ) -> torch.Tensor:
        sample_rate = 16000
        if ensure_loaded:
            audio_stage.load_model()

        speech_array = audio_window.astype(np.float32, copy=False)
        speech_array = audio_stage._loudness_norm(speech_array, sample_rate)

        if audio_stage.wav2vec_feature_extractor is not None:
            audio_feature_np = np.squeeze(
                audio_stage.wav2vec_feature_extractor(
                    speech_array, sampling_rate=sample_rate
                ).input_values
            )
            audio_feature_np = np.asarray(audio_feature_np, dtype=np.float32)
        else:
            audio_feature_np = np.asarray(speech_array, dtype=np.float32)

        audio_feature = torch.from_numpy(audio_feature_np).float().unsqueeze(0)
        if torch.cuda.is_available():
            try:
                audio_feature = audio_feature.pin_memory()
            except RuntimeError:
                pass
        return audio_feature

    def _encode_audio_feature(
        self,
        audio_stage: WanS2VAudioEncodingStage,
        audio_feature: torch.Tensor,
        *,
        target_audio_frames: int,
        audio_window_video_frames: int,
        wav2vec_graph_runner: _WanS2VWav2VecCudaGraphRunner | None = None,
    ) -> torch.Tensor:
        device = get_local_torch_device()
        if audio_feature.device != torch.device(device):
            audio_feature = audio_feature.to(
                device,
                non_blocking=bool(
                    audio_feature.device.type == "cpu" and audio_feature.is_pinned()
                ),
            )

        with set_forward_context(current_timestep=0, attn_metadata=None):
            if wav2vec_graph_runner is not None and wav2vec_graph_runner.can_replay(
                audio_feature, audio_window_video_frames
            ):
                audio_features = wav2vec_graph_runner.replay(audio_feature)
            else:
                audio_features = audio_stage.audio_encoder(
                    audio_feature,
                    num_video_frames=audio_window_video_frames,
                )

        if audio_features.shape[1] < target_audio_frames:
            pad = target_audio_frames - audio_features.shape[1]
            audio_features = torch.nn.functional.pad(
                audio_features, (0, 0, 0, 0, 0, pad)
            )
        else:
            audio_features = audio_features[:, -target_audio_frames:]

        return audio_features.permute(0, 2, 3, 1).contiguous()

    def _encode_audio_window(
        self,
        batch: Req,
        server_args: ServerArgs,
        audio_stage: WanS2VAudioEncodingStage,
        audio_window: np.ndarray,
        *,
        target_audio_frames: int,
        audio_window_video_frames: int,
        wav2vec_graph_runner: _WanS2VWav2VecCudaGraphRunner | None = None,
        ensure_loaded: bool = True,
    ) -> torch.Tensor:
        audio_feature = self._prepare_audio_feature_cpu(
            audio_stage,
            audio_window,
            ensure_loaded=ensure_loaded,
        )
        return self._encode_audio_feature(
            audio_stage,
            audio_feature,
            target_audio_frames=target_audio_frames,
            audio_window_video_frames=audio_window_video_frames,
            wav2vec_graph_runner=wav2vec_graph_runner,
        )

    def _prepare_wav2vec_cuda_graph(
        self,
        audio_stage: WanS2VAudioEncodingStage,
        *,
        audio_window_samples: int,
        audio_window_video_frames: int,
    ) -> _WanS2VWav2VecCudaGraphRunner | None:
        if not torch.cuda.is_available():
            return None

        audio_stage.load_model()
        sample_rate = 16000
        device = get_local_torch_device()
        sample_audio = np.zeros(audio_window_samples, dtype=np.float32)
        if audio_stage.wav2vec_feature_extractor is not None:
            sample_feature_np = np.squeeze(
                audio_stage.wav2vec_feature_extractor(
                    sample_audio,
                    sampling_rate=sample_rate,
                ).input_values
            )
        else:
            sample_feature_np = sample_audio
        sample_feature = (
            torch.from_numpy(np.asarray(sample_feature_np, dtype=np.float32))
            .float()
            .to(device)
            .unsqueeze(0)
        )

        def forward_fn(audio_feature: torch.Tensor) -> torch.Tensor:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return audio_stage.audio_encoder(
                    audio_feature,
                    num_video_frames=audio_window_video_frames,
                )

        cache = self._wav2vec_cuda_graph_cache()
        key = cache.key_from_sample(sample_feature, audio_window_video_frames)
        runner = cache.get(key)
        if runner is not None:
            logger.info(
                "Wan S2V Wav2Vec CUDA graph reused: input_shape=%s video_frames=%d",
                tuple(sample_feature.shape),
                audio_window_video_frames,
            )
            return runner

        runner = _WanS2VWav2VecCudaGraphRunner()
        runner.capture(forward_fn, sample_feature, audio_window_video_frames)
        cache.put(key, runner)
        logger.info(
            "Wan S2V Wav2Vec CUDA graph captured: input_shape=%s video_frames=%d",
            tuple(sample_feature.shape),
            audio_window_video_frames,
        )
        return runner

    @staticmethod
    def _wav2vec_cuda_graph_warmup_video_frames(
        batch: Req,
        server_args: ServerArgs,
        audio_window_seconds: float,
    ) -> tuple[int, ...]:
        fps_values: set[int] = set()

        def add_fps(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                for part in value.replace(";", ",").split(","):
                    add_fps(part.strip())
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add_fps(item)
                return
            try:
                fps = int(round(float(value)))
            except (TypeError, ValueError):
                return
            if fps > 0:
                fps_values.add(fps)

        add_fps(getattr(batch, "fps", None))
        add_fps(_pipeline_config_value(server_args, "fps", None))
        add_fps(
            _pipeline_config_value(
                server_args,
                "wan_s2v_realtime_wav2vec_cuda_graph_warmup_fps",
                (16, 24, 25),
            )
        )
        if not fps_values:
            fps_values.add(16)
        return tuple(
            sorted(
                {
                    max(1, int(round(float(audio_window_seconds) * fps)))
                    for fps in fps_values
                }
            )
        )

    def prewarm_realtime_cuda_graphs(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        if not torch.cuda.is_available():
            return
        if not bool(_pipeline_config_value(server_args, "stream_r1_mode", False)):
            return

        use_wav2vec_cuda_graph = bool(
            _pipeline_config_value(server_args, "wan_s2v_wav2vec_cuda_graph", False)
        )
        use_streaming_vae_cache = bool(
            _pipeline_config_value(server_args, "wan_s2v_streaming_vae_cache", True)
        )
        use_vae_cuda_graph = bool(
            _pipeline_config_value(server_args, "wan_s2v_vae_cuda_graph", False)
        ) and use_streaming_vae_cache
        if not use_wav2vec_cuda_graph and not use_vae_cuda_graph:
            return

        num_frame_per_block = int(
            batch.extra.get("num_frame_per_block")
            or _pipeline_config_value(server_args, "num_frame_per_block", 7)
        )
        block_public_frames = self._block_public_frames(
            server_args, num_frame_per_block
        )
        audio_window_seconds = float(
            _pipeline_config_value(
                server_args, "wan_s2v_realtime_audio_window_seconds", 8
            )
        )
        audio_window_samples = max(1, int(16000 * audio_window_seconds))
        wav2vec_video_frame_values = self._wav2vec_cuda_graph_warmup_video_frames(
            batch,
            server_args,
            audio_window_seconds,
        )

        if use_wav2vec_cuda_graph:
            try:
                audio_stage = self._get_stage(WanS2VAudioEncodingStage)
                for audio_window_video_frames in wav2vec_video_frame_values:
                    self._prepare_wav2vec_cuda_graph(
                        audio_stage,
                        audio_window_samples=audio_window_samples,
                        audio_window_video_frames=audio_window_video_frames,
                    )
            except Exception as exc:
                logger.warning(
                    "Wan S2V Wav2Vec CUDA graph startup prewarm failed: %s", exc
                )

        if not use_vae_cuda_graph:
            return
        try:
            latent_stage = self._get_stage(LatentPreparationStage)
            decoding_stage = self._get_stage(DecodingStage)
            decoding_stage.load_model()
            device = get_local_torch_device()
            dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
            sample_batch = self._copy_batch_for_block_prepare(batch)
            sample_latents = self._prepare_block_latents(
                sample_batch,
                server_args,
                latent_stage,
                block_public_frames,
            ).to(device=device, dtype=dit_dtype)
            vae_graph_cache = self._vae_cuda_graph_cache()
            if vae_graph_cache.get(sample_latents) is not None:
                logger.info(
                    "Wan S2V VAE decode CUDA graph reused during startup prewarm: "
                    "input_shape=%s cached_graphs=%d",
                    tuple(sample_latents.shape),
                    vae_graph_cache.cached_graph_count,
                )
                return

            stream_vae_state = _WanS2VStreamingVAEState(enabled=True)
            self._decode_block_frames(
                decoding_stage,
                sample_latents,
                server_args,
                stream_vae_state,
                vae_graph_cache=None,
            )
            self._decode_block_frames(
                decoding_stage,
                sample_latents,
                server_args,
                stream_vae_state,
                vae_graph_cache=vae_graph_cache,
            )
            logger.info(
                "Wan S2V VAE decode CUDA graph startup prewarm done: "
                "input_shape=%s cached_graphs=%d",
                tuple(sample_latents.shape),
                vae_graph_cache.cached_graph_count,
            )
        except Exception as exc:
            logger.warning(
                "Wan S2V VAE decode CUDA graph startup prewarm failed: %s", exc
            )
        finally:
            try:
                torch.cuda.synchronize(get_local_torch_device())
                self._get_stage(DecodingStage).vae.clear_cache()
            except Exception:
                pass

    def _prefetch_next_audio_chunk(
        self,
        *,
        batch: Req,
        server_args: ServerArgs,
        audio_stage: WanS2VAudioEncodingStage,
        latent_stage: LatentPreparationStage | None = None,
        session_dir: str,
        audio_chunk_idx: int,
        cancel_file: str | None,
        idle_policy: str,
        timeline_path: str | None,
        audio_window_snapshot: np.ndarray,
        target_audio_frames: int,
        audio_window_video_frames: int,
        wav2vec_graph_runner: _WanS2VWav2VecCudaGraphRunner | None,
        overlap_stream: torch.cuda.Stream | None,
        after_denoise_event: torch.cuda.Event | None,
        gpu_start_event: Event | None = None,
        prepare_block_idx: int | None = None,
        block_public_frames: int | None = None,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        dit_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        build_condition: bool = False,
    ) -> _PrefetchedAudioChunk:
        current_audio_idx, audio_chunk, audio_meta, end_requested = (
            self._next_audio_chunk(
                session_dir=session_dir,
                audio_chunk_idx=audio_chunk_idx,
                cancel_file=cancel_file,
                idle_policy=idle_policy,
                timeline_path=timeline_path,
            )
        )
        if end_requested:
            return _PrefetchedAudioChunk(
                audio_chunk_idx=current_audio_idx,
                audio=None,
                meta={},
                end_requested=True,
            )

        audio_window = _audio_window_after_extend(audio_window_snapshot, audio_chunk)
        audio_started = time.perf_counter()
        audio_feature = self._prepare_audio_feature_cpu(
            audio_stage,
            audio_window,
            ensure_loaded=False,
        )
        audio_cpu_s = time.perf_counter() - audio_started
        done_event = None
        if gpu_start_event is not None:
            gpu_start_event.wait()
        audio_gpu_started = time.perf_counter()
        if overlap_stream is not None and torch.cuda.is_available():
            device = get_local_torch_device()
            with torch.cuda.device(device):
                if after_denoise_event is not None:
                    overlap_stream.wait_event(after_denoise_event)
                with torch.cuda.stream(overlap_stream):
                    audio_input = self._encode_audio_feature(
                        audio_stage,
                        audio_feature,
                        target_audio_frames=target_audio_frames,
                        audio_window_video_frames=audio_window_video_frames,
                        wav2vec_graph_runner=wav2vec_graph_runner,
                    )
                    done_event = overlap_stream.record_event()
        else:
            audio_input = self._encode_audio_feature(
                audio_stage,
                audio_feature,
                target_audio_frames=target_audio_frames,
                audio_window_video_frames=audio_window_video_frames,
                wav2vec_graph_runner=wav2vec_graph_runner,
            )
        prepared_block = None
        if (
            prepare_block_idx is not None
            and latent_stage is not None
            and block_public_frames is not None
            and dit_dtype is not None
            and device is not None
        ):
            prepared_block = self._prepare_block_inputs(
                batch=batch,
                server_args=server_args,
                latent_stage=latent_stage,
                block_idx=prepare_block_idx,
                block_public_frames=block_public_frames,
                timesteps=timesteps,
                generator=generator,
                dit_dtype=dit_dtype,
                device=device,
                audio_input=audio_input,
                build_condition=build_condition,
                overlap_stream=overlap_stream,
            )
        audio_gpu_enqueue_s = time.perf_counter() - audio_gpu_started
        audio_s = time.perf_counter() - audio_started
        return _PrefetchedAudioChunk(
            audio_chunk_idx=current_audio_idx,
            audio=audio_chunk,
            meta=audio_meta,
            end_requested=False,
            audio_input=audio_input,
            event=done_event,
            audio_s=audio_s,
            audio_cpu_s=audio_cpu_s,
            audio_gpu_enqueue_s=audio_gpu_enqueue_s,
            prepared_block=prepared_block,
        )

    def _next_audio_chunk(
        self,
        *,
        session_dir: str,
        audio_chunk_idx: int,
        cancel_file: str | None,
        idle_policy: str,
        timeline_path: str | None,
    ) -> tuple[int, np.ndarray | None, dict[str, Any], bool]:
        """Return next usable audio chunk.

        The bool return is true when the session should end. With
        ``idle_policy=hold`` filler chunks are consumed and skipped without
        triggering GPU generation.
        """
        while True:
            audio = _wait_for_session_audio_chunk(
                session_dir,
                audio_chunk_idx,
                cancel_file=cancel_file,
                timeout=float(
                    os.environ.get("SGLANG_WAN_S2V_SESSION_AUDIO_TIMEOUT_S", "300")
                ),
            )
            if audio is None:
                return audio_chunk_idx, None, {}, True

            meta = _session_audio_chunk_meta(session_dir, audio_chunk_idx)
            is_filler = is_flashtalk_filler_audio_meta(meta)
            current_idx = audio_chunk_idx
            audio_chunk_idx += 1

            if idle_policy == "hold" and is_filler:
                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_filler_audio_held",
                    audio_chunk_idx=current_idx,
                    samples=int(len(audio)),
                )
                continue

            return current_idx, np.asarray(audio, dtype=np.float32), meta, False

    def _setup_frame_dir(
        self,
        batch: Req,
        server_args: ServerArgs,
        session_id: str,
        frames_per_chunk: int,
    ) -> tuple[str | None, ThreadPoolExecutor | None]:
        if _safe_world_rank() != 0:
            return None, None

        frame_dir = os.path.join(server_args.output_path, ".frames", session_id)
        os.makedirs(frame_dir, exist_ok=True)
        meta = {
            "num_chunks": None,
            "fps": batch.fps or 24,
            "frames_per_chunk": frames_per_chunk,
            "frames_per_first_chunk": frames_per_chunk,
            "variable_frames_per_chunk": True,
            "width": batch.width,
            "height": batch.height,
            "session": True,
            "runtime": "wan_s2v_realtime",
        }
        with open(os.path.join(frame_dir, "meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
        return frame_dir, ThreadPoolExecutor(max_workers=1)

    @torch.no_grad()
    def run(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        session_dir = batch.extra.get("session_dir")
        if not session_dir:
            raise RuntimeError(
                "Wan S2V realtime session requires batch.extra.session_dir"
            )

        session_id = (
            getattr(batch, "request_id", None)
            or getattr(batch, "output_file_name", None)
            or os.path.basename(session_dir.rstrip(os.sep))
        )
        timeline_path = batch.extra.get(
            "chunk_timeline_path"
        ) or flashtalk_chunk_timeline_path(session_dir)
        progress_dir = os.path.join(server_args.output_path, ".progress")
        os.makedirs(progress_dir, exist_ok=True)
        progress_file = os.path.join(progress_dir, session_id)
        cancel_file = os.path.join(progress_dir, f"{session_id}.cancel")

        denoising_dispatch = self._get_stage(WanS2VDenoisingDispatchStage)
        denoising_stage = denoising_dispatch.stream_r1_stage
        audio_stage = self._get_stage(WanS2VAudioEncodingStage)
        latent_stage = self._get_stage(LatentPreparationStage)
        image_stage = self._get_stage(ImageVAEEncodingStage)
        decoding_stage = self._get_stage(DecodingStage)

        stream_r1_mode = batch.extra.get("stream_r1_mode")
        if stream_r1_mode is None:
            stream_r1_mode = _pipeline_config_value(
                server_args, "stream_r1_mode", False
            )
        if not bool(stream_r1_mode):
            raise RuntimeError("Wan S2V realtime sessions require stream_r1_mode=true.")

        num_frame_per_block = int(
            batch.extra.get("num_frame_per_block")
            or _pipeline_config_value(server_args, "num_frame_per_block", 7)
        )
        block_public_frames = self._block_public_frames(
            server_args, num_frame_per_block
        )
        fps = int(batch.fps or 24)
        audio_window_seconds = float(
            _pipeline_config_value(
                server_args, "wan_s2v_realtime_audio_window_seconds", 8
            )
        )
        audio_window_samples = max(1, int(16000 * audio_window_seconds))
        audio_window_video_frames = max(1, int(round(audio_window_seconds * fps)))
        target_audio_frames = num_frame_per_block * 4
        idle_policy = str(
            _pipeline_config_value(server_args, "wan_s2v_idle_policy", "hold")
        ).lower()
        if idle_policy not in {"hold", "silence"}:
            idle_policy = "hold"
        use_wav2vec_cuda_graph = bool(
            _pipeline_config_value(server_args, "wan_s2v_wav2vec_cuda_graph", False)
        )
        use_audio_overlap = bool(
            _pipeline_config_value(server_args, "wan_s2v_audio_overlap", False)
        )
        if use_audio_overlap and not torch.cuda.is_available():
            use_audio_overlap = False
        use_latent_condition_overlap = bool(
            _pipeline_config_value(
                server_args,
                "wan_s2v_latent_condition_overlap",
                False,
            )
        )
        if use_latent_condition_overlap and not torch.cuda.is_available():
            use_latent_condition_overlap = False
        use_streaming_vae_cache = bool(
            _pipeline_config_value(server_args, "wan_s2v_streaming_vae_cache", True)
        )
        use_vae_cuda_graph = bool(
            _pipeline_config_value(server_args, "wan_s2v_vae_cuda_graph", False)
        )
        if use_vae_cuda_graph and not (
            torch.cuda.is_available() and use_streaming_vae_cache
        ):
            use_vae_cuda_graph = False
        adaptive_step_config = denoising_stage._resolve_adaptive_step_config(
            batch,
            server_args,
        )
        use_adaptive_steps = bool(adaptive_step_config.enabled)
        latent_warm_start_config = denoising_stage._resolve_latent_warm_start_config(
            batch,
            server_args,
        )
        clean_refresh_config = denoising_stage._resolve_clean_context_refresh_config(
            batch,
            server_args,
        )
        timestep_profile_config = denoising_stage._resolve_timestep_profile_config(
            batch,
            server_args,
        )
        timestep_ablation_config = denoising_stage._resolve_timestep_ablation_config(
            batch,
            server_args,
        )

        logger.info(
            "Wan S2V realtime session start: session=%s block_latent_frames=%d "
            "block_public_frames=%d fps=%d audio_window=%.2fs idle_policy=%s "
            "wav2vec_cuda_graph=%s audio_overlap=%s streaming_vae_cache=%s "
            "vae_cuda_graph=%s latent_condition_overlap=%s adaptive_steps=%s "
            "latent_warm_start=%s clean_refresh=%s/%d timestep_profile=%s "
            "timestep_ablation=%s",
            session_id,
            num_frame_per_block,
            block_public_frames,
            fps,
            audio_window_seconds,
            idle_policy,
            use_wav2vec_cuda_graph,
            use_audio_overlap,
            use_streaming_vae_cache,
            use_vae_cuda_graph,
            use_latent_condition_overlap,
            use_adaptive_steps,
            latent_warm_start_config.enabled,
            clean_refresh_config.mode,
            clean_refresh_config.interval,
            timestep_profile_config.enabled,
            timestep_ablation_config.mode,
        )
        emit_chunk_timeline(
            timeline_path,
            "wan_s2v_realtime_session_start",
            session_id=session_id,
            block_latent_frames=num_frame_per_block,
            block_public_frames=block_public_frames,
            fps=fps,
            audio_window_seconds=audio_window_seconds,
            idle_policy=idle_policy,
            wav2vec_cuda_graph=use_wav2vec_cuda_graph,
            audio_overlap=use_audio_overlap,
            streaming_vae_cache=use_streaming_vae_cache,
            vae_cuda_graph=use_vae_cuda_graph,
            latent_condition_overlap=use_latent_condition_overlap,
            adaptive_steps=use_adaptive_steps,
            adaptive_steps_log_only=adaptive_step_config.log_only,
            adaptive_steps_threshold=adaptive_step_config.threshold,
            adaptive_steps_aggressive_threshold=(
                adaptive_step_config.aggressive_threshold
            ),
            latent_warm_start={
                "enabled": latent_warm_start_config.enabled,
                "alpha": latent_warm_start_config.alpha,
                "mode": latent_warm_start_config.mode,
                "warmup_blocks": latent_warm_start_config.warmup_blocks,
                "timestep_index": latent_warm_start_config.timestep_index,
                "effective_sigma": latent_warm_start_config.effective_sigma,
            },
            clean_context_refresh={
                "mode": clean_refresh_config.mode,
                "interval": clean_refresh_config.interval,
                "warmup_blocks": clean_refresh_config.warmup_blocks,
            },
            timestep_profile={
                "enabled": timestep_profile_config.enabled,
                "log": timestep_profile_config.log,
                "nvtx": timestep_profile_config.nvtx,
                "synchronize": timestep_profile_config.synchronize,
            },
            timestep_ablation={
                "mode": timestep_ablation_config.mode,
                "step_indices": list(timestep_ablation_config.step_indices),
                "timestep_values": list(timestep_ablation_config.timestep_values),
                "block_indices": list(timestep_ablation_config.block_indices),
                "warmup_blocks": timestep_ablation_config.warmup_blocks,
                "value_tolerance": timestep_ablation_config.value_tolerance,
                "scale": timestep_ablation_config.scale,
                "log": timestep_ablation_config.log,
                "enabled": timestep_ablation_config.enabled,
            },
        )

        batch = self._prepare_reference_and_prompt(
            batch, server_args, block_public_frames
        )
        audio_ring = AudioRingBuffer(audio_window_samples)

        frame_dir, frame_executor = self._setup_frame_dir(
            batch, server_args, session_id, block_public_frames
        )
        frame_futures: list[Any] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()

        device = get_local_torch_device()
        output_stream = (
            torch.cuda.Stream(device=device)
            if frame_executor is not None and torch.cuda.is_available()
            else None
        )
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )
        audio_stage.load_model()
        wav2vec_graph_runner = None
        if use_wav2vec_cuda_graph:
            try:
                wav2vec_graph_runner = self._prepare_wav2vec_cuda_graph(
                    audio_stage,
                    audio_window_samples=audio_window_samples,
                    audio_window_video_frames=audio_window_video_frames,
                )
            except Exception as exc:
                logger.warning("Wan S2V Wav2Vec CUDA graph disabled: %s", exc)
                wav2vec_graph_runner = None

        audio_prefetch_pool: ThreadPoolExecutor | None = None
        audio_overlap_stream: torch.cuda.Stream | None = None
        prefetched_audio_future: Future | None = None
        prefetch_after_denoise_event: torch.cuda.Event | None = None
        prefetch_gpu_start_event: Event | None = None
        if use_audio_overlap or use_latent_condition_overlap:
            audio_prefetch_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="wan-s2v-audio-prefetch",
            )
            audio_overlap_stream = torch.cuda.Stream(device=device)
            logger.info(
                "Wan S2V prefetch overlap enabled: audio=%s latent_condition=%s",
                use_audio_overlap,
                use_latent_condition_overlap,
            )

        cache_state = None
        attention_request = None
        crossattn_cache: list[dict] | None = None
        frame_seq_length = None
        timesteps = None
        prompt_embeds = None
        reference_latents_ready = False
        stream_vae_state = _WanS2VStreamingVAEState(enabled=use_streaming_vae_cache)
        vae_graph_cache = self._vae_cuda_graph_cache() if use_vae_cuda_graph else None

        audio_chunk_idx = 0
        block_idx = 0
        frame_start_idx = 0
        end_requested = False
        previous_clean_latents: torch.Tensor | None = None
        timestep_graph_preoutput_required: bool | None = None
        timestep_graph_output_started = False
        buffered_frame_blocks: list[_BufferedWanS2VFrameBlock] = []

        decoding_stage.load_model()
        denoising_stage.load_model()
        try:
            while True:
                loop_started = time.perf_counter()
                audio_prefetched = False
                audio_cpu_s = 0.0
                audio_gpu_enqueue_s = 0.0
                audio_prefetch_wall_s = 0.0
                audio_prefetch_gate_s = 0.0
                audio_wait_s = 0.0
                latent_prefetched = False
                latent_wait_s = 0.0
                latent_prepare_wall_s = 0.0
                latent_prepare_gpu_s = 0.0
                step_noise_s = 0.0
                condition_s = 0.0
                latent_warm_start_s = 0.0
                latent_warm_start_applied = False
                clean_refresh_decision = None
                prepared_block = None
                if prefetched_audio_future is not None:
                    audio_wait_started = time.perf_counter()
                    try:
                        prefetched = prefetched_audio_future.result()
                    except Exception as exc:
                        logger.warning(
                            "Wan S2V audio prefetch failed; falling back to "
                            "synchronous audio encode: %s",
                            exc,
                        )
                        prefetched = None
                    audio_wait_s = time.perf_counter() - audio_wait_started
                    prefetched_audio_future = None
                else:
                    prefetched = None

                if prefetched is not None:
                    current_audio_idx = prefetched.audio_chunk_idx
                    end_requested = prefetched.end_requested
                    audio_chunk = prefetched.audio
                    audio_meta = prefetched.meta
                    prepared_block = prefetched.prepared_block
                    audio_chunk_idx = current_audio_idx + (0 if end_requested else 1)
                    if end_requested:
                        break
                    if prefetched.event is not None:
                        torch.cuda.current_stream(device).wait_event(prefetched.event)
                    audio_ring.extend(audio_chunk)
                    batch.extra["audio_input"] = prefetched.audio_input
                    audio_cpu_s = prefetched.audio_cpu_s
                    audio_gpu_enqueue_s = prefetched.audio_gpu_enqueue_s
                    audio_prefetch_wall_s = prefetched.audio_s
                    audio_prefetch_gate_s = max(
                        0.0,
                        prefetched.audio_s
                        - prefetched.audio_cpu_s
                        - prefetched.audio_gpu_enqueue_s,
                    )
                    audio_s = audio_wait_s
                    audio_prefetched = True
                    if prepared_block is not None:
                        latent_prefetched = True
                        latent_prepare_wall_s = prepared_block.prepare_s
                        latent_prepare_gpu_s = prepared_block.latent_s
                        step_noise_s = prepared_block.step_noise_s
                        condition_s = prepared_block.condition_s
                else:
                    current_audio_idx, audio_chunk, audio_meta, end_requested = (
                        self._next_audio_chunk(
                            session_dir=session_dir,
                            audio_chunk_idx=audio_chunk_idx,
                            cancel_file=cancel_file,
                            idle_policy=idle_policy,
                            timeline_path=timeline_path,
                        )
                    )
                    audio_chunk_idx = current_audio_idx + (0 if end_requested else 1)
                    if end_requested:
                        break
                    audio_ring.extend(audio_chunk)
                    audio_started = time.perf_counter()
                    batch.extra["audio_input"] = self._encode_audio_window(
                        batch,
                        server_args,
                        audio_stage,
                        audio_ring.snapshot(),
                        target_audio_frames=target_audio_frames,
                        audio_window_video_frames=audio_window_video_frames,
                        wav2vec_graph_runner=wav2vec_graph_runner,
                        ensure_loaded=False,
                    )
                    audio_s = time.perf_counter() - audio_started

                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_block_generation_start",
                    block_idx=block_idx,
                    audio_chunk_idx=current_audio_idx,
                    samples=int(len(audio_chunk)),
                    expected_output_frames=self._block_output_frames(
                        server_args,
                        num_frame_per_block,
                        block_idx,
                        use_streaming_vae_cache=stream_vae_state.enabled,
                    ),
                    chunk_source=audio_meta.get("chunk_source"),
                    is_filler=is_flashtalk_filler_audio_meta(audio_meta),
                    turn_id=audio_meta.get("turn_id"),
                    audio_prefetched=audio_prefetched,
                    latent_prefetched=latent_prefetched,
                )

                latent_started = time.perf_counter()
                block_step_noises: tuple[torch.Tensor, ...] | None = None
                bundle = None
                if prepared_block is not None and prepared_block.block_idx == block_idx:
                    latent_wait_started = time.perf_counter()
                    if prepared_block.event is not None:
                        torch.cuda.current_stream(device).wait_event(
                            prepared_block.event
                        )
                    latent_wait_s = time.perf_counter() - latent_wait_started
                    block_latents = prepared_block.latents
                    block_step_noises = prepared_block.step_noises_btchw
                    bundle = prepared_block.bundle
                    prompt_embeds = prepared_block.prompt_embeds
                else:
                    if prepared_block is not None:
                        logger.warning(
                            "Dropping prefetched Wan S2V block %s while executing "
                            "block %s",
                            prepared_block.block_idx,
                            block_idx,
                        )
                    block_latents = self._prepare_block_latents(
                        batch,
                        server_args,
                        latent_stage,
                        block_public_frames,
                    ).to(device=device, dtype=dit_dtype)
                    latent_prepare_gpu_s = time.perf_counter() - latent_started
                batch.latents = block_latents
                if not reference_latents_ready:
                    ref_started = time.perf_counter()
                    batch = self._prepare_reference_latents_once(
                        batch,
                        server_args,
                        image_stage,
                    )
                    reference_latents_ready = True
                    emit_chunk_timeline(
                        timeline_path,
                        "wan_s2v_reference_latents_ready",
                        block_idx=block_idx,
                        elapsed_ms=round(
                            (time.perf_counter() - ref_started) * 1000,
                            3,
                        ),
                    )

                if attention_request is None:
                    attention_request = denoising_stage._resolve_attention_request(
                        batch,
                        server_args,
                        block_latents.shape[2],
                    )
                    timesteps = denoising_stage._prepare_timesteps(
                        batch, server_args, device
                    )
                    if timesteps.numel() == 0:
                        raise ValueError(
                            "Wan S2V realtime session requires at least one timestep"
                        )
                    patch_size = (
                        server_args.pipeline_config.dit_config.arch_config.patch_size
                    )
                    _, _, _, latent_h, latent_w = block_latents.shape
                    frame_seq_length = (latent_h // patch_size[1]) * (
                        latent_w // patch_size[2]
                    )
                    denoising_stage._configure_transformer_attention(attention_request)
                    cache_state = denoising_stage._prepare_cache_state(
                        request=attention_request,
                        batch_size=block_latents.shape[0],
                        frame_seq_length=frame_seq_length,
                        dtype=dit_dtype,
                        device=device,
                    )
                    denoising_stage._guard_cache_runtime(cache_state)
                    crossattn_cache = denoising_stage._prepare_request_crossattn_cache(
                        True
                    )

                if bundle is None:
                    condition_started = time.perf_counter()
                    bundle = build_wan_s2v_condition_bundle(
                        batch,
                        server_args,
                        latents=block_latents,
                        dtype=dit_dtype,
                        device=device,
                    )
                    prompt_embeds = bundle.prompt_embeds
                    if isinstance(prompt_embeds, list):
                        prompt_embeds = prompt_embeds[0]
                    condition_s = time.perf_counter() - condition_started
                denoising_stage._maybe_cache_audio_embeddings(
                    bundle,
                    dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                )
                step_decision = denoising_stage._select_adaptive_timesteps(
                    batch=batch,
                    server_args=server_args,
                    block_bundle=bundle,
                    timesteps=timesteps,
                    block_index=block_idx,
                )
                block_timesteps = step_decision.timesteps
                latent_warm_start_started = time.perf_counter()
                block_latents, latent_warm_start_applied = (
                    denoising_stage.apply_stream_r1_latent_warm_start(
                        batch=batch,
                        server_args=server_args,
                        block_latents=block_latents,
                        previous_clean_latents=previous_clean_latents,
                        timesteps=block_timesteps,
                        block_index=block_idx,
                        config=latent_warm_start_config,
                    )
                )
                latent_warm_start_s = time.perf_counter() - latent_warm_start_started
                batch.latents = block_latents
                if block_step_noises is not None and len(block_step_noises) != max(
                    int(block_timesteps.numel()) - 1,
                    0,
                ):
                    block_step_noises = None
                if block_step_noises is None:
                    step_noise_started = time.perf_counter()
                    block_step_noises = self._prepare_step_noises(
                        block_latents,
                        block_timesteps,
                        generator,
                    )
                    step_noise_s = time.perf_counter() - step_noise_started
                latent_s = time.perf_counter() - latent_started
                if timestep_graph_preoutput_required is None:
                    timestep_graph_preoutput_required = (
                        self._requires_preoutput_timestep_cuda_graph(
                            denoising_stage=denoising_stage,
                            batch=batch,
                            server_args=server_args,
                            step_count=int(block_timesteps.numel()),
                        )
                    )
                    timestep_graph_output_started = (
                        not timestep_graph_preoutput_required
                    )

                # Timestep graph replay may include USP/NCCL work and KV cache
                # writes. Gate side-stream GPU prefetch until denoise finishes
                # so it cannot interleave with capture/replay.
                gate_prefetch_for_timestep_graph = (
                    self._should_gate_prefetch_for_timestep_cuda_graph(
                        denoising_stage=denoising_stage,
                        batch=batch,
                        server_args=server_args,
                        block_idx=block_idx,
                        step_count=int(block_timesteps.numel()),
                    )
                )

                if audio_prefetch_pool is not None and prefetched_audio_future is None:
                    if (
                        use_latent_condition_overlap
                        and not gate_prefetch_for_timestep_graph
                    ):
                        prefetch_after_denoise_event = None
                        prefetch_gpu_start_event = None
                    else:
                        prefetch_after_denoise_event = torch.cuda.Event()
                        prefetch_gpu_start_event = Event()
                    prefetched_audio_future = audio_prefetch_pool.submit(
                        self._prefetch_next_audio_chunk,
                        batch=batch,
                        server_args=server_args,
                        audio_stage=audio_stage,
                        latent_stage=(
                            latent_stage if use_latent_condition_overlap else None
                        ),
                        session_dir=session_dir,
                        audio_chunk_idx=audio_chunk_idx,
                        cancel_file=cancel_file,
                        idle_policy=idle_policy,
                        timeline_path=timeline_path,
                        audio_window_snapshot=audio_ring.snapshot(),
                        target_audio_frames=target_audio_frames,
                        audio_window_video_frames=audio_window_video_frames,
                        wav2vec_graph_runner=wav2vec_graph_runner,
                        overlap_stream=audio_overlap_stream,
                        after_denoise_event=prefetch_after_denoise_event,
                        gpu_start_event=prefetch_gpu_start_event,
                        prepare_block_idx=(
                            block_idx + 1 if use_latent_condition_overlap else None
                        ),
                        block_public_frames=block_public_frames,
                        timesteps=None if use_adaptive_steps else timesteps,
                        generator=generator,
                        dit_dtype=dit_dtype,
                        device=device,
                        build_condition=use_latent_condition_overlap,
                    )

                denoise_started = time.perf_counter()
                current_latents = denoising_stage.denoise_stream_r1_block(
                    batch=batch,
                    server_args=server_args,
                    block_latents=block_latents,
                    block_bundle=bundle,
                    block_start=block_idx * num_frame_per_block,
                    frame_seq_length=frame_seq_length,
                    timesteps=block_timesteps,
                    prompt_embeds=prompt_embeds,
                    cache_state=cache_state,
                    crossattn_cache=crossattn_cache,
                    generator=generator,
                    dit_dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                    audio_start_frame=0,
                    step_noises_btchw=block_step_noises,
                    block_index=block_idx,
                    allow_timestep_cuda_graph_capture=(
                        self._should_allow_timestep_cuda_graph_capture(
                            timestep_graph_output_started=timestep_graph_output_started
                        )
                    ),
                )
                timestep_profile_rows = list(
                    getattr(denoising_stage, "_last_timestep_profile_rows", [])
                )
                denoise_loop_s = time.perf_counter() - denoise_started
                clean_refresh_started = time.perf_counter()
                clean_refresh_decision = denoising_stage.select_clean_context_refresh(
                    batch=batch,
                    server_args=server_args,
                    block_index=block_idx,
                    config=clean_refresh_config,
                )
                if clean_refresh_decision.refresh:
                    denoising_stage._clean_context_refresh(
                        block_latents=current_latents,
                        prompt_embeds=prompt_embeds,
                        block_bundle=bundle,
                        current_start=(
                            block_idx * num_frame_per_block * frame_seq_length
                        ),
                        attention_request=attention_request,
                        cache_state=cache_state,
                        dtype=dit_dtype,
                        autocast_enabled=autocast_enabled,
                        forward_batch=batch,
                        crossattn_cache=crossattn_cache,
                        audio_start_frame=0,
                    )
                clean_refresh_s = time.perf_counter() - clean_refresh_started
                batch.latents = current_latents
                previous_clean_latents = current_latents.detach()
                denoise_s = denoise_loop_s + clean_refresh_s

                if prefetch_after_denoise_event is not None:
                    prefetch_after_denoise_event.record(
                        torch.cuda.current_stream(device)
                    )
                    if prefetch_gpu_start_event is not None:
                        prefetch_gpu_start_event.set()
                    prefetch_after_denoise_event = None
                    prefetch_gpu_start_event = None

                decode_started = time.perf_counter()
                frames = self._decode_block_frames(
                    decoding_stage,
                    batch.latents,
                    server_args,
                    stream_vae_state,
                    vae_graph_cache=vae_graph_cache,
                )
                frames = server_args.pipeline_config.post_decoding(frames, server_args)
                decode_s = time.perf_counter() - decode_started
                frame_count = int(frames.shape[2])
                if not timestep_graph_output_started or (
                    stream_vae_state.last_decode_mode
                    in {"graph_capture", "graph_replay"}
                ):
                    # VAE graph replay returns a reusable static output tensor.
                    # Any frame block that can outlive the current replay must
                    # own its contents before another replay overwrites it.
                    frames = frames.detach().clone()
                frame_block = _BufferedWanS2VFrameBlock(
                    block_idx=block_idx,
                    frames=frames,
                    frame_count=frame_count,
                    frame_start_idx=frame_start_idx,
                    audio_chunk=audio_chunk,
                    audio_chunk_idx=current_audio_idx,
                    audio_meta=audio_meta,
                    audio_prefetched=audio_prefetched,
                )
                frame_start_idx += frame_count

                stream_started = time.perf_counter()
                if timestep_graph_output_started:
                    self._save_streaming_frame_block(
                        frame_block=frame_block,
                        frame_dir=frame_dir,
                        frame_executor=frame_executor,
                        frame_futures=frame_futures,
                        timeline_path=timeline_path,
                        output_stream=output_stream,
                    )
                    self._write_progress(progress_file, block_idx)
                else:
                    buffered_frame_blocks.append(frame_block)
                    preoutput_error = self._timestep_cuda_graph_preoutput_error(
                        denoising_stage
                    )
                    if preoutput_error is not None:
                        raise RuntimeError(
                            "Wan S2V timestep CUDA graph pre-output capture failed "
                            f"with status: {preoutput_error}"
                        )
                    if self._timestep_cuda_graph_ready_for_output(denoising_stage):
                        timestep_graph_output_started = True
                        emit_chunk_timeline(
                            timeline_path,
                            "wan_s2v_timestep_cuda_graph_prewarm_done",
                            block_idx=block_idx,
                            buffered_blocks=len(buffered_frame_blocks),
                            cached_graphs=(
                                denoising_stage._timestep_cuda_graph_runner.cached_graph_count
                            ),
                        )
                        logger.info(
                            "Wan S2V timestep CUDA graph prewarm done: "
                            "block=%d buffered_blocks=%d cached_graphs=%d",
                            block_idx,
                            len(buffered_frame_blocks),
                            denoising_stage._timestep_cuda_graph_runner.cached_graph_count,
                        )
                        for buffered_block in buffered_frame_blocks:
                            self._save_streaming_frame_block(
                                frame_block=buffered_block,
                                frame_dir=frame_dir,
                                frame_executor=frame_executor,
                                frame_futures=frame_futures,
                                timeline_path=timeline_path,
                                output_stream=output_stream,
                            )
                        self._write_progress(progress_file, block_idx)
                        buffered_frame_blocks.clear()
                stream_s = time.perf_counter() - stream_started

                total_s = time.perf_counter() - loop_started
                logger.info(
                    "Wan S2V realtime block %d: audio=%.3fs latent=%.3fs "
                    "warm_start=%s/%.3fs steps=%d/%d denoise_loop=%.3fs "
                    "refresh=%s/%s/%.3fs decode=%.3fs stream=%.3fs total=%.3fs",
                    block_idx,
                    audio_s,
                    latent_s,
                    latent_warm_start_applied,
                    latent_warm_start_s,
                    step_decision.step_count,
                    step_decision.base_step_count,
                    denoise_loop_s,
                    clean_refresh_decision.refresh,
                    clean_refresh_decision.reason,
                    clean_refresh_s,
                    decode_s,
                    stream_s,
                    total_s,
                )
                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_block_generation_done",
                    block_idx=block_idx,
                    audio_chunk_idx=current_audio_idx,
                    frame_count=frame_count,
                    frame_start_idx=frame_start_idx - frame_count,
                    vae_decode_mode=stream_vae_state.last_decode_mode,
                    adaptive_steps={
                        "enabled": step_decision.enabled,
                        "log_only": step_decision.log_only,
                        "reduced": step_decision.reduced,
                        "reason": step_decision.reason,
                        "rel_l1": (
                            step_decision.rel_l1
                            if step_decision.rel_l1 is not None
                            and np.isfinite(step_decision.rel_l1)
                            else None
                        ),
                        "step_count": step_decision.step_count,
                        "base_step_count": step_decision.base_step_count,
                        "target_step_count": step_decision.target_step_count,
                    },
                    latent_warm_start={
                        "enabled": latent_warm_start_config.enabled,
                        "applied": latent_warm_start_applied,
                        "alpha": latent_warm_start_config.alpha,
                        "mode": latent_warm_start_config.mode,
                        "timestep_index": latent_warm_start_config.timestep_index,
                        "effective_sigma": latent_warm_start_config.effective_sigma,
                    },
                    clean_context_refresh={
                        "mode": clean_refresh_decision.mode,
                        "refresh": clean_refresh_decision.refresh,
                        "reason": clean_refresh_decision.reason,
                        "interval": clean_refresh_decision.interval,
                    },
                    timestep_profile=timestep_profile_rows,
                    timings={
                        "audio_ms": round(audio_s * 1000, 3),
                        "audio_cpu_ms": round(audio_cpu_s * 1000, 3),
                        "audio_gpu_enqueue_ms": round(
                            audio_gpu_enqueue_s * 1000,
                            3,
                        ),
                        "audio_prefetch_wall_ms": round(
                            audio_prefetch_wall_s * 1000,
                            3,
                        ),
                        "audio_prefetch_gate_ms": round(
                            audio_prefetch_gate_s * 1000,
                            3,
                        ),
                        "latent_ms": round(latent_s * 1000, 3),
                        "latent_prefetch_wait_ms": round(
                            latent_wait_s * 1000,
                            3,
                        ),
                        "latent_prefetch_wall_ms": round(
                            latent_prepare_wall_s * 1000,
                            3,
                        ),
                        "latent_prepare_ms": round(
                            latent_prepare_gpu_s * 1000,
                            3,
                        ),
                        "step_noise_ms": round(step_noise_s * 1000, 3),
                        "condition_ms": round(condition_s * 1000, 3),
                        "latent_warm_start_ms": round(
                            latent_warm_start_s * 1000,
                            3,
                        ),
                        "denoise_ms": round(denoise_s * 1000, 3),
                        "denoise_loop_ms": round(denoise_loop_s * 1000, 3),
                        "clean_refresh_ms": round(clean_refresh_s * 1000, 3),
                        "decode_ms": round(decode_s * 1000, 3),
                        "stream_ms": round(stream_s * 1000, 3),
                        "total_ms": round(total_s * 1000, 3),
                    },
                    audio_prefetched=audio_prefetched,
                    latent_prefetched=latent_prefetched,
                )
                block_idx += 1
        finally:
            if prefetch_gpu_start_event is not None:
                prefetch_gpu_start_event.set()
                prefetch_gpu_start_event = None
            if prefetched_audio_future is not None:
                prefetched_audio_future.cancel()
                prefetched_audio_future = None
            if audio_prefetch_pool is not None:
                try:
                    audio_prefetch_pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    audio_prefetch_pool.shutdown(wait=False)
            if audio_overlap_stream is not None:
                try:
                    audio_overlap_stream.synchronize()
                except Exception:
                    pass
            try:
                if stream_vae_state.initialized:
                    decoding_stage.vae.clear_cache()
            except Exception:
                pass
            try:
                denoising_stage.offload_model()
            except Exception:
                pass
            try:
                audio_stage.offload_model()
            except Exception:
                pass
            try:
                decoding_stage.offload_model()
            except Exception:
                pass
            self.pipeline._post_loop_cleanup(
                gc_was_enabled,
                frame_futures,
                frame_executor,
                frame_dir,
            )
            try:
                os.remove(progress_file)
            except FileNotFoundError:
                pass

        emit_chunk_timeline(
            timeline_path,
            "wan_s2v_realtime_session_done",
            session_id=session_id,
            blocks=block_idx,
            ended=end_requested,
        )
        logger.info(
            "Wan S2V realtime session finished: session=%s blocks=%d",
            session_id,
            block_idx,
        )
        return OutputBatch(output=None, output_file_paths=[], metrics=batch.metrics)
