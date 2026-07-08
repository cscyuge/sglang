# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mmap
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
    _tensor_sample_to_rgb24_array,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

FRAME_STORE_MAGIC = b"SGLRTFR1"
FRAME_STORE_HEADER_SIZE = 128
FRAME_STORE_STATUS_OFFSET = 8
FRAME_STORE_SIZE_OFFSET = 16
FRAME_STORE_MATERIALIZE_US_OFFSET = 24
FRAME_STORE_PRODUCER_WAIT_US_OFFSET = 32
FRAME_STORE_GPU_COPY_US_OFFSET = 40
FRAME_STORE_MMAP_WRITE_US_OFFSET = 48
FRAME_STORE_PRODUCER_DECODE_US_OFFSET = 56
FRAME_STORE_PRODUCER_POST_US_OFFSET = 64
FRAME_STORE_PRODUCER_CLONE_US_OFFSET = 72
FRAME_STORE_PRODUCER_TOTAL_US_OFFSET = 80
FRAME_STORE_PRODUCER_DENOISE_US_OFFSET = 88
FRAME_STORE_PRODUCER_REFRESH_US_OFFSET = 96
FRAME_STORE_PRODUCER_DENOISE_TO_READY_US_OFFSET = 104
FRAME_STORE_PENDING = 0
FRAME_STORE_READY = 1
FRAME_STORE_ERROR = 2
FRAME_STORE_DIR_ENV = "SGLANG_REALTIME_FRAME_STORE_DIR"
FRAME_STORE_TIMEOUT_ENV = "SGLANG_REALTIME_FRAME_STORE_TIMEOUT_MS"
FRAME_STORE_DEFAULT_TIMEOUT_MS = 30000.0
FRAME_STORE_WRITER_DEFER_ENV = "SGLANG_REALTIME_FRAME_STORE_DEFER_MS"
FRAME_STORE_DEFAULT_WRITER_DEFER_MS = 0.0
FRAME_STORE_FLUSH_ENV = "SGLANG_REALTIME_FRAME_STORE_FLUSH"
FRAME_STORE_DIRECT_TORCH_COPY_ENV = "SGLANG_REALTIME_FRAME_STORE_DIRECT_TORCH_COPY"
FRAME_STORE_CUDA_SIDE_STREAM_ENV = "SGLANG_REALTIME_FRAME_STORE_CUDA_SIDE_STREAM"
_RAW_RGB_FRAME_STORE_WRITE_REQUEST_ATTR = "_raw_rgb_frame_store_write_request"


@dataclass
class RealtimeRawFrameBatch:
    payload: bytes
    num_frames: int
    bytes_per_frame: int

    @property
    def raw_size(self) -> int:
        return len(self.payload)

    def __len__(self) -> int:
        return self.num_frames

    def slice_frames(self, start: int, stop: int) -> "RealtimeRawFrameBatch":
        start = max(0, start)
        stop = min(self.num_frames, stop)
        payload_start = start * self.bytes_per_frame
        payload_stop = stop * self.bytes_per_frame
        return RealtimeRawFrameBatch(
            payload=self.payload[payload_start:payload_stop],
            num_frames=max(0, stop - start),
            bytes_per_frame=self.bytes_per_frame,
        )

    def iter_frames(self):
        for offset in range(0, len(self.payload), self.bytes_per_frame):
            yield self.payload[offset : offset + self.bytes_per_frame]


@dataclass
class RealtimeFrameStoreHandle:
    path: str
    payload_size: int
    num_frames: int
    metadata: dict[str, Any]
    content_type: str = RAW_RGB_CONTENT_TYPE
    header_size: int = FRAME_STORE_HEADER_SIZE


@dataclass
class RealtimeFrameStoreLoadResult:
    frame_batches: list[RealtimeRawFrameBatch]
    wait_ms: float
    read_ms: float
    materialize_ms: float
    producer_wait_ms: float
    gpu_copy_ms: float
    mmap_write_ms: float
    producer_decode_ms: float
    producer_post_ms: float
    producer_clone_ms: float
    producer_total_ms: float
    producer_denoise_ms: float
    producer_refresh_ms: float
    producer_denoise_to_ready_ms: float


@dataclass
class _FrameStoreWriteRequest:
    samples: list[torch.Tensor]
    handles: list[RealtimeFrameStoreHandle]
    request_id: str
    chunk_idx: int
    ready_events: list[Any] | None = None
    producer_timing_events: dict[str, Any] | None = None


@dataclass
class _FrameStoreWriteTimings:
    producer_wait_ms: float = 0.0
    gpu_copy_ms: float = 0.0
    mmap_write_ms: float = 0.0
    producer_decode_ms: float = 0.0
    producer_post_ms: float = 0.0
    producer_clone_ms: float = 0.0
    producer_total_ms: float = 0.0
    producer_denoise_ms: float = 0.0
    producer_refresh_ms: float = 0.0
    producer_denoise_to_ready_ms: float = 0.0


_FRAME_STORE_WRITER_QUEUE: queue.SimpleQueue | None = None
_FRAME_STORE_WRITER_THREAD: threading.Thread | None = None
_FRAME_STORE_WRITER_LOCK = threading.Lock()
_FRAME_STORE_CUDA_STREAMS: dict[int, torch.cuda.Stream] = {}
_FRAME_STORE_PINNED_STAGING: dict[int, torch.Tensor] = {}
_FRAME_STORE_CUDA_RESOURCE_LOCK = threading.Lock()


def get_realtime_frame_store_timeout_ms() -> float:
    raw_env = os.environ.get(FRAME_STORE_TIMEOUT_ENV)
    if raw_env is not None and raw_env != "":
        return max(0.0, float(raw_env))
    return FRAME_STORE_DEFAULT_TIMEOUT_MS


def get_realtime_frame_store_defer_ms() -> float:
    raw_env = os.environ.get(FRAME_STORE_WRITER_DEFER_ENV)
    if raw_env is not None and raw_env != "":
        return max(0.0, float(raw_env))
    return FRAME_STORE_DEFAULT_WRITER_DEFER_MS


def should_flush_realtime_frame_store() -> bool:
    raw_env = os.environ.get(FRAME_STORE_FLUSH_ENV)
    if raw_env is None or raw_env == "":
        return False
    return raw_env.strip().lower() in {"1", "true", "yes", "on"}


def should_use_direct_torch_frame_store_copy() -> bool:
    raw_env = os.environ.get(FRAME_STORE_DIRECT_TORCH_COPY_ENV)
    if raw_env is None or raw_env == "":
        return True
    return raw_env.strip().lower() in {"1", "true", "yes", "on"}


def should_use_cuda_side_stream_frame_store() -> bool:
    raw_env = os.environ.get(FRAME_STORE_CUDA_SIDE_STREAM_ENV)
    if raw_env is None or raw_env == "":
        return True
    return raw_env.strip().lower() in {"1", "true", "yes", "on"}


def _ms_to_us_bytes(value_ms: float) -> bytes:
    return max(0, int(round(value_ms * 1000.0))).to_bytes(
        8,
        "little",
        signed=False,
    )


def _elapsed_event_ms(
    events: dict[str, Any] | None,
    start_name: str,
    end_name: str,
) -> float:
    if not events:
        return 0.0
    start_event = events.get(start_name)
    end_event = events.get(end_name)
    if start_event is None or end_event is None:
        return 0.0
    try:
        return max(0.0, float(start_event.elapsed_time(end_event)))
    except Exception:
        logger.exception(
            "failed to read realtime frame store producer CUDA timing: %s -> %s",
            start_name,
            end_name,
        )
        return 0.0


def _producer_event_timings(
    events: dict[str, Any] | None,
) -> _FrameStoreWriteTimings:
    if not events:
        return _FrameStoreWriteTimings()
    return _FrameStoreWriteTimings(
        producer_denoise_ms=_elapsed_event_ms(events, "denoise_start", "denoise_end"),
        producer_refresh_ms=_elapsed_event_ms(events, "refresh_start", "refresh_end"),
        producer_decode_ms=_elapsed_event_ms(events, "decode_start", "decode_end"),
        producer_post_ms=_elapsed_event_ms(events, "decode_end", "post_end"),
        producer_clone_ms=_elapsed_event_ms(events, "post_end", "ready"),
        producer_total_ms=_elapsed_event_ms(events, "decode_start", "ready"),
        producer_denoise_to_ready_ms=_elapsed_event_ms(
            events,
            "denoise_start",
            "ready",
        ),
    )


def _frame_store_dir() -> str:
    configured = os.environ.get(FRAME_STORE_DIR_ENV)
    if configured:
        directory = configured
    elif os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        directory = "/dev/shm/sglang_realtime_frames"
    else:
        directory = os.path.join(tempfile.gettempdir(), "sglang_realtime_frames")
    os.makedirs(directory, exist_ok=True)
    return directory


def _create_store_file(payload_size: int) -> str:
    path = os.path.join(_frame_store_dir(), f"{uuid.uuid4().hex}.rgb")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, FRAME_STORE_HEADER_SIZE + payload_size)
        with mmap.mmap(fd, FRAME_STORE_HEADER_SIZE + payload_size) as mm:
            mm[: len(FRAME_STORE_MAGIC)] = FRAME_STORE_MAGIC
            mm[FRAME_STORE_STATUS_OFFSET] = FRAME_STORE_PENDING
            mm[FRAME_STORE_SIZE_OFFSET : FRAME_STORE_SIZE_OFFSET + 8] = int(
                payload_size
            ).to_bytes(8, "little", signed=False)
            mm.flush()
    finally:
        os.close(fd)
    return path


def _cleanup_store_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("failed to unlink realtime frame store file %s: %s", path, exc)


def _mark_store_status(path: str, status: int) -> None:
    try:
        with open(path, "r+b") as fp:
            with mmap.mmap(fp.fileno(), FRAME_STORE_HEADER_SIZE) as mm:
                if mm[: len(FRAME_STORE_MAGIC)] != FRAME_STORE_MAGIC:
                    return
                mm[FRAME_STORE_STATUS_OFFSET] = status
                mm.flush()
    except Exception:
        logger.exception("failed to mark realtime frame store status path=%s", path)


def _iter_tensor_samples(output: torch.Tensor) -> list[torch.Tensor]:
    if output.dim() == 5:
        return list(output)
    return [output]


def _sample_frame_shape(sample: torch.Tensor) -> tuple[int, int, int]:
    if sample.dim() == 3:
        _, height, width = sample.shape
        return 1, int(height), int(width)
    if sample.dim() == 4:
        _, frames, height, width = sample.shape
        return int(frames), int(height), int(width)
    raise ValueError(f"unsupported realtime tensor frame shape: {tuple(sample.shape)}")


def can_use_raw_rgb_frame_store(output: Any, req: Req) -> bool:
    return (
        isinstance(output, torch.Tensor)
        and not req.enable_frame_interpolation
        and not req.enable_upscaling
    )


def create_raw_rgb_frame_store_handles(
    output: torch.Tensor,
    req: Req,
) -> tuple[list[RealtimeFrameStoreHandle], dict[str, Any]]:
    handles: list[RealtimeFrameStoreHandle] = []
    frame_metadata: dict[str, Any] = {}
    for sample in _iter_tensor_samples(output):
        num_frames, height, width = _sample_frame_shape(sample)
        bytes_per_frame = width * height * RAW_RGB_CHANNELS
        payload_size = num_frames * bytes_per_frame
        metadata = {
            "format": "rgb24",
            "width": width,
            "height": height,
            "channels": RAW_RGB_CHANNELS,
            "bytes_per_frame": bytes_per_frame,
            "frame_store": "shared_memory_file",
        }
        path = _create_store_file(payload_size)
        handles.append(
            RealtimeFrameStoreHandle(
                path=path,
                payload_size=payload_size,
                num_frames=num_frames,
                metadata=metadata,
            )
        )
        if not frame_metadata:
            frame_metadata = dict(metadata)

    frame_metadata["timings"] = {
        "raw_frame_materialize_async": True,
        "raw_frame_store_enqueue_ms": 0.0,
    }
    logger.info(
        "created realtime raw RGB frame store handles: request_id=%s chunk_idx=%s "
        "handles=%d total_bytes=%d",
        req.request_id,
        req.block_idx,
        len(handles),
        sum(handle.payload_size for handle in handles),
    )
    return handles, frame_metadata


def _write_sample_to_store(
    *,
    sample: torch.Tensor,
    handle: RealtimeFrameStoreHandle,
    ready_event: Any | None = None,
    producer_timing_events: dict[str, Any] | None = None,
) -> None:
    started = time.monotonic()
    fd = os.open(handle.path, os.O_RDWR)
    try:
        total_size = handle.header_size + handle.payload_size
        with mmap.mmap(fd, total_size) as mm:
            if mm[: len(FRAME_STORE_MAGIC)] != FRAME_STORE_MAGIC:
                raise RuntimeError("invalid realtime frame store magic")
            if (
                sample.is_cuda
                and should_use_cuda_side_stream_frame_store()
                and torch.cuda.is_available()
            ):
                timings = _write_sample_to_store_cuda_side_stream(
                    sample=sample,
                    mm=mm,
                    handle=handle,
                    ready_event=ready_event,
                )
            elif should_use_direct_torch_frame_store_copy():
                timings = _write_sample_to_store_direct_torch(
                    sample=sample,
                    mm=mm,
                    handle=handle,
                )
            else:
                timings = _write_sample_to_store_numpy(
                    sample=sample,
                    mm=mm,
                    handle=handle,
                )
            producer_timings = _producer_event_timings(producer_timing_events)
            timings.producer_decode_ms = producer_timings.producer_decode_ms
            timings.producer_post_ms = producer_timings.producer_post_ms
            timings.producer_clone_ms = producer_timings.producer_clone_ms
            timings.producer_total_ms = producer_timings.producer_total_ms
            timings.producer_denoise_ms = producer_timings.producer_denoise_ms
            timings.producer_refresh_ms = producer_timings.producer_refresh_ms
            timings.producer_denoise_to_ready_ms = (
                producer_timings.producer_denoise_to_ready_ms
            )
            materialize_us = max(0, int(round((time.monotonic() - started) * 1e6)))
            mm[
                FRAME_STORE_MATERIALIZE_US_OFFSET : FRAME_STORE_MATERIALIZE_US_OFFSET
                + 8
            ] = materialize_us.to_bytes(8, "little", signed=False)
            mm[
                FRAME_STORE_PRODUCER_WAIT_US_OFFSET : FRAME_STORE_PRODUCER_WAIT_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_wait_ms)
            mm[
                FRAME_STORE_GPU_COPY_US_OFFSET : FRAME_STORE_GPU_COPY_US_OFFSET + 8
            ] = _ms_to_us_bytes(timings.gpu_copy_ms)
            mm[
                FRAME_STORE_MMAP_WRITE_US_OFFSET : FRAME_STORE_MMAP_WRITE_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.mmap_write_ms)
            mm[
                FRAME_STORE_PRODUCER_DECODE_US_OFFSET : FRAME_STORE_PRODUCER_DECODE_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_decode_ms)
            mm[
                FRAME_STORE_PRODUCER_POST_US_OFFSET : FRAME_STORE_PRODUCER_POST_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_post_ms)
            mm[
                FRAME_STORE_PRODUCER_CLONE_US_OFFSET : FRAME_STORE_PRODUCER_CLONE_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_clone_ms)
            mm[
                FRAME_STORE_PRODUCER_TOTAL_US_OFFSET : FRAME_STORE_PRODUCER_TOTAL_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_total_ms)
            mm[
                FRAME_STORE_PRODUCER_DENOISE_US_OFFSET : FRAME_STORE_PRODUCER_DENOISE_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_denoise_ms)
            mm[
                FRAME_STORE_PRODUCER_REFRESH_US_OFFSET : FRAME_STORE_PRODUCER_REFRESH_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_refresh_ms)
            mm[
                FRAME_STORE_PRODUCER_DENOISE_TO_READY_US_OFFSET : FRAME_STORE_PRODUCER_DENOISE_TO_READY_US_OFFSET
                + 8
            ] = _ms_to_us_bytes(timings.producer_denoise_to_ready_ms)
            mm[FRAME_STORE_STATUS_OFFSET] = FRAME_STORE_READY
            if should_flush_realtime_frame_store():
                mm.flush()
    except Exception:
        _mark_store_status(handle.path, FRAME_STORE_ERROR)
        logger.exception(
            "failed to materialize realtime frame store path=%s", handle.path
        )
    finally:
        os.close(fd)


def _rgb24_tensor(sample: torch.Tensor) -> torch.Tensor:
    if sample.dim() == 3:
        sample = sample.unsqueeze(1)
    return (
        (sample * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
    )


def _write_sample_to_store_direct_torch(
    *,
    sample: torch.Tensor,
    mm: mmap.mmap,
    handle: RealtimeFrameStoreHandle,
) -> _FrameStoreWriteTimings:
    payload_array = np.ndarray(
        (handle.payload_size,),
        dtype=np.uint8,
        buffer=mm,
        offset=handle.header_size,
    )
    payload_tensor = torch.from_numpy(payload_array)
    rgb_tensor = _rgb24_tensor(sample).reshape(-1)
    try:
        if int(rgb_tensor.numel()) != handle.payload_size:
            raise RuntimeError(
                "realtime frame store payload size mismatch: "
                f"expected={handle.payload_size}, got={int(rgb_tensor.numel())}"
            )
        copy_started = time.monotonic()
        payload_tensor.copy_(rgb_tensor, non_blocking=False)
        copy_ms = (time.monotonic() - copy_started) * 1000.0
        return _FrameStoreWriteTimings(gpu_copy_ms=copy_ms)
    finally:
        del rgb_tensor
        del payload_tensor
        del payload_array


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _get_frame_store_cuda_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    with _FRAME_STORE_CUDA_RESOURCE_LOCK:
        stream = _FRAME_STORE_CUDA_STREAMS.get(device_index)
        if stream is None:
            with torch.cuda.device(device_index):
                stream = torch.cuda.Stream(device=device_index)
            _FRAME_STORE_CUDA_STREAMS[device_index] = stream
        return stream


def _get_frame_store_pinned_staging(payload_size: int) -> torch.Tensor:
    with _FRAME_STORE_CUDA_RESOURCE_LOCK:
        staging = _FRAME_STORE_PINNED_STAGING.get(payload_size)
        if staging is None:
            staging = torch.empty(
                (payload_size,),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            _FRAME_STORE_PINNED_STAGING[payload_size] = staging
        return staging


def _write_sample_to_store_cuda_side_stream(
    *,
    sample: torch.Tensor,
    mm: mmap.mmap,
    handle: RealtimeFrameStoreHandle,
    ready_event: Any | None = None,
) -> _FrameStoreWriteTimings:
    stream = _get_frame_store_cuda_stream(sample.device)
    staging = _get_frame_store_pinned_staging(handle.payload_size)
    wait_start_event = torch.cuda.Event(enable_timing=True)
    body_start_event = torch.cuda.Event(enable_timing=True)
    body_end_event = torch.cuda.Event(enable_timing=True)
    rgb_tensor = None
    with torch.cuda.stream(stream):
        wait_start_event.record(stream)
        if ready_event is not None:
            stream.wait_event(ready_event)
        body_start_event.record(stream)
        rgb_tensor = _rgb24_tensor(sample).reshape(-1)
        if int(rgb_tensor.numel()) != handle.payload_size:
            raise RuntimeError(
                "realtime frame store payload size mismatch: "
                f"expected={handle.payload_size}, got={int(rgb_tensor.numel())}"
            )
        staging.copy_(rgb_tensor, non_blocking=True)
        body_end_event.record(stream)
    stream.synchronize()

    payload_array = np.ndarray(
        (handle.payload_size,),
        dtype=np.uint8,
        buffer=mm,
        offset=handle.header_size,
    )
    try:
        mmap_started = time.monotonic()
        payload_array[:] = staging.numpy()
        mmap_write_ms = (time.monotonic() - mmap_started) * 1000.0
    finally:
        del payload_array
        if rgb_tensor is not None:
            del rgb_tensor
    return _FrameStoreWriteTimings(
        producer_wait_ms=wait_start_event.elapsed_time(body_start_event),
        gpu_copy_ms=body_start_event.elapsed_time(body_end_event),
        mmap_write_ms=mmap_write_ms,
    )


def _write_sample_to_store_numpy(
    *,
    sample: torch.Tensor,
    mm: mmap.mmap,
    handle: RealtimeFrameStoreHandle,
) -> _FrameStoreWriteTimings:
    convert_started = time.monotonic()
    frames = _tensor_sample_to_rgb24_array(sample)
    convert_ms = (time.monotonic() - convert_started) * 1000.0
    if not frames.flags.c_contiguous:
        frames = np.ascontiguousarray(frames)
    payload = memoryview(frames).cast("B")
    try:
        if len(payload) != handle.payload_size:
            raise RuntimeError(
                "realtime frame store payload size mismatch: "
                f"expected={handle.payload_size}, got={len(payload)}"
            )
        mmap_started = time.monotonic()
        mm[handle.header_size : handle.header_size + handle.payload_size] = payload
        mmap_write_ms = (time.monotonic() - mmap_started) * 1000.0
        return _FrameStoreWriteTimings(
            gpu_copy_ms=convert_ms,
            mmap_write_ms=mmap_write_ms,
        )
    finally:
        payload.release()


def _mark_store_handles_error(handles: list[RealtimeFrameStoreHandle]) -> None:
    for handle in handles:
        _mark_store_status(handle.path, FRAME_STORE_ERROR)


def _materialize_raw_rgb_frame_store_request(
    request: _FrameStoreWriteRequest,
) -> None:
    start = time.monotonic()
    try:
        if len(request.samples) != len(request.handles):
            raise RuntimeError(
                "realtime frame store request mismatch: "
                f"samples={len(request.samples)}, handles={len(request.handles)}"
            )
        ready_events = request.ready_events
        if ready_events is None:
            ready_events = [None] * len(request.samples)
        if len(ready_events) != len(request.samples):
            raise RuntimeError(
                "realtime frame store ready-event mismatch: "
                f"events={len(ready_events)}, samples={len(request.samples)}"
            )
        defer_ms = get_realtime_frame_store_defer_ms()
        if defer_ms > 0:
            time.sleep(defer_ms / 1000.0)
        for sample, handle, ready_event in zip(
            request.samples,
            request.handles,
            ready_events,
        ):
            _write_sample_to_store(
                sample=sample,
                handle=handle,
                ready_event=ready_event,
                producer_timing_events=request.producer_timing_events,
            )
    except Exception:
        _mark_store_handles_error(request.handles)
        logger.exception(
            "failed to materialize realtime frame store request: "
            "request_id=%s chunk_idx=%s handles=%d",
            request.request_id,
            request.chunk_idx,
            len(request.handles),
        )
        return

    logger.info(
        "realtime raw RGB frame store materialized: request_id=%s chunk_idx=%s "
        "handles=%d total_bytes=%d total=%.2fms",
        request.request_id,
        request.chunk_idx,
        len(request.handles),
        sum(handle.payload_size for handle in request.handles),
        (time.monotonic() - start) * 1000.0,
    )


def _frame_store_writer_loop(writer_queue: queue.SimpleQueue) -> None:
    while True:
        try:
            request = writer_queue.get()
            _materialize_raw_rgb_frame_store_request(request)
        except Exception:
            logger.exception("unexpected realtime frame store writer failure")


def _ensure_frame_store_writer_thread() -> tuple[queue.SimpleQueue, threading.Thread]:
    global _FRAME_STORE_WRITER_QUEUE, _FRAME_STORE_WRITER_THREAD

    with _FRAME_STORE_WRITER_LOCK:
        if _FRAME_STORE_WRITER_QUEUE is None:
            _FRAME_STORE_WRITER_QUEUE = queue.SimpleQueue()
        if (
            _FRAME_STORE_WRITER_THREAD is None
            or not _FRAME_STORE_WRITER_THREAD.is_alive()
        ):
            _FRAME_STORE_WRITER_THREAD = threading.Thread(
                target=_frame_store_writer_loop,
                args=(_FRAME_STORE_WRITER_QUEUE,),
                name="realtime-frame-store-writer",
                daemon=True,
            )
            _FRAME_STORE_WRITER_THREAD.start()
        return _FRAME_STORE_WRITER_QUEUE, _FRAME_STORE_WRITER_THREAD


def create_raw_rgb_frame_store_write_request(
    *,
    output: torch.Tensor,
    handles: list[RealtimeFrameStoreHandle],
    request_id: str,
    chunk_idx: int,
    producer_timing_events: dict[str, Any] | None = None,
) -> _FrameStoreWriteRequest:
    samples = _iter_tensor_samples(output)
    ready_events: list[Any] = []
    for sample in samples:
        if sample.is_cuda and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=producer_timing_events is not None)
            event.record(torch.cuda.current_stream(sample.device))
            ready_events.append(event)
        else:
            ready_events.append(None)
    if producer_timing_events is not None and ready_events:
        producer_timing_events = dict(producer_timing_events)
        producer_timing_events["ready"] = ready_events[0]
    return _FrameStoreWriteRequest(
        samples=samples,
        handles=handles,
        request_id=request_id,
        chunk_idx=chunk_idx,
        ready_events=ready_events,
        producer_timing_events=producer_timing_events,
    )


def attach_raw_rgb_frame_store_writer_request(
    output_batch: Any,
    *,
    output: torch.Tensor,
    handles: list[RealtimeFrameStoreHandle],
    request_id: str,
    chunk_idx: int,
    producer_timing_events: dict[str, Any] | None = None,
) -> None:
    setattr(
        output_batch,
        _RAW_RGB_FRAME_STORE_WRITE_REQUEST_ATTR,
        create_raw_rgb_frame_store_write_request(
            output=output,
            handles=handles,
            request_id=request_id,
            chunk_idx=chunk_idx,
            producer_timing_events=producer_timing_events,
        ),
    )


def pop_raw_rgb_frame_store_writer_request(output_batch: Any) -> Any | None:
    request = getattr(output_batch, _RAW_RGB_FRAME_STORE_WRITE_REQUEST_ATTR, None)
    if request is not None:
        delattr(output_batch, _RAW_RGB_FRAME_STORE_WRITE_REQUEST_ATTR)
    return request


def discard_raw_rgb_frame_store_writer_request(request: Any) -> None:
    handles = getattr(request, "handles", None) or []
    for handle in handles:
        _cleanup_store_file(handle.path)


def start_raw_rgb_frame_store_writer_request(
    request: _FrameStoreWriteRequest,
) -> threading.Thread:
    writer_queue, thread = _ensure_frame_store_writer_thread()
    writer_queue.put(request)
    return thread


def start_raw_rgb_frame_store_writer(
    *,
    output: torch.Tensor,
    handles: list[RealtimeFrameStoreHandle],
    request_id: str,
    chunk_idx: int,
    producer_timing_events: dict[str, Any] | None = None,
) -> threading.Thread:
    return start_raw_rgb_frame_store_writer_request(
        create_raw_rgb_frame_store_write_request(
            output=output,
            handles=handles,
            request_id=request_id,
            chunk_idx=chunk_idx,
            producer_timing_events=producer_timing_events,
        )
    )


def _wait_for_store_ready(
    handle: RealtimeFrameStoreHandle,
    *,
    timeout_ms: float,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    deadline = time.monotonic() + timeout_ms / 1000.0 if timeout_ms > 0 else None
    while True:
        with open(handle.path, "rb") as fp:
            with mmap.mmap(
                fp.fileno(), FRAME_STORE_HEADER_SIZE, access=mmap.ACCESS_READ
            ) as mm:
                if mm[: len(FRAME_STORE_MAGIC)] != FRAME_STORE_MAGIC:
                    raise RuntimeError("invalid realtime frame store magic")
                status = mm[FRAME_STORE_STATUS_OFFSET]
                materialize_us = int.from_bytes(
                    mm[
                        FRAME_STORE_MATERIALIZE_US_OFFSET : FRAME_STORE_MATERIALIZE_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_wait_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_WAIT_US_OFFSET : FRAME_STORE_PRODUCER_WAIT_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                gpu_copy_us = int.from_bytes(
                    mm[
                        FRAME_STORE_GPU_COPY_US_OFFSET : FRAME_STORE_GPU_COPY_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                mmap_write_us = int.from_bytes(
                    mm[
                        FRAME_STORE_MMAP_WRITE_US_OFFSET : FRAME_STORE_MMAP_WRITE_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_decode_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_DECODE_US_OFFSET : FRAME_STORE_PRODUCER_DECODE_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_post_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_POST_US_OFFSET : FRAME_STORE_PRODUCER_POST_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_clone_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_CLONE_US_OFFSET : FRAME_STORE_PRODUCER_CLONE_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_total_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_TOTAL_US_OFFSET : FRAME_STORE_PRODUCER_TOTAL_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_denoise_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_DENOISE_US_OFFSET : FRAME_STORE_PRODUCER_DENOISE_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_refresh_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_REFRESH_US_OFFSET : FRAME_STORE_PRODUCER_REFRESH_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
                producer_denoise_to_ready_us = int.from_bytes(
                    mm[
                        FRAME_STORE_PRODUCER_DENOISE_TO_READY_US_OFFSET : FRAME_STORE_PRODUCER_DENOISE_TO_READY_US_OFFSET
                        + 8
                    ],
                    "little",
                    signed=False,
                )
        if status == FRAME_STORE_READY:
            return (
                materialize_us / 1000.0,
                producer_wait_us / 1000.0,
                gpu_copy_us / 1000.0,
                mmap_write_us / 1000.0,
                producer_decode_us / 1000.0,
                producer_post_us / 1000.0,
                producer_clone_us / 1000.0,
                producer_total_us / 1000.0,
                producer_denoise_us / 1000.0,
                producer_refresh_us / 1000.0,
                producer_denoise_to_ready_us / 1000.0,
            )
        if status == FRAME_STORE_ERROR:
            raise RuntimeError("realtime frame store materialization failed")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for realtime frame store after {timeout_ms:.0f}ms"
            )
        time.sleep(0.001)


def load_raw_rgb_frame_store_handles(
    handles: list[RealtimeFrameStoreHandle],
    *,
    timeout_ms: float | None = None,
) -> RealtimeFrameStoreLoadResult:
    if timeout_ms is None:
        timeout_ms = get_realtime_frame_store_timeout_ms()

    frame_batches: list[RealtimeRawFrameBatch] = []
    wait_ms = 0.0
    read_ms = 0.0
    materialize_ms = 0.0
    producer_wait_ms = 0.0
    gpu_copy_ms = 0.0
    mmap_write_ms = 0.0
    producer_decode_ms = 0.0
    producer_post_ms = 0.0
    producer_clone_ms = 0.0
    producer_total_ms = 0.0
    producer_denoise_ms = 0.0
    producer_refresh_ms = 0.0
    producer_denoise_to_ready_ms = 0.0
    for handle in handles:
        wait_start = time.monotonic()
        read_start: float | None = None
        try:
            (
                materialize_delta_ms,
                producer_wait_delta_ms,
                gpu_copy_delta_ms,
                mmap_write_delta_ms,
                producer_decode_delta_ms,
                producer_post_delta_ms,
                producer_clone_delta_ms,
                producer_total_delta_ms,
                producer_denoise_delta_ms,
                producer_refresh_delta_ms,
                producer_denoise_to_ready_delta_ms,
            ) = _wait_for_store_ready(handle, timeout_ms=timeout_ms)
            materialize_ms += materialize_delta_ms
            producer_wait_ms += producer_wait_delta_ms
            gpu_copy_ms += gpu_copy_delta_ms
            mmap_write_ms += mmap_write_delta_ms
            producer_decode_ms += producer_decode_delta_ms
            producer_post_ms += producer_post_delta_ms
            producer_clone_ms += producer_clone_delta_ms
            producer_total_ms += producer_total_delta_ms
            producer_denoise_ms += producer_denoise_delta_ms
            producer_refresh_ms += producer_refresh_delta_ms
            producer_denoise_to_ready_ms += producer_denoise_to_ready_delta_ms
            wait_ms += (time.monotonic() - wait_start) * 1000.0

            read_start = time.monotonic()
            frame_bytes = int(handle.metadata["bytes_per_frame"])
            total_size = handle.header_size + handle.payload_size
            with open(handle.path, "rb") as fp:
                with mmap.mmap(
                    fp.fileno(), total_size, access=mmap.ACCESS_READ
                ) as mm:
                    payload = bytes(
                        mm[
                            handle.header_size : handle.header_size
                            + handle.payload_size
                        ]
                    )
            frame_batches.append(
                RealtimeRawFrameBatch(
                    payload=payload,
                    num_frames=handle.num_frames,
                    bytes_per_frame=frame_bytes,
                )
            )
        finally:
            _cleanup_store_file(handle.path)
        if read_start is not None:
            read_ms += (time.monotonic() - read_start) * 1000.0

    return RealtimeFrameStoreLoadResult(
        frame_batches=frame_batches,
        wait_ms=wait_ms,
        read_ms=read_ms,
        materialize_ms=materialize_ms,
        producer_wait_ms=producer_wait_ms,
        gpu_copy_ms=gpu_copy_ms,
        mmap_write_ms=mmap_write_ms,
        producer_decode_ms=producer_decode_ms,
        producer_post_ms=producer_post_ms,
        producer_clone_ms=producer_clone_ms,
        producer_total_ms=producer_total_ms,
        producer_denoise_ms=producer_denoise_ms,
        producer_refresh_ms=producer_refresh_ms,
        producer_denoise_to_ready_ms=producer_denoise_to_ready_ms,
    )
