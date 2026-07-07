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
FRAME_STORE_HEADER_SIZE = 64
FRAME_STORE_STATUS_OFFSET = 8
FRAME_STORE_SIZE_OFFSET = 16
FRAME_STORE_PENDING = 0
FRAME_STORE_READY = 1
FRAME_STORE_ERROR = 2
FRAME_STORE_DIR_ENV = "SGLANG_REALTIME_FRAME_STORE_DIR"
FRAME_STORE_TIMEOUT_ENV = "SGLANG_REALTIME_FRAME_STORE_TIMEOUT_MS"
FRAME_STORE_DEFAULT_TIMEOUT_MS = 30000.0
FRAME_STORE_WRITER_DEFER_ENV = "SGLANG_REALTIME_FRAME_STORE_DEFER_MS"
FRAME_STORE_DEFAULT_WRITER_DEFER_MS = 0.0
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


@dataclass
class _FrameStoreWriteRequest:
    samples: list[torch.Tensor]
    handles: list[RealtimeFrameStoreHandle]
    request_id: str
    chunk_idx: int


_FRAME_STORE_WRITER_QUEUE: queue.SimpleQueue | None = None
_FRAME_STORE_WRITER_THREAD: threading.Thread | None = None
_FRAME_STORE_WRITER_LOCK = threading.Lock()


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
) -> None:
    fd = os.open(handle.path, os.O_RDWR)
    try:
        total_size = handle.header_size + handle.payload_size
        with mmap.mmap(fd, total_size) as mm:
            if mm[: len(FRAME_STORE_MAGIC)] != FRAME_STORE_MAGIC:
                raise RuntimeError("invalid realtime frame store magic")
            frames = _tensor_sample_to_rgb24_array(sample)
            if not frames.flags.c_contiguous:
                frames = np.ascontiguousarray(frames)
            payload = memoryview(frames).cast("B")
            if len(payload) != handle.payload_size:
                raise RuntimeError(
                    "realtime frame store payload size mismatch: "
                    f"expected={handle.payload_size}, got={len(payload)}"
                )
            mm[handle.header_size : handle.header_size + handle.payload_size] = payload
            mm[FRAME_STORE_STATUS_OFFSET] = FRAME_STORE_READY
            mm.flush()
    except Exception:
        _mark_store_status(handle.path, FRAME_STORE_ERROR)
        logger.exception(
            "failed to materialize realtime frame store path=%s", handle.path
        )
    finally:
        os.close(fd)


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
        defer_ms = get_realtime_frame_store_defer_ms()
        if defer_ms > 0:
            time.sleep(defer_ms / 1000.0)
        for sample, handle in zip(request.samples, request.handles):
            _write_sample_to_store(sample=sample, handle=handle)
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
) -> _FrameStoreWriteRequest:
    return _FrameStoreWriteRequest(
        samples=_iter_tensor_samples(output),
        handles=handles,
        request_id=request_id,
        chunk_idx=chunk_idx,
    )


def attach_raw_rgb_frame_store_writer_request(
    output_batch: Any,
    *,
    output: torch.Tensor,
    handles: list[RealtimeFrameStoreHandle],
    request_id: str,
    chunk_idx: int,
) -> None:
    setattr(
        output_batch,
        _RAW_RGB_FRAME_STORE_WRITE_REQUEST_ATTR,
        create_raw_rgb_frame_store_write_request(
            output=output,
            handles=handles,
            request_id=request_id,
            chunk_idx=chunk_idx,
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
) -> threading.Thread:
    return start_raw_rgb_frame_store_writer_request(
        create_raw_rgb_frame_store_write_request(
            output=output,
            handles=handles,
            request_id=request_id,
            chunk_idx=chunk_idx,
        )
    )


def _wait_for_store_ready(
    handle: RealtimeFrameStoreHandle,
    *,
    timeout_ms: float,
) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0 if timeout_ms > 0 else None
    while True:
        with open(handle.path, "rb") as fp:
            with mmap.mmap(
                fp.fileno(), FRAME_STORE_HEADER_SIZE, access=mmap.ACCESS_READ
            ) as mm:
                if mm[: len(FRAME_STORE_MAGIC)] != FRAME_STORE_MAGIC:
                    raise RuntimeError("invalid realtime frame store magic")
                status = mm[FRAME_STORE_STATUS_OFFSET]
        if status == FRAME_STORE_READY:
            return
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
    for handle in handles:
        wait_start = time.monotonic()
        read_start: float | None = None
        try:
            _wait_for_store_ready(handle, timeout_ms=timeout_ms)
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
    )
