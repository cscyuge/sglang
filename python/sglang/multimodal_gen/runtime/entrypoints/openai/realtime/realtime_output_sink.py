# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeArtcOutputConfig,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_codeformer_artc import (
    CodeFormerArtcClient,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_frame_processor import (
    BaseRealtimeFrameProcessor,
    create_realtime_frame_processor,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
    RealtimeRawFrameBatch,
    _frame_shape_from_metadata,
    _raw_rgb_frame_metadata,
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError
from sglang.multimodal_gen.runtime.utils.realtime_frame_store import (
    discard_raw_rgb_frame_store_handles,
    load_raw_rgb_frame_store_handles,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )

logger = logging.getLogger(__name__)
RealtimeOutputTransport = Literal["ws", "artc"]
ARTC_CONTENT_TYPE = "video/artc"
ARTC_DEFAULT_QUEUE_SIZE = 2


def normalize_realtime_output_transport(
    request: RealtimeVideoGenerationsRequest,
) -> RealtimeOutputTransport:
    raw_transport = str(request.output_transport or "ws").strip().lower()
    if raw_transport == "websocket":
        raw_transport = "ws"
    if raw_transport not in {"ws", "artc"}:
        raise RealtimeProtocolError(
            "invalid_output_transport",
            f"unsupported realtime output_transport: {request.output_transport}",
            output_transport=request.output_transport,
        )
    return raw_transport  # type: ignore[return-value]


def _parse_size(size: str | None) -> tuple[int, int] | None:
    if not size or "x" not in str(size):
        return None
    try:
        width_s, height_s = str(size).lower().replace(" ", "").split("x", 1)
        width = int(width_s)
        height = int(height_s)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _coerce_artc_config(
    request: RealtimeVideoGenerationsRequest,
) -> RealtimeArtcOutputConfig:
    if request.artc is None:
        raise RealtimeProtocolError(
            "missing_artc_config",
            "output_transport='artc' requires an artc config",
        )
    return request.artc


def _audio_window_to_float32(samples: Any) -> np.ndarray | None:
    if samples is None:
        return None
    arr = np.asarray(samples)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32, copy=False)


def _result_num_frames(result: "OutputBatch") -> int:
    frame_batches = getattr(result, "raw_frame_batches", None)
    if frame_batches is not None:
        return sum(len(frames) for frames in frame_batches)
    handles = getattr(result, "raw_frame_store_handles", None)
    if handles is not None:
        return sum(int(getattr(handle, "num_frames", 0)) for handle in handles)
    return 0


def _result_raw_bytes(result: "OutputBatch") -> int:
    frame_batches = getattr(result, "raw_frame_batches", None)
    if frame_batches is not None:
        raw_bytes = 0
        for frame_batch in frame_batches:
            if isinstance(frame_batch, RealtimeRawFrameBatch):
                raw_bytes += frame_batch.raw_size
            else:
                raw_bytes += sum(len(frame) for frame in frame_batch)
        return raw_bytes
    handles = getattr(result, "raw_frame_store_handles", None)
    if handles is not None:
        return sum(int(getattr(handle, "payload_size", 0)) for handle in handles)
    return 0


def _result_frame_shape(
    result: "OutputBatch", batch: "Req"
) -> tuple[int, int, int] | None:
    metadata = getattr(result, "raw_frame_metadata", None) or _raw_rgb_frame_metadata(
        batch
    )
    return _frame_shape_from_metadata(metadata)


def _iter_raw_frames(frame_batches: list[Any]):
    for frame_batch in frame_batches:
        if isinstance(frame_batch, RealtimeRawFrameBatch):
            yield from frame_batch.iter_frames()
        else:
            yield from frame_batch


def _frames_from_result(
    result: "OutputBatch",
    batch: "Req",
) -> tuple[np.ndarray | None, dict[str, float]]:
    timings = {
        "frame_store_wait_ms": 0.0,
        "frame_store_read_ms": 0.0,
        "frame_store_materialize_ms": 0.0,
        "frame_store_producer_wait_ms": 0.0,
        "frame_store_gpu_copy_ms": 0.0,
        "frame_store_mmap_write_ms": 0.0,
        "frame_store_producer_decode_ms": 0.0,
        "frame_store_producer_post_ms": 0.0,
        "frame_store_producer_clone_ms": 0.0,
        "frame_store_producer_total_ms": 0.0,
        "frame_store_producer_denoise_ms": 0.0,
        "frame_store_producer_refresh_ms": 0.0,
        "frame_store_producer_denoise_to_ready_ms": 0.0,
    }
    frame_batches = getattr(result, "raw_frame_batches", None)
    if frame_batches is None:
        handles = getattr(result, "raw_frame_store_handles", None)
        if handles is not None:
            loaded = load_raw_rgb_frame_store_handles(handles)
            frame_batches = loaded.frame_batches
            timings.update(
                {
                    "frame_store_wait_ms": loaded.wait_ms,
                    "frame_store_read_ms": loaded.read_ms,
                    "frame_store_materialize_ms": loaded.materialize_ms,
                    "frame_store_producer_wait_ms": loaded.producer_wait_ms,
                    "frame_store_gpu_copy_ms": loaded.gpu_copy_ms,
                    "frame_store_mmap_write_ms": loaded.mmap_write_ms,
                    "frame_store_producer_decode_ms": loaded.producer_decode_ms,
                    "frame_store_producer_post_ms": loaded.producer_post_ms,
                    "frame_store_producer_clone_ms": loaded.producer_clone_ms,
                    "frame_store_producer_total_ms": loaded.producer_total_ms,
                    "frame_store_producer_denoise_ms": loaded.producer_denoise_ms,
                    "frame_store_producer_refresh_ms": loaded.producer_refresh_ms,
                    "frame_store_producer_denoise_to_ready_ms": (
                        loaded.producer_denoise_to_ready_ms
                    ),
                }
            )
    if not frame_batches:
        return None, timings

    shape = _result_frame_shape(result, batch)
    if shape is None:
        return None, timings
    height, width, channels = shape
    frame_bytes = width * height * channels
    num_frames = sum(len(frames) for frames in frame_batches)
    frames_np = np.empty((num_frames, height, width, channels), dtype=np.uint8)
    frame_idx = 0
    for raw in _iter_raw_frames(frame_batches):
        if not raw or len(raw) != frame_bytes:
            continue
        frames_np[frame_idx] = np.frombuffer(raw, dtype=np.uint8).reshape(
            height,
            width,
            channels,
        )
        frame_idx += 1
    if frame_idx <= 0:
        return None, timings
    return frames_np[:frame_idx], timings


@dataclass(slots=True)
class _ArtcOutputItem:
    session_id: str
    result: Any
    batch: Any
    enqueued_at: float
    queue_size: int


class BaseRealtimeOutputSink:
    transport: RealtimeOutputTransport = "ws"

    async def send(
        self,
        session: "GenerateSession",
        result: "OutputBatch",
        batch: "Req",
    ) -> RealtimeFrameSendStats:
        raise NotImplementedError

    def build_init_ack(self) -> dict[str, Any] | None:
        return {"output_transport": self.transport}

    def raise_if_failed(self) -> None:
        return None

    async def wait_failed(self) -> None:
        await asyncio.Future()

    async def close(self) -> None:
        return None

    async def cancel(self) -> None:
        await self.close()


class WebSocketRealtimeOutputSink(BaseRealtimeOutputSink):
    transport = "ws"

    def __init__(self, ws: "WebSocket") -> None:
        self.ws = ws

    async def send(
        self,
        session: "GenerateSession",
        result: "OutputBatch",
        batch: "Req",
    ) -> RealtimeFrameSendStats:
        if session.adapter is None:
            raise ValueError("realtime adapter is not initialized")
        return await session.adapter.send_output(self.ws, session, result, batch)


class ArtcRealtimeOutputSink(BaseRealtimeOutputSink):
    transport = "artc"

    def __init__(
        self,
        *,
        session_id: str,
        config: RealtimeArtcOutputConfig,
        width: int,
        height: int,
        fps: int,
        frame_processor: BaseRealtimeFrameProcessor,
    ) -> None:
        from sglang.multimodal_gen.runtime.utils.artc_pusher import ArtcPusher

        self.session_id = session_id
        self.config = config
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps) or 25
        self.frame_processor = frame_processor
        self.input_width = int(frame_processor.input_width)
        self.input_height = int(frame_processor.input_height)
        self.max_queue_size = int(config.queue_size or ARTC_DEFAULT_QUEUE_SIZE)
        self._queue: asyncio.Queue[_ArtcOutputItem] = asyncio.Queue(
            maxsize=self.max_queue_size
        )
        self._failed: BaseException | None = None
        self._failure_event = asyncio.Event()
        self._closed = False
        self._stopped = False
        self._pushed_chunks = 0
        self._pusher = ArtcPusher(
            artc_token=config.token,
            artc_channel=config.channel,
            artc_userid=config.userid or "sglang",
            width=self.width,
            height=self.height,
            fps=self.fps,
            session_id=session_id,
        )
        self._pusher.start_async()
        self._worker_task = asyncio.create_task(
            self._run(),
            name=f"realtime-artc-output-{session_id}",
        )

    @classmethod
    def from_request(
        cls,
        session: "GenerateSession",
        request: RealtimeVideoGenerationsRequest,
    ) -> "ArtcRealtimeOutputSink":
        config = _coerce_artc_config(request)
        size = _parse_size(request.size)
        width = size[0] if size is not None else request.width
        height = size[1] if size is not None else request.height
        if not width or not height:
            raise RealtimeProtocolError(
                "missing_artc_size",
                "output_transport='artc' requires request size or width/height",
                size=request.size,
                width=request.width,
                height=request.height,
            )
        if request.enable_upscaling:
            upscaling_scale = int(request.upscaling_scale or 1)
            width *= upscaling_scale
            height *= upscaling_scale
        fps = int(request.fps or 25)
        frame_processor = create_realtime_frame_processor(
            request,
            input_width=int(width),
            input_height=int(height),
            fps=fps,
        )
        return cls(
            session_id=session.id,
            config=config,
            width=int(frame_processor.output_width),
            height=int(frame_processor.output_height),
            fps=fps,
            frame_processor=frame_processor,
        )

    def build_init_ack(self) -> dict[str, Any] | None:
        ack = {
            "output_transport": "artc",
            "artc": {
                "channel": self.config.channel,
                "userid": self.config.userid or "sglang",
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "queue_size": self.max_queue_size,
            },
        }
        processor_ack = self.frame_processor.build_init_ack()
        if processor_ack is not None:
            ack["realtime_postprocess"] = processor_ack
        return ack

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed
        if getattr(self._pusher, "failed", False):
            raise RealtimeProtocolError(
                "artc_output_failed",
                "ARTC output sink failed",
                channel=self.config.channel,
            )

    async def wait_failed(self) -> None:
        await self._failure_event.wait()
        self.raise_if_failed()

    async def send(
        self,
        session: "GenerateSession",
        result: "OutputBatch",
        batch: "Req",
    ) -> RealtimeFrameSendStats:
        del session
        self.raise_if_failed()
        if self._closed:
            raise RealtimeProtocolError(
                "artc_output_closed",
                "ARTC output sink is closed",
                channel=self.config.channel,
            )
        started = time.perf_counter()
        item = _ArtcOutputItem(
            session_id=self.session_id,
            result=result,
            batch=batch,
            enqueued_at=started,
            queue_size=self._queue.qsize() + 1,
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self._discard_result_handles(result)
            raise RealtimeProtocolError(
                "artc_output_backpressure",
                "ARTC output queue is full",
                channel=self.config.channel,
                chunk_index=getattr(batch, "block_idx", None),
                queue_size=self._queue.qsize(),
                max_queue_size=self.max_queue_size,
            ) from exc

        stats = empty_frame_send_stats(ARTC_CONTENT_TYPE)
        stats["num_frames"] = _result_num_frames(result)
        stats["num_batches"] = 1 if stats["num_frames"] > 0 else 0
        stats["frame_shape"] = _result_frame_shape(result, batch)
        stats["raw_bytes"] = _result_raw_bytes(result)
        stats["artc_enqueue_wait_ms"] = (time.perf_counter() - started) * 1000.0
        stats["artc_queue_size"] = item.queue_size
        return stats

    async def close(self) -> None:
        self._closed = True
        if self._failed is None:
            await self._queue.join()
        if not self._worker_task.done():
            self._worker_task.cancel()
        await self._await_worker()
        await self._stop_pusher()
        self.raise_if_failed()

    async def cancel(self) -> None:
        self._closed = True
        self._discard_pending_items()
        if not self._worker_task.done():
            self._worker_task.cancel()
        await self._await_worker()
        await self._stop_pusher()

    async def _await_worker(self) -> None:
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass

    async def _stop_pusher(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await asyncio.to_thread(self._pusher.stop, 2.0)

    def _discard_pending_items(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._discard_result_handles(item.result)
            self._queue.task_done()

    async def _run(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    await asyncio.to_thread(self._push_item, item)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed = exc
            self._failure_event.set()
            logger.exception(
                "ARTC realtime output sink failed, session_id=%s channel=%s",
                self.session_id,
                self.config.channel,
            )
            self._discard_pending_items()

    def _push_item(self, item: _ArtcOutputItem) -> None:
        started = time.perf_counter()
        frames_np, frame_store_timings = _frames_from_result(item.result, item.batch)
        if frames_np is None:
            return
        frame_h, frame_w = int(frames_np.shape[1]), int(frames_np.shape[2])
        if (frame_w, frame_h) != (self.input_width, self.input_height):
            raise RealtimeProtocolError(
                "artc_frame_input_size_mismatch",
                "ARTC frame input size does not match configured processor input size",
                channel=self.config.channel,
                frame_width=frame_w,
                frame_height=frame_h,
                configured_width=self.input_width,
                configured_height=self.input_height,
            )
        processor_result = self.frame_processor.process(
            frames_np,
            session_id=item.session_id,
            chunk_idx=getattr(item.batch, "block_idx", None),
        )
        frames_np = processor_result.frames
        processor_stats = processor_result.stats
        frame_h, frame_w = int(frames_np.shape[1]), int(frames_np.shape[2])
        if (frame_w, frame_h) != (self.width, self.height):
            raise RealtimeProtocolError(
                "artc_frame_size_mismatch",
                "ARTC frame size does not match configured output size",
                channel=self.config.channel,
                frame_width=frame_w,
                frame_height=frame_h,
                configured_width=self.width,
                configured_height=self.height,
            )
        audio = _audio_window_to_float32(
            getattr(item.batch, "extra", {}).get("wan_s2v_audio_window")
        )
        audio_meta = getattr(item.batch, "extra", {}).get("wan_s2v_audio_window_meta")
        push_audio_meta = dict(audio_meta) if isinstance(audio_meta, dict) else {}
        if processor_stats:
            push_audio_meta.update(processor_stats)
        self._pusher.push_chunk(
            frames_np,
            audio_16k=audio,
            chunk_idx=getattr(item.batch, "block_idx", None),
            audio_chunk_idx=getattr(item.batch, "block_idx", None),
            session_id=item.session_id,
            audio_loaded=audio is not None,
            audio_chunk_meta=push_audio_meta or None,
            is_filler=False,
        )
        self._pushed_chunks += 1
        logger.info(
            "ARTC realtime chunk enqueued: session_id=%s channel=%s chunk_idx=%s "
            "frames=%d queue_delay=%.2fms push=%.2fms frame_store_wait=%.2fms "
            "frame_processor=%s processor_status=%s processor_total=%.2fms "
            "processor_http=%.2fms processor_passthrough=%s",
            item.session_id,
            self.config.channel,
            getattr(item.batch, "block_idx", None),
            int(frames_np.shape[0]),
            (started - item.enqueued_at) * 1000.0,
            (time.perf_counter() - started) * 1000.0,
            frame_store_timings.get("frame_store_wait_ms", 0.0),
            processor_stats.get("frame_processor_name", "none"),
            processor_stats.get("frame_processor_status", "noop"),
            float(processor_stats.get("frame_processor_total_ms", 0.0)),
            float(processor_stats.get("frame_processor_http_ms", 0.0)),
            processor_stats.get("frame_processor_passthrough", False),
        )

    @staticmethod
    def _discard_result_handles(result: "OutputBatch") -> None:
        handles = getattr(result, "raw_frame_store_handles", None)
        if handles is not None:
            discard_raw_rgb_frame_store_handles(handles)


class RemoteCodeFormerArtcOutputSink(BaseRealtimeOutputSink):
    """Send low-resolution chunks to CodeFormer, which owns the ARTC publisher."""

    transport = "artc"

    def __init__(
        self,
        *,
        session_id: str,
        config: RealtimeArtcOutputConfig,
        endpoint: str,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        fps: int,
        timeout_ms: float,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.endpoint = endpoint
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.width = int(output_width)
        self.height = int(output_height)
        self.fps = int(fps) or 25
        self.max_queue_size = int(config.queue_size or ARTC_DEFAULT_QUEUE_SIZE)
        self._queue: asyncio.Queue[_ArtcOutputItem] = asyncio.Queue(
            maxsize=self.max_queue_size
        )
        self._failed: BaseException | None = None
        self._failure_event = asyncio.Event()
        self._closed = False
        self._remote_closed = False
        self._pushed_chunks = 0
        self._worker_task: asyncio.Task | None = None
        self._client = CodeFormerArtcClient(
            endpoint=endpoint,
            session_id=session_id,
            timeout_ms=timeout_ms,
        )

    def _session_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video": {
                "input_width": self.input_width,
                "input_height": self.input_height,
                "output_width": self.width,
                "output_height": self.height,
                "fps": self.fps,
                "pix_fmt": "rgb24",
            },
            "artc": {
                "token": self.config.token,
                "channel": self.config.channel,
                "userid": self.config.userid or "codeformer",
                "queue_size": self.max_queue_size,
            },
        }

    def _start_worker(self) -> None:
        self._worker_task = asyncio.create_task(
            self._run(),
            name=f"realtime-codeformer-artc-output-{self.session_id}",
        )

    def _start_remote_sync(self) -> None:
        try:
            self._remote_status = self._client.create(self._session_payload())
        except Exception as exc:
            try:
                self._client.close(drain=False)
            except Exception:
                pass
            raise RealtimeProtocolError(
                "codeformer_artc_session_create_failed",
                f"CodeFormer ARTC session creation failed: {exc}",
                endpoint=self.endpoint,
                channel=self.config.channel,
            ) from exc
        self._start_worker()

    async def _start_remote_async(self) -> None:
        try:
            self._remote_status = await asyncio.to_thread(
                self._client.create,
                self._session_payload(),
            )
        except Exception as exc:
            try:
                await asyncio.to_thread(self._client.close, drain=False)
            except Exception:
                pass
            raise RealtimeProtocolError(
                "codeformer_artc_session_create_failed",
                f"CodeFormer ARTC session creation failed: {exc}",
                endpoint=self.endpoint,
                channel=self.config.channel,
            ) from exc
        self._start_worker()

    @classmethod
    def _from_request_unstarted(
        cls,
        session: "GenerateSession",
        request: RealtimeVideoGenerationsRequest,
    ) -> "RemoteCodeFormerArtcOutputSink":
        artc_config = _coerce_artc_config(request)
        postprocess = request.realtime_postprocess
        if postprocess is None or postprocess.type != "codeformer":
            raise RealtimeProtocolError(
                "missing_remote_artc_postprocess",
                "remote ARTC delivery requires realtime_postprocess type='codeformer'",
            )
        endpoint = str(
            postprocess.endpoint
            or os.environ.get("SGLANG_REALTIME_CODEFORMER_ARTC_ENDPOINT", "")
        ).strip()
        if not endpoint:
            raise RealtimeProtocolError(
                "missing_codeformer_artc_endpoint",
                "remote ARTC delivery requires a CodeFormer session endpoint",
            )
        size = _parse_size(request.size)
        width = size[0] if size is not None else request.width
        height = size[1] if size is not None else request.height
        if not width or not height:
            raise RealtimeProtocolError(
                "missing_artc_size",
                "output_transport='artc' requires request size or width/height",
                size=request.size,
                width=request.width,
                height=request.height,
            )
        if request.enable_upscaling:
            scale = int(request.upscaling_scale or 1)
            width *= scale
            height *= scale
        postprocess_scale = int(postprocess.scale or 2)
        sink = cls(
            session_id=session.id,
            config=artc_config,
            endpoint=endpoint,
            input_width=int(width),
            input_height=int(height),
            output_width=int(width) * postprocess_scale,
            output_height=int(height) * postprocess_scale,
            fps=int(request.fps or 25),
            timeout_ms=float(postprocess.timeout_ms or 0.0),
        )
        return sink

    @classmethod
    def from_request(
        cls,
        session: "GenerateSession",
        request: RealtimeVideoGenerationsRequest,
    ) -> "RemoteCodeFormerArtcOutputSink":
        sink = cls._from_request_unstarted(session, request)
        sink._start_remote_sync()
        return sink

    @classmethod
    async def from_request_async(
        cls,
        session: "GenerateSession",
        request: RealtimeVideoGenerationsRequest,
    ) -> "RemoteCodeFormerArtcOutputSink":
        sink = cls._from_request_unstarted(session, request)
        await sink._start_remote_async()
        return sink

    def build_init_ack(self) -> dict[str, Any] | None:
        return {
            "output_transport": "artc",
            "artc": {
                "channel": self.config.channel,
                "userid": self.config.userid or "codeformer",
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "queue_size": self.max_queue_size,
                "publisher": "codeformer",
            },
            "realtime_postprocess": {
                "type": "codeformer",
                "delivery": "artc",
                "endpoint": self.endpoint,
                "input_width": self.input_width,
                "input_height": self.input_height,
                "output_width": self.width,
                "output_height": self.height,
            },
        }

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed

    async def wait_failed(self) -> None:
        await self._failure_event.wait()
        self.raise_if_failed()

    async def send(
        self,
        session: "GenerateSession",
        result: "OutputBatch",
        batch: "Req",
    ) -> RealtimeFrameSendStats:
        del session
        self.raise_if_failed()
        if self._closed:
            raise RealtimeProtocolError(
                "codeformer_artc_output_closed",
                "CodeFormer ARTC output sink is closed",
                channel=self.config.channel,
            )
        started = time.perf_counter()
        item = _ArtcOutputItem(
            session_id=self.session_id,
            result=result,
            batch=batch,
            enqueued_at=started,
            queue_size=self._queue.qsize() + 1,
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self._discard_result_handles(result)
            raise RealtimeProtocolError(
                "codeformer_artc_output_backpressure",
                "CodeFormer ARTC output queue is full",
                channel=self.config.channel,
                chunk_index=getattr(batch, "block_idx", None),
                queue_size=self._queue.qsize(),
                max_queue_size=self.max_queue_size,
            ) from exc
        stats = empty_frame_send_stats(ARTC_CONTENT_TYPE)
        stats["num_frames"] = _result_num_frames(result)
        stats["num_batches"] = 1 if stats["num_frames"] > 0 else 0
        stats["frame_shape"] = _result_frame_shape(result, batch)
        stats["raw_bytes"] = _result_raw_bytes(result)
        stats["artc_enqueue_wait_ms"] = (time.perf_counter() - started) * 1000.0
        stats["artc_queue_size"] = item.queue_size
        return stats

    async def close(self) -> None:
        self._closed = True
        if self._failed is None:
            await self._queue.join()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        await self._await_worker()
        await self._close_remote(drain=True)
        self.raise_if_failed()

    async def cancel(self) -> None:
        self._closed = True
        self._discard_pending_items()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        await self._await_worker()
        try:
            await self._close_remote(drain=False)
        except Exception:
            logger.exception(
                "CodeFormer ARTC cancel failed, session_id=%s", self.session_id
            )

    async def _await_worker(self) -> None:
        if self._worker_task is None:
            return
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass

    async def _close_remote(self, *, drain: bool) -> None:
        if self._remote_closed:
            return
        self._remote_closed = True
        await asyncio.to_thread(self._client.close, drain=drain)

    def _discard_pending_items(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._discard_result_handles(item.result)
            self._queue.task_done()

    async def _run(self) -> None:
        poll_interval = max(
            0.1,
            float(os.environ.get("SGLANG_REALTIME_CODEFORMER_STATUS_POLL_S", "1")),
        )
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=poll_interval
                    )
                except asyncio.TimeoutError:
                    status = await asyncio.to_thread(self._client.status)
                    self._remote_status = status
                    if status.get("state") == "failed":
                        raise RealtimeProtocolError(
                            "codeformer_artc_output_failed",
                            "CodeFormer ARTC publisher failed",
                            channel=self.config.channel,
                            remote_error=status.get("last_error"),
                        )
                    continue
                try:
                    await asyncio.to_thread(self._push_item, item)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed = exc
            self._failure_event.set()
            logger.exception(
                "CodeFormer ARTC output sink failed, session_id=%s channel=%s",
                self.session_id,
                self.config.channel,
            )
            self._discard_pending_items()

    def _push_item(self, item: _ArtcOutputItem) -> None:
        started = time.perf_counter()
        frames_np, frame_store_timings = _frames_from_result(item.result, item.batch)
        if frames_np is None:
            return
        frame_h, frame_w = int(frames_np.shape[1]), int(frames_np.shape[2])
        if (frame_w, frame_h) != (self.input_width, self.input_height):
            raise RealtimeProtocolError(
                "codeformer_artc_frame_input_size_mismatch",
                "CodeFormer ARTC input size does not match configured size",
                frame_width=frame_w,
                frame_height=frame_h,
                configured_width=self.input_width,
                configured_height=self.input_height,
            )
        audio = _audio_window_to_float32(
            getattr(item.batch, "extra", {}).get("wan_s2v_audio_window")
        )
        audio_meta = getattr(item.batch, "extra", {}).get("wan_s2v_audio_window_meta")
        response = self._client.send_chunk(
            frames=frames_np,
            audio=audio,
            chunk_idx=int(getattr(item.batch, "block_idx", 0)),
            width=self.input_width,
            height=self.input_height,
            fps=self.fps,
            audio_meta=audio_meta if isinstance(audio_meta, dict) else {},
        )
        if response.get("status") != "enqueued":
            raise RealtimeProtocolError(
                "codeformer_artc_chunk_failed",
                "CodeFormer did not enqueue the ARTC chunk",
                response_status=response.get("status"),
            )
        self._remote_status = response
        self._pushed_chunks += 1
        remote_timing = response.get("timing") or {}
        logger.info(
            "CodeFormer ARTC chunk enqueued: session_id=%s channel=%s chunk_idx=%s "
            "frames=%d queue_delay=%.2fms request=%.2fms remote_total=%.2fms "
            "frame_store_wait=%.2fms",
            item.session_id,
            self.config.channel,
            getattr(item.batch, "block_idx", None),
            int(frames_np.shape[0]),
            (started - item.enqueued_at) * 1000.0,
            (time.perf_counter() - started) * 1000.0,
            float(remote_timing.get("end_to_end_ms", 0.0)),
            frame_store_timings.get("frame_store_wait_ms", 0.0),
        )

    @staticmethod
    def _discard_result_handles(result: "OutputBatch") -> None:
        handles = getattr(result, "raw_frame_store_handles", None)
        if handles is not None:
            discard_raw_rgb_frame_store_handles(handles)


def create_realtime_output_sink(
    ws: "WebSocket",
    session: "GenerateSession",
) -> BaseRealtimeOutputSink:
    if session.request is None:
        raise ValueError("realtime request is not initialized")
    transport = normalize_realtime_output_transport(session.request)
    if transport == "ws":
        return WebSocketRealtimeOutputSink(ws)
    postprocess = session.request.realtime_postprocess
    if (
        postprocess is not None
        and postprocess.type == "codeformer"
        and postprocess.delivery == "artc"
    ):
        return RemoteCodeFormerArtcOutputSink.from_request(session, session.request)
    return ArtcRealtimeOutputSink.from_request(session, session.request)


async def create_realtime_output_sink_async(
    ws: "WebSocket",
    session: "GenerateSession",
) -> BaseRealtimeOutputSink:
    if session.request is None:
        raise ValueError("realtime request is not initialized")
    postprocess = session.request.realtime_postprocess
    if (
        normalize_realtime_output_transport(session.request) == "artc"
        and postprocess is not None
        and postprocess.type == "codeformer"
        and postprocess.delivery == "artc"
    ):
        return await RemoteCodeFormerArtcOutputSink.from_request_async(
            session, session.request
        )
    return create_realtime_output_sink(ws, session)
