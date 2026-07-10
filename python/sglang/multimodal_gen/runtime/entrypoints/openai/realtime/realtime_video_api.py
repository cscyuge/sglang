# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec.msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeChunkContext,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
    send_realtime_ws_bytes,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_sink import (
    BaseRealtimeOutputSink,
    WebSocketRealtimeOutputSink,
    create_realtime_output_sink,
    create_realtime_output_sink_async,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSION_WAIT_SECONDS = 1.0
_ACTIVE_SESSION_WAIT_INTERVAL_SECONDS = 0.1
REALTIME_OUTPUT_QUEUE_SIZE_ENV = "SGLANG_REALTIME_OUTPUT_QUEUE_SIZE"
REALTIME_OUTPUT_QUEUE_SIZE_DEFAULT = 2
REALTIME_OUTPUT_ENQUEUE_TIMEOUT_ENV = "SGLANG_REALTIME_OUTPUT_ENQUEUE_TIMEOUT_MS"
REALTIME_OUTPUT_ENQUEUE_TIMEOUT_DEFAULT_MS = 10000.0


def get_realtime_output_queue_size() -> int:
    raw_env = os.environ.get(REALTIME_OUTPUT_QUEUE_SIZE_ENV)
    if raw_env is not None and raw_env != "":
        raw_value = raw_env
    else:
        try:
            raw_value = getattr(
                getattr(get_global_server_args(), "pipeline_config", None),
                "realtime_output_queue_size",
                REALTIME_OUTPUT_QUEUE_SIZE_DEFAULT,
            )
        except Exception:
            raw_value = REALTIME_OUTPUT_QUEUE_SIZE_DEFAULT
    return max(1, int(raw_value))


def get_realtime_output_enqueue_timeout_ms() -> float:
    raw_env = os.environ.get(REALTIME_OUTPUT_ENQUEUE_TIMEOUT_ENV)
    if raw_env is not None and raw_env != "":
        raw_value = raw_env
    else:
        try:
            raw_value = getattr(
                getattr(get_global_server_args(), "pipeline_config", None),
                "realtime_output_enqueue_timeout_ms",
                REALTIME_OUTPUT_ENQUEUE_TIMEOUT_DEFAULT_MS,
            )
        except Exception:
            raw_value = REALTIME_OUTPUT_ENQUEUE_TIMEOUT_DEFAULT_MS
    return max(0.0, float(raw_value))


def _realtime_error_code(error: Exception, default: str) -> str:
    return str(getattr(error, "code", None) or default)


def _realtime_error_message(error: Exception, fallback: str) -> str:
    return str(error).splitlines()[0] or fallback


def _realtime_error_details(
    error: Exception,
    **extra: Any,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    source = getattr(error, "details", None)
    if isinstance(source, dict):
        details.update(source)
    details.update({key: value for key, value in extra.items() if value is not None})
    return details


def _transport_ms(value: float) -> int:
    return max(0, int(value + 0.5))


def _session_elapsed_ms(
    session: GenerateSession,
    *,
    at_perf: float | None = None,
) -> float:
    started = getattr(session, "created_at_perf", None)
    if started is None:
        return 0.0
    now = time.perf_counter() if at_perf is None else at_perf
    return max(0.0, (now - started) * 1000.0)


def _merge_realtime_debug_payload(
    payload: dict,
    extra: dict | None,
) -> None:
    if not extra:
        return
    for key, value in extra.items():
        if key not in payload:
            payload[key] = value


def _realtime_worker_timings(result) -> dict[str, Any] | None:
    timings = getattr(result, "realtime_timings", None)
    if not isinstance(timings, dict):
        return None

    clean_timings: dict[str, Any] = {}
    for key, value in timings.items():
        clean_key = str(key)
        if isinstance(value, bool):
            clean_timings[clean_key] = value
            continue
        if isinstance(value, str):
            clean_timings[clean_key] = value
            continue
        try:
            clean_timings[clean_key] = _transport_ms(float(value))
        except (TypeError, ValueError):
            continue
    return clean_timings or None


async def _wait_for_active_session_slot(
    *,
    timeout_s: float = _ACTIVE_SESSION_WAIT_SECONDS,
    interval_s: float = _ACTIVE_SESSION_WAIT_INTERVAL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while _ACTIVE_SESSION_IDS and time.monotonic() < deadline:
        await asyncio.sleep(interval_s)
    return not _ACTIVE_SESSION_IDS


def _log_realtime_chunk_timing(
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    logger.info(
        "realtime chunk timing: session_id=%s request_id=%s "
        "chunk_idx=%s event_id=%s condition_kinds=%s "
        "request_prepare=%.2fms scheduler_forward=%.2fms "
        "output_enqueue_wait=%.2fms output_queue_delay=%.2fms "
        "output_queue_size=%d "
        "output_pace=%.2fms "
        "header_pack=%.2fms "
        "header_write=%.2fms frame_store_wait=%.2fms "
        "frame_store_read=%.2fms frame_store_materialize=%.2fms "
        "frame_store_producer_wait=%.2fms frame_store_gpu_copy=%.2fms "
        "frame_store_mmap_write=%.2fms "
        "frame_store_producer_denoise=%.2fms "
        "frame_store_producer_refresh=%.2fms "
        "frame_store_producer_decode=%.2fms frame_store_producer_post=%.2fms "
        "frame_store_producer_clone=%.2fms frame_store_producer_total=%.2fms "
        "frame_store_producer_denoise_to_ready=%.2fms "
        "raw_payload_build=%.2fms raw_write=%.2fms "
        "ws_write=%.2fms artc_enqueue_wait=%.2fms "
        "artc_queue_delay=%.2fms artc_queue_size=%d artc_push=%.2fms "
        "chunk_total=%.2fms batches=%d frames=%d "
        "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
        session.id,
        chunk.request_id,
        batch.block_idx,
        getattr(batch, "realtime_event_id", None),
        sorted(batch.condition_inputs) if batch.condition_inputs else [],
        request_prepare_ms,
        scheduler_forward_ms,
        send_stats.get("output_enqueue_wait_ms", 0.0),
        send_stats.get("output_queue_delay_ms", 0.0),
        send_stats.get("output_queue_size", 0),
        send_stats["pace_wait_ms"],
        send_stats["header_pack_ms"],
        send_stats["header_write_ms"],
        send_stats.get("frame_store_wait_ms", 0.0),
        send_stats.get("frame_store_read_ms", 0.0),
        send_stats.get("frame_store_materialize_ms", 0.0),
        send_stats.get("frame_store_producer_wait_ms", 0.0),
        send_stats.get("frame_store_gpu_copy_ms", 0.0),
        send_stats.get("frame_store_mmap_write_ms", 0.0),
        send_stats.get("frame_store_producer_denoise_ms", 0.0),
        send_stats.get("frame_store_producer_refresh_ms", 0.0),
        send_stats.get("frame_store_producer_decode_ms", 0.0),
        send_stats.get("frame_store_producer_post_ms", 0.0),
        send_stats.get("frame_store_producer_clone_ms", 0.0),
        send_stats.get("frame_store_producer_total_ms", 0.0),
        send_stats.get("frame_store_producer_denoise_to_ready_ms", 0.0),
        send_stats["raw_payload_build_ms"],
        send_stats["raw_write_ms"],
        send_stats["ws_write_ms"],
        send_stats.get("artc_enqueue_wait_ms", 0.0),
        send_stats.get("artc_queue_delay_ms", 0.0),
        send_stats.get("artc_queue_size", 0),
        send_stats.get("artc_push_ms", 0.0),
        chunk_total_ms,
        send_stats["num_batches"],
        send_stats["num_frames"],
        send_stats["frame_shape"],
        send_stats["raw_bytes"],
        send_stats["ws_payload_bytes"],
        send_stats["content_type"],
    )


async def _send_realtime_chunk_stats(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    result,
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
    chunk_started: float,
) -> None:
    payload = {
        "type": "chunk_stats",
        "session_id": session.id,
        "request_id": chunk.request_id,
        "chunk_index": batch.block_idx,
        "event_id": getattr(batch, "realtime_event_id", None),
        "server_chunk_start_ms": _transport_ms(
            _session_elapsed_ms(session, at_perf=chunk_started)
        ),
        "server_chunk_end_ms": _transport_ms(_session_elapsed_ms(session)),
        "request_prepare_ms": _transport_ms(request_prepare_ms),
        "scheduler_forward_ms": _transport_ms(scheduler_forward_ms),
        "output_enqueue_wait_ms": _transport_ms(
            send_stats.get("output_enqueue_wait_ms", 0.0)
        ),
        "output_queue_delay_ms": _transport_ms(
            send_stats.get("output_queue_delay_ms", 0.0)
        ),
        "output_queue_size": int(send_stats.get("output_queue_size", 0)),
        "pace_wait_ms": _transport_ms(send_stats["pace_wait_ms"]),
        "header_write_ms": _transport_ms(send_stats["header_write_ms"]),
        "frame_store_wait_ms": _transport_ms(
            send_stats.get("frame_store_wait_ms", 0.0)
        ),
        "frame_store_read_ms": _transport_ms(
            send_stats.get("frame_store_read_ms", 0.0)
        ),
        "frame_store_materialize_ms": _transport_ms(
            send_stats.get("frame_store_materialize_ms", 0.0)
        ),
        "frame_store_producer_wait_ms": _transport_ms(
            send_stats.get("frame_store_producer_wait_ms", 0.0)
        ),
        "frame_store_gpu_copy_ms": _transport_ms(
            send_stats.get("frame_store_gpu_copy_ms", 0.0)
        ),
        "frame_store_mmap_write_ms": _transport_ms(
            send_stats.get("frame_store_mmap_write_ms", 0.0)
        ),
        "frame_store_producer_denoise_ms": _transport_ms(
            send_stats.get("frame_store_producer_denoise_ms", 0.0)
        ),
        "frame_store_producer_refresh_ms": _transport_ms(
            send_stats.get("frame_store_producer_refresh_ms", 0.0)
        ),
        "frame_store_producer_decode_ms": _transport_ms(
            send_stats.get("frame_store_producer_decode_ms", 0.0)
        ),
        "frame_store_producer_post_ms": _transport_ms(
            send_stats.get("frame_store_producer_post_ms", 0.0)
        ),
        "frame_store_producer_clone_ms": _transport_ms(
            send_stats.get("frame_store_producer_clone_ms", 0.0)
        ),
        "frame_store_producer_total_ms": _transport_ms(
            send_stats.get("frame_store_producer_total_ms", 0.0)
        ),
        "frame_store_producer_denoise_to_ready_ms": _transport_ms(
            send_stats.get("frame_store_producer_denoise_to_ready_ms", 0.0)
        ),
        "raw_payload_build_ms": _transport_ms(send_stats["raw_payload_build_ms"]),
        "raw_write_ms": _transport_ms(send_stats["raw_write_ms"]),
        "ws_write_ms": _transport_ms(send_stats["ws_write_ms"]),
        "artc_enqueue_wait_ms": _transport_ms(
            send_stats.get("artc_enqueue_wait_ms", 0.0)
        ),
        "artc_queue_delay_ms": _transport_ms(
            send_stats.get("artc_queue_delay_ms", 0.0)
        ),
        "artc_queue_size": int(send_stats.get("artc_queue_size", 0)),
        "artc_push_ms": _transport_ms(send_stats.get("artc_push_ms", 0.0)),
        "artc_dropped_chunks": int(send_stats.get("artc_dropped_chunks", 0)),
        "chunk_total_ms": _transport_ms(chunk_total_ms),
        "num_batches": send_stats["num_batches"],
        "num_frames": send_stats["num_frames"],
        "raw_bytes": send_stats["raw_bytes"],
        "ws_payload_bytes": send_stats["ws_payload_bytes"],
        "content_type": send_stats["content_type"],
    }
    worker_timings = _realtime_worker_timings(result)
    if worker_timings is not None:
        payload["worker_timings"] = worker_timings
        worker_total_ms = worker_timings.get("total_ms")
        if worker_total_ms is not None:
            payload["scheduler_overhead_ms"] = _transport_ms(
                max(0.0, scheduler_forward_ms - worker_total_ms)
            )
    if session.adapter is not None and hasattr(
        session.adapter, "build_chunk_stats_extra"
    ):
        _merge_realtime_debug_payload(
            payload,
            session.adapter.build_chunk_stats_extra(session, batch, result),
        )
    await send_realtime_ws_bytes(ws, msgspec.msgpack.encode(payload))


@dataclass(slots=True)
class RealtimeOutputItem:
    chunk: RealtimeChunkContext
    batch: Any
    result: Any
    request_prepare_ms: float
    scheduler_forward_ms: float
    chunk_started: float
    enqueued_at: float
    output_enqueue_wait_ms: float
    output_queue_size: int


class RealtimeOutputPipeline:
    """Bounded per-session output pipeline for encode/build/send work."""

    def __init__(
        self,
        ws: WebSocket,
        session: GenerateSession,
        *,
        max_queue_size: int | None = None,
        enqueue_timeout_ms: float | None = None,
    ) -> None:
        self.ws = ws
        self.session = session
        self.max_queue_size = max_queue_size or get_realtime_output_queue_size()
        self.enqueue_timeout_ms = (
            get_realtime_output_enqueue_timeout_ms()
            if enqueue_timeout_ms is None
            else max(0.0, float(enqueue_timeout_ms))
        )
        self._queue: asyncio.Queue[RealtimeOutputItem] = asyncio.Queue()
        self._slots = asyncio.Semaphore(self.max_queue_size)
        self._failed: BaseException | None = None
        self._failure_event = asyncio.Event()
        self._closed = False
        self.output_sink = session.output_sink
        if self.output_sink is None:
            self.output_sink = create_realtime_output_sink(ws, session)
            session.set_output_sink(self.output_sink)
        self._worker_task = asyncio.create_task(
            self._run(),
            name=f"realtime-output-{session.id}",
        )

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed
        self.output_sink.raise_if_failed()

    async def wait_failed(self) -> None:
        pipeline_failure_task = asyncio.create_task(self._failure_event.wait())
        sink_failure_task = asyncio.create_task(self.output_sink.wait_failed())
        try:
            done, pending = await asyncio.wait(
                {pipeline_failure_task, sink_failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                await _await_realtime_task(task)
            for task in done:
                await task
            self.raise_if_failed()
        finally:
            for task in (pipeline_failure_task, sink_failure_task):
                if not task.done():
                    task.cancel()
                    await _await_realtime_task(task)

    async def submit(
        self,
        *,
        chunk: RealtimeChunkContext,
        batch: "Req",
        result,
        request_prepare_ms: float,
        scheduler_forward_ms: float,
        chunk_started: float,
    ) -> None:
        if self._closed:
            raise RealtimeProtocolError(
                "output_pipeline_closed",
                "Realtime output pipeline is already closed",
                session_id=self.session.id,
                chunk_index=getattr(batch, "block_idx", None),
            )
        self.raise_if_failed()

        wait_started = time.perf_counter()
        try:
            if self.enqueue_timeout_ms <= 0:
                await self._slots.acquire()
            else:
                await asyncio.wait_for(
                    self._slots.acquire(),
                    timeout=self.enqueue_timeout_ms / 1000.0,
                )
        except asyncio.TimeoutError as exc:
            raise RealtimeProtocolError(
                "output_backpressure_timeout",
                "Timed out waiting for realtime output queue slot",
                session_id=self.session.id,
                chunk_index=getattr(batch, "block_idx", None),
                queue_size=self._queue.qsize(),
                max_queue_size=self.max_queue_size,
                timeout_ms=round(self.enqueue_timeout_ms, 3),
            ) from exc

        try:
            self.raise_if_failed()
            enqueued_at = time.perf_counter()
            item = RealtimeOutputItem(
                chunk=chunk,
                batch=batch,
                result=result,
                request_prepare_ms=request_prepare_ms,
                scheduler_forward_ms=scheduler_forward_ms,
                chunk_started=chunk_started,
                enqueued_at=enqueued_at,
                output_enqueue_wait_ms=(enqueued_at - wait_started) * 1000.0,
                output_queue_size=self._queue.qsize() + 1,
            )
            self._queue.put_nowait(item)
        except Exception:
            self._slots.release()
            raise

    async def drain(self) -> None:
        await self._queue.join()
        self.raise_if_failed()

    async def close(self) -> None:
        self._closed = True
        await self.drain()
        await self.output_sink.close()
        if not self._worker_task.done():
            self._worker_task.cancel()
        await _await_realtime_task(self._worker_task)

    async def cancel(self) -> None:
        self._closed = True
        if not self._worker_task.done():
            self._worker_task.cancel()
        await _await_realtime_task(self._worker_task)
        await self.output_sink.cancel()

    async def _run(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    output_queue_delay_ms = (
                        time.perf_counter() - item.enqueued_at
                    ) * 1000.0
                    await _send_output_and_log(
                        self.ws,
                        self.session,
                        item.chunk,
                        item.batch,
                        item.result,
                        item.request_prepare_ms,
                        item.scheduler_forward_ms,
                        item.chunk_started,
                        output_enqueue_wait_ms=item.output_enqueue_wait_ms,
                        output_queue_delay_ms=output_queue_delay_ms,
                        output_queue_size=item.output_queue_size,
                        output_sink=self.output_sink,
                    )
                finally:
                    self._queue.task_done()
                    self._slots.release()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed = exc
            self._failure_event.set()
            logger.error(
                "realtime output pipeline failed, session_id=%s, error=%s",
                self.session.id,
                exc,
            )
            while True:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._queue.task_done()
                self._slots.release()


async def _wait_for_next_chunk_or_output_failure(
    adapter,
    session: GenerateSession,
    output_pipeline: RealtimeOutputPipeline,
) -> None:
    wait_task = asyncio.create_task(adapter.wait_for_next_chunk(session))
    failure_task = asyncio.create_task(output_pipeline.wait_failed())
    try:
        done, pending = await asyncio.wait(
            {wait_task, failure_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            await _await_realtime_task(task)
        for task in done:
            await task
    finally:
        for task in (wait_task, failure_task):
            if not task.done():
                task.cancel()
                await _await_realtime_task(task)


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    adapter = session.adapter
    if adapter is None:
        raise ValueError("realtime adapter is not initialized")

    output_pipeline = RealtimeOutputPipeline(ws, session)
    ended_by_adapter = False
    while not session.reached_max_chunks():
        try:
            output_pipeline.raise_if_failed()
            # send to scheduler and generate video chunk
            server_args = get_global_server_args()

            try:
                await _wait_for_next_chunk_or_output_failure(
                    adapter,
                    session,
                    output_pipeline,
                )
            except StopAsyncIteration:
                logger.info(
                    "generation ended by realtime adapter, session_id=%s",
                    session.id,
                )
                ended_by_adapter = True
                break

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()

            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(
                session,
                server_args,
                chunk,
            )
            if batch.condition_inputs:
                logger.debug(
                    "consume realtime conditions, session_id=%s, block_idx=%s, kinds=%s",
                    session.id,
                    batch.block_idx,
                    sorted(batch.condition_inputs),
                )
            request_prepare_ms = timer.mark_ms()

            _, result = await process_generation_batch(async_scheduler_client, batch)
            scheduler_forward_ms = timer.mark_ms()

            # finish
            adapter.on_chunk_complete(session, result)
            await output_pipeline.submit(
                chunk=chunk,
                batch=batch,
                result=result,
                request_prepare_ms=request_prepare_ms,
                scheduler_forward_ms=scheduler_forward_ms,
                chunk_started=chunk_started,
            )

        except asyncio.CancelledError:
            await output_pipeline.cancel()
            logger.info("generation completed, session_id=%s", session.id)
            break
        except WebSocketDisconnect:
            await output_pipeline.cancel()
            logger.info(
                "client disconnected during generation, session_id=%s", session.id
            )
            break
        except Exception as e:
            await output_pipeline.cancel()
            err_msg = str(e).splitlines()[0]
            error_code = _realtime_error_code(e, "generation_error")
            logger.error(
                "error during generate loop, code=%s: %s",
                error_code,
                err_msg,
            )
            try:
                content = (
                    err_msg
                    if error_code == "output_write_timeout"
                    else f"error during generate loop: {err_msg}"
                )
                await write_error_msg(
                    content,
                    ws,
                    code=error_code,
                    details=_realtime_error_details(e, session_id=session.id),
                )
            except Exception as send_error:
                logger.error(
                    "error during sending complete msg: %s",
                    send_error,
                )
            break
    else:
        await output_pipeline.close()
        logger.info(
            "generation reached max chunks, session_id=%s, max_chunks=%s",
            session.id,
            session.request.max_chunks if session.request is not None else None,
        )
        return

    if ended_by_adapter:
        await output_pipeline.close()


async def _send_output_and_log(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    result,
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_started: float,
    *,
    output_enqueue_wait_ms: float = 0.0,
    output_queue_delay_ms: float = 0.0,
    output_queue_size: int = 0,
    output_sink: BaseRealtimeOutputSink | None = None,
) -> RealtimeFrameSendStats:
    if session.adapter is None:
        raise ValueError("realtime adapter is not initialized")
    if output_sink is None:
        output_sink = session.output_sink
        if output_sink is None:
            if session.request is None:
                output_sink = WebSocketRealtimeOutputSink(ws)
            else:
                output_sink = create_realtime_output_sink(ws, session)
            session.set_output_sink(output_sink)
    pace_wait_ms = await _wait_for_realtime_output_slot(session, batch, result)
    send_stats = await output_sink.send(session, result, batch)
    send_stats["pace_wait_ms"] = pace_wait_ms
    send_stats["output_enqueue_wait_ms"] = output_enqueue_wait_ms
    send_stats["output_queue_delay_ms"] = output_queue_delay_ms
    send_stats["output_queue_size"] = output_queue_size
    chunk_total_ms = (time.perf_counter() - chunk_started) * 1000
    _log_realtime_chunk_timing(
        session,
        chunk,
        batch,
        request_prepare_ms,
        scheduler_forward_ms,
        chunk_total_ms,
        send_stats,
    )
    await _send_realtime_chunk_stats(
        ws,
        session,
        chunk,
        batch,
        result,
        request_prepare_ms,
        scheduler_forward_ms,
        chunk_total_ms,
        send_stats,
        chunk_started,
    )
    return send_stats


def _result_num_frames(result) -> int:
    if result.raw_frame_batches is not None:
        return sum(len(frames) for frames in result.raw_frame_batches)
    handles = getattr(result, "raw_frame_store_handles", None)
    if handles is not None:
        return sum(int(getattr(handle, "num_frames", 0)) for handle in handles)
    return 0


def _output_pacing_fps(batch: "Req") -> float:
    fps = float(batch.fps or 0)
    if batch.enable_frame_interpolation:
        fps *= 2 ** int(batch.frame_interpolation_exp or 1)
    return fps


async def _wait_for_realtime_output_slot(
    session: GenerateSession,
    batch: "Req",
    result,
) -> float:
    if not getattr(batch, "realtime_output_pacing", False):
        return 0.0

    frame_count = _result_num_frames(result)
    output_fps = _output_pacing_fps(batch)
    if frame_count <= 0 or output_fps <= 0:
        return 0.0

    now = time.perf_counter()
    next_send_at = session.output_pace_next_send_at
    if next_send_at is None:
        next_send_at = now
    if (
        batch.realtime_event_id is not None
        and batch.realtime_event_id != session.output_pace_last_event_id
    ):
        next_send_at = min(next_send_at, now)
        session.output_pace_last_event_id = batch.realtime_event_id

    wait_s = max(0.0, next_send_at - now)
    if wait_s > 0:
        await asyncio.sleep(wait_s)

    send_started_at = time.perf_counter()
    session.output_pace_next_send_at = (
        max(next_send_at, send_started_at) + frame_count / output_fps
    )
    return wait_s * 1000


async def _await_realtime_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    try:
        await task
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass
    except Exception as e:
        logger.debug("realtime task exited with error: %s", e)


async def _send_realtime_init_ack(
    ws: WebSocket,
    session: GenerateSession,
    request: RealtimeVideoGenerationsRequest,
) -> None:
    payload = {
        "type": "init_ack",
        "session_id": session.id,
        "server_ack_ms": _transport_ms(_session_elapsed_ms(session)),
        "server_session_start_wall_ms": _transport_ms(
            getattr(session, "created_at_wall_ms", 0.0)
        ),
        "server_ack_wall_ms": _transport_ms(time.time() * 1000.0),
        "request": {
            "fps": request.fps,
            "max_chunks": request.max_chunks,
            "size": request.size,
            "realtime_output_format": request.realtime_output_format,
            "realtime_output_pacing": bool(request.realtime_output_pacing),
        },
    }
    if session.adapter is not None:
        _merge_realtime_debug_payload(
            payload,
            session.adapter.build_init_ack(session, request),
        )
    if session.output_sink is not None:
        _merge_realtime_debug_payload(payload, session.output_sink.build_init_ack())
    await send_realtime_ws_bytes(ws, msgspec.msgpack.encode(payload))


async def _send_realtime_event_ack(
    ws: WebSocket,
    session: GenerateSession,
    event: RealtimeEvent,
    event_log: str,
    *,
    message_bytes: int,
    recv_perf: float,
    ack_perf: float,
) -> None:
    payload = {
        "type": "event_ack",
        "session_id": session.id,
        "event_id": event.event_id,
        "kind": event.kind,
        "server_recv_ms": _transport_ms(
            _session_elapsed_ms(session, at_perf=recv_perf)
        ),
        "server_ack_ms": _transport_ms(_session_elapsed_ms(session, at_perf=ack_perf)),
        "ingest_ms": _transport_ms((ack_perf - recv_perf) * 1000.0),
        "message_bytes": message_bytes,
    }
    if session.adapter is not None:
        _merge_realtime_debug_payload(
            payload,
            session.adapter.build_event_ack(session, event, event_log),
        )
    await send_realtime_ws_bytes(ws, msgspec.msgpack.encode(payload))


async def _listen_events(ws: WebSocket, session: GenerateSession):
    """listen for user events: usually condition inputs"""
    async for message in ws.iter_bytes():
        recv_perf = time.perf_counter()
        data = None
        try:
            data = msgspec.msgpack.decode(message)
            if not isinstance(data, dict):
                raise ValueError("realtime event must be a map")
            realtime_event = RealtimeEvent.model_validate(data)
            if session.adapter is None:
                raise ValueError("realtime adapter is not initialized")
            event_log = session.adapter.ingest_event(session, realtime_event)
            logger.info(
                "receive realtime event, session_id=%s, event_id=%s, %s",
                session.id,
                realtime_event.event_id,
                event_log,
            )
            ack_perf = time.perf_counter()
            await _send_realtime_event_ack(
                ws,
                session,
                realtime_event,
                event_log,
                message_bytes=len(message),
                recv_perf=recv_perf,
                ack_perf=ack_perf,
            )
        except Exception as e:
            if _realtime_error_code(e, "") == "output_write_timeout":
                logger.warning(
                    "event ack write timeout, session_id=%s, error=%s",
                    session.id,
                    e,
                )
                raise
            event_kind = data.get("kind") if isinstance(data, dict) else None
            event_id = data.get("event_id") if isinstance(data, dict) else None
            error_code = _realtime_error_code(e, "invalid_event")
            err_msg = _realtime_error_message(e, "invalid event")
            logger.warning(
                "invalid event, kind=%s, code=%s, error=%s",
                event_kind,
                error_code,
                err_msg,
            )
            await write_error_msg(
                err_msg,
                ws,
                code=error_code,
                details=_realtime_error_details(
                    e,
                    kind=event_kind,
                    event_id=event_id,
                ),
            )
            continue


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = msgspec.msgpack.decode(await ws.receive_bytes())
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            adapter = get_realtime_model_adapter(get_global_server_args())
            session.set_adapter(adapter)
            await adapter.on_init(session, realtime_req)

            # Keep session state update atomic with validated request.
            session.set_request(realtime_req)
            session.set_output_sink(
                await create_realtime_output_sink_async(ws, session)
            )
            await _send_realtime_init_ack(ws, session, realtime_req)
            break
        except WebSocketDisconnect:
            raise
        except Exception as e:
            if _realtime_error_code(e, "") == "output_write_timeout":
                logger.warning(
                    "init ack write timeout, session_id=%s, error=%s",
                    session.id,
                    e,
                )
                raise
            error_code = _realtime_error_code(e, "invalid_generate_request")
            err_msg = _realtime_error_message(e, "invalid generate request")
            logger.warning(
                "invalid generate request, session_id=%s, code=%s, error=%s",
                session.id,
                error_code,
                err_msg,
            )
            await write_error_msg(
                err_msg,
                ws,
                code=error_code,
                details=_realtime_error_details(e, session_id=session.id),
            )
            continue


async def _cleanup_realtime_session(
    session: GenerateSession,
    generate_task: asyncio.Task | None,
    listen_task: asyncio.Task | None,
) -> None:
    logger.info("terminating session, session_id=%s", session.id)
    for task in (generate_task, listen_task):
        if task and not task.done():
            task.cancel()
    for task in (generate_task, listen_task):
        if task is None:
            continue
        await _await_realtime_task(task)
    try:
        await async_scheduler_client.forward(
            ReleaseRealtimeSessionReq(session_id=session.id)
        )
    except Exception as e:
        logger.warning(
            "failed to release realtime session on scheduler, session_id=%s, error=%s",
            session.id,
            e,
        )
    if session.input_temp_dir is not None:
        shutil.rmtree(session.input_temp_dir, ignore_errors=True)
    if session.output_sink is not None:
        await session.output_sink.cancel()
    session.dispose()


async def _close_realtime_websocket(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except (RuntimeError, WebSocketDisconnect):
        pass


async def _wait_for_server_warmup(websocket: WebSocket) -> None:
    warmup_done = getattr(websocket.app.state, "server_warmup_done", None)
    if warmup_done is not None and not warmup_done.is_set():
        await warmup_done.wait()


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    await _wait_for_server_warmup(websocket)
    if _ACTIVE_SESSION_IDS and not await _wait_for_active_session_slot():
        logger.warning(
            "reject realtime session because another session is active: %s",
            sorted(_ACTIVE_SESSION_IDS),
        )
        try:
            await write_error_msg(
                "another realtime session is already active",
                websocket,
                code="session_busy",
                details={"active_session_ids": sorted(_ACTIVE_SESSION_IDS)},
            )
        finally:
            await websocket.close(code=1008)
        return

    session = GenerateSession()
    _ACTIVE_SESSION_IDS.add(session.id)
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(websocket, session)

        # continuously generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # continuously listen for user events
        listen_task = asyncio.create_task(_listen_events(websocket, session))

        wait_tasks = [generate_task, listen_task]
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if generate_task.done() and session.reached_max_chunks():
            await _close_realtime_websocket(
                websocket,
                code=1000,
                reason="generation complete",
            )

    except WebSocketDisconnect:
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        try:
            await _cleanup_realtime_session(session, generate_task, listen_task)
        finally:
            _ACTIVE_SESSION_IDS.discard(session.id)


async def write_error_msg(
    error_msg: str,
    websocket: WebSocket,
    *,
    code: str = "server_error",
    details: dict[str, Any] | None = None,
):
    payload: dict[str, Any] = {
        "type": "error",
        "code": code,
        "content": error_msg,
    }
    if details:
        payload["details"] = details
    await send_realtime_ws_bytes(websocket, msgspec.msgpack.encode(payload))
