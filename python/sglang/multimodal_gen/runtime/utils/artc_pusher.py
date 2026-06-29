"""Push video/audio via AliRTC SDK (ARTC protocol).

The AliRTC Python wrapper and native CoreService run in a dedicated worker
process. The main process only owns lifecycle control and chunk delivery. This
keeps SDK hangs, wrapper thread state, and CoreService cleanup isolated from the
model-serving process.
"""

import importlib.util
import logging
import os
import pickle
import queue
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np


def _fallback_compact_chunk_trace_fields(**kwargs):
    fields = {}
    meta = kwargs.pop("meta", None) or {}
    for key, value in meta.items():
        if key in {
            "session_id",
            "turn_id",
            "client_chunk_idx",
            "client_chunk_ms",
            "chunk_source",
            "is_filler",
            "is_first_real_chunk",
            "allow_preempt_filler",
            "turn_start_policy",
            "client_turn_t0_wall_ms",
            "client_t0_to_post_start_ms",
            "client_post_start_wall_ms",
            "client_input_rms",
            "client_input_peak",
            "queue_size",
            "pending_filler_ms",
            "audio_queue_ms",
            "video_queue_ms",
            "pending_filler_chunks",
            "pending_real_chunks",
            "wait_after_received_ms",
            "session_chunk_idx",
            "input_audio_ms",
            "input_rms",
            "input_peak",
            "generate_ms",
            "audio_ms",
            "video_frames",
            "first_audio_pts",
            "first_video_pts",
            "last_audio_pts",
            "last_video_pts",
            "frame_idx",
            "is_silent",
            "rms",
            "peak",
            "pending_frames",
            "pusher_userid",
            "state",
            "reason",
            "error_code",
            "error_message",
            "audio_published",
            "video_published",
            "dual_stream_enabled",
            "low_stream_profile",
        }:
            fields[key] = value
    for key, value in kwargs.items():
        if value is not None:
            fields[key] = value
    return fields


def _load_chunk_timeline_helpers():
    # The ARTC worker is launched as this file directly. Importing through the
    # sglang package triggers the full multimodal runtime import stack and adds
    # several seconds to every ARTC startup.
    timeline_path = os.path.join(os.path.dirname(__file__), "chunk_timeline.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "_sglang_artc_chunk_timeline", timeline_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load chunk_timeline from {timeline_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.emit_chunk_timeline,
            getattr(
                module,
                "compact_chunk_trace_fields",
                _fallback_compact_chunk_trace_fields,
            ),
        )
    except Exception:

        def _noop_emit_chunk_timeline(path, event, **fields):
            return

        return _noop_emit_chunk_timeline, _fallback_compact_chunk_trace_fields


emit_chunk_timeline, compact_chunk_trace_fields = _load_chunk_timeline_helpers()


logger = logging.getLogger(__name__)

_SDK_DIR = os.path.join(os.path.dirname(__file__), "alirtc")
_CHUNK = "chunk"
_CLEAR_BUFFER = "clear_buffer"
_STOP = "stop"


def _ensure_sdk_importable(sdk_path: str) -> str:
    """Make AliRTC Python and native libraries discoverable."""
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    lib_dir = os.path.join(sdk_path, "Release", "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + ld_path
    return lib_dir


class _IpcTimeout(TimeoutError):
    pass


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _send_message(fd: int, message) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    _write_all(fd, struct.pack("!I", len(data)))
    _write_all(fd, data)


def _recv_exact(fd: int, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("ARTC worker pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(fd: int, timeout: Optional[float] = None):
    if timeout is not None:
        readable, _, _ = select.select([fd], [], [], timeout)
        if not readable:
            raise _IpcTimeout()
    header = _recv_exact(fd, 4)
    size = struct.unpack("!I", header)[0]
    return pickle.loads(_recv_exact(fd, size))


def _put_status(status_fd: int, kind: str, message: str = "") -> None:
    try:
        _send_message(status_fd, (kind, message))
    except Exception:
        pass


def _audio_int16_stats(audio_slice) -> dict[str, float | int | bool]:
    if audio_slice is None or len(audio_slice) == 0:
        return {"rms": 0.0, "peak": 0, "is_silent": True}
    audio_i32 = np.asarray(audio_slice, dtype=np.int32)
    peak = int(np.max(np.abs(audio_i32))) if len(audio_i32) else 0
    rms = (
        float(np.sqrt(np.mean(np.square(audio_i32.astype(np.float64)))))
        if len(audio_i32)
        else 0.0
    )
    try:
        non_silent_threshold = int(
            os.environ.get("SGLANG_ARTC_NON_SILENT_INT16_THRESHOLD", "16")
        )
    except ValueError:
        non_silent_threshold = 16
    is_silent = peak <= max(0, non_silent_threshold)
    return {"rms": round(rms, 3), "peak": peak, "is_silent": is_silent}


class _WorkerEventHandler:
    """AliRTC callbacks scoped to one worker process and one channel."""

    def __init__(
        self,
        status_fd: int,
        timeline_path: Optional[str] = None,
        session_id: Optional[str] = None,
        channel: Optional[str] = None,
        pusher_userid: Optional[str] = None,
        dual_stream_enabled: Optional[bool] = None,
        low_stream_profile: Optional[str] = None,
    ):
        self._status_fd = status_fd
        self._timeline_path = timeline_path
        self._session_id = session_id
        self._channel = channel
        self._pusher_userid = pusher_userid
        self._dual_stream_enabled = dual_stream_enabled
        self._low_stream_profile = low_stream_profile
        self.joined = threading.Event()
        self.left = threading.Event()
        self.audio_published = threading.Event()
        self.video_published = threading.Event()
        self.failed = False
        self.push_video_full = False
        self.push_audio_full = False

    @staticmethod
    def _state_value(value):
        return getattr(value, "value", value)

    def _emit(self, event: str, **fields) -> None:
        emit_chunk_timeline(
            self._timeline_path,
            event,
            session_id=self._session_id,
            channel=self._channel,
            pusher_userid=self._pusher_userid,
            audio_published=self.audio_published.is_set(),
            video_published=self.video_published.is_set(),
            dual_stream_enabled=self._dual_stream_enabled,
            low_stream_profile=self._low_stream_profile,
            wall_clock=time.time(),
            **fields,
        )

    def OnAudioPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC audio publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        if self._state_value(newState) == 2:
            self.audio_published.set()
        self._emit(
            "artc_publish_state",
            media="audio",
            old_state=self._state_value(oldState),
            new_state=self._state_value(newState),
            state=self._state_value(newState),
            elapsed=elapsed,
            callback_channel=channel,
        )

    def OnVideoPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC video publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        if self._state_value(newState) == 2:
            self.video_published.set()
        self._emit(
            "artc_publish_state",
            media="video",
            old_state=self._state_value(oldState),
            new_state=self._state_value(newState),
            state=self._state_value(newState),
            elapsed=elapsed,
            callback_channel=channel,
        )

    def OnPushAudioFrameBufferFull(self, isFull):
        self.push_audio_full = isFull
        if isFull:
            logger.debug("ARTC audio buffer full")

    def OnPushVideoFrameBufferFull(self, isFull):
        self.push_video_full = isFull
        if isFull:
            logger.debug("ARTC video buffer full")

    def OnError(self, error_code):
        self.failed = True
        msg = f"ARTC SDK error: {error_code}"
        logger.error(msg)
        self._emit(
            "artc_error", error_code=error_code, error_message=msg, message=msg
        )
        self._emit(
            "artc_push_error", error_code=error_code, error_message=msg, message=msg
        )
        self._emit(
            "error",
            component="artc",
            error_code=error_code,
            error_message=msg,
            message=msg,
        )
        _put_status(self._status_fd, "failed", msg)

    def OnConnectionStatusChanged(self, status, reason):
        logger.info("ARTC connection: status=%s reason=%s", status, reason)
        self._emit(
            "artc_connection_state",
            status=self._state_value(status),
            state=self._state_value(status),
            reason=self._state_value(reason),
        )
        self._emit(
            "connection_state",
            component="artc",
            status=self._state_value(status),
            state=self._state_value(status),
            reason=self._state_value(reason),
        )

    def OnJoinChannelResult(self, result, channel, userId):
        logger.info(
            "ARTC JoinChannel result=%s channel=%s user=%s",
            result,
            channel,
            userId,
        )
        if result != 0:
            self.failed = True
            self._emit(
                "artc_error",
                error_code=result,
                error_message=f"ARTC JoinChannel failed: {result}",
                message=f"ARTC JoinChannel failed: {result}",
                callback_channel=channel,
                user_id=userId,
            )
            self._emit(
                "artc_push_error",
                error_code=result,
                error_message=f"ARTC JoinChannel failed: {result}",
                message=f"ARTC JoinChannel failed: {result}",
                callback_channel=channel,
                user_id=userId,
            )
            self._emit(
                "error",
                component="artc",
                error_code=result,
                error_message=f"ARTC JoinChannel failed: {result}",
                message=f"ARTC JoinChannel failed: {result}",
                callback_channel=channel,
                user_id=userId,
            )
            _put_status(
                self._status_fd,
                "failed",
                f"ARTC JoinChannel failed: {result}",
            )
        self.joined.set()

    def OnLeaveChannelResult(self, result):
        logger.info("ARTC LeaveChannel result=%s", result)
        self.left.set()

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            pass

        return _noop


def _create_engine(config: dict, handler: _WorkerEventHandler):
    lib_dir = _ensure_sdk_importable(config["sdk_path"])
    core_service = os.path.join(lib_dir, "AliRtcCoreService")

    from AliRTCEngine import CreateAliRTCEngine  # noqa: E402

    log_path = os.environ.get("SGLANG_ARTC_LOG_PATH", "/tmp/artc_sdk_logs")
    os.makedirs(log_path, exist_ok=True)

    t0 = time.perf_counter()
    engine = CreateAliRTCEngine(
        eventHandler=handler,
        lowPort=int(os.environ.get("SGLANG_ARTC_LOW_PORT", "42000")),
        highPort=int(os.environ.get("SGLANG_ARTC_HIGH_PORT", "45000")),
        logPath=log_path,
        coreServicePath=core_service,
        h5mode=False,
        extra="{}",
    )
    logger.info(
        "ARTC worker CreateAliRTCEngine finished in %.3fs",
        time.perf_counter() - t0,
    )
    return engine


def _configure_engine(engine, config: dict) -> None:
    from AliRTCLinuxSdkDefine import (  # noqa: E402
        AliEngineClientRole,
        AliEngineFrameRate,
        AliEngineRotationMode,
        AliEngineVideoEncoderConfiguration,
        AliEngineVideoEncoderOrientationMode,
        AliEngineVideoMirrorMode,
        RenderMode,
        VideoSource,
    )

    frame_rate_enum = {
        5: AliEngineFrameRate.AliEngineFrameRateFps5,
        10: AliEngineFrameRate.AliEngineFrameRateFps10,
        15: AliEngineFrameRate.AliEngineFrameRateFps15,
        20: AliEngineFrameRate.AliEngineFrameRateFps20,
        25: AliEngineFrameRate.AliEngineFrameRateFps25,
        30: AliEngineFrameRate.AliEngineFrameRateFps30,
        60: AliEngineFrameRate.AliEngineFrameRateFps60,
    }.get(config["fps"], AliEngineFrameRate.AliEngineFrameRateFps25)

    video_cfg = AliEngineVideoEncoderConfiguration(
        width=config["width"],
        height=config["height"],
        f=frame_rate_enum,
        b=2000,
        ori=AliEngineVideoEncoderOrientationMode.AliEngineVideoEncoderOrientationModeAdaptive,
        mr=AliEngineVideoMirrorMode.AliEngineVideoMirrorModeDisabled,
        rotation=AliEngineRotationMode.AliEngineRotationMode_0,
    )
    engine.SetVideoEncoderConfiguration(video_cfg)
    engine.SetExternalVideoSource(
        True,
        VideoSource.VideoSourceCamera,
        RenderMode.RenderModeFill,
    )
    engine.SetExternalAudioSource(True, 16000, 1)
    engine.PublishLocalVideoStream(True)
    engine.PublishLocalAudioStream(True)
    engine.SetClientRole(AliEngineClientRole.AliEngineClientRoleInteractive)


def _join_channel(engine, handler: _WorkerEventHandler, config: dict) -> None:
    from AliRTCLinuxSdkDefine import (  # noqa: E402
        JoinChannelConfig,
        PublishAvsyncMode,
        PublishMode,
    )

    join_cfg = JoinChannelConfig()
    avsync_mode = str(config.get("avsync_mode") or "nodelay").strip().lower()
    if avsync_mode in ("pts", "with_pts", "withpts"):
        join_cfg.publishAvsyncMode = PublishAvsyncMode.PublishAvsyncWithPts
    else:
        join_cfg.publishAvsyncMode = PublishAvsyncMode.PublishAvsyncNoDelay
    join_cfg.publishMode = PublishMode.PublishAutomatically
    engine.JoinChannel(
        config["token"],
        config["channel"],
        config["userid"],
        config["userid"],
        join_cfg,
    )

    if not handler.joined.wait(timeout=float(config["join_timeout"])):
        raise RuntimeError("ARTC JoinChannel timed out")
    if handler.failed:
        raise RuntimeError("ARTC JoinChannel failed")

    handler.audio_published.wait(timeout=5.0)
    handler.video_published.wait(timeout=5.0)
    logger.info(
        "ARTC worker ready: channel=%s user=%s %dx%d@%dfps avsync=%s",
        config["channel"],
        config["userid"],
        config["width"],
        config["height"],
        config["fps"],
        avsync_mode,
    )


def _drain_worker_queue(
    engine, handler: _WorkerEventHandler, config: dict, command_fd: int
):
    from AliRTCLinuxSdkDefine import (  # noqa: E402
        VideoBufferType,
        VideoDataFormat,
        VideoDataSample,
        VideoSource,
    )

    fps = int(config["fps"])
    width = int(config["width"])
    height = int(config["height"])
    ms_per_frame = 1000 // fps
    samples_per_frame = 16000 // fps
    v_ts = 0
    a_ts = 0
    timeline_path = config.get("timeline_path")
    frame_pacing = bool(config.get("frame_pacing", False))
    reset_pts_on_clear = bool(config.get("reset_pts_on_clear", True))
    frame_interval_s = 1.0 / max(fps, 1)
    next_frame_at = time.monotonic()
    last_pts_reset_meta = None

    def _worker_meta_is_filler(meta: dict) -> bool:
        explicit = meta.get("is_filler")
        if explicit is not None:
            if isinstance(explicit, str):
                return explicit.strip().lower() in ("1", "true", "yes", "on")
            return bool(explicit)
        return bool(meta.get("used_silence")) or not bool(meta.get("audio_loaded"))

    while True:
        try:
            item = _recv_message(command_fd, timeout=2.0)
        except _IpcTimeout:
            if handler.failed:
                raise RuntimeError("ARTC SDK failed during push loop")
            continue
        except EOFError:
            break

        if item is None or item[0] == _STOP:
            break
        if item[0] == _CLEAR_BUFFER:
            meta = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            clear_started = time.monotonic()
            try:
                old_next_video_pts = v_ts
                old_next_audio_pts = a_ts
                engine.ClearDataBuffer()
                handler.push_audio_full = False
                handler.push_video_full = False
                if reset_pts_on_clear:
                    v_ts = 0
                    a_ts = 0
                last_pts_reset_meta = {
                    "old_next_audio_pts": old_next_audio_pts,
                    "old_next_video_pts": old_next_video_pts,
                    "new_first_audio_pts": a_ts,
                    "new_first_video_pts": v_ts,
                    "reset_applied": reset_pts_on_clear,
                    "reason": meta.get("reason") or "clear_buffer",
                }
                next_frame_at = time.monotonic()
                emit_chunk_timeline(
                    timeline_path,
                    "artc_worker_buffer_cleared",
                    session_id=config.get("session_id"),
                    channel=config.get("channel"),
                    pusher_userid=config.get("userid"),
                    chunk_idx=meta.get("chunk_idx"),
                    audio_chunk_idx=meta.get("audio_chunk_idx"),
                    turn_id=meta.get("turn_id"),
                    reason=meta.get("reason"),
                    reset_pts=reset_pts_on_clear,
                    elapsed_ms=round((time.monotonic() - clear_started) * 1000, 3),
                )
                emit_chunk_timeline(
                    timeline_path,
                    "pts_reset_on_real_audio",
                    session_id=config.get("session_id"),
                    channel=config.get("channel"),
                    pusher_userid=config.get("userid"),
                    chunk_idx=meta.get("chunk_idx"),
                    audio_chunk_idx=meta.get("audio_chunk_idx"),
                    turn_id=meta.get("turn_id"),
                    old_next_audio_pts=old_next_audio_pts,
                    old_next_video_pts=old_next_video_pts,
                    new_first_audio_pts=a_ts,
                    new_first_video_pts=v_ts,
                    reset_applied=reset_pts_on_clear,
                    reason=meta.get("reason") or "clear_buffer",
                )
            except Exception as exc:
                logger.warning("ARTC ClearDataBuffer failed: %s", exc)
            continue
        if item[0] != _CHUNK:
            continue

        frames_np, audio_int16 = item[1], item[2]
        meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        chunk_idx = meta.get("chunk_idx")
        audio_chunk_idx = meta.get("audio_chunk_idx")
        num_frames = frames_np.shape[0]
        chunk_started = time.monotonic()
        is_filler_chunk = _worker_meta_is_filler(meta)
        pending_frames = meta.get("pending_frames")
        if not is_filler_chunk:
            emit_chunk_timeline(
                timeline_path,
                "pts_real_audio_alignment",
                **compact_chunk_trace_fields(
                    session_id=config.get("session_id"),
                    meta=meta,
                    chunk_idx=chunk_idx,
                    audio_chunk_idx=audio_chunk_idx,
                    pts=a_ts,
                    wall_clock=time.time(),
                    old_next_audio_pts=(
                        last_pts_reset_meta.get("old_next_audio_pts")
                        if last_pts_reset_meta
                        else a_ts
                    ),
                    old_next_video_pts=(
                        last_pts_reset_meta.get("old_next_video_pts")
                        if last_pts_reset_meta
                        else v_ts
                    ),
                    new_first_audio_pts=a_ts,
                    new_first_video_pts=v_ts,
                    reset_applied=(
                        last_pts_reset_meta.get("reset_applied")
                        if last_pts_reset_meta
                        else False
                    ),
                    reason=(
                        last_pts_reset_meta.get("reason")
                        if last_pts_reset_meta
                        else "real_audio_chunk"
                    ),
                ),
            )
            last_pts_reset_meta = None
        emit_chunk_timeline(
            timeline_path,
            "artc_worker_chunk_received",
            **compact_chunk_trace_fields(
                session_id=config.get("session_id"),
                meta=meta,
                chunk_idx=chunk_idx,
                audio_chunk_idx=audio_chunk_idx,
                pts=a_ts,
                pending_frames=pending_frames,
            ),
            used_silence=meta.get("used_silence"),
            audio_loaded=meta.get("audio_loaded"),
            audio_prefetched=meta.get("audio_prefetched"),
            frame_count=int(num_frames),
            audio_samples=int(len(audio_int16)) if audio_int16 is not None else 0,
            video_pts_ms=v_ts,
            audio_pts_ms=a_ts,
        )
        first_non_silent_audio_pushed = False
        for i in range(num_frames):
            if handler.failed:
                raise RuntimeError("ARTC SDK failed during frame push")

            if frame_pacing:
                now = time.monotonic()
                if next_frame_at > now:
                    time.sleep(next_frame_at - now)
                next_frame_at = max(next_frame_at + frame_interval_s, time.monotonic())

            while handler.push_video_full:
                time.sleep(0.001)
                if handler.failed:
                    raise RuntimeError("ARTC SDK failed while video buffer full")

            frame = frames_np[i]
            video_sample = VideoDataSample()
            video_sample.width = width
            video_sample.height = height
            video_sample.format = VideoDataFormat.VideoDataFormatRGB24
            video_sample.bufferType = VideoBufferType.VideoBufferTypeRawData
            video_sample.data = frame.tobytes()
            video_sample.dataLen = width * height * 3
            video_sample.timeStamp = v_ts
            video_sample.strideY = 0
            video_sample.strideU = 0
            video_sample.strideV = 0
            video_sample.rotation = 0
            video_push_started = time.monotonic()
            try:
                engine.PushExternalVideoFrame(
                    video_sample, VideoSource.VideoSourceCamera
                )
            except Exception as exc:
                emit_chunk_timeline(
                    timeline_path,
                    "artc_push_error",
                    **compact_chunk_trace_fields(
                        session_id=config.get("session_id"),
                        meta=meta,
                        chunk_idx=chunk_idx,
                        audio_chunk_idx=audio_chunk_idx,
                        pts=v_ts,
                        wall_clock=time.time(),
                        frame_idx=i,
                        pending_frames=pending_frames,
                        error_message=str(exc),
                        error_code=type(exc).__name__,
                        message=str(exc),
                        media="video",
                    ),
                )
                raise
            if i == 0:
                emit_chunk_timeline(
                    timeline_path,
                    "artc_worker_first_video_frame_pushed",
                    **compact_chunk_trace_fields(
                        session_id=config.get("session_id"),
                        meta=meta,
                        chunk_idx=chunk_idx,
                        audio_chunk_idx=audio_chunk_idx,
                        pts=v_ts,
                    ),
                    video_pts_ms=v_ts,
                    elapsed_ms=round((time.monotonic() - chunk_started) * 1000, 3),
                )
                emit_chunk_timeline(
                    timeline_path,
                    "artc_first_video_frame_push_done",
                    **compact_chunk_trace_fields(
                        session_id=config.get("session_id"),
                        meta=meta,
                        chunk_idx=chunk_idx,
                        audio_chunk_idx=audio_chunk_idx,
                        pts=v_ts,
                        wall_clock=time.time(),
                        frame_idx=i,
                        pending_frames=pending_frames,
                        audio_queue_ms=meta.get("audio_queue_ms"),
                        video_queue_ms=meta.get("video_queue_ms"),
                    ),
                    video_pts_ms=v_ts,
                    frame_index=i,
                    push_ms=round(
                        (time.monotonic() - video_push_started) * 1000,
                        3,
                    ),
                    wall_ms=int(round(time.time() * 1000.0)),
                    elapsed_ms=round((time.monotonic() - chunk_started) * 1000, 3),
                )
            v_ts += ms_per_frame

            if audio_int16 is not None:
                while handler.push_audio_full:
                    time.sleep(0.001)
                    if handler.failed:
                        raise RuntimeError("ARTC SDK failed while audio buffer full")

                start = i * samples_per_frame
                end = min(start + samples_per_frame, len(audio_int16))
                audio_slice = audio_int16[start:end]
                if len(audio_slice) < samples_per_frame:
                    audio_slice = np.pad(
                        audio_slice,
                        (0, samples_per_frame - len(audio_slice)),
                    )
                audio_stats = _audio_int16_stats(audio_slice)
                audio_bytes = audio_slice.tobytes()
                audio_push_started = time.monotonic()
                if i == 0:
                    emit_chunk_timeline(
                        timeline_path,
                        "artc_first_audio_frame_push_start",
                        **compact_chunk_trace_fields(
                            session_id=config.get("session_id"),
                            meta=meta,
                            chunk_idx=chunk_idx,
                            audio_chunk_idx=audio_chunk_idx,
                            pts=a_ts,
                            wall_clock=time.time(),
                            frame_idx=i,
                            is_silent=audio_stats.get("is_silent"),
                            rms=audio_stats.get("rms"),
                            peak=audio_stats.get("peak"),
                            pending_frames=pending_frames,
                            audio_queue_ms=meta.get("audio_queue_ms"),
                            video_queue_ms=meta.get("video_queue_ms"),
                        ),
                        audio_pts_ms=a_ts,
                        frame_index=i,
                        wall_ms=int(round(time.time() * 1000.0)),
                        audio_samples=int(len(audio_slice)),
                    )
                try:
                    engine.PushExternalAudioFrameRawData(
                        audio_bytes, len(audio_bytes), a_ts
                    )
                except Exception as exc:
                    emit_chunk_timeline(
                        timeline_path,
                        "artc_push_error",
                        **compact_chunk_trace_fields(
                            session_id=config.get("session_id"),
                            meta=meta,
                            chunk_idx=chunk_idx,
                            audio_chunk_idx=audio_chunk_idx,
                            pts=a_ts,
                            wall_clock=time.time(),
                            frame_idx=i,
                            is_silent=audio_stats.get("is_silent"),
                            rms=audio_stats.get("rms"),
                            peak=audio_stats.get("peak"),
                            pending_frames=pending_frames,
                            error_message=str(exc),
                            error_code=type(exc).__name__,
                            message=str(exc),
                            media="audio",
                        ),
                    )
                    raise
                if i == 0:
                    emit_chunk_timeline(
                        timeline_path,
                        "artc_worker_first_audio_frame_pushed",
                        **compact_chunk_trace_fields(
                            session_id=config.get("session_id"),
                            meta=meta,
                            chunk_idx=chunk_idx,
                            audio_chunk_idx=audio_chunk_idx,
                            pts=a_ts,
                        ),
                        audio_pts_ms=a_ts,
                        elapsed_ms=round(
                            (time.monotonic() - chunk_started) * 1000, 3
                        ),
                    )
                    emit_chunk_timeline(
                        timeline_path,
                        "artc_first_audio_frame_push_done",
                        **compact_chunk_trace_fields(
                            session_id=config.get("session_id"),
                            meta=meta,
                            chunk_idx=chunk_idx,
                            audio_chunk_idx=audio_chunk_idx,
                            pts=a_ts,
                            wall_clock=time.time(),
                            frame_idx=i,
                            is_silent=audio_stats.get("is_silent"),
                            rms=audio_stats.get("rms"),
                            peak=audio_stats.get("peak"),
                            pending_frames=pending_frames,
                            audio_queue_ms=meta.get("audio_queue_ms"),
                            video_queue_ms=meta.get("video_queue_ms"),
                        ),
                        audio_pts_ms=a_ts,
                        frame_index=i,
                        push_ms=round(
                            (time.monotonic() - audio_push_started) * 1000, 3
                        ),
                        wall_ms=int(round(time.time() * 1000.0)),
                        elapsed_ms=round(
                            (time.monotonic() - chunk_started) * 1000, 3
                        ),
                    )
                if (
                    not first_non_silent_audio_pushed
                    and len(audio_slice) > 0
                    and not bool(audio_stats.get("is_silent"))
                ):
                    first_non_silent_audio_pushed = True
                    emit_chunk_timeline(
                        timeline_path,
                        "artc_first_non_silent_audio_frame_push_done",
                        **compact_chunk_trace_fields(
                            session_id=config.get("session_id"),
                            meta=meta,
                            chunk_idx=chunk_idx,
                            audio_chunk_idx=audio_chunk_idx,
                            pts=a_ts,
                            wall_clock=time.time(),
                            frame_idx=i,
                            is_silent=False,
                            rms=audio_stats.get("rms"),
                            peak=audio_stats.get("peak"),
                            pending_frames=pending_frames,
                            audio_queue_ms=meta.get("audio_queue_ms"),
                            video_queue_ms=meta.get("video_queue_ms"),
                        ),
                        audio_pts_ms=a_ts,
                        frame_index=i,
                        audio_peak=audio_stats.get("peak"),
                        push_ms=round(
                            (time.monotonic() - audio_push_started) * 1000, 3
                        ),
                        wall_ms=int(round(time.time() * 1000.0)),
                        elapsed_ms=round(
                            (time.monotonic() - chunk_started) * 1000, 3
                        ),
                    )
                a_ts += ms_per_frame
        emit_chunk_timeline(
            timeline_path,
            "artc_worker_chunk_pushed",
            **compact_chunk_trace_fields(
                session_id=config.get("session_id"),
                meta=meta,
                chunk_idx=chunk_idx,
                audio_chunk_idx=audio_chunk_idx,
                pts=a_ts,
            ),
            frame_count=int(num_frames),
            audio_samples=int(len(audio_int16)) if audio_int16 is not None else 0,
            elapsed_ms=round((time.monotonic() - chunk_started) * 1000, 3),
            next_video_pts_ms=v_ts,
            next_audio_pts_ms=a_ts,
            video_buffer_full=handler.push_video_full,
            audio_buffer_full=handler.push_audio_full,
        )


def _leave_and_release(engine, handler: _WorkerEventHandler, config: dict) -> None:
    if engine is None:
        return

    left_completed = False
    try:
        handler.left.clear()
        engine.LeaveChannel()
        left_timeout = float(config["leave_timeout"])
        left_completed = handler.left.wait(timeout=left_timeout)
        if not left_completed:
            logger.warning(
                "ARTC worker LeaveChannel did not complete within %.1fs for channel=%s",
                left_timeout,
                config["channel"],
            )
    except Exception as exc:
        logger.warning("ARTC worker LeaveChannel error: %s", exc)

    if not left_completed:
        logger.warning(
            "ARTC worker killing its process group after leave timeout to avoid "
            "orphaned AliRtcCoreService"
        )
        _kill_own_process_group_if_isolated(config)
        return

    try:
        engine.Release()
    except Exception as exc:
        logger.warning("ARTC worker Release error: %s", exc)


def _kill_own_process_group_if_isolated(config: dict) -> None:
    if not config.get("process_group_isolated"):
        logger.error(
            "ARTC worker process group is not isolated; refusing to kill process group"
        )
        return
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception as exc:
        logger.error("ARTC worker failed to kill process group: %s", exc)


def _artc_worker_main(config: dict, command_fd: int, status_fd: int) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )

    handler = _WorkerEventHandler(
        status_fd,
        timeline_path=config.get("timeline_path"),
        session_id=config.get("session_id"),
        channel=config.get("channel"),
        pusher_userid=config.get("userid"),
        dual_stream_enabled=config.get("dual_stream_enabled"),
        low_stream_profile=config.get("low_stream_profile"),
    )
    engine = None
    try:
        engine = _create_engine(config, handler)
        _configure_engine(engine, config)
        _join_channel(engine, handler, config)
        _put_status(status_fd, "ready", config["channel"])
        _drain_worker_queue(engine, handler, config, command_fd)
    except Exception as exc:
        logger.exception("ARTC worker failed for channel=%s", config["channel"])
        emit_chunk_timeline(
            config.get("timeline_path"),
            "artc_error",
            session_id=config.get("session_id"),
            channel=config.get("channel"),
            pusher_userid=config.get("userid"),
            wall_clock=time.time(),
            error_message=str(exc),
            message=str(exc),
        )
        emit_chunk_timeline(
            config.get("timeline_path"),
            "artc_push_error",
            session_id=config.get("session_id"),
            channel=config.get("channel"),
            pusher_userid=config.get("userid"),
            wall_clock=time.time(),
            error_message=str(exc),
            message=str(exc),
        )
        emit_chunk_timeline(
            config.get("timeline_path"),
            "error",
            component="artc",
            session_id=config.get("session_id"),
            channel=config.get("channel"),
            pusher_userid=config.get("userid"),
            wall_clock=time.time(),
            error_message=str(exc),
            message=str(exc),
        )
        _put_status(status_fd, "failed", str(exc))
    finally:
        _leave_and_release(engine, handler, config)
        _put_status(status_fd, "stopped", config["channel"])


def _artc_worker_subprocess_main(command_fd: int, status_fd: int) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )

    process_group_isolated = False
    try:
        os.setsid()
        process_group_isolated = True
    except Exception:
        pass

    try:
        config = _recv_message(command_fd, timeout=10.0)
        config["process_group_isolated"] = process_group_isolated
        _artc_worker_main(config, command_fd, status_fd)
    finally:
        for fd in (command_fd, status_fd):
            try:
                os.close(fd)
            except Exception:
                pass


class ArtcPusher:
    """Push video/audio via AliRTC SDK (ARTC protocol).

    Duck-typing compatible with the old StreamPusher interface:
    - ``start()`` / ``push_chunk(frames_np, audio_16k)`` / ``stop()``
    - ``.failed`` property
    - ``._started`` attribute (checked by pipeline for lazy-start)
    """

    def __init__(
        self,
        artc_token: str,
        artc_channel: str,
        artc_userid: str = "sglang",
        width: int = 448,
        height: int = 448,
        fps: int = 25,
        queue_maxsize: int = 8,
        sdk_path: Optional[str] = None,
        timeline_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self._token = artc_token
        self._channel = artc_channel
        self._userid = artc_userid
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_maxsize = queue_maxsize
        self._sdk_path = sdk_path or _SDK_DIR
        self._timeline_path = timeline_path
        self._session_id = session_id

        self._command_queue: Optional[queue.Queue] = None
        self._command_write_fd: Optional[int] = None
        self._status_read_fd: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._sender_thread: Optional[threading.Thread] = None

        self._started = False
        self._failed = False
        self._start_thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._start_done = threading.Event()
        self._start_requested = False
        self._stop_requested = threading.Event()
        self._last_enqueued_real_turn_id: Optional[str] = None
        self._last_enqueued_was_filler = False
        self._real_chunk_enqueued = False
        self._real_playout_until_monotonic = 0.0

    @property
    def failed(self) -> bool:
        self._poll_status()
        return self._failed

    @property
    def start_requested(self) -> bool:
        return self._start_requested

    @property
    def start_in_progress(self) -> bool:
        return self._start_requested and not self._start_done.is_set()

    def wait_until_started(self, timeout: float | None = None) -> bool:
        """Wait for async startup to finish."""
        if self._started:
            return True
        if not self._start_requested:
            return False
        finished = self._start_done.wait(timeout=timeout)
        self._poll_status()
        return finished and self._started and not self._failed

    def start(self) -> None:
        """Start the dedicated AliRTC worker process and wait until it joins."""
        if self._started:
            self._start_done.set()
            return

        with self._start_lock:
            if self._started:
                self._start_done.set()
                return

            already_requested = self._start_requested
            self._start_requested = True
            if not already_requested:
                self._stop_requested.clear()
            self._start_done.clear()
            self._failed = False

            self._ensure_queues()
            process = self._process
            if process is not None and process.poll() is not None:
                self._close_queues()
                self._ensure_queues()
                process = None
            if process is None or process.poll() is not None:
                try:
                    self._start_subprocess_worker()
                except Exception as exc:
                    logger.error("ARTC worker subprocess start failed: %s", exc)
                    self._failed = True
                    self._terminate_worker(timeout=2.0)
                    self._start_requested = False
                    self._start_done.set()
                    self._close_queues()
                    return
                process = self._process

            start_timeout = float(os.environ.get("SGLANG_ARTC_START_TIMEOUT", "15.0"))
            deadline = time.monotonic() + start_timeout
            t0 = time.perf_counter()
            while time.monotonic() < deadline:
                if self._started:
                    self._start_done.set()
                    return
                if self._failed:
                    self._terminate_worker(timeout=2.0)
                    self._start_requested = False
                    self._start_done.set()
                    return
                if self._stop_requested.is_set():
                    self._send_stop()
                    self._terminate_worker(timeout=2.0)
                    self._start_requested = False
                    self._start_done.set()
                    return

                event = self._get_status_event(timeout=0.1)
                if event is None:
                    if process.poll() is not None:
                        self._failed = True
                        logger.error(
                            "ARTC worker exited before ready: channel=%s exitcode=%s",
                            self._channel,
                            process.returncode,
                        )
                        self._start_requested = False
                        self._start_done.set()
                        return
                    continue

                kind, message = event
                if kind == "ready":
                    self._started = True
                    self._start_done.set()
                    self._ensure_sender_thread()
                    logger.info(
                        "ARTC start() finished in %.3fs for channel=%s worker_pid=%s",
                        time.perf_counter() - t0,
                        self._channel,
                        process.pid,
                    )
                    return
                if kind == "failed":
                    self._failed = True
                    self._terminate_worker(timeout=2.0)
                    logger.error("ARTC worker init failed: %s", message)
                    self._start_requested = False
                    self._start_done.set()
                    return

            self._failed = True
            self._terminate_worker(timeout=2.0)
            logger.error(
                "ARTC worker did not become ready within %.1fs for channel=%s",
                start_timeout,
                self._channel,
            )
            self._start_requested = False
            self._start_done.set()

    def start_async(self) -> None:
        """Launch worker startup in the background."""
        if self._started:
            self._start_done.set()
            return
        with self._start_lock:
            if self._started:
                self._start_done.set()
                return
            self._start_requested = True
            self._stop_requested.clear()
            self._ensure_queues()
            if self._start_thread is not None and self._start_thread.is_alive():
                return
            self._start_done.clear()
            self._start_thread = threading.Thread(
                target=self.start, daemon=True, name="artc-start"
            )
            self._start_thread.start()

    def push_chunk(
        self,
        frames_np: np.ndarray,
        audio_16k: Optional[np.ndarray] = None,
        chunk_idx: Optional[int] = None,
        audio_chunk_idx: Optional[int] = None,
        used_silence: Optional[bool] = None,
        audio_loaded: Optional[bool] = None,
        audio_prefetched: Optional[bool] = None,
        chunk_source: Optional[str] = None,
        is_filler: Optional[bool] = None,
        turn_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio_chunk_meta: Optional[dict] = None,
    ) -> None:
        """Enqueue a chunk of RGB video frames and optional 16 kHz mono audio."""
        if self.failed:
            return
        if not self._started and not self._start_requested:
            return

        audio_int16 = None
        if audio_16k is not None and len(audio_16k) > 0:
            audio_int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)

        self._ensure_queues()
        frame_count = int(frames_np.shape[0])
        duration_s = frame_count / max(self._fps, 1)
        meta = dict(audio_chunk_meta or {})
        meta.update({
            "session_id": session_id or self._session_id or meta.get("session_id"),
            "chunk_idx": chunk_idx,
            "audio_chunk_idx": audio_chunk_idx,
            "used_silence": used_silence,
            "audio_loaded": audio_loaded,
            "audio_prefetched": audio_prefetched,
            "chunk_source": chunk_source or meta.get("chunk_source"),
            "is_filler": is_filler
            if is_filler is not None
            else meta.get("is_filler"),
            "turn_id": turn_id or meta.get("turn_id"),
            "frame_count": frame_count,
            "duration_s": duration_s,
            "enqueue_monotonic_s": time.monotonic(),
        })
        item = (_CHUNK, frames_np, audio_int16, meta)

        meta_is_filler = self._meta_is_filler(meta)
        effective_turn_id = meta.get("turn_id")
        dropped_filler = 0
        dropped_stale = 0
        clear_enqueued = False

        if meta_is_filler:
            if not self._started:
                # ARTC startup can take several seconds. Keeping every prestart
                # filler chunk creates a playback backlog before the first real
                # response, so retain at most the latest placeholder.
                dropped_filler += self.drop_filler_chunks(
                    reason="prestart_latest_filler"
                )
        else:
            dropped_filler += self.drop_filler_chunks(reason="real_chunk_enqueue")
            dropped_stale += self.drop_stale_chunks(
                active_turn_id=effective_turn_id,
                reason="real_chunk_enqueue",
            )
            if self._should_clear_before_real(meta):
                clear_enqueued = self._enqueue_clear_buffer(meta)

        queue_dropped = False
        enqueued = False
        try:
            self._command_queue.put_nowait(item)
            enqueued = True
        except queue.Full:
            dropped_filler += self.drop_filler_chunks(reason="queue_full")
            if meta_is_filler:
                if dropped_filler:
                    try:
                        self._command_queue.put_nowait(item)
                        enqueued = True
                    except queue.Full:
                        pass
                logger.warning(
                    "ARTC pusher queue full, dropped filler chunk=%s source=%s",
                    chunk_idx,
                    chunk_source,
                )
            else:
                queue_dropped = bool(self.drop_oldest_chunk(reason="queue_full"))
                try:
                    self._command_queue.put_nowait(item)
                    enqueued = True
                except queue.Full:
                    pass
                logger.warning("ARTC pusher queue full, dropped oldest chunk")
        if meta_is_filler:
            self._last_enqueued_was_filler = True
        elif enqueued:
            self._last_enqueued_was_filler = False
            self._real_chunk_enqueued = True
            now = time.monotonic()
            self._real_playout_until_monotonic = (
                max(now, self._real_playout_until_monotonic) + duration_s
            )
            if effective_turn_id:
                self._last_enqueued_real_turn_id = effective_turn_id
        current_queue_stats = self.queue_stats()
        if self._command_queue is not None:
            meta["queue_size"] = self._command_queue.qsize()
        if current_queue_stats.get("video_queue_ms") is not None:
            meta["video_queue_ms"] = current_queue_stats.get("video_queue_ms")
        if current_queue_stats.get("artc_pending_filler_ms") is not None:
            meta["pending_filler_ms"] = current_queue_stats.get(
                "artc_pending_filler_ms"
            )
        if current_queue_stats.get("pending_frames") is not None:
            meta["pending_frames"] = current_queue_stats.get("pending_frames")
        meta["pusher_userid"] = self._userid
        emit_chunk_timeline(
            self._timeline_path,
            "artc_chunk_enqueued",
            **compact_chunk_trace_fields(
                session_id=session_id or self._session_id,
                meta=meta,
                chunk_idx=chunk_idx,
                audio_chunk_idx=audio_chunk_idx,
                queue_size=self._command_queue.qsize()
                if self._command_queue
                else None,
                video_queue_ms=current_queue_stats.get("video_queue_ms"),
            ),
            used_silence=used_silence,
            audio_loaded=audio_loaded,
            audio_prefetched=audio_prefetched,
            frame_count=frame_count,
            audio_samples=int(len(audio_int16)) if audio_int16 is not None else 0,
            queue_dropped=queue_dropped,
            dropped_filler=dropped_filler,
            dropped_stale=dropped_stale,
            clear_enqueued=clear_enqueued,
            enqueued=enqueued,
            pusher_started=bool(self._started),
            real_playout_ahead_ms=round(
                max(0.0, self._real_playout_until_monotonic - time.monotonic())
                * 1000,
                1,
            )
            if not meta_is_filler and enqueued
            else None,
        )

    @staticmethod
    def _meta_is_filler(meta: dict) -> bool:
        explicit = meta.get("is_filler")
        if explicit is not None:
            if isinstance(explicit, str):
                return explicit.strip().lower() in ("1", "true", "yes", "on")
            return bool(explicit)
        return bool(meta.get("used_silence")) or not bool(meta.get("audio_loaded"))

    def _item_is_filler(self, item) -> bool:
        if not isinstance(item, tuple) or len(item) < 4 or item[0] != _CHUNK:
            return False
        meta = item[3] if isinstance(item[3], dict) else {}
        return self._meta_is_filler(meta)

    @staticmethod
    def _item_meta(item) -> dict:
        if not isinstance(item, tuple):
            return {}
        if item and item[0] == _CHUNK and len(item) > 3 and isinstance(item[3], dict):
            return item[3]
        if item and item[0] == _CLEAR_BUFFER and len(item) > 1 and isinstance(item[1], dict):
            return item[1]
        return {}

    def queue_stats(self) -> dict[str, float | int | None]:
        if self._command_queue is None:
            return {"video_queue_ms": None}
        items = list(self._command_queue.queue)
        video_queue_ms = 0.0
        pending_filler_ms = 0.0
        pending_frames = 0
        for item in items:
            if not isinstance(item, tuple) or not item or item[0] != _CHUNK:
                continue
            meta = self._item_meta(item)
            duration_ms = float(meta.get("duration_s") or 0.0) * 1000.0
            video_queue_ms += duration_ms
            try:
                pending_frames += int(meta.get("frame_count") or 0)
            except (TypeError, ValueError):
                pass
            if self._meta_is_filler(meta):
                pending_filler_ms += duration_ms
        return {
            "artc_queue_size": self._command_queue.qsize(),
            "video_queue_ms": round(video_queue_ms, 3),
            "artc_pending_filler_ms": round(pending_filler_ms, 3),
            "pending_frames": pending_frames,
        }

    def _should_clear_before_real(self, meta: dict) -> bool:
        turn_id = meta.get("turn_id")
        # ClearDataBuffer drops media that has reached the ARTC SDK but has not
        # yet played.  Never clear while real digital-human speech is expected
        # to still be buffered; doing so is visible as skipped video/mouth frames.
        if self._real_chunk_enqueued:
            real_ahead_s = self._real_playout_until_monotonic - time.monotonic()
            try:
                guard_s = float(
                    os.environ.get("SGLANG_ARTC_CLEAR_REAL_GUARD_S", "0.25")
                )
            except ValueError:
                guard_s = 0.25
            guard_s = max(0.0, guard_s)
            if real_ahead_s > guard_s:
                emit_chunk_timeline(
                    self._timeline_path,
                    "artc_clear_buffer_suppressed",
                    chunk_idx=meta.get("chunk_idx"),
                    audio_chunk_idx=meta.get("audio_chunk_idx"),
                    turn_id=turn_id,
                    reason="real_playout_active",
                    real_playout_ahead_ms=round(real_ahead_s * 1000, 1),
                    guard_ms=round(guard_s * 1000, 1),
                )
                return False
        if self._last_enqueued_was_filler:
            return True
        if turn_id and turn_id != self._last_enqueued_real_turn_id:
            return True
        if self._last_enqueued_real_turn_id is None and turn_id:
            return True
        return False

    def _enqueue_clear_buffer(self, meta: dict) -> bool:
        if self._command_queue is None:
            return False
        clear_meta = {
            "chunk_idx": meta.get("chunk_idx"),
            "audio_chunk_idx": meta.get("audio_chunk_idx"),
            "turn_id": meta.get("turn_id"),
            "reason": "real_turn_boundary",
            "enqueue_monotonic_s": time.monotonic(),
        }
        item = (_CLEAR_BUFFER, clear_meta)
        enqueued = self._put_control_item(item)
        emit_chunk_timeline(
            self._timeline_path,
            "artc_clear_buffer_enqueued",
            chunk_idx=clear_meta["chunk_idx"],
            audio_chunk_idx=clear_meta["audio_chunk_idx"],
            turn_id=clear_meta["turn_id"],
            reason=clear_meta["reason"],
            enqueued=enqueued,
            queue_size=self._command_queue.qsize()
            if self._command_queue is not None
            else None,
        )
        return enqueued

    def _put_control_item(self, item) -> bool:
        if self._command_queue is None:
            return False
        try:
            self._command_queue.put_nowait(item)
            return True
        except queue.Full:
            self.drop_filler_chunks(reason="control_queue_full")
            self.drop_oldest_chunk(reason="control_queue_full")
            try:
                self._command_queue.put_nowait(item)
                return True
            except queue.Full:
                return False

    def drop_oldest_chunk(self, reason: str = "") -> int:
        if self._command_queue is None:
            return 0

        kept = []
        dropped = 0
        while True:
            try:
                item = self._command_queue.get_nowait()
            except queue.Empty:
                break
            if dropped == 0 and isinstance(item, tuple) and item and item[0] == _CHUNK:
                dropped = 1
                continue
            kept.append(item)
        for item in kept:
            try:
                self._command_queue.put_nowait(item)
            except queue.Full:
                break
        if dropped:
            emit_chunk_timeline(
                self._timeline_path,
                "artc_oldest_chunk_dropped",
                reason=reason,
                queue_size=self._command_queue.qsize()
                if self._command_queue is not None
                else None,
            )
        return dropped

    def drop_filler_chunks_with_stats(
        self,
        reason: str = "",
    ) -> dict[str, Optional[float] | int]:
        """Drop queued filler chunks that have not been sent to the worker."""
        stats = {
            "dropped_filler_chunks": 0,
            "dropped_filler_audio_ms": 0.0,
            "dropped_filler_video_frames": 0,
            "first_dropped_pts": None,
            "last_dropped_pts": None,
        }
        if self._command_queue is None:
            return stats

        kept = []
        while True:
            try:
                item = self._command_queue.get_nowait()
            except queue.Empty:
                break
            if self._item_is_filler(item):
                meta = self._item_meta(item)
                stats["dropped_filler_chunks"] += 1
                duration_ms = float(meta.get("duration_s") or 0.0) * 1000.0
                stats["dropped_filler_audio_ms"] += duration_ms
                try:
                    frame_count = int(meta.get("frame_count") or 0)
                except (TypeError, ValueError):
                    frame_count = 0
                stats["dropped_filler_video_frames"] += frame_count
                first_pts = meta.get("first_audio_pts")
                if first_pts is None:
                    try:
                        chunk_idx = int(meta.get("chunk_idx"))
                        first_pts = chunk_idx * duration_ms
                    except (TypeError, ValueError):
                        first_pts = None
                last_pts = meta.get("last_audio_pts")
                if last_pts is None and first_pts is not None:
                    last_pts = float(first_pts) + max(
                        0.0,
                        duration_ms - 1000.0 / max(self._fps, 1),
                    )
                if first_pts is not None and stats["first_dropped_pts"] is None:
                    stats["first_dropped_pts"] = round(float(first_pts), 3)
                if last_pts is not None:
                    stats["last_dropped_pts"] = round(float(last_pts), 3)
            else:
                kept.append(item)
        for item in kept:
            try:
                self._command_queue.put_nowait(item)
            except queue.Full:
                break
        if stats["dropped_filler_chunks"]:
            stats["dropped_filler_audio_ms"] = round(
                float(stats["dropped_filler_audio_ms"]), 3
            )
            emit_chunk_timeline(
                self._timeline_path,
                "artc_filler_chunks_dropped",
                dropped_chunks=stats["dropped_filler_chunks"],
                **stats,
                reason=reason,
                queue_size=self._command_queue.qsize()
                if self._command_queue is not None
                else None,
            )
            logger.info(
                "ARTC pusher dropped %d queued filler chunks reason=%s",
                stats["dropped_filler_chunks"],
                reason,
            )
        return stats

    def drop_filler_chunks(self, reason: str = "") -> int:
        """Drop queued filler chunks that have not been sent to the worker."""
        stats = self.drop_filler_chunks_with_stats(reason=reason)
        return int(stats.get("dropped_filler_chunks") or 0)

    def drop_stale_chunks(self, active_turn_id: Optional[str], reason: str = "") -> int:
        """Drop queued filler from older turns before a new real turn starts.

        Real chunks are preserved even when their turn id is older. Dropping
        them causes visible skips in the digital-human speech; a small backlog is
        preferable to swallowing already generated mouth/video frames.
        """
        if self._command_queue is None or not active_turn_id:
            return 0

        kept = []
        dropped = 0
        kept_stale_real = 0
        while True:
            try:
                item = self._command_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item and item[0] == _CHUNK:
                meta = self._item_meta(item)
                item_turn_id = meta.get("turn_id")
                if self._meta_is_filler(meta):
                    dropped += 1
                    continue
                if item_turn_id and item_turn_id != active_turn_id:
                    kept_stale_real += 1
            kept.append(item)
        for item in kept:
            try:
                self._command_queue.put_nowait(item)
            except queue.Full:
                break
        if dropped:
            emit_chunk_timeline(
                self._timeline_path,
                "artc_stale_chunks_dropped",
                dropped_chunks=dropped,
                active_turn_id=active_turn_id,
                reason=reason,
                queue_size=self._command_queue.qsize()
                if self._command_queue is not None
                else None,
            )
            logger.info(
                "ARTC pusher dropped %d stale chunks for turn=%s reason=%s",
                dropped,
                active_turn_id,
                reason,
            )
        if kept_stale_real:
            emit_chunk_timeline(
                self._timeline_path,
                "artc_stale_real_chunks_preserved",
                kept_chunks=kept_stale_real,
                active_turn_id=active_turn_id,
                reason=reason,
                queue_size=self._command_queue.qsize()
                if self._command_queue is not None
                else None,
            )
            logger.info(
                "ARTC pusher preserved %d stale real chunks for turn=%s reason=%s",
                kept_stale_real,
                active_turn_id,
                reason,
            )
        return dropped

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker process; kill its process group on timeout."""
        if not self._started and not self._start_requested and self._process is None:
            return

        self._stop_requested.set()
        self._send_stop()

        if self._start_thread is not None and self._start_thread.is_alive():
            self._start_thread.join(timeout=timeout)
            if self._start_thread.is_alive():
                logger.warning("ARTC start thread did not exit within %.1fs", timeout)

        process = self._process
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ARTC worker did not exit within %.1fs for channel=%s",
                    timeout,
                    self._channel,
                )
                self._terminate_worker(timeout=2.0)

        self._poll_status()
        self._started = False
        self._start_requested = False
        self._start_done.set()
        self._close_queues()

    def _worker_config(self) -> dict:
        avsync_mode = os.environ.get("SGLANG_ARTC_AVSYNC_MODE", "nodelay").strip()
        frame_pacing_env = os.environ.get("SGLANG_ARTC_FRAME_PACING")
        if frame_pacing_env is None:
            frame_pacing = avsync_mode.lower() not in ("pts", "with_pts", "withpts")
        else:
            frame_pacing = frame_pacing_env.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        return {
            "token": self._token,
            "channel": self._channel,
            "userid": self._userid,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "queue_maxsize": self._queue_maxsize,
            "sdk_path": self._sdk_path,
            "timeline_path": self._timeline_path,
            "session_id": self._session_id,
            "avsync_mode": avsync_mode,
            "frame_pacing": frame_pacing,
            "dual_stream_enabled": os.environ.get(
                "SGLANG_ARTC_DUAL_STREAM_ENABLED", "0"
            )
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
            "low_stream_profile": os.environ.get(
                "SGLANG_ARTC_LOW_STREAM_PROFILE", ""
            ).strip()
            or None,
            "reset_pts_on_clear": os.environ.get(
                "SGLANG_ARTC_RESET_PTS_ON_CLEAR", "1"
            )
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
            "join_timeout": float(os.environ.get("SGLANG_ARTC_JOIN_TIMEOUT", "10.0")),
            "leave_timeout": float(os.environ.get("SGLANG_ARTC_LEAVE_TIMEOUT", "2.0")),
        }

    def _ensure_queues(self) -> None:
        if self._command_queue is None:
            self._command_queue = queue.Queue(maxsize=self._queue_maxsize)

    def _start_subprocess_worker(self) -> None:
        command_read_fd, self._command_write_fd = os.pipe()
        self._status_read_fd, status_write_fd = os.pipe()

        self._process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--artc-worker",
                str(command_read_fd),
                str(status_write_fd),
            ],
            stdin=subprocess.DEVNULL,
            pass_fds=(command_read_fd, status_write_fd),
            close_fds=True,
        )
        os.close(command_read_fd)
        os.close(status_write_fd)
        _send_message(self._command_write_fd, self._worker_config())

    def _ensure_sender_thread(self) -> None:
        if self._command_queue is None or self._command_write_fd is None:
            return
        if self._sender_thread is not None and self._sender_thread.is_alive():
            return
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="artc-sender",
        )
        self._sender_thread.start()

    def _send_stop(self) -> None:
        if self._command_queue is None:
            return
        try:
            self._command_queue.put_nowait((_STOP,))
        except queue.Full:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._command_queue.put_nowait((_STOP,))
            except queue.Full:
                pass

    def _get_status_event(self, timeout: float):
        if self._status_read_fd is None:
            return None
        try:
            return _recv_message(self._status_read_fd, timeout=timeout)
        except queue.Empty:
            return None
        except _IpcTimeout:
            return None
        except EOFError:
            return ("failed", "ARTC worker status pipe closed")

    def _poll_status(self) -> None:
        if self._status_read_fd is not None:
            while True:
                readable, _, _ = select.select([self._status_read_fd], [], [], 0)
                if not readable:
                    break
                try:
                    kind, message = _recv_message(self._status_read_fd, timeout=0.1)
                except _IpcTimeout:
                    break
                except EOFError:
                    if self._started and not self._stop_requested.is_set():
                        self._failed = True
                        logger.error("ARTC worker status pipe closed")
                    break
                if kind == "failed":
                    self._failed = True
                    logger.error("ARTC worker reported failure: %s", message)
                elif kind == "ready":
                    self._started = True
                    self._start_done.set()
                    self._ensure_sender_thread()
                elif kind == "stopped":
                    self._started = False

        process = self._process
        if (
            process is not None
            and process.poll() is not None
            and self._started
            and not self._stop_requested.is_set()
        ):
            self._failed = True
            logger.error(
                "ARTC worker exited unexpectedly: channel=%s exitcode=%s",
                self._channel,
                process.returncode,
            )

    def _terminate_worker(self, timeout: float) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    def _sender_loop(self) -> None:
        while True:
            try:
                item = self._command_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_requested.is_set():
                    return
                continue

            command_fd = self._command_write_fd
            if command_fd is None:
                return
            try:
                _send_message(command_fd, item)
            except Exception:
                if item[0] != _STOP:
                    self._failed = True
                return
            if item[0] == _STOP:
                return
            meta = self._item_meta(item)
            if item[0] == _CLEAR_BUFFER:
                emit_chunk_timeline(
                    self._timeline_path,
                    "artc_clear_buffer_ipc_sent",
                    **compact_chunk_trace_fields(
                        session_id=self._session_id,
                        meta=meta,
                        chunk_idx=meta.get("chunk_idx"),
                        audio_chunk_idx=meta.get("audio_chunk_idx"),
                        queue_size=self._command_queue.qsize()
                        if self._command_queue is not None
                        else None,
                        video_queue_ms=self.queue_stats().get("video_queue_ms"),
                    ),
                    enqueue_to_ipc_ms=round(
                        (
                            time.monotonic()
                            - meta.get("enqueue_monotonic_s", time.monotonic())
                        )
                        * 1000,
                        3,
                    ),
                )
            else:
                emit_chunk_timeline(
                    self._timeline_path,
                    "artc_chunk_ipc_sent",
                    **compact_chunk_trace_fields(
                        session_id=self._session_id,
                        meta=meta,
                        chunk_idx=meta.get("chunk_idx"),
                        audio_chunk_idx=meta.get("audio_chunk_idx"),
                        queue_size=self._command_queue.qsize()
                        if self._command_queue is not None
                        else None,
                        video_queue_ms=self.queue_stats().get("video_queue_ms"),
                    ),
                    enqueue_to_ipc_ms=round(
                        (
                            time.monotonic()
                            - meta.get("enqueue_monotonic_s", time.monotonic())
                        )
                        * 1000,
                        3,
                    ),
                )

    def _close_queues(self) -> None:
        for fd in (self._command_write_fd, self._status_read_fd):
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        self._command_queue = None
        self._command_write_fd = None
        self._status_read_fd = None
        self._process = None
        self._sender_thread = None


def _main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--artc-worker":
        _artc_worker_subprocess_main(int(sys.argv[2]), int(sys.argv[3]))
        return
    raise SystemExit("artc_pusher.py is an internal worker module")


if __name__ == "__main__":
    _main()
