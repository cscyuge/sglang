"""Push video/audio via AliRTC SDK (ARTC protocol).

Provides ``ArtcPusher``, a queue-based background thread that accepts raw
RGB video frames and PCM audio, and pushes them through the AliRTC SDK.
The SDK handles H.264 encoding internally — no PyAV dependency needed.

Audio is accepted as float32 16 kHz mono and converted to int16 in-place
(no resampling required, unlike the old RTMP/SRT path).
"""

import atexit
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# AliRTC SDK root — sibling ``alirtc/`` package
_SDK_DIR = os.path.join(os.path.dirname(__file__), "alirtc")
_SDK_LIB_DIR = os.path.join(_SDK_DIR, "Release", "lib")


class _ArtcStartCancelled(RuntimeError):
    pass


def _ensure_sdk_importable(sdk_path: str) -> str:
    """Make AliRTC Python and native libraries discoverable."""
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    lib_dir = os.path.join(sdk_path, "Release", "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + ld_path
    return lib_dir


class _ReusableEventHandler:
    """AliRTC event handler whose callbacks are routed to the active pusher."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pusher: Optional["ArtcPusher"] = None

    def set_pusher(self, pusher: Optional["ArtcPusher"]) -> None:
        with self._lock:
            self._pusher = pusher

    def _get_pusher(self) -> Optional["ArtcPusher"]:
        with self._lock:
            return self._pusher

    def OnAudioPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC audio publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        pusher = self._get_pusher()
        if pusher is not None and getattr(newState, "value", newState) == 2:
            pusher._audio_published.set()

    def OnVideoPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC video publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        pusher = self._get_pusher()
        if pusher is not None and getattr(newState, "value", newState) == 2:
            pusher._video_published.set()

    def OnPushAudioFrameBufferFull(self, isFull):
        pusher = self._get_pusher()
        if pusher is not None:
            pusher._push_audio_full = isFull
        if isFull:
            logger.debug("ARTC audio buffer full")

    def OnPushVideoFrameBufferFull(self, isFull):
        pusher = self._get_pusher()
        if pusher is not None:
            pusher._push_video_full = isFull
        if isFull:
            logger.debug("ARTC video buffer full")

    def OnError(self, error_code):
        logger.error("ARTC SDK error: %s", error_code)
        pusher = self._get_pusher()
        if pusher is not None:
            pusher._failed = True

    def OnConnectionStatusChanged(self, status, reason):
        logger.info("ARTC connection: status=%s reason=%s", status, reason)

    def OnJoinChannelResult(self, result, channel, userId):
        logger.info(
            "ARTC JoinChannel result=%s channel=%s user=%s",
            result, channel, userId,
        )
        pusher = self._get_pusher()
        if pusher is None:
            return
        if result == 0:
            pusher._joined.set()
        else:
            logger.error("ARTC JoinChannel failed: %s", result)
            pusher._failed = True
            pusher._joined.set()

    def OnLeaveChannelResult(self, result):
        logger.info("ARTC LeaveChannel result=%s", result)
        pusher = self._get_pusher()
        if pusher is not None:
            pusher._left.set()

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            pass
        return _noop


class _ArtcEngineManager:
    """Process-local single-engine ARTC manager.

    The AliRTC SDK startup path can degrade when repeatedly creating and
    releasing engines. This manager serializes sessions and reuses one engine
    across channels, while still leaving the channel at session boundaries.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._engine = None
        self._handler = _ReusableEventHandler()
        self._active_owner: Optional["ArtcPusher"] = None
        self._sdk_path: Optional[str] = None
        self._sdk_defs = None
        self._create_count = 0
        self._session_count = 0

    def start_session(self, owner: "ArtcPusher"):
        """Create/reuse the engine, configure it, and join owner's channel."""
        t_wait = time.perf_counter()
        with self._cond:
            while self._active_owner is not None and self._active_owner is not owner:
                if owner._stop_requested.is_set():
                    raise _ArtcStartCancelled(
                        "ARTC start cancelled before engine acquisition"
                    )
                self._cond.wait(timeout=1.0)
            if owner._stop_requested.is_set():
                raise _ArtcStartCancelled(
                    "ARTC start cancelled before engine acquisition"
                )
            wait_s = time.perf_counter() - t_wait
            if wait_s > 0.01:
                logger.info("ARTC engine manager waited %.3fs for previous session", wait_s)
            self._active_owner = owner
            self._handler.set_pusher(owner)

        try:
            t0 = time.perf_counter()
            engine = self._get_or_create_engine(owner._sdk_path)
            create_or_reuse_s = time.perf_counter() - t0
            self._handler.set_pusher(owner)

            owner._reset_session_events()
            owner._engine = engine

            t_cfg = time.perf_counter()
            self._configure_engine(owner)
            config_s = time.perf_counter() - t_cfg

            t_join = time.perf_counter()
            self._join_channel(owner)
            join_s = time.perf_counter() - t_join

            with self._cond:
                self._session_count += 1
                session_count = self._session_count
                create_count = self._create_count
            logger.info(
                "ARTC engine session ready: channel=%s reused=%s sessions=%d creates=%d "
                "engine=%.3fs config=%.3fs join=%.3fs",
                owner._channel,
                create_count < session_count,
                session_count,
                create_count,
                create_or_reuse_s,
                config_s,
                join_s,
            )
            return engine
        except Exception:
            self.end_session(owner, release_on_error=True)
            raise

    def end_session(
        self,
        owner: "ArtcPusher",
        timeout: float = 2.0,
        release_on_error: bool = False,
    ) -> None:
        """Leave the current channel and make the reusable engine available."""
        with self._cond:
            if self._active_owner is not owner:
                return
            engine = self._engine
            handler = self._handler

        left_completed = False
        if engine is not None:
            owner._left.clear()
            try:
                engine.LeaveChannel()
                left_completed = owner._left.wait(timeout=timeout)
                if not left_completed:
                    logger.warning(
                        "ARTC LeaveChannel did not complete within %.1fs for channel=%s",
                        timeout,
                        owner._channel,
                    )
                    release_on_error = True
            except Exception as exc:
                logger.warning("ARTC LeaveChannel error: %s", exc)
                release_on_error = True

        with self._cond:
            if release_on_error and self._engine is not None:
                self._retire_engine_locked(
                    engine=engine,
                    handler=handler,
                    owner=owner,
                    left_completed=left_completed,
                )
            else:
                self._handler.set_pusher(None)
            if self._active_owner is owner:
                self._active_owner = None
            self._cond.notify_all()
        owner._engine = None

    def release(self) -> None:
        with self._cond:
            self._handler.set_pusher(None)
            self._active_owner = None
            self._release_engine_locked()
            self._cond.notify_all()

    def _get_or_create_engine(self, sdk_path: str):
        with self._cond:
            if self._engine is not None and self._sdk_path == sdk_path:
                return self._engine
            if self._engine is not None:
                logger.info("ARTC SDK path changed; releasing reusable engine")
                self._release_engine_locked()

            lib_dir = _ensure_sdk_importable(sdk_path)
            core_service = os.path.join(lib_dir, "AliRtcCoreService")

            from AliRTCEngine import CreateAliRTCEngine  # noqa: E402

            log_path = os.environ.get("SGLANG_ARTC_LOG_PATH", "/tmp/artc_sdk_logs")
            os.makedirs(log_path, exist_ok=True)

            t0 = time.perf_counter()
            self._engine = CreateAliRTCEngine(
                eventHandler=self._handler,
                lowPort=int(os.environ.get("SGLANG_ARTC_LOW_PORT", "40000")),
                highPort=int(os.environ.get("SGLANG_ARTC_HIGH_PORT", "40100")),
                logPath=log_path,
                coreServicePath=core_service,
                h5mode=False,
                extra="{}",
            )
            self._sdk_path = sdk_path
            self._sdk_defs = None
            self._create_count += 1
            logger.info(
                "ARTC CreateAliRTCEngine finished in %.3fs (creates=%d)",
                time.perf_counter() - t0,
                self._create_count,
            )
            return self._engine

    def _defs(self, sdk_path: str):
        if self._sdk_defs is not None:
            return self._sdk_defs

        _ensure_sdk_importable(sdk_path)
        from AliRTCLinuxSdkDefine import (  # noqa: E402
            AliEngineClientRole,
            AliEngineFrameRate,
            AliEngineRotationMode,
            AliEngineVideoEncoderConfiguration,
            AliEngineVideoEncoderOrientationMode,
            AliEngineVideoMirrorMode,
            JoinChannelConfig,
            PublishAvsyncMode,
            PublishMode,
            RenderMode,
            VideoSource,
        )

        self._sdk_defs = {
            "AliEngineClientRole": AliEngineClientRole,
            "AliEngineFrameRate": AliEngineFrameRate,
            "AliEngineRotationMode": AliEngineRotationMode,
            "AliEngineVideoEncoderConfiguration": AliEngineVideoEncoderConfiguration,
            "AliEngineVideoEncoderOrientationMode": AliEngineVideoEncoderOrientationMode,
            "AliEngineVideoMirrorMode": AliEngineVideoMirrorMode,
            "JoinChannelConfig": JoinChannelConfig,
            "PublishAvsyncMode": PublishAvsyncMode,
            "PublishMode": PublishMode,
            "RenderMode": RenderMode,
            "VideoSource": VideoSource,
        }
        return self._sdk_defs

    def _configure_engine(self, owner: "ArtcPusher") -> None:
        d = self._defs(owner._sdk_path)
        frame_rate_enum = {
            5: d["AliEngineFrameRate"].AliEngineFrameRateFps5,
            10: d["AliEngineFrameRate"].AliEngineFrameRateFps10,
            15: d["AliEngineFrameRate"].AliEngineFrameRateFps15,
            20: d["AliEngineFrameRate"].AliEngineFrameRateFps20,
            25: d["AliEngineFrameRate"].AliEngineFrameRateFps25,
            30: d["AliEngineFrameRate"].AliEngineFrameRateFps30,
            60: d["AliEngineFrameRate"].AliEngineFrameRateFps60,
        }.get(owner._fps, d["AliEngineFrameRate"].AliEngineFrameRateFps25)

        video_cfg = d["AliEngineVideoEncoderConfiguration"](
            width=owner._width,
            height=owner._height,
            f=frame_rate_enum,
            b=2000,
            ori=d[
                "AliEngineVideoEncoderOrientationMode"
            ].AliEngineVideoEncoderOrientationModeAdaptive,
            mr=d["AliEngineVideoMirrorMode"].AliEngineVideoMirrorModeDisabled,
            rotation=d["AliEngineRotationMode"].AliEngineRotationMode_0,
        )
        engine = self._engine
        engine.SetVideoEncoderConfiguration(video_cfg)
        engine.SetExternalVideoSource(
            True,
            d["VideoSource"].VideoSourceCamera,
            d["RenderMode"].RenderModeFill,
        )
        engine.SetExternalAudioSource(True, 16000, 1)
        engine.PublishLocalVideoStream(True)
        engine.PublishLocalAudioStream(True)
        engine.SetClientRole(d["AliEngineClientRole"].AliEngineClientRoleInteractive)

    def _join_channel(self, owner: "ArtcPusher") -> None:
        d = self._defs(owner._sdk_path)
        join_cfg = d["JoinChannelConfig"]()
        join_cfg.publishAvsyncMode = d["PublishAvsyncMode"].PublishAvsyncWithPts
        join_cfg.publishMode = d["PublishMode"].PublishAutomatically
        self._engine.JoinChannel(
            owner._token,
            owner._channel,
            owner._userid,
            owner._userid,
            join_cfg,
        )

        if not owner._joined.wait(timeout=10.0):
            raise RuntimeError("ARTC JoinChannel timed out")
        if owner._failed:
            raise RuntimeError("ARTC JoinChannel failed")

        owner._audio_published.wait(timeout=5.0)
        owner._video_published.wait(timeout=5.0)
        logger.info(
            "ARTC engine ready: channel=%s user=%s %dx%d@%dfps",
            owner._channel,
            owner._userid,
            owner._width,
            owner._height,
            owner._fps,
        )

    def _release_engine_locked(self) -> None:
        if self._engine is None:
            return
        self._handler.set_pusher(None)
        try:
            self._engine.Release()
        except Exception as exc:
            logger.warning("ARTC Release error: %s", exc)
        self._engine = None
        self._handler = _ReusableEventHandler()
        self._sdk_path = None
        self._sdk_defs = None

    def _retire_engine_locked(
        self,
        engine,
        handler: _ReusableEventHandler,
        owner: "ArtcPusher",
        left_completed: bool,
    ) -> None:
        """Detach a suspect engine without synchronously destroying it."""
        if engine is None or self._engine is not engine:
            return

        self._engine = None
        self._handler = _ReusableEventHandler()
        self._sdk_path = None
        self._sdk_defs = None

        def _release_after_leave() -> None:
            if not left_completed and not owner._left.wait(timeout=60.0):
                logger.warning(
                    "ARTC retired engine did not leave channel=%s within 60s; "
                    "skipping Release to avoid blocking cleanup",
                    owner._channel,
                )
                handler.set_pusher(None)
                return
            handler.set_pusher(None)
            try:
                engine.Release()
            except Exception as exc:
                logger.warning("ARTC retired engine Release error: %s", exc)

        threading.Thread(
            target=_release_after_leave,
            daemon=True,
            name="artc-retired-release",
        ).start()


_ENGINE_MANAGER = _ArtcEngineManager()
atexit.register(_ENGINE_MANAGER.release)


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
    ):
        self._token = artc_token
        self._channel = artc_channel
        self._userid = artc_userid
        self._width = width
        self._height = height
        self._fps = fps

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._started = False
        self._failed = False
        self._thread: Optional[threading.Thread] = None
        self._start_thread: Optional[threading.Thread] = None
        self._engine = None
        self._sdk_path = sdk_path or _SDK_DIR
        self._start_lock = threading.Lock()
        self._start_done = threading.Event()
        self._start_requested = False
        self._stop_requested = threading.Event()

        # Synchronisation events
        self._joined = threading.Event()
        self._left = threading.Event()
        self._audio_published = threading.Event()
        self._video_published = threading.Event()
        self._push_video_full = False
        self._push_audio_full = False

        # Monotonic PTS counters (milliseconds)
        self._v_ts = 0
        self._a_ts = 0

    def _reset_session_events(self) -> None:
        self._joined.clear()
        self._left.clear()
        self._audio_published.clear()
        self._video_published.clear()
        self._push_video_full = False
        self._push_audio_full = False

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def start_requested(self) -> bool:
        return self._start_requested

    @property
    def start_in_progress(self) -> bool:
        return self._start_requested and not self._start_done.is_set()

    def wait_until_started(self, timeout: float | None = None) -> bool:
        """Wait for async startup to finish.

        Returns True if the drain thread is ready to accept frames.
        """
        if self._started:
            return True
        if not self._start_requested:
            return False
        finished = self._start_done.wait(timeout=timeout)
        return finished and self._started and not self._failed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create engine, join channel, and start the drain thread."""
        if self._started:
            self._start_done.set()
            return

        with self._start_lock:
            if self._started:
                self._start_done.set()
                return
            self._start_requested = True
            self._start_done.clear()
            if self._stop_requested.is_set():
                self._start_done.set()
                self._start_requested = False
                return
            self._failed = False
            self._v_ts = 0
            self._a_ts = 0
            t0 = time.perf_counter()
            try:
                _ENGINE_MANAGER.start_session(self)
            except _ArtcStartCancelled:
                self._start_done.set()
                self._start_requested = False
                return
            except Exception as exc:
                logger.error("ARTC engine init failed: %s", exc)
                self._failed = True
                self._start_done.set()
                self._start_requested = False
                return

            if self._stop_requested.is_set():
                _ENGINE_MANAGER.end_session(self)
                self._start_done.set()
                self._start_requested = False
                logger.info(
                    "ARTC start() finished after stop request in %.3fs for channel=%s",
                    time.perf_counter() - t0,
                    self._channel,
                )
                return

            self._thread = threading.Thread(
                target=self._drain_loop, daemon=True, name="artc-push"
            )
            self._thread.start()
            self._started = True
            self._start_done.set()
            logger.info(
                "ARTC start() finished in %.3fs for channel=%s",
                time.perf_counter() - t0,
                self._channel,
            )

    def start_async(self) -> None:
        """Launch startup in the background so chunk generation can overlap it."""
        if self._started:
            self._start_done.set()
            return
        with self._start_lock:
            if self._started:
                self._start_done.set()
                return
            self._start_requested = True
            self._stop_requested.clear()
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
    ) -> None:
        """Enqueue a chunk of video frames + audio for pushing.

        Parameters
        ----------
        frames_np : np.ndarray
            Video frames, shape ``(T, H, W, 3)`` dtype ``uint8`` (RGB24).
        audio_16k : np.ndarray or None
            Float32 mono 16 kHz PCM audio for this chunk.
        """
        if self._failed:
            return
        if not self._started and not self._start_requested:
            return

        # Convert audio float32 → int16
        audio_int16 = None
        if audio_16k is not None and len(audio_16k) > 0:
            audio_int16 = np.clip(
                audio_16k * 32767, -32768, 32767
            ).astype(np.int16)

        item = (frames_np, audio_int16)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop oldest to make room
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass
            logger.warning("ARTC pusher queue full — dropped oldest chunk")

    def stop(self, timeout: float = 10.0) -> None:
        """Drain queue and leave channel.

        If async startup is still running, this method never races it by
        releasing the engine concurrently. On timeout, the start thread observes
        ``_stop_requested`` and performs the leave once startup returns.
        """
        if not self._started and not self._start_requested:
            return

        self._stop_requested.set()

        if self._start_thread is not None and self._start_thread.is_alive():
            self._start_thread.join(timeout=timeout)
            if self._start_thread.is_alive():
                logger.warning(
                    "ARTC start thread did not exit within %.1fs", timeout
                )
                return

        release_on_error = False
        if self._started:
            # Drain the queue to make room for sentinel
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)

            if self._thread is not None:
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    logger.warning(
                        "ARTC pusher thread did not exit within %.1fs", timeout
                    )
                    self._failed = True
                    self._thread.join(timeout=1.0)
                    release_on_error = self._thread.is_alive()

        _ENGINE_MANAGER.end_session(
            self,
            timeout=min(2.0, timeout),
            release_on_error=release_on_error,
        )
        self._started = False
        self._start_requested = False
        self._start_done.set()

    # ------------------------------------------------------------------
    # Background drain thread
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        """Drain the queue and push frames/audio to AliRTC SDK.

        Uses a "skip-to-latest" strategy: before pushing a chunk, drain
        any accumulated items from the queue and keep only the newest one.
        This bounds latency to ~1 chunk (~1.12s) while still pushing all
        28 frames of each selected chunk (no intra-chunk frame drops).
        Intermediate chunks are skipped only when the queue accumulates
        (generation outpacing real-time), roughly 1 skip per 9 chunks.
        """
        try:
            from AliRTCLinuxSdkDefine import (
                VideoBufferType,
                VideoDataFormat,
                VideoDataSample,
                VideoSource,
            )
        except ImportError:
            logger.error("Failed to import AliRTC SDK defines in drain thread")
            self._failed = True
            return

        ms_per_frame = 1000 // self._fps  # 40ms @ 25fps
        samples_per_frame = 16000 // self._fps  # 640 @ 25fps

        while True:
            try:
                item = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            if item is None:
                break  # sentinel

            # --- Skip to latest when queue is near-full ---
            # Only skip when the queue is severely backed up (>= 6/8).
            # During normal catch-up (1-2 pending), push all chunks to
            # avoid dropping valid audio/video content.
            _qsize = self._queue.qsize()
            if _qsize >= self._queue.maxsize - 2:
                _skipped = 0
                _skipped_frames = 0
                while True:
                    try:
                        newer = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if newer is None:
                        # Sentinel — put it back so outer loop sees it
                        self._queue.put(None)
                        break
                    _skipped += 1
                    _skipped_frames += item[0].shape[0]
                    item = newer
                if _skipped > 0:
                    # Advance timestamps for skipped chunks to keep A/V sync
                    self._v_ts += _skipped_frames * ms_per_frame
                    self._a_ts += _skipped_frames * ms_per_frame
                    logger.info(
                        "ARTC drain: queue near-full, skipped %d chunk(s)",
                        _skipped,
                    )

            frames_np, audio_int16 = item
            num_frames = frames_np.shape[0]

            for i in range(num_frames):
                if self._failed:
                    return

                # --- Push video frame ---
                # Wait if SDK buffer is full
                while self._push_video_full:
                    time.sleep(0.001)
                    if self._failed:
                        return

                frame = frames_np[i]
                video_sample = VideoDataSample()
                video_sample.width = self._width
                video_sample.height = self._height
                video_sample.format = VideoDataFormat.VideoDataFormatRGB24
                video_sample.bufferType = VideoBufferType.VideoBufferTypeRawData
                video_sample.data = frame.tobytes()
                video_sample.dataLen = self._width * self._height * 3
                video_sample.timeStamp = self._v_ts
                video_sample.strideY = 0
                video_sample.strideU = 0
                video_sample.strideV = 0
                video_sample.rotation = 0
                self._engine.PushExternalVideoFrame(
                    video_sample, VideoSource.VideoSourceCamera
                )
                self._v_ts += ms_per_frame

                # --- Push corresponding audio slice ---
                if audio_int16 is not None:
                    while self._push_audio_full:
                        time.sleep(0.001)
                        if self._failed:
                            return

                    start = i * samples_per_frame
                    end = min(start + samples_per_frame, len(audio_int16))
                    audio_slice = audio_int16[start:end]
                    # Pad if short
                    if len(audio_slice) < samples_per_frame:
                        audio_slice = np.pad(
                            audio_slice,
                            (0, samples_per_frame - len(audio_slice)),
                        )
                    audio_bytes = audio_slice.tobytes()
                    self._engine.PushExternalAudioFrameRawData(
                        audio_bytes, len(audio_bytes), self._a_ts
                    )
                    self._a_ts += ms_per_frame
