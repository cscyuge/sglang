"""Push video/audio via AliRTC SDK (ARTC protocol).

Provides ``ArtcPusher``, a queue-based background thread that accepts raw
RGB video frames and PCM audio, and pushes them through the AliRTC SDK.
The SDK handles H.264 encoding internally — no PyAV dependency needed.

Audio is accepted as float32 16 kHz mono and converted to int16 in-place
(no resampling required, unlike the old RTMP/SRT path).
"""

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


class _EventHandler:
    """Minimal AliRTC event handler that tracks publish-readiness and errors."""

    def __init__(self, pusher: "ArtcPusher"):
        self._pusher = pusher

    # -- publish state ---------------------------------------------------
    def OnAudioPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC audio publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        # AliEnginePublishState: Published = 2
        if getattr(newState, "value", newState) == 2:
            self._pusher._audio_published.set()

    def OnVideoPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC video publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        if getattr(newState, "value", newState) == 2:
            self._pusher._video_published.set()

    # -- buffer full flags -----------------------------------------------
    def OnPushAudioFrameBufferFull(self, isFull):
        self._pusher._push_audio_full = isFull
        if isFull:
            logger.debug("ARTC audio buffer full")

    def OnPushVideoFrameBufferFull(self, isFull):
        self._pusher._push_video_full = isFull
        if isFull:
            logger.debug("ARTC video buffer full")

    # -- error / connection ----------------------------------------------
    def OnError(self, error_code):
        logger.error("ARTC SDK error: %s", error_code)
        self._pusher._failed = True

    def OnConnectionStatusChanged(self, status, reason):
        logger.info("ARTC connection: status=%s reason=%s", status, reason)

    def OnJoinChannelResult(self, result, channel, userId):
        logger.info(
            "ARTC JoinChannel result=%s channel=%s user=%s",
            result, channel, userId,
        )
        if result == 0:
            self._pusher._joined.set()
        else:
            logger.error("ARTC JoinChannel failed: %s", result)
            self._pusher._failed = True

    def OnLeaveChannelResult(self, result):
        logger.info("ARTC LeaveChannel result=%s", result)

    # Catch-all for callbacks we don't handle
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            pass
        return _noop


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
        self._engine = None
        self._sdk_path = sdk_path or _SDK_DIR

        # Synchronisation events
        self._joined = threading.Event()
        self._audio_published = threading.Event()
        self._video_published = threading.Event()
        self._push_video_full = False
        self._push_audio_full = False

        # Monotonic PTS counters (milliseconds)
        self._v_ts = 0
        self._a_ts = 0

    @property
    def failed(self) -> bool:
        return self._failed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create engine, join channel, and start the drain thread."""
        if self._started:
            return

        try:
            self._init_engine()
        except Exception as exc:
            logger.error("ARTC engine init failed: %s", exc)
            self._failed = True
            return

        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="artc-push"
        )
        self._thread.start()
        self._started = True

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
        if self._failed or not self._started:
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
        """Drain queue, leave channel, release engine."""
        if not self._started:
            return

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

        self._cleanup_engine()
        self._started = False

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        """Initialise AliRTC engine and join the channel."""
        # Ensure SDK Python modules are importable
        sdk_py = self._sdk_path
        if sdk_py not in sys.path:
            sys.path.insert(0, sdk_py)

        # Ensure native libs are on LD_LIBRARY_PATH / can be found
        lib_dir = os.path.join(self._sdk_path, "Release", "lib")
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + ld_path

        core_service = os.path.join(lib_dir, "AliRtcCoreService")

        from AliRTCEngine import CreateAliRTCEngine  # noqa: E402
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

        handler = _EventHandler(self)
        log_path = "/tmp/artc_sdk_logs"
        os.makedirs(log_path, exist_ok=True)

        self._engine = CreateAliRTCEngine(
            eventHandler=handler,
            lowPort=40000,
            highPort=40100,
            logPath=log_path,
            coreServicePath=core_service,
            h5mode=False,
            extra="{}",
        )

        # Configure video encoder — constructor requires all positional args
        # Map fps int to SDK enum (default to 25fps if no exact match)
        _fps_map = {
            5: AliEngineFrameRate.AliEngineFrameRateFps5,
            10: AliEngineFrameRate.AliEngineFrameRateFps10,
            15: AliEngineFrameRate.AliEngineFrameRateFps15,
            20: AliEngineFrameRate.AliEngineFrameRateFps20,
            25: AliEngineFrameRate.AliEngineFrameRateFps25,
            30: AliEngineFrameRate.AliEngineFrameRateFps30,
            60: AliEngineFrameRate.AliEngineFrameRateFps60,
        }
        frame_rate_enum = _fps_map.get(
            self._fps, AliEngineFrameRate.AliEngineFrameRateFps25
        )
        video_cfg = AliEngineVideoEncoderConfiguration(
            width=self._width,
            height=self._height,
            f=frame_rate_enum,
            b=2000,
            ori=AliEngineVideoEncoderOrientationMode.AliEngineVideoEncoderOrientationModeAdaptive,
            mr=AliEngineVideoMirrorMode.AliEngineVideoMirrorModeDisabled,
            rotation=AliEngineRotationMode.AliEngineRotationMode_0,
        )
        self._engine.SetVideoEncoderConfiguration(video_cfg)

        # External video/audio sources
        self._engine.SetExternalVideoSource(
            True, VideoSource.VideoSourceCamera, RenderMode.RenderModeFill
        )
        self._engine.SetExternalAudioSource(True, 16000, 1)

        # Publish streams
        self._engine.PublishLocalVideoStream(True)
        self._engine.PublishLocalAudioStream(True)

        # Set interactive role (broadcaster)
        self._engine.SetClientRole(AliEngineClientRole.AliEngineClientRoleInteractive)

        # Join channel
        join_cfg = JoinChannelConfig()
        join_cfg.publishAvsyncMode = PublishAvsyncMode.PublishAvsyncWithPts
        join_cfg.publishMode = PublishMode.PublishAutomatically
        self._engine.JoinChannel(
            self._token, self._channel, self._userid, self._userid, join_cfg
        )

        # Wait for join + publish readiness
        if not self._joined.wait(timeout=10.0):
            raise RuntimeError("ARTC JoinChannel timed out")
        if self._failed:
            raise RuntimeError("ARTC JoinChannel failed")

        # Give publish callbacks a moment (they may arrive shortly after join)
        self._audio_published.wait(timeout=5.0)
        self._video_published.wait(timeout=5.0)
        logger.info(
            "ARTC engine ready: channel=%s user=%s %dx%d@%dfps",
            self._channel, self._userid, self._width, self._height, self._fps,
        )

    def _cleanup_engine(self) -> None:
        """Leave channel and release engine resources."""
        if self._engine is None:
            return
        try:
            self._engine.LeaveChannel()
        except Exception as exc:
            logger.warning("ARTC LeaveChannel error: %s", exc)
        try:
            self._engine.Release()
        except Exception as exc:
            logger.warning("ARTC Release error: %s", exc)
        self._engine = None

    # ------------------------------------------------------------------
    # Background drain thread
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        """Drain the queue and push frames/audio to AliRTC SDK."""
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
