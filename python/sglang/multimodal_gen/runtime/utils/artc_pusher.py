"""Push video/audio via AliRTC SDK (ARTC protocol).

The AliRTC Python wrapper and native CoreService run in a dedicated worker
process. The main process only owns lifecycle control and chunk delivery. This
keeps SDK hangs, wrapper thread state, and CoreService cleanup isolated from the
model-serving process.
"""

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

logger = logging.getLogger(__name__)

_SDK_DIR = os.path.join(os.path.dirname(__file__), "alirtc")
_CHUNK = "chunk"
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


class _WorkerEventHandler:
    """AliRTC callbacks scoped to one worker process and one channel."""

    def __init__(self, status_fd: int):
        self._status_fd = status_fd
        self.joined = threading.Event()
        self.left = threading.Event()
        self.audio_published = threading.Event()
        self.video_published = threading.Event()
        self.failed = False
        self.push_video_full = False
        self.push_audio_full = False

    def OnAudioPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC audio publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        if getattr(newState, "value", newState) == 2:
            self.audio_published.set()

    def OnVideoPublishStateChanged(self, oldState, newState, elapsed, channel):
        logger.debug(
            "ARTC video publish: %s -> %s (ch=%s)", oldState, newState, channel
        )
        if getattr(newState, "value", newState) == 2:
            self.video_published.set()

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
        _put_status(self._status_fd, "failed", msg)

    def OnConnectionStatusChanged(self, status, reason):
        logger.info("ARTC connection: status=%s reason=%s", status, reason)

    def OnJoinChannelResult(self, result, channel, userId):
        logger.info(
            "ARTC JoinChannel result=%s channel=%s user=%s",
            result,
            channel,
            userId,
        )
        if result != 0:
            self.failed = True
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
    join_cfg.publishAvsyncMode = PublishAvsyncMode.PublishAvsyncWithPts
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
        "ARTC worker ready: channel=%s user=%s %dx%d@%dfps",
        config["channel"],
        config["userid"],
        config["width"],
        config["height"],
        config["fps"],
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
        if item[0] != _CHUNK:
            continue

        frames_np, audio_int16 = item[1], item[2]
        num_frames = frames_np.shape[0]
        for i in range(num_frames):
            if handler.failed:
                raise RuntimeError("ARTC SDK failed during frame push")

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
            engine.PushExternalVideoFrame(video_sample, VideoSource.VideoSourceCamera)
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
                audio_bytes = audio_slice.tobytes()
                engine.PushExternalAudioFrameRawData(
                    audio_bytes, len(audio_bytes), a_ts
                )
                a_ts += ms_per_frame


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

    handler = _WorkerEventHandler(status_fd)
    engine = None
    try:
        engine = _create_engine(config, handler)
        _configure_engine(engine, config)
        _join_channel(engine, handler, config)
        _put_status(status_fd, "ready", config["channel"])
        _drain_worker_queue(engine, handler, config, command_fd)
    except Exception as exc:
        logger.exception("ARTC worker failed for channel=%s", config["channel"])
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
    ):
        self._token = artc_token
        self._channel = artc_channel
        self._userid = artc_userid
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_maxsize = queue_maxsize
        self._sdk_path = sdk_path or _SDK_DIR

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
        item = (_CHUNK, frames_np, audio_int16)
        try:
            self._command_queue.put_nowait(item)
        except queue.Full:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._command_queue.put_nowait(item)
            except queue.Full:
                pass
            logger.warning("ARTC pusher queue full, dropped oldest chunk")

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
        return {
            "token": self._token,
            "channel": self._channel,
            "userid": self._userid,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "queue_maxsize": self._queue_maxsize,
            "sdk_path": self._sdk_path,
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
