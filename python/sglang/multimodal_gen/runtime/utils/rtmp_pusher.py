"""Push-streaming via PyAV (H.264 + AAC → RTMP/FLV or SRT/MPEG-TS).

Provides ``StreamPusher``, a queue-based background thread that accepts raw
video frames (uint8 numpy) and audio samples (float32 16 kHz), encodes them
to H.264 + AAC via libav, and pushes to an RTMP or SRT ingest URL.

Supported URL schemes:
- ``rtmp://...`` — RTMP over FLV container
- ``srt://...``  — SRT over MPEG-TS container (Alibaba Cloud, etc.)
"""

import logging
import queue
import threading
from typing import Optional, Tuple
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

_INPUT_AUDIO_SR = 16000
_OUTPUT_AUDIO_SR = 48000
_AUDIO_RESAMPLE_RATIO = _OUTPUT_AUDIO_SR / _INPUT_AUDIO_SR  # 3.0


def _resample_16k_to_48k_int16(audio_f32: np.ndarray) -> np.ndarray:
    """Resample 16 kHz float32 mono audio to 48 kHz int16.

    Uses the same linear-interpolation algorithm as the fMP4 streamer in
    ``video_api.py`` (lines 940-948).
    """
    out_len = int(len(audio_f32) * _AUDIO_RESAMPLE_RATIO)
    indices = np.arange(out_len) / _AUDIO_RESAMPLE_RATIO
    left = np.floor(indices).astype(np.intp)
    np.clip(left, 0, len(audio_f32) - 1, out=left)
    right = np.minimum(left + 1, len(audio_f32) - 1)
    frac = (indices - left).astype(np.float32)
    resampled = audio_f32[left] * (1 - frac) + audio_f32[right] * frac
    return np.clip(resampled * 32767, -32768, 32767).astype(np.int16)


def _is_connection_closed(exc: BaseException) -> bool:
    """Return True if *exc* indicates a broken pipe / connection reset."""
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    # PyAV may wrap the OS error; check string as fallback
    msg = str(exc).lower()
    return "broken pipe" in msg or "connection reset" in msg


class StreamPusher:
    """Queue-based background thread for pushing H.264+AAC to RTMP or SRT.

    Automatically detects the protocol from the URL scheme:
    - ``rtmp://`` → FLV container
    - ``srt://``  → MPEG-TS container

    Usage::

        pusher = StreamPusher("srt://push.example.com:1105?streamid=...",
                              width=448, height=448, fps=25)
        pusher.start()
        for chunk_frames, chunk_audio in generate():
            pusher.push_chunk(chunk_frames, chunk_audio)
        pusher.stop()

    Parameters
    ----------
    url : str
        Ingest URL.  Supported schemes: ``rtmp://``, ``srt://``.
    width, height : int
        Video frame dimensions.
    fps : int
        Video frame rate.  Defaults to 25 (FlashTalk default).
    queue_maxsize : int
        Maximum queued chunks.  When full, the oldest chunk is dropped.
    """

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int = 25,
        queue_maxsize: int = 8,
    ):
        self._url = url
        self._width = width
        self._height = height
        self._fps = fps

        # Detect protocol from URL scheme
        scheme = urlparse(url).scheme.lower()
        if scheme == "srt":
            self._container_format = "mpegts"
            self._protocol = "srt"
        else:
            self._container_format = "flv"
            self._protocol = "rtmp"

        self._queue: queue.Queue[Optional[Tuple[np.ndarray, Optional[np.ndarray]]]] = (
            queue.Queue(maxsize=queue_maxsize)
        )
        self._thread: Optional[threading.Thread] = None
        self._container = None  # set by _run(); used by stop() to force-close
        self._container_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._failed = False
        self._started = False
        self._abandoned = False
        self._frame_count = 0

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        """Spawn the background encoding/push thread."""
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"{self._protocol}-push"
        )
        self._thread.start()
        self._started = True

    def push_chunk(
        self,
        frames_np: np.ndarray,
        audio_16k: Optional[np.ndarray] = None,
    ) -> None:
        """Enqueue a video chunk for encoding and push.

        Parameters
        ----------
        frames_np : np.ndarray
            Video frames, shape ``(T, H, W, 3)`` dtype ``uint8``.
        audio_16k : np.ndarray or None
            Raw audio for this chunk, float32 mono at 16 kHz.
            If *None*, silence is pushed for the audio track.
        """
        if self._failed or not self._started:
            return
        item = (frames_np, audio_16k)
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
            logger.warning("Stream pusher queue full — dropped oldest chunk")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the thread to flush remaining items and exit."""
        if not self._started:
            return
        # Signal the drain loop to stop checking the queue.
        self._stop_event.set()
        # Drain the queue first to guarantee room for the sentinel.
        # Without this, put(None) can block forever if the drain thread
        # is stuck (e.g., broken pipe in mux) and the queue is full.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Stream pusher thread did not exit within %.1fs", timeout)
                # Force-close the container so PyAV's __dealloc__ doesn't
                # try to write a trailer on a broken connection (segfault).
                # The drain thread's _run() skips container.close() when
                # _container is None, avoiding a deadlock where both
                # threads call close() on the same PyAV container.
                self._force_close_container()
                # Mark as abandoned so the _run() finally block also
                # skips container.close() on its thread-local reference.
                self._abandoned = True
        self._started = False

    def _force_close_container(self) -> None:
        """Best-effort close of the output container from the main thread.

        Uses a timeout to avoid blocking indefinitely when
        ``container.close()`` → ``av_write_trailer()`` hangs on a
        broken TCP connection (half-open socket, no RST received).
        """
        with self._container_lock:
            c = self._container
            if c is None:
                return
            self._container = None  # clear first so _run() finally block skips it

        def _do_close():
            try:
                c.close()
            except Exception:
                pass

        closer = threading.Thread(target=_do_close, daemon=True, name="rtmp-close")
        closer.start()
        closer.join(timeout=5.0)
        if closer.is_alive():
            logger.warning("Stream container close did not finish within 5 s, abandoning")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import av

            container = av.open(
                self._url, mode="w", format=self._container_format
            )
            with self._container_lock:
                self._container = container  # expose for force-close in stop()
        except Exception as exc:
            logger.error(
                "%s pusher: failed to open %s: %s",
                self._protocol.upper(),
                self._url,
                exc,
            )
            self._failed = True
            return

        try:
            # Video stream — H.264 ultrafast / zerolatency
            v_stream = container.add_stream("libx264", rate=self._fps)
            v_stream.width = self._width
            v_stream.height = self._height
            v_stream.pix_fmt = "yuv420p"
            v_opts = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "g": str(self._fps),  # GOP = 1 second
            }
            if self._protocol == "srt":
                # MPEG-TS requires Annex B byte-stream format with SPS/PPS
                # repeated at each keyframe for mid-stream joins.
                v_opts["x264opts"] = "repeat-headers=1"
            v_stream.options = v_opts

            # Audio stream — AAC mono 48 kHz
            a_stream = container.add_stream("aac", rate=_OUTPUT_AUDIO_SR)
            a_stream.layout = "mono"

            audio_samples_per_frame = _OUTPUT_AUDIO_SR // self._fps

            self._drain_loop(container, v_stream, a_stream, audio_samples_per_frame)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("%s connection closed by peer — stream ended", self._protocol.upper())
        except Exception as exc:
            # PyAV may wrap OS errors; treat broken-pipe / reset as non-fatal
            if _is_connection_closed(exc):
                logger.info("%s connection closed by peer — stream ended", self._protocol.upper())
            else:
                logger.error("%s pusher thread error: %s", self._protocol.upper(), exc)
                self._failed = True
        finally:
            # Only close if _container is still set AND stop() hasn't
            # abandoned us.  If stop() called _force_close_container(),
            # _container is None and we must NOT call close() again
            # (PyAV internal lock deadlock).  If _abandoned is True,
            # stop() already force-closed the container and the
            # thread-local `container` ref is stale — skip close to
            # avoid double-close and potential __dealloc__ hangs.
            if self._abandoned:
                return
            with self._container_lock:
                should_close = self._container is not None
                self._container = None
            if should_close:
                try:
                    container.close()
                except Exception:
                    pass

    def _drain_loop(self, container, v_stream, a_stream, audio_samples_per_frame) -> None:
        import av

        while True:
            # Use a timeout so we periodically check _stop_event.
            # This prevents blocking forever when mux() was hanging and
            # stop() drained the queue + sent a sentinel we missed.
            try:
                item = self._queue.get(timeout=2.0)
            except queue.Empty:
                if self._stop_event.is_set():
                    logger.debug("Stream drain loop exiting — stop event set")
                    break
                continue

            if item is None:
                # Flush encoders — ignore broken-pipe / connection-reset
                # errors that occur when the CDN closes first.
                try:
                    for pkt in v_stream.encode(None):
                        container.mux(pkt)
                    for pkt in a_stream.encode(None):
                        container.mux(pkt)
                except Exception as exc:
                    logger.debug("Stream flush on stop (harmless): %s", exc)
                break

            frames_np, audio_16k = item
            num_frames = frames_np.shape[0]

            # Resample audio
            if audio_16k is not None and len(audio_16k) > 0:
                audio_48k_s16 = _resample_16k_to_48k_int16(audio_16k)
            else:
                # Silence for the duration of these frames
                total_samples = num_frames * audio_samples_per_frame
                audio_48k_s16 = np.zeros(total_samples, dtype=np.int16)

            audio_pos = 0
            for i in range(num_frames):
                # Encode video frame
                vf = av.VideoFrame.from_ndarray(frames_np[i], format="rgb24")
                vf.pts = self._frame_count
                for pkt in v_stream.encode(vf):
                    container.mux(pkt)

                # Encode corresponding audio samples
                chunk_end = min(audio_pos + audio_samples_per_frame, len(audio_48k_s16))
                a_samples = audio_48k_s16[audio_pos:chunk_end]
                if len(a_samples) < audio_samples_per_frame:
                    a_samples = np.pad(
                        a_samples, (0, audio_samples_per_frame - len(a_samples))
                    )
                af = av.AudioFrame.from_ndarray(
                    a_samples.reshape(1, -1), format="s16", layout="mono"
                )
                af.sample_rate = _OUTPUT_AUDIO_SR
                af.pts = self._frame_count * audio_samples_per_frame
                for pkt in a_stream.encode(af):
                    container.mux(pkt)

                audio_pos = chunk_end
                self._frame_count += 1


# Backwards-compatible alias
RTMPPusher = StreamPusher
