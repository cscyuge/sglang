"""GrpcFramePusher — writes H.264-encoded frames to session_dir/grpc_frames/.

Duck-typed to match StreamPusher interface (start / push_chunk / stop / failed).
The gRPC servicer picks up the files via filesystem polling.
"""

from __future__ import annotations

import logging
import os
import queue
import struct
import tempfile
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Binary frame header layout (21 bytes total):
#   [4B] chunk_index   uint32 LE
#   [8B] pts_ms        uint64 LE
#   [1B] is_keyframe   uint8
#   [4B] h264_len      uint32 LE
#   [4B] pcm_len       uint32 LE
#   [NB] h264_data
#   [MB] pcm_data
_FRAME_HEADER_FMT = "<IQB2I"
_FRAME_HEADER_SIZE = struct.calcsize(_FRAME_HEADER_FMT)  # 21 bytes

# video_meta.bin layout:
#   [4B] width   [4B] height   [4B] fps   [4B] extra_data_len   [NB] extra_data
_META_HEADER_FMT = "<4I"
_META_HEADER_SIZE = struct.calcsize(_META_HEADER_FMT)  # 16 bytes


class GrpcFramePusher:
    """Encodes video frames to H.264 and writes them to grpc_frames/ for the
    gRPC servicer to pick up.  Provides the same interface as StreamPusher."""

    def __init__(
        self,
        session_dir: str,
        width: int,
        height: int,
        fps: int = 25,
        queue_maxsize: int = 8,
    ):
        self._session_dir = session_dir
        self._width = width
        self._height = height
        self._fps = fps

        self._frames_dir = os.path.join(session_dir, "grpc_frames")
        os.makedirs(self._frames_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._failed = False
        self._frame_count = 0
        self._chunk_index = 0
        self._meta_written = False

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        """Start the background encoding thread."""
        self._thread = threading.Thread(
            target=self._run, name="grpc-frame-encoder", daemon=True
        )
        self._thread.start()

    def push_chunk(self, frames_np: np.ndarray, audio_16k: Optional[np.ndarray] = None) -> None:
        """Enqueue a chunk of video frames + optional audio for encoding.

        Args:
            frames_np: (T, H, W, 3) uint8 RGB frames.
            audio_16k: float32 mono PCM at 16 kHz, or None.
        """
        if self._failed:
            return
        try:
            self._queue.put_nowait((frames_np, audio_16k))
        except queue.Full:
            # Drop oldest to keep up
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((frames_np, audio_16k))
            except queue.Full:
                pass

    def stop(self, timeout: float = 10.0) -> None:
        """Drain the queue, write end sentinel, and join the thread."""
        if self._thread is None:
            return
        # Send None sentinel to stop the encoding thread
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("gRPC frame encoder thread did not exit in %.1fs", timeout)
        # Write end sentinel for the servicer
        end_path = os.path.join(self._frames_dir, "end")
        try:
            with open(end_path, "w"):
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import av

            # Create a standalone H.264 codec context (no container mux)
            codec_ctx = av.CodecContext.create("libx264", "w")
            codec_ctx.width = self._width
            codec_ctx.height = self._height
            codec_ctx.pix_fmt = "yuv420p"
            codec_ctx.time_base = av.Fraction(1, self._fps)
            codec_ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "g": str(self._fps),  # GOP = 1 second
            }
            codec_ctx.open()

            # Extract SPS/PPS from extradata
            extradata = bytes(codec_ctx.extradata) if codec_ctx.extradata else b""

            # Write video_meta.bin
            self._write_video_meta(extradata)

            self._drain_loop(codec_ctx)

            # Flush encoder
            try:
                for pkt in codec_ctx.encode(None):
                    self._write_frame_packet(pkt, b"")
            except Exception:
                pass

            codec_ctx.close()
        except Exception as exc:
            logger.error("gRPC frame encoder error: %s", exc)
            self._failed = True

    def _drain_loop(self, codec_ctx) -> None:
        import av

        while True:
            item = self._queue.get()
            if item is None:
                break

            frames_np, audio_16k = item
            num_frames = frames_np.shape[0]

            # Convert audio to raw PCM bytes (float32 LE)
            if audio_16k is not None and len(audio_16k) > 0:
                pcm_bytes = np.ascontiguousarray(audio_16k, dtype=np.float32).tobytes()
            else:
                pcm_bytes = b""

            # Encode each video frame, accumulate H.264 data per chunk
            h264_chunks = []
            first_keyframe = False
            for i in range(num_frames):
                vf = av.VideoFrame.from_ndarray(frames_np[i], format="rgb24")
                vf.pts = self._frame_count
                self._frame_count += 1
                for pkt in codec_ctx.encode(vf):
                    h264_chunks.append(bytes(pkt))
                    if pkt.is_keyframe and not first_keyframe:
                        first_keyframe = True

            h264_data = b"".join(h264_chunks)
            pts_ms = int(self._chunk_index * num_frames * 1000 / self._fps)

            self._write_frame_file(
                chunk_index=self._chunk_index,
                pts_ms=pts_ms,
                is_keyframe=first_keyframe,
                h264_data=h264_data,
                pcm_data=pcm_bytes,
            )
            self._chunk_index += 1

    def _write_video_meta(self, extradata: bytes) -> None:
        """Write video_meta.bin with codec parameters."""
        header = struct.pack(
            _META_HEADER_FMT,
            self._width,
            self._height,
            self._fps,
            len(extradata),
        )
        meta_path = os.path.join(self._frames_dir, "video_meta.bin")
        self._atomic_write(meta_path, header + extradata)
        self._meta_written = True
        logger.info(
            "gRPC video_meta written: %dx%d @ %d fps, extradata=%d bytes",
            self._width, self._height, self._fps, len(extradata),
        )

    def _write_frame_file(
        self,
        chunk_index: int,
        pts_ms: int,
        is_keyframe: bool,
        h264_data: bytes,
        pcm_data: bytes,
    ) -> None:
        """Write a frame_{chunk_index:06d}.bin file atomically."""
        header = struct.pack(
            _FRAME_HEADER_FMT,
            chunk_index,
            pts_ms,
            1 if is_keyframe else 0,
            len(h264_data),
            len(pcm_data),
        )
        fname = f"frame_{chunk_index:06d}.bin"
        path = os.path.join(self._frames_dir, fname)
        self._atomic_write(path, header + h264_data + pcm_data)

    def _write_frame_packet(self, pkt, pcm_data: bytes) -> None:
        """Write a single encoder packet as a frame file (used during flush)."""
        pts_ms = int(self._chunk_index * 1000 / self._fps) if pkt.pts is not None else 0
        h264_data = bytes(pkt)
        self._write_frame_file(
            chunk_index=self._chunk_index,
            pts_ms=pts_ms,
            is_keyframe=pkt.is_keyframe,
            h264_data=h264_data,
            pcm_data=pcm_data,
        )
        self._chunk_index += 1

    @staticmethod
    def _atomic_write(path: str, data: bytes) -> None:
        """Write data atomically via temp file + rename."""
        dir_name = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name)
        try:
            os.write(fd, data)
            os.close(fd)
            os.rename(tmp_path, path)
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
