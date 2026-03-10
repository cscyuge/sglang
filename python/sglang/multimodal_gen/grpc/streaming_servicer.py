"""gRPC servicer for FlashTalk frame streaming.

Polls session_dir/grpc_frames/ for encoded frame files written by
GrpcFramePusher and yields them as FramePacket messages.
"""

from __future__ import annotations

import logging
import os
import struct
import time

import grpc

from sglang.multimodal_gen.grpc import flashtalk_streaming_pb2 as pb2
from sglang.multimodal_gen.grpc import flashtalk_streaming_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

# Must match grpc_frame_pusher.py
_FRAME_HEADER_FMT = "<IQB2I"
_FRAME_HEADER_SIZE = struct.calcsize(_FRAME_HEADER_FMT)  # 21 bytes
_META_HEADER_FMT = "<4I"
_META_HEADER_SIZE = struct.calcsize(_META_HEADER_FMT)  # 16 bytes

# Polling / timeout constants
_POLL_INTERVAL_S = 0.1
_META_TIMEOUT_S = 60.0
_FRAME_TIMEOUT_S = 120.0  # max wait between consecutive frames


def _get_session_dir(session_id: str) -> str:
    """Resolve the session directory from session_id.

    Mirrors the convention in video_api.py: outputs/sessions/<session_id>/
    """
    return os.path.join("outputs", "sessions", session_id)


def _parse_video_meta(data: bytes) -> pb2.VideoMeta:
    """Parse video_meta.bin into a VideoMeta protobuf message."""
    width, height, fps, extra_len = struct.unpack_from(_META_HEADER_FMT, data)
    extra_data = data[_META_HEADER_SIZE : _META_HEADER_SIZE + extra_len]
    return pb2.VideoMeta(
        width=width,
        height=height,
        fps=fps,
        codec_extra_data=extra_data,
    )


def _parse_frame_file(data: bytes):
    """Parse a frame_XXXXXX.bin file.

    Returns (chunk_index, pts_ms, is_keyframe, h264_data, pcm_data).
    """
    chunk_index, pts_ms, is_kf, h264_len, pcm_len = struct.unpack_from(
        _FRAME_HEADER_FMT, data
    )
    offset = _FRAME_HEADER_SIZE
    h264_data = data[offset : offset + h264_len]
    offset += h264_len
    pcm_data = data[offset : offset + pcm_len]
    return chunk_index, pts_ms, bool(is_kf), h264_data, pcm_data


class FlashTalkStreamingServicer(pb2_grpc.FlashTalkStreamingServicer):
    """Implements the FlashTalkStreaming gRPC service."""

    def StreamFrames(self, request, context):
        """Server-streaming RPC: yields FramePacket messages for a session."""
        session_id = request.session_id
        session_dir = _get_session_dir(session_id)
        frames_dir = os.path.join(session_dir, "grpc_frames")

        logger.info("StreamFrames started for session %s", session_id)

        # Wait for video_meta.bin to appear
        meta_path = os.path.join(frames_dir, "video_meta.bin")
        video_meta = self._wait_for_meta(meta_path, context)
        if video_meta is None:
            if not context.is_active():
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(
                    f"Timed out waiting for video metadata (session={session_id})"
                )
            return

        # Stream frames
        next_index = 0
        include_meta = True  # attach VideoMeta to first packet

        while context.is_active():
            # Check for end sentinel
            end_path = os.path.join(frames_dir, "end")
            frame_path = os.path.join(frames_dir, f"frame_{next_index:06d}.bin")

            if os.path.exists(frame_path):
                try:
                    with open(frame_path, "rb") as f:
                        data = f.read()
                    chunk_index, pts_ms, is_kf, h264_data, pcm_data = _parse_frame_file(data)
                except Exception as exc:
                    logger.warning("Failed to read frame %d: %s", next_index, exc)
                    time.sleep(_POLL_INTERVAL_S)
                    continue

                pkt = pb2.FramePacket(
                    chunk_index=chunk_index,
                    h264_data=h264_data,
                    pcm_audio=pcm_data,
                    pts_ms=pts_ms,
                    is_keyframe=is_kf,
                )
                if include_meta:
                    pkt.video_meta.CopyFrom(video_meta)
                    include_meta = False

                yield pkt
                next_index += 1
                last_frame_time = time.monotonic()

            elif os.path.exists(end_path):
                logger.info("Session %s ended, closing stream", session_id)
                break
            else:
                # Poll
                if next_index > 0 and (time.monotonic() - last_frame_time) > _FRAME_TIMEOUT_S:
                    logger.warning(
                        "Timed out waiting for frame %d (session=%s)", next_index, session_id
                    )
                    break
                time.sleep(_POLL_INTERVAL_S)

        logger.info(
            "StreamFrames ended for session %s (sent %d frames)", session_id, next_index
        )

    def _wait_for_meta(self, meta_path: str, context) -> pb2.VideoMeta | None:
        """Poll for video_meta.bin, return parsed VideoMeta or None on timeout."""
        deadline = time.monotonic() + _META_TIMEOUT_S
        while time.monotonic() < deadline:
            if not context.is_active():
                return None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "rb") as f:
                        data = f.read()
                    return _parse_video_meta(data)
                except Exception as exc:
                    logger.warning("Failed to parse video_meta.bin: %s", exc)
            time.sleep(_POLL_INTERVAL_S)
        return None
