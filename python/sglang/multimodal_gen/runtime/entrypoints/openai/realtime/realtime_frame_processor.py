# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        RealtimePostprocessConfig,
        RealtimeVideoGenerationsRequest,
    )

logger = logging.getLogger(__name__)

_FRAME_PROCESSOR_PIX_FMTS = {"rgb24", "bgr24"}
_FrameProcessorPolicy = Literal["passthrough", "error"]


@dataclass(slots=True)
class RealtimeFrameProcessorResult:
    frames: np.ndarray
    stats: dict[str, Any]


class BaseRealtimeFrameProcessor:
    name = "none"
    enabled = False

    def __init__(self, *, input_width: int, input_height: int, fps: int) -> None:
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.output_width = int(input_width)
        self.output_height = int(input_height)
        self.fps = int(fps) or 25

    def build_init_ack(self) -> dict[str, Any] | None:
        return None

    def process(
        self,
        frames: np.ndarray,
        *,
        session_id: str,
        chunk_idx: int | None,
    ) -> RealtimeFrameProcessorResult:
        del session_id, chunk_idx
        return RealtimeFrameProcessorResult(frames=frames, stats={})


class NoopRealtimeFrameProcessor(BaseRealtimeFrameProcessor):
    pass


def _normalize_pix_fmt(value: str | None, *, default: str = "rgb24") -> str:
    pix_fmt = str(value or default).strip().lower()
    if pix_fmt not in _FRAME_PROCESSOR_PIX_FMTS:
        raise RealtimeProtocolError(
            "invalid_realtime_postprocess_pix_fmt",
            f"unsupported realtime postprocess pix_fmt: {value}",
            pix_fmt=value,
        )
    return pix_fmt


def _normalize_policy(
    value: str | None,
    *,
    default: _FrameProcessorPolicy = "passthrough",
) -> _FrameProcessorPolicy:
    policy = str(value or default).strip().lower()
    if policy not in {"passthrough", "error"}:
        raise RealtimeProtocolError(
            "invalid_realtime_postprocess_policy",
            f"unsupported realtime postprocess policy: {value}",
            policy=value,
        )
    return policy  # type: ignore[return-value]


def _as_rgb24(frames: np.ndarray, pix_fmt: str) -> np.ndarray:
    if pix_fmt == "rgb24":
        return frames
    return np.ascontiguousarray(frames[..., ::-1])


def _from_rgb24(frames: np.ndarray, pix_fmt: str) -> np.ndarray:
    if pix_fmt == "rgb24":
        return frames
    return np.ascontiguousarray(frames[..., ::-1])


def _resize_nearest(frames: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if frames.shape[1] == height and frames.shape[2] == width:
        return np.ascontiguousarray(frames)
    in_h = int(frames.shape[1])
    in_w = int(frames.shape[2])
    if width % in_w == 0 and height % in_h == 0:
        scale_w = width // in_w
        scale_h = height // in_h
        return np.ascontiguousarray(
            np.repeat(np.repeat(frames, scale_h, axis=1), scale_w, axis=2)
        )
    # Generic fallback without pulling a video/image dependency into the hot path.
    y_idx = (np.arange(height) * in_h // height).astype(np.int64)
    x_idx = (np.arange(width) * in_w // width).astype(np.int64)
    return np.ascontiguousarray(frames[:, y_idx][:, :, x_idx])


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, socket.timeout))
    return False


class RemoteCodeFormerFrameProcessor(BaseRealtimeFrameProcessor):
    name = "codeformer"
    enabled = True

    def __init__(
        self,
        *,
        config: "RealtimePostprocessConfig",
        input_width: int,
        input_height: int,
        fps: int,
    ) -> None:
        super().__init__(input_width=input_width, input_height=input_height, fps=fps)
        endpoint = str(
            config.endpoint or os.environ.get("SGLANG_REALTIME_CODEFORMER_ENDPOINT", "")
        ).strip()
        if not endpoint:
            raise RealtimeProtocolError(
                "missing_realtime_postprocess_endpoint",
                "realtime_postprocess type='codeformer' requires endpoint",
            )
        self.endpoint = endpoint
        self.scale = int(config.scale or 2)
        if self.scale < 1:
            raise RealtimeProtocolError(
                "invalid_realtime_postprocess_scale",
                "realtime_postprocess scale must be >= 1",
                scale=config.scale,
            )
        self.output_width = self.input_width * self.scale
        self.output_height = self.input_height * self.scale
        self.timeout_ms = max(0.0, float(config.timeout_ms or 0.0))
        self.on_timeout = _normalize_policy(config.on_timeout)
        self.on_busy = _normalize_policy(config.on_busy)
        self.on_error = _normalize_policy(config.on_error)
        self.input_pix_fmt = _normalize_pix_fmt(config.input_pix_fmt, default="rgb24")
        self.output_pix_fmt = _normalize_pix_fmt(config.output_pix_fmt, default="rgb24")

    def build_init_ack(self) -> dict[str, Any] | None:
        return {
            "type": "codeformer",
            "endpoint": self.endpoint,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "scale": self.scale,
            "timeout_ms": self.timeout_ms,
            "on_timeout": self.on_timeout,
            "on_busy": self.on_busy,
            "on_error": self.on_error,
        }

    def process(
        self,
        frames: np.ndarray,
        *,
        session_id: str,
        chunk_idx: int | None,
    ) -> RealtimeFrameProcessorResult:
        started = time.perf_counter()
        stats: dict[str, Any] = {
            "frame_processor_name": self.name,
            "frame_processor_enabled": True,
            "frame_processor_status": "started",
            "frame_processor_passthrough": False,
            "frame_processor_total_ms": 0.0,
            "frame_processor_http_ms": 0.0,
            "frame_processor_payload_build_ms": 0.0,
            "frame_processor_materialize_ms": 0.0,
            "frame_processor_output_width": self.output_width,
            "frame_processor_output_height": self.output_height,
        }
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RealtimeProtocolError(
                "invalid_frame_processor_input",
                "realtime frame processor expects NHWC rgb24 frames",
                shape=tuple(frames.shape),
            )
        if int(frames.shape[1]) != self.input_height or int(frames.shape[2]) != self.input_width:
            raise RealtimeProtocolError(
                "frame_processor_input_size_mismatch",
                "realtime frame processor input size does not match configured size",
                frame_width=int(frames.shape[2]),
                frame_height=int(frames.shape[1]),
                configured_width=self.input_width,
                configured_height=self.input_height,
            )

        payload_started = time.perf_counter()
        payload_frames = _from_rgb24(np.ascontiguousarray(frames), self.input_pix_fmt)
        payload = payload_frames.tobytes()
        stats["frame_processor_payload_build_ms"] = (
            time.perf_counter() - payload_started
        ) * 1000.0
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Width": str(self.input_width),
            "X-Height": str(self.input_height),
            "X-Num-Frames": str(int(frames.shape[0])),
            "X-Pix-Fmt": self.input_pix_fmt,
            "X-Output-Pix-Fmt": self.output_pix_fmt,
            "X-Scale": str(self.scale),
            "X-Output-Width": str(self.output_width),
            "X-Output-Height": str(self.output_height),
            "X-Session-Id": session_id,
            "X-Chunk-Index": "" if chunk_idx is None else str(chunk_idx),
            "X-Fps": str(self.fps),
            "X-Deadline-Ms": str(int(self.timeout_ms)) if self.timeout_ms > 0 else "",
        }
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        timeout_s = self.timeout_ms / 1000.0 if self.timeout_ms > 0 else None
        http_started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                response_body = response.read()
                response_headers = response.headers
            stats["frame_processor_http_ms"] = (time.perf_counter() - http_started) * 1000.0
        except urllib.error.HTTPError as exc:
            stats["frame_processor_http_ms"] = (time.perf_counter() - http_started) * 1000.0
            if exc.code == 429:
                return self._passthrough(
                    frames,
                    stats,
                    started=started,
                    reason="busy",
                    policy=self.on_busy,
                    error=f"HTTP {exc.code}",
                )
            return self._passthrough(
                frames,
                stats,
                started=started,
                reason="http_error",
                policy=self.on_error,
                error=f"HTTP {exc.code}: {exc.reason}",
            )
        except Exception as exc:
            stats["frame_processor_http_ms"] = (time.perf_counter() - http_started) * 1000.0
            if _is_timeout_error(exc):
                return self._passthrough(
                    frames,
                    stats,
                    started=started,
                    reason="timeout",
                    policy=self.on_timeout,
                    error=str(exc),
                )
            return self._passthrough(
                frames,
                stats,
                started=started,
                reason="error",
                policy=self.on_error,
                error=str(exc),
            )

        materialize_started = time.perf_counter()
        try:
            out_width = int(response_headers.get("X-Width", self.output_width))
            out_height = int(response_headers.get("X-Height", self.output_height))
            out_frames = int(response_headers.get("X-Num-Frames", int(frames.shape[0])))
            out_pix_fmt = _normalize_pix_fmt(
                response_headers.get("X-Pix-Fmt"), default=self.output_pix_fmt
            )
            if (out_width, out_height) != (self.output_width, self.output_height):
                raise ValueError(
                    f"bad CodeFormer output size: expected {self.output_width}x{self.output_height}, "
                    f"got {out_width}x{out_height}"
                )
            expected_bytes = out_width * out_height * 3 * out_frames
            if len(response_body) != expected_bytes:
                raise ValueError(
                    f"bad CodeFormer response size: expected {expected_bytes} bytes, got {len(response_body)}"
                )
            output = np.frombuffer(response_body, dtype=np.uint8).reshape(
                out_frames,
                out_height,
                out_width,
                3,
            )
            output = _as_rgb24(output, out_pix_fmt).copy()
            if out_frames != int(frames.shape[0]):
                raise ValueError(
                    f"bad CodeFormer frame count: expected {frames.shape[0]}, got {out_frames}"
                )
            stats["frame_processor_materialize_ms"] = (
                time.perf_counter() - materialize_started
            ) * 1000.0
            stats["frame_processor_status"] = "ok"
            stats["frame_processor_remote_timing"] = response_headers.get("X-Timing", "")
            stats["frame_processor_output_width"] = out_width
            stats["frame_processor_output_height"] = out_height
            stats["frame_processor_total_ms"] = (time.perf_counter() - started) * 1000.0
            return RealtimeFrameProcessorResult(frames=output, stats=stats)
        except Exception as exc:
            stats["frame_processor_materialize_ms"] = (
                time.perf_counter() - materialize_started
            ) * 1000.0
            return self._passthrough(
                frames,
                stats,
                started=started,
                reason="bad_response",
                policy=self.on_error,
                error=str(exc),
            )

    def _passthrough(
        self,
        frames: np.ndarray,
        stats: dict[str, Any],
        *,
        started: float,
        reason: str,
        policy: _FrameProcessorPolicy,
        error: str,
    ) -> RealtimeFrameProcessorResult:
        stats["frame_processor_status"] = reason
        stats["frame_processor_error"] = error
        if policy == "error":
            raise RealtimeProtocolError(
                "realtime_frame_processor_failed",
                f"realtime frame processor failed: {reason}: {error}",
                processor=self.name,
                reason=reason,
                endpoint=self.endpoint,
            )
        output = _resize_nearest(
            frames,
            width=self.output_width,
            height=self.output_height,
        )
        stats["frame_processor_passthrough"] = True
        stats["frame_processor_output_width"] = self.output_width
        stats["frame_processor_output_height"] = self.output_height
        stats["frame_processor_total_ms"] = (time.perf_counter() - started) * 1000.0
        logger.warning(
            "realtime frame processor passthrough: processor=%s reason=%s error=%s",
            self.name,
            reason,
            error,
        )
        return RealtimeFrameProcessorResult(frames=output, stats=stats)


def create_realtime_frame_processor(
    request: "RealtimeVideoGenerationsRequest",
    *,
    input_width: int,
    input_height: int,
    fps: int,
) -> BaseRealtimeFrameProcessor:
    config = request.realtime_postprocess
    if config is None or str(config.type or "none").lower() == "none":
        return NoopRealtimeFrameProcessor(
            input_width=input_width,
            input_height=input_height,
            fps=fps,
        )
    processor_type = str(config.type).strip().lower()
    if processor_type == "codeformer":
        return RemoteCodeFormerFrameProcessor(
            config=config,
            input_width=input_width,
            input_height=input_height,
            fps=fps,
        )
    raise RealtimeProtocolError(
        "unsupported_realtime_postprocess",
        f"unsupported realtime_postprocess type: {config.type}",
        postprocess_type=config.type,
    )
