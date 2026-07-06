# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError
from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState

try:
    import scipy.signal as scipy_signal
except ImportError:  # pragma: no cover
    scipy_signal = None


WAN_S2V_REALTIME_DEFAULT_FPS = 16
WAN_S2V_REALTIME_SAMPLE_RATE = 16000
WAN_S2V_REALTIME_VAE_TEMPORAL_SCALE = 4
WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS = 60000.0


@dataclass(slots=True)
class WanS2VAudioWindow:
    chunk_idx: int
    pts_start_ms: float
    pts_end_ms: float
    samples: np.ndarray
    is_final: bool


class WanS2VAudioTimelineState(BaseRealtimeState):
    """Sample-accurate Wan S2V realtime audio timeline.

    The endpoint adapter uses this state to validate client-owned audio deltas
    and emit model-sized audio windows. Worker-side Wan S2V runners can reuse
    the same state class once per-chunk generation is wired to RealtimeSession.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = WAN_S2V_REALTIME_SAMPLE_RATE
        self.fps = WAN_S2V_REALTIME_DEFAULT_FPS
        self.num_frame_per_block = 3
        self.vae_temporal_scale = WAN_S2V_REALTIME_VAE_TEMPORAL_SCALE
        self.first_public_frames = 9
        self.steady_public_frames = 12
        self.first_window_samples = 9000
        self.steady_window_samples = 12000
        self.pad_final_window = True
        self.allow_resample = True
        self.max_buffered_audio_ms = WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS
        self.max_buffered_audio_samples: int | None = None
        self._update_max_buffered_audio_samples()
        self.reset()

    def configure(
        self,
        *,
        fps: int,
        num_frame_per_block: int,
        sample_rate: int = WAN_S2V_REALTIME_SAMPLE_RATE,
        vae_temporal_scale: int = WAN_S2V_REALTIME_VAE_TEMPORAL_SCALE,
        pad_final_window: bool = True,
        allow_resample: bool = True,
        max_buffered_audio_ms: float | None = (
            WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS
        ),
    ) -> None:
        if sample_rate != WAN_S2V_REALTIME_SAMPLE_RATE:
            raise ValueError("Wan S2V realtime timeline output must be 16 kHz")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        if max_buffered_audio_ms is None:
            max_buffered_audio_ms = self.max_buffered_audio_ms
        if float(max_buffered_audio_ms) < 0:
            raise ValueError("max_buffered_audio_ms must be non-negative")
        self.sample_rate = int(sample_rate)
        self.fps = int(fps)
        self.num_frame_per_block = int(num_frame_per_block)
        self.vae_temporal_scale = int(vae_temporal_scale)
        self.pad_final_window = bool(pad_final_window)
        self.allow_resample = bool(allow_resample)
        self.max_buffered_audio_ms = float(max_buffered_audio_ms)
        self._update_max_buffered_audio_samples()
        self.first_public_frames = (
            self.num_frame_per_block - 1
        ) * self.vae_temporal_scale + 1
        self.steady_public_frames = (
            self.num_frame_per_block * self.vae_temporal_scale
        )
        self.first_window_samples = max(
            1,
            int(round(self.first_public_frames * self.sample_rate / self.fps)),
        )
        self.steady_window_samples = max(
            1,
            int(round(self.steady_public_frames * self.sample_rate / self.fps)),
        )

    def reset(self) -> None:
        self.next_seq = 0
        self.next_pts_ms = 0.0
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.audio_buffer_start_pts_ms = 0.0
        self.audio_end_received = False
        self.final_seq: int | None = None
        self.generated_windows = 0
        self.latest_event_id: int | None = None
        self.latest_event_debug: dict[str, Any] | None = None

    def _update_max_buffered_audio_samples(self) -> None:
        if self.max_buffered_audio_ms <= 0:
            self.max_buffered_audio_samples = None
            return
        self.max_buffered_audio_samples = max(
            1,
            int(round(self.max_buffered_audio_ms * self.sample_rate / 1000.0)),
        )

    def _buffered_audio_ms(self, samples: int | None = None) -> float:
        sample_count = self.audio_buffer.size if samples is None else int(samples)
        return sample_count * 1000.0 / self.sample_rate

    def _pts_end_ms(self) -> float:
        return self.audio_buffer_start_pts_ms + self._buffered_audio_ms()

    def debug_snapshot(self) -> dict[str, Any]:
        required_samples = self._required_window_samples()
        return {
            "timeline_sample_rate": self.sample_rate,
            "next_seq": self.next_seq,
            "next_pts_ms": round(self.next_pts_ms, 3),
            "queue_samples": int(self.audio_buffer.size),
            "queue_ms": round(self._buffered_audio_ms(), 3),
            "queue_start_pts_ms": round(self.audio_buffer_start_pts_ms, 3),
            "queue_end_pts_ms": round(self._pts_end_ms(), 3),
            "ready_window": self.has_ready_window(),
            "required_window_samples": int(required_samples),
            "required_window_ms": round(self._buffered_audio_ms(required_samples), 3),
            "audio_end_received": self.audio_end_received,
            "final_seq": self.final_seq,
            "generated_windows": self.generated_windows,
            "max_buffered_audio_ms": round(self.max_buffered_audio_ms, 3),
        }

    def _validate_buffer_capacity(self, *, incoming_samples: int, seq: int) -> None:
        if self.max_buffered_audio_samples is None:
            return
        projected_samples = self.audio_buffer.size + int(incoming_samples)
        if projected_samples <= self.max_buffered_audio_samples:
            return
        raise RealtimeProtocolError(
            "audio_buffer_overflow",
            "audio.delta would exceed max buffered audio",
            seq=seq,
            buffered_ms=round(self._buffered_audio_ms(), 3),
            incoming_ms=round(self._buffered_audio_ms(incoming_samples), 3),
            projected_ms=round(self._buffered_audio_ms(projected_samples), 3),
            max_buffered_audio_ms=round(self.max_buffered_audio_ms, 3),
            buffered_samples=int(self.audio_buffer.size),
            incoming_samples=int(incoming_samples),
            max_buffered_audio_samples=int(self.max_buffered_audio_samples),
        )

    def _required_window_samples(self) -> int:
        if self.generated_windows == 0:
            return self.first_window_samples
        return self.steady_window_samples

    def _decode_audio_samples(self, payload: dict[str, Any]) -> np.ndarray:
        fmt = str(payload.get("format") or payload.get("audio_format") or "pcm16")
        raw = payload.get("audio")
        if raw is None:
            raw = payload.get("payload")
        if raw is None:
            b64_payload = payload.get("audio_b64") or payload.get("payload_b64")
            if b64_payload is not None:
                raw = base64.b64decode(b64_payload)
        if raw is None:
            if payload.get("is_silence"):
                sample_count = int(payload.get("sample_count") or 0)
                if sample_count <= 0:
                    raise RealtimeProtocolError(
                        "invalid_audio_delta",
                        "silent audio.delta requires sample_count",
                    )
                return np.zeros(sample_count, dtype=np.float32)
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta requires audio bytes",
            )
        if isinstance(raw, str):
            raw = base64.b64decode(raw)
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta audio payload must be bytes",
            )
        raw_bytes = bytes(raw)
        if fmt == "pcm16":
            return np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if fmt == "f32le":
            return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32)
        raise RealtimeProtocolError(
            "unsupported_audio_format",
            f"unsupported Wan S2V realtime audio format: {fmt}",
            audio_format=fmt,
        )

    def _resample_to_timeline_rate(
        self,
        samples: np.ndarray,
        *,
        source_sample_rate: int,
    ) -> np.ndarray:
        if source_sample_rate == self.sample_rate:
            return samples.astype(np.float32, copy=False)
        if not self.allow_resample:
            raise RealtimeProtocolError(
                "audio_resample_not_allowed",
                "Wan S2V realtime timeline does not resample audio.delta",
                source_sample_rate=source_sample_rate,
                timeline_sample_rate=self.sample_rate,
            )
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)

        target_count = int(round(samples.size * self.sample_rate / source_sample_rate))
        target_count = max(1, target_count)
        if scipy_signal is not None:
            divisor = math.gcd(source_sample_rate, self.sample_rate)
            resampled = scipy_signal.resample_poly(
                samples,
                self.sample_rate // divisor,
                source_sample_rate // divisor,
            ).astype(np.float32, copy=False)
        else:
            source_t = np.arange(samples.size, dtype=np.float64) / source_sample_rate
            target_t = np.arange(target_count, dtype=np.float64) / self.sample_rate
            resampled = np.interp(target_t, source_t, samples).astype(np.float32)

        if resampled.size > target_count:
            resampled = resampled[:target_count]
        elif resampled.size < target_count:
            pad_mode = "edge" if resampled.size else "constant"
            resampled = np.pad(resampled, (0, target_count - resampled.size), pad_mode)
        return resampled.astype(np.float32, copy=False)

    def receive_audio_delta(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        if not isinstance(payload, dict):
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta payload must be a map",
            )
        if self.audio_end_received:
            raise RealtimeProtocolError(
                "audio_delta_after_end",
                "cannot receive audio.delta after audio.end",
            )
        if "seq" not in payload:
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta requires seq",
            )
        if "pts_ms" not in payload:
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta requires pts_ms",
            )

        seq = int(payload["seq"])
        if seq != self.next_seq:
            raise RealtimeProtocolError(
                "audio_seq_mismatch",
                f"audio.delta seq mismatch: expected {self.next_seq}, got {seq}",
                expected_seq=self.next_seq,
                got_seq=seq,
            )

        source_sample_rate = int(payload.get("sample_rate") or self.sample_rate)
        if source_sample_rate <= 0:
            raise RealtimeProtocolError(
                "invalid_audio_delta",
                "audio.delta sample_rate must be positive",
                sample_rate=source_sample_rate,
            )
        channels = int(payload.get("channels") or 1)
        if channels != 1:
            raise RealtimeProtocolError(
                "unsupported_audio_channels",
                "Wan S2V realtime timeline requires mono audio.delta",
                channels=channels,
            )

        source_samples = self._decode_audio_samples(payload)
        declared_sample_count = payload.get("sample_count")
        if (
            declared_sample_count is not None
            and int(declared_sample_count) != len(source_samples)
        ):
            raise RealtimeProtocolError(
                "audio_sample_count_mismatch",
                "audio.delta sample_count mismatch: "
                f"declared={declared_sample_count} decoded={len(source_samples)}",
                declared_sample_count=int(declared_sample_count),
                decoded_sample_count=len(source_samples),
            )

        pts_ms = float(payload["pts_ms"])
        if abs(pts_ms - self.next_pts_ms) > 0.5:
            raise RealtimeProtocolError(
                "audio_pts_mismatch",
                "audio.delta pts mismatch: "
                f"expected={self.next_pts_ms:.3f} got={pts_ms:.3f}",
                expected_pts_ms=round(self.next_pts_ms, 3),
                got_pts_ms=round(pts_ms, 3),
            )

        samples = self._resample_to_timeline_rate(
            source_samples,
            source_sample_rate=source_sample_rate,
        )
        self._validate_buffer_capacity(incoming_samples=len(samples), seq=seq)
        if self.audio_buffer.size == 0:
            self.audio_buffer_start_pts_ms = pts_ms
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])
        self.next_seq += 1
        self.next_pts_ms = pts_ms + len(source_samples) * 1000.0 / source_sample_rate
        self.latest_event_id = event_id
        self.latest_event_debug = {
            "kind": "audio.delta",
            "event_id": event_id,
            "seq": seq,
            "pts_ms": round(pts_ms, 3),
            "source_sample_rate": source_sample_rate,
            "source_samples": len(source_samples),
            "source_duration_ms": round(
                len(source_samples) * 1000.0 / source_sample_rate, 3
            ),
            "timeline_samples": len(samples),
            "timeline_duration_ms": round(self._buffered_audio_ms(len(samples)), 3),
        }
        return (
            "kind=audio.delta "
            f"seq={seq} source_samples={len(source_samples)} "
            f"samples={len(samples)} queue_samples={self.audio_buffer.size} "
            f"queue_ms={self._buffered_audio_ms():.3f}"
        )

    def receive_audio_end(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        if payload is not None and not isinstance(payload, dict):
            raise RealtimeProtocolError(
                "invalid_audio_end",
                "audio.end payload must be a map",
            )
        payload = payload or {}
        final_seq = payload.get("final_seq")
        if final_seq is not None and int(final_seq) != self.next_seq - 1:
            raise RealtimeProtocolError(
                "audio_end_final_seq_mismatch",
                "audio.end final_seq mismatch: "
                f"expected={self.next_seq - 1} got={final_seq}",
                expected_final_seq=self.next_seq - 1,
                got_final_seq=int(final_seq),
            )
        self.audio_end_received = True
        self.final_seq = None if final_seq is None else int(final_seq)
        self.latest_event_id = event_id
        self.latest_event_debug = {
            "kind": "audio.end",
            "event_id": event_id,
            "final_seq": self.final_seq,
        }
        return (
            "kind=audio.end "
            f"final_seq={self.final_seq} queue_samples={self.audio_buffer.size} "
            f"queue_ms={self._buffered_audio_ms():.3f}"
        )

    def has_ready_window(self) -> bool:
        required = self._required_window_samples()
        if self.audio_buffer.size >= required:
            return True
        return self.audio_end_received and self.audio_buffer.size > 0

    def is_drained(self) -> bool:
        return self.audio_end_received and self.audio_buffer.size == 0

    def pop_window(self) -> WanS2VAudioWindow:
        required = self._required_window_samples()
        is_final = self.audio_end_received and self.audio_buffer.size <= required
        if self.audio_buffer.size < required:
            if not self.audio_end_received or not self.pad_final_window:
                raise ValueError("audio window is not ready")
            samples = np.pad(self.audio_buffer, (0, required - self.audio_buffer.size))
            consumed = self.audio_buffer.size
        else:
            samples = self.audio_buffer[:required]
            consumed = required
        pts_start_ms = self.audio_buffer_start_pts_ms
        pts_end_ms = pts_start_ms + consumed * 1000.0 / self.sample_rate
        self.audio_buffer = self.audio_buffer[consumed:]
        self.audio_buffer_start_pts_ms = pts_end_ms
        window = WanS2VAudioWindow(
            chunk_idx=self.generated_windows,
            pts_start_ms=pts_start_ms,
            pts_end_ms=pts_end_ms,
            samples=np.asarray(samples, dtype=np.float32),
            is_final=is_final,
        )
        self.generated_windows += 1
        return window

    def clear(self) -> None:
        self.reset()

    def dispose(self) -> None:
        self.clear()
