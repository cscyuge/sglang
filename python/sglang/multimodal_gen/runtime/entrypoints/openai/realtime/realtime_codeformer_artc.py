# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import struct
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

_MAGIC = b"CFA1"
_HEADER_STRUCT = struct.Struct("!4sI")
_MAX_HEADER_BYTES = 1024 * 1024


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def encode_codeformer_artc_chunk(
    header: dict[str, Any],
    frames: bytes,
    audio: bytes = b"",
) -> bytes:
    payload_header = dict(header)
    payload_header.setdefault("version", 1)
    payload_header["frame_bytes"] = len(frames)
    payload_header["audio_bytes"] = len(audio)
    encoded = json.dumps(
        payload_header,
        ensure_ascii=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    if len(encoded) > _MAX_HEADER_BYTES:
        raise ValueError("CodeFormer ARTC chunk header is too large")
    return _HEADER_STRUCT.pack(_MAGIC, len(encoded)) + encoded + frames + audio


class CodeFormerArtcClient:
    def __init__(
        self,
        *,
        endpoint: str,
        session_id: str,
        timeout_ms: float,
    ) -> None:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint:
            raise ValueError("CodeFormer ARTC session endpoint is required")
        self.endpoint = endpoint
        self.session_id = session_id
        self.timeout_s = timeout_ms / 1000.0 if timeout_ms > 0 else None
        self.session_timeout_s = (
            float(
                os.environ.get("SGLANG_REALTIME_CODEFORMER_SESSION_TIMEOUT_MS", "30000")
            )
            / 1000.0
        )

    @property
    def session_url(self) -> str:
        return f"{self.endpoint}/{urllib.parse.quote(self.session_id, safe='')}"

    @staticmethod
    def _decode_json(response) -> dict[str, Any]:
        body = response.read()
        payload = json.loads(body) if body else {}
        if not isinstance(payload, dict):
            raise ValueError("CodeFormer returned a non-object JSON response")
        return payload

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.session_timeout_s
        ) as response:
            return self._decode_json(response)

    def send_chunk(
        self,
        *,
        frames: np.ndarray,
        audio: np.ndarray | None,
        chunk_idx: int,
        width: int,
        height: int,
        fps: int,
        audio_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        audio_bytes = b""
        audio_samples = 0
        if audio is not None and len(audio):
            audio_array = np.ascontiguousarray(audio, dtype="<f4")
            audio_bytes = audio_array.tobytes()
            audio_samples = int(audio_array.size)
        body = encode_codeformer_artc_chunk(
            {
                "chunk_idx": int(chunk_idx),
                "width": int(width),
                "height": int(height),
                "num_frames": int(frames.shape[0]),
                "pix_fmt": "rgb24",
                "fps": int(fps),
                "audio_sample_rate": 16000,
                "audio_dtype": "float32le",
                "audio_samples": audio_samples,
                "audio_meta": dict(audio_meta or {}),
            },
            frames.tobytes(),
            audio_bytes,
        )
        request = urllib.request.Request(
            f"{self.session_url}/chunks",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return self._decode_json(response)

    def status(self) -> dict[str, Any]:
        request = urllib.request.Request(self.session_url, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return self._decode_json(response)

    def close(self, *, drain: bool) -> dict[str, Any]:
        mode = "drain" if drain else "cancel"
        request = urllib.request.Request(
            f"{self.session_url}?mode={mode}",
            method="DELETE",
        )
        with urllib.request.urlopen(
            request, timeout=self.session_timeout_s
        ) as response:
            return self._decode_json(response)
