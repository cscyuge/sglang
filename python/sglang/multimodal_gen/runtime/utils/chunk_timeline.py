# SPDX-License-Identifier: Apache-2.0
"""Lightweight JSONL timeline for FlashTalk live-session chunks."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any


def _enabled() -> bool:
    value = os.environ.get("SGLANG_FLASHTALK_CHUNK_TIMELINE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def flashtalk_chunk_timeline_path(session_dir: str | None) -> str | None:
    if not session_dir:
        return None
    override = os.environ.get("SGLANG_FLASHTALK_CHUNK_TIMELINE_PATH", "").strip()
    if override:
        return override
    return os.path.join(session_dir, "chunk_timeline.jsonl")


def flashtalk_audio_chunk_meta_path(
    session_dir: str | None,
    chunk_idx: int,
) -> str | None:
    if not session_dir:
        return None
    return os.path.join(session_dir, "audio_chunks", f"chunk_{chunk_idx:04d}.json")


def read_flashtalk_audio_chunk_meta(
    session_dir: str | None,
    chunk_idx: int,
) -> dict[str, Any]:
    path = flashtalk_audio_chunk_meta_path(session_dir, chunk_idx)
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_flashtalk_audio_chunk_meta(
    session_dir: str | None,
    chunk_idx: int,
    meta: dict[str, Any],
) -> None:
    path = flashtalk_audio_chunk_meta_path(session_dir, chunk_idx)
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=True, default=str)
        os.replace(tmp_path, path)
    except Exception:
        return


def is_flashtalk_filler_audio_meta(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    explicit = meta.get("is_filler")
    if explicit is not None:
        if isinstance(explicit, str):
            return explicit.strip().lower() in ("1", "true", "yes", "on")
        return bool(explicit)
    source = str(meta.get("chunk_source") or meta.get("source") or "").lower()
    return source in {
        "filler",
        "silence",
        "idle_silence",
        "warmup_silence",
        "internal_silence",
    }


def emit_chunk_timeline(path: str | None, event: str, **fields: Any) -> None:
    if not path or not _enabled():
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_time_s": time.time(),
        "monotonic_s": time.monotonic(),
        "pid": os.getpid(),
        "event": event,
    }
    record.update(fields)

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    except Exception:
        # Timeline must never affect serving.
        return
