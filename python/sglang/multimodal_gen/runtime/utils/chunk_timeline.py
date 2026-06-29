# SPDX-License-Identifier: Apache-2.0
"""Lightweight JSONL timeline for FlashTalk live-session chunks."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

CHUNK_TRACE_META_KEYS = (
    "turn_id",
    "client_chunk_idx",
    "client_chunk_ms",
    "chunk_source",
    "is_filler",
    "is_first_real_chunk",
    "allow_preempt_filler",
    "turn_start_policy",
    "client_turn_t0_wall_ms",
    "client_t0_to_post_start_ms",
    "client_post_start_wall_ms",
    "client_input_rms",
    "client_input_peak",
    "worker_received_wall_ms",
    "received_monotonic_s",
    "queue_size_before",
    "queue_size_after",
    "pending_filler_chunks",
    "pending_real_chunks",
    "wait_after_received_ms",
    "session_chunk_idx",
    "input_audio_ms",
    "input_rms",
    "input_peak",
    "generate_ms",
    "audio_ms",
    "video_frames",
    "first_audio_pts",
    "first_video_pts",
    "last_audio_pts",
    "last_video_pts",
    "frame_idx",
    "is_silent",
    "rms",
    "peak",
    "pending_frames",
    "audio_delta_seq_start",
    "audio_delta_seq_end",
    "audio_delta_frames",
    "audio_delta_sources",
    "pts_start_ms",
    "pts_end_ms",
    "real_audio_ms",
    "silence_audio_ms",
    "fifo_level_ms",
    "client_audio_delta_mode",
    "client_audio_lead_ms",
    "pusher_userid",
    "state",
    "reason",
    "error_code",
    "error_message",
    "audio_published",
    "video_published",
    "dual_stream_enabled",
    "low_stream_profile",
)


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


def flashtalk_audio_chunk_consumed_idx_path(session_dir: str | None) -> str | None:
    if not session_dir:
        return None
    return os.path.join(session_dir, "audio_chunks", ".consumed_idx")


def read_flashtalk_audio_consumed_idx(session_dir: str | None) -> int:
    path = flashtalk_audio_chunk_consumed_idx_path(session_dir)
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as fp:
            return max(0, int((fp.read() or "0").strip()))
    except Exception:
        return 0


def write_flashtalk_audio_consumed_idx(
    session_dir: str | None,
    next_chunk_idx: int,
) -> None:
    path = flashtalk_audio_chunk_consumed_idx_path(session_dir)
    if not path:
        return
    try:
        next_chunk_idx = max(0, int(next_chunk_idx))
    except Exception:
        return
    try:
        current = read_flashtalk_audio_consumed_idx(session_dir)
        if next_chunk_idx < current:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            fp.write(str(next_chunk_idx))
        os.replace(tmp_path, path)
    except Exception:
        return


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
        "response_pending_silence",
        "warmup_silence",
        "internal_silence",
    }


def compact_chunk_trace_fields(
    *,
    session_id: str | None = None,
    meta: dict[str, Any] | None = None,
    chunk_idx: int | None = None,
    audio_chunk_idx: int | None = None,
    pts: int | float | None = None,
    wall_clock: int | float | None = None,
    queue_size: int | None = None,
    pending_filler_ms: int | float | None = None,
    audio_queue_ms: int | float | None = None,
    video_queue_ms: int | float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a consistent field set for chunk/worker/ARTC trace events."""
    meta = meta or {}
    fields: dict[str, Any] = {}

    resolved_session_id = session_id or meta.get("session_id")
    if resolved_session_id is not None:
        fields["session_id"] = resolved_session_id
    if chunk_idx is not None:
        fields["chunk_idx"] = chunk_idx
    if audio_chunk_idx is not None:
        fields["audio_chunk_idx"] = audio_chunk_idx

    for key in CHUNK_TRACE_META_KEYS:
        if key in meta:
            fields[key] = meta.get(key)

    if "is_filler" not in fields and meta:
        fields["is_filler"] = is_flashtalk_filler_audio_meta(meta)

    if pts is not None:
        fields["pts"] = pts
    if wall_clock is not None:
        fields["wall_clock"] = wall_clock
    queue_size = queue_size if queue_size is not None else meta.get("queue_size")
    pending_filler_ms = (
        pending_filler_ms
        if pending_filler_ms is not None
        else meta.get("pending_filler_ms")
    )
    audio_queue_ms = (
        audio_queue_ms if audio_queue_ms is not None else meta.get("audio_queue_ms")
    )
    video_queue_ms = (
        video_queue_ms if video_queue_ms is not None else meta.get("video_queue_ms")
    )
    if queue_size is not None:
        fields["queue_size"] = queue_size
    if pending_filler_ms is not None:
        fields["pending_filler_ms"] = pending_filler_ms
    if audio_queue_ms is not None:
        fields["audio_queue_ms"] = audio_queue_ms
    if video_queue_ms is not None:
        fields["video_queue_ms"] = video_queue_ms

    for key, value in extra.items():
        if value is not None:
            fields[key] = value
    return fields


def emit_chunk_timeline(path: str | None, event: str, **fields: Any) -> None:
    if not path or not _enabled():
        return

    wall_now = time.time()
    record = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_time_s": wall_now,
        "wall_clock": wall_now,
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
