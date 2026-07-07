# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.realtime.states.wan_s2v_audio import (
    WAN_S2V_REALTIME_DEFAULT_FPS,
    WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS,
    WAN_S2V_REALTIME_SAMPLE_RATE,
    WanS2VAudioTimelineState,
    WanS2VAudioWindow,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _resolve_float_config(
    server_args: ServerArgs | None,
    *,
    attr: str,
    env_name: str,
    default: float,
) -> float:
    raw_env = os.environ.get(env_name)
    if raw_env is not None and raw_env != "":
        return float(raw_env)
    return float(getattr(getattr(server_args, "pipeline_config", None), attr, default))


def _resolve_bool_config(
    server_args: ServerArgs | None,
    *,
    attr: str,
    env_name: str,
    default: bool,
) -> bool:
    raw_env = os.environ.get(env_name)
    if raw_env is not None and raw_env != "":
        return raw_env.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(getattr(server_args, "pipeline_config", None), attr, default))


def _audio_window_meta(window: WanS2VAudioWindow) -> dict[str, Any]:
    return {
        "chunk_idx": window.chunk_idx,
        "pts_start_ms": window.pts_start_ms,
        "pts_end_ms": window.pts_end_ms,
        "sample_count": int(len(window.samples)),
        "is_final": window.is_final,
    }


class WanS2VRealtimeAdapterState:
    """Endpoint-side async wrapper around the reusable Wan S2V timeline."""

    def __init__(self) -> None:
        self.timeline = WanS2VAudioTimelineState()
        self.audio_ready = asyncio.Event()
        self.reserved_prefetch_window: WanS2VAudioWindow | None = None

    def configure(
        self,
        *,
        fps: int,
        num_frame_per_block: int,
        sample_rate: int = WAN_S2V_REALTIME_SAMPLE_RATE,
        pad_final_window: bool = True,
        max_buffered_audio_ms: float = WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS,
    ) -> None:
        self.timeline.configure(
            fps=fps,
            num_frame_per_block=num_frame_per_block,
            sample_rate=sample_rate,
            pad_final_window=pad_final_window,
            max_buffered_audio_ms=max_buffered_audio_ms,
        )
        self.reserved_prefetch_window = None

    def has_ready_window(self) -> bool:
        return (
            self.reserved_prefetch_window is not None
            or self.timeline.has_ready_window()
        )

    def _wake_if_progress_possible(self) -> None:
        if self.has_ready_window() or self.timeline.is_drained():
            self.audio_ready.set()

    def receive_audio_delta(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        event_log = self.timeline.receive_audio_delta(payload, event_id=event_id)
        self._wake_if_progress_possible()
        return event_log

    def receive_audio_end(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        event_log = self.timeline.receive_audio_end(payload, event_id=event_id)
        self._wake_if_progress_possible()
        return event_log

    async def wait_for_ready_window(self) -> None:
        while not self.has_ready_window():
            if self.timeline.is_drained():
                raise StopAsyncIteration
            self.audio_ready.clear()
            if self.has_ready_window() or self.timeline.is_drained():
                break
            await self.audio_ready.wait()
        if self.timeline.is_drained() and not self.has_ready_window():
            raise StopAsyncIteration

    def pop_window(self) -> WanS2VAudioWindow:
        if self.reserved_prefetch_window is not None:
            window = self.reserved_prefetch_window
            self.reserved_prefetch_window = None
        else:
            window = self.timeline.pop_window()
        if not self.has_ready_window():
            self.audio_ready.clear()
        return window

    def reserve_prefetch_window(self) -> WanS2VAudioWindow | None:
        if self.reserved_prefetch_window is not None:
            return None
        if not self.timeline.has_ready_window():
            return None
        self.reserved_prefetch_window = self.timeline.pop_window()
        self.audio_ready.set()
        return self.reserved_prefetch_window

    def debug_snapshot(self) -> dict[str, Any]:
        snapshot = self.timeline.debug_snapshot()
        reserved = self.reserved_prefetch_window
        snapshot["reserved_prefetch_window"] = (
            None if reserved is None else _audio_window_meta(reserved)
        )
        snapshot["ready_window"] = self.has_ready_window()
        return snapshot

    def clear(self) -> None:
        self.timeline.clear()
        self.reserved_prefetch_window = None
        self.audio_ready.set()


class WanS2VRealtimeAdapter(BaseRealtimeModelAdapter):
    """WebSocket adapter for Wan2.2-S2V realtime audio-driven generation."""

    def create_state(self) -> WanS2VRealtimeAdapterState:
        return WanS2VRealtimeAdapterState()

    def _state(self, session: GenerateSession) -> WanS2VRealtimeAdapterState:
        state = session.adapter_state
        if not isinstance(state, WanS2VRealtimeAdapterState):
            raise TypeError("Wan S2V realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        request.fps = int(request.fps or WAN_S2V_REALTIME_DEFAULT_FPS)
        if request.realtime_output_format is None:
            request.realtime_output_format = "raw"
        server_args = None
        try:
            from sglang.multimodal_gen.runtime.server_args import (
                get_global_server_args,
            )

            server_args = get_global_server_args()
        except Exception:
            server_args = None
        num_frame_per_block = int(
            getattr(
                getattr(server_args, "pipeline_config", None),
                "num_frame_per_block",
                3,
            )
        )
        max_buffered_audio_ms = _resolve_float_config(
            server_args,
            attr="wan_s2v_realtime_max_buffered_audio_ms",
            env_name="WAN_S2V_REALTIME_MAX_BUFFERED_AUDIO_MS",
            default=WAN_S2V_REALTIME_DEFAULT_MAX_BUFFERED_AUDIO_MS,
        )
        state = self._state(session)
        state.configure(
            fps=request.fps,
            num_frame_per_block=num_frame_per_block,
            sample_rate=WAN_S2V_REALTIME_SAMPLE_RATE,
            max_buffered_audio_ms=max_buffered_audio_ms,
        )
        await save_realtime_first_frame(
            session,
            request,
            required_error="Wan S2V realtime requires first_frame",
        )

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind in {"audio.delta", "audio_delta"}:
            return state.receive_audio_delta(event.payload, event_id=event.event_id)
        if event.kind in {"audio.end", "audio_end"}:
            return state.receive_audio_end(event.payload, event_id=event.event_id)
        raise ValueError(f"unsupported Wan S2V realtime event kind: {event.kind}")

    def build_init_ack(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> dict[str, Any] | None:
        timeline = self._state(session).timeline
        first_window_ms = timeline.first_window_samples * 1000.0 / timeline.sample_rate
        steady_window_ms = (
            timeline.steady_window_samples * 1000.0 / timeline.sample_rate
        )
        return {
            "realtime": {
                "fps": request.fps,
                "max_chunks": request.max_chunks,
                "realtime_output_format": request.realtime_output_format,
                "realtime_output_pacing": bool(request.realtime_output_pacing),
            },
            "audio_timeline": {
                "sample_rate": timeline.sample_rate,
                "num_frame_per_block": timeline.num_frame_per_block,
                "first_public_frames": timeline.first_public_frames,
                "steady_public_frames": timeline.steady_public_frames,
                "first_window_samples": timeline.first_window_samples,
                "steady_window_samples": timeline.steady_window_samples,
                "first_window_ms": round(first_window_ms, 3),
                "steady_window_ms": round(steady_window_ms, 3),
                "pad_final_window": timeline.pad_final_window,
                "max_buffered_audio_ms": round(timeline.max_buffered_audio_ms, 3),
            },
        }

    def build_event_ack(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
        event_log: str,
    ) -> dict[str, Any] | None:
        del event, event_log
        state = self._state(session)
        timeline = state.timeline
        return {
            "audio_event": dict(timeline.latest_event_debug or {}),
            "audio_queue": state.debug_snapshot(),
        }

    def build_chunk_stats_extra(
        self,
        session: GenerateSession,
        batch,
        result,
    ) -> dict[str, Any] | None:
        del session, result
        meta = getattr(batch, "extra", {}).get("wan_s2v_audio_window_meta")
        if not isinstance(meta, dict):
            return None
        pts_start_ms = float(meta.get("pts_start_ms") or 0.0)
        pts_end_ms = float(meta.get("pts_end_ms") or 0.0)
        return {
            "audio_window": {
                "chunk_idx": int(meta.get("chunk_idx") or 0),
                "pts_start_ms": round(pts_start_ms, 3),
                "pts_end_ms": round(pts_end_ms, 3),
                "duration_ms": round(max(0.0, pts_end_ms - pts_start_ms), 3),
                "sample_count": int(meta.get("sample_count") or 0),
                "is_final": bool(meta.get("is_final")),
                "event_id": getattr(batch, "realtime_event_id", None),
            }
        }

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        await self._state(session).wait_for_ready_window()

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        del session, chunk
        return int(getattr(server_args.pipeline_config, "num_frame_per_block", 3))

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ):
        batch = super().prepare_next_request(session, server_args, chunk)
        state = self._state(session)
        window = state.pop_window()
        batch.extra["wan_s2v_realtime_per_chunk"] = True
        batch.extra["stream_r1_mode"] = True
        batch.extra["num_frame_per_block"] = self.get_chunk_size(
            session, server_args, chunk
        )
        batch.extra["wan_s2v_audio_window"] = window.samples
        batch.extra["wan_s2v_audio_window_meta"] = _audio_window_meta(window)
        batch.extra["wan_s2v_audio_is_final"] = window.is_final
        if self._should_prefetch_next_audio_window(session, server_args, chunk, window):
            prefetch_window = state.reserve_prefetch_window()
            if prefetch_window is not None:
                batch.extra["wan_s2v_prefetch_audio_window"] = prefetch_window.samples
                batch.extra["wan_s2v_prefetch_audio_window_meta"] = _audio_window_meta(
                    prefetch_window
                )
        return batch

    def _should_prefetch_next_audio_window(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        window: WanS2VAudioWindow,
    ) -> bool:
        if window.is_final:
            return False
        request = session.request
        if (
            request is not None
            and request.max_chunks is not None
            and chunk.index + 1 >= request.max_chunks
        ):
            return False
        return _resolve_bool_config(
            server_args,
            attr="wan_s2v_ws_audio_cpu_prefetch",
            env_name="WAN_S2V_WS_AUDIO_CPU_PREFETCH",
            default=False,
        )

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        del server_args, chunk_size
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        condition_inputs = (
            dict(request.condition_inputs or {}) if chunk.index == 0 else {}
        )
        return RealtimeChunkInputs(
            prompt=request.prompt,
            condition_inputs=condition_inputs,
        )

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        temporal = int(
            getattr(
                server_args.pipeline_config.vae_config.arch_config,
                "scale_factor_temporal",
                4,
            )
        )
        num_frames = max(1, (int(chunk_size) - 1) * temporal + 1)
        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            chunk_size=chunk_size,
        )

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._state(session).timeline.latest_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, WanS2VRealtimeAdapterState):
            state.clear()
