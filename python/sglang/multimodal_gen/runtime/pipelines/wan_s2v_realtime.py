# SPDX-License-Identifier: Apache-2.0
"""Realtime session runner for Wan2.2-S2V Stream-R1 inference."""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_rank,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VAudioEncodingStage,
    WanS2VDenoisingDispatchStage,
    build_wan_s2v_condition_bundle,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.chunk_timeline import (
    emit_chunk_timeline,
    flashtalk_chunk_timeline_path,
    is_flashtalk_filler_audio_meta,
    read_flashtalk_audio_chunk_meta,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class AudioRingBuffer:
    """Fixed-size numpy ring buffer for 16 kHz session audio."""

    __slots__ = ("_buf", "_pos", "_cap")

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._pos = 0
        self._cap = capacity

    def extend(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        if n >= self._cap:
            np.copyto(self._buf, samples[-self._cap :])
            self._pos = 0
            return
        end = self._pos + n
        if end <= self._cap:
            self._buf[self._pos : end] = samples
        else:
            first = self._cap - self._pos
            self._buf[self._pos :] = samples[:first]
            self._buf[: n - first] = samples[first:]
        self._pos = end % self._cap

    def snapshot(self) -> np.ndarray:
        if self._pos == 0:
            return self._buf.copy()
        return np.concatenate((self._buf[self._pos :], self._buf[: self._pos]))


def _session_audio_chunk_path(session_dir: str, chunk_idx: int) -> str:
    return os.path.join(session_dir, "audio_chunks", f"chunk_{chunk_idx:04d}.npy")


def _session_audio_chunk_meta(session_dir: str, chunk_idx: int) -> dict[str, Any]:
    return read_flashtalk_audio_chunk_meta(session_dir, chunk_idx)


def _wait_for_session_audio_chunk(
    session_dir: str,
    chunk_idx: int,
    cancel_file: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 0.05,
) -> np.ndarray | None:
    chunk_path = _session_audio_chunk_path(session_dir, chunk_idx)
    end_path = os.path.join(session_dir, "end")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(end_path):
            return None
        if cancel_file and os.path.exists(cancel_file):
            return None
        if os.path.exists(chunk_path):
            try:
                return np.load(chunk_path)
            except Exception:
                time.sleep(0.01)
                try:
                    return np.load(chunk_path)
                except Exception:
                    return None
        time.sleep(poll_interval)
    return None


def _safe_world_rank() -> int:
    try:
        return get_world_rank()
    except Exception:
        return 0


def _pipeline_config_value(
    server_args: ServerArgs,
    key: str,
    default: Any,
) -> Any:
    return getattr(server_args.pipeline_config, key, default)


class WanS2VRealtimeSessionRunner:
    """Consume session audio chunks and emit Wan S2V frames block-by-block.

    This runner is intentionally separate from the normal batch pipeline. It
    reuses the same model stages, but keeps prompt/reference state resident and
    drives Stream-R1 one latent block at a time.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.stages = pipeline.stages

    def _get_stage(self, stage_type: type) -> Any:
        for stage in self.stages:
            if isinstance(stage, stage_type):
                return stage
        raise RuntimeError(f"{stage_type.__name__} not found in Wan S2V pipeline")

    def _block_public_frames(
        self,
        server_args: ServerArgs,
        num_frame_per_block: int,
    ) -> int:
        temporal = int(
            getattr(
                server_args.pipeline_config.vae_config.arch_config,
                "scale_factor_temporal",
                4,
            )
        )
        return max(1, (num_frame_per_block - 1) * temporal + 1)

    def _prepare_reference_and_prompt(
        self,
        batch: Req,
        server_args: ServerArgs,
        block_public_frames: int,
    ) -> Req:
        original_num_frames = batch.num_frames
        batch.extra["wan_s2v_realtime_original_num_frames"] = original_num_frames
        batch.extra["wan_s2v_realtime_block_num_frames"] = block_public_frames
        batch.num_frames = block_public_frames

        for stage_type in (
            InputValidationStage,
            TextEncodingStage,
            ImageVAEEncodingStage,
        ):
            stage = self._get_stage(stage_type)
            batch = stage(batch, server_args)
        return batch

    def _prepare_block_latents(
        self,
        batch: Req,
        server_args: ServerArgs,
        latent_stage: LatentPreparationStage,
        block_public_frames: int,
    ) -> torch.Tensor:
        batch.num_frames = block_public_frames
        batch.latents = None
        batch = latent_stage(batch, server_args)
        return batch.latents

    def _encode_audio_window(
        self,
        batch: Req,
        server_args: ServerArgs,
        audio_stage: WanS2VAudioEncodingStage,
        audio_window: np.ndarray,
        *,
        target_audio_frames: int,
        audio_window_video_frames: int,
    ) -> torch.Tensor:
        device = get_local_torch_device()
        sample_rate = 16000
        audio_stage.load_model()

        speech_array = audio_window.astype(np.float32, copy=False)
        speech_array = audio_stage._loudness_norm(speech_array, sample_rate)

        if audio_stage.wav2vec_feature_extractor is not None:
            audio_feature_np = np.squeeze(
                audio_stage.wav2vec_feature_extractor(
                    speech_array, sampling_rate=sample_rate
                ).input_values
            )
            audio_feature = (
                torch.from_numpy(audio_feature_np).float().to(device).unsqueeze(0)
            )
        else:
            audio_feature = (
                torch.from_numpy(speech_array).float().to(device).unsqueeze(0)
            )

        with set_forward_context(current_timestep=0, attn_metadata=None):
            audio_features = audio_stage.audio_encoder(
                audio_feature,
                num_video_frames=audio_window_video_frames,
            )

        if audio_features.shape[1] < target_audio_frames:
            pad = target_audio_frames - audio_features.shape[1]
            audio_features = torch.nn.functional.pad(
                audio_features, (0, 0, 0, 0, 0, pad)
            )
        else:
            audio_features = audio_features[:, -target_audio_frames:]

        return audio_features.permute(0, 2, 3, 1).contiguous()

    def _next_audio_chunk(
        self,
        *,
        session_dir: str,
        audio_chunk_idx: int,
        cancel_file: str | None,
        idle_policy: str,
        timeline_path: str | None,
    ) -> tuple[int, np.ndarray | None, dict[str, Any], bool]:
        """Return next usable audio chunk.

        The bool return is true when the session should end. With
        ``idle_policy=hold`` filler chunks are consumed and skipped without
        triggering GPU generation.
        """
        while True:
            audio = _wait_for_session_audio_chunk(
                session_dir,
                audio_chunk_idx,
                cancel_file=cancel_file,
                timeout=float(
                    os.environ.get("SGLANG_WAN_S2V_SESSION_AUDIO_TIMEOUT_S", "300")
                ),
            )
            if audio is None:
                return audio_chunk_idx, None, {}, True

            meta = _session_audio_chunk_meta(session_dir, audio_chunk_idx)
            is_filler = is_flashtalk_filler_audio_meta(meta)
            current_idx = audio_chunk_idx
            audio_chunk_idx += 1

            if idle_policy == "hold" and is_filler:
                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_filler_audio_held",
                    audio_chunk_idx=current_idx,
                    samples=int(len(audio)),
                )
                continue

            return current_idx, np.asarray(audio, dtype=np.float32), meta, False

    def _setup_frame_dir(
        self,
        batch: Req,
        server_args: ServerArgs,
        session_id: str,
        frames_per_chunk: int,
    ) -> tuple[str | None, ThreadPoolExecutor | None]:
        if _safe_world_rank() != 0:
            return None, None

        frame_dir = os.path.join(server_args.output_path, ".frames", session_id)
        os.makedirs(frame_dir, exist_ok=True)
        meta = {
            "num_chunks": None,
            "fps": batch.fps or 24,
            "frames_per_chunk": frames_per_chunk,
            "width": batch.width,
            "height": batch.height,
            "session": True,
            "runtime": "wan_s2v_realtime",
        }
        with open(os.path.join(frame_dir, "meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
        return frame_dir, ThreadPoolExecutor(max_workers=1)

    @torch.no_grad()
    def run(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        session_dir = batch.extra.get("session_dir")
        if not session_dir:
            raise RuntimeError(
                "Wan S2V realtime session requires batch.extra.session_dir"
            )

        session_id = (
            getattr(batch, "request_id", None)
            or getattr(batch, "output_file_name", None)
            or os.path.basename(session_dir.rstrip(os.sep))
        )
        timeline_path = batch.extra.get(
            "chunk_timeline_path"
        ) or flashtalk_chunk_timeline_path(session_dir)
        progress_dir = os.path.join(server_args.output_path, ".progress")
        os.makedirs(progress_dir, exist_ok=True)
        progress_file = os.path.join(progress_dir, session_id)
        cancel_file = os.path.join(progress_dir, f"{session_id}.cancel")

        denoising_dispatch = self._get_stage(WanS2VDenoisingDispatchStage)
        denoising_stage = denoising_dispatch.stream_r1_stage
        audio_stage = self._get_stage(WanS2VAudioEncodingStage)
        latent_stage = self._get_stage(LatentPreparationStage)
        decoding_stage = self._get_stage(DecodingStage)

        stream_r1_mode = batch.extra.get("stream_r1_mode")
        if stream_r1_mode is None:
            stream_r1_mode = _pipeline_config_value(
                server_args, "stream_r1_mode", False
            )
        if not bool(stream_r1_mode):
            raise RuntimeError("Wan S2V realtime sessions require stream_r1_mode=true.")

        num_frame_per_block = int(
            batch.extra.get("num_frame_per_block")
            or _pipeline_config_value(server_args, "num_frame_per_block", 7)
        )
        block_public_frames = self._block_public_frames(
            server_args, num_frame_per_block
        )
        fps = int(batch.fps or 24)
        audio_window_seconds = float(
            _pipeline_config_value(
                server_args, "wan_s2v_realtime_audio_window_seconds", 8
            )
        )
        audio_window_samples = max(1, int(16000 * audio_window_seconds))
        audio_window_video_frames = max(1, int(round(audio_window_seconds * fps)))
        target_audio_frames = num_frame_per_block * 4
        idle_policy = str(
            _pipeline_config_value(server_args, "wan_s2v_idle_policy", "hold")
        ).lower()
        if idle_policy not in {"hold", "silence"}:
            idle_policy = "hold"

        logger.info(
            "Wan S2V realtime session start: session=%s block_latent_frames=%d "
            "block_public_frames=%d fps=%d audio_window=%.2fs idle_policy=%s",
            session_id,
            num_frame_per_block,
            block_public_frames,
            fps,
            audio_window_seconds,
            idle_policy,
        )
        emit_chunk_timeline(
            timeline_path,
            "wan_s2v_realtime_session_start",
            session_id=session_id,
            block_latent_frames=num_frame_per_block,
            block_public_frames=block_public_frames,
            fps=fps,
            audio_window_seconds=audio_window_seconds,
            idle_policy=idle_policy,
        )

        batch = self._prepare_reference_and_prompt(
            batch, server_args, block_public_frames
        )
        audio_ring = AudioRingBuffer(audio_window_samples)

        frame_dir, frame_executor = self._setup_frame_dir(
            batch, server_args, session_id, block_public_frames
        )
        frame_futures: list[Any] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()

        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.precision]
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        autocast_enabled = (
            dit_dtype != torch.float32 and not server_args.disable_autocast
        )

        cache_state = None
        attention_request = None
        crossattn_cache: list[dict] | None = None
        frame_seq_length = None
        timesteps = None
        prompt_embeds = None

        audio_chunk_idx = 0
        block_idx = 0
        end_requested = False

        decoding_stage.load_model()
        denoising_stage.load_model()
        try:
            while True:
                loop_started = time.perf_counter()
                current_audio_idx, audio_chunk, audio_meta, end_requested = (
                    self._next_audio_chunk(
                        session_dir=session_dir,
                        audio_chunk_idx=audio_chunk_idx,
                        cancel_file=cancel_file,
                        idle_policy=idle_policy,
                        timeline_path=timeline_path,
                    )
                )
                audio_chunk_idx = current_audio_idx + (0 if end_requested else 1)
                if end_requested:
                    break

                audio_ring.extend(audio_chunk)
                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_block_generation_start",
                    block_idx=block_idx,
                    audio_chunk_idx=current_audio_idx,
                    samples=int(len(audio_chunk)),
                    chunk_source=audio_meta.get("chunk_source"),
                    is_filler=is_flashtalk_filler_audio_meta(audio_meta),
                    turn_id=audio_meta.get("turn_id"),
                )

                audio_started = time.perf_counter()
                batch.extra["audio_input"] = self._encode_audio_window(
                    batch,
                    server_args,
                    audio_stage,
                    audio_ring.snapshot(),
                    target_audio_frames=target_audio_frames,
                    audio_window_video_frames=audio_window_video_frames,
                )
                audio_s = time.perf_counter() - audio_started

                latent_started = time.perf_counter()
                block_latents = self._prepare_block_latents(
                    batch,
                    server_args,
                    latent_stage,
                    block_public_frames,
                ).to(device=device, dtype=dit_dtype)
                batch.latents = block_latents
                latent_s = time.perf_counter() - latent_started

                if attention_request is None:
                    attention_request = denoising_stage._resolve_attention_request(
                        batch,
                        server_args,
                        block_latents.shape[2],
                    )
                    timesteps = denoising_stage._prepare_timesteps(
                        batch, server_args, device
                    )
                    if timesteps.numel() == 0:
                        raise ValueError(
                            "Wan S2V realtime session requires at least one timestep"
                        )
                    patch_size = (
                        server_args.pipeline_config.dit_config.arch_config.patch_size
                    )
                    _, _, _, latent_h, latent_w = block_latents.shape
                    frame_seq_length = (latent_h // patch_size[1]) * (
                        latent_w // patch_size[2]
                    )
                    denoising_stage._configure_transformer_attention(attention_request)
                    cache_state = denoising_stage._prepare_cache_state(
                        request=attention_request,
                        batch_size=block_latents.shape[0],
                        frame_seq_length=frame_seq_length,
                        dtype=dit_dtype,
                        device=device,
                    )
                    denoising_stage._guard_cache_runtime(cache_state)
                    crossattn_cache = [
                        {} for _ in range(len(denoising_stage.transformer.blocks))
                    ]

                bundle = build_wan_s2v_condition_bundle(
                    batch,
                    server_args,
                    latents=block_latents,
                    dtype=dit_dtype,
                    device=device,
                )
                prompt_embeds = bundle.prompt_embeds
                if isinstance(prompt_embeds, list):
                    prompt_embeds = prompt_embeds[0]
                denoising_stage._maybe_cache_audio_embeddings(
                    bundle,
                    dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                )

                denoise_started = time.perf_counter()
                current_latents = denoising_stage.denoise_stream_r1_block(
                    batch=batch,
                    block_latents=block_latents,
                    block_bundle=bundle,
                    block_start=block_idx * num_frame_per_block,
                    frame_seq_length=frame_seq_length,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    cache_state=cache_state,
                    crossattn_cache=crossattn_cache,
                    generator=generator,
                    dit_dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                    audio_start_frame=0,
                )
                denoising_stage._clean_context_refresh(
                    block_latents=current_latents,
                    prompt_embeds=prompt_embeds,
                    block_bundle=bundle,
                    current_start=block_idx * num_frame_per_block * frame_seq_length,
                    attention_request=attention_request,
                    cache_state=cache_state,
                    dtype=dit_dtype,
                    autocast_enabled=autocast_enabled,
                    forward_batch=batch,
                    crossattn_cache=crossattn_cache,
                    audio_start_frame=0,
                )
                batch.latents = current_latents
                denoise_s = time.perf_counter() - denoise_started

                decode_started = time.perf_counter()
                frames = decoding_stage.decode(batch.latents, server_args)
                frames = server_args.pipeline_config.post_decoding(frames, server_args)
                decode_s = time.perf_counter() - decode_started

                stream_started = time.perf_counter()
                self.pipeline._save_streaming_frames(
                    frames,
                    block_idx,
                    frame_dir,
                    frame_executor,
                    frame_futures,
                    int(frames.shape[2]),
                    chunk_audio_data=audio_chunk,
                    timeline_path=timeline_path,
                    audio_chunk_idx=current_audio_idx,
                    used_silence=False,
                    audio_loaded=True,
                    audio_prefetched=False,
                    chunk_source=audio_meta.get("chunk_source") or "audio",
                    is_filler=is_flashtalk_filler_audio_meta(audio_meta),
                    turn_id=audio_meta.get("turn_id"),
                )
                stream_s = time.perf_counter() - stream_started

                if _safe_world_rank() == 0:
                    try:
                        with open(progress_file, "w", encoding="utf-8") as fp:
                            fp.write(f"{block_idx + 1} -1")
                    except Exception:
                        pass

                total_s = time.perf_counter() - loop_started
                logger.info(
                    "Wan S2V realtime block %d: audio=%.3fs latent=%.3fs "
                    "denoise=%.3fs decode=%.3fs stream=%.3fs total=%.3fs",
                    block_idx,
                    audio_s,
                    latent_s,
                    denoise_s,
                    decode_s,
                    stream_s,
                    total_s,
                )
                emit_chunk_timeline(
                    timeline_path,
                    "wan_s2v_block_generation_done",
                    block_idx=block_idx,
                    audio_chunk_idx=current_audio_idx,
                    timings={
                        "audio_ms": round(audio_s * 1000, 3),
                        "latent_ms": round(latent_s * 1000, 3),
                        "denoise_ms": round(denoise_s * 1000, 3),
                        "decode_ms": round(decode_s * 1000, 3),
                        "stream_ms": round(stream_s * 1000, 3),
                        "total_ms": round(total_s * 1000, 3),
                    },
                )
                block_idx += 1
        finally:
            try:
                denoising_stage.offload_model()
            except Exception:
                pass
            try:
                audio_stage.offload_model()
            except Exception:
                pass
            try:
                decoding_stage.offload_model()
            except Exception:
                pass
            self.pipeline._post_loop_cleanup(
                gc_was_enabled,
                frame_futures,
                frame_executor,
                frame_dir,
            )
            try:
                os.remove(progress_file)
            except FileNotFoundError:
                pass

        emit_chunk_timeline(
            timeline_path,
            "wan_s2v_realtime_session_done",
            session_id=session_id,
            blocks=block_idx,
            ended=end_requested,
        )
        logger.info(
            "Wan S2V realtime session finished: session=%s blocks=%d",
            session_id,
            block_idx,
        )
        return OutputBatch(output=None, output_file_paths=[], metrics=batch.metrics)
