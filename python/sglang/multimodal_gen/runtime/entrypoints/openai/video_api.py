# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import fractions
import json
import os
import shutil
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    SessionResponse,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    add_common_data_to_response,
    build_sampling_params,
    merge_image_input_list,
    process_generation_batch,
    save_audio_to_path,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


def _build_video_sampling_params(request_id: str, request: VideoGenerationsRequest):
    """Resolve video-specific defaults (fps, seconds → num_frames) then
    delegate to the shared build_sampling_params.

    When neither ``num_frames`` nor ``fps`` is explicitly provided by the
    caller, we pass *None* so that model-specific SamplingParams defaults
    take precedence (e.g. FlashTalkSamplingParams uses num_frames=33,
    fps=25 instead of the generic 24 fps × 4 s = 96 frames).
    """
    if request.num_frames is not None:
        num_frames = request.num_frames
        fps = request.fps if request.fps is not None else DEFAULT_FPS
    elif request.fps is not None:
        fps = request.fps
        seconds = (
            request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
        )
        num_frames = fps * seconds
    else:
        # Neither num_frames nor fps specified – let model defaults apply
        num_frames = None
        fps = None

    return build_sampling_params(
        request_id,
        prompt=request.prompt,
        size=request.size,
        num_frames=num_frames,
        fps=fps,
        image_path=request.input_reference,
        output_file_name=request_id,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        guidance_scale_2=request.guidance_scale_2,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        enable_frame_interpolation=request.enable_frame_interpolation,
        frame_interpolation_exp=request.frame_interpolation_exp,
        frame_interpolation_scale=request.frame_interpolation_scale,
        frame_interpolation_model_path=request.frame_interpolation_model_path,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        audio_path=getattr(request, "audio_path", None),
        audio_encode_mode=getattr(request, "audio_encode_mode", None),
    )


# extract metadata which http_server needs to know
def _video_job_from_sampling(
    request_id: str, req: VideoGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
        "stream_url": f"/v1/videos/{request_id}/stream",
        "events_url": f"/v1/videos/{request_id}/events",
        "audio_path": getattr(req, "audio_path", None),
    }


async def _save_first_input_image(
    image_sources, request_id: str, uploads_dir: str
) -> str | None:
    """Save the first input image from a list of sources and return its path."""
    image_list = merge_image_input_list(image_sources)
    if not image_list:
        return None
    image = image_list[0]

    os.makedirs(uploads_dir, exist_ok=True)

    filename = image.filename if hasattr(image, "filename") else "url_image"
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(image, target_path)


async def _save_audio_input(
    audio_upload: Optional[UploadFile],
    audio_url: Optional[str],
    request_id: str,
    uploads_dir: str = "inputs/uploads",
) -> Optional[str]:
    """Resolve an audio UploadFile or URL to a local path."""
    source = audio_upload or audio_url
    if source is None:
        return None

    os.makedirs(uploads_dir, exist_ok=True)

    if isinstance(source, str):
        # URL or base64
        target_path = os.path.join(uploads_dir, f"{request_id}_audio")
    else:
        filename = source.filename if hasattr(source, "filename") else "audio"
        target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")

    return await save_audio_to_path(source, target_path)


def _progress_dir() -> str:
    """Return the directory used for file-based progress / cancel IPC."""
    server_args = get_global_server_args()
    d = os.path.join(server_args.output_path, ".progress")
    os.makedirs(d, exist_ok=True)
    return d


def _frame_dir_for_job(job_id: str) -> str:
    """Return the .frames/{job_id}/ directory used for streaming frame IPC."""
    server_args = get_global_server_args()
    return os.path.join(server_args.output_path, ".frames", job_id)


def _session_dir_for_id(session_id: str) -> str:
    """Return the .sessions/{session_id}/ directory for session IPC."""
    server_args = get_global_server_args()
    return os.path.join(server_args.output_path, ".sessions", session_id)


async def _cleanup_frame_dir(frame_dir: str, delay: float = 5.0) -> None:
    """Remove a streaming frame directory after a delay."""
    await asyncio.sleep(delay)
    try:
        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)
    except Exception as e:
        logger.warning("Failed to clean up frame dir %s: %s", frame_dir, e)


async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    progress_dir = _progress_dir()
    progress_file = os.path.join(progress_dir, job_id)
    cancel_file = os.path.join(progress_dir, f"{job_id}.cancel")

    await VIDEO_STORE.update_fields(job_id, {"status": "processing"})
    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

        # Determine final status — check if a cancel was requested
        was_cancelled = os.path.exists(cancel_file)
        final_status = "cancelled" if was_cancelled else "completed"

        persistent_path = (
            save_file_path if not cloud_url and output_persistent else None
        )
        update_fields = {
            "status": final_status,
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": persistent_path,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.exception("Job %s dispatch failed:", job_id)
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
    finally:
        # Clean up progress and cancel sentinel files
        for f in (progress_file, cancel_file):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        # Schedule deferred cleanup of streaming frame directory
        frame_dir = _frame_dir_for_job(job_id)
        if os.path.isdir(frame_dir):
            asyncio.create_task(_cleanup_frame_dir(frame_dir, delay=120.0))
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)


# -- Session store (in-memory, matches VIDEO_STORE pattern) --
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}


async def _dispatch_session_async(session_id: str, batch: Req) -> None:
    """Dispatch a session batch to the scheduler.

    The pipeline will enter an open-ended chunk loop, waiting for audio
    chunk files to appear in the session directory.
    """
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    progress_dir = _progress_dir()
    cancel_file = os.path.join(progress_dir, f"{session_id}.cancel")

    _SESSION_STORE[session_id]["status"] = "running"
    await VIDEO_STORE.update_fields(session_id, {"status": "processing"})
    try:
        _, result = await process_generation_batch(async_scheduler_client, batch)
        was_cancelled = os.path.exists(cancel_file)
        final_status = "cancelled" if was_cancelled else "ended"
        _SESSION_STORE[session_id]["status"] = final_status
        await VIDEO_STORE.update_fields(
            session_id, {"status": final_status, "completed_at": int(time.time())}
        )
    except Exception as e:
        logger.exception("Session %s failed:", session_id)
        _SESSION_STORE[session_id]["status"] = "failed"
        _SESSION_STORE[session_id]["error"] = {"message": str(e)}
        await VIDEO_STORE.update_fields(
            session_id, {"status": "failed", "error": {"message": str(e)}}
        )
    finally:
        try:
            os.remove(cancel_file)
        except FileNotFoundError:
            pass
        # Deferred cleanup of frame dir and session dir
        frame_dir = _frame_dir_for_job(session_id)
        session_dir = _session_dir_for_id(session_id)
        if os.path.isdir(frame_dir):
            asyncio.create_task(_cleanup_frame_dir(frame_dir, delay=300.0))
        if os.path.isdir(session_dir):
            asyncio.create_task(_cleanup_frame_dir(session_dir, delay=300.0))
        # Deferred cleanup of in-memory session entry.
        # 60s grace period allows clients to fetch final status/events after
        # the session ends before the entry is removed from memory.
        _SESSION_CLEANUP_DELAY = 60

        async def _remove_session():
            await asyncio.sleep(_SESSION_CLEANUP_DELAY)
            _SESSION_STORE.pop(session_id, None)

        asyncio.create_task(_remove_session())


# ======================================================================
# Session endpoints
# ======================================================================


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: Request,
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    seed: Optional[int] = Form(1024),
    size: Optional[str] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    rtmp_push_url: Optional[str] = Form(None),
):
    """Create a persistent generation session for live streaming.

    The session initializes the pipeline with the reference image and prompt,
    then waits for audio chunks to be pushed via
    ``POST /v1/videos/sessions/{session_id}/chunks``.

    Video frames stream out in real-time via
    ``GET /v1/videos/sessions/{session_id}/stream`` (MJPEG) or
    ``GET /v1/videos/sessions/{session_id}/events`` (SSE).
    """
    server_args = get_global_server_args()
    session_id = generate_request_id()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Auto-end any existing sessions.  The scheduler processes sessions
    # serially; orphan sessions (client disconnected without DELETE) would
    # block the pipeline for up to 300 s.  Writing the ``end`` sentinel
    # causes _wait_for_session_audio_chunk to return immediately.
    for sid, sdata in list(_SESSION_STORE.items()):
        if sdata["status"] in ("created", "running"):
            try:
                end_path = os.path.join(_session_dir_for_id(sid), "end")
                os.makedirs(os.path.dirname(end_path), exist_ok=True)
                with open(end_path, "w"):
                    pass
            except Exception:
                pass
            sdata["status"] = "ended"
            logger.info("Auto-ended previous session %s", sid)

    # Save reference image
    uploads_dir = server_args.input_save_path or os.path.join("inputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    image_sources = merge_image_input_list(input_reference, reference_url)
    try:
        input_path = await _save_first_input_image(
            image_sources, session_id, uploads_dir
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e}")

    if not input_path:
        raise HTTPException(
            status_code=400, detail="input_reference is required for sessions"
        )

    # Create session directory
    session_dir = _session_dir_for_id(session_id)
    os.makedirs(os.path.join(session_dir, "audio_chunks"), exist_ok=True)

    # Build sampling params (minimal — no audio, short duration placeholder)
    req = VideoGenerationsRequest(
        prompt=prompt,
        input_reference=input_path,
        seed=seed,
        size=size or "",
        seconds=4,  # placeholder; session runs indefinitely
        num_inference_steps=num_inference_steps,
        enable_teacache=enable_teacache,
        **({"guidance_scale": guidance_scale} if guidance_scale is not None else {}),
    )
    try:
        sampling_params = _build_video_sampling_params(session_id, req)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build batch and inject session mode flags
    batch = prepare_request(server_args=server_args, sampling_params=sampling_params)
    batch.extra["session_mode"] = True
    batch.extra["session_dir"] = session_dir
    batch.extra["rtmp_push_url"] = rtmp_push_url

    # Store session metadata
    session_data = {
        "session_id": session_id,
        "object": "video.session",
        "status": "created",
        "stream_url": f"/v1/videos/{session_id}/stream",
        "events_url": f"/v1/videos/{session_id}/events",
        "rtmp_push_url": rtmp_push_url,
        "created_at": int(time.time()),
        "chunks_received": 0,
        "chunks_processed": 0,
    }
    _SESSION_STORE[session_id] = session_data

    # Also register as a video job so /stream and /events endpoints work
    await VIDEO_STORE.upsert(
        session_id,
        {
            "id": session_id,
            "object": "video",
            "status": "processing",
            "progress": 0,
            "created_at": session_data["created_at"],
            "stream_url": session_data["stream_url"],
            "events_url": session_data["events_url"],
        },
    )

    # Dispatch (blocks GPU worker until session ends)
    asyncio.create_task(_dispatch_session_async(session_id, batch))

    return SessionResponse(**session_data)


@router.post("/sessions/{session_id}/chunks")
async def push_session_chunk(
    session_id: str = Path(...),
    audio: Optional[UploadFile] = File(None),
):
    """Push an audio chunk to a running session.

    The audio is saved as a ``.npy`` float32 array in the session's
    ``audio_chunks/`` directory. The pipeline picks it up automatically.

    Accepts either:
    - A raw WAV/MP3 file (decoded via librosa)
    - A ``.npy`` file (raw float32 samples at 16 kHz)
    """
    if session_id not in _SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _SESSION_STORE[session_id]
    if session["status"] not in ("created", "running"):
        raise HTTPException(status_code=400, detail=f"Session is {session['status']}")

    if audio is None:
        raise HTTPException(status_code=400, detail="audio file is required")

    session_dir = _session_dir_for_id(session_id)
    chunk_idx = session["chunks_received"]
    session["chunks_received"] = chunk_idx + 1  # atomic increment before await
    chunk_path = os.path.join(session_dir, "audio_chunks", f"chunk_{chunk_idx:04d}.npy")

    # Read uploaded audio and convert to float32 numpy
    audio_bytes = await audio.read()
    filename = audio.filename or ""

    if filename.endswith(".npy"):
        # Raw numpy array
        audio_array = np.load(BytesIO(audio_bytes))
    else:
        # Decode audio file (WAV, MP3, etc.) via soundfile or librosa
        import soundfile as sf

        try:
            audio_array, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
            if sr != 16000:
                # Resample to 16kHz
                import librosa

                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        except Exception as sf_err:
            # Fallback to librosa for formats soundfile can't handle
            logger.debug(
                "soundfile decode failed (%s), falling back to librosa", sf_err
            )
            import tempfile

            import librosa

            with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                audio_array, _ = librosa.load(tmp_path, sr=16000, mono=True)
            finally:
                os.unlink(tmp_path)

    audio_array = audio_array.astype(np.float32)

    # Atomic write: write to tmp then rename
    tmp_path = chunk_path.removesuffix(".npy") + ".tmp.npy"
    np.save(tmp_path, audio_array)
    os.rename(tmp_path, chunk_path)

    return {
        "success": True,
        "session_id": session_id,
        "chunk_idx": chunk_idx,
        "samples": len(audio_array),
        "duration_s": round(len(audio_array) / 16000, 3),
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str = Path(...)):
    """Get session status."""
    if session_id not in _SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**_SESSION_STORE[session_id])


@router.delete("/sessions/{session_id}")
async def end_session(session_id: str = Path(...)):
    """End a running session.

    Writes the ``end`` sentinel file so the pipeline exits its chunk loop.
    """
    if session_id not in _SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _SESSION_STORE[session_id]
    session_dir = _session_dir_for_id(session_id)

    # Write end sentinel
    end_path = os.path.join(session_dir, "end")
    with open(end_path, "w") as f:
        pass

    session["status"] = "ended"
    await VIDEO_STORE.update_fields(
        session_id, {"status": "ended", "completed_at": int(time.time())}
    )
    return {"success": True, "session_id": session_id, "status": "ended"}


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    enable_frame_interpolation: Optional[bool] = Form(False),
    frame_interpolation_exp: Optional[int] = Form(1),
    frame_interpolation_scale: Optional[float] = Form(1.0),
    frame_interpolation_model_path: Optional[str] = Form(None),
    output_quality: Optional[str] = Form("default"),
    output_compression: Optional[int] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    # Resolve input upload directory (may be a temp dir when saving is disabled)
    temp_dirs: list[str] = []
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
        temp_dirs.append(uploads_dir)

    # Resolve output directory
    effective_output_path = server_args.output_path
    output_persistent = True
    if "multipart/form-data" not in content_type:
        # JSON body may carry a per-request output_path; checked after parsing below
        pass

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        # Validate image input based on model task type
        image_sources = merge_image_input_list(input_reference, reference_url)
        if task_type.requires_image_input() and not image_sources:
            raise HTTPException(
                status_code=400,
                detail="input_reference or reference_url is required for image-to-video generation",
            )
        try:
            input_path = await _save_first_input_image(
                image_sources, request_id, uploads_dir
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        # Resolve audio input (multipart upload or URL)
        try:
            audio_path = await _save_audio_input(
                audio, audio_url, request_id, uploads_dir
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process audio source: {str(e)}"
            )

        # Parse extra_body JSON (if provided in multipart form) to get fps/num_frames overrides
        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = json.loads(extra_body)
            except Exception:
                extra_from_form = {}

        fps_val = fps if fps is not None else extra_from_form.get("fps")
        num_frames_val = (
            num_frames if num_frames is not None else extra_from_form.get("num_frames")
        )

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            model=model,
            seconds=seconds if seconds is not None else 4,
            size=size,
            fps=fps_val,
            num_frames=num_frames_val,
            seed=seed,
            generator_device=generator_device,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
            enable_frame_interpolation=enable_frame_interpolation,
            frame_interpolation_exp=frame_interpolation_exp,
            frame_interpolation_scale=frame_interpolation_scale,
            frame_interpolation_model_path=frame_interpolation_model_path,
            output_compression=output_compression,
            output_quality=output_quality,
            audio_path=audio_path,
            **(
                {"guidance_scale": guidance_scale} if guidance_scale is not None else {}
            ),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            # If client uses extra_body, merge it into the top-level payload
            payload: Dict[str, Any] = dict(body or {})
            extra = payload.pop("extra_body", None)
            if isinstance(extra, dict):
                # Shallow-merge: only keys like fps/num_frames are expected
                payload.update(extra)
            # openai may turn extra_body to extra_json
            extra_json = payload.pop("extra_json", None)
            if isinstance(extra_json, dict):
                payload.update(extra_json)
            # Validate image input based on model task type
            has_image_input = payload.get("reference_url") or payload.get(
                "input_reference"
            )
            if task_type.requires_image_input() and not has_image_input:
                raise HTTPException(
                    status_code=400,
                    detail="input_reference or reference_url is required for image-to-video generation",
                )
            # for non-multipart/form-data type
            if payload.get("reference_url"):
                try:
                    input_path = await _save_first_input_image(
                        payload.get("reference_url"), request_id, uploads_dir
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    )
                payload["input_reference"] = input_path
            # Resolve audio_url → audio_path for JSON requests
            if payload.get("audio_url") and not payload.get("audio_path"):
                try:
                    audio_path = await _save_audio_input(
                        None, payload.pop("audio_url"), request_id, uploads_dir
                    )
                    payload["audio_path"] = audio_path
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process audio source: {str(e)}",
                    )
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # Resolve per-request output_path override
    effective_output_path = req.output_path or server_args.output_path
    if effective_output_path is None:
        output_tmp = tempfile.mkdtemp(prefix="sglang_output_")
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False

    # Inject resolved output_path so _build_video_sampling_params picks it up
    req.output_path = effective_output_path

    logger.debug(f"Server received from create_video endpoint: req={req}")

    try:
        sampling_params = _build_video_sampling_params(request_id, req)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = _video_job_from_sampling(request_id, req, sampling_params)
    await VIDEO_STORE.upsert(request_id, job)

    # Build Req for scheduler
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )
    # Add diffusers_kwargs if provided
    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(
        _dispatch_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        )
    )
    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    # Normalize order
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await VIDEO_STORE.list_values()

    reverse = order != "asc"
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=reverse)

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    items = [VideoResponse(**j) for j in jobs]
    return VideoListResponse(data=items)


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    # Enrich with chunk-level progress when job is in-flight
    if job.get("status") == "processing":
        try:
            progress_file = os.path.join(_progress_dir(), video_id)
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    parts = f.read().strip().split()
                if len(parts) == 2:
                    current, total = int(parts[0]), int(parts[1])
                    job["chunks_completed"] = current
                    job["num_chunks"] = total
                    job["progress"] = int(100 * current / total) if total > 0 else 0
        except Exception:
            pass  # progress file may be mid-write; skip gracefully

    return VideoResponse(**job)


@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    status = job.get("status", "")
    if status in ("processing", "queued"):
        # Write a cancel sentinel so the pipeline can detect cancellation
        try:
            cancel_path = os.path.join(_progress_dir(), f"{video_id}.cancel")
            with open(cancel_path, "w") as f:
                f.write("")
        except Exception:
            pass
        job["status"] = "cancelling"
        await VIDEO_STORE.update_fields(video_id, {"status": "cancelling"})
    else:
        # Job already finished/failed — just remove it
        await VIDEO_STORE.pop(video_id)
        job["status"] = "deleted"

    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Video has been uploaded to cloud storage. Please use the cloud URL: {job.get('url')}",
        )

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )


def _read_jpeg_as_rgb(path: str) -> np.ndarray:
    """Read a JPEG file and return an RGB uint8 numpy array (H, W, 3)."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _load_audio_for_fmp4(
    audio_path: str, target_sample_rate: int = 48000
) -> Optional[np.ndarray]:
    """Decode audio file to int16 mono numpy at *target_sample_rate* via PyAV.

    Returns None if the file cannot be decoded.
    """
    try:
        import av

        container = av.open(audio_path)
        try:
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=target_sample_rate
            )
            frames = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().flatten()
                    frames.append(arr)
        finally:
            container.close()
        if not frames:
            return None
        return np.concatenate(frames).astype(np.int16)
    except Exception as e:
        logger.warning("Failed to load audio for fMP4 from %s: %s", audio_path, e)
        return None


async def _wait_for_meta(frame_dir: str, timeout: float = 60.0) -> Optional[dict]:
    """Poll for meta.json in *frame_dir*, return parsed dict or None on timeout."""
    meta_path = os.path.join(frame_dir, "meta.json")
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass  # file may be mid-write
        await asyncio.sleep(0.1)
    return None


async def _fmp4_generator(
    frame_dir: str,
    fps: int,
    buffer_frames: int,
    audio_path: Optional[str],
    session_audio_dir: Optional[str] = None,
):
    """Async generator that yields fragmented MP4 bytes (H.264 + optional AAC).

    Mirrors the MJPEG generator pattern: polls for frame_NNNNN.jpg files in
    *frame_dir*, encodes them into an fMP4 container, and yields bytes
    incrementally so browsers can play via Media Source Extensions (MSE).

    For session mode, *session_audio_dir* points to the directory containing
    incremental audio chunk files (chunk_NNNN.npy at 16 kHz float32).
    Audio is resampled to 48 kHz and muxed into the fMP4 stream on-the-fly.
    """
    import av

    AUDIO_SAMPLE_RATE = 48000
    SESSION_AUDIO_INPUT_SR = 16000

    is_session_audio = session_audio_dir is not None

    # ── Load audio (if provided as a single file) ──
    audio_samples: Optional[np.ndarray] = None
    if not is_session_audio and audio_path and os.path.exists(audio_path):
        audio_samples = await asyncio.to_thread(
            _load_audio_for_fmp4, audio_path, AUDIO_SAMPLE_RATE
        )

    # ── Wait for meta.json to get resolution ──
    meta = await _wait_for_meta(frame_dir)
    if meta is None:
        logger.warning("fMP4 stream: timed out waiting for meta.json in %s", frame_dir)
        return

    width = meta.get("width", 512)
    height = meta.get("height", 512)

    # ── Open PyAV fMP4 container writing to BytesIO ──
    buf = BytesIO()
    container = av.open(
        buf,
        mode="w",
        format="mp4",
        options={
            "movflags": "frag_keyframe+empty_moov+default_base_moof",
        },
    )

    # Video stream
    video_stream = container.add_stream("libx264", rate=fps)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "yuv420p"
    video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
    video_stream.time_base = fractions.Fraction(1, fps)
    if is_session_audio:
        # Use a small GOP to produce fMP4 fragments frequently for low-latency
        # streaming while avoiding the bandwidth cost of all-I-frame encoding.
        video_stream.gop_size = 5

    # Audio stream (optional for file mode, always for session mode)
    audio_stream = None
    audio_samples_per_frame = 0
    has_audio = (
        audio_samples is not None and len(audio_samples) > 0
    ) or is_session_audio
    if has_audio:
        audio_stream = container.add_stream("aac", rate=AUDIO_SAMPLE_RATE)
        audio_stream.layout = "mono"
        audio_stream.time_base = fractions.Fraction(1, AUDIO_SAMPLE_RATE)
        audio_samples_per_frame = AUDIO_SAMPLE_RATE // fps

    # ── Session audio state ──
    # For session mode, we load audio chunks incrementally from .npy files.
    # Each chunk is 17920 samples at 16 kHz (1.12s).  We resample to 48 kHz
    # and maintain a buffer that the per-frame encoder drains.
    _session_audio_buf: Optional[np.ndarray] = None  # int16 @ 48 kHz
    _session_audio_buf_pos = 0  # how much of _session_audio_buf has been consumed
    _session_chunk_idx = 0  # next chunk file to try loading

    def _load_session_audio_chunks() -> None:
        """Try to load any new session audio chunks into _session_audio_buf."""
        nonlocal _session_audio_buf, _session_audio_buf_pos, _session_chunk_idx
        loaded_any = False
        while True:
            path = os.path.join(
                session_audio_dir, f"chunk_{_session_chunk_idx:04d}.npy"
            )
            if not os.path.exists(path):
                break
            try:
                raw = np.load(path)  # float32 @ 16 kHz
                # Simple linear-interpolation upsample 16 kHz → 48 kHz (ratio=3)
                ratio = AUDIO_SAMPLE_RATE / SESSION_AUDIO_INPUT_SR
                out_len = int(len(raw) * ratio)
                indices = np.arange(out_len) / ratio
                left = np.floor(indices).astype(np.intp)
                frac = (indices - left).astype(np.float32)
                np.clip(left, 0, len(raw) - 1, out=left)
                right = np.minimum(left + 1, len(raw) - 1)
                resampled = raw[left] * (1 - frac) + raw[right] * frac
                chunk_s16 = np.clip(resampled * 32767, -32768, 32767).astype(np.int16)
                if _session_audio_buf is None:
                    _session_audio_buf = chunk_s16
                    _session_audio_buf_pos = 0
                else:
                    _session_audio_buf = np.concatenate(
                        [
                            _session_audio_buf[_session_audio_buf_pos:],
                            chunk_s16,
                        ]
                    )
                    _session_audio_buf_pos = 0
                _session_chunk_idx += 1
                loaded_any = True
            except Exception as e:
                logger.debug(
                    "Session audio chunk %d load error: %s", _session_chunk_idx, e
                )
                break
        return

    def _get_session_audio_for_frame(n_samples: int) -> Optional[np.ndarray]:
        """Get *n_samples* int16 samples from session audio buffer.

        Returns silence (zeros) if no audio data is available yet, to keep the
        audio track in sync with video and prevent MSE from stalling.
        """
        nonlocal _session_audio_buf, _session_audio_buf_pos
        if not is_session_audio:
            return None
        # Try loading new chunks if buffer is low
        remaining = 0
        if _session_audio_buf is not None:
            remaining = len(_session_audio_buf) - _session_audio_buf_pos
        if remaining < n_samples:
            _load_session_audio_chunks()
            if _session_audio_buf is not None:
                remaining = len(_session_audio_buf) - _session_audio_buf_pos
        if _session_audio_buf is None or remaining <= 0:
            # Return silence to keep audio track in sync with video
            return np.zeros(n_samples, dtype=np.int16)
        end = _session_audio_buf_pos + n_samples
        chunk = _session_audio_buf[_session_audio_buf_pos:end]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        _session_audio_buf_pos = min(end, len(_session_audio_buf))
        return chunk

    def _flush_buf() -> bytes:
        """Read new bytes from the BytesIO buffer and reset it."""
        data = buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        return data

    # ── Pre-buffer phase ──
    stall_count = 0
    max_stall = 600  # 60s timeout

    if buffer_frames > 0:
        while True:
            ready = 0
            while os.path.exists(os.path.join(frame_dir, f"frame_{ready:05d}.jpg")):
                ready += 1
            done_path = os.path.join(frame_dir, "done")
            if ready >= buffer_frames or os.path.exists(done_path):
                break
            stall_count += 1
            if stall_count >= max_stall:
                container.close()
                return
            await asyncio.sleep(0.1)
        stall_count = 0

    # ── Streaming loop ──
    frame_idx = 0
    frame_interval = 1.0 / fps
    playback_epoch = None
    audio_pos = 0  # current position in audio_samples

    def _encode_one_frame(fidx: int, rgb) -> Optional[bytes]:
        """Encode a single video+audio frame and return flushed fMP4 bytes."""
        nonlocal audio_pos

        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = fidx
        video_frame.time_base = fractions.Fraction(1, fps)
        for packet in video_stream.encode(video_frame):
            container.mux(packet)

        if audio_stream is not None:
            audio_chunk = None
            if is_session_audio:
                audio_chunk = _get_session_audio_for_frame(audio_samples_per_frame)
            elif audio_samples is not None:
                start = audio_pos
                end = min(start + audio_samples_per_frame, len(audio_samples))
                if start < len(audio_samples):
                    audio_chunk = audio_samples[start:end]
                    if len(audio_chunk) < audio_samples_per_frame:
                        audio_chunk = np.pad(
                            audio_chunk, (0, audio_samples_per_frame - len(audio_chunk))
                        )
                audio_pos = end
            if audio_chunk is not None:
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_chunk.reshape(1, -1), format="s16", layout="mono"
                )
                audio_frame.sample_rate = AUDIO_SAMPLE_RATE
                audio_frame.pts = fidx * audio_samples_per_frame
                audio_frame.time_base = fractions.Fraction(1, AUDIO_SAMPLE_RATE)
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)

        return _flush_buf()

    async def _pace_playback(fidx: int) -> None:
        """Wall-clock pacing for non-session mode."""
        nonlocal playback_epoch
        if is_session_audio:
            return
        now = asyncio.get_event_loop().time()
        if playback_epoch is None:
            playback_epoch = now
        target = playback_epoch + fidx * frame_interval
        delay = target - now
        if delay > 0:
            await asyncio.sleep(delay)

    # Wrap the streaming loop in try/finally to ensure the container is
    # closed even when the client disconnects (GeneratorExit at yield).
    _container_closed = False
    try:
        while True:
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.jpg")
            if os.path.exists(frame_path):
                stall_count = 0
                try:
                    rgb = await asyncio.to_thread(_read_jpeg_as_rgb, frame_path)
                    new_bytes = _encode_one_frame(frame_idx, rgb)
                    if new_bytes:
                        if is_session_audio and frame_idx < 5:
                            logger.info(
                                "fMP4 session: yielding %d bytes at frame %d",
                                len(new_bytes),
                                frame_idx,
                            )
                        yield new_bytes

                except Exception as e:
                    logger.debug("fMP4 encode error at frame %d: %s", frame_idx, e)
                    continue  # retry on next iteration

                frame_idx += 1
                await _pace_playback(frame_idx)
            else:
                # Check if generation is done
                done_path = os.path.join(frame_dir, "done")
                if os.path.exists(done_path):
                    # Drain remaining frames
                    while True:
                        frame_path = os.path.join(
                            frame_dir, f"frame_{frame_idx:05d}.jpg"
                        )
                        if not os.path.exists(frame_path):
                            break
                        try:
                            rgb = await asyncio.to_thread(
                                _read_jpeg_as_rgb, frame_path
                            )
                            new_bytes = _encode_one_frame(frame_idx, rgb)
                            if new_bytes:
                                yield new_bytes
                        except Exception:
                            break

                        frame_idx += 1
                        await _pace_playback(frame_idx)
                    break

                stall_count += 1
                if stall_count >= max_stall:
                    break
                await asyncio.sleep(0.1)

        # ── Flush encoders and close container ──
        try:
            for packet in video_stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                for packet in audio_stream.encode():
                    container.mux(packet)
            container.close()
            _container_closed = True
            final_bytes = _flush_buf()
            if final_bytes:
                yield final_bytes
        except Exception as e:
            logger.debug("fMP4 finalize error: %s", e)
    finally:
        if not _container_closed:
            try:
                container.close()
            except Exception:
                pass

    # Schedule cleanup (session mode cleanup is handled by _dispatch_session_async)
    if not is_session_audio:
        asyncio.create_task(_cleanup_frame_dir(frame_dir, delay=5.0))


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str = Path(...),
    format: str = Query("mjpeg", pattern="^(mjpeg|fmp4)$"),
    fps: int = Query(25, ge=1, le=60),
    jpeg_quality: int = Query(85, ge=1, le=100),
    buffer_frames: int = Query(
        0,
        ge=0,
        le=500,
        description="Number of frames to pre-buffer before starting playback. "
        "When generation is faster than real-time, buffering absorbs timing "
        "jitter and ensures video duration matches audio exactly. "
        "Recommended: 5*fps (e.g. 125 for 25fps = 5 second buffer).",
    ),
):
    """Stream video frames as MJPEG or fMP4.

    Use ``format=mjpeg`` (default) for MJPEG multipart response, or
    ``format=fmp4`` for fragmented MP4 (H.264 + AAC) playable via MSE.

    Polls the .frames/{video_id}/ directory for frame_NNNNN.jpg files written
    by the GPU worker pipeline and yields them as a stream.
    """
    frame_dir = _frame_dir_for_job(video_id)

    # Validate job exists
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if format == "fmp4":
        audio_path = job.get("audio_path")
        # For sessions, pass the audio_chunks directory for incremental audio muxing
        session_audio_dir = None
        if video_id in _SESSION_STORE:
            session_dir = _session_dir_for_id(video_id)
            chunks_dir = os.path.join(session_dir, "audio_chunks")
            if os.path.isdir(chunks_dir):
                session_audio_dir = chunks_dir
        return StreamingResponse(
            _fmp4_generator(
                frame_dir,
                fps,
                buffer_frames,
                audio_path,
                session_audio_dir=session_audio_dir,
            ),
            media_type="video/mp4",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _mjpeg_generator():
        frame_idx = 0
        frame_interval = 1.0 / fps
        stall_count = 0
        max_stall = 600  # 600 × 0.1s = 60s timeout for first chunk / compile
        playback_epoch = None  # wall-clock ref for drift-free pacing

        # ── Pre-buffer: wait until enough frames exist before streaming ──
        # This absorbs timing jitter and prevents the video from running
        # longer than the audio when generation is faster than real-time.
        if buffer_frames > 0:
            while True:
                # Count available frames
                ready = 0
                while os.path.exists(os.path.join(frame_dir, f"frame_{ready:05d}.jpg")):
                    ready += 1
                done_path = os.path.join(frame_dir, "done")
                if ready >= buffer_frames or os.path.exists(done_path):
                    break
                stall_count += 1
                if stall_count >= max_stall:
                    return  # timeout
                await asyncio.sleep(0.1)
            stall_count = 0

        while True:
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.jpg")
            if os.path.exists(frame_path):
                stall_count = 0
                try:
                    with open(frame_path, "rb") as f:
                        jpeg_data = f.read()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n"
                        b"\r\n" + jpeg_data + b"\r\n"
                    )
                except Exception:
                    pass  # file may be mid-write; retry next iteration
                    continue
                frame_idx += 1
                # Wall-clock pacing: sleep until the target time for the
                # next frame so per-frame I/O overhead doesn't accumulate.
                now = asyncio.get_event_loop().time()
                if playback_epoch is None:
                    playback_epoch = now
                target = playback_epoch + frame_idx * frame_interval
                delay = target - now
                if delay > 0:
                    await asyncio.sleep(delay)
                # else: behind schedule (after stall), serve next immediately
            else:
                # Check if generation is done
                done_path = os.path.join(frame_dir, "done")
                if os.path.exists(done_path):
                    # Drain any remaining frames with the same wall-clock pacing
                    while True:
                        frame_path = os.path.join(
                            frame_dir, f"frame_{frame_idx:05d}.jpg"
                        )
                        if not os.path.exists(frame_path):
                            break
                        try:
                            with open(frame_path, "rb") as f:
                                jpeg_data = f.read()
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: "
                                + str(len(jpeg_data)).encode()
                                + b"\r\n"
                                b"\r\n" + jpeg_data + b"\r\n"
                            )
                        except Exception:
                            break
                        frame_idx += 1
                        if playback_epoch is not None:
                            target = playback_epoch + frame_idx * frame_interval
                            delay = target - asyncio.get_event_loop().time()
                            if delay > 0:
                                await asyncio.sleep(delay)
                    break
                stall_count += 1
                if stall_count >= max_stall:
                    break
                await asyncio.sleep(0.1)

        # Schedule cleanup
        asyncio.create_task(_cleanup_frame_dir(frame_dir, delay=5.0))

    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{video_id}/events")
async def stream_video_events(
    video_id: str = Path(...),
    include_frames: bool = Query(False),
):
    """Stream generation progress as Server-Sent Events (SSE).

    Events emitted:
      - ``event: meta``  — once, with generation metadata (num_chunks, fps, …)
      - ``event: chunk`` — per chunk: chunk_idx, frames_start, frames_end, progress_pct
      - ``event: frame`` — per frame (opt-in): frame_idx + base64 JPEG data
      - ``event: done``  — generation complete
      - ``data: [DONE]`` — stream end (matches SRT convention)
    """
    import base64

    frame_dir = _frame_dir_for_job(video_id)

    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    async def _sse_generator():
        meta_sent = False
        chunks_seen = 0
        frames_seen = 0
        stall_count = 0
        max_stall = 600  # 60s timeout

        while True:
            # Emit meta event once the meta.json appears
            if not meta_sent:
                meta_path = os.path.join(frame_dir, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                        meta_sent = True
                        frames_per_chunk = meta.get("frames_per_chunk", 28)
                        num_chunks = meta.get("num_chunks") or 0
                    except Exception:
                        pass

            if not meta_sent:
                stall_count += 1
                if stall_count >= max_stall:
                    break
                await asyncio.sleep(0.1)
                continue

            # Check for new frames (detect chunk boundaries)
            new_frames_found = False
            while True:
                frame_path = os.path.join(frame_dir, f"frame_{frames_seen:05d}.jpg")
                if not os.path.exists(frame_path):
                    break
                new_frames_found = True

                if include_frames:
                    try:
                        with open(frame_path, "rb") as f:
                            jpeg_data = f.read()
                        b64 = base64.b64encode(jpeg_data).decode("ascii")
                        event_data = json.dumps(
                            {"frame_idx": frames_seen, "data_b64": b64}
                        )
                        yield f"event: frame\ndata: {event_data}\n\n"
                    except Exception:
                        pass

                frames_seen += 1

                # Emit chunk event at chunk boundaries
                if frames_seen % frames_per_chunk == 0:
                    chunk_idx = (frames_seen // frames_per_chunk) - 1
                    progress_pct = (
                        int(100 * (chunk_idx + 1) / num_chunks) if num_chunks > 0 else 0
                    )
                    chunk_data = json.dumps(
                        {
                            "chunk_idx": chunk_idx,
                            "frames_start": chunk_idx * frames_per_chunk,
                            "frames_end": frames_seen,
                            "progress_pct": progress_pct,
                        }
                    )
                    yield f"event: chunk\ndata: {chunk_data}\n\n"
                    chunks_seen = chunk_idx + 1

            if new_frames_found:
                stall_count = 0
            else:
                # Check if done
                done_path = os.path.join(frame_dir, "done")
                if os.path.exists(done_path):
                    # Emit chunk event for any remaining partial chunk
                    if frames_seen % frames_per_chunk != 0 and frames_seen > 0:
                        chunk_idx = frames_seen // frames_per_chunk
                        chunk_data = json.dumps(
                            {
                                "chunk_idx": chunk_idx,
                                "frames_start": chunk_idx * frames_per_chunk,
                                "frames_end": frames_seen,
                                "progress_pct": 100,
                            }
                        )
                        yield f"event: chunk\ndata: {chunk_data}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                stall_count += 1
                if stall_count >= max_stall:
                    break
                await asyncio.sleep(0.1)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
