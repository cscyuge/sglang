import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    file_path: Optional[str] = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class ImageGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "auto"
    response_format: Optional[str] = "url"  # url | b64_json
    size: Optional[str] = "1024x1024"  # e.g., 1024x1024
    style: Optional[str] = "vivid"
    background: Optional[str] = "auto"  # transparent | opaque | auto
    output_format: Optional[str] = None  # png | jpeg | webp
    user: Optional[str] = None
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    negative_prompt: Optional[str] = None
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    enable_teacache: Optional[bool] = False
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = (
        "queued"  # queued | processing | completed | failed | cancelling | cancelled
    )
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    num_outputs: Optional[int] = None
    stream_url: Optional[str] = None
    events_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None
    num_chunks: Optional[int] = None
    chunks_completed: Optional[int] = None


class VideoGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    input_reference: Optional[str] = None
    reference_url: Optional[str] = None
    model: Optional[str] = None
    n: Optional[int] = 1
    num_outputs_per_prompt: Optional[int] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    negative_prompt: Optional[str] = None
    max_sequence_length: Optional[int] = None
    flow_shift: Optional[float] = None
    enable_teacache: Optional[bool] = False
    # Frame interpolation
    enable_frame_interpolation: Optional[bool] = False
    frame_interpolation_exp: Optional[int] = 1  # 1=2×, 2=4×
    frame_interpolation_scale: Optional[float] = 1.0
    frame_interpolation_model_path: Optional[str] = None
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    output_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # FlashTalk audio input
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    audio_encode_mode: Optional[str] = None  # "stream" or "batch"
    # Performance profiling
    perf_dump_path: Optional[str] = None


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str


class RealtimeArtcOutputConfig(BaseModel):
    token: str
    channel: str
    userid: Optional[str] = None
    queue_size: Optional[int] = Field(default=None, ge=1)


class RealtimeVideoGenerationsRequest(VideoGenerationsRequest):
    type: Literal["init"]
    # WebSocket does not support multipart/form-data image uploads.
    first_frame: Optional[bytes | str] = None
    condition_inputs: Optional[Dict[str, Any]] = None
    max_chunks: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = 42
    guidance_scale: Optional[float] = 1.0
    size: Optional[str] = "832x480"
    profile: Optional[bool] = False
    num_profiled_timesteps: Optional[int] = None
    profile_all_stages: Optional[bool] = False
    output_transport: Optional[Literal["ws", "websocket", "artc"]] = "ws"
    artc: Optional[RealtimeArtcOutputConfig] = None
    realtime_output_format: Optional[Literal["raw", "webp", "jpeg", "h264"]] = None
    realtime_preview_max_width: Optional[int] = None
    realtime_output_pacing: Optional[bool] = False
    realtime_causal_sink_size: Optional[int] = None
    realtime_causal_kv_cache_num_frames: Optional[int] = None


class RealtimeEvent(BaseModel):
    type: Literal["event"]
    kind: str
    payload: Any = None
    event_id: Optional[int] = None


class WebRTCIceServer(BaseModel):
    urls: Union[str, List[str]]
    username: Optional[str] = None
    credential: Optional[str] = None


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str = "offer"
    fps: int = Field(default=25, ge=1, le=60)
    include_audio: bool = True
    buffer_frames: int = Field(default=0, ge=0, le=500)
    ice_servers: Optional[List[WebRTCIceServer]] = None


class WebRTCAnswerResponse(BaseModel):
    sdp: str
    type: str = "answer"


# Session API protocol models (live-streaming FlashTalk)
class SessionResponse(BaseModel):
    session_id: str
    object: str = "video.session"
    status: str = "created"  # created | running | ended | failed
    runtime: Optional[str] = None
    model_id: Optional[str] = None
    wan_s2v_realtime: Optional[bool] = None
    stream_r1_mode: Optional[bool] = None
    num_frame_per_block: Optional[int] = None
    stream_url: Optional[str] = None
    events_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    artc_channel: Optional[str] = None
    chunk_timeline_path: Optional[str] = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    chunks_received: int = 0
    chunks_processed: int = 0
    error: Optional[Dict[str, Any]] = None


# Mesh API protocol models
class MeshResponse(BaseModel):
    id: str
    object: str = "mesh"
    model: str = ""
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    format: str = "glb"
    url: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class MeshGenerationsRequest(BaseModel):
    prompt: str = "generate 3d mesh"
    input_image: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    generator_device: Optional[str] = "cuda"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    output_format: Optional[str] = "glb"


class MeshListResponse(BaseModel):
    data: List[MeshResponse]
    object: str = "list"


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None
