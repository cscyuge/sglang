from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamFramesRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class FramePacket(_message.Message):
    __slots__ = ("chunk_index", "h264_data", "pcm_audio", "pts_ms", "is_keyframe", "video_meta")
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    H264_DATA_FIELD_NUMBER: _ClassVar[int]
    PCM_AUDIO_FIELD_NUMBER: _ClassVar[int]
    PTS_MS_FIELD_NUMBER: _ClassVar[int]
    IS_KEYFRAME_FIELD_NUMBER: _ClassVar[int]
    VIDEO_META_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    h264_data: bytes
    pcm_audio: bytes
    pts_ms: int
    is_keyframe: bool
    video_meta: VideoMeta
    def __init__(self, chunk_index: _Optional[int] = ..., h264_data: _Optional[bytes] = ..., pcm_audio: _Optional[bytes] = ..., pts_ms: _Optional[int] = ..., is_keyframe: bool = ..., video_meta: _Optional[_Union[VideoMeta, _Mapping]] = ...) -> None: ...

class VideoMeta(_message.Message):
    __slots__ = ("width", "height", "fps", "codec_extra_data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    CODEC_EXTRA_DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    fps: int
    codec_extra_data: bytes
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., fps: _Optional[int] = ..., codec_extra_data: _Optional[bytes] = ...) -> None: ...
