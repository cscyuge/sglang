# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import pickle
import struct
import sys
import urllib.error
from types import SimpleNamespace

import msgspec.msgpack
import numpy as np
import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimePostprocessConfig,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_output_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_frame_processor import (
    RemoteCodeFormerFrameProcessor,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_sink import (
    ARTC_CONTENT_TYPE,
    ArtcRealtimeOutputSink,
    RemoteCodeFormerArtcOutputSink,
    WebSocketRealtimeOutputSink,
    create_realtime_output_sink,
    create_realtime_output_sink_async,
    normalize_realtime_output_transport,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.errors import RealtimeProtocolError
from sglang.multimodal_gen.runtime.utils.realtime_frame_store import (
    attach_raw_rgb_frame_store_writer_request,
    create_raw_rgb_frame_store_handles,
    pop_raw_rgb_frame_store_writer_request,
    start_raw_rgb_frame_store_writer_request,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
    build_delta_gzip_raw_rgb_payload,
    build_raw_rgb_frame_batches,
    restore_delta_gzip_raw_rgb_payload,
)


def _unpack_frame_batch_messages(payloads):
    messages = []
    payload_iter = iter(payloads)
    for payload in payload_iter:
        message = msgspec.msgpack.decode(payload)
        message_type = message.pop("type")
        if message_type == "frame_batch":
            frame_payload = message.pop("payload")
        else:
            assert message_type == "frame_batch_header"
            frame_payload = next(payload_iter)
        messages.append((message, frame_payload))
    return messages


def test_raw_rgb_frame_batches_preserve_frame_bytes_and_metadata():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    grayscale = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rgba = np.array(
        [
            [[5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20]],
        ],
        dtype=np.uint8,
    )

    def post_process_sample(*_args, **_kwargs):
        return [grayscale, rgba]

    frame_batches, metadata = build_raw_rgb_frame_batches(
        object(),
        req,
        output_batch,
        post_process_sample,
    )

    timings = metadata.pop("timings", None)
    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert timings["raw_frame_materialize_ms"] >= 0
    assert len(frame_batches) == 1
    assert frame_batches[0][0] == bytes([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    assert frame_batches[0][1] == bytes([5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
    assert RAW_RGB_CONTENT_TYPE == "application/x-raw-rgb"


def test_raw_rgb_frame_batches_use_tensor_fast_path_without_postprocess():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    output = torch.tensor(
        [[[[[0.0]], [[0.25]]], [[[0.5]], [[0.75]]], [[[1.0]], [[1.0]]]]]
    )

    def post_process_sample(*_args, **_kwargs):
        raise AssertionError("tensor realtime output should not use postprocess")

    frame_batches, metadata = build_raw_rgb_frame_batches(
        output,
        req,
        output_batch,
        post_process_sample,
    )

    timings = metadata.pop("timings", None)
    assert metadata == {
        "format": "rgb24",
        "width": 1,
        "height": 1,
        "channels": 3,
        "bytes_per_frame": 3,
    }
    assert timings["raw_frame_materialize_ms"] >= 0
    assert frame_batches == [[bytes([0, 127, 255]), bytes([63, 191, 255])]]


def test_raw_rgb_realtime_output_adapter_reads_frame_store_handles():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-frame-store",
            width=1,
            height=1,
            fps=16,
            enable_upscaling=False,
            realtime_event_id=7,
        )
        output = torch.tensor(
            [[[[[0.0]], [[0.25]]], [[[0.5]], [[0.75]]], [[[1.0]], [[1.0]]]]]
        )
        handles, metadata = create_raw_rgb_frame_store_handles(output, batch)
        result = OutputBatch(
            raw_frame_store_handles=handles,
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )
        attach_raw_rgb_frame_store_writer_request(
            result,
            output=output,
            handles=handles,
            request_id=batch.request_id,
            chunk_idx=batch.block_idx,
        )
        write_request = pop_raw_rgb_frame_store_writer_request(result)
        assert write_request is not None
        pickle.dumps(result)
        start_raw_rgb_frame_store_writer_request(write_request)

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, handles

    payloads, stats, handles = asyncio.run(run())

    [(header, payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert header["encoding"] == "raw"
    assert header["event_id"] == 7
    assert header["num_frames"] == 2
    assert payload == bytes([0, 127, 255, 63, 191, 255])
    assert stats["raw_bytes"] == 6
    assert stats["num_frames"] == 2
    assert stats["frame_store_wait_ms"] >= 0
    assert stats["frame_store_read_ms"] >= 0
    assert stats["frame_store_materialize_ms"] >= 0
    assert stats["frame_store_producer_wait_ms"] >= 0
    assert stats["frame_store_gpu_copy_ms"] >= 0
    assert stats["frame_store_mmap_write_ms"] >= 0
    assert stats["frame_store_producer_decode_ms"] >= 0
    assert stats["frame_store_producer_post_ms"] >= 0
    assert stats["frame_store_producer_clone_ms"] >= 0
    assert stats["frame_store_producer_total_ms"] >= 0
    assert stats["frame_store_producer_denoise_ms"] >= 0
    assert stats["frame_store_producer_refresh_ms"] >= 0
    assert stats["frame_store_producer_denoise_to_ready_ms"] >= 0
    assert all(not os.path.exists(handle.path) for handle in handles)


def test_output_batch_uses_raw_frame_transport_names():
    output_batch = OutputBatch(
        raw_frame_batches=[[b"rgb"]],
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
        raw_frame_metadata={"format": "rgb24"},
    )

    assert output_batch.raw_frame_batches == [[b"rgb"]]
    assert output_batch.raw_frame_content_type == RAW_RGB_CONTENT_TYPE
    assert output_batch.raw_frame_metadata == {"format": "rgb24"}


def test_delta_gzip_raw_rgb_payload_roundtrips_exactly():
    frames = [
        bytes([1, 2, 3, 4, 5, 6]),
        bytes([1, 2, 4, 4, 6, 6]),
        bytes([2, 2, 4, 5, 6, 7]),
    ]

    payload = build_delta_gzip_raw_rgb_payload(frames)
    restored = restore_delta_gzip_raw_rgb_payload(
        payload,
        bytes_per_frame=6,
        num_frames=3,
    )

    assert restored == b"".join(frames)


def test_raw_rgb_realtime_output_adapter_uses_lossless_raw_payload_by_default():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, frame0 + frame1

    payloads, stats, expected_frames = asyncio.run(run())

    [(first_header, first_payload)] = _unpack_frame_batch_messages(payloads)
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert first_header["encoding"] == "raw"
    assert first_header["event_id"] == 3
    assert first_header["format"] == "rgb24"
    assert first_header["channels"] == 3
    assert first_header["bytes_per_frame"] == 3000
    assert first_header["raw_size"] == 6000
    assert first_header["total_size"] == len(first_payload)
    assert first_header["num_frames"] == 2
    assert first_header["num_frame_batches"] == 1
    assert first_header["frame_batch_index"] == 0
    assert "delta_reference" not in first_header
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 2
    assert first_payload == expected_frames


def test_raw_rgb_realtime_output_adapter_offloads_default_lossless_payload_build(
    monkeypatch,
):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )

    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-offload-raw",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, frame0 + frame1

    payloads, expected_frames = asyncio.run(run())

    assert [call[0] for call in calls] == [
        realtime_output_adapter._build_transport_payload,
    ]
    [(first_header, first_payload)] = _unpack_frame_batch_messages(payloads)
    assert first_header["encoding"] == "raw"
    assert "delta_reference" not in first_header
    assert first_payload == expected_frames


def test_raw_rgb_realtime_output_adapter_can_send_uncompressed_raw_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-raw",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, frame0 + frame1

    payloads, stats, expected_frames = asyncio.run(run())

    [(first_header, first_payload)] = _unpack_frame_batch_messages(payloads)
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert first_header["encoding"] == "raw"
    assert first_header["raw_size"] == 6000
    assert first_header["total_size"] == 6000
    assert first_header["num_frames"] == 2
    assert first_header["num_frame_batches"] == 1
    assert first_header["frame_batch_index"] == 0
    assert first_payload == expected_frames
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 2
    assert stats["ws_payload_bytes"] == sum(len(payload) for payload in payloads)


def test_raw_rgb_realtime_output_adapter_can_send_h264_annexb_chunks(monkeypatch):
    def fake_h264_encoder(transport_frames, *, width, height, fps, crf):
        assert len(transport_frames) == 2
        assert width == 2
        assert height == 2
        assert fps == 16
        assert crf == 27
        return b"\x00\x00\x00\x01fake-h264"

    monkeypatch.setattr(
        realtime_output_adapter,
        "_encode_raw_rgb_frames_to_h264_annexb",
        fake_h264_encoder,
    )

    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([255, 0, 0]) * 4
        frame1 = bytes([0, 255, 0]) * 4
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-h264",
            width=2,
            height=2,
            fps=16,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_output_format="h264",
            output_compression=27,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 2,
                "channels": 3,
                "bytes_per_frame": 12,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    [(header, payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == realtime_output_adapter.H264_FRAME_CONTENT_TYPE
    assert header["format"] == "h264_annexb"
    assert header["encoding"] == "h264_annexb"
    assert header["codec"] == "h264"
    assert header["source_format"] == "rgb24"
    assert header["pixel_format"] == "yuv420p"
    assert header["width"] == 2
    assert header["height"] == 2
    assert header["fps"] == 16
    assert header["raw_size"] == 24
    assert header["num_frames"] == 2
    assert payload == b"\x00\x00\x00\x01fake-h264"
    assert stats["content_type"] == realtime_output_adapter.H264_FRAME_CONTENT_TYPE
    assert stats["raw_bytes"] == 24
    assert stats["ws_payload_bytes"] == sum(len(payload) for payload in payloads)


def test_raw_rgb_realtime_output_adapter_does_not_require_previous_frame_reference():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        base_batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
        )
        next_batch = SimpleNamespace(
            block_idx=1,
            request_id="req-2",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
        )
        metadata = {
            "format": "rgb24",
            "width": 2,
            "height": 1,
            "channels": 3,
            "bytes_per_frame": 6,
        }
        first = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3, 4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )
        second = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 4, 4, 6, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )

        await adapter.send(ws, SimpleNamespace(), first, base_batch)
        await adapter.send(ws, SimpleNamespace(), second, next_batch)
        return ws.payloads

    payloads = asyncio.run(run())

    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert second_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert "delta_reference" not in first_header
    assert "delta_reference" not in second_header
    assert first_payload == bytes([1, 2, 3, 4, 5, 6])
    assert second_payload == bytes([1, 2, 4, 4, 6, 6])


def test_raw_rgb_realtime_output_adapter_splits_large_frame_batches():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frames = [bytes([idx, idx + 1, idx + 2]) for idx in range(17)]
        batch = SimpleNamespace(
            block_idx=4,
            request_id="req-split",
            width=1,
            height=1,
            enable_upscaling=False,
            realtime_event_id=12,
        )
        result = OutputBatch(
            raw_frame_batches=[frames],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    headers = [header for header, _ in _unpack_frame_batch_messages(payloads)]
    assert len(headers) == 2
    assert [header["chunk_index"] for header in headers] == [4, 4]
    assert [header["frame_batch_index"] for header in headers] == [0, 1]
    assert [header["num_frame_batches"] for header in headers] == [2, 2]
    assert [header["num_frames"] for header in headers] == [16, 1]
    assert [header["is_final_frame_batch"] for header in headers] == [False, True]
    assert "delta_reference" not in headers[0]
    assert "delta_reference" not in headers[1]
    assert stats["num_batches"] == 2
    assert stats["num_frames"] == 17


def test_raw_rgb_realtime_output_adapter_sends_large_payload_separately():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame = bytes([7]) * (72 * 1024)
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-large",
            width=len(frame) // 3,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": len(frame) // 3,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": len(frame),
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    assert len(payloads) == 2
    header = msgspec.msgpack.decode(payloads[0])
    assert header["type"] == "frame_batch_header"
    assert "payload" not in header
    assert header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert header["total_size"] == len(payloads[1])
    assert payloads[1] == bytes([7]) * (72 * 1024)
    assert stats["raw_bytes"] == len(payloads[1])
    assert stats["ws_payload_bytes"] == len(payloads[0]) + len(payloads[1])
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_can_send_webp_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["format"] == "webp"
    assert header["encoding"] == "webp"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"RIFF")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_offloads_preview_encoding(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )

    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame_count = realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE + 1
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp-offload",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[
                [
                    bytes([idx % 256, 0, 0, 0, 255, idx % 256])
                    for idx in range(frame_count)
                ]
            ],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads

    payloads = asyncio.run(run())

    assert [call[0] for call in calls] == [
        realtime_output_adapter._encode_rgb_frame_to_webp
    ] * (realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE + 1)
    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert first_header["encoding"] == "webp"
    assert first_header["num_frames"] == (
        realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE
    )
    assert first_header["frame_batch_index"] == 0
    assert first_header["num_frame_batches"] == 2
    assert first_header["is_final_frame_batch"] is False
    assert len(first_header["payload_lengths"]) == (
        realtime_output_adapter.ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE
    )
    assert second_header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert second_header["encoding"] == "webp"
    assert second_header["num_frames"] == 1
    assert second_header["frame_batch_index"] == 1
    assert second_header["num_frame_batches"] == 2
    assert second_header["is_final_frame_batch"] is True
    assert len(second_header["payload_lengths"]) == 1
    assert first_payload.startswith(b"RIFF")
    assert second_payload.startswith(b"RIFF")


def test_raw_rgb_realtime_output_adapter_can_send_jpeg_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-jpeg",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="jpeg",
            output_compression=85,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == JPEG_FRAME_CONTENT_TYPE
    assert header["format"] == "jpeg"
    assert header["encoding"] == "jpeg"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"\xff\xd8")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_realtime_output_transport_defaults_to_websocket():
    request = RealtimeVideoGenerationsRequest(type="init", prompt="p", size="1x1")
    session = GenerateSession()
    session.set_request(request)

    sink = create_realtime_output_sink(SimpleNamespace(), session)

    assert normalize_realtime_output_transport(request) == "ws"
    assert isinstance(sink, WebSocketRealtimeOutputSink)
    assert sink.build_init_ack() == {"output_transport": "ws"}


def test_realtime_output_transport_requires_artc_config():
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="p",
        size="1x1",
        output_transport="artc",
    )
    session = GenerateSession()
    session.set_request(request)

    try:
        create_realtime_output_sink(SimpleNamespace(), session)
    except RealtimeProtocolError as exc:
        assert exc.code == "missing_artc_config"
    else:
        raise AssertionError("ARTC output transport should require artc config")


def test_artc_realtime_output_sink_pushes_raw_rgb_chunks_asynchronously(monkeypatch):
    class _FakeArtcPusher:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.started = False
            self.stopped = False
            _FakeArtcPusher.instances.append(self)

        def start_async(self):
            self.started = True

        @property
        def failed(self):
            return False

        def push_chunk(self, frames_np, **kwargs):
            self.calls.append((frames_np.copy(), kwargs))

        def stop(self, timeout=10.0):
            self.stop_timeout = timeout
            self.stopped = True

    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.utils.artc_pusher",
        SimpleNamespace(ArtcPusher=_FakeArtcPusher),
    )

    async def run():
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="p",
            size="1x1",
            fps=16,
            output_transport="artc",
            artc={"token": "token-1", "channel": "channel-1", "queue_size": 2},
        )
        session = GenerateSession()
        session.set_request(request)
        sink = create_realtime_output_sink(SimpleNamespace(), session)
        batch = SimpleNamespace(
            block_idx=3,
            request_id="req-artc",
            width=1,
            height=1,
            fps=16,
            enable_upscaling=False,
            realtime_event_id=5,
            extra={
                "wan_s2v_audio_window": np.array([0, 16384], dtype=np.int16),
                "wan_s2v_audio_window_meta": {"sample_count": 2},
            },
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3]), bytes([4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )

        stats = await sink.send(session, result, batch)
        await sink.close()
        return sink, stats

    sink, stats = asyncio.run(run())

    assert isinstance(sink, ArtcRealtimeOutputSink)
    assert stats["content_type"] == ARTC_CONTENT_TYPE
    assert stats["num_frames"] == 2
    assert stats["num_batches"] == 1
    assert stats["frame_shape"] == (1, 1, 3)
    assert stats["raw_bytes"] == 6
    assert stats["artc_queue_size"] == 1

    [pusher] = _FakeArtcPusher.instances
    assert pusher.started is True
    assert pusher.stopped is True
    assert pusher.kwargs["artc_channel"] == "channel-1"
    assert pusher.kwargs["width"] == 1
    assert pusher.kwargs["height"] == 1
    assert pusher.kwargs["fps"] == 16
    [(frames_np, push_kwargs)] = pusher.calls
    assert frames_np.shape == (2, 1, 1, 3)
    assert frames_np.dtype == np.uint8
    assert frames_np.tolist() == [[[[1, 2, 3]]], [[[4, 5, 6]]]]
    assert push_kwargs["chunk_idx"] == 3
    assert push_kwargs["audio_chunk_idx"] == 3
    assert push_kwargs["audio_loaded"] is True
    assert push_kwargs["audio_chunk_meta"] == {"sample_count": 2}
    np.testing.assert_allclose(push_kwargs["audio_16k"], np.array([0.0, 0.5]))


def test_artc_realtime_output_sink_applies_remote_codeformer_processor(monkeypatch):
    class _FakeArtcPusher:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            _FakeArtcPusher.instances.append(self)

        def start_async(self):
            pass

        @property
        def failed(self):
            return False

        def push_chunk(self, frames_np, **kwargs):
            self.calls.append((frames_np.copy(), kwargs))

        def stop(self, timeout=10.0):
            self.stop_timeout = timeout

    class _FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.headers = {
                "X-Width": "2",
                "X-Height": "2",
                "X-Num-Frames": "2",
                "X-Pix-Fmt": "rgb24",
                "X-Timing": "frames=2 total=11ms",
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.payload

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = request.data
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        frame0 = bytes([10, 20, 30] * 4)
        frame1 = bytes([40, 50, 60] * 4)
        return _FakeResponse(frame0 + frame1)

    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.utils.artc_pusher",
        SimpleNamespace(ArtcPusher=_FakeArtcPusher),
    )
    monkeypatch.setenv(
        "SGLANG_REALTIME_CODEFORMER_ENDPOINT",
        "http://codeformer.local/v1/realtime/sr/raw",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_frame_processor.urllib.request.urlopen",
        fake_urlopen,
    )

    async def run():
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="p",
            size="1x1",
            fps=16,
            output_transport="artc",
            artc={"token": "token-1", "channel": "channel-1", "queue_size": 2},
            realtime_postprocess={
                "type": "codeformer",
                "scale": 2,
                "timeout_ms": 123,
            },
        )
        session = GenerateSession()
        session.set_request(request)
        sink = create_realtime_output_sink(SimpleNamespace(), session)
        ack = sink.build_init_ack()
        batch = SimpleNamespace(
            block_idx=3,
            request_id="req-artc",
            width=1,
            height=1,
            fps=16,
            enable_upscaling=False,
            realtime_event_id=5,
            extra={},
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3]), bytes([4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )
        stats = await sink.send(session, result, batch)
        await sink.close()
        return ack, stats

    ack, stats = asyncio.run(run())

    assert stats["content_type"] == ARTC_CONTENT_TYPE
    assert stats["num_frames"] == 2
    assert captured["url"] == "http://codeformer.local/v1/realtime/sr/raw"
    assert captured["timeout"] == 0.123
    assert captured["body"] == bytes([1, 2, 3, 4, 5, 6])
    assert captured["headers"]["x-scale"] == "2"
    assert captured["headers"]["x-output-width"] == "2"
    assert captured["headers"]["x-output-height"] == "2"
    assert ack["artc"]["width"] == 2
    assert ack["artc"]["height"] == 2
    assert ack["realtime_postprocess"]["input_width"] == 1
    assert ack["realtime_postprocess"]["output_width"] == 2

    [pusher] = _FakeArtcPusher.instances
    assert pusher.kwargs["width"] == 2
    assert pusher.kwargs["height"] == 2
    [(frames_np, push_kwargs)] = pusher.calls
    assert frames_np.shape == (2, 2, 2, 3)
    assert frames_np.tolist()[0] == [
        [[10, 20, 30], [10, 20, 30]],
        [[10, 20, 30], [10, 20, 30]],
    ]
    assert frames_np.tolist()[1] == [
        [[40, 50, 60], [40, 50, 60]],
        [[40, 50, 60], [40, 50, 60]],
    ]
    assert push_kwargs["audio_chunk_meta"]["frame_processor_status"] == "ok"
    assert push_kwargs["audio_chunk_meta"]["frame_processor_passthrough"] is False
    assert (
        push_kwargs["audio_chunk_meta"]["frame_processor_remote_timing"]
        == "frames=2 total=11ms"
    )


def test_remote_codeformer_processor_busy_passthrough_resizes(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise urllib.error.HTTPError(
            url="http://codeformer.local/v1/realtime/sr/raw",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_frame_processor.urllib.request.urlopen",
        fake_urlopen,
    )
    processor = RemoteCodeFormerFrameProcessor(
        config=RealtimePostprocessConfig(
            type="codeformer",
            endpoint="http://codeformer.local/v1/realtime/sr/raw",
            scale=2,
            timeout_ms=50,
            on_busy="passthrough",
        ),
        input_width=1,
        input_height=1,
        fps=16,
    )
    frames = np.array([[[[1, 2, 3]]]], dtype=np.uint8)

    result = processor.process(frames, session_id="s", chunk_idx=0)

    assert result.frames.shape == (1, 2, 2, 3)
    assert result.frames.tolist() == [[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]]
    assert result.stats["frame_processor_status"] == "busy"
    assert result.stats["frame_processor_passthrough"] is True
    assert result.stats["frame_processor_output_width"] == 2
    assert result.stats["frame_processor_output_height"] == 2


def test_remote_codeformer_artc_sink_sends_versioned_video_audio_chunk(monkeypatch):
    captured = []

    class _FakeResponse:
        def __init__(self, payload, status=200):
            self.payload = json.dumps(payload).encode("utf-8")
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(request, timeout=None):
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": request.data,
                "timeout": timeout,
            }
        )
        if request.get_method() == "POST" and request.full_url.endswith("/sessions"):
            return _FakeResponse({"state": "ready"}, status=201)
        if request.get_method() == "POST" and request.full_url.endswith("/chunks"):
            return _FakeResponse(
                {
                    "status": "enqueued",
                    "timing": {"end_to_end_ms": 12.5},
                    "queue": {},
                }
            )
        if request.get_method() == "DELETE":
            return _FakeResponse({"status": "closed"})
        return _FakeResponse({"state": "ready"})

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_codeformer_artc.urllib.request.urlopen",
        fake_urlopen,
    )

    async def run():
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="p",
            size="1x1",
            fps=16,
            output_transport="artc",
            artc={"token": "secret-token", "channel": "channel-1", "queue_size": 2},
            realtime_postprocess={
                "type": "codeformer",
                "delivery": "artc",
                "endpoint": "http://codeformer.local/v1/realtime/sessions",
                "scale": 2,
                "timeout_ms": 123,
            },
        )
        session = GenerateSession()
        session.set_request(request)
        sink = await create_realtime_output_sink_async(SimpleNamespace(), session)
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-remote-artc",
            width=1,
            height=1,
            fps=16,
            enable_upscaling=False,
            realtime_event_id=5,
            extra={
                "wan_s2v_audio_window": np.array([0, 16384], dtype=np.int16),
                "wan_s2v_audio_window_meta": {"sample_count": 2},
            },
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3]), bytes([4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )
        stats = await sink.send(session, result, batch)
        ack = sink.build_init_ack()
        await sink.close()
        return sink, stats, ack

    sink, stats, ack = asyncio.run(run())

    assert isinstance(sink, RemoteCodeFormerArtcOutputSink)
    assert stats["content_type"] == ARTC_CONTENT_TYPE
    assert ack["artc"]["publisher"] == "codeformer"
    assert ack["artc"]["width"] == 2
    assert ack["artc"]["height"] == 2
    assert [item["method"] for item in captured] == ["POST", "POST", "DELETE"]
    create_payload = json.loads(captured[0]["body"])
    assert create_payload["artc"]["token"] == "secret-token"
    assert create_payload["video"]["output_width"] == 2
    envelope = captured[1]["body"]
    magic, header_size = struct.unpack("!4sI", envelope[:8])
    assert magic == b"CFA1"
    header = json.loads(envelope[8 : 8 + header_size])
    assert header["chunk_idx"] == 0
    assert header["num_frames"] == 2
    assert header["frame_bytes"] == 6
    assert header["audio_samples"] == 2
    assert header["audio_meta"] == {"sample_count": 2}
    assert envelope[8 + header_size : 8 + header_size + 6] == bytes([1, 2, 3, 4, 5, 6])
    assert captured[-1]["url"].endswith("?mode=drain")


def test_remote_codeformer_artc_sink_propagates_publisher_failure(monkeypatch):
    class _FakeResponse:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(request, timeout=None):
        if request.get_method() == "GET":
            return _FakeResponse(
                {"state": "failed", "last_error": "publisher disconnected"}
            )
        if request.get_method() == "DELETE":
            return _FakeResponse({"status": "closed"})
        return _FakeResponse({"state": "ready"})

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_codeformer_artc.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setenv("SGLANG_REALTIME_CODEFORMER_STATUS_POLL_S", "0.1")

    async def run():
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="p",
            size="1x1",
            fps=16,
            output_transport="artc",
            artc={"token": "token", "channel": "channel"},
            realtime_postprocess={
                "type": "codeformer",
                "delivery": "artc",
                "endpoint": "http://codeformer.local/v1/realtime/sessions",
            },
        )
        session = GenerateSession()
        session.set_request(request)
        sink = await create_realtime_output_sink_async(SimpleNamespace(), session)
        try:
            await asyncio.wait_for(sink.wait_failed(), timeout=1.0)
        except RealtimeProtocolError as exc:
            assert exc.code == "codeformer_artc_output_failed"
        else:
            raise AssertionError("remote publisher failure was not propagated")
        await sink.cancel()

    asyncio.run(run())
