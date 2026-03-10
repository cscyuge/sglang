# FlashTalk gRPC 帧流接入文档

## 概述

SGLang 推理端新增 gRPC server-streaming 输出通道，业务服务端可直接拉取 H.264 编码帧和原始 PCM 音频，
绕过 RTMP CDN 中转，降低端到端延迟。业务服务端拿到裸流后可通过 WebRTC 推给客户端。

**整体时序：**

```
业务服务端                            SGLang 推理端 (:30000 HTTP + :30001 gRPC)
    │                                       │
    │─ POST /v1/videos/sessions ──────────>│  stream_mode=grpc
    │<─ SessionResponse {session_id} ──────│
    │                                       │
    │─ gRPC StreamFrames(session_id) ─────>│  长连接，server push
    │<─ stream FramePacket ────────────────│  H.264 NALUs + PCM
    │                                       │
    │─ POST .../chunks (audio WAV/npy) ───>│  可多次调用
    │─ POST .../chunks ───────────────────>│
    │─ ...                                  │
    │                                       │
    │─ DELETE /v1/videos/sessions/{id} ───>│  结束会话
    │<─ gRPC stream EOF ──────────────────│
```

---

## 1. 推理端启动

在现有启动命令中添加 `--grpc-port` 参数即可开启 gRPC 服务：

```bash
sglang serve \
  --model-path <FlashTalk模型路径> \
  --num-gpus 8 \
  --port 30000 \
  --grpc-port 30001
```

- HTTP API 监听 `:30000`（不变）
- gRPC 帧流服务监听 `:30001`（新增）
- 不传 `--grpc-port` 时 gRPC 服务不启动，完全向后兼容

---

## 2. 创建会话

**请求：**

```
POST http://<host>:30000/v1/videos/sessions
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 生成提示词 |
| `input_reference` | file | 是 | 参考人脸图片（JPG/PNG） |
| `reference_url` | string | 否 | 参考图片 URL（与 `input_reference` 二选一） |
| `stream_mode` | string | 否 | 设为 `"grpc"` 启用 gRPC 帧流输出 |
| `seed` | int | 否 | 随机种子，默认 1024 |
| `size` | string | 否 | 分辨率，如 `"448x448"` |
| `num_inference_steps` | int | 否 | 推理步数，默认 4 |
| `guidance_scale` | float | 否 | CFG 引导强度 |
| `enable_teacache` | bool | 否 | 启用 TeaCache 加速，默认 false |
| `rtmp_push_url` | string | 否 | RTMP/SRT 推流地址（与 gRPC 互斥，优先级更高） |

> **注意：** `rtmp_push_url` 和 `stream_mode=grpc` 互斥。如果同时传了 `rtmp_push_url`，
> 将走 RTMP/SRT 推流路径，`stream_mode` 被忽略。

**响应：**

```json
{
  "session_id": "req-abc123...",
  "object": "video.session",
  "status": "created",
  "stream_mode": "grpc",
  "stream_url": "/v1/videos/req-abc123.../stream",
  "events_url": "/v1/videos/req-abc123.../events",
  "rtmp_push_url": null,
  "created_at": 1741584000,
  "chunks_received": 0,
  "chunks_processed": 0,
  "error": null
}
```

拿到 `session_id` 后立即发起 gRPC 连接。

---

## 3. gRPC 拉流

### 3.1 Proto 定义

```protobuf
syntax = "proto3";
package sglang.multimodal_gen;

service FlashTalkStreaming {
  // 服务端流式 RPC：持续推送编码后的帧
  rpc StreamFrames(StreamFramesRequest) returns (stream FramePacket) {}
}

message StreamFramesRequest {
  string session_id = 1;
}

message FramePacket {
  uint32 chunk_index = 1;      // 递增的 chunk 序号
  bytes  h264_data   = 2;      // H.264 NAL units (Annex B start code)
  bytes  pcm_audio   = 3;      // 16kHz float32 mono PCM (little-endian)
  uint64 pts_ms      = 4;      // 展示时间戳 (毫秒)
  bool   is_keyframe = 5;      // 该 chunk 是否包含关键帧
  VideoMeta video_meta = 6;    // 仅首包携带
}

message VideoMeta {
  uint32 width  = 1;
  uint32 height = 2;
  uint32 fps    = 3;
  bytes  codec_extra_data = 4; // H.264 SPS + PPS (用于初始化解码器)
}
```

Proto 文件位于 `python/sglang/multimodal_gen/grpc/flashtalk_streaming.proto`，
可用 `grpc_tools.protoc` 生成任意语言的客户端 stub。

### 3.2 连接流程

1. 创建 gRPC channel 连接 `<host>:30001`
2. 调用 `StreamFrames(session_id=<session_id>)`
3. 循环接收 `FramePacket`：
   - 首包的 `video_meta` 字段非空，包含视频宽高、帧率和 H.264 SPS/PPS
   - 后续包的 `video_meta` 为空
4. 会话结束时 stream 自动关闭（server EOF）

### 3.3 数据格式说明

| 字段 | 格式 | 说明 |
|------|------|------|
| `h264_data` | Annex B byte stream | 每个 chunk 包含该时间段内所有帧的 NAL units，可直接喂给解码器。编码参数：`libx264 ultrafast zerolatency yuv420p`，GOP = fps（每秒一个关键帧） |
| `pcm_audio` | float32 LE, mono, 16kHz | 原始 PCM 采样，每个 float32 值范围 `[-1.0, 1.0]`。字节数 = 采样数 × 4。若无音频则为空 bytes |
| `codec_extra_data` | H.264 SPS + PPS (AVCC or Annex B) | 来自 libx264 extradata，初始化 WebRTC 编码器或解码器时需要 |
| `pts_ms` | 毫秒时间戳 | 从 0 开始递增，可用于音视频同步 |

### 3.4 Python 客户端示例

```python
import grpc
import struct
import numpy as np

# 从 proto 生成的 stub（或直接复制推理端的 pb2 文件）
from flashtalk_streaming_pb2 import StreamFramesRequest
from flashtalk_streaming_pb2_grpc import FlashTalkStreamingStub


def pull_frames(host: str, grpc_port: int, session_id: str):
    channel = grpc.insecure_channel(f"{host}:{grpc_port}")
    stub = FlashTalkStreamingStub(channel)

    request = StreamFramesRequest(session_id=session_id)
    video_meta = None

    for packet in stub.StreamFrames(request):
        # 首包解析视频元数据
        if packet.video_meta.width > 0:
            video_meta = packet.video_meta
            print(f"Video: {video_meta.width}x{video_meta.height} "
                  f"@ {video_meta.fps}fps, "
                  f"SPS/PPS: {len(video_meta.codec_extra_data)} bytes")

        # H.264 裸流 → 送入 WebRTC / 解码器
        h264_nalu = packet.h264_data  # bytes

        # PCM 音频 → 送入 WebRTC / 播放器
        if packet.pcm_audio:
            pcm = np.frombuffer(packet.pcm_audio, dtype=np.float32)
            # pcm: 16kHz mono float32, shape=(N,)

        print(f"chunk={packet.chunk_index} pts={packet.pts_ms}ms "
              f"keyframe={packet.is_keyframe} "
              f"h264={len(h264_nalu)}B audio={len(packet.pcm_audio)}B")

    print("Stream ended")
    channel.close()
```

### 3.5 Go 客户端示例

```go
package main

import (
    "context"
    "fmt"
    "io"
    "log"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"

    pb "your_module/flashtalk_streaming" // protoc 生成
)

func pullFrames(host string, grpcPort int, sessionID string) error {
    addr := fmt.Sprintf("%s:%d", host, grpcPort)
    conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        return err
    }
    defer conn.Close()

    client := pb.NewFlashTalkStreamingClient(conn)
    stream, err := client.StreamFrames(context.Background(),
        &pb.StreamFramesRequest{SessionId: sessionID})
    if err != nil {
        return err
    }

    for {
        pkt, err := stream.Recv()
        if err == io.EOF {
            log.Println("Stream ended")
            break
        }
        if err != nil {
            return err
        }

        if meta := pkt.GetVideoMeta(); meta.GetWidth() > 0 {
            log.Printf("Video: %dx%d @%dfps SPS/PPS=%d bytes",
                meta.Width, meta.Height, meta.Fps,
                len(meta.CodecExtraData))
        }

        // pkt.H264Data → 送入 WebRTC video track
        // pkt.PcmAudio → 送入 WebRTC audio track (需重采样到 48kHz)
        log.Printf("chunk=%d pts=%dms kf=%v h264=%dB pcm=%dB",
            pkt.ChunkIndex, pkt.PtsMs, pkt.IsKeyframe,
            len(pkt.H264Data), len(pkt.PcmAudio))
    }
    return nil
}
```

---

## 4. 推送音频 chunks

与现有接口完全一致，不受 `stream_mode` 影响：

```
POST http://<host>:30000/v1/videos/sessions/{session_id}/chunks
Content-Type: multipart/form-data
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio` | file | WAV / MP3 文件（自动重采样到 16kHz），或 `.npy` 文件（float32, 16kHz） |

**响应：**

```json
{
  "success": true,
  "session_id": "req-abc123...",
  "chunk_idx": 0,
  "samples": 32000,
  "duration_s": 2.0
}
```

每推一个 audio chunk，推理端会生成对应的视频帧并通过 gRPC 推出。
建议每个 chunk 的音频时长为 2 秒左右。

---

## 5. 结束会话

```
DELETE http://<host>:30000/v1/videos/sessions/{session_id}
```

**响应：**

```json
{
  "success": true,
  "session_id": "req-abc123...",
  "status": "ended"
}
```

调用后：
- 推理端写入 end 哨兵，pipeline 退出 chunk 循环
- gRPC stream 发送完剩余帧后自动 EOF
- 客户端的 `StreamFrames` 迭代正常结束

---

## 6. 完整接入流程（伪代码）

```python
import requests
import grpc
import threading
from flashtalk_streaming_pb2 import StreamFramesRequest
from flashtalk_streaming_pb2_grpc import FlashTalkStreamingStub

INFER_HOST = "10.0.0.1"
HTTP_PORT  = 30000
GRPC_PORT  = 30001
BASE_URL   = f"http://{INFER_HOST}:{HTTP_PORT}/v1/videos"

# ① 创建会话
with open("face.jpg", "rb") as f:
    resp = requests.post(f"{BASE_URL}/sessions", data={
        "prompt": "一个正在说话的女性",
        "stream_mode": "grpc",
        "num_inference_steps": 4,
    }, files={"input_reference": f})
session_id = resp.json()["session_id"]

# ② 后台拉流（gRPC）
def grpc_receiver():
    channel = grpc.insecure_channel(f"{INFER_HOST}:{GRPC_PORT}")
    stub = FlashTalkStreamingStub(channel)
    for pkt in stub.StreamFrames(StreamFramesRequest(session_id=session_id)):
        # 送入 WebRTC pipeline
        webrtc_push_video(pkt.h264_data, pkt.pts_ms, pkt.is_keyframe)
        if pkt.pcm_audio:
            webrtc_push_audio(pkt.pcm_audio, sample_rate=16000)
    channel.close()

t = threading.Thread(target=grpc_receiver, daemon=True)
t.start()

# ③ 逐段推送音频
for audio_chunk_path in audio_chunks:
    with open(audio_chunk_path, "rb") as f:
        requests.post(f"{BASE_URL}/sessions/{session_id}/chunks",
                      files={"audio": f})

# ④ 结束会话
requests.delete(f"{BASE_URL}/sessions/{session_id}")
t.join(timeout=10)
```

---

## 7. 注意事项

### 时序

- gRPC `StreamFrames` 应在创建会话后、推送第一个 audio chunk **之前**发起
- 推理端在收到第一个 audio chunk 后才开始生成视频帧
- gRPC 服务端会等待最多 **60 秒**让 pipeline 初始化（等待 `video_meta.bin`）
- 帧间最大等待超时为 **120 秒**，超时后 stream 自动关闭

### 音视频同步

- `pts_ms` 从 0 开始单调递增
- 每个 `FramePacket` 的 `pcm_audio` 与 `h264_data` 在时间上是对齐的（同一个 chunk 的音视频）
- 音频采样率固定 16kHz，若需 48kHz（WebRTC 常用）请在业务端重采样

### 并发限制

- 当前 pipeline 串行处理会话：同一时间只有一个活跃会话
- 创建新会话时会自动结束旧会话
- 一个 `session_id` 只能有一个 gRPC `StreamFrames` 消费者

### 与 RTMP/SRT 模式的关系

| | RTMP/SRT | gRPC |
|---|---|---|
| 触发条件 | 传 `rtmp_push_url` | 传 `stream_mode=grpc`（且不传 `rtmp_push_url`） |
| 推流方向 | 推理端 → CDN → 客户端拉流 | 业务端 gRPC 主动拉 → WebRTC → 客户端 |
| 数据格式 | FLV/MPEG-TS 容器（H.264 + AAC 48kHz） | 裸 H.264 NALUs + PCM float32 16kHz |
| 额外延迟 | CDN 中转 1-3 秒 | 无中转，帧级延迟 |
| 部署要求 | 需要 CDN 或流媒体服务器 | 业务端需实现 gRPC client + WebRTC |

---

## 8. 错误处理

| 场景 | gRPC 表现 |
|------|-----------|
| `session_id` 不存在 | stream 等待 60 秒后返回 `DEADLINE_EXCEEDED` |
| 推理端 OOM / 崩溃 | stream 断开，客户端收到 `UNAVAILABLE` |
| 客户端主动取消 | 调用 `context.cancel()`，服务端检测到后清理退出 |
| 长时间无新帧（>120s） | stream 自动关闭（server EOF） |
| 会话被 DELETE | 剩余帧发完后 stream 正常 EOF |
