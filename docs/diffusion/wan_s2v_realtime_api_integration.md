# Wan2.2-S2V 实时生成接口接入文档

日期：2026-07-06

适用接口：`/v1/realtime_video/generate`

本文面向调用端，说明如何接入当前实时生成 WebSocket 接口，以及如何做音频时间线、视频输出和音画编排。

## 1. 接口概览

实时生成接口使用 WebSocket 长连接：

```text
ws://<host>:<port>/v1/realtime_video/generate
```

所有 WebSocket 消息均为 msgpack 二进制消息。本文用 JSON 展示字段结构，真实传输时请用 msgpack 编码。

典型流程：

1. 建立 WebSocket 连接。
2. 发送 `init`，创建实时生成会话。
3. 接收 `init_ack`，读取音频窗口参数和服务端限制。
4. 持续发送 `audio.delta`。
5. 持续接收 `event_ack`、视频 payload 和 `chunk_stats`。
6. 音频结束时发送 `audio.end`。
7. 服务端生成完成后结束会话；异常时返回 `error`。

调用端必须持续读取 WebSocket 输出，不能只发送不读取。否则服务端可能因为输出写入超时而释放会话。

## 2. 时间线职责

调用端拥有音频时间线控制权。

调用端负责：

- 按顺序发送音频 delta。
- 保证 `seq` 连续。
- 保证 `pts_ms` 连续。
- 决定是否发送静音。
- 根据 `event_ack.audio_queue.queue_ms` 控制提前发送量。
- 根据 `chunk_stats.audio_window` 对齐音频和视频。

服务端负责：

- 校验音频时间线。
- 音频足够一个模型窗口后生成视频 chunk。
- 返回 ack、视频 payload、chunk stats 和错误信息。
- 在输入缓冲过大或输出读取过慢时保护会话。

服务端不会因为墙钟时间流逝自动补静音。如果业务需要静音推进时间线，调用端应发送真实静音样本；如果业务希望暂停时间线，则不要发送音频。

## 3. 建立连接

Python 示例：

```python
import msgspec.msgpack
import websockets

ws_url = "ws://127.0.0.1:30000/v1/realtime_video/generate"

async with websockets.connect(
    ws_url,
    max_size=None,
    ping_interval=None,
    open_timeout=30,
) as ws:
    await ws.send(msgspec.msgpack.encode(init_message))
```

建议：

- `max_size=None`，避免 raw 输出或较大控制消息被客户端限制拦截。
- 接收协程应一直运行，及时处理所有服务端消息。
- 当前服务端默认单活实时会话；如果已有会话在运行，新连接可能收到 `session_busy`。

## 4. `init`

连接建立后的第一条消息必须是 `init`。

示例：

```json
{
  "type": "init",
  "prompt": "A person is talking. Only the foreground person is moving, the background remains static.",
  "first_frame": "data:image/jpeg;base64,...",
  "fps": 16,
  "size": "480x832",
  "max_chunks": 20,
  "realtime_output_format": "h264",
  "realtime_output_pacing": false,
  "seed": 42,
  "guidance_scale": 1.0
}
```

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `type` | 是 | 固定为 `init`。 |
| `prompt` | 是 | 生成 prompt。 |
| `first_frame` | 是 | 首帧图像。推荐 `data:image/jpeg;base64,...` 或 msgpack bytes。 |
| `fps` | 否 | 默认 16。建议调用端按 `init_ack` 返回值确认实际窗口。 |
| `size` | 否 | 例如 `480x832`。 |
| `max_chunks` | 否 | 最多生成多少个视频 chunk；长会话可按业务策略设置。 |
| `realtime_output_format` | 否 | `h264` / `raw` / `jpeg` / `webp`，默认 `raw`。真实接入建议用 `h264`。 |
| `realtime_output_pacing` | 否 | 是否由服务端按输出 fps 节奏发送。低延迟接入通常设为 `false`。 |
| `seed` | 否 | 随机种子。 |
| `guidance_scale` | 否 | 生成参数，默认按服务端配置。 |

## 5. `init_ack`

服务端成功初始化后返回 `init_ack`。

示例：

```json
{
  "type": "init_ack",
  "session_id": "68abe073d38a428599b1409a5a1a182c",
  "server_ack_ms": 12,
  "request": {
    "fps": 16,
    "max_chunks": 20,
    "size": "480x832",
    "realtime_output_format": "h264",
    "realtime_output_pacing": false
  },
  "audio_timeline": {
    "sample_rate": 16000,
    "first_public_frames": 9,
    "steady_public_frames": 12,
    "first_window_samples": 9000,
    "steady_window_samples": 12000,
    "first_window_ms": 562.5,
    "steady_window_ms": 750.0,
    "pad_final_window": true,
    "max_buffered_audio_ms": 60000.0
  }
}
```

调用端应使用：

- `session_id`：日志关联。
- `audio_timeline.sample_rate`：服务端音频时间线采样率，目前为 16000。
- `first_window_ms`：首个 chunk 需要的音频时长。
- `steady_window_ms`：后续 chunk 需要的音频时长。
- `first_public_frames` / `steady_public_frames`：首个和后续 chunk 的视频帧数。
- `max_buffered_audio_ms`：服务端允许的最大未消费音频时长。

## 6. 音频输入

音频通过 `event` 消息发送。

推荐格式：

- `pcm16`：int16 little-endian。适合在线接入，带宽较小。
- `f32le`：float32 little-endian。适合需要更高精度或需要和源音频严格对齐的场景。

要求：

- 单声道。
- `seq` 从 0 开始连续递增。
- `pts_ms` 从 0 开始，表示该 delta 在源音频时间线中的起点。
- `sample_count` 应与 payload 中的源音频样本数一致。

### `audio.delta`

示例：

```json
{
  "type": "event",
  "event_id": 1,
  "kind": "audio.delta",
  "payload": {
    "seq": 0,
    "pts_ms": 0.0,
    "sample_rate": 16000,
    "channels": 1,
    "format": "pcm16",
    "sample_count": 1600,
    "audio": "<bytes>"
  }
}
```

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `type` | 是 | 固定为 `event`。 |
| `event_id` | 建议 | 调用端事件 id，用于和 ack / stats 关联。 |
| `kind` | 是 | `audio.delta`。 |
| `payload.seq` | 是 | 当前音频 delta 序号。 |
| `payload.pts_ms` | 是 | 当前音频 delta 的起始 PTS，单位 ms。 |
| `payload.sample_rate` | 否 | 源音频采样率，推荐 16000。 |
| `payload.channels` | 否 | 必须为 1。 |
| `payload.format` | 否 | `pcm16` 或 `f32le`，默认 `pcm16`。 |
| `payload.sample_count` | 建议 | 源音频样本数。 |
| `payload.audio` | 是 | msgpack bytes。base64 string 可用但不推荐在线链路使用。 |

### `audio.end`

当调用端确认后续没有音频时发送：

```json
{
  "type": "event",
  "event_id": 101,
  "kind": "audio.end",
  "payload": {
    "final_seq": 99
  }
}
```

说明：

- `final_seq` 是最后一个已发送并被接受的 `audio.delta.seq`。
- 如果剩余音频不足一个完整窗口，服务端会根据 `init_ack.audio_timeline.pad_final_window` 处理最终 chunk。
- 发送 `audio.end` 后不能再发送 `audio.delta`。

## 7. `event_ack`

每个被服务端接受的 `audio.delta` / `audio.end` 会返回 `event_ack`。

示例：

```json
{
  "type": "event_ack",
  "session_id": "68abe073d38a428599b1409a5a1a182c",
  "event_id": 11,
  "kind": "audio.delta",
  "server_recv_ms": 10,
  "server_ack_ms": 11,
  "audio_queue": {
    "next_seq": 1,
    "next_pts_ms": 100.0,
    "queue_ms": 100.0,
    "ready_window": false,
    "required_window_ms": 562.5,
    "generated_windows": 0,
    "max_buffered_audio_ms": 60000.0
  }
}
```

调用端重点使用：

- `audio_queue.next_seq`：服务端下一条期望收到的 seq。
- `audio_queue.next_pts_ms`：服务端下一条期望收到的音频 PTS。
- `audio_queue.queue_ms`：服务端已接收但未被模型消费的音频时长。
- `audio_queue.ready_window`：是否已有足够音频生成下一个 chunk。
- `audio_queue.required_window_ms`：下一个 chunk 需要的音频窗口长度。

发送策略：

- 不要一次性无限推完整段音频。
- 在线低延迟模式建议让 `queue_ms` 维持在 1 到 2 个 steady window 附近。
- 当 `queue_ms` 过高时暂停或降低发送速度。
- 收到 `audio_buffer_overflow` 时，该 delta 未被接受；调用端可等待后重发同一个 `seq`。

## 8. 视频输出

服务端可能返回两种视频 payload 形式：

- `frame_batch_header` + 下一条二进制 payload。
- `frame_batch`，header 和 payload 打包在同一条 msgpack 消息中。

调用端必须同时支持这两种形式。

### H.264 输出

当 `realtime_output_format="h264"` 时，输出是 H.264 Annex-B 字节流。

典型 header：

```json
{
  "type": "frame_batch_header",
  "request_id": "...",
  "chunk_index": 1,
  "content_type": "video/h264",
  "num_frames": 12,
  "total_size": 10322,
  "format": "h264_annexb",
  "encoding": "h264_annexb",
  "codec": "h264",
  "width": 480,
  "height": 832,
  "fps": 16,
  "event_id": 15
}
```

下一条二进制 payload 是该 chunk 的 H.264 Annex-B 数据。调用端应把它作为视频码流交给解码器，并使用 `chunk_stats.audio_window` 决定它在音频时间线上的位置。

解码建议：

- native 客户端可使用系统 H.264 decoder 或 ffmpeg。
- 浏览器可使用 WebCodecs；如果使用 MSE/MP4 pipeline，需要先 remux。
- 不要把 H.264 payload 当作 JPEG/WebP 图片序列处理。

### raw 输出

当 `realtime_output_format="raw"` 时，payload 是 RGB24 连续帧。

解析方式：

```python
frame_bytes = width * height * 3
for i in range(num_frames):
    frame = payload[i * frame_bytes : (i + 1) * frame_bytes]
```

raw 带宽很大，只建议在明确需要未压缩帧时使用。

### JPEG / WebP 输出

当 `realtime_output_format="jpeg"` 或 `"webp"` 时，payload 是多个编码图片拼接，使用 `payload_lengths` 切分。

这两种格式只建议用于预览，不建议作为主实时视频传输。

## 9. `chunk_stats`

每个视频 chunk 输出后，服务端返回 `chunk_stats`。

示例：

```json
{
  "type": "chunk_stats",
  "session_id": "68abe073d38a428599b1409a5a1a182c",
  "request_id": "...",
  "chunk_index": 0,
  "event_id": 7,
  "server_chunk_start_ms": 180,
  "server_chunk_end_ms": 2798,
  "num_frames": 9,
  "content_type": "video/h264",
  "ws_payload_bytes": 8884,
  "audio_window": {
    "chunk_idx": 0,
    "pts_start_ms": 0.0,
    "pts_end_ms": 562.5,
    "duration_ms": 562.5,
    "sample_count": 9000,
    "is_final": false,
    "event_id": 7
  }
}
```

调用端重点使用：

- `chunk_index`：视频 chunk 序号。
- `num_frames`：该 chunk 的帧数。
- `content_type`：payload 类型。
- `audio_window.pts_start_ms` / `pts_end_ms`：该 chunk 对应的源音频时间范围。
- `audio_window.duration_ms`：该 chunk 覆盖的音频时长。
- `audio_window.is_final`：是否为最终 chunk。
- `server_chunk_start_ms` / `server_chunk_end_ms`：服务端相对会话时间，可用于端到端调试。

音画对齐建议：

- 以音频播放时钟为主时钟。
- 视频按 `chunk_stats.audio_window` 放到音频时间线上。
- 不要假设 chunk 到达时间等于视频 PTS。
- 如果实时播放落后，可由调用端按业务策略丢帧或加速追赶。

## 10. Backpressure

### 输入侧

服务端限制最大未消费音频时长。上限通过 `init_ack.audio_timeline.max_buffered_audio_ms` 返回。

如果某个 `audio.delta` 会超过上限，服务端返回：

```json
{
  "type": "error",
  "code": "audio_buffer_overflow",
  "content": "audio.delta would exceed max buffered audio",
  "details": {
    "seq": 4,
    "buffered_ms": 1000.0,
    "incoming_ms": 750.0,
    "projected_ms": 1750.0,
    "max_buffered_audio_ms": 1500.0
  }
}
```

该 delta 不会被接受，服务端期望的 `seq` 不会前进。调用端可以稍后重发同一个 `seq`。

### 输出侧

调用端必须持续读取输出。如果长时间不读，服务端可能返回：

```json
{
  "type": "error",
  "code": "output_write_timeout",
  "content": "WebSocket write timed out"
}
```

收到该错误后应释放本地状态并重新建立会话。

## 11. 错误处理

错误统一格式：

```json
{
  "type": "error",
  "code": "audio_seq_mismatch",
  "content": "audio.delta seq mismatch: expected 1, got 2",
  "details": {
    "expected_seq": 1,
    "got_seq": 2
  }
}
```

常见错误：

| code | 含义 | 处理建议 |
| --- | --- | --- |
| `invalid_generate_request` | `init` 格式或参数错误。 | 修正后重新连接。 |
| `invalid_event` | event 格式或 kind 错误。 | 修正 event。 |
| `invalid_audio_delta` | 音频 payload 缺失或非法。 | 修正音频消息。 |
| `audio_seq_mismatch` | `seq` 不连续。 | 按 `details.expected_seq` 重发。 |
| `audio_pts_mismatch` | `pts_ms` 不连续。 | 修正 PTS。 |
| `audio_sample_count_mismatch` | `sample_count` 与 payload 不一致。 | 修正 sample_count 或 payload。 |
| `unsupported_audio_format` | 音频格式不支持。 | 使用 `pcm16` 或 `f32le`。 |
| `unsupported_audio_channels` | 非单声道。 | 调用端先转 mono。 |
| `audio_delta_after_end` | `audio.end` 后继续发音频。 | 重新建立会话。 |
| `audio_end_final_seq_mismatch` | `final_seq` 不匹配。 | 修正 `final_seq`。 |
| `audio_buffer_overflow` | 输入音频队列超限。 | 降低提前发送量，稍后重发。 |
| `output_write_timeout` | 调用端读取输出太慢。 | 持续读取输出或重连。 |
| `session_busy` | 服务端已有活跃会话。 | 等待后重试。 |

## 12. 推荐客户端结构

推荐调用端使用两个协程：

- sender：读取音频源，根据 `event_ack.audio_queue.queue_ms` 控制发送。
- receiver：持续读取所有服务端消息，处理 ack、视频 payload、stats 和 error。

伪代码：

```python
async def sender(ws, queue_state):
    seq = 0
    pts_ms = 0.0
    target_queue_ms = 1500.0

    while has_audio():
        if queue_state.queue_ms > target_queue_ms:
            await asyncio.sleep(0.02)
            continue

        samples = read_next_audio_delta()
        await ws.send(msgspec.msgpack.encode(build_audio_delta(seq, pts_ms, samples)))
        seq += 1
        pts_ms += len(samples) * 1000.0 / sample_rate

    await ws.send(msgspec.msgpack.encode(build_audio_end(seq - 1)))


async def receiver(ws, queue_state):
    pending_header = None

    async for raw in ws:
        if pending_header is not None:
            handle_video_payload(pending_header, raw)
            pending_header = None
            continue

        msg = msgspec.msgpack.decode(raw)
        msg_type = msg.get("type")

        if msg_type == "init_ack":
            configure_from_init_ack(msg)
        elif msg_type == "event_ack":
            queue_state.update(msg["audio_queue"])
        elif msg_type == "frame_batch_header":
            pending_header = msg
        elif msg_type == "frame_batch":
            handle_packed_video_payload(msg)
        elif msg_type == "chunk_stats":
            update_video_timeline(msg["audio_window"], msg)
        elif msg_type == "error":
            handle_error(msg)
```

## 13. 接入建议

- 在线接入优先使用 `realtime_output_format="h264"`。
- 音频 delta 建议 20ms 到 100ms。
- `queue_ms` 建议维持在约 1 到 2 个 steady window。
- 以音频播放时钟为主时钟。
- 需要静音推进时发送静音样本；需要暂停时间线时停止发送音频。
- 新接入请使用 `/v1/realtime_video/generate`，不要优先接入旧的 `/v1/videos/sessions`。
