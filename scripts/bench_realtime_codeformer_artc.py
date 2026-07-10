"""Smoke test SGLang -> CodeFormer -> ARTC without logging credentials."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
import wave
from pathlib import Path

import msgspec.msgpack
import websockets
from websockets.exceptions import ConnectionClosedOK


def _load_env(path: str) -> dict[str, str]:
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value.strip().strip('"').strip("'")
    return result


def _artc_token(
    app_id: str,
    app_key: str,
    channel: str,
    user_id: str,
) -> str:
    timestamp = int(time.time()) + 86400
    digest = hashlib.sha256(
        f"{app_id}{app_key}{channel}{user_id}{timestamp}".encode()
    ).hexdigest()
    payload = {
        "appid": app_id,
        "channelid": channel,
        "userid": user_id,
        "nonce": "",
        "timestamp": timestamp,
        "token": digest,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _read_pcm16(path: str, sample_count: int) -> bytes:
    with wave.open(path, "rb") as wav:
        if (
            wav.getframerate() != 16000
            or wav.getnchannels() != 1
            or wav.getsampwidth() != 2
        ):
            raise ValueError("audio must be 16 kHz mono PCM16 WAV")
        return wav.readframes(sample_count)


def _decode_control(raw) -> dict:
    message = msgspec.msgpack.decode(raw) if isinstance(raw, bytes) else json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("server returned a non-object control message")
    return message


async def _run(args) -> None:
    env = _load_env(args.env)
    channel = args.channel or f"cf_direct_{int(time.time())}"
    audio_samples = args.audio_samples or (9000 + max(0, args.chunks - 1) * 12000)
    audio = _read_pcm16(args.audio, audio_samples)
    image_data = base64.b64encode(Path(args.image).read_bytes()).decode("ascii")
    init = {
        "type": "init",
        "prompt": args.prompt,
        "first_frame": f"data:image/jpeg;base64,{image_data}",
        "fps": args.fps,
        "size": args.size,
        "max_chunks": args.chunks,
        "output_transport": "artc",
        "artc": {
            "token": _artc_token(
                env["ARTC_APP_ID"],
                env["ARTC_APP_KEY"],
                channel,
                args.userid,
            ),
            "channel": channel,
            "userid": args.userid,
            "queue_size": 2,
        },
        "realtime_postprocess": {
            "type": "codeformer",
            "delivery": "artc",
            "scale": args.scale,
            "timeout_ms": args.timeout_ms,
        },
        "seed": 42,
        "guidance_scale": 1.0,
    }
    started = time.perf_counter()
    async with websockets.connect(
        args.url,
        max_size=None,
        ping_interval=None,
        open_timeout=30,
    ) as ws:
        await ws.send(msgspec.msgpack.encode(init))
        while True:
            message = _decode_control(await asyncio.wait_for(ws.recv(), timeout=60))
            if message.get("type") == "error":
                raise RuntimeError(message)
            if message.get("type") == "init_ack":
                init_ack = message
                break
        artc_ack = init_ack.get("artc") or {}
        width_s, height_s = args.size.split("x", 1)
        if artc_ack.get("publisher") != "codeformer":
            raise RuntimeError(f"unexpected ARTC publisher: {artc_ack}")
        if (artc_ack.get("width"), artc_ack.get("height")) != (
            int(width_s) * args.scale,
            int(height_s) * args.scale,
        ):
            raise RuntimeError(f"unexpected ARTC output dimensions: {artc_ack}")
        await ws.send(
            msgspec.msgpack.encode(
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
                        "sample_count": len(audio) // 2,
                        "audio": audio,
                    },
                }
            )
        )
        await ws.send(
            msgspec.msgpack.encode(
                {
                    "type": "event",
                    "event_id": 2,
                    "kind": "audio.end",
                    "payload": {"final_seq": 0},
                }
            )
        )
        result = None
        chunk_stats = []
        try:
            while True:
                message = _decode_control(
                    await asyncio.wait_for(ws.recv(), timeout=args.timeout_s)
                )
                if message.get("type") == "error":
                    raise RuntimeError(message)
                if message.get("type") == "chunk_stats":
                    chunk_stats.append(message)
                    result = {
                        "channel": channel,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                        "chunk_index": message.get("chunk_index"),
                        "content_type": message.get("content_type"),
                        "artc": artc_ack,
                    }
        except ConnectionClosedOK:
            pass
        if result is None:
            raise RuntimeError("connection closed before chunk_stats")
        if len(chunk_stats) != args.chunks:
            raise RuntimeError(
                f"expected {args.chunks} chunk_stats, got {len(chunk_stats)}"
            )
        result["chunks"] = len(chunk_stats)
        result["drained_ms"] = round((time.perf_counter() - started) * 1000, 3)
        print("E2E_RESULT " + json.dumps(result, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:30000/v1/realtime_video/generate",
    )
    parser.add_argument("--env", default="/mnt/mycephfs/ljx/.env")
    parser.add_argument("--image", default="/mnt/mycephfs/ljx/assets/mouth.jpg")
    parser.add_argument("--audio", default="/mnt/mycephfs/ljx/assets/mouth.wav")
    parser.add_argument("--audio-samples", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--size", default="480x832")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--userid", default="codeformer")
    parser.add_argument("--channel")
    parser.add_argument("--timeout-ms", type=float, default=120000)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument(
        "--prompt",
        default=(
            "A person is talking. Only the foreground person is moving, "
            "the background remains static."
        ),
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
