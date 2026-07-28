#!/usr/bin/env python3
"""Tune the Stream-R1 segmented K/V pack at the Wan S2V SP2 shape."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from sglang.jit_kernel.diffusion.triton.stream_r1_segmented_pack import (
    fused_pack_segmented_kv,
)


def _samples(fn, *, warmup: int, repeats: int, trials: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-seq-len", type=int, default=14040)
    parser.add_argument("--condition-seq-len", type=int, default=3874)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-rows", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--block-hd", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--num-warps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    noisy_shape = (1, args.noisy_seq_len, args.heads, args.head_dim)
    condition_shape = (1, args.condition_seq_len, args.heads, args.head_dim)
    noisy_key = torch.randn(noisy_shape, device=device, dtype=dtype)
    noisy_value = torch.randn_like(noisy_key)
    condition_key = torch.randn(condition_shape, device=device, dtype=dtype)
    condition_value = torch.randn_like(condition_key)
    plan = [
        torch.tensor(values, device=device, dtype=torch.int32)
        for values in (
            [0, 0],
            [0, args.noisy_seq_len],
            [0, args.noisy_seq_len],
            [args.noisy_seq_len, args.condition_seq_len],
        )
    ]
    total_tokens = args.noisy_seq_len + args.condition_seq_len
    out = (
        torch.empty(
            (total_tokens, args.heads, args.head_dim), device=device, dtype=dtype
        ),
        torch.empty(
            (total_tokens, args.heads, args.head_dim), device=device, dtype=dtype
        ),
    )
    expected_key = torch.cat((noisy_key[0], condition_key[0]), dim=0)
    expected_value = torch.cat((noisy_value[0], condition_value[0]), dim=0)
    results = []

    for block_rows, block_hd, num_warps in itertools.product(
        args.block_rows, args.block_hd, args.num_warps
    ):
        def run():
            return fused_pack_segmented_kv(
                noisy_key,
                noisy_value,
                condition_key,
                condition_value,
                *plan,
                total_tokens=total_tokens,
                noisy_seq_len=args.noisy_seq_len,
                max_length=args.noisy_seq_len,
                out=out,
                block_rows=block_rows,
                block_hd=block_hd,
                num_warps=num_warps,
            )

        try:
            run()
            torch.cuda.synchronize()
            torch.testing.assert_close(out[0], expected_key, rtol=0, atol=0)
            torch.testing.assert_close(out[1], expected_value, rtol=0, atol=0)
            samples = _samples(
                run,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
            )
            item = {
                "block_rows": block_rows,
                "block_hd": block_hd,
                "num_warps": num_warps,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "exact": True,
            }
            print(
                f"rows={block_rows} hd={block_hd} warps={num_warps}: "
                f"{item['median_ms']:.6f} ms",
                flush=True,
            )
        except Exception as exc:
            item = {
                "block_rows": block_rows,
                "block_hd": block_hd,
                "num_warps": num_warps,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"rows={block_rows} hd={block_hd} warps={num_warps}: "
                f"ERROR {item['error']}",
                flush=True,
            )
        results.append(item)

    valid = [item for item in results if "median_ms" in item]
    valid.sort(key=lambda item: item["median_ms"])
    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "noisy": noisy_shape,
            "condition": condition_shape,
            "total_tokens": total_tokens,
        },
        "best": valid[0] if valid else None,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": payload["best"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
