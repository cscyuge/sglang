#!/usr/bin/env python3
"""Benchmark FA4 SM120 tile shapes for the steady Wan S2V attention plan."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _parse_tiles(value: str) -> list[tuple[int, int]]:
    tiles = []
    for item in value.split(","):
        try:
            tile_m, tile_n = (int(dim) for dim in item.lower().split("x"))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated MxN tiles, got {value!r}"
            ) from exc
        if tile_m <= 0 or tile_n <= 0:
            raise argparse.ArgumentTypeError("tile dimensions must be positive")
        tiles.append((tile_m, tile_n))
    return tiles


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": values,
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_kernel(fn, *, warmup: int, repeats: int, trials: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--tiles",
        type=_parse_tiles,
        default=_parse_tiles("128x64,64x64,64x128,128x128"),
    )
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--noisy-query", type=int, default=1560)
    parser.add_argument("--condition-query", type=int, default=3874)
    parser.add_argument("--cached-noisy-kv", type=int, default=14040)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    total_q = args.noisy_query + args.condition_query
    noisy_kv = args.cached_noisy_kv + args.condition_query
    total_kv = noisy_kv + args.condition_query
    q = torch.randn(
        total_q,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        total_kv,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor(
        [0, args.noisy_query, total_q],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0, noisy_kv, total_kv],
        dtype=torch.int32,
        device=device,
    )

    from flash_attn_4_sm120.interface import _flash_attn_fwd

    def run(tile: tuple[int, int]) -> torch.Tensor:
        output, _ = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(args.noisy_query, args.condition_query),
            max_seqlen_k=max(noisy_kv, args.condition_query),
            causal=False,
            tile_mn=tile,
        )
        return output

    results = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "shape": {
            "q": list(q.shape),
            "k": list(k.shape),
            "cu_seqlens_q": cu_seqlens_q.tolist(),
            "cu_seqlens_k": cu_seqlens_k.tolist(),
            "max_seqlen_q": max(args.noisy_query, args.condition_query),
            "max_seqlen_k": max(noisy_kv, args.condition_query),
        },
        "tiles": [],
    }
    reference = None
    for tile in args.tiles:
        record: dict[str, object] = {"tile_m": tile[0], "tile_n": tile[1]}
        print(f"[tile] {tile[0]}x{tile[1]}", flush=True)
        try:
            output = run(tile)
            torch.cuda.synchronize()
            if reference is None:
                reference = output.detach().clone()
                record["reference"] = True
            else:
                delta = (output.float() - reference.float()).abs()
                record["max_abs_diff_vs_reference"] = float(delta.max().item())
                record["mean_abs_diff_vs_reference"] = float(delta.mean().item())
            record.update(
                _stats(
                    _time_kernel(
                        lambda: run(tile),
                        warmup=args.warmup,
                        repeats=args.repeats,
                        trials=args.trials,
                    )
                )
            )
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[failed] {record['error']}", flush=True)
        results["tiles"].append(record)

    successful = [item for item in results["tiles"] if item["status"] == "ok"]
    baseline = next(
        (
            item
            for item in successful
            if item["tile_m"] == 128 and item["tile_n"] == 64
        ),
        successful[0] if successful else None,
    )
    if baseline is not None:
        for item in successful:
            item["speedup_vs_128x64"] = (
                float(baseline["p50_ms"]) / float(item["p50_ms"])
            )
    successful.sort(key=lambda item: float(item["p50_ms"]))
    results["best_tile"] = (
        [successful[0]["tile_m"], successful[0]["tile_n"]]
        if successful
        else None
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"best_tile": results["best_tile"], "json_out": str(args.json_out)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
