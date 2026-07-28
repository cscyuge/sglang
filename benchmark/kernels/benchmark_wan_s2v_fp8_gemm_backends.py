#!/usr/bin/env python3
"""Benchmark Wan S2V block-FP8 linear backends at the SP2 production shapes."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    cutlass_w8a8_block_fp8_linear_with_fallback,
    flashinfer_gemm_w8a8_block_fp8_linear_with_fallback,
    triton_w8a8_block_fp8_linear,
)


BLOCK_SIZE = [128, 128]
MODEL_DIM = 5120
FFN_DIM = 13824


@dataclass(frozen=True)
class GemmSpec:
    name: str
    m: int
    n: int
    k: int
    count: int


def _specs(m: int) -> list[GemmSpec]:
    return [
        GemmSpec("self_qkv", m, 3 * MODEL_DIM, MODEL_DIM, 1),
        GemmSpec("model_square", m, MODEL_DIM, MODEL_DIM, 3),
        GemmSpec("ffn_in", m, FFN_DIM, MODEL_DIM, 1),
        GemmSpec("ffn_out", m, MODEL_DIM, FFN_DIM, 1),
    ]


def _capture_samples(
    fn, *, repeats: int, graph_replays: int, trials: int
) -> list[float]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(graph_replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (repeats * graph_replays))
    return samples


def _stats(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2717)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--graph-replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    backends = {
        "flashinfer_cutlass": flashinfer_gemm_w8a8_block_fp8_linear_with_fallback,
        "sgl_cutlass": cutlass_w8a8_block_fp8_linear_with_fallback,
        "triton": triton_w8a8_block_fp8_linear,
    }
    result = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "block_size": BLOCK_SIZE,
        "m": args.m,
        "ops": [],
    }

    for spec in _specs(args.m):
        print(f"[shape] {spec.name} M={spec.m} N={spec.n} K={spec.k}", flush=True)
        x = torch.randn((spec.m, spec.k), device=device, dtype=torch.bfloat16)
        weight = torch.randn(
            (spec.n, spec.k), device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        weight_scale = torch.ones(
            (spec.n // BLOCK_SIZE[0], spec.k // BLOCK_SIZE[1]),
            device=device,
            dtype=torch.float32,
        )
        op = {**asdict(spec), "backends": {}}
        reference = None
        for name, fn in backends.items():
            try:
                eager_output = fn(
                    x, weight, BLOCK_SIZE, weight_scale, input_scale=None, bias=None
                )
                torch.cuda.synchronize()
                if reference is None:
                    reference = eager_output
                    error = {"max_abs": 0.0, "mean_abs": 0.0}
                else:
                    diff = (eager_output.float() - reference.float()).abs()
                    error = {
                        "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean()),
                    }

                samples = _capture_samples(
                    lambda: fn(
                        x,
                        weight,
                        BLOCK_SIZE,
                        weight_scale,
                        input_scale=None,
                        bias=None,
                    ),
                    repeats=args.repeats,
                    graph_replays=args.graph_replays,
                    trials=args.trials,
                )
                op["backends"][name] = {
                    **_stats(samples),
                    "error_vs_flashinfer_cutlass": error,
                }
                print(
                    f"  {name}: {statistics.median(samples):.6f} ms",
                    flush=True,
                )
            except Exception as exc:
                op["backends"][name] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
                print(f"  {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
        result["ops"].append(op)
        del x, weight, weight_scale, reference
        gc.collect()
        torch.cuda.empty_cache()

    totals = {}
    for backend in backends:
        values = []
        for op in result["ops"]:
            measurement = op["backends"][backend]
            if "median_ms" not in measurement:
                values = []
                break
            values.append(measurement["median_ms"] * op["count"])
        totals[backend] = sum(values) if values else None
    baseline = totals["flashinfer_cutlass"]
    result["weighted_layer_ms"] = {
        backend: {
            "ms": value,
            "speedup_vs_flashinfer_cutlass": (
                baseline / value if baseline is not None and value is not None else None
            ),
        }
        for backend, value in totals.items()
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["weighted_layer_ms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
