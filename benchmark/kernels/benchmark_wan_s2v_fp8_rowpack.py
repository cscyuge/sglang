#!/usr/bin/env python3
"""Benchmark fused Wan S2V FP8 rowpack preparation on its steady shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _time(fn, *, warmup: int, repeats: int, trials: int) -> list[float]:
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


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    from sglang.jit_kernel.diffusion.triton.usp_fp8_comm import (
        aligned_rowpack_shape,
        blockwise_quant_fp8,
        blockwise_quant_fp8_rowpack,
        blockwise_quant_qkv_fp8_rowpack,
        pack_fp8_payload_scale_aligned,
    )
    from sglang.jit_kernel.diffusion.triton.usp_permute import (
        fused_pack_qkv_for_all_to_all,
    )

    group_size = 128
    world_size = 2
    qkv_shape = (1, 2717, 40, 128)
    q = torch.randn(qkv_shape, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qkv_payload_shape = (120, 1, 2717, 128)
    qkv_scale_shape = (120, 1, 2717, 1)
    qkv_rowpack_shape = aligned_rowpack_shape(
        qkv_payload_shape,
        scale_cols=1,
        world_size=world_size,
    )
    qkv_packed = torch.empty(
        qkv_payload_shape, device=device, dtype=torch.bfloat16
    )
    qkv_payload = torch.empty(
        qkv_payload_shape, device=device, dtype=torch.float8_e4m3fn
    )
    qkv_scale = torch.empty(
        qkv_scale_shape, device=device, dtype=torch.float32
    )
    qkv_reference_out = torch.empty(
        qkv_rowpack_shape, device=device, dtype=torch.uint8
    )
    qkv_fused_out = torch.empty_like(qkv_reference_out)

    def qkv_reference() -> None:
        fused_pack_qkv_for_all_to_all(q, k, v, out=qkv_packed)
        blockwise_quant_fp8(
            qkv_packed,
            group_size=group_size,
            out_q=qkv_payload,
            out_scale=qkv_scale,
        )
        pack_fp8_payload_scale_aligned(
            qkv_payload,
            qkv_scale,
            world_size=world_size,
            out=qkv_reference_out,
        )

    def qkv_fused() -> None:
        blockwise_quant_qkv_fp8_rowpack(
            q,
            k,
            v,
            group_size=group_size,
            world_size=world_size,
            out=qkv_fused_out,
        )

    output_shape = (5434, 1, 20, 128)
    output = torch.randn(output_shape, device=device, dtype=torch.bfloat16)
    output_scale_shape = (5434, 1, 20, 1)
    output_rowpack_shape = aligned_rowpack_shape(
        output_shape,
        scale_cols=1,
        world_size=world_size,
    )
    output_payload = torch.empty(
        output_shape, device=device, dtype=torch.float8_e4m3fn
    )
    output_scale = torch.empty(
        output_scale_shape, device=device, dtype=torch.float32
    )
    output_reference_out = torch.empty(
        output_rowpack_shape, device=device, dtype=torch.uint8
    )
    output_fused_out = torch.empty_like(output_reference_out)

    def output_reference() -> None:
        blockwise_quant_fp8(
            output,
            group_size=group_size,
            out_q=output_payload,
            out_scale=output_scale,
        )
        pack_fp8_payload_scale_aligned(
            output_payload,
            output_scale,
            world_size=world_size,
            out=output_reference_out,
        )

    def output_fused() -> None:
        blockwise_quant_fp8_rowpack(
            output,
            group_size=group_size,
            world_size=world_size,
            out=output_fused_out,
        )

    results = {
        "device": torch.cuda.get_device_name(device),
        "qkv": {
            "input_shape": list(qkv_shape),
            "rowpack_shape": list(qkv_rowpack_shape),
            "reference": _summary(
                _time(
                    qkv_reference,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    trials=args.trials,
                )
            ),
            "fused": _summary(
                _time(
                    qkv_fused,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    trials=args.trials,
                )
            ),
        },
        "output": {
            "input_shape": list(output_shape),
            "rowpack_shape": list(output_rowpack_shape),
            "reference": _summary(
                _time(
                    output_reference,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    trials=args.trials,
                )
            ),
            "fused": _summary(
                _time(
                    output_fused,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    trials=args.trials,
                )
            ),
        },
    }
    for record in (results["qkv"], results["output"]):
        record["speedup"] = (
            float(record["reference"]["p50_ms"])
            / float(record["fused"]["p50_ms"])
        )
        record["saved_ms"] = (
            float(record["reference"]["p50_ms"])
            - float(record["fused"]["p50_ms"])
        )
    results["saved_ms_per_layer"] = (
        float(results["qkv"]["saved_ms"]) + float(results["output"]["saved_ms"])
    )
    results["projected_saved_ms_per_timestep_40_layers"] = (
        float(results["saved_ms_per_layer"]) * 40
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
