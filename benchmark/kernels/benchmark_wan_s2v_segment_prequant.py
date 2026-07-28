#!/usr/bin/env python3
"""Benchmark byte-identical Wan S2V segment-modulate FP8 prequant fusion."""

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
    parser.add_argument("--seq-len", type=int, default=2717)
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument("--seg-idx", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    from sglang.jit_kernel.diffusion.triton.wan_s2v_segment import (
        gelu_tanh_quant_fp8,
        segment_gate_add,
        segment_modulate,
        segment_modulate_quant_fp8,
    )
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    shape = (1, args.seq_len, args.hidden_dim)
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    shift = torch.randn(
        (1, 2, args.hidden_dim),
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn_like(shift)

    def reference():
        modulated = segment_modulate(
            x,
            shift,
            scale,
            args.seg_idx,
            out_dtype=torch.bfloat16,
        )
        q, q_scale = sglang_per_token_group_quant_fp8(
            modulated.view(-1, args.hidden_dim),
            128,
            column_major_scales=True,
        )
        return q.view_as(modulated), q_scale.transpose(-1, -2)

    def fused():
        return segment_modulate_quant_fp8(
            x,
            shift,
            scale,
            args.seg_idx,
        )

    expected_q, expected_scale = reference()
    actual_q, actual_scale = fused()
    if not torch.equal(actual_q, expected_q):
        differing_values = torch.count_nonzero(actual_q != expected_q).item()
        raise AssertionError(
            f"fused segment prequant differs in {differing_values} FP8 bytes"
        )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)

    segment_reference_samples = _time(
        reference,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
    )
    segment_fused_samples = _time(
        fused,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
    )
    segment_reference_summary = _summary(segment_reference_samples)
    segment_fused_summary = _summary(segment_fused_samples)
    segment_reference_p50 = float(segment_reference_summary["p50_ms"])
    segment_fused_p50 = float(segment_fused_summary["p50_ms"])
    segment_saved_ms = segment_reference_p50 - segment_fused_p50

    gelu_shape = (1, args.seq_len, 13824)
    gelu_input = torch.randn(
        gelu_shape,
        device=device,
        dtype=torch.bfloat16,
    )

    def gelu_reference():
        activated = torch.nn.functional.gelu(gelu_input, approximate="tanh")
        q, q_scale = sglang_per_token_group_quant_fp8(
            activated.view(-1, activated.shape[-1]),
            128,
            column_major_scales=True,
        )
        return q.view_as(activated), q_scale.transpose(-1, -2)

    def gelu_fused():
        return gelu_tanh_quant_fp8(gelu_input)

    expected_q, expected_scale = gelu_reference()
    actual_q, actual_scale = gelu_fused()
    if not torch.equal(actual_q, expected_q):
        differing_values = torch.count_nonzero(actual_q != expected_q).item()
        raise AssertionError(
            f"fused GELU prequant differs in {differing_values} FP8 bytes"
        )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    gelu_reference_samples = _time(
        gelu_reference,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
    )
    gelu_fused_samples = _time(
        gelu_fused,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
    )
    gelu_reference_summary = _summary(gelu_reference_samples)
    gelu_fused_summary = _summary(gelu_fused_samples)
    gelu_reference_p50 = float(gelu_reference_summary["p50_ms"])
    gelu_fused_p50 = float(gelu_fused_summary["p50_ms"])
    gelu_saved_ms = gelu_reference_p50 - gelu_fused_p50

    gelu_bias = torch.randn(
        (gelu_shape[-1],),
        device=device,
        dtype=torch.bfloat16,
    )

    def gelu_bias_reference():
        biased = (gelu_input + gelu_bias).to(torch.bfloat16)
        activated = torch.nn.functional.gelu(biased, approximate="tanh")
        q, q_scale = sglang_per_token_group_quant_fp8(
            activated.view(-1, activated.shape[-1]),
            128,
            column_major_scales=True,
        )
        return q.view_as(activated), q_scale.transpose(-1, -2)

    def gelu_bias_fused():
        return gelu_tanh_quant_fp8(gelu_input, bias=gelu_bias)

    expected_q, expected_scale = gelu_bias_reference()
    actual_q, actual_scale = gelu_bias_fused()
    if not torch.equal(actual_q, expected_q):
        differing_values = torch.count_nonzero(actual_q != expected_q).item()
        raise AssertionError(
            f"fused bias+GELU prequant differs in {differing_values} FP8 bytes"
        )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    gelu_bias_reference_summary = _summary(
        _time(
            gelu_bias_reference,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
    )
    gelu_bias_fused_summary = _summary(
        _time(
            gelu_bias_fused,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
    )
    gelu_bias_reference_p50 = float(gelu_bias_reference_summary["p50_ms"])
    gelu_bias_fused_p50 = float(gelu_bias_fused_summary["p50_ms"])
    gelu_bias_saved_ms = gelu_bias_reference_p50 - gelu_bias_fused_p50

    gate_shape = (1, args.seq_len, args.hidden_dim)
    residual = torch.randn(gate_shape, device=device, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(
        (1, 2, args.hidden_dim),
        device=device,
        dtype=torch.float32,
    )
    gate_bias = torch.randn(
        (args.hidden_dim,),
        device=device,
        dtype=torch.bfloat16,
    )

    def gate_bias_reference():
        biased_update = (update + gate_bias).to(torch.bfloat16)
        return segment_gate_add(
            residual,
            biased_update,
            gate,
            args.seg_idx,
        )

    def gate_bias_fused():
        return segment_gate_add(
            residual,
            update,
            gate,
            args.seg_idx,
            bias=gate_bias,
        )

    expected_gate = gate_bias_reference()
    actual_gate = gate_bias_fused()
    if not torch.equal(actual_gate, expected_gate):
        differing_values = torch.count_nonzero(actual_gate != expected_gate).item()
        raise AssertionError(
            f"fused bias+gate add differs in {differing_values} BF16 values"
        )
    gate_bias_reference_summary = _summary(
        _time(
            gate_bias_reference,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
    )
    gate_bias_fused_summary = _summary(
        _time(
            gate_bias_fused,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
        )
    )
    gate_bias_reference_p50 = float(gate_bias_reference_summary["p50_ms"])
    gate_bias_fused_p50 = float(gate_bias_fused_summary["p50_ms"])
    gate_bias_saved_ms = gate_bias_reference_p50 - gate_bias_fused_p50
    result = {
        "device": torch.cuda.get_device_name(device),
        "group_size": 128,
        "byte_identical": True,
        "segment_modulate": {
            "input_shape": list(shape),
            "seg_idx": args.seg_idx,
            "reference": segment_reference_summary,
            "fused": segment_fused_summary,
            "speedup": segment_reference_p50 / segment_fused_p50,
            "saved_ms_per_call": segment_saved_ms,
        },
        "gelu_tanh": {
            "input_shape": list(gelu_shape),
            "reference": gelu_reference_summary,
            "fused": gelu_fused_summary,
            "speedup": gelu_reference_p50 / gelu_fused_p50,
            "saved_ms_per_call": gelu_saved_ms,
        },
        "gelu_tanh_with_linear_bias": {
            "input_shape": list(gelu_shape),
            "reference": gelu_bias_reference_summary,
            "fused": gelu_bias_fused_summary,
            "speedup": gelu_bias_reference_p50 / gelu_bias_fused_p50,
            "saved_ms_per_call": gelu_bias_saved_ms,
        },
        "segment_gate_add_with_linear_bias": {
            "input_shape": list(gate_shape),
            "reference": gate_bias_reference_summary,
            "fused": gate_bias_fused_summary,
            "speedup": gate_bias_reference_p50 / gate_bias_fused_p50,
            "saved_ms_per_call": gate_bias_saved_ms,
        },
        "projected_saved_ms_per_block_40_layers": (segment_saved_ms * 2 + gelu_saved_ms)
        * 40,
        "projected_bias_saved_ms_per_block_40_layers": (
            gelu_bias_saved_ms + 2 * gate_bias_saved_ms
        )
        * 40,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
