#!/usr/bin/env python3
"""Sweep NCCL settings with the Wan S2V TPP/SP2 communication contract.

Every candidate launches ``run_wan_s2v_tpp_nccl_microbench.sh`` with the same
rank mapping, stage P2P payload, and ordered QKV/output FP8 rowpack all-to-all
pattern. Only NCCL environment variables differ between candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


TUNING_ENV_KEYS = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_P2P_LEVEL",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_BUFFSIZE",
)

DEFAULT_PROFILES: dict[str, dict[str, str]] = {
    "baseline": {},
    "p2p_sys": {"NCCL_P2P_LEVEL": "SYS"},
    "p2p_sys_ch1": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
    },
    "p2p_sys_ch2": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "2",
    },
    "p2p_sys_ch4": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
    },
    "p2p_sys_ch8": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "8",
        "NCCL_MAX_NCHANNELS": "8",
    },
    "p2p_sys_ch4_simple": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_PROTO": "Simple",
    },
    "p2p_sys_ch4_ll": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_PROTO": "LL",
    },
    "p2p_sys_ch4_ll128": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_PROTO": "LL128",
    },
    "p2p_sys_ch4_buff1m": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_BUFFSIZE": "1048576",
    },
    "p2p_sys_ch4_buff8m": {
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_BUFFSIZE": "8388608",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_profiles(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return DEFAULT_PROFILES
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("profile JSON must be a non-empty object")
    profiles = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError("profile JSON must map names to environment objects")
        unknown = sorted(set(values) - set(TUNING_ENV_KEYS))
        if unknown:
            raise ValueError(f"profile {name!r} has unsupported keys: {unknown}")
        profiles[name] = {str(key): str(value) for key, value in values.items()}
    return profiles


def _build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        type=Path,
        default=script_dir / "run_wan_s2v_tpp_nccl_microbench.sh",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path)
    parser.add_argument(
        "--profiles",
        help="Optional comma-separated profile names; defaults to all profiles.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--warmup-blocks", type=int, default=8)
    parser.add_argument("--master-port-base", type=int, default=29620)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    args = _build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.blocks <= 0 or not 0 <= args.warmup_blocks < args.blocks:
        raise ValueError("--warmup-blocks must be in [0, blocks)")
    profiles = _load_profiles(args.profiles_json)
    if args.profiles:
        requested = [item.strip() for item in args.profiles.split(",") if item.strip()]
        missing = [name for name in requested if name not in profiles]
        if missing:
            raise ValueError(f"unknown profiles: {missing}")
        profiles = {name: profiles[name] for name in requested}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    run_index = 0
    for profile_name, tuning_env in profiles.items():
        for repeat_index in range(args.repeats):
            run_dir = args.output_dir / profile_name / f"repeat_{repeat_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            json_output = run_dir / "result.json"
            log_path = run_dir / "runner.log"
            env = dict(os.environ)
            for key in TUNING_ENV_KEYS:
                env.pop(key, None)
            env.update(tuning_env)
            env.update(
                {
                    "NUM_PROCESSES": "8",
                    "STAGE_RANKS": "2,4,6,0",
                    "STAGE_PARALLEL_SIZE": "2",
                    "BLOCKS": str(args.blocks),
                    "WARMUP_BLOCKS": str(args.warmup_blocks),
                    "SHAPE": "16,1,60,104",
                    "DTYPE": "fp32",
                    "VALIDATE_EVERY": "1",
                    "MATMUL_SIZE": "0",
                    "SP_COLLECTIVE_ELEMENTS": "0",
                    "SP_COLLECTIVE_PATTERN": (
                        "uint8:124x1x2717x128,uint8:5604x1x20x128"
                    ),
                    "SP_COLLECTIVE_PATTERN_REPEATS": "40",
                    "TIMEOUT_SECONDS": str(args.timeout_seconds),
                    "MASTER_PORT": str(args.master_port_base + run_index),
                    "JSON_OUTPUT": str(json_output),
                    "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
                }
            )
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log_fp:
                completed = subprocess.run(
                    [str(args.runner)],
                    env=env,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_seconds + 60,
                    check=False,
                )
            record: dict[str, Any] = {
                "profile": profile_name,
                "repeat": repeat_index,
                "environment": tuning_env,
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
                "result_path": str(json_output),
                "log_path": str(log_path),
                "status": "failed",
            }
            if completed.returncode == 0 and json_output.exists():
                result = json.loads(json_output.read_text(encoding="utf-8"))
                interarrival = result["decoder_interarrival_ms"]
                record.update(
                    {
                        "status": result.get("status"),
                        "decoder_blocks_per_second": result.get(
                            "decoder_blocks_per_second"
                        ),
                        "interarrival_mean_ms": interarrival.get("mean"),
                        "interarrival_p50_ms": interarrival.get("p50"),
                        "interarrival_p95_ms": interarrival.get("p95"),
                        "communication_contract_sha256": result[
                            "communication_contract"
                        ]["sha256"],
                    }
                )
            run_records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            run_index += 1

    successful_hashes = {
        item["communication_contract_sha256"]
        for item in run_records
        if item["status"] == "ok"
    }
    if len(successful_hashes) > 1:
        raise RuntimeError(
            f"communication contract changed across candidates: {successful_hashes}"
        )

    profile_summaries = []
    for profile_name, tuning_env in profiles.items():
        records = [
            item
            for item in run_records
            if item["profile"] == profile_name and item["status"] == "ok"
        ]
        profile_summaries.append(
            {
                "profile": profile_name,
                "environment": tuning_env,
                "successful_repeats": len(records),
                "failed_repeats": args.repeats - len(records),
                "interarrival_mean_ms": _mean(
                    [float(item["interarrival_mean_ms"]) for item in records]
                ),
                "interarrival_p50_ms": _mean(
                    [float(item["interarrival_p50_ms"]) for item in records]
                ),
                "interarrival_p95_ms": _mean(
                    [float(item["interarrival_p95_ms"]) for item in records]
                ),
                "decoder_blocks_per_second": _mean(
                    [float(item["decoder_blocks_per_second"]) for item in records]
                ),
            }
        )
    profile_summaries.sort(
        key=lambda item: (
            item["interarrival_mean_ms"] is None,
            item["interarrival_mean_ms"] or float("inf"),
        )
    )
    baseline = next(
        (item for item in profile_summaries if item["profile"] == "baseline"),
        None,
    )
    baseline_ms = baseline["interarrival_mean_ms"] if baseline else None
    for item in profile_summaries:
        candidate_ms = item["interarrival_mean_ms"]
        item["improvement_vs_baseline_pct"] = (
            (baseline_ms - candidate_ms) / baseline_ms * 100.0
            if baseline_ms and candidate_ms
            else None
        )

    summary = {
        "status": "ok"
        if any(item["successful_repeats"] for item in profile_summaries)
        else "failed",
        "blocks": args.blocks,
        "warmup_blocks": args.warmup_blocks,
        "repeats": args.repeats,
        "communication_contract_sha256": (
            next(iter(successful_hashes)) if successful_hashes else None
        ),
        "profile_summaries": profile_summaries,
        "runs": run_records,
    }
    summary_path = args.output_dir / "sweep_summary.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_path": str(summary_path),
                "best_profile": (
                    profile_summaries[0]["profile"] if profile_summaries else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
