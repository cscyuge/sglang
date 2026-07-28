#!/usr/bin/env python3
"""Benchmark the blocking NCCL order used by LiveAvatar TPP.

The measured loop intentionally uses the default NCCL process group and the
same communication order as LiveAvatar:

    first DiT stage: create -> SP compute -> dist.send on every lane
    middle DiT stage(s): dist.recv -> SP compute -> dist.send on every lane
    decode stage: dist.recv -> validate on every lane

There are no per-link process groups, acknowledgements, artificial rank
staggering, or CUDA synchronizations around send/recv. This keeps the
microbenchmark independent from the SGLang runtime while exercising the
ordering contract that the production TPP transport would rely on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


HEADER_BASE = 127
HEADER_DIGITS = 5
PAYLOAD_MODULUS = 97


def _parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return parsed


def _parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(_parse_int_csv(value))
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {shape}")
    return shape


def _parse_collective_pattern(
    value: str,
) -> list[tuple[str, tuple[int, ...]]]:
    if not value.strip():
        return []
    pattern = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            dtype_name, raw_shape = item.split(":", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "collective pattern entries must use dtype:dimxdim syntax, "
                f"got {item!r}"
            ) from exc
        dtype_name = dtype_name.strip().lower()
        if dtype_name not in ("uint8", "bf16", "fp16", "fp32"):
            raise argparse.ArgumentTypeError(
                f"unsupported collective dtype {dtype_name!r}"
            )
        try:
            shape = tuple(int(dim) for dim in raw_shape.lower().split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid collective shape in {item!r}"
            ) from exc
        if not shape or any(dim <= 0 for dim in shape):
            raise argparse.ArgumentTypeError(
                f"collective dimensions must be positive in {item!r}"
            )
        pattern.append((dtype_name, shape))
    if not pattern:
        raise argparse.ArgumentTypeError("collective pattern must not be empty")
    return pattern


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "uint8": torch.uint8,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _encode_block_id(block_idx: int) -> list[int]:
    if block_idx < 0 or block_idx >= HEADER_BASE**HEADER_DIGITS:
        raise ValueError(
            f"block index {block_idx} exceeds the encodable range "
            f"[0, {HEADER_BASE**HEADER_DIGITS})"
        )
    digits = []
    value = block_idx
    for _ in range(HEADER_DIGITS):
        digits.append(value % HEADER_BASE)
        value //= HEADER_BASE
    return digits


def _decode_block_id(digits: list[int]) -> int:
    value = 0
    multiplier = 1
    for digit in digits:
        value += digit * multiplier
        multiplier *= HEADER_BASE
    return value


def _payload_value(block_idx: int) -> int:
    return (block_idx * 17 + 11) % PAYLOAD_MODULUS


def _new_payload(
    block_idx: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.full(
        shape,
        _payload_value(block_idx),
        dtype=dtype,
        device=device,
    )
    flat = tensor.view(-1)
    flat[:HEADER_DIGITS].copy_(
        torch.tensor(_encode_block_id(block_idx), dtype=dtype, device=device)
    )
    return tensor


def _validate_payload(
    tensor: torch.Tensor,
    block_idx: int,
    total_increment: int,
) -> tuple[int, int]:
    expected = torch.full_like(
        tensor,
        _payload_value(block_idx) + total_increment,
    )
    expected.view(-1)[:HEADER_DIGITS].copy_(
        torch.tensor(
            [
                digit + total_increment
                for digit in _encode_block_id(block_idx)
            ],
            dtype=tensor.dtype,
            device=tensor.device,
        )
    )
    mismatch_count = int(torch.count_nonzero(tensor != expected).item())
    observed_digits = (
        tensor.view(-1)[:HEADER_DIGITS].float().cpu() - total_increment
    ).round().to(torch.int64)
    observed_block = _decode_block_id(observed_digits.tolist())
    return mismatch_count, observed_block


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _rank_output_path(output_path: Path, rank: int) -> Path:
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{output_path.stem}.rank{rank}{suffix}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_topology(
    stage_ranks: list[int],
    world_size: int,
    stage_parallel_size: int = 1,
) -> None:
    if len(stage_ranks) < 2:
        raise ValueError(
            "stage_ranks must contain at least one DiT stage and one decoder"
        )
    if stage_parallel_size <= 0:
        raise ValueError("stage_parallel_size must be positive")
    if len(stage_ranks) * stage_parallel_size != world_size:
        raise ValueError(
            f"stage_ranks has {len(stage_ranks)} stage leaders with "
            f"stage_parallel_size={stage_parallel_size}, but world_size={world_size}"
        )
    if any(rank % stage_parallel_size != 0 for rank in stage_ranks):
        raise ValueError(
            "stage_ranks must contain stage-group-aligned leader ranks: "
            f"stage_ranks={stage_ranks}, stage_parallel_size={stage_parallel_size}"
        )
    expected_leaders = list(range(0, world_size, stage_parallel_size))
    if sorted(stage_ranks) != expected_leaders:
        raise ValueError(
            "stage_ranks must be a permutation of all stage leaders: "
            f"stage_ranks={stage_ranks}, expected={expected_leaders}"
        )


def _create_compute_workspace(
    size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if size <= 0:
        return None
    left = torch.full((size, size), 0.001, dtype=dtype, device=device)
    right = torch.full((size, size), 0.002, dtype=dtype, device=device)
    output = torch.empty((size, size), dtype=dtype, device=device)
    return left, right, output


def _create_collective_pattern_workspaces(
    pattern: list[tuple[str, tuple[int, ...]]],
    device: torch.device,
) -> list[tuple[str, tuple[int, ...], torch.Tensor, torch.Tensor]]:
    return [
        (
            dtype_name,
            shape,
            torch.full(
                shape,
                index + 1,
                dtype=_dtype_from_name(dtype_name),
                device=device,
            ),
            torch.empty(
                shape,
                dtype=_dtype_from_name(dtype_name),
                device=device,
            ),
        )
        for index, (dtype_name, shape) in enumerate(pattern)
    ]


def _communication_contract(
    *,
    stage_ranks: list[int],
    stage_parallel_size: int,
    collective_pattern: list[tuple[str, tuple[int, ...]]],
    collective_pattern_repeats: int,
    legacy_collective_elements: int,
    legacy_collective_repeats: int,
) -> dict[str, Any]:
    if collective_pattern:
        sp_sequence = [
            {
                "operation": "all_to_all_single",
                "dtype": dtype_name,
                "shape": list(shape),
            }
            for _ in range(collective_pattern_repeats)
            for dtype_name, shape in collective_pattern
        ]
    elif legacy_collective_elements > 0 and stage_parallel_size > 1:
        sp_sequence = [
            {
                "operation": "all_to_all_single",
                "dtype": "payload_dtype",
                "shape": [legacy_collective_elements],
            }
            for _ in range(legacy_collective_repeats)
        ]
    else:
        sp_sequence = []

    rank_sequences = {}
    for stage_index, stage_leader in enumerate(stage_ranks):
        for lane_index in range(stage_parallel_size):
            rank = stage_leader + lane_index
            operations = [
                {
                    "operation": "create"
                    if stage_index == 0
                    else "recv",
                    "peer": (
                        None
                        if stage_index == 0
                        else stage_ranks[stage_index - 1] + lane_index
                    ),
                }
            ]
            if stage_index + 1 < len(stage_ranks):
                operations.extend(sp_sequence)
                operations.append({"operation": "compute_tail"})
                operations.append(
                    {
                        "operation": "send",
                        "peer": stage_ranks[stage_index + 1] + lane_index,
                    }
                )
            else:
                operations.append({"operation": "validate"})
            rank_sequences[str(rank)] = operations
    contract = {
        "version": 1,
        "ordering": (
            "first stage: create -> SP sequence -> compute tail -> send; "
            "middle stages: recv -> SP sequence -> compute tail -> send; "
            "decode stage: recv -> validate"
        ),
        "stage_ranks": stage_ranks,
        "stage_parallel_size": stage_parallel_size,
        "rank_sequences": rank_sequences,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    return contract


def _run_compute(
    tensor: torch.Tensor,
    stage_index: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    matmul_repeats: int,
    sp_group: dist.ProcessGroup | None,
    collective_workspace: tuple[torch.Tensor, torch.Tensor] | None,
    sp_collective_repeats: int,
    collective_pattern_workspaces: list[
        tuple[str, tuple[int, ...], torch.Tensor, torch.Tensor]
    ],
    collective_pattern_repeats: int,
) -> int:
    if workspace is not None:
        left, right, output = workspace
        for _ in range(matmul_repeats):
            torch.mm(left, right, out=output)
    if sp_group is not None and collective_workspace is not None:
        collective_input, collective_output = collective_workspace
        for _ in range(sp_collective_repeats):
            dist.all_to_all_single(
                collective_output,
                collective_input,
                group=sp_group,
            )
            collective_input, collective_output = (
                collective_output,
                collective_input,
            )
    collective_calls = (
        sp_collective_repeats
        if sp_group is not None and collective_workspace is not None
        else 0
    )
    if sp_group is not None and collective_pattern_workspaces:
        for _ in range(collective_pattern_repeats):
            for _, _, collective_input, collective_output in (
                collective_pattern_workspaces
            ):
                dist.all_to_all_single(
                    collective_output,
                    collective_input,
                    group=sp_group,
                )
                collective_calls += 1
    tensor.add_(stage_index + 1)
    return collective_calls


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an independent blocking NCCL microbenchmark with the exact "
            "recv -> compute -> send ordering used by LiveAvatar TPP."
        )
    )
    parser.add_argument(
        "--stage-ranks",
        type=_parse_int_csv,
        default=_parse_int_csv("0,1,2,3"),
        help=(
            "Logical pipeline order, ending in the decode rank. "
            "Use 0,1,2,3 for the official linear mapping or 1,2,3,0 for "
            "the current SGLang Wan S2V mapping."
        ),
    )
    parser.add_argument(
        "--stage-parallel-size",
        type=int,
        default=1,
        help=(
            "Uniform number of ranks per stage. stage-ranks names each "
            "contiguous group's leader."
        ),
    )
    parser.add_argument("--blocks", type=int, default=1000)
    parser.add_argument("--warmup-blocks", type=int, default=20)
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=_parse_shape("16,1,60,104"),
        help="Latent tensor shape. The default matches 480x832, one latent frame.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=1,
        help="Validate every Nth block on the decoder; 0 disables validation.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=0,
        help="Optional square matmul size used to enqueue compute before each send.",
    )
    parser.add_argument(
        "--matmul-repeats",
        type=int,
        default=1,
        help="Number of optional matmuls per DiT stage and block.",
    )
    parser.add_argument(
        "--sp-collective-elements",
        type=int,
        default=0,
        help=(
            "Elements in the optional stage-local all-to-all workspace. "
            "Use a positive value to stress SP collective/P2P ordering."
        ),
    )
    parser.add_argument(
        "--sp-collective-repeats",
        type=int,
        default=1,
        help="Number of stage-local all-to-all operations per DiT stage and block.",
    )
    parser.add_argument(
        "--sp-collective-pattern",
        type=_parse_collective_pattern,
        default=[],
        help=(
            "Ordered stage-local all-to-all pattern using comma-separated "
            "dtype:dimxdim entries. For the current Wan S2V SP2 FP8 rowpack "
            "path use uint8:124x1x2717x128,uint8:5604x1x20x128."
        ),
    )
    parser.add_argument(
        "--sp-collective-pattern-repeats",
        type=int,
        default=1,
        help="Number of repetitions of the ordered SP collective pattern.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Combined JSON path. Per-rank JSON files are written alongside it.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run payload and parser checks on CPU without initializing NCCL.",
    )
    return parser


def _run_self_test() -> None:
    shape = (16, 1, 8, 8)
    total_increment = 1 + 2 + 3
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for block_idx in (0, 1, 126, 127, 1000, 100_000):
            tensor = _new_payload(block_idx, shape, dtype, torch.device("cpu"))
            tensor.add_(1).add_(2).add_(3)
            mismatch_count, observed_block = _validate_payload(
                tensor, block_idx, total_increment
            )
            if mismatch_count or observed_block != block_idx:
                raise AssertionError(
                    f"dtype={dtype}, block={block_idx}, mismatches={mismatch_count}, "
                    f"observed={observed_block}"
                )
    _validate_topology([1, 2, 3, 0], 4)
    _validate_topology([2, 4, 6, 0], 8, stage_parallel_size=2)
    assert _parse_shape("16,1,60,104") == (16, 1, 60, 104)
    assert _parse_collective_pattern(
        "uint8:124x1x2717x128,uint8:5604x1x20x128"
    ) == [
        ("uint8", (124, 1, 2717, 128)),
        ("uint8", (5604, 1, 20, 128)),
    ]
    contract = _communication_contract(
        stage_ranks=[2, 4, 6, 0],
        stage_parallel_size=2,
        collective_pattern=[("uint8", (16, 8)), ("uint8", (8, 16))],
        collective_pattern_repeats=40,
        legacy_collective_elements=0,
        legacy_collective_repeats=1,
    )
    assert len(contract["rank_sequences"]["2"]) == 83
    assert len(contract["sha256"]) == 64
    print("self-test: ok")


def _run_benchmark(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _validate_topology(
        args.stage_ranks,
        world_size,
        stage_parallel_size=args.stage_parallel_size,
    )

    if args.blocks <= 0:
        raise ValueError("--blocks must be positive")
    if args.warmup_blocks < 0 or args.warmup_blocks >= args.blocks:
        raise ValueError("--warmup-blocks must be in [0, blocks)")
    if args.validate_every < 0:
        raise ValueError("--validate-every must be non-negative")
    if args.matmul_size < 0 or args.matmul_repeats <= 0:
        raise ValueError(
            "matmul size must be non-negative and repeats must be positive"
        )
    if args.sp_collective_elements < 0 or args.sp_collective_repeats <= 0:
        raise ValueError(
            "SP collective elements must be non-negative and repeats must be positive"
        )
    if args.sp_collective_pattern_repeats <= 0:
        raise ValueError("--sp-collective-pattern-repeats must be positive")
    if args.sp_collective_elements > 0 and args.sp_collective_pattern:
        raise ValueError(
            "--sp-collective-elements and --sp-collective-pattern are mutually exclusive"
        )
    if (
        args.sp_collective_elements > 0
        and args.sp_collective_elements % args.stage_parallel_size != 0
    ):
        raise ValueError(
            "--sp-collective-elements must be divisible by --stage-parallel-size"
        )
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.blocks >= HEADER_BASE**HEADER_DIGITS:
        raise ValueError("--blocks exceeds the payload header range")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = _dtype_from_name(args.dtype)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.timeout_seconds),
    )

    stage_groups: dict[int, dist.ProcessGroup] = {}
    for stage_leader in range(0, world_size, args.stage_parallel_size):
        stage_group_ranks = list(
            range(stage_leader, stage_leader + args.stage_parallel_size)
        )
        stage_groups[stage_leader] = dist.new_group(stage_group_ranks)

    stage_leader = (rank // args.stage_parallel_size) * args.stage_parallel_size
    lane_index = rank - stage_leader
    stage_index = args.stage_ranks.index(stage_leader)
    decoder_rank = args.stage_ranks[-1]
    is_decoder = stage_leader == decoder_rank
    is_decoder_leader = rank == decoder_rank
    is_dit_rank = not is_decoder
    previous_rank = (
        args.stage_ranks[stage_index - 1] + lane_index
        if stage_index > 0
        else None
    )
    next_rank = (
        args.stage_ranks[stage_index + 1] + lane_index
        if stage_index + 1 < len(args.stage_ranks)
        else None
    )
    total_increment = sum(range(1, len(args.stage_ranks)))
    numel = 1
    for dim in args.shape:
        numel *= dim
    payload_bytes = numel * torch.empty((), dtype=dtype).element_size()

    workspace = (
        _create_compute_workspace(args.matmul_size, dtype, device)
        if is_dit_rank
        else None
    )
    collective_workspace = (
        (
            torch.full(
                (args.sp_collective_elements,),
                rank + 1,
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                (args.sp_collective_elements,),
                dtype=dtype,
                device=device,
            ),
        )
        if is_dit_rank
        and args.stage_parallel_size > 1
        and args.sp_collective_elements > 0
        else None
    )
    collective_pattern_workspaces = (
        _create_collective_pattern_workspaces(args.sp_collective_pattern, device)
        if is_dit_rank
        and args.stage_parallel_size > 1
        and args.sp_collective_pattern
        else []
    )
    communication_contract = _communication_contract(
        stage_ranks=args.stage_ranks,
        stage_parallel_size=args.stage_parallel_size,
        collective_pattern=args.sp_collective_pattern,
        collective_pattern_repeats=args.sp_collective_pattern_repeats,
        legacy_collective_elements=args.sp_collective_elements,
        legacy_collective_repeats=args.sp_collective_repeats,
    )
    torch.cuda.synchronize(device)

    start_record = {
        "event": "start",
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "stage_index": stage_index,
        "stage_leader": stage_leader,
        "stage_parallel_size": args.stage_parallel_size,
        "lane_index": lane_index,
        "stage_ranks": args.stage_ranks,
        "previous_rank": previous_rank,
        "next_rank": next_rank,
        "decoder_rank": decoder_rank,
        "device": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "dtype": args.dtype,
        "shape": args.shape,
        "payload_bytes": payload_bytes,
        "communication_contract_sha256": communication_contract["sha256"],
    }
    print(json.dumps(start_record, sort_keys=True), flush=True)

    # This is the single initialization barrier used by the official pipeline.
    dist.barrier()

    recv_call_ms: list[float] = []
    send_call_ms: list[float] = []
    decoder_interarrival_ms: list[float] = []
    validated_blocks = 0
    last_decoder_arrival: float | None = None
    measured_start: float | None = None
    measured_end: float | None = None
    local_measure_start: float | None = None
    sp_collective_calls = 0

    for block_idx in range(args.blocks):
        if block_idx == args.warmup_blocks:
            local_measure_start = time.perf_counter()

        if previous_rank is None:
            tensor = _new_payload(block_idx, args.shape, dtype, device)
        else:
            tensor = torch.empty(args.shape, dtype=dtype, device=device)
            recv_start = time.perf_counter()
            dist.recv(tensor, src=previous_rank)
            recv_end = time.perf_counter()
            if block_idx >= args.warmup_blocks:
                recv_call_ms.append((recv_end - recv_start) * 1000.0)

        if is_dit_rank:
            sp_collective_calls += _run_compute(
                tensor,
                stage_index,
                workspace,
                args.matmul_repeats,
                (
                    stage_groups[stage_leader]
                    if args.stage_parallel_size > 1
                    else None
                ),
                collective_workspace,
                args.sp_collective_repeats,
                collective_pattern_workspaces,
                args.sp_collective_pattern_repeats,
            )
            send_start = time.perf_counter()
            dist.send(tensor.contiguous(), dst=next_rank)
            send_end = time.perf_counter()
            if block_idx >= args.warmup_blocks:
                send_call_ms.append((send_end - send_start) * 1000.0)
        else:
            should_validate = (
                args.validate_every > 0
                and (
                    block_idx % args.validate_every == 0
                    or block_idx == args.blocks - 1
                )
            )
            if should_validate:
                mismatch_count, observed_block = _validate_payload(
                    tensor,
                    block_idx,
                    total_increment,
                )
                if mismatch_count or observed_block != block_idx:
                    raise RuntimeError(
                        "payload validation failed: "
                        f"block={block_idx}, observed_block={observed_block}, "
                        f"mismatch_count={mismatch_count}, numel={numel}"
                    )
                validated_blocks += 1

            arrival = time.perf_counter()
            if is_decoder_leader and block_idx == args.warmup_blocks - 1:
                measured_start = arrival
            elif is_decoder_leader and block_idx >= args.warmup_blocks:
                if measured_start is None:
                    measured_start = arrival if block_idx > 0 else local_measure_start
                if last_decoder_arrival is not None:
                    decoder_interarrival_ms.append(
                        (arrival - last_decoder_arrival) * 1000.0
                    )
                measured_end = arrival
            last_decoder_arrival = arrival

    torch.cuda.synchronize(device)
    local_measure_end = time.perf_counter()

    if is_decoder_leader and args.warmup_blocks > 0:
        # Include the first measured completion interval: warmup[-1] -> measured[0].
        expected_intervals = args.blocks - args.warmup_blocks
        if len(decoder_interarrival_ms) != expected_intervals:
            raise RuntimeError(
                "unexpected decoder interval count: "
                f"got={len(decoder_interarrival_ms)}, expected={expected_intervals}"
            )

    local_elapsed_seconds = (
        local_measure_end - local_measure_start
        if local_measure_start is not None
        else None
    )
    decoder_elapsed_seconds = (
        measured_end - measured_start
        if is_decoder_leader
        and measured_start is not None
        and measured_end is not None
        else None
    )
    measured_blocks = args.blocks - args.warmup_blocks
    expected_sp_collective_calls_per_block = (
        len(args.sp_collective_pattern) * args.sp_collective_pattern_repeats
        if args.sp_collective_pattern
        else (
            args.sp_collective_repeats
            if args.sp_collective_elements > 0
            and args.stage_parallel_size > 1
            else 0
        )
    )
    expected_sp_collective_calls = (
        args.blocks * expected_sp_collective_calls_per_block if is_dit_rank else 0
    )
    if sp_collective_calls != expected_sp_collective_calls:
        raise RuntimeError(
            "SP collective sequence count mismatch: "
            f"rank={rank}, actual={sp_collective_calls}, "
            f"expected={expected_sp_collective_calls}"
        )
    decoder_blocks_per_second = (
        measured_blocks / decoder_elapsed_seconds
        if decoder_elapsed_seconds and decoder_elapsed_seconds > 0
        else None
    )
    per_link_gib_per_second = (
        payload_bytes * decoder_blocks_per_second / (1024**3)
        if decoder_blocks_per_second is not None
        else None
    )
    aggregate_wire_gib_per_second = (
        per_link_gib_per_second
        * (len(args.stage_ranks) - 1)
        * args.stage_parallel_size
        if per_link_gib_per_second is not None
        else None
    )

    rank_summary: dict[str, Any] = {
        "event": "rank_summary",
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "stage_index": stage_index,
        "stage_leader": stage_leader,
        "stage_parallel_size": args.stage_parallel_size,
        "lane_index": lane_index,
        "stage_role": "decode" if is_decoder else "dit",
        "stage_ranks": args.stage_ranks,
        "blocks": args.blocks,
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": measured_blocks,
        "shape": args.shape,
        "dtype": args.dtype,
        "payload_bytes": payload_bytes,
        "local_elapsed_seconds": local_elapsed_seconds,
        "recv_call_ms": _stats(recv_call_ms),
        "send_call_ms": _stats(send_call_ms),
        "validated_blocks": validated_blocks if is_decoder else None,
        "decoder_elapsed_seconds": decoder_elapsed_seconds,
        "decoder_blocks_per_second": decoder_blocks_per_second,
        "decoder_interarrival_ms": _stats(decoder_interarrival_ms),
        "per_link_gib_per_second": per_link_gib_per_second,
        "aggregate_wire_gib_per_second": aggregate_wire_gib_per_second,
        "matmul_size": args.matmul_size,
        "matmul_repeats": args.matmul_repeats,
        "sp_collective_elements": args.sp_collective_elements,
        "sp_collective_repeats": args.sp_collective_repeats,
        "sp_collective_pattern": [
            {"dtype": dtype_name, "shape": list(shape)}
            for dtype_name, shape in args.sp_collective_pattern
        ],
        "sp_collective_pattern_repeats": args.sp_collective_pattern_repeats,
        "sp_collective_calls": sp_collective_calls,
        "sp_collective_calls_per_block": expected_sp_collective_calls_per_block,
        "communication_contract_sha256": communication_contract["sha256"],
    }
    print(json.dumps(rank_summary, sort_keys=True), flush=True)

    if args.json_output is not None:
        _write_json(_rank_output_path(args.json_output, rank), rank_summary)

    # This is the single completion barrier used by the official pipeline.
    dist.barrier()

    if is_decoder_leader and args.json_output is not None:
        rank_summaries = []
        for peer_rank in range(world_size):
            rank_path = _rank_output_path(args.json_output, peer_rank)
            rank_summaries.append(json.loads(rank_path.read_text(encoding="utf-8")))
        combined_summary = {
            "event": "summary",
            "status": "ok",
            "official_order": "recv -> compute -> send (blocking, default NCCL group)",
            "stage_ranks": args.stage_ranks,
            "stage_parallel_size": args.stage_parallel_size,
            "decoder_rank": decoder_rank,
            "blocks": args.blocks,
            "warmup_blocks": args.warmup_blocks,
            "shape": args.shape,
            "dtype": args.dtype,
            "payload_bytes": payload_bytes,
            "logical_transfers_per_block": (
                len(args.stage_ranks) - 1
            )
            * args.stage_parallel_size,
            "validated_blocks": validated_blocks,
            "decoder_elapsed_seconds": decoder_elapsed_seconds,
            "decoder_blocks_per_second": decoder_blocks_per_second,
            "decoder_interarrival_ms": _stats(decoder_interarrival_ms),
            "per_link_gib_per_second": per_link_gib_per_second,
            "aggregate_wire_gib_per_second": aggregate_wire_gib_per_second,
            "matmul_size": args.matmul_size,
            "matmul_repeats": args.matmul_repeats,
            "sp_collective_elements": args.sp_collective_elements,
            "sp_collective_repeats": args.sp_collective_repeats,
            "sp_collective_pattern": [
                {"dtype": dtype_name, "shape": list(shape)}
                for dtype_name, shape in args.sp_collective_pattern
            ],
            "sp_collective_pattern_repeats": args.sp_collective_pattern_repeats,
            "sp_collective_calls_per_dit_rank_block": (
                expected_sp_collective_calls_per_block
            ),
            "communication_contract": communication_contract,
            "rank_summaries": rank_summaries,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "NCCL_DEBUG",
                    "NCCL_ALGO",
                    "NCCL_PROTO",
                    "NCCL_P2P_LEVEL",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_MAX_NCHANNELS",
                    "NCCL_BUFFSIZE",
                    "TORCH_NCCL_BLOCKING_WAIT",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
                )
            },
        }
        _write_json(args.json_output, combined_summary)
        print(
            json.dumps(
                {
                    "event": "result",
                    "status": "ok",
                    "json_output": str(args.json_output),
                    "decoder_blocks_per_second": decoder_blocks_per_second,
                    "validated_blocks": validated_blocks,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.destroy_process_group()


def main() -> None:
    args = _build_parser().parse_args()
    if args.self_test:
        _run_self_test()
        return
    _run_benchmark(args)


if __name__ == "__main__":
    main()
