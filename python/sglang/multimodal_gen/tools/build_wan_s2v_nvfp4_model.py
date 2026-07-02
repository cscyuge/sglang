"""Build a Wan2.2-S2V Stream-R1 DMD NVFP4 model directory.

This tool mirrors the existing Wan S2V FP8 whole-model layout: the output
directory is a self-contained model path with DMD weights merged into the DiT
checkpoint and a top-level ``quantization_config``. Unlike the generic
ModelOpt transformer builder, Wan S2V uses the original Wan checkpoint layout
instead of a Diffusers ``transformer/`` subdirectory.
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
DEFAULT_QUANT_BLOCK_START = 3
DEFAULT_QUANT_BLOCK_END = 37

QUANTIZED_LINEAR_SUFFIXES = (
    ".self_attn.q.weight",
    ".self_attn.k.weight",
    ".self_attn.v.weight",
    ".self_attn.o.weight",
    ".cross_attn.q.weight",
    ".cross_attn.k.weight",
    ".cross_attn.v.weight",
    ".cross_attn.o.weight",
    ".ffn.0.weight",
    ".ffn.2.weight",
)

CHECKPOINT_TO_RUNTIME_MODULE_TYPES = {
    "self_attn.q": "to_q",
    "self_attn.k": "to_k",
    "self_attn.v": "to_v",
    "self_attn.o": "to_out",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out",
    "ffn.0": "ffn.fc_in",
    "ffn.2": "ffn.fc_out",
}

NON_SHARD_IGNORE = shutil.ignore_patterns(
    "*.safetensors",
    "*.safetensors.index.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_non_shard_files(base_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_dir, output_dir, ignore=NON_SHARD_IGNORE)


def _load_weight_map(model_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model_dir / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    index = _load_json(index_path)
    return dict(index["weight_map"]), dict(index.get("metadata") or {})


def _load_dmd_state(path: Path, state_key: str, prefix: str) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or state_key not in checkpoint:
        raise ValueError(f"Expected checkpoint dict with key {state_key!r}: {path}")
    state = checkpoint[state_key]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint key {state_key!r} is not a state dict: {path}")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        out_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        normalized[out_key] = value.detach().cpu().contiguous()
    return normalized


def _block_index(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blocks":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _module_type_for_weight(name: str) -> str | None:
    if not name.endswith(".weight"):
        return None
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None
    return ".".join(parts[2:-1])


def _matches_any_pattern(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _parse_block_range(value: str) -> tuple[int, int]:
    if ":" in value:
        start_s, end_s = value.split(":", 1)
    elif "-" in value:
        start_s, end_s = value.split("-", 1)
        end_s = str(int(end_s) + 1)
    else:
        start_s = value
        end_s = str(int(value) + 1)
    start = int(start_s)
    end = int(end_s)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid BF16 block range: {value!r}")
    return start, end


def _block_in_ranges(block_idx: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= block_idx < end for start, end in ranges)


def _ignore_patterns_for_bf16_modules(
    module_patterns: tuple[str, ...],
    block_ranges: tuple[tuple[int, int], ...],
    *,
    num_layers: int = 40,
) -> list[str]:
    ignore: list[str] = []
    seen: set[str] = set()

    def append(pattern: str) -> None:
        if pattern not in seen:
            ignore.append(pattern)
            seen.add(pattern)

    for pattern in module_patterns:
        append(f"blocks.*.{pattern}")
        for source, target in CHECKPOINT_TO_RUNTIME_MODULE_TYPES.items():
            if fnmatch.fnmatchcase(source, pattern):
                append(f"blocks.*.{target}")
    for start, end in block_ranges:
        for idx in range(max(0, start), min(num_layers, end)):
            append(f"blocks.{idx}.*")
    return ignore


def _should_quantize_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    block_start: int,
    block_end: int,
    bf16_module_patterns: tuple[str, ...] = (),
    bf16_block_ranges: tuple[tuple[int, int], ...] = (),
) -> bool:
    block_idx = _block_index(name)
    if block_idx is None or block_idx < block_start or block_idx >= block_end:
        return False
    if _block_in_ranges(block_idx, bf16_block_ranges):
        return False
    module_type = _module_type_for_weight(name)
    if module_type is not None and _matches_any_pattern(
        module_type,
        bf16_module_patterns,
    ):
        return False
    if tensor.ndim != 2 or tensor.shape[-1] % 16 != 0:
        return False
    return any(name.endswith(suffix) for suffix in QUANTIZED_LINEAR_SUFFIXES)


def _module_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a weight tensor name, got {name}")
    return name[: -len(".weight")]


def _nvfp4_quantize_weight(
    weight: torch.Tensor,
    *,
    device: torch.device,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    weight_gpu = weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    qweight, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(
        weight_gpu.contiguous(),
        group_size,
        try_tensorrt=False,
    )
    packed = qweight._quantized_data.detach().cpu().contiguous()
    block_scale = weight_scale.detach().cpu().contiguous()
    scale_2 = weight_scale_2.detach().reshape(1).cpu().to(torch.float32).contiguous()
    input_scale = torch.ones((1,), dtype=torch.float32)
    del weight_gpu, qweight, weight_scale, weight_scale_2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return packed, block_scale, scale_2, input_scale


def _nvfp4_quant_config(
    *,
    block_start: int,
    block_end: int,
    group_size: int,
    bf16_module_patterns: tuple[str, ...] = (),
    bf16_block_ranges: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    ignore = [
        *[f"blocks.{i}.*" for i in range(0, block_start)],
        *[f"blocks.{i}.*" for i in range(block_end, 40)],
        *_ignore_patterns_for_bf16_modules(
            bf16_module_patterns,
            bf16_block_ranges,
        ),
        "patch_embedding*",
        "cond_encoder*",
        "condition_embedder*",
        "proj_out*",
        "head*",
        "text_embedding*",
        "time_embedding*",
        "time_projection*",
        "trainable_cond_mask*",
        "casual_audio_encoder*",
        "audio_injector*",
        "frame_packer*",
    ]
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "group_size": group_size,
                    "num_bits": 4,
                    "type": "float",
                },
                "targets": ["Linear"],
                "weights": {
                    "dynamic": False,
                    "group_size": group_size,
                    "num_bits": 4,
                    "type": "float",
                },
            }
        },
        "ignore": ignore,
        "kv_cache_scheme": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float",
        },
        "producer": {
            "name": "sglang-wan-s2v-nvfp4-builder",
            "version": "1",
        },
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
        "quant_type": "NVFP4",
        "swap_weight_nibbles": False,
    }


def _copy_readme(output_dir: Path, stats: Mapping[str, Any]) -> None:
    readme = output_dir / "README_STREAM_R1_NVFP4.md"
    readme.write_text(
        "# Wan2.2-S2V Stream-R1 DMD NVFP4\n\n"
        "This directory was generated by "
        "`sglang.multimodal_gen.tools.build_wan_s2v_nvfp4_model`.\n\n"
        "DMD checkpoint weights are merged before quantization. Wan S2V DiT "
        "blocks in the configured range are stored as ModelOpt-compatible "
        "NVFP4; audio injection, audio encoder, frame packer, patch/condition "
        "embeddings, output head, and boundary blocks remain BF16.\n\n"
        "```json\n"
        + json.dumps(dict(stats), indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def build_wan_s2v_nvfp4_model(
    *,
    base_model_dir: str,
    dmd_checkpoint: str,
    output_dir: str,
    checkpoint_state_key: str = "generator",
    checkpoint_prefix: str = "model.",
    group_size: int = 16,
    block_start: int = DEFAULT_QUANT_BLOCK_START,
    block_end: int = DEFAULT_QUANT_BLOCK_END,
    bf16_module_patterns: tuple[str, ...] = (),
    bf16_block_ranges: tuple[tuple[int, int], ...] = (),
    device: str = "cuda:0",
    overwrite: bool = False,
) -> dict[str, Any]:
    base_path = Path(base_model_dir).expanduser().resolve()
    dmd_path = Path(dmd_checkpoint).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_path}")
    if not dmd_path.is_file():
        raise FileNotFoundError(f"DMD checkpoint not found: {dmd_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_path}")
    if block_start < 0 or block_end <= block_start:
        raise ValueError("Invalid block quantization range")
    if group_size != 16:
        raise ValueError("Wan S2V NVFP4 currently expects group_size=16")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NVFP4 quantization")

    print(f"Copying non-shard files: {base_path} -> {output_path}", flush=True)
    _copy_non_shard_files(base_path, output_path)

    print(f"Loading DMD checkpoint: {dmd_path}", flush=True)
    dmd_state = _load_dmd_state(dmd_path, checkpoint_state_key, checkpoint_prefix)

    base_weight_map, base_metadata = _load_weight_map(base_path)
    base_files: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in base_weight_map.items():
        base_files[filename].append(tensor_name)

    quant_config = _nvfp4_quant_config(
        block_start=block_start,
        block_end=block_end,
        group_size=group_size,
        bf16_module_patterns=bf16_module_patterns,
        bf16_block_ranges=bf16_block_ranges,
    )
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    config_path = output_path / "config.json"
    config = _load_json(config_path)
    config["quantization_config"] = quant_config
    _write_json(config_path, config)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_tensors = 0
    bf16_tensors = 0
    dmd_overrides = 0

    for shard_idx, (filename, tensor_names) in enumerate(sorted(base_files.items()), 1):
        shard_tensors: dict[str, torch.Tensor] = {}
        shard_path = base_path / filename
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            for name in tensor_names:
                tensor = dmd_state.get(name)
                if tensor is not None:
                    dmd_overrides += 1
                else:
                    tensor = f.get_tensor(name)
                tensor = tensor.contiguous()
                if _should_quantize_tensor(
                    name,
                    tensor,
                    block_start=block_start,
                    block_end=block_end,
                    bf16_module_patterns=bf16_module_patterns,
                    bf16_block_ranges=bf16_block_ranges,
                ):
                    packed, weight_scale, weight_scale_2, input_scale = (
                        _nvfp4_quantize_weight(
                            tensor,
                            device=target_device,
                            group_size=group_size,
                        )
                    )
                    module_name = _module_name_for_weight(name)
                    shard_tensors[name] = packed
                    shard_tensors[f"{module_name}.weight_scale"] = weight_scale
                    shard_tensors[f"{module_name}.weight_scale_2"] = weight_scale_2
                    shard_tensors[f"{module_name}.input_scale"] = input_scale
                    quantized_tensors += 1
                else:
                    shard_tensors[name] = tensor
                    bf16_tensors += 1
                del tensor

        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        output_shard = output_path / filename
        print(
            f"[{shard_idx}/{len(base_files)}] writing {output_shard.name}: "
            f"{len(shard_tensors)} tensors",
            flush=True,
        )
        save_file(shard_tensors, output_shard, metadata=metadata)
        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()
        del shard_tensors
        gc.collect()

    _write_json(
        output_path / INDEX_FILENAME,
        {
            "metadata": {
                **base_metadata,
                "total_size": total_size,
            },
            "weight_map": updated_weight_map,
        },
    )

    stats = {
        "base_model_dir": str(base_path),
        "dmd_checkpoint": str(dmd_path),
        "output_dir": str(output_path),
        "quantized_weight_tensors": quantized_tensors,
        "bf16_tensors": bf16_tensors,
        "dmd_overrides": dmd_overrides,
        "output_tensors": len(updated_weight_map),
        "output_shards": len(base_files),
        "group_size": group_size,
        "quantized_block_start": block_start,
        "quantized_block_end": block_end,
        "bf16_module_patterns": list(bf16_module_patterns),
        "bf16_block_ranges": [list(item) for item in bf16_block_ranges],
        "total_size": total_size,
    }
    _copy_readme(output_path, stats)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Wan S2V Stream-R1 DMD NVFP4 model dir."
    )
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--dmd-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-state-key", default="generator")
    parser.add_argument("--checkpoint-prefix", default="model.")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--quantized-block-start", type=int, default=3)
    parser.add_argument("--quantized-block-end", type=int, default=37)
    parser.add_argument(
        "--bf16-module-pattern",
        action="append",
        default=[],
        help=(
            "Keep matching block module types in BF16. Supports shell-style "
            "patterns such as cross_attn.k, cross_attn.v, or cross_attn.*."
        ),
    )
    parser.add_argument(
        "--bf16-block-range",
        action="append",
        default=[],
        help=(
            "Keep a block range in BF16. Use start:end with an exclusive end "
            "(for example 31:37), start-end with an inclusive end, or a single block."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_wan_s2v_nvfp4_model(
        base_model_dir=args.base_model_dir,
        dmd_checkpoint=args.dmd_checkpoint,
        output_dir=args.output_dir,
        checkpoint_state_key=args.checkpoint_state_key,
        checkpoint_prefix=args.checkpoint_prefix,
        group_size=args.group_size,
        block_start=args.quantized_block_start,
        block_end=args.quantized_block_end,
        bf16_module_patterns=tuple(args.bf16_module_pattern or ()),
        bf16_block_ranges=tuple(
            _parse_block_range(value) for value in (args.bf16_block_range or ())
        ),
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
