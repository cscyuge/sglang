# SPDX-License-Identifier: Apache-2.0
"""Merge a Stream-R1 Wan S2V DMD checkpoint and quantize the result to FP8.

This builds a standalone FP8 Wan2.2-S2V model directory for Stream-R1 DMD
inference. The output transformer weights already include the DMD checkpoint,
so runtime configs should not set stream_r1_generator_checkpoint_path again.

Usage:
    python scripts/quantize_stream_r1_wan_s2v_fp8.py \
        --input-path /data/model/Wan2.2-S2V-14B \
        --stream-r1-checkpoint /data/models/exp_dmd_chunk3_2node_resume4000_model5200.pt \
        --output-path /data/model/Wan2.2-S2V-14B-StreamR1-DMD-FP8

Output format matches scripts/quantize_flashtalk_fp8.py:
    weight           : torch.float8_e4m3fn
    weight_scale_inv : torch.bfloat16, shape (M / 128, N / 128)

The script quantizes large linear weights under blocks.* and keeps embeddings,
norms, biases, modulation tensors, audio injector, VAE/text/audio side
directories, and all other tensors in their checkpoint dtype.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

FP8_E4M3_MAX = 448.0
BLOCK_SIZE = 128

STATE_DICT_KEYS = ("generator_ema", "generator", "model", "state_dict")
WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
)
ROOT_MODULE_PREFIXES = ("module.", "model.")

# Wan2.2-S2V block-level linear weights. These are the same block names used by
# FlashTalk for the shared Wan transformer backbone, minus FlashTalk-specific
# audio/img block modules that are harmless to leave in the pattern list.
QUANTIZE_PATTERNS = [
    r"^blocks\.\d+\.self_attn\.[qkvo]\.weight$",
    r"^blocks\.\d+\.cross_attn\.[qkvo]\.weight$",
    r"^blocks\.\d+\.cross_attn\.[kv]_img\.weight$",
    r"^blocks\.\d+\.audio_cross_attn\.q_linear\.weight$",
    r"^blocks\.\d+\.audio_cross_attn\.kv_linear\.weight$",
    r"^blocks\.\d+\.audio_cross_attn\.proj\.weight$",
    r"^blocks\.\d+\.ffn\.[02]\.weight$",
]

QUANTIZE_RE = [re.compile(pattern) for pattern in QUANTIZE_PATTERNS]


def should_quantize(name: str) -> bool:
    return any(pattern.match(name) for pattern in QUANTIZE_RE)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def is_mapping_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def select_stream_r1_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool,
) -> tuple[Mapping[str, Any], str | None]:
    candidate_keys = STATE_DICT_KEYS if use_ema else STATE_DICT_KEYS[1:]
    for key in candidate_keys:
        value = checkpoint.get(key)
        if is_mapping_state_dict(value):
            return value, key
    if is_mapping_state_dict(checkpoint):
        return checkpoint, None
    raise TypeError("Stream-R1 checkpoint must contain a mapping state dict")


def clean_stream_r1_key(key: str) -> str:
    clean_key = key
    for prefix in WRAPPER_PREFIXES:
        clean_key = clean_key.replace(prefix, "")

    stripped = True
    while stripped:
        stripped = False
        for prefix in ROOT_MODULE_PREFIXES:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                stripped = True
                break
    return clean_key


def load_stream_r1_checkpoint(
    checkpoint_path: Path,
    *,
    use_ema: bool,
) -> tuple[dict[str, torch.Tensor], str | None, tuple[str, ...]]:
    print(f"Loading Stream-R1 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not is_mapping_state_dict(checkpoint):
        raise TypeError("Stream-R1 checkpoint must load to a mapping")

    state_dict, source_key = select_stream_r1_state_dict(checkpoint, use_ema=use_ema)
    cleaned: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if isinstance(value, nn.Parameter):
            value = value.detach()
        if not torch.is_tensor(value):
            skipped.append(key)
            continue
        cleaned[clean_stream_r1_key(key)] = value.detach().cpu()

    if not cleaned:
        raise ValueError("Stream-R1 checkpoint did not contain tensor weights")

    print(
        "Loaded Stream-R1 state dict "
        f"(source={source_key or '<root>'}, tensors={len(cleaned)}, "
        f"skipped={len(skipped)})"
    )
    return cleaned, source_key, tuple(skipped)


def quantize_block_fp8(
    weight: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {weight.ndim}D")
    out_features, in_features = weight.shape
    if out_features % block_size != 0 or in_features % block_size != 0:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} is not divisible by {block_size}"
        )

    weight_f32 = weight.float()
    blocks = weight_f32.reshape(
        out_features // block_size,
        block_size,
        in_features // block_size,
        block_size,
    )
    max_abs = blocks.abs().amax(dim=(1, 3))
    scale_inv = max_abs.clamp(min=1e-12) / FP8_E4M3_MAX
    scaled = blocks / scale_inv[:, None, :, None]
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    weight_fp8 = scaled.reshape(out_features, in_features).to(torch.float8_e4m3fn)
    return weight_fp8, scale_inv.to(torch.bfloat16)


def relative_quantization_error(
    original: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int,
) -> float:
    out_features, in_features = original.shape
    dequant = (
        weight_fp8.float().reshape(
            out_features // block_size,
            block_size,
            in_features // block_size,
            block_size,
        )
        * scale_inv.float()[:, None, :, None]
    ).reshape(out_features, in_features)
    denom = original.float().abs().mean() + 1e-12
    return ((dequant - original.float()).abs() / denom).mean().item()


def load_index(input_path: Path) -> tuple[dict[str, Any] | None, dict[str, str]]:
    index_path = input_path / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        return None, {}
    with open(index_path) as f:
        index = json.load(f)
    weight_map = dict(index.get("weight_map", {}))
    return index, weight_map


def validate_overlay(
    *,
    checkpoint_state: Mapping[str, torch.Tensor],
    weight_map: Mapping[str, str],
    strict_overlay: bool,
) -> None:
    if not weight_map:
        return

    base_keys = set(weight_map)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(base_keys - checkpoint_keys)
    extra = sorted(checkpoint_keys - base_keys)

    if strict_overlay and missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            "Stream-R1 checkpoint does not cover all base transformer tensors. "
            f"Missing {len(missing)} keys, first keys: {preview}"
        )

    print(
        "Overlay validation: "
        f"base_keys={len(base_keys)}, checkpoint_keys={len(checkpoint_keys)}, "
        f"missing={len(missing)}, extra={len(extra)}"
    )
    if missing:
        print(
            "  Warning: missing checkpoint keys will keep base weights: "
            + ", ".join(missing[:10])
        )
    if extra:
        print(
            "  Warning: extra checkpoint keys not present in base model: "
            + ", ".join(extra[:10])
        )


def copy_config_and_auxiliary_files(
    *,
    input_path: Path,
    output_path: Path,
    index: dict[str, Any] | None,
    new_weight_map_entries: Mapping[str, str],
    output_total_size: int,
    block_size: int,
    checkpoint_path: Path,
    source_key: str | None,
    quantized_tensors: int,
) -> None:
    handled_top_level = {".json", ".safetensors"}

    print("\nCopying config and auxiliary files...")
    for config_file in sorted(input_path.glob("*.json")):
        if config_file.name == "config.json":
            with open(config_file) as f:
                config = json.load(f)
            config["quantization_config"] = {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [block_size, block_size],
            }
            with open(output_path / config_file.name, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  {config_file.name} (with quantization_config added)")
        elif config_file.name == "diffusion_pytorch_model.safetensors.index.json":
            if index is None:
                continue
            output_index = dict(index)
            weight_map = dict(output_index.get("weight_map", {}))
            weight_map.update(new_weight_map_entries)
            output_index["weight_map"] = dict(sorted(weight_map.items()))
            metadata = dict(output_index.get("metadata", {}))
            metadata["total_size"] = output_total_size
            output_index["metadata"] = metadata
            with open(output_path / config_file.name, "w") as f:
                json.dump(output_index, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(
                f"  {config_file.name} "
                f"(with {len(new_weight_map_entries)} scale_inv entries added)"
            )
        else:
            shutil.copy2(config_file, output_path / config_file.name)
            print(f"  {config_file.name}")

    for item in sorted(input_path.iterdir()):
        dest = output_path / item.name
        if dest.exists() or dest.is_symlink():
            continue
        if item.is_dir():
            if item.name.startswith("."):
                continue
            shutil.copytree(item, dest)
            print(f"  {item.name}/ (directory)")
        elif item.is_file() and item.suffix not in handled_top_level:
            shutil.copy2(item, dest)
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.name} ({size_mb:.1f} MB)")

    readme = output_path / "README_STREAM_R1_FP8.md"
    with open(readme, "w") as f:
        f.write("# Stream-R1 Wan S2V DMD FP8 Weights\n\n")
        f.write(
            "This directory was generated by "
            "`scripts/quantize_stream_r1_wan_s2v_fp8.py`.\n\n"
        )
        f.write(f"- Base model: `{input_path}`\n")
        f.write(f"- Stream-R1 checkpoint: `{checkpoint_path}`\n")
        f.write(f"- Checkpoint source key: `{source_key or '<root>'}`\n")
        f.write(f"- Quantized tensors: `{quantized_tensors}`\n")
        f.write(f"- FP8 block size: `{block_size}x{block_size}`\n\n")
        f.write(
            "The transformer safetensors already include the Stream-R1 DMD "
            "checkpoint. Do not set `stream_r1_generator_checkpoint_path` when "
            "using this model directory, otherwise runtime will try to overlay "
            "the bf16 DMD checkpoint onto FP8 parameters.\n"
        )
    print(f"  {readme.name}")


def quantize_stream_r1_wan_s2v_model(
    *,
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    block_size: int = BLOCK_SIZE,
    use_ema: bool = False,
    strict_overlay: bool = True,
    overwrite: bool = False,
    verify_error: bool = False,
) -> None:
    input_dir = Path(input_path)
    checkpoint_file = Path(checkpoint_path)
    output_dir = Path(output_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input model directory not found: {input_dir}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Stream-R1 checkpoint file not found: {checkpoint_file}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index, weight_map = load_index(input_dir)
    checkpoint_state, source_key, skipped_keys = load_stream_r1_checkpoint(
        checkpoint_file,
        use_ema=use_ema,
    )
    validate_overlay(
        checkpoint_state=checkpoint_state,
        weight_map=weight_map,
        strict_overlay=strict_overlay,
    )

    safetensors_files = sorted(input_dir.glob("diffusion_pytorch_model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No diffusion safetensors files found in {input_dir}")

    print(f"Found {len(safetensors_files)} diffusion safetensors shards")
    t_start = time.time()
    total_quantized = 0
    total_kept = 0
    total_overlayed = 0
    total_params_quantized = 0
    total_params_kept = 0
    output_total_size = 0
    new_weight_map_entries: dict[str, str] = {}
    seen_base_keys: set[str] = set()

    for safetensors_file in safetensors_files:
        print(f"\nProcessing {safetensors_file.name}...")
        tensors = load_file(str(safetensors_file))
        output_tensors: dict[str, torch.Tensor] = {}
        file_quantized = 0
        file_kept = 0
        file_overlayed = 0

        for name, base_tensor in tensors.items():
            seen_base_keys.add(name)
            tensor = checkpoint_state.get(name, base_tensor)
            if name in checkpoint_state:
                file_overlayed += 1

            if tuple(tensor.shape) != tuple(base_tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: base={tuple(base_tensor.shape)} "
                    f"checkpoint={tuple(tensor.shape)}"
                )

            if should_quantize(name):
                weight_fp8, scale_inv = quantize_block_fp8(tensor, block_size)
                output_tensors[name] = weight_fp8
                output_tensors[f"{name}_scale_inv"] = scale_inv
                shard_name = safetensors_file.name
                new_weight_map_entries[f"{name}_scale_inv"] = shard_name

                file_quantized += 1
                total_params_quantized += tensor.numel()
                output_total_size += tensor_nbytes(weight_fp8) + tensor_nbytes(
                    scale_inv
                )

                msg = f"  {name:56s} {str(tuple(tensor.shape)):20s} -> fp8"
                if verify_error:
                    rel_err = relative_quantization_error(
                        tensor,
                        weight_fp8,
                        scale_inv,
                        block_size,
                    )
                    msg += f"  (rel_err={rel_err:.6f})"
                print(msg)
            else:
                output_tensors[name] = tensor
                file_kept += 1
                total_params_kept += tensor.numel()
                output_total_size += tensor_nbytes(tensor)

        total_quantized += file_quantized
        total_kept += file_kept
        total_overlayed += file_overlayed

        out_file = output_dir / safetensors_file.name
        print(
            f"  Saving {out_file.name} "
            f"({file_quantized} quantized, {file_kept} kept, "
            f"{file_overlayed} overlayed)"
        )
        save_file(output_tensors, str(out_file))

        del tensors
        del output_tensors

    if strict_overlay and not weight_map:
        missing_seen = sorted(seen_base_keys - set(checkpoint_state))
        if missing_seen:
            preview = ", ".join(missing_seen[:10])
            raise KeyError(
                "Stream-R1 checkpoint did not cover all processed base tensors. "
                f"Missing {len(missing_seen)} keys, first keys: {preview}"
            )

    copy_config_and_auxiliary_files(
        input_path=input_dir,
        output_path=output_dir,
        index=index,
        new_weight_map_entries=new_weight_map_entries,
        output_total_size=output_total_size,
        block_size=block_size,
        checkpoint_path=checkpoint_file,
        source_key=source_key,
        quantized_tensors=total_quantized,
    )

    elapsed = time.time() - t_start
    input_size = sum(file.stat().st_size for file in safetensors_files)
    output_size = sum(file.stat().st_size for file in output_dir.glob("*.safetensors"))

    print(f"\n{'=' * 70}")
    print(f"Stream-R1 Wan S2V FP8 quantization complete in {elapsed:.1f}s")
    print(f"  Checkpoint source:  {source_key or '<root>'}")
    print(f"  Skipped ckpt keys:  {len(skipped_keys)}")
    print(f"  Overlayed tensors:  {total_overlayed}")
    print(f"  Quantized tensors:  {total_quantized}")
    print(f"  Kept tensors:       {total_kept}")
    print(
        f"  Quantized params:   {total_params_quantized:,} "
        f"({total_params_quantized * 2 / 1e9:.2f} GB bf16 -> "
        f"{total_params_quantized / 1e9:.2f} GB fp8)"
    )
    print(
        f"  Kept params:        {total_params_kept:,} "
        f"({total_params_kept * 2 / 1e9:.2f} GB bf16)"
    )
    if input_size > 0:
        print(
            f"  Transformer disk:   {input_size / 1e9:.2f} GB -> "
            f"{output_size / 1e9:.2f} GB "
            f"({100 * (1 - output_size / input_size):.1f}% reduction)"
        )
    print(f"  Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a Stream-R1 Wan2.2-S2V DMD checkpoint and quantize the "
            "merged transformer safetensors to FP8 block-wise format."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the original Wan2.2-S2V model directory.",
    )
    parser.add_argument(
        "--stream-r1-checkpoint",
        required=True,
        help="Path to the Stream-R1 DMD checkpoint .pt file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the merged FP8 model directory.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help="Block size for block-wise FP8 quantization.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use generator_ema from the Stream-R1 checkpoint when present.",
    )
    parser.add_argument(
        "--no-strict-overlay",
        dest="strict_overlay",
        action="store_false",
        help="Allow checkpoint-missing base tensors to keep original base weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists and is non-empty.",
    )
    parser.add_argument(
        "--verify-error",
        action="store_true",
        help="Compute and print relative quantization error for each quantized tensor.",
    )
    parser.set_defaults(strict_overlay=True)
    args = parser.parse_args()

    quantize_stream_r1_wan_s2v_model(
        input_path=args.input_path,
        checkpoint_path=args.stream_r1_checkpoint,
        output_path=args.output_path,
        block_size=args.block_size,
        use_ema=args.use_ema,
        strict_overlay=args.strict_overlay,
        overwrite=args.overwrite,
        verify_error=args.verify_error,
    )


if __name__ == "__main__":
    main()
