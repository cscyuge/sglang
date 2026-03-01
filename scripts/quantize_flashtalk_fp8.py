# SPDX-License-Identifier: Apache-2.0
"""Offline FP8 block-wise quantization for FlashTalk-14B.

Quantizes MLP and QKVO weights to float8_e4m3fn with 128x128 block-wise
scaling, matching the Qwen3-FP8 / DeepSeek-V3 format for DeepGEMM.

Embedding, norm, bias, modulation, head, and audio_proj layers are kept
in bfloat16.

Usage:
    python scripts/quantize_flashtalk_fp8.py \
        --input-path /path/to/SoulX-FlashTalk-14B \
        --output-path /path/to/SoulX-FlashTalk-14B-FP8

Output format (per quantized weight):
    weight          : torch.float8_e4m3fn, shape (M, N)
    weight_scale_inv: torch.bfloat16,      shape (M/128, N/128)

config.json is copied with added quantization_config:
    {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128]
    }
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# FP8 E4M3 representable max
FP8_E4M3_MAX = 448.0

# Block size for block-wise quantization
BLOCK_SIZE = 128

# Patterns of tensor names to quantize (original FlashTalk checkpoint naming).
# Only large linear weight matrices are quantized.
QUANTIZE_PATTERNS = [
    # Self-attention QKVO
    r"^blocks\.\d+\.self_attn\.[qkvo]\.weight$",
    # Text/image cross-attention QKVO + k_img/v_img
    r"^blocks\.\d+\.cross_attn\.[qkvo]\.weight$",
    r"^blocks\.\d+\.cross_attn\.[kv]_img\.weight$",
    # Audio cross-attention
    r"^blocks\.\d+\.audio_cross_attn\.q_linear\.weight$",
    r"^blocks\.\d+\.audio_cross_attn\.kv_linear\.weight$",
    r"^blocks\.\d+\.audio_cross_attn\.proj\.weight$",
    # FFN (MLP)
    r"^blocks\.\d+\.ffn\.[02]\.weight$",
]

_QUANTIZE_RE = [re.compile(p) for p in QUANTIZE_PATTERNS]


def should_quantize(name: str) -> bool:
    """Check if a tensor name matches any quantization pattern."""
    return any(r.match(name) for r in _QUANTIZE_RE)


def quantize_block_fp8(
    weight: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to FP8 E4M3 with block-wise scaling.

    Args:
        weight: (M, N) bfloat16/float32 weight tensor.
                M and N must be divisible by block_size.
        block_size: block dimension (default 128).

    Returns:
        weight_fp8: (M, N) torch.float8_e4m3fn
        scale_inv:  (M // block_size, N // block_size) torch.bfloat16
                    Dequantization: weight ≈ weight_fp8 * scale_inv
    """
    assert weight.ndim == 2, f"Expected 2D tensor, got {weight.ndim}D"
    M, N = weight.shape
    assert M % block_size == 0, f"M={M} not divisible by {block_size}"
    assert N % block_size == 0, f"N={N} not divisible by {block_size}"

    # Reshape into blocks: (M/B, B, N/B, B)
    blocks = weight.reshape(M // block_size, block_size, N // block_size, block_size)

    # Per-block max absolute value: (M/B, N/B)
    max_abs = blocks.abs().amax(dim=(1, 3))

    # Compute inverse scale (dequantization multiplier)
    # scale_inv = max_abs / FP8_MAX, with floor to avoid underflow
    scale_inv = max_abs.clamp(min=1e-12) / FP8_E4M3_MAX

    # Scale weight into FP8 range: weight_scaled = weight / scale_inv
    # Expand scale_inv for broadcasting: (M/B, 1, N/B, 1)
    scale_inv_expanded = scale_inv[:, None, :, None]
    blocks_scaled = blocks / scale_inv_expanded

    # Clamp and cast to FP8 E4M3
    blocks_scaled = blocks_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    weight_fp8 = blocks_scaled.reshape(M, N).to(torch.float8_e4m3fn)

    scale_inv = scale_inv.to(torch.bfloat16)
    return weight_fp8, scale_inv


def quantize_model(input_path: str, output_path: str) -> None:
    """Quantize FlashTalk model weights to FP8 block-wise format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    st_files = sorted(input_path.glob("*.safetensors"))
    if not st_files:
        # Try subdirectory (some models store weights in a subfolder)
        st_files = sorted(input_path.glob("**/*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {input_path}")

    print(f"Found {len(st_files)} safetensors files in {input_path}")

    total_quantized = 0
    total_kept = 0
    total_params_quantized = 0
    total_params_kept = 0
    t_start = time.time()

    for st_file in st_files:
        print(f"\nProcessing {st_file.name}...")
        tensors = load_file(str(st_file))

        output_tensors = {}
        file_quantized = 0
        file_kept = 0

        for name, tensor in tensors.items():
            if should_quantize(name):
                # Quantize to FP8
                weight_f32 = tensor.float()  # compute in fp32 for precision
                weight_fp8, scale_inv = quantize_block_fp8(weight_f32)

                output_tensors[name] = weight_fp8
                output_tensors[f"{name}_scale_inv"] = scale_inv

                file_quantized += 1
                total_params_quantized += tensor.numel()

                # Verify round-trip error
                with torch.no_grad():
                    M, N = tensor.shape
                    dequant = (
                        weight_fp8.float().reshape(
                            M // BLOCK_SIZE, BLOCK_SIZE, N // BLOCK_SIZE, BLOCK_SIZE
                        )
                        * scale_inv.float()[:, None, :, None]
                    )
                    dequant = dequant.reshape(M, N)
                    rel_err = (
                        (dequant - weight_f32).abs() / (weight_f32.abs().mean() + 1e-12)
                    ).mean()
                    print(
                        f"  {name:60s} {str(tuple(tensor.shape)):20s} "
                        f"-> fp8  (rel_err={rel_err:.6f})"
                    )
            else:
                # Keep in original dtype (bfloat16)
                output_tensors[name] = tensor
                file_kept += 1
                total_params_kept += tensor.numel()

        total_quantized += file_quantized
        total_kept += file_kept

        # Save output safetensors
        out_file = output_path / st_file.name
        print(
            f"  Saving {out_file.name} ({file_quantized} quantized, {file_kept} kept)"
        )
        save_file(output_tensors, str(out_file))

    t_elapsed = time.time() - t_start

    # Copy config files and update safetensors index
    print("\nCopying config and auxiliary files...")
    for config_file in input_path.glob("*.json"):
        if config_file.name == "config.json":
            # Add quantization config
            with open(config_file) as f:
                config = json.load(f)
            config["quantization_config"] = {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
            }
            out_config = output_path / config_file.name
            with open(out_config, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  {config_file.name} (with quantization_config added)")
        elif config_file.name.endswith(".safetensors.index.json"):
            # Update safetensors index with scale_inv entries
            with open(config_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # For each quantized weight, add its scale_inv to the same shard file
            new_entries = {}
            for tensor_name, shard_file in weight_map.items():
                if should_quantize(tensor_name):
                    new_entries[f"{tensor_name}_scale_inv"] = shard_file
            weight_map.update(new_entries)
            index["weight_map"] = dict(sorted(weight_map.items()))
            out_index = output_path / config_file.name
            with open(out_index, "w") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            print(
                f"  {config_file.name} (with {len(new_entries)} scale_inv entries added)"
            )
        else:
            shutil.copy2(config_file, output_path / config_file.name)
            print(f"  {config_file.name}")

    # Copy all other files (non-json, non-safetensors) from input directory
    handled_suffixes = {".json", ".safetensors"}
    for item in sorted(input_path.iterdir()):
        dest = output_path / item.name
        if dest.exists() or dest.is_symlink():
            continue
        if item.is_dir():
            # Skip hidden directories like .cache, .git
            if item.name.startswith("."):
                continue
            shutil.copytree(item, dest)
            print(f"  {item.name}/ (directory)")
        elif item.is_file() and item.suffix not in handled_suffixes:
            shutil.copy2(item, dest)
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.name} ({size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Quantization complete in {t_elapsed:.1f}s")
    print(f"  Quantized tensors: {total_quantized}")
    print(f"  Kept tensors:      {total_kept}")
    print(
        f"  Quantized params:  {total_params_quantized:,} "
        f"({total_params_quantized * 2 / 1e9:.2f} GB bf16 -> "
        f"{total_params_quantized / 1e9:.2f} GB fp8)"
    )
    print(
        f"  Kept params:       {total_params_kept:,} "
        f"({total_params_kept * 2 / 1e9:.2f} GB bf16)"
    )

    input_size = sum(f.stat().st_size for f in input_path.glob("*.safetensors"))
    output_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors"))
    print(
        f"  Disk size:         {input_size / 1e9:.2f} GB -> {output_size / 1e9:.2f} GB "
        f"({100 * (1 - output_size / input_size):.1f}% reduction)"
    )
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize FlashTalk-14B to FP8 block-wise format"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to original FlashTalk model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for block-wise quantization (default: 128)",
    )
    args = parser.parse_args()

    global BLOCK_SIZE
    BLOCK_SIZE = args.block_size

    quantize_model(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
