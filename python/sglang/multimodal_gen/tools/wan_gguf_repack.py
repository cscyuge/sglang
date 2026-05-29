"""Repack Wan GGUF transformer checkpoints into a Diffusers model tree.

This tool is for ComfyUI Wan GGUF checkpoints that store transformer weights in
original Wan naming, for example ``blocks.0.self_attn.q.weight``. It dequantizes
GGUF tensors into regular safetensors shards and writes a Diffusers-compatible
model directory that SGLang diffusion can load.
"""

import argparse
import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any

import gguf
import numpy as np
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.tools.wan_comfy_repack import (
    DEFAULT_MAX_SHARD_SIZE_GB,
    _copy_or_link_base_model_skeleton,
    _prepare_output_dir,
    _read_json,
    _require_dir,
    _require_file,
    _resolve_transformer_sources,
    _tensor_nbytes,
    preserve_workflow_files,
)
from sglang.multimodal_gen.tools.wan_repack import (
    CASCADE_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    convert_transformer_key,
    get_transformer_dirs,
)

logger = init_logger(__name__)

DEFAULT_DEQUANT_DTYPE = "float16"
DIRECT_GGUF_TYPES = {
    "F32",
    "F16",
    "BF16",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
}
TORCH_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
SAFETENSORS_METADATA = {
    "format": "pt",
    "source_format": "gguf_wan",
}


@dataclass(frozen=True)
class GGUFConversionStats:
    source_path: pathlib.Path
    output_dir: pathlib.Path
    tensor_count: int
    quantized_tensor_count: int
    total_size: int
    shard_count: int


def _gguf_type_name(tensor_type: Any) -> str:
    return getattr(tensor_type, "name", str(tensor_type))


def _is_quantized_tensor(tensor: gguf.ReaderTensor) -> bool:
    return _gguf_type_name(tensor.tensor_type) not in DIRECT_GGUF_TYPES


def _get_field_contents(reader: gguf.GGUFReader, key: str) -> Any | None:
    field = reader.fields.get(key)
    if field is None:
        return None
    return field.contents()


def _validate_wan_architecture(
    reader: gguf.GGUFReader, source_path: pathlib.Path
) -> None:
    architecture = _get_field_contents(reader, "general.architecture")
    if architecture != "wan":
        raise ValueError(
            f"{source_path} has general.architecture={architecture!r}; expected 'wan'."
        )


def _validate_patch_embedding_shape(
    *,
    reader: gguf.GGUFReader,
    source_path: pathlib.Path,
    transformer_config_path: pathlib.Path,
) -> None:
    config = _read_json(transformer_config_path)
    expected_in_channels = config.get("in_channels")
    if expected_in_channels is None:
        return

    patch_tensor = next(
        (tensor for tensor in reader.tensors if tensor.name == "patch_embedding.weight"),
        None,
    )
    if patch_tensor is None:
        raise KeyError(f"{source_path} does not contain patch_embedding.weight")

    shape = tuple(int(dim) for dim in patch_tensor.data.shape)
    if len(shape) < 2:
        raise ValueError(
            f"patch_embedding.weight has unexpected shape {shape} in {source_path}"
        )
    if shape[1] != expected_in_channels:
        raise ValueError(
            f"{source_path} patch_embedding input channels ({shape[1]}) do not match "
            f"{transformer_config_path} in_channels ({expected_in_channels})"
        )


def _convert_gguf_tensor(
    tensor: gguf.ReaderTensor,
    *,
    dequant_dtype: torch.dtype,
    cast_all_floating_tensors: bool,
) -> torch.Tensor:
    if _is_quantized_tensor(tensor):
        array = gguf.dequantize(tensor.data, tensor.tensor_type)
        out = torch.from_numpy(np.asarray(array)).to(dequant_dtype)
        return out.contiguous()

    # GGUFReader returns correctly ordered tensor.data; tensor.shape is the raw
    # GGUF metadata shape and may be reversed for non-2D tensors.
    out = torch.from_numpy(np.array(tensor.data, copy=True))
    if cast_all_floating_tensors and out.is_floating_point():
        out = out.to(dequant_dtype)
    return out.contiguous()


def _save_shard(
    shard: dict[str, torch.Tensor],
    tmp_shards: list[tuple[pathlib.Path, list[str]]],
    output_dir: pathlib.Path,
    *,
    dequant_dtype_name: str,
) -> None:
    if not shard:
        return
    tmp_path = (
        output_dir / f".diffusion_pytorch_model-{len(tmp_shards) + 1:05d}.safetensors"
    )
    metadata = dict(SAFETENSORS_METADATA)
    metadata["dequant_dtype"] = dequant_dtype_name
    save_file(shard, tmp_path, metadata=metadata)
    tmp_shards.append((tmp_path, list(shard.keys())))
    logger.info("Wrote shard %s with %d tensors", tmp_path.name, len(shard))


def convert_gguf_transformer(
    *,
    model_type: str,
    source_path: pathlib.Path,
    output_dir: pathlib.Path,
    transformer_config_path: pathlib.Path,
    max_shard_size_gb: float = DEFAULT_MAX_SHARD_SIZE_GB,
    dequant_dtype_name: str = DEFAULT_DEQUANT_DTYPE,
    cast_all_floating_tensors: bool = False,
    validate_shapes: bool = True,
    validate_architecture: bool = True,
) -> GGUFConversionStats:
    """Convert one Wan GGUF transformer file into Diffusers safetensors shards."""
    _require_file(source_path, "Wan GGUF transformer weights")
    _require_file(transformer_config_path, "Base transformer config")
    output_dir.mkdir(parents=True, exist_ok=True)

    if dequant_dtype_name not in TORCH_DTYPE_BY_NAME:
        raise ValueError(
            f"Unsupported dequant dtype {dequant_dtype_name!r}; "
            f"choose one of {sorted(TORCH_DTYPE_BY_NAME)}"
        )
    dequant_dtype = TORCH_DTYPE_BY_NAME[dequant_dtype_name]

    max_shard_bytes = int(max_shard_size_gb * 1024**3)
    if max_shard_bytes <= 0:
        raise ValueError("--max-shard-size-gb must be positive")

    reader = gguf.GGUFReader(str(source_path))
    if validate_architecture:
        _validate_wan_architecture(reader, source_path)
    if validate_shapes:
        _validate_patch_embedding_shape(
            reader=reader,
            source_path=source_path,
            transformer_config_path=transformer_config_path,
        )

    seen_converted_keys: set[str] = set()
    weight_map: dict[str, str] = {}
    tmp_shards: list[tuple[pathlib.Path, list[str]]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    total_size = 0
    quantized_tensor_count = 0

    for tensor in sorted(reader.tensors, key=lambda item: item.name):
        converted_key = convert_transformer_key(tensor.name)
        if converted_key in seen_converted_keys:
            raise ValueError(
                f"Duplicate converted key {converted_key!r} while converting {source_path}"
            )
        seen_converted_keys.add(converted_key)

        is_quantized = _is_quantized_tensor(tensor)
        if is_quantized:
            quantized_tensor_count += 1

        converted_tensor = _convert_gguf_tensor(
            tensor,
            dequant_dtype=dequant_dtype,
            cast_all_floating_tensors=cast_all_floating_tensors,
        )
        tensor_size = _tensor_nbytes(converted_tensor)
        if current_shard and current_size + tensor_size > max_shard_bytes:
            _save_shard(
                current_shard,
                tmp_shards,
                output_dir,
                dequant_dtype_name=dequant_dtype_name,
            )
            current_shard = {}
            current_size = 0

        current_shard[converted_key] = converted_tensor
        current_size += tensor_size
        total_size += tensor_size

    _save_shard(
        current_shard,
        tmp_shards,
        output_dir,
        dequant_dtype_name=dequant_dtype_name,
    )

    total_shards = len(tmp_shards)
    if total_shards == 0:
        raise ValueError(f"No tensors found in {source_path}")

    for index, (tmp_path, shard_keys) in enumerate(tmp_shards, start=1):
        file_name = (
            f"diffusion_pytorch_model-{index:05d}-of-{total_shards:05d}.safetensors"
        )
        final_path = output_dir / file_name
        tmp_path.replace(final_path)
        for shard_key in shard_keys:
            weight_map[shard_key] = file_name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    shutil.copy2(transformer_config_path, output_dir / "config.json")
    conversion_config = {
        "source_path": str(source_path),
        "source_format": "gguf_wan",
        "model_type": model_type,
        "dequant_dtype": dequant_dtype_name,
        "cast_all_floating_tensors": cast_all_floating_tensors,
        "tensor_count": len(weight_map),
        "quantized_tensor_count": quantized_tensor_count,
        "total_size": total_size,
        "shard_count": total_shards,
    }
    with open(output_dir / "sglang_gguf_conversion.json", "w") as f:
        json.dump(conversion_config, f, indent=2, sort_keys=True)

    logger.info(
        "Converted %s %s to %s (%d tensors, %d quantized, %d shard(s))",
        model_type,
        source_path,
        output_dir,
        len(weight_map),
        quantized_tensor_count,
        total_shards,
    )
    return GGUFConversionStats(
        source_path=source_path,
        output_dir=output_dir,
        tensor_count=len(weight_map),
        quantized_tensor_count=quantized_tensor_count,
        total_size=total_size,
        shard_count=total_shards,
    )


def repack_gguf_wan(
    *,
    model_type: str,
    original_model_path: pathlib.Path,
    output_path: pathlib.Path,
    high_path: pathlib.Path | None = None,
    low_path: pathlib.Path | None = None,
    transformer_path: pathlib.Path | None = None,
    workflow_path: pathlib.Path | None = None,
    copy_components: bool = False,
    overwrite: bool = False,
    max_shard_size_gb: float = DEFAULT_MAX_SHARD_SIZE_GB,
    dequant_dtype_name: str = DEFAULT_DEQUANT_DTYPE,
    cast_all_floating_tensors: bool = False,
    validate_shapes: bool = True,
    validate_architecture: bool = True,
) -> list[GGUFConversionStats]:
    _require_dir(original_model_path, "Base Diffusers model")
    transformer_sources = _resolve_transformer_sources(
        model_type,
        high_path=high_path,
        low_path=low_path,
        transformer_path=transformer_path,
    )
    transformer_dirs = get_transformer_dirs(model_type)

    for source in transformer_sources:
        _require_file(source.weights_path, f"{source.output_dir_name} GGUF weights")

    _prepare_output_dir(output_path, overwrite)
    _copy_or_link_base_model_skeleton(
        original_model_path,
        output_path,
        transformer_dirs,
        copy_components=copy_components,
        replace_text_encoder=False,
    )

    if workflow_path is not None:
        preserve_workflow_files(workflow_path, output_path)

    stats = []
    for source in transformer_sources:
        config_path = original_model_path / source.output_dir_name / "config.json"
        stats.append(
            convert_gguf_transformer(
                model_type=model_type,
                source_path=source.weights_path,
                output_dir=output_path / source.output_dir_name,
                transformer_config_path=config_path,
                max_shard_size_gb=max_shard_size_gb,
                dequant_dtype_name=dequant_dtype_name,
                cast_all_floating_tensors=cast_all_floating_tensors,
                validate_shapes=validate_shapes,
                validate_architecture=validate_architecture,
            )
        )

    logger.info("Done. Repacked model saved to: %s", output_path)
    return stats


def get_args():
    parser = argparse.ArgumentParser(
        description="Repack Wan GGUF transformer checkpoints into a Diffusers model tree"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to convert",
    )
    parser.add_argument(
        "--original-model-path",
        type=str,
        required=True,
        help="Path to the base HF Diffusers model",
    )
    parser.add_argument(
        "--high-path",
        type=str,
        default=None,
        help="High-noise transformer GGUF path for Wan2.2 cascade models",
    )
    parser.add_argument(
        "--low-path",
        type=str,
        default=None,
        help="Low-noise transformer GGUF path for Wan2.2 cascade models",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Single transformer GGUF path for non-cascade models",
    )
    parser.add_argument(
        "--workflow-path",
        type=str,
        default=None,
        help="Optional ComfyUI workflow JSON file or directory to copy into output/workflow",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the repacked Diffusers model",
    )
    parser.add_argument(
        "--copy-components",
        action="store_true",
        help="Copy base model components instead of symlinking them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output path if it already exists",
    )
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=DEFAULT_MAX_SHARD_SIZE_GB,
        help="Maximum output transformer shard size in GiB",
    )
    parser.add_argument(
        "--dequant-dtype",
        type=str,
        default=DEFAULT_DEQUANT_DTYPE,
        choices=sorted(TORCH_DTYPE_BY_NAME),
        help="Torch dtype used for dequantized GGUF tensors",
    )
    parser.add_argument(
        "--cast-all-floating-tensors",
        action="store_true",
        help="Cast non-quantized floating tensors to --dequant-dtype too",
    )
    parser.add_argument(
        "--no-validate-shapes",
        action="store_true",
        help="Skip patch_embedding shape checks against base transformer config",
    )
    parser.add_argument(
        "--no-validate-architecture",
        action="store_true",
        help="Skip GGUF general.architecture == 'wan' validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.model_type in CASCADE_MODEL_TYPES and (
        args.high_path is None or args.low_path is None
    ):
        raise ValueError(f"{args.model_type} requires both --high-path and --low-path")

    repack_gguf_wan(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model_path),
        output_path=pathlib.Path(args.output_path),
        high_path=pathlib.Path(args.high_path) if args.high_path else None,
        low_path=pathlib.Path(args.low_path) if args.low_path else None,
        transformer_path=(
            pathlib.Path(args.transformer_path) if args.transformer_path else None
        ),
        workflow_path=pathlib.Path(args.workflow_path) if args.workflow_path else None,
        copy_components=args.copy_components,
        overwrite=args.overwrite,
        max_shard_size_gb=args.max_shard_size_gb,
        dequant_dtype_name=args.dequant_dtype,
        cast_all_floating_tensors=args.cast_all_floating_tensors,
        validate_shapes=not args.no_validate_shapes,
        validate_architecture=not args.no_validate_architecture,
    )
