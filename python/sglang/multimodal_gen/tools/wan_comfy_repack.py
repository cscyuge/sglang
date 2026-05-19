"""Repack ComfyUI Wan single-file checkpoints into a Diffusers model tree.

This is intended for Wan2.2 Remix-style checkpoints that contain only the
transformer weights in ComfyUI/original Wan naming, for example
``model.diffusion_model.blocks.0.self_attn.q.weight``.
"""

import argparse
import json
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import Iterable

from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.tools.wan_repack import (
    CASCADE_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    convert_transformer_key,
    get_transformer_dirs,
)

logger = init_logger(__name__)

COMFYUI_TRANSFORMER_PREFIX = "model.diffusion_model."
DEFAULT_MAX_SHARD_SIZE_GB = 4.0
SAFETENSORS_METADATA = {
    "format": "pt",
    "source_format": "comfyui_wan",
}


@dataclass(frozen=True)
class TransformerSource:
    output_dir_name: str
    weights_path: pathlib.Path


def _tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _read_json(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _require_file(path: pathlib.Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} does not exist or is not a file: {path}"
        )


def _require_dir(path: pathlib.Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} does not exist or is not a directory: {path}"
        )


def _prepare_output_dir(output_path: pathlib.Path, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Pass --overwrite to replace it."
            )
        if output_path.is_symlink() or output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)
    output_path.mkdir(parents=True)


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path, copy: bool) -> None:
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        return

    os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())


def _copy_or_link_base_model_skeleton(
    original_model_path: pathlib.Path,
    output_path: pathlib.Path,
    transformer_dirs: Iterable[str],
    *,
    copy_components: bool,
    replace_text_encoder: bool,
) -> None:
    skip_names = set(transformer_dirs)
    if replace_text_encoder:
        skip_names.add("text_encoder")

    for child in original_model_path.iterdir():
        if child.name in skip_names:
            continue
        _link_or_copy(child, output_path / child.name, copy_components)


def _copy_or_link_text_encoder_skeleton(
    original_model_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    copy_components: bool,
) -> pathlib.Path:
    src_dir = original_model_path / "text_encoder"
    out_dir = output_path / "text_encoder"
    _require_dir(src_dir, "Base text_encoder directory")
    out_dir.mkdir()

    for child in src_dir.iterdir():
        if child.name.endswith(".safetensors") or child.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        _link_or_copy(child, out_dir / child.name, copy_components)

    return out_dir


def _safe_open_keys(path: pathlib.Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def _base_text_encoder_keys(base_text_encoder_dir: pathlib.Path) -> set[str]:
    keys: set[str] = set()
    for path in _list_safetensors_files(str(base_text_encoder_dir)):
        keys.update(_safe_open_keys(pathlib.Path(path)))
    if not keys:
        raise FileNotFoundError(
            f"No text_encoder safetensors found in {base_text_encoder_dir}"
        )
    return keys


def replace_text_encoder_weights(
    original_model_path: pathlib.Path,
    output_path: pathlib.Path,
    weights_path: pathlib.Path,
    *,
    copy_components: bool,
    check_keys: bool,
) -> None:
    _require_file(weights_path, "Text encoder weights")
    out_dir = _copy_or_link_text_encoder_skeleton(
        original_model_path, output_path, copy_components=copy_components
    )

    if check_keys:
        base_keys = _base_text_encoder_keys(original_model_path / "text_encoder")
        replacement_keys = set(_safe_open_keys(weights_path))
        if replacement_keys != base_keys:
            missing = sorted(base_keys - replacement_keys)[:10]
            extra = sorted(replacement_keys - base_keys)[:10]
            raise ValueError(
                "Replacement text encoder keys do not match the base text_encoder. "
                f"missing={missing}, extra={extra}"
            )

    _link_or_copy(weights_path, out_dir / "model.safetensors", copy_components)
    logger.info("Replaced text_encoder weights with %s", weights_path)


def preserve_workflow_files(
    workflow_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Copy ComfyUI workflow JSON files for provenance.

    SGLang does not consume ComfyUI workflow graphs at runtime. They are kept
    next to the converted checkpoint as reproducibility/reference artifacts.
    """
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow path does not exist: {workflow_path}")

    out_dir = output_path / "workflow"
    out_dir.mkdir(exist_ok=True)

    if workflow_path.is_file():
        shutil.copy2(workflow_path, out_dir / workflow_path.name)
        logger.info("Copied workflow file %s", workflow_path)
        return

    if not workflow_path.is_dir():
        raise ValueError(
            f"Workflow path is neither a file nor a directory: {workflow_path}"
        )

    copied = 0
    for child in workflow_path.iterdir():
        if child.is_file():
            shutil.copy2(child, out_dir / child.name)
            copied += 1
    logger.info("Copied %d workflow file(s) from %s", copied, workflow_path)


def _resolve_transformer_sources(
    model_type: str,
    *,
    high_path: pathlib.Path | None,
    low_path: pathlib.Path | None,
    transformer_path: pathlib.Path | None,
) -> list[TransformerSource]:
    transformer_dirs = get_transformer_dirs(model_type)
    if model_type in CASCADE_MODEL_TYPES:
        if transformer_path is not None:
            raise ValueError(
                "--transformer-path is only valid for single-transformer models"
            )
        if high_path is None or low_path is None:
            raise ValueError(f"{model_type} requires both --high-path and --low-path")
        return [
            TransformerSource(transformer_dirs[0], high_path),
            TransformerSource(transformer_dirs[1], low_path),
        ]

    if high_path is not None or low_path is not None:
        raise ValueError("--high-path/--low-path are only valid for cascade models")
    if transformer_path is None:
        raise ValueError(f"{model_type} requires --transformer-path")
    return [TransformerSource(transformer_dirs[0], transformer_path)]


def _validate_patch_embedding_shape(
    *,
    source_path: pathlib.Path,
    source_prefix: str,
    transformer_config_path: pathlib.Path,
) -> None:
    config = _read_json(transformer_config_path)
    expected_in_channels = config.get("in_channels")
    patch_key = f"{source_prefix}patch_embedding.weight"

    with safe_open(str(source_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if patch_key not in keys:
            raise KeyError(f"{source_path} does not contain {patch_key}")
        shape = tuple(f.get_tensor(patch_key).shape)

    if len(shape) < 2:
        raise ValueError(f"{patch_key} has unexpected shape {shape} in {source_path}")
    if expected_in_channels is not None and shape[1] != expected_in_channels:
        raise ValueError(
            f"{source_path} patch_embedding input channels ({shape[1]}) do not match "
            f"{transformer_config_path} in_channels ({expected_in_channels})"
        )


def _save_shard(
    shard: dict[str, object],
    tmp_shards: list[tuple[pathlib.Path, list[str]]],
    output_dir: pathlib.Path,
) -> None:
    if not shard:
        return
    tmp_path = (
        output_dir / f".diffusion_pytorch_model-{len(tmp_shards) + 1:05d}.safetensors"
    )
    save_file(shard, tmp_path, metadata=SAFETENSORS_METADATA)
    tmp_shards.append((tmp_path, list(shard.keys())))
    logger.info("Wrote shard %s with %d tensors", tmp_path.name, len(shard))


def convert_comfyui_transformer(
    *,
    model_type: str,
    source_path: pathlib.Path,
    output_dir: pathlib.Path,
    transformer_config_path: pathlib.Path,
    source_prefix: str = COMFYUI_TRANSFORMER_PREFIX,
    max_shard_size_gb: float = DEFAULT_MAX_SHARD_SIZE_GB,
    validate_shapes: bool = True,
) -> None:
    """Convert one ComfyUI/original Wan transformer safetensors file."""
    _require_file(source_path, "ComfyUI transformer weights")
    _require_file(transformer_config_path, "Base transformer config")
    output_dir.mkdir(parents=True, exist_ok=True)

    if validate_shapes:
        _validate_patch_embedding_shape(
            source_path=source_path,
            source_prefix=source_prefix,
            transformer_config_path=transformer_config_path,
        )

    max_shard_bytes = int(max_shard_size_gb * 1024**3)
    if max_shard_bytes <= 0:
        raise ValueError("--max-shard-size-gb must be positive")

    seen_converted_keys: set[str] = set()
    weight_map: dict[str, str] = {}
    tmp_shards: list[tuple[pathlib.Path, list[str]]] = []
    current_shard: dict[str, object] = {}
    current_size = 0
    total_size = 0

    with safe_open(str(source_path), framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            if not key.startswith(source_prefix):
                raise ValueError(
                    f"{source_path} contains non-ComfyUI Wan key {key!r}; "
                    f"expected prefix {source_prefix!r}"
                )
            converted_key = convert_transformer_key(key, source_prefix=source_prefix)
            if converted_key in seen_converted_keys:
                raise ValueError(
                    f"Duplicate converted key {converted_key!r} while converting {source_path}"
                )
            seen_converted_keys.add(converted_key)

            tensor = f.get_tensor(key)
            tensor_size = _tensor_nbytes(tensor)
            if current_shard and current_size + tensor_size > max_shard_bytes:
                _save_shard(current_shard, tmp_shards, output_dir)
                current_shard = {}
                current_size = 0

            current_shard[converted_key] = tensor
            current_size += tensor_size
            total_size += tensor_size

    _save_shard(current_shard, tmp_shards, output_dir)

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
    logger.info(
        "Converted %s %s to %s (%d tensors, %d shard(s))",
        model_type,
        source_path,
        output_dir,
        len(weight_map),
        total_shards,
    )


def repack_comfyui_wan(
    *,
    model_type: str,
    original_model_path: pathlib.Path,
    output_path: pathlib.Path,
    high_path: pathlib.Path | None = None,
    low_path: pathlib.Path | None = None,
    transformer_path: pathlib.Path | None = None,
    text_encoder_weights_path: pathlib.Path | None = None,
    workflow_path: pathlib.Path | None = None,
    copy_components: bool = False,
    overwrite: bool = False,
    max_shard_size_gb: float = DEFAULT_MAX_SHARD_SIZE_GB,
    source_prefix: str = COMFYUI_TRANSFORMER_PREFIX,
    validate_shapes: bool = True,
    check_text_encoder_keys: bool = True,
) -> None:
    _require_dir(original_model_path, "Base Diffusers model")
    transformer_sources = _resolve_transformer_sources(
        model_type,
        high_path=high_path,
        low_path=low_path,
        transformer_path=transformer_path,
    )
    transformer_dirs = get_transformer_dirs(model_type)

    for source in transformer_sources:
        _require_file(source.weights_path, f"{source.output_dir_name} weights")

    _prepare_output_dir(output_path, overwrite)
    _copy_or_link_base_model_skeleton(
        original_model_path,
        output_path,
        transformer_dirs,
        copy_components=copy_components,
        replace_text_encoder=text_encoder_weights_path is not None,
    )

    if text_encoder_weights_path is not None:
        replace_text_encoder_weights(
            original_model_path,
            output_path,
            text_encoder_weights_path,
            copy_components=copy_components,
            check_keys=check_text_encoder_keys,
        )

    if workflow_path is not None:
        preserve_workflow_files(workflow_path, output_path)

    for source in transformer_sources:
        config_path = original_model_path / source.output_dir_name / "config.json"
        convert_comfyui_transformer(
            model_type=model_type,
            source_path=source.weights_path,
            output_dir=output_path / source.output_dir_name,
            transformer_config_path=config_path,
            source_prefix=source_prefix,
            max_shard_size_gb=max_shard_size_gb,
            validate_shapes=validate_shapes,
        )

    logger.info("Done. Repacked model saved to: %s", output_path)


def get_args():
    parser = argparse.ArgumentParser(
        description="Repack ComfyUI/original Wan safetensors into a Diffusers model tree"
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
        help="High-noise transformer safetensors path for Wan2.2 cascade models",
    )
    parser.add_argument(
        "--low-path",
        type=str,
        default=None,
        help="Low-noise transformer safetensors path for Wan2.2 cascade models",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Single transformer safetensors path for non-cascade models",
    )
    parser.add_argument(
        "--text-encoder-weights-path",
        type=str,
        default=None,
        help="Optional replacement UMT5 text_encoder safetensors path",
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
        "--source-prefix",
        type=str,
        default=COMFYUI_TRANSFORMER_PREFIX,
        help="Prefix to strip from ComfyUI transformer keys",
    )
    parser.add_argument(
        "--no-validate-shapes",
        action="store_true",
        help="Skip patch_embedding shape checks against base transformer config",
    )
    parser.add_argument(
        "--no-text-encoder-key-check",
        action="store_true",
        help="Skip key-set validation for --text-encoder-weights-path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    repack_comfyui_wan(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model_path),
        output_path=pathlib.Path(args.output_path),
        high_path=pathlib.Path(args.high_path) if args.high_path else None,
        low_path=pathlib.Path(args.low_path) if args.low_path else None,
        transformer_path=(
            pathlib.Path(args.transformer_path) if args.transformer_path else None
        ),
        text_encoder_weights_path=(
            pathlib.Path(args.text_encoder_weights_path)
            if args.text_encoder_weights_path
            else None
        ),
        workflow_path=pathlib.Path(args.workflow_path) if args.workflow_path else None,
        copy_components=args.copy_components,
        overwrite=args.overwrite,
        max_shard_size_gb=args.max_shard_size_gb,
        source_prefix=args.source_prefix,
        validate_shapes=not args.no_validate_shapes,
        check_text_encoder_keys=not args.no_text_encoder_key_check,
    )
