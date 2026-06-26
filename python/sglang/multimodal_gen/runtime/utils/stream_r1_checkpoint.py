# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from glob import glob
import os
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader.utils import hf_to_custom_state_dict

_STATE_DICT_KEYS = ("generator_ema", "generator", "model", "state_dict")
_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
)
_ROOT_MODULE_PREFIXES = ("module.", "model.")


@dataclass(frozen=True)
class StreamR1CheckpointLoadInfo:
    checkpoint_path: str
    source_key: str | None
    num_tensors: int
    skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _is_mapping_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(k, str) for k in value)


def select_stream_r1_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool,
) -> tuple[Mapping[str, Any], str | None]:
    """Select the Stream-R1 generator state dict from a training checkpoint."""

    candidate_keys = _STATE_DICT_KEYS if use_ema else _STATE_DICT_KEYS[1:]
    for key in candidate_keys:
        value = checkpoint.get(key)
        if _is_mapping_state_dict(value):
            return value, key
    if _is_mapping_state_dict(checkpoint):
        return checkpoint, None
    raise TypeError("Stream-R1 checkpoint must be a mapping state dict")


def clean_stream_r1_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Remove wrapper prefixes commonly produced by FSDP/checkpoint wrappers."""

    cleaned: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if isinstance(value, nn.Parameter):
            value = value.detach()
        if not torch.is_tensor(value):
            skipped.append(key)
            continue

        clean_key = key
        for prefix in _WRAPPER_PREFIXES:
            clean_key = clean_key.replace(prefix, "")
        stripped = True
        while stripped:
            stripped = False
            for prefix in _ROOT_MODULE_PREFIXES:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    stripped = True
                    break
        cleaned[clean_key] = value

    return cleaned, tuple(skipped)


def resolve_stream_r1_checkpoint_path(checkpoint_path: str) -> str:
    path = os.path.expanduser(checkpoint_path)
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        base_name = os.path.basename(os.path.normpath(path))
        preferred_paths = [
            os.path.join(path, f"{base_name}.pt"),
            os.path.join(path, "model.pt"),
            os.path.join(path, "checkpoint.pt"),
        ]
        for preferred_path in preferred_paths:
            if os.path.isfile(preferred_path):
                return preferred_path

        candidates: list[str] = []
        for pattern in ("*.pt", "*.pth", "*.bin"):
            candidates.extend(glob(os.path.join(path, pattern)))
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                "No Stream-R1 checkpoint file found under directory: "
                f"{checkpoint_path}"
            )
        raise ValueError(
            "Expected one Stream-R1 checkpoint file under directory "
            f"{checkpoint_path}, found: {candidates}"
        )

    raise FileNotFoundError(
        f"Stream-R1 generator checkpoint not found: {checkpoint_path}"
    )


def load_stream_r1_generator_checkpoint(
    module: nn.Module,
    checkpoint_path: str,
    *,
    use_ema: bool = True,
    strict: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
) -> StreamR1CheckpointLoadInfo:
    """Overlay a Stream-R1 DMD generator checkpoint onto a loaded transformer."""

    if not checkpoint_path:
        raise ValueError("checkpoint_path must be non-empty")
    checkpoint_path = resolve_stream_r1_checkpoint_path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not _is_mapping_state_dict(checkpoint):
        raise TypeError("Stream-R1 checkpoint must load to a mapping")

    selected_state_dict, source_key = select_stream_r1_state_dict(
        checkpoint, use_ema=use_ema
    )
    cleaned_state_dict, skipped_keys = clean_stream_r1_state_dict_keys(
        selected_state_dict
    )
    if not cleaned_state_dict:
        raise ValueError(
            f"Stream-R1 checkpoint has no tensor values under {source_key or '<root>'}"
        )

    if param_names_mapping is not None:
        cleaned_state_dict, _ = hf_to_custom_state_dict(
            cleaned_state_dict,
            param_names_mapping,
            valid_target_names=set(module.state_dict().keys()),
        )

    incompatible = module.load_state_dict(cleaned_state_dict, strict=strict)
    if incompatible.unexpected_keys and len(incompatible.unexpected_keys) == len(
        cleaned_state_dict
    ):
        raise RuntimeError(
            "Stream-R1 checkpoint did not match any transformer parameters. "
            "Check that the checkpoint belongs to the Wan S2V generator."
        )
    return StreamR1CheckpointLoadInfo(
        checkpoint_path=checkpoint_path,
        source_key=source_key,
        num_tensors=len(cleaned_state_dict),
        skipped_keys=skipped_keys,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
