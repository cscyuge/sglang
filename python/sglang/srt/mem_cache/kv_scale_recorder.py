from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

__all__ = [
    "KVScaleRecorder",
    "set_active_recorder",
    "get_active_recorder",
    "clear_active_recorder",
    "is_active",
    "record",
]


@dataclass
class _LayerStats:
    max_k: torch.Tensor
    max_v: torch.Tensor


class KVScaleRecorder:
    """Track per-layer max abs values for K/V tensors."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.layer_stats: Dict[int, _LayerStats] = {}
        self.start_ts = time.time()
        self.sample_count = 0

    def _ensure_layer(self, layer_id: int) -> _LayerStats:
        stats = self.layer_stats.get(layer_id)
        if stats is None:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            stats = _LayerStats(max_k=zero.clone(), max_v=zero.clone())
            self.layer_stats[layer_id] = stats
        return stats

    @torch.no_grad()
    def record(
        self,
        layer_id: int,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> None:
        stats = self._ensure_layer(layer_id)

        if cache_k is not None and cache_k.numel() > 0:
            max_k = torch.amax(cache_k.detach().abs()).to(torch.float32)
            stats.max_k = torch.maximum(stats.max_k, max_k)
            self.sample_count += cache_k.shape[0]

        if cache_v is not None and cache_v.numel() > 0:
            max_v = torch.amax(cache_v.detach().abs()).to(torch.float32)
            stats.max_v = torch.maximum(stats.max_v, max_v)

    def _scale_map(self) -> Dict[int, Dict[str, float]]:
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        result: Dict[int, Dict[str, float]] = {}
        for layer_id, stats in self.layer_stats.items():
            max_k = float(stats.max_k.cpu().item())
            max_v = float(stats.max_v.cpu().item())
            layer_entry: Dict[str, float] = {}
            if max_k > 0.0:
                layer_entry["k_scale"] = max_k / fp8_max
                layer_entry["k_absmax"] = max_k
            if max_v > 0.0:
                layer_entry["v_scale"] = max_v / fp8_max
                layer_entry["v_absmax"] = max_v
            if layer_entry:
                result[layer_id] = layer_entry
        return result

    def summary(self) -> Dict[str, Dict[str, float]]:
        layer_map = self._scale_map()
        k_map: Dict[str, float] = {}
        v_map: Dict[str, float] = {}
        for layer_id, entry in layer_map.items():
            key = str(layer_id)
            if "k_scale" in entry:
                k_map[key] = entry["k_scale"]
            if "v_scale" in entry:
                v_map[key] = entry["v_scale"]
        return {"k_scale": k_map, "v_scale": v_map}


_ACTIVE_RECORDER: Optional[KVScaleRecorder] = None


def set_active_recorder(recorder: KVScaleRecorder) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder


def get_active_recorder() -> Optional[KVScaleRecorder]:
    return _ACTIVE_RECORDER


def clear_active_recorder() -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = None


def is_active() -> bool:
    return _ACTIVE_RECORDER is not None


def record(
    layer_id: int,
    cache_k: Optional[torch.Tensor],
    cache_v: Optional[torch.Tensor],
) -> None:
    recorder = _ACTIVE_RECORDER
    if recorder is None:
        return
    recorder.record(layer_id, cache_k, cache_v)
