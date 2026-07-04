# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V DiT runtime model.

This adapts the official ``WanModel_S2V`` structure to SGLang's DiT runtime
layers so the original and block-FP8 checkpoints can be loaded by the existing
FSDP/quantization loader.
"""

import math
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.configs.models.dits import WanS2VConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanS2VArchConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.wan_s2v_stream_r1 import (
    WanS2VKVCacheBlock,
    WanS2VStreamR1AttentionLayout,
    WanS2VTimestepStaticMetadataBuffers,
    run_wan_s2v_stream_r1_cached_self_attention,
    update_wan_s2v_stream_r1_cached_self_attention_kv_cache,
    validate_wan_s2v_stream_r1_forward_cache,
    wan_s2v_stream_r1_uses_head_sharded_sp_kv_cache,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:
    from diffusers.models.attention import AdaLayerNorm
except ImportError:  # pragma: no cover
    AdaLayerNorm = None

logger = init_logger(__name__)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)


def _as_list_4d(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(x, list):
        return x
    if x.dim() == 4:
        return [x]
    if x.dim() == 5:
        return [u for u in x]
    raise ValueError(f"Expected 4D/5D tensor or list, got shape {tuple(x.shape)}")


def _build_s2v_noisy_rope_grid_sizes(
    grid_sizes: torch.Tensor,
    *,
    stream_r1_mode: bool = False,
    current_start: int = 0,
    frame_seq_length: int = 0,
) -> list[list[torch.Tensor]]:
    if not stream_r1_mode:
        return [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

    current_start = int(current_start)
    frame_seq_length = int(frame_seq_length)
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    if current_start < 0:
        raise ValueError("current_start must be non-negative")
    if current_start % frame_seq_length != 0:
        raise ValueError("current_start must be frame-aligned")

    frame_offset = current_start // frame_seq_length
    start = torch.zeros_like(grid_sizes)
    start[:, 0] = frame_offset
    end = grid_sizes.clone()
    end[:, 0] += frame_offset
    return [[start, end, grid_sizes]]


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _linspace_index(
    start: int,
    end: int,
    steps: int,
    device: torch.device,
) -> np.ndarray | torch.Tensor:
    if (
        torch.device(device).type == "cuda"
        and _cuda_graph_capture_active()
    ):
        if steps == 1:
            return torch.full((1,), int(start), dtype=torch.long, device=device)
        return torch.linspace(
            float(start), float(end), int(steps), device=device
        ).to(torch.long)
    return np.linspace(start, end, steps).astype(int)


def _clone_as_mutable_cache_tensor(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().clone()


def _is_inference_tensor(tensor: torch.Tensor) -> bool:
    is_inference = getattr(tensor, "is_inference", None)
    return bool(is_inference()) if callable(is_inference) else False


def _rope_precompute(
    x: torch.Tensor,
    grid_sizes,
    freqs: torch.Tensor | list[torch.Tensor],
    start=None,
) -> torch.Tensor:
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    trainable_freqs = None
    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = torch.empty((b, s, n, c), dtype=torch.complex64, device=x.device)
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g, g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            f_o, h_o, w_o = int(f_o), int(h_o), int(w_o)
            f, h, w = int(f), int(h), int(w)
            t_f, t_h, t_w = int(t_f), int(t_h), int(t_w)
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len <= 0:
                continue
            if t_f > 0:
                if f_o >= 0:
                    f_sam = _linspace_index(
                        f_o, t_f + f_o - 1, seq_f, freqs[0].device
                    )
                else:
                    f_sam = _linspace_index(
                        -f_o, -t_f - f_o + 1, seq_f, freqs[0].device
                    )
                h_sam = _linspace_index(
                    h_o, t_h + h_o - 1, seq_h, freqs[1].device
                )
                w_sam = _linspace_index(
                    w_o, t_w + w_o - 1, seq_w, freqs[2].device
                )
                assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                freqs_i = torch.cat(
                    [
                        freqs_0.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam]
                        .view(1, seq_h, 1, -1)
                        .expand(seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam]
                        .view(1, 1, seq_w, -1)
                        .expand(seq_f, seq_h, seq_w, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, 1, -1)
            elif t_f < 0 and trainable_freqs is not None:
                freqs_i = trainable_freqs.unsqueeze(1)
            else:
                continue
            output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


def _rope_precompute_s2v_stream_r1_tensor_current_start(
    x: torch.Tensor,
    noisy_grid_sizes: torch.Tensor,
    ref_grid_sizes,
    freqs: torch.Tensor | list[torch.Tensor],
    *,
    current_start: torch.Tensor,
    frame_seq_length: int,
) -> torch.Tensor:
    """Precompute S2V noisy/ref RoPE with tensor-driven noisy frame offset."""

    frame_seq_length = int(frame_seq_length)
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    if noisy_grid_sizes.dim() != 2 or noisy_grid_sizes.shape[1] != 3:
        raise ValueError("noisy_grid_sizes must have shape [B, 3]")

    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    base_freqs = freqs[0] if isinstance(freqs, list) else freqs
    freq_f, freq_h, freq_w = base_freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3],
        dim=1,
    )
    output = torch.empty((b, s, n, c), dtype=torch.complex64, device=x.device)
    current_start = current_start.to(device=freq_f.device, dtype=torch.long).reshape(())
    frame_offset = torch.div(
        current_start,
        frame_seq_length,
        rounding_mode="floor",
    )

    if noisy_grid_sizes.device.type != "cpu":
        if _cuda_graph_capture_active():
            raise RuntimeError(
                "Stream-R1 tensor-current-start RoPE requires host noisy grid sizes "
                "during CUDA graph capture"
            )
        noisy_grid_sizes = noisy_grid_sizes.cpu()

    noisy_seq_len = 0
    for i in range(noisy_grid_sizes.shape[0]):
        seq_f = int(noisy_grid_sizes[i, 0])
        seq_h = int(noisy_grid_sizes[i, 1])
        seq_w = int(noisy_grid_sizes[i, 2])
        seq_len = int(seq_f * seq_h * seq_w)
        if seq_len <= 0:
            continue
        noisy_seq_len = max(noisy_seq_len, seq_len)
        f_sam = frame_offset + torch.arange(
            seq_f,
            dtype=torch.long,
            device=freq_f.device,
        )
        h_sam = torch.arange(seq_h, dtype=torch.long, device=freq_h.device)
        w_sam = torch.arange(seq_w, dtype=torch.long, device=freq_w.device)
        freqs_i = torch.cat(
            [
                freq_f[f_sam]
                .view(seq_f, 1, 1, -1)
                .expand(seq_f, seq_h, seq_w, -1),
                freq_h[h_sam]
                .view(1, seq_h, 1, -1)
                .expand(seq_f, seq_h, seq_w, -1),
                freq_w[w_sam]
                .view(1, 1, seq_w, -1)
                .expand(seq_f, seq_h, seq_w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        output[i, :seq_len] = freqs_i

    if ref_grid_sizes:
        ref_freqs = _rope_precompute(x[:, noisy_seq_len:], ref_grid_sizes, freqs)
        ref_seq_len = ref_freqs.shape[1]
        output[:, noisy_seq_len : noisy_seq_len + ref_seq_len] = ref_freqs
    return output


def _slice_s2v_audio_embeddings_with_tensor_start(
    audio_emb: torch.Tensor,
    *,
    audio_start_frame: torch.Tensor,
    motion_prefix_frames: torch.Tensor | int,
    latent_frames: int,
) -> torch.Tensor:
    """Slice S2V audio embeddings with tensor-provided chunk start metadata."""

    latent_frames = int(latent_frames)
    if latent_frames <= 0:
        raise ValueError("latent_frames must be positive")
    if audio_emb.dim() < 2:
        raise ValueError("audio embeddings must have at least batch/time dimensions")
    audio_start = audio_start_frame.to(device=audio_emb.device, dtype=torch.long).reshape(
        ()
    )
    if isinstance(motion_prefix_frames, torch.Tensor):
        motion_prefix = motion_prefix_frames.to(
            device=audio_emb.device,
            dtype=torch.long,
        ).reshape(())
    else:
        motion_prefix = torch.tensor(
            int(motion_prefix_frames),
            dtype=torch.long,
            device=audio_emb.device,
        )
    indices = (
        audio_start
        + motion_prefix
        + torch.arange(latent_frames, dtype=torch.long, device=audio_emb.device)
    )
    return torch.index_select(audio_emb, dim=1, index=indices)


def _rope_apply_precomputed(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    freqs = freqs[:, : x.shape[1]]
    if x.device.type == "cuda":
        try:
            from sglang.jit_kernel.diffusion.triton.wan_s2v_rope import (
                apply_wan_s2v_rope,
            )

            return apply_wan_s2v_rope(x, freqs)
        except Exception:
            pass

    out = torch.view_as_complex(x.to(torch.float32).reshape(*x.shape[:-1], -1, 2))
    if freqs.dtype != torch.complex64:
        freqs = freqs.to(torch.complex64)
    out = torch.view_as_real(out * freqs).flatten(3)
    return out.to(x.dtype)


def _segment_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    seg_idx: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x.device.type == "cuda":
        try:
            from sglang.jit_kernel.diffusion.triton.wan_s2v_segment import (
                segment_modulate,
            )

            return segment_modulate(
                x, shift, scale, seg_idx, out_dtype=out_dtype
            )
        except Exception:
            pass

    seg_idx = min(max(0, seg_idx), x.size(1))
    parts = [
        x[:, :seg_idx] * (1 + scale[:, 0:1]) + shift[:, 0:1],
        x[:, seg_idx:] * (1 + scale[:, 1:2]) + shift[:, 1:2],
    ]
    out = torch.cat(parts, dim=1)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def _segment_gate(x: torch.Tensor, gate: torch.Tensor, seg_idx: int) -> torch.Tensor:
    seg_idx = min(max(0, seg_idx), x.size(1))
    return torch.cat(
        [x[:, :seg_idx] * gate[:, 0:1], x[:, seg_idx:] * gate[:, 1:2]], dim=1
    )


def _segment_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    seg_idx: int,
) -> torch.Tensor:
    if residual.device.type == "cuda":
        try:
            from sglang.jit_kernel.diffusion.triton.wan_s2v_segment import (
                segment_gate_add,
            )

            return segment_gate_add(residual, update, gate, seg_idx)
        except Exception:
            pass

    return (residual + _segment_gate(update, gate, seg_idx)).to(residual.dtype)


def _pad_stream_r1_attention_mask_for_sp(
    attn_mask: torch.Tensor,
    original_seq_len: int,
    pad_tokens: int,
) -> torch.Tensor:
    if pad_tokens <= 0:
        return attn_mask
    if attn_mask.dim() != 3:
        raise ValueError("Stream-R1 SP attention mask must be [B, S, S]")
    if (
        attn_mask.shape[-2] != original_seq_len
        or attn_mask.shape[-1] != original_seq_len
    ):
        raise ValueError(
            "Stream-R1 SP attention mask shape does not match the unpadded sequence"
        )

    padded_seq_len = original_seq_len + pad_tokens
    padded_mask = torch.ones(
        (attn_mask.shape[0], padded_seq_len, padded_seq_len),
        dtype=attn_mask.dtype,
        device=attn_mask.device,
    )
    padded_mask[:, :original_seq_len, :original_seq_len] = attn_mask
    padded_mask[:, :original_seq_len, original_seq_len:] = False
    return padded_mask


class CausalConv1d(nn.Module):
    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.time_causal_padding, mode=self.pad_mode))


class MotionEncoderTC(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_heads: int, need_global: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, stride=2)
        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "b t c -> b c t")
        x_ori = x.clone()
        b, _, _ = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.act(self.norm1(x))
        x = self.conv2(rearrange(x, "b t c -> b c t"))
        x = self.act(self.norm2(rearrange(x, "b c t -> b t c")))
        x = self.conv3(rearrange(x, "b t c -> b c t"))
        x = self.act(self.norm3(rearrange(x, "b c t -> b t c")))
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        x_local = torch.cat(
            [x, self.padding_tokens.repeat(b, x.shape[1], 1, 1)], dim=-2
        )
        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = self.act(self.norm1(rearrange(x, "b c t -> b t c")))
        x = self.conv2(rearrange(x, "b t c -> b c t"))
        x = self.act(self.norm2(rearrange(x, "b c t -> b t c")))
        x = self.conv3(rearrange(x, "b t c -> b c t"))
        x = self.act(self.norm3(rearrange(x, "b c t -> b t c")))
        x = self.final_linear(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 25,
        out_dim: int = 5120,
        num_token: int = 4,
        need_global: bool = True,
    ):
        super().__init__()
        self.encoder = MotionEncoderTC(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global,
        )
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor):
        weights = self.act(self.weights)
        weighted_feat = ((features * weights) / weights.sum(dim=1, keepdims=True)).sum(
            dim=1
        )
        weighted_feat = weighted_feat.permute(0, 2, 1)
        return self.encoder(weighted_feat)


class FramePackMotioner(nn.Module):
    def __init__(
        self,
        inner_dim: int = 5120,
        num_heads: int = 40,
        zip_frame_buckets: tuple[int, int, int] = (1, 2, 16),
        drop_mode: str = "padd",
    ):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_bucket_values = tuple(int(item) for item in zip_frame_buckets)
        self.register_buffer(
            "zip_frame_buckets",
            torch.tensor(self.zip_frame_bucket_values, dtype=torch.long),
            persistent=False,
        )
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        d = inner_dim // num_heads
        self.register_buffer(
            "freqs",
            torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ),
            persistent=False,
        )
        self.drop_mode = drop_mode

    def forward(self, motion_latents: list[torch.Tensor], add_last_motion: int = 2):
        mot, mot_remb = [], []
        buckets = self.zip_frame_bucket_values
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(
                16,
                sum(buckets),
                lat_height,
                lat_width,
                device=m.device,
                dtype=m.dtype,
            )
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]
            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = sum(buckets[: len(buckets) - add_last_motion - 1])
                if zero_end_frame > 0:
                    padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_4x, clean_2x, clean_post = padd_lat.split(
                list(buckets)[::-1], dim=2
            )
            clean_post = self.proj(clean_post).flatten(2).transpose(1, 2)
            clean_2x = self.proj_2x(clean_2x).flatten(2).transpose(1, 2)
            clean_4x = self.proj_4x(clean_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_post = clean_post[:, :0]
                if add_last_motion < 1:
                    clean_2x = clean_2x[:, :0]
            motion_lat = torch.cat([clean_post, clean_2x, clean_4x], dim=1)

            grid_sizes = []
            if not (add_last_motion < 2 and self.drop_mode == "drop"):
                grid_sizes.append(
                    [
                        torch.tensor([-sum(buckets[:1]), 0, 0]).view(1, 3),
                        torch.tensor(
                            [
                                -sum(buckets[:1]) + buckets[0],
                                lat_height // 2,
                                lat_width // 2,
                            ]
                        ).view(1, 3),
                        torch.tensor(
                            [buckets[0], lat_height // 2, lat_width // 2]
                        ).view(1, 3),
                    ]
                )
            if not (add_last_motion < 1 and self.drop_mode == "drop"):
                grid_sizes.append(
                    [
                        torch.tensor([-sum(buckets[:2]), 0, 0]).view(1, 3),
                        torch.tensor(
                            [
                                -sum(buckets[:2]) + buckets[1] // 2,
                                lat_height // 4,
                                lat_width // 4,
                            ]
                        ).view(1, 3),
                        torch.tensor(
                            [buckets[1], lat_height // 2, lat_width // 2]
                        ).view(1, 3),
                    ]
                )
            grid_sizes.append(
                [
                    torch.tensor([-sum(buckets[:3]), 0, 0]).view(1, 3),
                    torch.tensor(
                        [
                            -sum(buckets[:3]) + buckets[2] // 4,
                            lat_height // 8,
                            lat_width // 8,
                        ]
                    ).view(1, 3),
                    torch.tensor(
                        [buckets[2], lat_height // 2, lat_width // 2]
                    ).view(1, 3),
                ]
            )
            freqs = self.freqs.to(device=m.device)
            motion_rope_emb = _rope_precompute(
                motion_lat.detach().view(
                    1,
                    motion_lat.shape[1],
                    self.num_heads,
                    self.inner_dim // self.num_heads,
                ),
                grid_sizes,
                freqs,
            )
            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class WanS2VTransformerBlock(WanTransformerBlock):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("fused_qkv", True)
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: list[torch.Tensor | int],
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        stream_r1_kv_cache: WanS2VKVCacheBlock | None = None,
        stream_r1_attention_layout: WanS2VStreamR1AttentionLayout | None = None,
        cache_start: int | None = None,
        stream_r1_sequence_shard_enabled: bool = False,
        stream_r1_sp_pad_tokens: int = 0,
        stream_r1_cache_update_only: bool = False,
        stream_r1_graph_kv_update: bool = False,
        crossattn_kv_cache: dict | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype
        seg_idx = int(temb[1])
        e = self.scale_shift_table.unsqueeze(2) + temb[0].float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            x.squeeze(1) for x in e.chunk(6, dim=1)
        )

        norm_hidden_states = _segment_modulate(
            self.norm1.norm(hidden_states),
            shift_msa,
            scale_msa,
            seg_idx,
            out_dtype=orig_dtype,
        )
        query, key, value = self._project_self_attn_qkv(norm_hidden_states)
        if self.norm_q is not None:
            query = (
                tensor_parallel_rms_norm(query, self.norm_q)
                if self.tp_rmsnorm
                else self.norm_q(query)
            )
        if self.norm_k is not None:
            key = (
                tensor_parallel_rms_norm(key, self.norm_k)
                if self.tp_rmsnorm
                else self.norm_k(key)
            )
        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        query = _rope_apply_precomputed(query, freqs_cis).to(orig_dtype)
        key = _rope_apply_precomputed(key, freqs_cis).to(orig_dtype)
        if stream_r1_cache_update_only:
            if stream_r1_kv_cache is None or stream_r1_attention_layout is None:
                raise ValueError(
                    "Stream-R1 cache-update-only block requires KV cache and layout"
                )
            update_wan_s2v_stream_r1_cached_self_attention_kv_cache(
                query=query,
                key=key,
                value=value,
                kv_cache=stream_r1_kv_cache,
                layout=stream_r1_attention_layout,
                cache_start=cache_start,
                sequence_shard_enabled=stream_r1_sequence_shard_enabled,
                sp_pad_tokens=stream_r1_sp_pad_tokens,
                graph_kv_update=stream_r1_graph_kv_update,
            )
            return hidden_states.to(orig_dtype)
        if stream_r1_kv_cache is not None or stream_r1_attention_layout is not None:
            attn_output = run_wan_s2v_stream_r1_cached_self_attention(
                self.attn1,
                query=query,
                key=key,
                value=value,
                kv_cache=stream_r1_kv_cache,
                layout=stream_r1_attention_layout,
                cache_start=cache_start,
                sequence_shard_enabled=stream_r1_sequence_shard_enabled,
                sp_pad_tokens=stream_r1_sp_pad_tokens,
                graph_kv_update=stream_r1_graph_kv_update,
            ).flatten(2)
        else:
            attn_output = self.attn1(query, key, value, attn_mask=attn_mask).flatten(2)
        attn_output, _ = self.to_out(attn_output)
        hidden_states = _segment_gate_add(
            hidden_states, attn_output.squeeze(1), gate_msa, seg_idx
        )
        hidden_states = hidden_states.to(orig_dtype)

        # Populate or refresh cross-attn K/V cache. Server startup graph
        # pre-capture reuses these tensor addresses across real sessions, so a
        # new request refreshes contents in place instead of replacing tensors.
        if crossattn_kv_cache is not None and (
            bool(crossattn_kv_cache.get("needs_update", False))
            or not isinstance(crossattn_kv_cache.get("k"), torch.Tensor)
            or not isinstance(crossattn_kv_cache.get("v"), torch.Tensor)
        ):
            ctx_k, _ = self.attn2.to_k(encoder_hidden_states)
            if self.attn2.tp_rmsnorm:
                ctx_k = tensor_parallel_rms_norm(ctx_k, self.attn2.norm_k)
            else:
                ctx_k = self.attn2.norm_k(ctx_k)
            ctx_k = ctx_k.unflatten(
                2, (self.attn2.local_num_heads, self.attn2.head_dim)
            )
            ctx_v, _ = self.attn2.to_v(encoder_hidden_states)
            ctx_v = ctx_v.unflatten(
                2, (self.attn2.local_num_heads, self.attn2.head_dim)
            )
            cached_k = crossattn_kv_cache.get("k")
            cached_v = crossattn_kv_cache.get("v")
            if (
                isinstance(cached_k, torch.Tensor)
                and cached_k.shape == ctx_k.shape
                and cached_k.dtype == ctx_k.dtype
                and cached_k.device == ctx_k.device
                and not _is_inference_tensor(cached_k)
            ):
                cached_k.copy_(ctx_k)
                ctx_k = cached_k
            else:
                ctx_k = _clone_as_mutable_cache_tensor(ctx_k)
            if (
                isinstance(cached_v, torch.Tensor)
                and cached_v.shape == ctx_v.shape
                and cached_v.dtype == ctx_v.dtype
                and cached_v.device == ctx_v.device
                and not _is_inference_tensor(cached_v)
            ):
                cached_v.copy_(ctx_v)
                ctx_v = cached_v
            else:
                ctx_v = _clone_as_mutable_cache_tensor(ctx_v)
            crossattn_kv_cache["k"] = ctx_k
            crossattn_kv_cache["v"] = ctx_v
            crossattn_kv_cache["needs_update"] = False

        attn_output = self.attn2(
            self.self_attn_residual_norm.norm(hidden_states),
            context=encoder_hidden_states,
            context_lens=None,
            cached_kv=crossattn_kv_cache,
        )
        hidden_states = hidden_states + attn_output
        norm_hidden_states = _segment_modulate(
            self.cross_attn_residual_norm.norm(hidden_states),
            c_shift_msa,
            c_scale_msa,
            seg_idx,
            out_dtype=orig_dtype,
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = _segment_gate_add(hidden_states, ff_output, c_gate_msa, seg_idx)
        return hidden_states.to(orig_dtype)


class WanS2VAudioInjector(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        inject_layers: list[int],
        enable_adain: bool,
        adain_dim: int,
        supported_attention_backends: set[AttentionBackendEnum],
    ):
        super().__init__()
        self.injected_block_id = {layer: i for i, layer in enumerate(inject_layers)}
        self.injector = nn.ModuleList(
            [
                WanT2VCrossAttention(
                    dim,
                    num_heads,
                    qk_norm="rms_norm_across_heads",
                    eps=1e-6,
                    prefix=f"audio_injector.injector.{i}",
                    supported_attention_backends=supported_attention_backends,
                )
                for i in range(len(inject_layers))
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                for _ in inject_layers
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                for _ in inject_layers
            ]
        )
        if enable_adain:
            if AdaLayerNorm is None:
                raise ImportError("diffusers is required for Wan S2V AdaLayerNorm")
            self.injector_adain_layers = nn.ModuleList(
                [
                    AdaLayerNorm(
                        output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1
                    )
                    for _ in inject_layers
                ]
            )


class WanS2VTransformer3DModel(WanTransformer3DModel):
    _fsdp_shard_conditions = WanS2VConfig()._fsdp_shard_conditions
    _compile_conditions = WanS2VConfig()._compile_conditions
    _supported_attention_backends = WanS2VConfig()._supported_attention_backends
    param_names_mapping = WanS2VArchConfig().param_names_mapping

    def __init__(
        self,
        config: WanS2VConfig,
        hf_config: dict[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super(WanTransformer3DModel, self).__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.out_channels
        self.patch_size = arch.patch_size
        self.text_len = arch.text_len
        self.freq_dim = arch.freq_dim
        self.enable_framepack = arch.enable_framepack
        self.enable_motioner = arch.enable_motioner
        self.add_last_motion = arch.add_last_motion
        self.zero_timestep = arch.zero_timestep
        self.enable_adain = arch.enable_adain
        self.adain_mode = arch.adain_mode
        try:
            self.sp_size = get_sp_world_size()
        except AssertionError:
            self.sp_size = 1
        self.use_context_parallel = False
        self.sequence_shard_start = 0

        self.patch_embedding = PatchEmbed(
            in_chans=arch.in_channels,
            embed_dim=inner_dim,
            patch_size=arch.patch_size,
            flatten=False,
        )
        self.cond_encoder = PatchEmbed(
            in_chans=arch.cond_dim,
            embed_dim=inner_dim,
            patch_size=arch.patch_size,
            flatten=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=arch.freq_dim,
            text_embed_dim=arch.text_dim,
        )

        attn_backend = get_global_server_args().attention_backend
        if attn_backend and attn_backend.lower() == "video_sparse_attn":
            logger.warning(
                "Wan S2V does not use video_sparse_attn; falling back to original attention"
            )
        self.blocks = nn.ModuleList(
            [
                WanS2VTransformerBlock(
                    inner_dim,
                    arch.ffn_dim,
                    arch.num_attention_heads,
                    arch.qk_norm,
                    arch.cross_attn_norm,
                    arch.eps,
                    None,
                    self._supported_attention_backends,
                    prefix=f"blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(arch.num_layers)
            ]
        )

        self.norm_out = LayerNormScaleShift(
            inner_dim, eps=arch.eps, elementwise_affine=False, dtype=torch.float32
        )
        self.proj_out = ReplicatedLinear(
            inner_dim,
            arch.out_channels * math.prod(arch.patch_size),
            bias=True,
            prefix="proj_out",
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )
        self.trainable_cond_mask = nn.Embedding(3, inner_dim)
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=arch.audio_dim,
            out_dim=inner_dim,
            num_token=arch.num_audio_token,
            need_global=arch.enable_adain,
        )
        self.audio_injector = WanS2VAudioInjector(
            dim=inner_dim,
            num_heads=arch.num_attention_heads,
            inject_layers=arch.audio_inject_layers,
            enable_adain=arch.enable_adain,
            adain_dim=inner_dim,
            supported_attention_backends=self._supported_attention_backends,
        )
        if arch.enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=inner_dim,
                num_heads=arch.num_attention_heads,
                drop_mode=arch.framepack_drop_mode,
        )

        d = inner_dim // arch.num_attention_heads
        self.register_buffer(
            "freqs",
            torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ),
            persistent=False,
        )
        self.layer_names = ["blocks"]
        self.cnt = 0
        self.stream_r1_local_attn_size: int | None = None
        self.stream_r1_sink_size: int | None = None
        self.stream_r1_num_frame_per_block: int | None = None
        self.stream_r1_kv_cache_requested = False
        self._logged_stream_r1_sp_no_kv_mask = False
        self._logged_stream_r1_sp_kv_mask = False
        self.__post_init__()

    def _process_motion_frame_pack(
        self,
        motion_latents: list[torch.Tensor],
        drop_motion_frames: bool = False,
        add_last_motion: int = 2,
    ):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        return flattern_mot, mot_remb

    def _inject_motion(
        self,
        x,
        seq_lens,
        rope_embs,
        mask_input,
        motion_latents,
        drop_motion_frames=False,
        add_last_motion=2,
    ):
        if self.enable_framepack:
            mot, mot_remb = self._process_motion_frame_pack(
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion,
            )
        else:
            mot, mot_remb = [], []
        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor(
                [r.size(1) for r in mot], dtype=torch.long
            )
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [
                torch.cat(
                    [
                        m,
                        2
                        * torch.ones(
                            [1, u.shape[1] - m.shape[1]],
                            device=m.device,
                            dtype=m.dtype,
                        ),
                    ],
                    dim=1,
                )
                for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def _run_audio_injector(
        self,
        audio_attn_id: int,
        input_hidden_states: torch.Tensor,
        attn_audio_emb: torch.Tensor,
        audio_global_temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.enable_adain and self.adain_mode == "attn_norm":
            if audio_global_temb is None:
                raise ValueError(
                    "audio_global_temb is required for Wan S2V AdaIN injection"
                )
            attn_hidden_states = self.audio_injector.injector_adain_layers[
                audio_attn_id
            ](input_hidden_states, temb=audio_global_temb)
        else:
            attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                audio_attn_id
            ](input_hidden_states)
        return self.audio_injector.injector[audio_attn_id](
            x=attn_hidden_states,
            context=attn_audio_emb,
            context_lens=torch.ones(
                attn_hidden_states.shape[0],
                dtype=torch.long,
                device=attn_hidden_states.device,
            )
            * attn_audio_emb.shape[1],
        )

    def _after_transformer_block(self, block_idx: int, hidden_states: torch.Tensor):
        if block_idx not in self.audio_injector.injected_block_id:
            return hidden_states

        audio_attn_id = self.audio_injector.injected_block_id[block_idx]
        audio_emb = self.merged_audio_emb
        num_frames = audio_emb.shape[1]
        if self.use_context_parallel:
            return self._after_transformer_block_sequence_shard(
                audio_attn_id,
                hidden_states,
                audio_emb,
                num_frames,
            )

        input_hidden_states = hidden_states[:, : self.original_seq_len].clone()
        input_hidden_states = rearrange(
            input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames
        )
        attn_audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
        audio_global_temb = None
        if self.enable_adain and self.adain_mode == "attn_norm":
            audio_emb_global = rearrange(self.audio_emb_global, "b t n c -> (b t) n c")
            audio_global_temb = audio_emb_global[:, 0]
        residual_out = self._run_audio_injector(
            audio_attn_id,
            input_hidden_states,
            attn_audio_emb,
            audio_global_temb=audio_global_temb,
        )
        residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
        hidden_states[:, : self.original_seq_len] = (
            hidden_states[:, : self.original_seq_len] + residual_out
        )
        return hidden_states

    def _after_transformer_block_sequence_shard(
        self,
        audio_attn_id: int,
        hidden_states: torch.Tensor,
        audio_emb: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        if self.original_seq_len % num_frames != 0:
            raise ValueError(
                "Wan S2V audio injection requires original_seq_len to be frame-aligned"
            )

        frame_seq_len = self.original_seq_len // num_frames
        local_start_global = int(self.sequence_shard_start)
        local_end_global = local_start_global + hidden_states.shape[1]
        inject_start_global = max(local_start_global, 0)
        inject_end_global = min(local_end_global, int(self.original_seq_len))
        if inject_start_global >= inject_end_global:
            return hidden_states

        current_global = inject_start_global
        while current_global < inject_end_global:
            frame_idx = current_global // frame_seq_len
            next_frame_global = (frame_idx + 1) * frame_seq_len
            segment_end_global = min(inject_end_global, next_frame_global)
            local_start = current_global - local_start_global
            local_end = segment_end_global - local_start_global

            input_hidden_states = hidden_states[:, local_start:local_end].clone()
            audio_global_temb = None
            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_global_temb = self.audio_emb_global[:, frame_idx, 0]
            residual_out = self._run_audio_injector(
                audio_attn_id,
                input_hidden_states,
                audio_emb[:, frame_idx],
                audio_global_temb=audio_global_temb,
            )
            hidden_states[:, local_start:local_end] = (
                hidden_states[:, local_start:local_end] + residual_out
            )
            current_global = segment_end_global
        return hidden_states

    def set_stream_r1_attention(
        self,
        local_attn_size: int,
        sink_size: int,
        *,
        num_frame_per_block: int | None = None,
        kv_cache: bool = False,
    ) -> None:
        if local_attn_size <= 0:
            raise ValueError("local_attn_size must be positive")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        if sink_size >= local_attn_size:
            raise ValueError("sink_size must be smaller than local_attn_size")
        if num_frame_per_block is not None and num_frame_per_block <= 0:
            raise ValueError("num_frame_per_block must be positive")
        self.stream_r1_local_attn_size = int(local_attn_size)
        self.stream_r1_sink_size = int(sink_size)
        self.stream_r1_num_frame_per_block = (
            int(num_frame_per_block) if num_frame_per_block is not None else None
        )
        self.stream_r1_kv_cache_requested = bool(kv_cache)

    def forward_with_plan_buffers(
        self,
        *,
        timestep_metadata_buffers: WanS2VTimestepStaticMetadataBuffers,
        **kwargs,
    ) -> torch.Tensor:
        """Forward adapter driven by stable timestep metadata buffers.

        This is the Phase-4 bridge into a graph-friendly forward path. The
        current implementation materializes metadata as Python scalars and
        delegates to ``forward``; later patches should move the corresponding
        audio/RoPE/attention/KV code to consume tensor metadata directly.
        """

        return self.forward(
            timestep_metadata_buffers=timestep_metadata_buffers,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor | None = None,
        t: torch.LongTensor | None = None,
        ref_latents: torch.Tensor | list[torch.Tensor] | None = None,
        motion_latents: torch.Tensor | list[torch.Tensor] | None = None,
        cond_states: torch.Tensor | list[torch.Tensor] | None = None,
        audio_input: torch.Tensor | None = None,
        audio_emb: Any | None = None,
        audio_emb_global: torch.Tensor | None = None,
        kv_cache: list | None = None,
        crossattn_cache: list | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
        audio_start_frame: int | None = None,
        motion_frames: list[int] | tuple[int, int] = (73, 19),
        add_last_motion: int = 2,
        drop_motion_frames: bool = False,
        stream_r1_mode: bool = False,
        stream_r1_refresh_only: bool = False,
        stream_r1_audio_emb_pre_sliced: bool = False,
        stream_r1_graph_kv_update: bool = False,
        timestep_metadata_buffers: WanS2VTimestepStaticMetadataBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:
        timestep = timestep if timestep is not None else t
        if timestep is None:
            raise ValueError("WanS2VTransformer3DModel.forward requires timestep/t")
        if timestep_metadata_buffers is not None:
            timestep_metadata = timestep_metadata_buffers.to_forward_kwargs()
            current_start = timestep_metadata["current_start"]
            cache_start = timestep_metadata["cache_start"]
            audio_start_frame = timestep_metadata["audio_start_frame"]
            motion_frames = timestep_metadata["motion_frames"]
            add_last_motion = timestep_metadata["add_last_motion"]
            drop_motion_frames = timestep_metadata["drop_motion_frames"]
            stream_r1_mode = timestep_metadata["stream_r1_mode"]
        validate_wan_s2v_stream_r1_forward_cache(
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            stream_r1_mode=stream_r1_mode,
            num_transformer_blocks=len(self.blocks),
        )
        if stream_r1_mode and self.stream_r1_kv_cache_requested and kv_cache is None:
            raise NotImplementedError(
                "Stream-R1 S2V KV cache was configured, but no KV cache was "
                "provided to the transformer. S2V attention-kernel cache "
                "mutation is not implemented in this phase."
            )
        if stream_r1_refresh_only and (not stream_r1_mode or kv_cache is None):
            raise ValueError(
                "Wan S2V refresh-only forward requires stream_r1_mode=True and kv_cache"
            )
        if cache_start is not None and kv_cache is None:
            logger.debug("Wan S2V cache_start is ignored while KV cache is disabled")
        if audio_input is None and audio_emb is None:
            raise ValueError("Wan S2V requires audio_input or audio_emb")
        if ref_latents is None:
            raise ValueError("Wan S2V requires ref_latents")
        if motion_latents is None:
            raise ValueError("Wan S2V requires motion_latents")

        x_list = _as_list_4d(hidden_states)
        latent_frames = x_list[0].shape[1]
        latent_h, latent_w = x_list[0].shape[-2:]
        frame_seq_length = (latent_h // self.patch_size[1]) * (
            latent_w // self.patch_size[2]
        )
        if audio_start_frame is None:
            audio_start_frame = (
                int(current_start) // frame_seq_length if frame_seq_length else 0
            )
        else:
            audio_start_frame = int(audio_start_frame)
            if audio_start_frame < 0:
                raise ValueError("audio_start_frame must be non-negative")
        ref_list = _as_list_4d(ref_latents)
        motion_list = _as_list_4d(motion_latents)
        if cond_states is None:
            cond_list = [torch.zeros_like(u) for u in x_list]
        else:
            cond_list = _as_list_4d(cond_states)
        if isinstance(encoder_hidden_states, torch.Tensor):
            context_list = [u for u in encoder_hidden_states]
        else:
            context_list = encoder_hidden_states

        add_last_motion = int(self.add_last_motion) * add_last_motion
        if audio_emb is None:
            assert audio_input is not None
            audio_input = torch.cat(
                [audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input],
                dim=-1,
            )
            audio_emb_res = self.casual_audio_encoder(audio_input)
        else:
            audio_emb_res = audio_emb
        use_tensor_audio_slice = (
            stream_r1_mode and timestep_metadata_buffers is not None
        )
        audio_emb_pre_sliced = bool(stream_r1_audio_emb_pre_sliced)
        audio_slice_start = motion_frames[1] + audio_start_frame
        audio_slice_end = audio_slice_start + latent_frames
        if self.enable_adain:
            if isinstance(audio_emb_res, tuple):
                audio_emb_global, audio_emb = audio_emb_res
            elif audio_emb_global is None:
                raise ValueError(
                    "Wan S2V requires audio_emb_global when enable_adain=True "
                    "and audio_emb is provided without a tuple"
                )
            else:
                audio_emb = audio_emb_res
            if audio_emb_pre_sliced:
                if audio_emb_global.shape[1] != latent_frames:
                    raise ValueError(
                        "pre-sliced Wan S2V global audio embeddings must match "
                        f"latent_frames={latent_frames}, "
                        f"got={audio_emb_global.shape[1]}"
                    )
                self.audio_emb_global = audio_emb_global
            elif audio_slice_end > audio_emb_global.shape[1]:
                raise ValueError(
                    "Wan S2V global audio embeddings do not cover the requested "
                    "latent chunk: "
                    f"chunk_start={audio_start_frame}, "
                    f"chunk_frames={latent_frames}, "
                    f"available_audio_frames={audio_emb_global.shape[1] - motion_frames[1]}"
                )
            elif use_tensor_audio_slice:
                self.audio_emb_global = _slice_s2v_audio_embeddings_with_tensor_start(
                    audio_emb_global,
                    audio_start_frame=timestep_metadata_buffers.scalar_tensor(
                        "audio_start_frame"
                    ),
                    motion_prefix_frames=timestep_metadata_buffers.motion_frames[1],
                    latent_frames=latent_frames,
                )
            else:
                self.audio_emb_global = audio_emb_global[
                    :,
                    audio_slice_start:audio_slice_end,
                ].clone()
        else:
            audio_emb = audio_emb_res
        if audio_emb_pre_sliced:
            if audio_emb.shape[1] != latent_frames:
                raise ValueError(
                    "pre-sliced Wan S2V audio embeddings must match "
                    f"latent_frames={latent_frames}, got={audio_emb.shape[1]}"
                )
            self.merged_audio_emb = audio_emb
        elif audio_slice_end > audio_emb.shape[1]:
            raise ValueError(
                "Wan S2V audio embeddings do not cover the requested latent chunk: "
                f"chunk_start={audio_start_frame}, chunk_frames={latent_frames}, "
                f"available_audio_frames={audio_emb.shape[1] - motion_frames[1]}"
            )
        elif use_tensor_audio_slice:
            self.merged_audio_emb = _slice_s2v_audio_embeddings_with_tensor_start(
                audio_emb,
                audio_start_frame=timestep_metadata_buffers.scalar_tensor(
                    "audio_start_frame"
                ),
                motion_prefix_frames=timestep_metadata_buffers.motion_frames[1],
                latent_frames=latent_frames,
            )
        else:
            self.merged_audio_emb = audio_emb[
                :,
                audio_slice_start:audio_slice_end,
                :,
            ]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x_list]
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_list]
        x = [x_ + pose for x_, pose in zip(x, cond)]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        original_grid_sizes = deepcopy(grid_sizes)
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        noisy_grid_sizes_rope = _build_s2v_noisy_rope_grid_sizes(
            grid_sizes,
            stream_r1_mode=stream_r1_mode,
            current_start=current_start,
            frame_seq_length=frame_seq_length,
        )
        use_tensor_current_start_rope = (
            stream_r1_mode and timestep_metadata_buffers is not None
        )

        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_list]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0]).view(1, 3).repeat(batch_size, 1),
                torch.tensor([31, height, width]).view(1, 3).repeat(batch_size, 1),
                torch.tensor([1, height, width]).view(1, 3).repeat(batch_size, 1),
            ]
        ]
        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0].item()
        seq_lens = seq_lens + torch.tensor(
            [r.size(1) for r in ref], dtype=torch.long
        )
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=u.device) for u in x
        ]
        for mask in mask_input:
            mask[:, self.original_seq_len :] = 1

        x_cat = torch.cat(x)
        b, s, n, d = (
            x_cat.size(0),
            x_cat.size(1),
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
        )
        freqs = self.freqs.to(device=x_cat.device)
        x_for_rope = x_cat.detach().view(b, s, n, d)
        if use_tensor_current_start_rope:
            pre_compute_freqs = _rope_precompute_s2v_stream_r1_tensor_current_start(
                x_for_rope,
                grid_sizes,
                ref_grid_sizes,
                freqs,
                current_start=timestep_metadata_buffers.scalar_tensor(
                    "current_start"
                ),
                frame_seq_length=frame_seq_length,
            )
        else:
            pre_compute_freqs = _rope_precompute(
                x_for_rope,
                noisy_grid_sizes_rope + ref_grid_sizes,
                freqs,
            )
        x = [u.unsqueeze(0) for u in x_cat]
        pre_compute_freqs = [u.unsqueeze(0) for u in pre_compute_freqs]
        x, seq_lens, pre_compute_freqs, mask_input = self._inject_motion(
            x,
            seq_lens,
            pre_compute_freqs,
            mask_input,
            motion_list,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
        )
        x = torch.cat(x, dim=0)
        pre_compute_freqs = torch.cat(pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)
        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        if self.zero_timestep:
            timestep = torch.cat(
                [
                    timestep,
                    torch.zeros([1], dtype=timestep.dtype, device=timestep.device),
                ]
            )
        temb = self.condition_embedder.time_embedder(timestep)
        timestep_proj = self.condition_embedder.time_modulation(temb).unflatten(
            1, (6, self.hidden_size)
        )
        if self.zero_timestep:
            temb = temb[:-1]
            zero_e0 = timestep_proj[-1:]
            timestep_proj = torch.cat(
                [
                    timestep_proj[:-1].unsqueeze(2),
                    zero_e0.unsqueeze(2).repeat(timestep_proj.size(0) - 1, 1, 1, 1),
                ],
                dim=2,
            )
            timestep_proj = [timestep_proj, self.original_seq_len]
        else:
            timestep_proj = [timestep_proj.unsqueeze(2).repeat(1, 1, 2, 1), 0]

        context = self.condition_embedder.text_embedder(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context_list
                ]
            )
        )
        context = context.to(x.dtype)

        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and forward_batch.enable_sequence_shard
            and self.sp_size > 1
        )
        self.use_context_parallel = sequence_shard_enabled
        self.sequence_shard_start = 0
        stream_r1_attn_mask = None
        stream_r1_attention_layout = None
        seq_len_before_sp_pad = int(x.shape[1])
        seq_shard_pad = 0
        if stream_r1_mode and self.stream_r1_local_attn_size is not None:
            stream_r1_attention_layout = WanS2VStreamR1AttentionLayout(
                noisy_seq_len=int(self.original_seq_len),
                total_seq_len=int(x.shape[1]),
                frame_seq_length=frame_seq_length,
                num_frame_per_block=(
                    self.stream_r1_num_frame_per_block or latent_frames
                ),
                local_attn_size=self.stream_r1_local_attn_size,
                sink_size=self.stream_r1_sink_size or 0,
                current_start=int(current_start),
            )
            if kv_cache is None:
                stream_r1_attn_mask = (
                    stream_r1_attention_layout.build_no_kv_attention_mask(x.device)
                )
                if sequence_shard_enabled and not self._logged_stream_r1_sp_no_kv_mask:
                    logger.info(
                        "Stream-R1 S2V no-KV SP is using an SP-aware local/sink "
                        "attention mask."
                    )
                    self._logged_stream_r1_sp_no_kv_mask = True
            elif sequence_shard_enabled and not self._logged_stream_r1_sp_kv_mask:
                if wan_s2v_stream_r1_uses_head_sharded_sp_kv_cache():
                    logger.info(
                        "Stream-R1 S2V KV cache SP is using packed varlen "
                        "attention with head-sharded KV cache."
                    )
                else:
                    logger.info(
                        "Stream-R1 S2V KV cache SP is using replicated mixed-KV "
                        "local/sink attention masks."
                    )
                self._logged_stream_r1_sp_kv_mask = True
        if sequence_shard_enabled:
            sp_world_size = get_sp_world_size()
            seq_shard_pad = (-seq_len_before_sp_pad) % sp_world_size
            if seq_shard_pad:
                x_pad = torch.zeros(
                    (x.shape[0], seq_shard_pad, x.shape[2]),
                    dtype=x.dtype,
                    device=x.device,
                )
                freq_pad = torch.ones(
                    (
                        pre_compute_freqs.shape[0],
                        seq_shard_pad,
                        *pre_compute_freqs.shape[2:],
                    ),
                    dtype=pre_compute_freqs.dtype,
                    device=pre_compute_freqs.device,
                )
                if stream_r1_attn_mask is not None:
                    stream_r1_attn_mask = _pad_stream_r1_attention_mask_for_sp(
                        stream_r1_attn_mask,
                        seq_len_before_sp_pad,
                        seq_shard_pad,
                    )
                    sp_key_padding_mask = None
                else:
                    sp_key_padding_mask = torch.ones(
                        (x.shape[0], seq_len_before_sp_pad + seq_shard_pad),
                        dtype=torch.bool,
                        device=x.device,
                    )
                    sp_key_padding_mask[:, seq_len_before_sp_pad:] = False
                x = torch.cat([x, x_pad], dim=1)
                pre_compute_freqs = torch.cat([pre_compute_freqs, freq_pad], dim=1)
            else:
                sp_key_padding_mask = None

            sp_rank = get_sp_group().rank_in_group
            chunks = torch.chunk(x, sp_world_size, dim=1)
            sq_size = [u.shape[1] for u in chunks]
            sq_start_size = sum(sq_size[:sp_rank])
            self.sequence_shard_start = int(sq_start_size)
            x = chunks[sp_rank]
            timestep_proj[1] = timestep_proj[1] - sq_start_size
            pre_compute_freqs = torch.chunk(pre_compute_freqs, sp_world_size, dim=1)[
                sp_rank
            ]
            if sp_key_padding_mask is not None:
                stream_r1_attn_mask = torch.chunk(
                    sp_key_padding_mask, sp_world_size, dim=1
                )[sp_rank]

        for idx, block in enumerate(self.blocks):
            refresh_last_block = stream_r1_refresh_only and idx == len(self.blocks) - 1
            x = block(
                x,
                context,
                timestep_proj,
                pre_compute_freqs,
                attn_mask=stream_r1_attn_mask,
                stream_r1_kv_cache=(
                    kv_cache[idx] if stream_r1_mode and kv_cache is not None else None
                ),
                stream_r1_attention_layout=(
                    stream_r1_attention_layout
                    if stream_r1_mode and kv_cache is not None
                    else None
                ),
                cache_start=cache_start,
                stream_r1_sequence_shard_enabled=(
                    sequence_shard_enabled and stream_r1_mode and kv_cache is not None
                ),
                stream_r1_sp_pad_tokens=seq_shard_pad,
                stream_r1_cache_update_only=refresh_last_block,
                stream_r1_graph_kv_update=stream_r1_graph_kv_update,
                crossattn_kv_cache=(
                    crossattn_cache[idx]
                    if stream_r1_mode and crossattn_cache is not None
                    else None
                ),
            )
            if refresh_last_block:
                return x
            x = self._after_transformer_block(idx, x)

        if stream_r1_refresh_only:
            return x

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_out(x, shift, scale)
        x, _ = self.proj_out(x)
        if sequence_shard_enabled:
            x = sequence_model_parallel_all_gather(x.contiguous(), dim=1)
            if seq_shard_pad:
                x = x[:, :seq_len_before_sp_pad]
        x = x[:, : self.original_seq_len]

        output = self._unpatchify(x, original_grid_sizes)
        return (
            torch.stack(output) if isinstance(hidden_states, torch.Tensor) else output
        )

    def _unpatchify(
        self, x: torch.Tensor, grid_sizes: torch.Tensor
    ) -> list[torch.Tensor]:
        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u.float())
        return out


EntryClass = WanS2VTransformer3DModel
