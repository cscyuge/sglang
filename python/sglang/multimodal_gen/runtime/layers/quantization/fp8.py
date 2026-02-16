# SPDX-License-Identifier: Apache-2.0
"""FP8 block-wise quantization support for multimodal_gen linear layers.

Supports loading FP8 checkpoints with block-wise weight_scale_inv and
performing FP8 GEMM via triton/DeepGEMM backends (dispatched from srt).

Usage:
    1. Quantize model offline with scripts/quantize_flashtalk_fp8.py
    2. The pipeline detects quantization_config in config.json
    3. apply_fp8_quant_to_model() patches block linear layers before loading
    4. Weights load as fp8 + scale, forward uses FP8 GEMM
"""

import logging

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)

# FP8 dtypes that should not be cast to param_dtype during loading
FP8_DTYPES = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    }
)


def _get_fp8_block_linear_fn():
    """Lazy-load the best available FP8 block linear function."""
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            dispatch_w8a8_block_fp8_linear,
        )

        return dispatch_w8a8_block_fp8_linear()
    except Exception:
        logger.info(
            "dispatch_w8a8_block_fp8_linear not available, "
            "using triton FP8 GEMM directly."
        )

    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
        w8a8_block_fp8_matmul_triton,
    )

    def triton_fp8_linear(
        input, weight, block_size, weight_scale, input_scale=None, bias=None
    ):
        assert input_scale is None
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )
        output = w8a8_block_fp8_matmul_triton(
            q_input,
            weight,
            x_scale,
            weight_scale,
            block_size,
            output_dtype=input_2d.dtype,
        )
        if bias is not None:
            output += bias
        return output.to(dtype=input_2d.dtype).view(*output_shape)

    return triton_fp8_linear


class Fp8BlockQuantConfig(QuantizationConfig):
    """Config for FP8 block-wise quantization (e.g., 128x128 blocks)."""

    def __init__(self, weight_block_size: list[int]):
        super().__init__()
        self.weight_block_size = weight_block_size
        self._w8a8_block_fp8_linear = None

    @property
    def w8a8_block_fp8_linear(self):
        if self._w8a8_block_fp8_linear is None:
            self._w8a8_block_fp8_linear = _get_fp8_block_linear_fn()
        return self._w8a8_block_fp8_linear

    def get_name(self):
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls):
        return 89

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        weight_block_size = config.get("weight_block_size", [128, 128])
        return cls(weight_block_size=weight_block_size)

    def get_quant_method(self, layer, prefix=""):
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return Fp8BlockLinearMethod(self)
        return None


class Fp8BlockLinearMethod(QuantizeMethodBase):
    """Linear method for FP8 block-wise quantized weights.

    Expects the layer to have:
        - weight: (N, K) float8_e4m3fn
        - weight_scale_inv: (N/block_n, K/block_k) float32
    """

    def __init__(self, quant_config: Fp8BlockQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError(
            "Fp8BlockLinearMethod.create_weights is not used. "
            "Use apply_fp8_quant_to_model() to patch existing model."
        )

    def apply(self, layer, x, bias=None):
        # Ensure consistent output dtype (bf16) — inputs from different
        # encoders (text, image, audio) may arrive in different dtypes,
        # but attention requires q/k/v to match.
        x = x.to(torch.bfloat16)
        return self.quant_config.w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )

    def process_weights_after_loading(self, layer):
        """Convert weight_scale_inv to float32 (checkpoint may store as bf16)."""
        if hasattr(layer, "weight_scale_inv"):
            layer.weight_scale_inv = nn.Parameter(
                layer.weight_scale_inv.data.float(), requires_grad=False
            )


def apply_fp8_quant_to_model(
    model: nn.Module, quant_config: Fp8BlockQuantConfig
) -> int:
    """Patch block-level linear layers for FP8 quantization.

    Must be called after model creation on meta device, before weight loading.
    For each ColumnParallelLinear/RowParallelLinear inside transformer blocks:
      - Replaces weight parameter with float8_e4m3fn dtype
      - Adds weight_scale_inv parameter (float32)
      - Replaces quant_method with Fp8BlockLinearMethod

    Args:
        model: Model on meta device.
        quant_config: FP8 block quantization config.

    Returns:
        Number of layers patched.
    """
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase

    block_size = quant_config.weight_block_size
    fp8_method = Fp8BlockLinearMethod(quant_config)
    patched = 0

    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        # Only quantize block-level linear layers
        if not name.startswith("blocks."):
            continue

        old_weight = module.weight
        out_features, in_features = old_weight.shape

        # Check divisibility by block size
        if out_features % block_size[0] != 0 or in_features % block_size[1] != 0:
            logger.warning(
                "Skipping FP8 for %s: shape (%d, %d) not divisible by %s",
                name,
                out_features,
                in_features,
                block_size,
            )
            continue

        # 1. Replace weight with fp8 dtype
        new_weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float8_e4m3fn,
                device=old_weight.device,
            ),
            requires_grad=False,
        )
        # Copy dim metadata (but NOT weight_loader — the new nn.Parameter
        # doesn't have load_column_parallel_weight that weight_loader expects;
        # for TP=1/SP, fsdp_load will just use the full tensor directly)
        for attr in ("output_dim", "input_dim"):
            val = getattr(old_weight, attr, None)
            if val is not None:
                setattr(new_weight, attr, val)
        module.weight = new_weight

        # 2. Add weight_scale_inv parameter
        scale_shape = (
            out_features // block_size[0],
            in_features // block_size[1],
        )
        scale = nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.float32, device=old_weight.device),
            requires_grad=False,
        )
        for attr in ("output_dim", "input_dim"):
            val = getattr(old_weight, attr, None)
            if val is not None:
                setattr(scale, attr, val)
        module.register_parameter("weight_scale_inv", scale)

        # 3. Replace quant method
        module.quant_method = fp8_method
        patched += 1

    logger.info("Applied FP8 block quantization to %d linear layers", patched)
    return patched


def process_fp8_weights_after_loading(model: nn.Module) -> None:
    """Post-process FP8 weights after checkpoint loading.

    Converts weight_scale_inv from bf16 (checkpoint dtype) to float32
    (required by FP8 GEMM kernels).
    """
    processed = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_scale_inv") and hasattr(module, "quant_method"):
            if isinstance(module.quant_method, Fp8BlockLinearMethod):
                module.quant_method.process_weights_after_loading(module)
                processed += 1
    if processed > 0:
        logger.info(
            "Post-processed %d FP8 layers (scale_inv -> float32)", processed
        )
