import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod


class TestFp8WeightScaleLayout(unittest.TestCase):
    def test_flashinfer_cutlass_scale_is_prepacked(self):
        method = object.__new__(Fp8LinearMethod)
        method.block_quant = True
        method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
        method._uses_flashinfer_cutlass_block_fp8 = lambda: True

        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(torch.empty(512, 1024), requires_grad=False),
        )
        source = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
        layer.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(source, requires_grad=False),
        )

        method._process_block_fp8_linear_weight_scale(layer)

        self.assertEqual(tuple(layer.weight_scale_inv_cutlass.shape), (8, 4))
        self.assertTrue(layer.weight_scale_inv_cutlass.is_contiguous())
        self.assertTrue(
            torch.equal(layer.weight_scale_inv_cutlass, source.T.contiguous())
        )
        self.assertTrue(torch.equal(layer.weight_scale_inv, source))


if __name__ == "__main__":
    unittest.main()
