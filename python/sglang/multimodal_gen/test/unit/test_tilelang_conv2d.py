"""Unit tests for TileLang Conv2D implicit GEMM kernel."""

import unittest

import torch
import torch.nn as nn


class TestTileLangConv2D(unittest.TestCase):
    """Test TileLang Conv2D kernel correctness against cuDNN reference."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.multimodal_gen.runtime.kernels.tilelang_conv2d import (
                is_available,
            )

            if not is_available():
                raise unittest.SkipTest("TileLang not available")
        except ImportError:
            raise unittest.SkipTest("TileLang not installed")

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        cls.device = "cuda"
        cls.dtype = torch.bfloat16

    def _run_conv2d_test(self, C_in, C_out, H, W, max_allowed_diff=3.0):
        """Run TileLang conv2d and compare with cuDNN reference."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv2d import (
            tilelang_conv2d_forward,
        )

        torch.manual_seed(42)
        B = 1
        KH = KW = 3

        # Random input and weights
        x = torch.randn(B, C_in, H, W, device=self.device, dtype=self.dtype)
        weight = torch.randn(
            C_out, C_in, KH, KW, device=self.device, dtype=self.dtype
        )
        bias = torch.randn(C_out, device=self.device, dtype=self.dtype)

        # TileLang
        out_tl = tilelang_conv2d_forward(x, weight, bias)

        # cuDNN reference
        conv_ref = nn.Conv2d(C_in, C_out, 3, padding=0, bias=True).to(
            device=self.device, dtype=self.dtype
        )
        conv_ref.weight.data = weight.clone()
        conv_ref.bias.data = bias.clone()
        out_ref = conv_ref(x).contiguous()

        # Compare
        max_diff = (out_tl.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_tl.float() - out_ref.float()).abs().mean().item()

        self.assertEqual(out_tl.shape, out_ref.shape)
        self.assertLess(
            max_diff,
            max_allowed_diff,
            f"max_diff={max_diff:.6e} too large for {C_in}->{C_out} ({H},{W})",
        )
        self.assertLess(mean_diff, 0.01)

    def test_384_192_small(self):
        """up_block_0: 384->192, small spatial (post-upsample from 97x57)."""
        self._run_conv2d_test(384, 192, 194, 114)

    def test_384_192_large(self):
        """up_block_1: 384->192, large spatial (post-upsample from 193x113)."""
        self._run_conv2d_test(384, 192, 386, 226)

    def test_192_96(self):
        """up_block_2: 192->96, large spatial (post-upsample from 385x225)."""
        self._run_conv2d_test(192, 96, 770, 450)

    def test_384_192_medium(self):
        """Medium spatial size for 384->192."""
        self._run_conv2d_test(384, 192, 98, 58)

    def test_192_96_medium(self):
        """Medium spatial size for 192->96."""
        self._run_conv2d_test(192, 96, 386, 226)

    def test_output_channels_last(self):
        """Verify output is channels_last format."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv2d import (
            tilelang_conv2d_forward,
        )

        torch.manual_seed(42)
        x = torch.randn(1, 384, 194, 114, device=self.device, dtype=self.dtype)
        w = torch.randn(192, 384, 3, 3, device=self.device, dtype=self.dtype)
        out = tilelang_conv2d_forward(x, w, None)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(out.shape, (1, 192, 192, 112))

    def test_no_bias(self):
        """Test without bias."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv2d import (
            tilelang_conv2d_forward,
        )

        torch.manual_seed(42)
        x = torch.randn(1, 384, 194, 114, device=self.device, dtype=self.dtype)
        w = torch.randn(192, 384, 3, 3, device=self.device, dtype=self.dtype)
        out = tilelang_conv2d_forward(x, w, None)
        self.assertEqual(out.shape, (1, 192, 192, 112))
        self.assertFalse(torch.isnan(out).any())

    def test_channels_last_input(self):
        """Verify channels_last input avoids extra copy."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv2d import (
            tilelang_conv2d_forward,
        )

        torch.manual_seed(42)
        x = torch.randn(1, 384, 98, 58, device=self.device, dtype=self.dtype)
        x_cl = x.contiguous(memory_format=torch.channels_last)
        w = torch.randn(192, 384, 3, 3, device=self.device, dtype=self.dtype)
        bias = torch.randn(192, device=self.device, dtype=self.dtype)

        out_contiguous = tilelang_conv2d_forward(x, w, bias)
        out_cl = tilelang_conv2d_forward(x_cl, w, bias)

        max_diff = (out_contiguous.float() - out_cl.float()).abs().max().item()
        self.assertLess(max_diff, 1e-6, "channels_last input should produce same result")


if __name__ == "__main__":
    unittest.main()
