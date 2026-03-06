"""Unit tests for TileLang Conv3D implicit GEMM kernel."""

import unittest

import torch
import torch.nn as nn


class TestTileLangConv3D(unittest.TestCase):
    """Test TileLang Conv3D kernel correctness against cuDNN reference."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.multimodal_gen.runtime.kernels.tilelang_conv3d import (
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

    def _run_conv3d_test(self, C_in, C_out, D, H, W, max_allowed_diff=3.0):
        """Run TileLang conv3d and compare with cuDNN reference."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv3d import (
            prepare_weight,
            tilelang_conv3d_forward,
        )

        torch.manual_seed(42)
        B = 1
        KD = KH = KW = 3

        # Random input and weights
        x = torch.randn(B, C_in, D, H, W, device=self.device, dtype=self.dtype)
        weight = torch.randn(
            C_out, C_in, KD, KH, KW, device=self.device, dtype=self.dtype
        )
        bias = torch.randn(C_out, device=self.device, dtype=self.dtype)

        # TileLang
        out_tl = tilelang_conv3d_forward(x, weight, bias)

        # cuDNN reference
        conv_ref = nn.Conv3d(C_in, C_out, 3, padding=0, bias=True).to(
            device=self.device, dtype=self.dtype
        )
        conv_ref.weight.data = weight.clone()
        conv_ref.bias.data = bias.clone()
        conv_ref.weight.data = conv_ref.weight.data.contiguous(
            memory_format=torch.channels_last_3d
        )
        x_cl3d = x.contiguous(memory_format=torch.channels_last_3d)
        out_ref = conv_ref(x_cl3d).contiguous()

        # Compare
        max_diff = (out_tl.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_tl.float() - out_ref.float()).abs().mean().item()

        self.assertEqual(out_tl.shape, out_ref.shape)
        self.assertLess(
            max_diff,
            max_allowed_diff,
            f"max_diff={max_diff:.6e} too large for {C_in}->{C_out} ({D},{H},{W})",
        )
        self.assertLess(mean_diff, 0.01)

    def test_conv_in_16_384(self):
        """conv_in: 16->384, small spatial."""
        self._run_conv3d_test(16, 384, 3, 98, 58)

    def test_mid_384_384_small(self):
        """mid block: 384->384, small spatial."""
        self._run_conv3d_test(384, 384, 3, 98, 58)

    def test_mid_384_384_large(self):
        """mid block: 384->384, larger spatial."""
        self._run_conv3d_test(384, 384, 3, 194, 114)

    def test_up2_192_192(self):
        """up_block 2: 192->192."""
        self._run_conv3d_test(192, 192, 3, 386, 226)

    def test_up3_96_96(self):
        """up_block 3: 96->96."""
        self._run_conv3d_test(96, 96, 3, 770, 450)

    def test_output_contiguous(self):
        """Verify output is contiguous NCDHW."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv3d import (
            tilelang_conv3d_forward,
        )

        torch.manual_seed(42)
        x = torch.randn(1, 16, 3, 98, 58, device=self.device, dtype=self.dtype)
        w = torch.randn(384, 16, 3, 3, 3, device=self.device, dtype=self.dtype)
        out = tilelang_conv3d_forward(x, w, None)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.shape, (1, 384, 1, 96, 56))

    def test_no_bias(self):
        """Test without bias."""
        from sglang.multimodal_gen.runtime.kernels.tilelang_conv3d import (
            tilelang_conv3d_forward,
        )

        torch.manual_seed(42)
        x = torch.randn(1, 16, 3, 98, 58, device=self.device, dtype=self.dtype)
        w = torch.randn(384, 16, 3, 3, 3, device=self.device, dtype=self.dtype)
        out = tilelang_conv3d_forward(x, w, None)
        self.assertEqual(out.shape, (1, 384, 1, 96, 56))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
