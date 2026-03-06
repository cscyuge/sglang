"""Unit tests for Triton fused L2Norm + SiLU kernel."""

import unittest

import torch
import torch.nn.functional as F


class TestFusedNormSilu(unittest.TestCase):
    """Test fused L2Norm+SiLU kernel correctness against PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        try:
            from sglang.multimodal_gen.runtime.kernels.fused_norm_silu import (
                is_available,
            )

            if not is_available():
                raise unittest.SkipTest("Triton not available")
        except ImportError:
            raise unittest.SkipTest("fused_norm_silu not importable")

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        cls.device = "cuda"
        cls.dtype = torch.bfloat16

    def _make_norm(self, C):
        from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import (
            WanRMS_norm,
        )

        return WanRMS_norm(C, channel_first=True, images=False).to(
            self.device, self.dtype
        )

    def _run_test(self, C, T, H, W, apply_silu=True, max_diff=0.1):
        from sglang.multimodal_gen.runtime.kernels.fused_norm_silu import (
            fused_rms_norm_silu,
        )

        torch.manual_seed(42)
        B = 1
        norm = self._make_norm(C)
        x = torch.randn(B, C, T, H, W, device=self.device, dtype=self.dtype)

        # Fused
        out = fused_rms_norm_silu(x, norm, apply_silu=apply_silu)

        # Reference
        ref = norm(x)
        if apply_silu:
            ref = F.silu(ref)

        diff = (out.float() - ref.float()).abs()
        self.assertEqual(out.shape, ref.shape)
        self.assertLess(
            diff.max().item(),
            max_diff,
            f"max_diff={diff.max().item():.6e} for C={C} silu={apply_silu}",
        )
        self.assertLess(diff.mean().item(), 0.01)

    def test_c384_silu(self):
        """C=384 (mid block) with SiLU."""
        self._run_test(384, 3, 98, 58)

    def test_c192_silu(self):
        """C=192 (up_block 2) with SiLU."""
        self._run_test(192, 3, 194, 114)

    def test_c96_silu(self):
        """C=96 (up_block 3) with SiLU."""
        self._run_test(96, 3, 386, 226)

    def test_c384_no_silu(self):
        """C=384 norm only (no SiLU)."""
        self._run_test(384, 3, 98, 58, apply_silu=False)

    def test_cl3d_input(self):
        """CL3D (channels_last_3d) input format."""
        from sglang.multimodal_gen.runtime.kernels.fused_norm_silu import (
            fused_rms_norm_silu,
        )

        torch.manual_seed(42)
        C = 384
        norm = self._make_norm(C)
        x = torch.randn(1, C, 3, 98, 58, device=self.device, dtype=self.dtype)
        x_cl3d = x.contiguous(memory_format=torch.channels_last_3d)

        out = fused_rms_norm_silu(x_cl3d, norm, apply_silu=True)
        ref = F.silu(norm(x))

        self.assertEqual(out.shape, ref.shape)
        diff = (out.float() - ref.float()).abs().max().item()
        self.assertLess(diff, 0.1)

    def test_output_format_preserved(self):
        """Output should match input memory format."""
        from sglang.multimodal_gen.runtime.kernels.fused_norm_silu import (
            fused_rms_norm_silu,
        )

        torch.manual_seed(42)
        C = 384
        norm = self._make_norm(C)
        x = torch.randn(1, C, 3, 98, 58, device=self.device, dtype=self.dtype)

        out_ncthw = fused_rms_norm_silu(x, norm)
        self.assertEqual(out_ncthw.shape, x.shape)
        self.assertFalse(torch.isnan(out_ncthw).any())


if __name__ == "__main__":
    unittest.main()
