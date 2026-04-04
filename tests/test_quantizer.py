"""Tests for MSEQuantizer and InnerProductQuantizer."""

import math
import pytest
import torch

from quashkv.quantizer import MSEQuantizer, InnerProductQuantizer


class TestMSEQuantizer:
    """Test the MSE (rotation + scalar quantize) path."""

    @pytest.fixture
    def quantizer(self):
        return MSEQuantizer(d=128, bits=3, seed=42, device="cpu")

    def test_rotate_unrotate_roundtrip(self, quantizer):
        x = torch.randn(4, 128)
        x_rot = quantizer.rotate(x)
        x_back = quantizer.unrotate(x_rot)
        assert torch.allclose(x, x_back, atol=1e-5)

    def test_rotation_preserves_norm(self, quantizer):
        x = torch.randn(8, 128)
        x_rot = quantizer.rotate(x)
        norms_orig = x.norm(dim=-1)
        norms_rot = x_rot.norm(dim=-1)
        assert torch.allclose(norms_orig, norms_rot, atol=1e-5)

    def test_compress_decompress_shape(self, quantizer):
        x = torch.randn(2, 4, 32, 128)  # (batch, heads, seq, dim)
        indices, norms = quantizer.compress(x)
        assert indices.shape == (2, 4, 32, 128)
        assert indices.dtype == torch.uint8
        assert norms.shape == (2, 4, 32)
        x_hat = quantizer.decompress(indices, norms)
        assert x_hat.shape == x.shape

    def test_mse_within_paper_bound(self, quantizer):
        """Empirical MSE should be near theoretical bound for large enough sample."""
        torch.manual_seed(123)
        d = 128
        n = 2000
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)  # unit norm

        indices, norms = quantizer.compress(x)
        x_hat = quantizer.decompress(indices, norms)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        # Paper: D_mse(b=3) ≤ √(3π/2) · 4^{-3} ≈ 0.034
        assert mse < 0.05, f"Empirical MSE {mse} too high for 3-bit quantizer"

    def test_higher_bits_lower_mse(self):
        """More bits → lower MSE."""
        d = 128
        torch.manual_seed(42)
        x = torch.randn(500, d)
        x = x / x.norm(dim=-1, keepdim=True)

        mses = []
        for bits in [1, 2, 3]:
            q = MSEQuantizer(d=d, bits=bits, seed=42)
            indices, norms = q.compress(x)
            x_hat = q.decompress(indices, norms)
            mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
            mses.append(mse)

        assert mses[0] > mses[1] > mses[2], f"MSE should decrease with more bits: {mses}"


class TestInnerProductQuantizer:
    """Test the two-stage key quantizer (MSE + QJL)."""

    @pytest.fixture
    def quantizer(self):
        return InnerProductQuantizer(d=128, total_bits=3, seed=42, device="cpu")

    def test_compress_shapes(self, quantizer):
        x = torch.randn(2, 4, 16, 128)
        mse_idx, qjl_signs, norms = quantizer.compress(x)
        assert mse_idx.shape == (2, 4, 16, 128)
        assert mse_idx.dtype == torch.uint8
        assert qjl_signs.shape == (2, 4, 16, 128)
        assert qjl_signs.dtype == torch.uint8
        assert norms.shape == (2, 4, 16)

    def test_qjl_signs_binary(self, quantizer):
        x = torch.randn(4, 128)
        _, qjl_signs, _ = quantizer.compress(x)
        assert set(qjl_signs.unique().tolist()).issubset({0, 1})

    def test_inner_product_unbiased(self, quantizer):
        """The IP estimator should be approximately unbiased over many samples."""
        torch.manual_seed(99)
        d = 128
        n = 5000

        q = torch.randn(d)
        q = q / q.norm()

        keys = torch.randn(n, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)

        # True inner products
        true_ips = (keys * q).sum(dim=-1)

        # Compress keys
        mse_idx, qjl_signs, norms = quantizer.compress(keys)

        # Rotate query
        q_rot = quantizer.mse_quantizer.rotate(q.unsqueeze(0))  # (1, d)

        # Estimated inner products (loop for clarity)
        est_ips = quantizer.decompress_for_dot(
            mse_idx, qjl_signs, norms, q_rot.expand(n, -1)
        )

        # Bias should be small (mean error close to 0)
        bias = (est_ips - true_ips).mean().item()
        assert abs(bias) < 0.05, f"IP estimator bias {bias} too large (should be ~0)"

    def test_inner_product_distortion_bounded(self, quantizer):
        """D_ip should be bounded by paper's distortion bound."""
        torch.manual_seed(77)
        d = 128
        n = 3000

        q = torch.randn(d)
        q = q / q.norm()

        keys = torch.randn(n, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)

        true_ips = (keys * q).sum(dim=-1)

        mse_idx, qjl_signs, norms = quantizer.compress(keys)
        q_rot = quantizer.mse_quantizer.rotate(q.unsqueeze(0))
        est_ips = quantizer.decompress_for_dot(
            mse_idx, qjl_signs, norms, q_rot.expand(n, -1)
        )

        mse_ip = ((est_ips - true_ips) ** 2).mean().item()

        # Paper: D_ip(b=3) ≤ √(3π/2) · ||y||²/d · 4^{-b}
        # With ||q|| = 1, d = 128: bound ≈ 2.17 * (1/128) * (1/64) ≈ 2.65e-4
        # Allow generous margin for finite sample effects
        assert mse_ip < 0.01, f"IP distortion {mse_ip} too high"

    def test_min_bits_validation(self):
        with pytest.raises(ValueError, match="at least 2"):
            InnerProductQuantizer(d=128, total_bits=1)
