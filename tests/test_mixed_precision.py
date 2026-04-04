"""Tests for mixed-precision quantization with outlier handling."""

import pytest
import torch

from quashkv.mixed_precision import (
    MixedPrecisionMSEQuantizer,
    MixedPrecisionIPQuantizer,
)


class TestMixedPrecisionMSEQuantizer:

    def test_init(self):
        q = MixedPrecisionMSEQuantizer(d=128, outlier_bits=3, regular_bits=2, n_outliers=32)
        assert q.d == 128
        assert q.outlier_bits == 3
        assert q.regular_bits == 2
        assert q.n_outliers == 32

    def test_effective_bits(self):
        q = MixedPrecisionMSEQuantizer(d=128, outlier_bits=3, regular_bits=2, n_outliers=32)
        expected = (32 * 3 + 96 * 2) / 128
        assert abs(q.effective_bits - expected) < 1e-6

    def test_effective_bits_all_outlier(self):
        q = MixedPrecisionMSEQuantizer(d=64, outlier_bits=3, regular_bits=2, n_outliers=64)
        assert abs(q.effective_bits - 3.0) < 1e-6

    def test_compress_without_calibration(self):
        """Without calibration, uses regular codebook everywhere."""
        q = MixedPrecisionMSEQuantizer(d=64, outlier_bits=3, regular_bits=2, n_outliers=16)
        x = torch.randn(10, 64)
        indices, norms = q.compress(x)
        assert indices.shape == (10, 64)
        assert norms.shape == (10,)
        assert indices.dtype == torch.uint8

    def test_calibrate_and_compress(self):
        torch.manual_seed(42)
        d = 64
        q = MixedPrecisionMSEQuantizer(d=d, outlier_bits=3, regular_bits=2, n_outliers=16)

        # Calibrate on some data
        calib = torch.randn(200, d)
        q.calibrate(calib)
        assert q.calibrated
        assert q._outlier_mask.sum().item() == 16

        # Compress
        x = torch.randn(10, d)
        indices, norms = q.compress(x)
        assert indices.shape == (10, d)
        assert norms.shape == (10,)

    def test_roundtrip_quality(self):
        """Compressed → decompressed should approximate original."""
        torch.manual_seed(42)
        d = 128
        q = MixedPrecisionMSEQuantizer(d=d, outlier_bits=3, regular_bits=2, n_outliers=32)

        calib = torch.randn(500, d)
        q.calibrate(calib)

        x = torch.randn(50, d)
        indices, norms = q.compress(x)
        x_hat = q.decompress(indices, norms)

        cosine = torch.nn.functional.cosine_similarity(
            x.flatten(), x_hat.flatten(), dim=0
        ).item()
        assert cosine > 0.7, f"Cosine {cosine} too low for mixed 2.5-bit"

    def test_mixed_better_than_uniform_low(self):
        """Mixed 2.5-bit (3+2) should beat uniform 2-bit."""
        torch.manual_seed(42)
        d = 128
        x = torch.randn(200, d)
        calib = torch.randn(500, d)

        # Uniform 2-bit
        from quashkv.quantizer import MSEQuantizer
        q2 = MSEQuantizer(d=d, bits=2, seed=42)
        idx_2, norms_2 = q2.compress(x)
        x_hat_2 = q2.decompress(idx_2, norms_2)
        mse_2 = ((x - x_hat_2) ** 2).mean().item()

        # Mixed 2.5-bit
        qm = MixedPrecisionMSEQuantizer(d=d, outlier_bits=3, regular_bits=2, n_outliers=32)
        qm.calibrate(calib)
        idx_m, norms_m = qm.compress(x)
        x_hat_m = qm.decompress(idx_m, norms_m)
        mse_m = ((x - x_hat_m) ** 2).mean().item()

        assert mse_m < mse_2, f"Mixed MSE {mse_m:.4f} should be < uniform 2-bit MSE {mse_2:.4f}"

    def test_set_outlier_channels(self):
        d = 64
        q = MixedPrecisionMSEQuantizer(d=d, outlier_bits=3, regular_bits=2, n_outliers=8)
        q.set_outlier_channels(torch.tensor([0, 1, 2, 3, 60, 61, 62, 63]))
        assert q.calibrated
        assert q._outlier_mask.sum().item() == 8
        assert q._outlier_mask[0] is True or q._outlier_mask[0].item() is True
        assert q._outlier_mask[63] is True or q._outlier_mask[63].item() is True

    def test_4d_input(self):
        """Should work with (B, H, S, D) tensors."""
        torch.manual_seed(42)
        d = 64
        q = MixedPrecisionMSEQuantizer(d=d, outlier_bits=3, regular_bits=2, n_outliers=16)
        q.calibrate(torch.randn(100, d))

        x = torch.randn(1, 2, 16, d)
        indices, norms = q.compress(x)
        assert indices.shape == (1, 2, 16, d)
        assert norms.shape == (1, 2, 16)

        x_hat = q.decompress(indices, norms)
        assert x_hat.shape == (1, 2, 16, d)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="n_outliers"):
            MixedPrecisionMSEQuantizer(d=64, n_outliers=100)

        with pytest.raises(ValueError, match="outlier_bits"):
            MixedPrecisionMSEQuantizer(d=64, outlier_bits=1, regular_bits=2)


class TestMixedPrecisionIPQuantizer:

    def test_init(self):
        q = MixedPrecisionIPQuantizer(d=128, outlier_bits=3, regular_bits=2, n_outliers=32)
        assert q.d == 128
        # MSE part uses bits-1
        assert q.mse_quantizer.outlier_bits == 2
        assert q.mse_quantizer.regular_bits == 1

    def test_effective_bits(self):
        q = MixedPrecisionIPQuantizer(d=128, outlier_bits=3, regular_bits=2, n_outliers=32)
        # MSE: (32*2 + 96*1)/128 = 1.25, plus 1 for QJL = 2.25
        expected = (32 * 2 + 96 * 1) / 128 + 1.0
        assert abs(q.effective_bits - expected) < 1e-6

    def test_compress_shapes(self):
        q = MixedPrecisionIPQuantizer(d=64, outlier_bits=3, regular_bits=2, n_outliers=16)
        q.calibrate(torch.randn(100, 64))

        x = torch.randn(10, 64)
        mse_idx, qjl, norms = q.compress(x)
        assert mse_idx.shape == (10, 64)
        assert qjl.shape == (10, 64)
        assert norms.shape == (10,)

    def test_compress_4d(self):
        q = MixedPrecisionIPQuantizer(d=64, outlier_bits=4, regular_bits=3, n_outliers=16)
        q.calibrate(torch.randn(100, 64))

        x = torch.randn(1, 2, 16, 64)
        mse_idx, qjl, norms = q.compress(x)
        assert mse_idx.shape == (1, 2, 16, 64)
        assert qjl.shape == (1, 2, 16, 64)
        assert norms.shape == (1, 2, 16)

    def test_qjl_binary(self):
        """QJL signs should be 0 or 1."""
        q = MixedPrecisionIPQuantizer(d=64, outlier_bits=3, regular_bits=2, n_outliers=16)
        q.calibrate(torch.randn(100, 64))

        x = torch.randn(50, 64)
        _, qjl, _ = q.compress(x)
        assert ((qjl == 0) | (qjl == 1)).all()

    def test_min_bits_required(self):
        with pytest.raises(ValueError, match="at least 2"):
            MixedPrecisionIPQuantizer(d=64, outlier_bits=1, regular_bits=1)

    def test_unbiased_ip(self):
        """Inner product estimation should be approximately unbiased."""
        torch.manual_seed(42)
        d = 128
        n_trials = 500

        q = MixedPrecisionIPQuantizer(d=d, outlier_bits=4, regular_bits=3, n_outliers=32)
        q.calibrate(torch.randn(500, d))

        x = torch.randn(n_trials, d)
        y = torch.randn(1, d)

        mse_idx, qjl, norms = q.compress(x)

        # Estimate IP: norm * (mse_dot + qjl_correction)
        x_rot = q.mse_quantizer.rotate(x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        y_rot = y @ q.mse_quantizer.Pi.T

        # MSE part
        x_rot_hat = q._dequantize_mixed(mse_idx)
        mse_dot = (y_rot * x_rot_hat).sum(dim=-1)

        # QJL part
        sign_vals = qjl.float() * 2.0 - 1.0
        correction = q.qjl_scale * (y_rot.abs() * q.S * sign_vals).sum(dim=-1)

        estimated_ip = norms * (mse_dot + correction)
        true_ip = (y * x).sum(dim=-1)

        # Check approximate unbiasedness: mean of estimated should be close to mean of true
        bias = (estimated_ip - true_ip).mean().item()
        assert abs(bias) < 0.5, f"Bias {bias:.4f} too large (should be near 0)"
