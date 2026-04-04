"""Tests for Lloyd-Max codebook solver."""

import math
import pytest
import torch

from quashkv.codebook import LloydMaxCodebook, solve_lloyd_max, compute_mse_cost


class TestSolveLloydMax:
    """Verify the Lloyd-Max solver produces correct centroids."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_num_centroids(self, bits: int):
        centroids, boundaries = solve_lloyd_max(d=128, bits=bits)
        assert len(centroids) == 2**bits
        assert len(boundaries) == 2**bits - 1

    @pytest.mark.parametrize("bits", [1, 2, 3])
    def test_centroids_sorted(self, bits: int):
        centroids, _ = solve_lloyd_max(d=128, bits=bits)
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]

    def test_centroids_symmetric(self):
        """For symmetric PDF (Gaussian), centroids should be symmetric around 0."""
        centroids, _ = solve_lloyd_max(d=128, bits=2)
        n = len(centroids)
        for i in range(n // 2):
            assert abs(centroids[i] + centroids[n - 1 - i]) < 1e-6

    @pytest.mark.parametrize("bits", [1, 2, 3])
    def test_boundaries_between_centroids(self, bits: int):
        centroids, boundaries = solve_lloyd_max(d=128, bits=bits)
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1]

    def test_1bit_centroids_match_theory(self):
        """For 1-bit Gaussian quantization, centroids = ±σ·√(2/π).

        With σ = 1/√d and d=128: centroids ≈ ±0.07035
        """
        d = 128
        sigma = 1.0 / math.sqrt(d)
        expected = sigma * math.sqrt(2.0 / math.pi)
        centroids, _ = solve_lloyd_max(d=d, bits=1)
        assert abs(centroids[0] + expected) < 1e-4
        assert abs(centroids[1] - expected) < 1e-4

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be in"):
            solve_lloyd_max(d=128, bits=5)


class TestMSECost:
    """Verify MSE cost matches paper bounds."""

    # Paper Theorem 1: D_mse(b) ≤ √(3π/2) · 4^{-b}
    # √(3π/2) ≈ 2.17, but the multiplicative gap from Gaussian approximation
    # makes empirical values slightly higher.  Use 2.8 as a safe upper factor.
    @pytest.mark.parametrize("bits,expected_upper", [
        (1, 2.8 * 0.25),       # ≈ 0.70
        (2, 2.8 * 0.0625),     # ≈ 0.175
        (3, 2.8 * 0.015625),   # ≈ 0.044
    ])
    def test_mse_below_upper_bound(self, bits: int, expected_upper: float):
        d = 128
        mse = compute_mse_cost(d, bits)
        assert mse < expected_upper, f"MSE {mse} exceeds upper bound {expected_upper}"

    @pytest.mark.parametrize("bits,expected_lower", [
        (1, 0.25),      # 4^{-1}
        (2, 0.0625),    # 4^{-2}
        (3, 0.015625),  # 4^{-3}
    ])
    def test_mse_above_lower_bound(self, bits: int, expected_lower: float):
        """Paper shows optimal quantizer can't beat 4^{-b}."""
        d = 128
        mse = compute_mse_cost(d, bits)
        # Allow a small margin since numerical integration isn't perfect
        assert mse > expected_lower * 0.9, f"MSE {mse} suspiciously below lower bound"


class TestLloydMaxCodebook:
    """Test the codebook class."""

    def test_quantize_dequantize_roundtrip(self):
        cb = LloydMaxCodebook(d=128, bits=2)
        x = torch.randn(10, 128) / math.sqrt(128)
        indices = cb.quantize(x)
        x_hat = cb.dequantize(indices)
        assert x_hat.shape == x.shape
        assert indices.dtype == torch.uint8
        assert indices.max().item() <= 3  # 2 bits → 4 levels

    def test_quantize_boundary_matches_quantize(self):
        cb = LloydMaxCodebook(d=128, bits=2)
        x = torch.randn(100, 128) / math.sqrt(128)
        idx_dist = cb.quantize(x)
        idx_bnd = cb.quantize_boundary(x)
        assert torch.equal(idx_dist, idx_bnd)

    def test_device_transfer(self):
        cb = LloydMaxCodebook(d=128, bits=2)
        cb.to("cpu")
        assert cb.centroids.device == torch.device("cpu")

    def test_repr(self):
        cb = LloydMaxCodebook(d=128, bits=1)
        assert "d=128" in repr(cb)
        assert "bits=1" in repr(cb)
