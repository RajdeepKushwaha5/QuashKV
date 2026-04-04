"""
Mixed-precision quantization with outlier channel handling.

Implements the paper's outlier channel splitting strategy (Section 4.3):
  - Detect which rotated channels have highest variance (outliers)
  - Allocate more bits to outlier channels, fewer to regular channels
  - Example: 2.5-bit = 32 outlier channels at 3 bits + 96 at 2 bits

This can improve quality at the same average bit rate, or maintain
quality with fewer average bits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .codebook import LloydMaxCodebook
from .quantizer import _random_orthogonal, _random_signs
from .constants import DEFAULT_SEED, QJL_COEFF


class MixedPrecisionMSEQuantizer:
    """MSE quantizer with per-channel adaptive bit allocation.

    Splits channels into outlier and regular groups, using a separate
    codebook for each.

    Parameters
    ----------
    d : int
        Vector dimension.
    outlier_bits : int
        Bits for outlier channels.
    regular_bits : int
        Bits for regular channels.
    n_outliers : int
        Number of channels to treat as outliers.
    seed : int
        Random seed for rotation matrix.
    device : str or torch.device
        Device for tensors.
    """

    def __init__(
        self,
        d: int,
        outlier_bits: int = 3,
        regular_bits: int = 2,
        n_outliers: int = 32,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
    ):
        if n_outliers < 0 or n_outliers > d:
            raise ValueError(f"n_outliers must be in [0, {d}], got {n_outliers}")
        if outlier_bits < regular_bits:
            raise ValueError("outlier_bits should be >= regular_bits")

        self.d = d
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.n_outliers = n_outliers
        self.seed = seed
        self.device = torch.device(device)

        # Rotation matrix (shared)
        self.Pi = _random_orthogonal(d, seed, self.device)

        # Two codebooks at different bit-widths
        self.outlier_codebook = LloydMaxCodebook(d, outlier_bits)
        self.outlier_codebook.to(self.device)
        self.regular_codebook = LloydMaxCodebook(d, regular_bits)
        self.regular_codebook.to(self.device)

        # Outlier channel mask (detected via calibrate() or set manually)
        self._outlier_indices: Optional[torch.Tensor] = None
        self._outlier_mask: Optional[torch.Tensor] = None

    @property
    def effective_bits(self) -> float:
        """Average bits per coordinate."""
        n_o = self.n_outliers
        n_r = self.d - n_o
        return (n_o * self.outlier_bits + n_r * self.regular_bits) / self.d

    @property
    def calibrated(self) -> bool:
        return self._outlier_mask is not None

    def calibrate(self, data: torch.Tensor) -> None:
        """Detect outlier channels from calibration data.

        Rotates the data, measures per-channel variance in rotated space,
        and marks the top-k highest variance channels as outliers.

        Parameters
        ----------
        data : (N, d) float — representative sample of vectors
        """
        if data.dim() != 2 or data.shape[1] != self.d:
            raise ValueError(f"Expected (N, {self.d}) calibration data")

        data = data.to(self.device)
        norms = data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_rot = (data / norms) @ self.Pi.T

        per_channel_var = x_rot.var(dim=0)  # (d,)
        _, top_k = per_channel_var.topk(self.n_outliers)

        self._outlier_indices = top_k.sort().values
        mask = torch.zeros(self.d, dtype=torch.bool, device=self.device)
        mask[self._outlier_indices] = True
        self._outlier_mask = mask

    def set_outlier_channels(self, indices: torch.Tensor) -> None:
        """Manually set which channels are outliers."""
        self._outlier_indices = indices.sort().values.to(self.device)
        mask = torch.zeros(self.d, dtype=torch.bool, device=self.device)
        mask[self._outlier_indices] = True
        self._outlier_mask = mask

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, x_rot: torch.Tensor) -> torch.Tensor:
        return x_rot @ self.Pi

    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress with mixed precision.

        Parameters
        ----------
        x : (..., d) float

        Returns
        -------
        indices : (..., d) uint8  (indices are from *different* codebooks by channel)
        norms : (...) float32
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_rot = (x / norms) @ self.Pi.T

        if self._outlier_mask is None:
            # No calibration → regular bits everywhere
            indices = self.regular_codebook.quantize_boundary(x_rot)
        else:
            indices = torch.empty_like(x_rot, dtype=torch.uint8)
            mask = self._outlier_mask
            indices[..., mask] = self.outlier_codebook.quantize_boundary(
                x_rot[..., mask]
            )
            indices[..., ~mask] = self.regular_codebook.quantize_boundary(
                x_rot[..., ~mask]
            )

        return indices, norms.squeeze(-1)

    def decompress(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Decompress mixed-precision indices.

        Parameters
        ----------
        indices : (..., d) uint8
        norms : (...) float32

        Returns
        -------
        x_hat : (..., d) float32
        """
        if self._outlier_mask is None:
            x_rot_hat = self.regular_codebook.dequantize(indices)
        else:
            x_rot_hat = torch.empty(
                *indices.shape, dtype=torch.float32, device=indices.device,
            )
            mask = self._outlier_mask
            x_rot_hat[..., mask] = self.outlier_codebook.dequantize(
                indices[..., mask]
            )
            x_rot_hat[..., ~mask] = self.regular_codebook.dequantize(
                indices[..., ~mask]
            )

        x_unit_hat = x_rot_hat @ self.Pi
        return x_unit_hat * norms.unsqueeze(-1)

    def to(self, device: str | torch.device) -> "MixedPrecisionMSEQuantizer":
        self.device = torch.device(device)
        self.Pi = self.Pi.to(self.device)
        self.outlier_codebook.to(self.device)
        self.regular_codebook.to(self.device)
        if self._outlier_mask is not None:
            self._outlier_mask = self._outlier_mask.to(self.device)
            self._outlier_indices = self._outlier_indices.to(self.device)
        return self


class MixedPrecisionIPQuantizer:
    """Two-stage IP quantizer with per-channel mixed precision.

    Uses MixedPrecisionMSEQuantizer at (bits - 1) for MSE stage,
    plus 1-bit QJL on the residual for all channels.

    Parameters
    ----------
    d : int
        Vector dimension.
    outlier_bits : int
        Total bits for outlier channels (MSE uses outlier_bits - 1).
    regular_bits : int
        Total bits for regular channels (MSE uses regular_bits - 1).
    n_outliers : int
        Number of outlier channels.
    seed : int
        Random seed.
    device : str or torch.device
    """

    def __init__(
        self,
        d: int,
        outlier_bits: int = 3,
        regular_bits: int = 2,
        n_outliers: int = 32,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
    ):
        if regular_bits < 2:
            raise ValueError("IP quantizer needs at least 2 total bits")

        self.d = d
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.n_outliers = n_outliers
        self.seed = seed
        self.device = torch.device(device)

        # MSE stage: mixed precision at (bits - 1)
        self.mse_quantizer = MixedPrecisionMSEQuantizer(
            d=d,
            outlier_bits=outlier_bits - 1,
            regular_bits=regular_bits - 1,
            n_outliers=n_outliers,
            seed=seed,
            device=device,
        )

        # QJL stage (1-bit for all channels)
        self.S = _random_signs(d, seed, self.device)
        self.qjl_scale = QJL_COEFF / d

    @property
    def effective_bits(self) -> float:
        return self.mse_quantizer.effective_bits + 1.0  # +1 for QJL

    @property
    def calibrated(self) -> bool:
        return self.mse_quantizer.calibrated

    def calibrate(self, data: torch.Tensor) -> None:
        """Calibrate outlier channels from representative data."""
        self.mse_quantizer.calibrate(data)

    def set_outlier_channels(self, indices: torch.Tensor) -> None:
        self.mse_quantizer.set_outlier_channels(indices)

    def compress(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress with mixed-precision MSE + QJL.

        Returns
        -------
        mse_indices : (..., d) uint8
        qjl_signs : (..., d) uint8
        norms : (...) float32
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # MSE stage
        x_rot = self.mse_quantizer.rotate(x_unit)
        mse_indices = self._quantize_mixed(x_rot)
        x_rot_hat = self._dequantize_mixed(mse_indices)

        # QJL on residual
        residual = x_rot - x_rot_hat
        qjl_signs = ((residual * self.S) >= 0).to(torch.uint8)

        return mse_indices, qjl_signs, norms.squeeze(-1)

    def _quantize_mixed(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Apply mixed-precision boundary quantization."""
        mq = self.mse_quantizer
        if mq._outlier_mask is None:
            return mq.regular_codebook.quantize_boundary(x_rot)
        indices = torch.empty_like(x_rot, dtype=torch.uint8)
        mask = mq._outlier_mask
        indices[..., mask] = mq.outlier_codebook.quantize_boundary(x_rot[..., mask])
        indices[..., ~mask] = mq.regular_codebook.quantize_boundary(x_rot[..., ~mask])
        return indices

    def _dequantize_mixed(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize mixed-precision indices (in rotated space)."""
        mq = self.mse_quantizer
        if mq._outlier_mask is None:
            return mq.regular_codebook.dequantize(indices)
        x_rot_hat = torch.empty(
            *indices.shape, dtype=torch.float32, device=indices.device,
        )
        mask = mq._outlier_mask
        x_rot_hat[..., mask] = mq.outlier_codebook.dequantize(indices[..., mask])
        x_rot_hat[..., ~mask] = mq.regular_codebook.dequantize(indices[..., ~mask])
        return x_rot_hat

    def to(self, device: str | torch.device) -> "MixedPrecisionIPQuantizer":
        self.device = torch.device(device)
        self.mse_quantizer.to(device)
        self.S = self.S.to(device)
        return self
