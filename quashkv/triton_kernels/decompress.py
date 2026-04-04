"""
Fused decompression kernels.

Provides fused_decompress_mse and fused_decompress_ip which combine
dequantize → unrotate → rescale into a single pass.

Triton kernel strategy:
  - Each program instance processes one vector.
  - Load indices → centroid gather → tl.dot with Pi (un-rotate) →
    multiply by norm → write output.
  - Centroid table (≤16 entries for 4-bit) fits in shared memory.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ======================================================================
# PyTorch reference implementations
# ======================================================================

def _decompress_mse_pytorch(
    indices: torch.Tensor,
    norms: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Fused MSE decompression in PyTorch.

    dequantize → unrotate → rescale.

    Parameters
    ----------
    indices : (..., d) uint8
    norms : (...) float32
    Pi : (d, d) rotation matrix
    centroids : (2^b,) centroid values

    Returns
    -------
    x_hat : (..., d) float32
    """
    x_rot_hat = centroids[indices.long()]
    x_unit_hat = x_rot_hat @ Pi  # Pi.T.T = Pi for un-rotation
    return x_unit_hat * norms.unsqueeze(-1)


def _decompress_ip_pytorch(
    mse_indices: torch.Tensor,
    qjl_signs: torch.Tensor,
    norms: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    S: torch.Tensor,
    qjl_scale: float,
) -> torch.Tensor:
    """Fused IP decompression in PyTorch.

    Reconstructs x̃ = x̃_mse + x̃_qjl:
      x̃_mse = norm * Pi^T @ centroid[idx]
      x̃_qjl = norm * qjl_scale * Pi^T @ (S * (2*sign - 1))

    Parameters
    ----------
    mse_indices : (..., d) uint8
    qjl_signs : (..., d) uint8 (0 or 1)
    norms : (...) float32
    Pi : (d, d) rotation matrix
    centroids : (2^{b-1},) MSE centroid values
    S : (d,) QJL random sign vector
    qjl_scale : float, sqrt(pi/2) / d

    Returns
    -------
    x_hat : (..., d) float32
    """
    # MSE component
    mse_hat = centroids[mse_indices.long()]  # (..., d) in rotated space
    mse_unrot = mse_hat @ Pi

    # QJL component
    sign_vals = qjl_signs.float() * 2.0 - 1.0  # (..., d) → ±1
    qjl_rot = qjl_scale * (S * sign_vals)  # (..., d) in rotated space
    qjl_unrot = qjl_rot @ Pi

    return (mse_unrot + qjl_unrot) * norms.unsqueeze(-1)


# ======================================================================
# Triton kernel stubs
# ======================================================================

if HAS_TRITON:
    @triton.jit
    def _decompress_mse_kernel(
        idx_ptr, norms_ptr, Pi_ptr, centroids_ptr, out_ptr,
        d: tl.constexpr, n_levels: tl.constexpr,
        stride_idx_row, stride_Pi_row, stride_out_row,
    ):
        """Triton kernel: fused centroid-gather → un-rotate → rescale.

        Algorithm:
          1. Load centroids table into shared memory (tiny: ≤16 floats)
          2. Load indices[row, :], gather centroids → rotated_hat
          3. Un-rotate: output[j] = dot(rotated_hat, Pi[:, j])
          4. Multiply by norms[row]

        The centroid gather is essentially a tiny LUT in shared memory.
        """
        # TODO: implement when GPU is available
        pass


# ======================================================================
# Dispatch functions
# ======================================================================

def fused_decompress_mse(
    indices: torch.Tensor,
    norms: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Fused MSE decompression: dequantize → unrotate → rescale.

    Parameters
    ----------
    indices : (..., d) uint8
    norms : (...) float32
    Pi : (d, d) rotation matrix
    centroids : (2^b,) centroid values

    Returns
    -------
    x_hat : (..., d) float32
    """
    if HAS_TRITON and indices.is_cuda:
        # TODO: launch _decompress_mse_kernel
        pass
    return _decompress_mse_pytorch(indices, norms, Pi, centroids)


def fused_decompress_ip(
    mse_indices: torch.Tensor,
    qjl_signs: torch.Tensor,
    norms: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    S: torch.Tensor,
    qjl_scale: float,
) -> torch.Tensor:
    """Fused IP decompression: dequantize MSE + QJL → unrotate → rescale.

    Parameters
    ----------
    mse_indices : (..., d) uint8
    qjl_signs : (..., d) uint8
    norms : (...) float32
    Pi : (d, d) rotation matrix
    centroids : (2^{b-1},) MSE centroid values
    S : (d,) QJL sign vector
    qjl_scale : float

    Returns
    -------
    x_hat : (..., d) float32
    """
    if HAS_TRITON and mse_indices.is_cuda:
        # TODO: launch Triton kernel
        pass
    return _decompress_ip_pytorch(
        mse_indices, qjl_signs, norms, Pi, centroids, S, qjl_scale,
    )
