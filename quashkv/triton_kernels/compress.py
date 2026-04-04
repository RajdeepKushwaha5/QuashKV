"""
Fused compression kernels.

Provides fused_compress_mse and fused_compress_ip which combine
normalize → rotate → quantize into a single pass.

When Triton is available, uses GPU-optimized kernels. Otherwise
falls back to PyTorch reference implementations.

Triton kernel strategy (for when GPU is available):
  - Each program instance processes one row (one vector).
  - Load vector tile → normalize in registers → tl.dot with Pi tile →
    boundary comparisons → write packed indices.
  - Rotation is the bottleneck: O(d²) per vector, fits in shared memory
    for d ≤ 256 (common head_dim values: 64, 96, 128).
"""

from __future__ import annotations

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

def _compress_mse_pytorch(
    x: torch.Tensor,
    Pi: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MSE compression in PyTorch.

    normalize → rotate → boundary-quantize, single pass.

    Parameters
    ----------
    x : (..., d) float
    Pi : (d, d) rotation matrix
    boundaries : (2^b - 1,) sorted boundaries

    Returns
    -------
    indices : (..., d) uint8
    norms : (...) float32
    """
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_rot = (x / norms) @ Pi.T
    # Vectorized boundary quantization (replaces Python loop in codebook)
    idx = torch.bucketize(x_rot, boundaries)
    return idx.to(torch.uint8), norms.squeeze(-1)


def _compress_ip_pytorch(
    x: torch.Tensor,
    Pi: torch.Tensor,
    boundaries: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused inner-product compression in PyTorch.

    normalize → rotate → MSE boundary-quantize → residual → QJL sign hash.

    Parameters
    ----------
    x : (..., d) float
    Pi : (d, d) rotation matrix
    boundaries : (2^{b-1} - 1,) sorted MSE boundaries
    S : (d,) QJL random sign vector
    centroids : (2^{b-1},) MSE centroid values

    Returns
    -------
    mse_indices : (..., d) uint8
    qjl_signs : (..., d) uint8 (0 or 1)
    norms : (...) float32
    """
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_rot = (x / norms) @ Pi.T

    # MSE quantization via vectorized boundary comparisons
    mse_indices = torch.bucketize(x_rot, boundaries).to(torch.uint8)

    # QJL sign hash on the residual (x_rot - dequantized)
    x_rot_hat = centroids[mse_indices.long()]
    residual = x_rot - x_rot_hat
    qjl_signs = ((residual * S) >= 0).to(torch.uint8)

    return mse_indices, qjl_signs, norms.squeeze(-1)


# ======================================================================
# Triton kernel stubs (to be filled when GPU is available)
# ======================================================================

if HAS_TRITON:
    @triton.jit
    def _compress_mse_kernel(
        # Pointers
        x_ptr, Pi_ptr, bounds_ptr, idx_ptr, norms_ptr,
        # Dimensions
        d: tl.constexpr, n_bounds: tl.constexpr,
        # Strides
        stride_x_row, stride_Pi_row,
    ):
        """Triton kernel: fused normalize → rotate → quantize for one vector.

        Algorithm:
          1. Load x[row, :] from global memory
          2. Compute norm = sqrt(sum(x²)), normalize in registers
          3. For each output coordinate j:
             rotated_j = dot(x_normalized, Pi[j, :])
             index_j = binary search in boundaries
          4. Write indices and norm back

        Block mapping: one program per (row × output_tile).
        For d=128, entire rotation fits in one program's registers.
        """
        # TODO: implement when GPU is available
        pass

    @triton.jit
    def _compress_ip_kernel(
        x_ptr, Pi_ptr, bounds_ptr, S_ptr,
        mse_idx_ptr, qjl_ptr, norms_ptr,
        d: tl.constexpr, n_bounds: tl.constexpr,
        stride_x_row, stride_Pi_row,
    ):
        """Triton kernel: fused normalize → rotate → MSE quantize → QJL.

        Same as _compress_mse_kernel but adds the QJL sign computation:
          qjl[j] = 1 if S[j] * rotated[j] > 0 else 0
        """
        # TODO: implement when GPU is available
        pass


# ======================================================================
# Dispatch functions
# ======================================================================

def fused_compress_mse(
    x: torch.Tensor,
    Pi: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MSE compression: normalize → rotate → quantize.

    Automatically dispatches to Triton kernel on CUDA, PyTorch on CPU.

    Parameters
    ----------
    x : (..., d) float tensor
    Pi : (d, d) rotation matrix
    boundaries : (2^b - 1,) sorted decision boundaries

    Returns
    -------
    indices : (..., d) uint8
    norms : (...) float32
    """
    if HAS_TRITON and x.is_cuda:
        # TODO: launch _compress_mse_kernel
        pass
    return _compress_mse_pytorch(x, Pi, boundaries)


def fused_compress_ip(
    x: torch.Tensor,
    Pi: torch.Tensor,
    boundaries: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused IP compression: normalize → rotate → MSE quantize → QJL sign.

    Parameters
    ----------
    x : (..., d) float tensor
    Pi : (d, d) rotation matrix
    boundaries : (2^{b-1} - 1,) sorted MSE boundaries
    S : (d,) QJL random sign vector
    centroids : (2^{b-1},) MSE centroid values

    Returns
    -------
    mse_indices : (..., d) uint8
    qjl_signs : (..., d) uint8
    norms : (...) float32
    """
    if HAS_TRITON and x.is_cuda:
        # TODO: launch _compress_ip_kernel
        pass
    return _compress_ip_pytorch(x, Pi, boundaries, S, centroids)
