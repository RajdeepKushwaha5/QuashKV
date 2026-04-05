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
# Triton kernels
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

        Grid: (N,) where N = total number of vectors (product of batch dims).
        Each program processes one d-dimensional vector.
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, d)

        # 1. Load x[row, :]
        x_ptrs = x_ptr + row * stride_x_row + col_offsets
        x = tl.load(x_ptrs)

        # 2. Compute norm and normalize
        norm_sq = tl.sum(x * x, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        x_unit = x / norm

        # 3. Rotate: x_rot[j] = dot(x_unit, Pi[j, :]) for all j
        #    We iterate over output coordinates, doing dot products
        x_rot = tl.zeros([d], dtype=tl.float32)
        for j in tl.static_range(d):
            Pi_row_ptrs = Pi_ptr + j * stride_Pi_row + col_offsets
            Pi_row = tl.load(Pi_row_ptrs)
            dot_val = tl.sum(x_unit * Pi_row, axis=0)
            # Store rotated value at position j
            x_rot = tl.where(col_offsets == j, dot_val, x_rot)

        # 4. Boundary quantization via linear scan
        #    For small n_bounds (<=15 for 4-bit), this is efficient
        indices = tl.zeros([d], dtype=tl.int32)
        for b_idx in tl.static_range(n_bounds):
            bound_val = tl.load(bounds_ptr + b_idx)
            indices += tl.where(x_rot > bound_val, 1, 0).to(tl.int32)

        # 5. Store results
        idx_ptrs = idx_ptr + row * d + col_offsets
        tl.store(idx_ptrs, indices.to(tl.uint8))
        tl.store(norms_ptr + row, norm)

    @triton.jit
    def _compress_ip_kernel(
        x_ptr, Pi_ptr, bounds_ptr, S_ptr, centroids_ptr,
        mse_idx_ptr, qjl_ptr, norms_ptr,
        d: tl.constexpr, n_bounds: tl.constexpr, n_centroids: tl.constexpr,
        stride_x_row, stride_Pi_row,
    ):
        """Triton kernel: fused normalize → rotate → MSE quantize → QJL sign.

        Grid: (N,) where N = total number of vectors.
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, d)

        # 1. Load and normalize
        x_ptrs = x_ptr + row * stride_x_row + col_offsets
        x = tl.load(x_ptrs)
        norm_sq = tl.sum(x * x, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        x_unit = x / norm

        # 2. Rotate
        x_rot = tl.zeros([d], dtype=tl.float32)
        for j in tl.static_range(d):
            Pi_row_ptrs = Pi_ptr + j * stride_Pi_row + col_offsets
            Pi_row = tl.load(Pi_row_ptrs)
            dot_val = tl.sum(x_unit * Pi_row, axis=0)
            x_rot = tl.where(col_offsets == j, dot_val, x_rot)

        # 3. MSE boundary quantization
        indices = tl.zeros([d], dtype=tl.int32)
        for b_idx in tl.static_range(n_bounds):
            bound_val = tl.load(bounds_ptr + b_idx)
            indices += tl.where(x_rot > bound_val, 1, 0).to(tl.int32)

        # 4. Gather MSE centroids and compute residual
        #    centroid_vals[j] = centroids[indices[j]]
        centroid_vals = tl.zeros([d], dtype=tl.float32)
        for c_idx in tl.static_range(n_centroids):
            centroid_val = tl.load(centroids_ptr + c_idx)
            centroid_vals = tl.where(indices == c_idx, centroid_val, centroid_vals)
        residual = x_rot - centroid_vals

        # 5. QJL sign: sign[j] = 1 if S[j] * residual[j] >= 0 else 0
        S = tl.load(S_ptr + col_offsets)
        qjl_signs = tl.where(residual * S >= 0, 1, 0)

        # 6. Store
        tl.store(mse_idx_ptr + row * d + col_offsets, indices.to(tl.uint8))
        tl.store(qjl_ptr + row * d + col_offsets, qjl_signs.to(tl.uint8))
        tl.store(norms_ptr + row, norm)


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
        orig_shape = x.shape
        d = orig_shape[-1]
        x_flat = x.reshape(-1, d).contiguous().float()
        N = x_flat.shape[0]
        Pi_c = Pi.contiguous().float()
        bounds_c = boundaries.contiguous().float()

        idx_out = torch.empty(N, d, dtype=torch.uint8, device=x.device)
        norms_out = torch.empty(N, dtype=torch.float32, device=x.device)

        n_bounds = bounds_c.shape[0]
        _compress_mse_kernel[(N,)](
            x_flat, Pi_c, bounds_c, idx_out, norms_out,
            d=d, n_bounds=n_bounds,
            stride_x_row=x_flat.stride(0),
            stride_Pi_row=Pi_c.stride(0),
        )
        return idx_out.reshape(*orig_shape), norms_out.reshape(*orig_shape[:-1])
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
        orig_shape = x.shape
        d = orig_shape[-1]
        x_flat = x.reshape(-1, d).contiguous().float()
        N = x_flat.shape[0]
        Pi_c = Pi.contiguous().float()
        bounds_c = boundaries.contiguous().float()
        S_c = S.contiguous().float()
        cents_c = centroids.contiguous().float()

        mse_out = torch.empty(N, d, dtype=torch.uint8, device=x.device)
        qjl_out = torch.empty(N, d, dtype=torch.uint8, device=x.device)
        norms_out = torch.empty(N, dtype=torch.float32, device=x.device)

        n_bounds = bounds_c.shape[0]
        n_centroids = cents_c.shape[0]
        _compress_ip_kernel[(N,)](
            x_flat, Pi_c, bounds_c, S_c, cents_c,
            mse_out, qjl_out, norms_out,
            d=d, n_bounds=n_bounds, n_centroids=n_centroids,
            stride_x_row=x_flat.stride(0),
            stride_Pi_row=Pi_c.stride(0),
        )
        return (
            mse_out.reshape(*orig_shape),
            qjl_out.reshape(*orig_shape),
            norms_out.reshape(*orig_shape[:-1]),
        )
    return _compress_ip_pytorch(x, Pi, boundaries, S, centroids)
