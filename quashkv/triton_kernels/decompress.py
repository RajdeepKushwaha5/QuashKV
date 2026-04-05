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

        Grid: (N,) where N = total number of vectors.
        Each program reconstructs one d-dimensional vector.
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, d)

        # 1. Load indices and gather centroids (tiny LUT: <=16 entries)
        idx_ptrs = idx_ptr + row * stride_idx_row + col_offsets
        indices = tl.load(idx_ptrs).to(tl.int32)

        centroid_vals = tl.zeros([d], dtype=tl.float32)
        for c_idx in tl.static_range(n_levels):
            c_val = tl.load(centroids_ptr + c_idx)
            centroid_vals = tl.where(indices == c_idx, c_val, centroid_vals)

        # 2. Un-rotate: out[j] = dot(centroid_vals, Pi[:, j]) = dot(centroid_vals, Pi[j, :])
        #    Since Pi is orthogonal, un-rotate = multiply by Pi (not Pi^T).
        #    But the PyTorch ref does x_rot_hat @ Pi, which for row vectors means
        #    out[j] = sum_k centroid_vals[k] * Pi[k, j].
        #    We compute this as: for each output dim j, dot centroid_vals with Pi column j.
        #    Pi column j = Pi[0,j], Pi[1,j], ... = Pi_ptr + j + k*stride_Pi_row
        norm = tl.load(norms_ptr + row)

        out_vals = tl.zeros([d], dtype=tl.float32)
        for j in tl.static_range(d):
            # Load Pi[:, j] — column j of Pi
            Pi_col_ptrs = Pi_ptr + col_offsets * stride_Pi_row + j
            Pi_col = tl.load(Pi_col_ptrs)
            dot_val = tl.sum(centroid_vals * Pi_col, axis=0)
            out_vals = tl.where(col_offsets == j, dot_val * norm, out_vals)

        # 3. Store
        out_ptrs = out_ptr + row * stride_out_row + col_offsets
        tl.store(out_ptrs, out_vals)

    @triton.jit
    def _decompress_ip_kernel(
        mse_idx_ptr, qjl_ptr, norms_ptr, Pi_ptr, centroids_ptr, S_ptr,
        out_ptr,
        d: tl.constexpr, n_levels: tl.constexpr,
        qjl_scale,
        stride_idx_row, stride_Pi_row, stride_out_row,
    ):
        """Triton kernel: fused IP decompression (MSE + QJL → un-rotate → rescale).

        Grid: (N,) where N = total number of vectors.
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, d)

        # 1. Load MSE indices and gather centroids
        mse_indices = tl.load(mse_idx_ptr + row * stride_idx_row + col_offsets).to(tl.int32)
        mse_vals = tl.zeros([d], dtype=tl.float32)
        for c_idx in tl.static_range(n_levels):
            c_val = tl.load(centroids_ptr + c_idx)
            mse_vals = tl.where(mse_indices == c_idx, c_val, mse_vals)

        # 2. QJL component: qjl_scale * S * (2*sign - 1)
        qjl_signs = tl.load(qjl_ptr + row * stride_idx_row + col_offsets).to(tl.float32)
        S = tl.load(S_ptr + col_offsets)
        sign_vals = qjl_signs * 2.0 - 1.0
        qjl_vals = qjl_scale * S * sign_vals

        # 3. Combined rotated-space representation
        combined = mse_vals + qjl_vals

        # 4. Un-rotate and scale
        norm = tl.load(norms_ptr + row)
        out_vals = tl.zeros([d], dtype=tl.float32)
        for j in tl.static_range(d):
            Pi_col_ptrs = Pi_ptr + col_offsets * stride_Pi_row + j
            Pi_col = tl.load(Pi_col_ptrs)
            dot_val = tl.sum(combined * Pi_col, axis=0)
            out_vals = tl.where(col_offsets == j, dot_val * norm, out_vals)

        tl.store(out_ptr + row * stride_out_row + col_offsets, out_vals)


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
        orig_shape = indices.shape
        d = orig_shape[-1]
        idx_flat = indices.reshape(-1, d).contiguous()
        N = idx_flat.shape[0]
        norms_flat = norms.reshape(-1).contiguous().float()
        Pi_c = Pi.contiguous().float()
        cents_c = centroids.contiguous().float()

        out = torch.empty(N, d, dtype=torch.float32, device=indices.device)
        n_levels = cents_c.shape[0]
        _decompress_mse_kernel[(N,)](
            idx_flat, norms_flat, Pi_c, cents_c, out,
            d=d, n_levels=n_levels,
            stride_idx_row=idx_flat.stride(0),
            stride_Pi_row=Pi_c.stride(0),
            stride_out_row=out.stride(0),
        )
        return out.reshape(*orig_shape)
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
        orig_shape = mse_indices.shape
        d = orig_shape[-1]
        idx_flat = mse_indices.reshape(-1, d).contiguous()
        qjl_flat = qjl_signs.reshape(-1, d).contiguous()
        N = idx_flat.shape[0]
        norms_flat = norms.reshape(-1).contiguous().float()
        Pi_c = Pi.contiguous().float()
        cents_c = centroids.contiguous().float()
        S_c = S.contiguous().float()

        out = torch.empty(N, d, dtype=torch.float32, device=mse_indices.device)
        n_levels = cents_c.shape[0]
        _decompress_ip_kernel[(N,)](
            idx_flat, qjl_flat, norms_flat, Pi_c, cents_c, S_c,
            out,
            d=d, n_levels=n_levels,
            qjl_scale=qjl_scale,
            stride_idx_row=idx_flat.stride(0),
            stride_Pi_row=Pi_c.stride(0),
            stride_out_row=out.stride(0),
        )
        return out.reshape(*orig_shape)
    return _decompress_ip_pytorch(
        mse_indices, qjl_signs, norms, Pi, centroids, S, qjl_scale,
    )
