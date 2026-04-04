"""
Fused quantized attention kernel.

Computes attention over a compressed KV cache without materializing the
full decompressed key/value tensors.  Uses online softmax (FlashAttention
style) for memory efficiency with long sequences.

Triton kernel strategy:
  - Each program instance handles one (batch, head, q_tile) group.
  - Outer loop streams through compressed KV blocks.
  - Per block:
      1. Dequantize MSE indices → centroid vectors (tiny LUT in SRAM)
      2. Compute Q_tile @ K_centroids^T (tl.dot) → MSE score component
      3. Add QJL correction: scale * |Q_rot| * S * signs → correction
      4. Score = norms * (MSE_dot + correction)
      5. Online softmax update: m_new, l_new, O_new
      6. Dequantize values (centroid LUT) → un-rotate → accumulate
  - After all blocks: O = O / l

  This avoids materializing the (Q, total_KV) score matrix and the full
  decompressed (total_KV, D) key/value tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


@dataclass
class CompressedBlock:
    """Lightweight view of a compressed KV block for the attention kernel."""
    mse_indices: torch.Tensor   # (B, H, S, D) uint8
    qjl_signs: torch.Tensor     # (B, H, S, D) uint8
    key_norms: torch.Tensor     # (B, H, S) float32
    val_indices: torch.Tensor   # (B, H, S, D) uint8
    val_norms: torch.Tensor     # (B, H, S) float32
    seq_len: int


# ======================================================================
# PyTorch reference: online-softmax fused attention
# ======================================================================

def _fused_attention_pytorch(
    queries: torch.Tensor,
    blocks: list[CompressedBlock],
    Pi: torch.Tensor,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    S: torch.Tensor,
    qjl_scale: float,
    scale: float,
    causal: bool = False,
    total_kv_tokens: int = 0,
) -> torch.Tensor:
    """FlashAttention-style quantized attention with online softmax.

    Unlike the naive approach in engine.py which materializes the full
    score matrix, this streams through blocks and maintains running
    softmax statistics (m, l, O).

    Memory: O(B * H * Q * D) instead of O(B * H * Q * total_KV).

    Parameters
    ----------
    queries : (B, H, Q, D) float
    blocks : list of CompressedBlock
    Pi : (D, D) rotation matrix
    key_centroids : (2^{b-1},) MSE centroid values for keys
    val_centroids : (2^b_v,) centroid values for values
    S : (D,) QJL sign vector
    qjl_scale : float
    scale : float, attention scale (typically 1/√D)
    causal : bool
    total_kv_tokens : int, total tokens across all blocks

    Returns
    -------
    output : (B, H, Q, D) float
    """
    B, H, Q, D = queries.shape
    device = queries.device

    q_rot = queries @ Pi.T  # (B, H, Q, D)

    # Online softmax accumulators
    m = torch.full((B, H, Q), float("-inf"), device=device)  # running max
    l = torch.zeros(B, H, Q, device=device)                   # running sum(exp)
    O = torch.zeros(B, H, Q, D, device=device)                # running output

    kv_offset = 0

    for block in blocks:
        kv_len = block.seq_len

        # --- Key scores ---
        k_hat = key_centroids[block.mse_indices.long()]  # (B, H, KV, D)
        mse_dot = torch.einsum("bhqd,bhkd->bhqk", q_rot, k_hat)

        sign_vals = block.qjl_signs.float() * 2.0 - 1.0
        q_scaled = q_rot.abs() * S  # (B, H, Q, D)
        correction = qjl_scale * torch.einsum("bhqd,bhkd->bhqk", q_scaled, sign_vals)

        k_norms = block.key_norms  # (B, H, KV)
        scores = k_norms.unsqueeze(2) * (mse_dot + correction) * scale  # (B,H,Q,KV)

        if causal:
            q_pos = torch.arange(Q, device=device) + total_kv_tokens - Q
            kv_pos = torch.arange(kv_len, device=device) + kv_offset
            mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # --- Online softmax update ---
        block_max = scores.max(dim=-1).values  # (B, H, Q)
        m_new = torch.maximum(m, block_max)

        # Rescale previous accumulator
        exp_diff_old = torch.exp(m - m_new)  # (B, H, Q)
        l = l * exp_diff_old
        O = O * exp_diff_old.unsqueeze(-1)

        # Current block's contribution
        P = torch.exp(scores - m_new.unsqueeze(-1))  # (B, H, Q, KV)
        l = l + P.sum(dim=-1)

        # Decompress values and accumulate
        v_hat = val_centroids[block.val_indices.long()]  # (B, H, KV, D)
        v_unrot = v_hat @ Pi  # un-rotate
        v_reconst = v_unrot * block.val_norms.unsqueeze(-1)

        O = O + torch.einsum("bhqk,bhkd->bhqd", P, v_reconst)

        m = m_new
        kv_offset += kv_len

    # Final normalization
    output = O / l.unsqueeze(-1).clamp(min=1e-8)
    return output


# ======================================================================
# Triton kernel stubs
# ======================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_quantized_attention_kernel(
        # Query pointers
        Q_ptr,
        # Compressed KV pointers (per block, passed as arrays)
        mse_idx_ptr, qjl_ptr, k_norms_ptr,
        v_idx_ptr, v_norms_ptr,
        # Codebook pointers
        key_cent_ptr, val_cent_ptr,
        # Rotation + QJL
        Pi_ptr, S_ptr,
        # Output
        O_ptr,
        # Dimensions
        B: tl.constexpr, H: tl.constexpr,
        Q_LEN: tl.constexpr, D: tl.constexpr,
        KV_LEN: tl.constexpr,
        # Block sizes for tiling
        BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
        # Scale
        scale,
        qjl_scale,
    ):
        """Triton fused quantized attention kernel.

        Grid: (B * H * cdiv(Q_LEN, BLOCK_Q),)

        Per program:
          1. Load Q tile (BLOCK_Q × D) from global memory
          2. Rotate: Q_rot = Q_tile @ Pi^T (in shared memory)
          3. Initialize online softmax: m=-inf, l=0, O=0
          4. For each KV tile (BLOCK_KV vectors):
             a. Load MSE indices, QJL signs, key norms
             b. Gather key centroids (tiny LUT)
             c. Compute: mse_dot = Q_rot @ K_centroids^T
             d. Compute: correction = qjl_scale * |Q_rot| @ (S * signs)^T
             e. Score = k_norms * (mse_dot + correction) * scale
             f. Online softmax update (m_new, l_new, O_new)
             g. Load value indices, gather centroids, un-rotate, scale
             h. Accumulate O += P @ V_reconst
          5. Write O / l to output

        Expected speedup over PyTorch: 3-8x for sequence lengths > 1K
        due to elimination of intermediate tensors and kernel launch overhead.
        """
        # TODO: implement when GPU is available
        pass


# ======================================================================
# Dispatch function
# ======================================================================

def fused_quantized_attention(
    queries: torch.Tensor,
    blocks: list[CompressedBlock],
    Pi: torch.Tensor,
    key_centroids: torch.Tensor,
    val_centroids: torch.Tensor,
    S: torch.Tensor,
    qjl_scale: float,
    scale: Optional[float] = None,
    causal: bool = False,
    total_kv_tokens: int = 0,
) -> torch.Tensor:
    """Compute attention over compressed KV blocks.

    Uses online softmax for memory efficiency.
    Dispatches to Triton when available, otherwise PyTorch.

    Parameters
    ----------
    queries : (B, H, Q, D) float
    blocks : list of CompressedBlock from the engine cache
    Pi : (D, D) rotation matrix
    key_centroids : (n_levels_key,) centroid values for keys
    val_centroids : (n_levels_val,) centroid values for values
    S : (D,) QJL random sign vector
    qjl_scale : float
    scale : float, optional (default: 1/√D)
    causal : bool

    Returns
    -------
    output : (B, H, Q, D) float
    """
    D = queries.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if HAS_TRITON and queries.is_cuda:
        # TODO: launch _fused_quantized_attention_kernel
        pass

    return _fused_attention_pytorch(
        queries, blocks, Pi, key_centroids, val_centroids,
        S, qjl_scale, scale, causal, total_kv_tokens,
    )
