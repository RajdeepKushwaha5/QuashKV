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
        # Pointers
        Q_ptr,
        mse_idx_ptr, qjl_ptr, k_norms_ptr,
        v_idx_ptr, v_norms_ptr,
        key_cent_ptr, val_cent_ptr,
        Pi_ptr, S_ptr,
        O_ptr,
        # Strides for Q/O: (B, H, Q, D) — last dim stride assumed 1
        stride_q_b, stride_q_h, stride_q_s,
        # Strides for KV indices: (B, H, KV, D) — last dim stride assumed 1
        stride_kv_b, stride_kv_h, stride_kv_s,
        # Strides for KV norms: (B, H, KV)
        stride_kn_b, stride_kn_h,
        # Pi stride: (D, D) row-major
        stride_Pi_row,
        # Dimensions
        B: tl.constexpr, H: tl.constexpr,
        Q_LEN: tl.constexpr, D: tl.constexpr,
        KV_LEN: tl.constexpr,
        # Block sizes for query / KV tiling (powers of 2, ≥ 16)
        BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
        # Codebook sizes
        N_KEY_CENTS: tl.constexpr, N_VAL_CENTS: tl.constexpr,
        # Scalars
        scale,
        qjl_scale,
    ):
        """Fused quantized attention with online softmax (FlashAttention-style).

        Grid: (cdiv(Q_LEN, BLOCK_Q) * B * H,)

        Each program handles BLOCK_Q queries, streaming through all KV
        tokens in tiles of BLOCK_KV.  Keys are scored directly in
        rotated space (no key un-rotation needed).  Values are
        decompressed on-the-fly via centroid gather + un-rotation.
        """
        # Map program ID → (batch, head, q_block)
        pid = tl.program_id(0)
        n_q_blocks = (Q_LEN + BLOCK_Q - 1) // BLOCK_Q
        bh_idx = pid // n_q_blocks
        q_block_idx = pid % n_q_blocks
        b_idx = bh_idx // H
        h_idx = bh_idx % H

        q_start = q_block_idx * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, D)
        q_mask = q_offs < Q_LEN

        # 1. Load Q tile: (BLOCK_Q, D)
        q_ptrs = (Q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
                  + q_offs[:, None] * stride_q_s + d_offs[None, :])
        Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        # 2. Load rotation matrix Pi: (D, D) and rotate queries
        Pi_block = tl.load(
            Pi_ptr + d_offs[:, None] * stride_Pi_row + d_offs[None, :]
        ).to(tl.float32)
        Q_rot = tl.dot(Q_tile, tl.trans(Pi_block))  # Q @ Pi^T

        # 3. Precompute |Q_rot| * S for QJL correction
        S_vec = tl.load(S_ptr + d_offs).to(tl.float32)
        Q_abs_S = tl.abs(Q_rot) * S_vec[None, :]  # (BLOCK_Q, D)

        # 4. Online softmax accumulators
        m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        O_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

        # Base offsets for this (batch, head)
        kv_off = b_idx * stride_kv_b + h_idx * stride_kv_h
        kn_off = b_idx * stride_kn_b + h_idx * stride_kn_h

        # 5. Stream through KV tiles
        for kv_start in range(0, KV_LEN, BLOCK_KV):
            kv_offs = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_offs < KV_LEN

            # --- Key score computation (in rotated space) ---
            # Load MSE indices and gather key centroids via LUT
            mse_ptrs = (mse_idx_ptr + kv_off
                        + kv_offs[:, None] * stride_kv_s + d_offs[None, :])
            mse_idx = tl.load(mse_ptrs, mask=kv_mask[:, None], other=0).to(tl.int32)

            k_hat = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
            for c in tl.static_range(N_KEY_CENTS):
                c_val = tl.load(key_cent_ptr + c)
                k_hat = tl.where(mse_idx == c, c_val, k_hat)

            # MSE dot: Q_rot @ k_hat^T → (BLOCK_Q, BLOCK_KV)
            scores = tl.dot(Q_rot, tl.trans(k_hat))

            # QJL correction: qjl_scale * |Q_rot|*S @ sign_vals^T
            qjl_ptrs = (qjl_ptr + kv_off
                        + kv_offs[:, None] * stride_kv_s + d_offs[None, :])
            qjl_raw = tl.load(qjl_ptrs, mask=kv_mask[:, None], other=0).to(tl.float32)
            sign_vals = qjl_raw * 2.0 - 1.0  # {0,1} → {-1,+1}
            correction = qjl_scale * tl.dot(Q_abs_S, tl.trans(sign_vals))

            # Scale by key norms
            k_norms = tl.load(k_norms_ptr + kn_off + kv_offs,
                              mask=kv_mask, other=0.0).to(tl.float32)
            scores = k_norms[None, :] * (scores + correction) * scale
            scores = tl.where(kv_mask[None, :], scores, float('-inf'))

            # --- Online softmax update ---
            block_max = tl.max(scores, axis=1)  # (BLOCK_Q,)
            m_new = tl.maximum(m_i, block_max)
            exp_old = tl.exp(m_i - m_new)
            l_i = l_i * exp_old
            O_i = O_i * exp_old[:, None]

            P = tl.exp(scores - m_new[:, None])  # (BLOCK_Q, BLOCK_KV)
            l_i = l_i + tl.sum(P, axis=1)

            # --- Value decompression on-the-fly ---
            v_ptrs = (v_idx_ptr + kv_off
                      + kv_offs[:, None] * stride_kv_s + d_offs[None, :])
            v_idx = tl.load(v_ptrs, mask=kv_mask[:, None], other=0).to(tl.int32)

            val_hat = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
            for c in tl.static_range(N_VAL_CENTS):
                c_val = tl.load(val_cent_ptr + c)
                val_hat = tl.where(v_idx == c, c_val, val_hat)

            # Un-rotate: val_hat @ Pi (Pi is orthogonal, so Pi^{-1} = Pi^T)
            val_unrot = tl.dot(val_hat, Pi_block)

            # Scale by value norms
            v_norms = tl.load(v_norms_ptr + kn_off + kv_offs,
                              mask=kv_mask, other=0.0).to(tl.float32)
            val_reconst = val_unrot * v_norms[:, None]

            # Accumulate O += P @ V
            O_i = O_i + tl.dot(P.to(tl.float32), val_reconst)
            m_i = m_new

        # 6. Final normalization and store
        O_i = O_i / tl.maximum(l_i[:, None], 1e-8)
        o_ptrs = (O_ptr + b_idx * stride_q_b + h_idx * stride_q_h
                  + q_offs[:, None] * stride_q_s + d_offs[None, :])
        tl.store(o_ptrs, O_i, mask=q_mask[:, None])


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

    if HAS_TRITON and queries.is_cuda and not causal:
        B, H, Q, _ = queries.shape

        # Concatenate all blocks along sequence dimension
        all_mse = torch.cat([b.mse_indices for b in blocks], dim=2).contiguous()
        all_qjl = torch.cat([b.qjl_signs for b in blocks], dim=2).contiguous()
        all_knorms = torch.cat([b.key_norms for b in blocks], dim=2).contiguous().float()
        all_vidx = torch.cat([b.val_indices for b in blocks], dim=2).contiguous()
        all_vnorms = torch.cat([b.val_norms for b in blocks], dim=2).contiguous().float()

        KV = all_mse.shape[2]
        queries_c = queries.contiguous().float()
        Pi_c = Pi.contiguous().float()
        kcents = key_centroids.contiguous().float()
        vcents = val_centroids.contiguous().float()
        S_c = S.contiguous().float()

        O = torch.zeros(B, H, Q, D, dtype=torch.float32, device=queries.device)

        BLOCK_Q = max(16, triton.next_power_of_2(min(Q, 64)))
        BLOCK_KV = max(16, triton.next_power_of_2(min(KV, 64)))

        grid = (triton.cdiv(Q, BLOCK_Q) * B * H,)

        _fused_quantized_attention_kernel[grid](
            queries_c,
            all_mse, all_qjl, all_knorms,
            all_vidx, all_vnorms,
            kcents, vcents,
            Pi_c, S_c,
            O,
            queries_c.stride(0), queries_c.stride(1), queries_c.stride(2),
            all_mse.stride(0), all_mse.stride(1), all_mse.stride(2),
            all_knorms.stride(0), all_knorms.stride(1),
            Pi_c.stride(0),
            B=B, H=H, Q_LEN=Q, D=D, KV_LEN=KV,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
            N_KEY_CENTS=kcents.shape[0], N_VAL_CENTS=vcents.shape[0],
            scale=scale, qjl_scale=qjl_scale,
        )
        return O

    return _fused_attention_pytorch(
        queries, blocks, Pi, key_centroids, val_centroids,
        S, qjl_scale, scale, causal, total_kv_tokens,
    )
