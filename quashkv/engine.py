"""
QuashKV engine — the top-level orchestrator for KV cache compression.

This is the class users interact with.  It manages:
 • InnerProductQuantizer for **keys** (2-bit MSE + 1-bit QJL by default)
 • MSEQuantizer for **values** (3-bit MSE by default)
 • Compressed KV storage
 • Fused attention against the compressed cache

Usage
-----
    engine = QuashKVEngine(head_dim=128)
    engine.append(keys, values)          # compress & store a new KV block
    attn_out = engine.attention(queries)  # attend over entire compressed cache
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch

from .quantizer import MSEQuantizer, InnerProductQuantizer
from .packing import pack_bits, unpack_bits
from .constants import DEFAULT_TOTAL_BITS, DEFAULT_SEED


@dataclass
class CompressedKVBlock:
    """One compressed KV block (one layer, all heads at once).

    Tensors may be bit-packed (when engine.pack_storage=True) or raw uint8.

    For keys:
        mse_indices : packed or (batch, n_heads, seq_len, head_dim) uint8
        qjl_signs   : packed or (batch, n_heads, seq_len, head_dim) uint8
        key_norms    : (batch, n_heads, seq_len) float32

    For values:
        val_indices  : packed or (batch, n_heads, seq_len, head_dim) uint8
        val_norms    : (batch, n_heads, seq_len) float32
    """
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    key_norms: torch.Tensor
    val_indices: torch.Tensor
    val_norms: torch.Tensor
    seq_len: int
    packed: bool = False


class QuashKVEngine:
    """Orchestrator for TurboQuant-style KV cache compression.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head (e.g. 128).
    key_bits : int
        Total bits per coordinate for keys.  Default 3 (2-bit MSE + 1-bit QJL).
    value_bits : int
        Bits per coordinate for values.  Default 3 (3-bit MSE).
    seed : int
        Random seed for rotation matrix and QJL signs.
    device : str or torch.device
        Device to use.

    Notes
    -----
    The rotation matrix Π is shared between the key and value quantizers
    (same seed → same Π).  This is important: the query must be rotated
    once and used for both key matching and value retrieval.
    """

    def __init__(
        self,
        head_dim: int = 128,
        key_bits: int = DEFAULT_TOTAL_BITS,
        value_bits: int = DEFAULT_TOTAL_BITS,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
        pack_storage: bool = False,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.seed = seed
        self.device = torch.device(device)
        self.pack_storage = pack_storage

        # Key quantizer: (b-1)-bit MSE + 1-bit QJL
        self.key_quantizer = InnerProductQuantizer(
            d=head_dim, total_bits=key_bits, seed=seed, device=device,
        )

        # Value quantizer: b-bit MSE (no QJL needed for values)
        self.value_quantizer = MSEQuantizer(
            d=head_dim, bits=value_bits, seed=seed, device=device,
        )

        # Compressed cache: list of blocks
        self._cache: list[CompressedKVBlock] = []
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_keys(
        self, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress key vectors.

        Parameters
        ----------
        keys : torch.Tensor, shape (batch, n_heads, seq_len, head_dim)

        Returns
        -------
        mse_indices, qjl_signs, norms
        """
        return self.key_quantizer.compress(keys)

    def compress_values(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress value vectors.

        Parameters
        ----------
        values : torch.Tensor, shape (batch, n_heads, seq_len, head_dim)

        Returns
        -------
        indices, norms
        """
        return self.value_quantizer.compress(values)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Compress and append a new KV block to the cache.

        Parameters
        ----------
        keys : (batch, n_heads, seq_len, head_dim)
        values : (batch, n_heads, seq_len, head_dim)
        """
        mse_idx, qjl_sgn, k_norms = self.compress_keys(keys)
        v_idx, v_norms = self.compress_values(values)

        if self.pack_storage:
            mse_bits = self.key_bits - 1  # MSE uses (total - 1) bits
            mse_idx = pack_bits(mse_idx, mse_bits)
            qjl_sgn = pack_bits(qjl_sgn, 1)
            v_idx = pack_bits(v_idx, self.value_bits)

        block = CompressedKVBlock(
            mse_indices=mse_idx,
            qjl_signs=qjl_sgn,
            key_norms=k_norms,
            val_indices=v_idx,
            val_norms=v_norms,
            seq_len=keys.shape[-2],
            packed=self.pack_storage,
        )
        self._cache.append(block)
        self._total_tokens += block.seq_len

    def clear(self) -> None:
        """Clear the compressed cache."""
        self._cache.clear()
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_blocks(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _unpack_block(self, block: CompressedKVBlock) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return (mse_indices, qjl_signs, val_indices) unpacked to uint8."""
        if not block.packed:
            return block.mse_indices, block.qjl_signs, block.val_indices
        d = self.head_dim
        mse_bits = self.key_bits - 1
        mse_idx = unpack_bits(block.mse_indices, mse_bits, d)
        qjl_sgn = unpack_bits(block.qjl_signs, 1, d)
        v_idx = unpack_bits(block.val_indices, self.value_bits, d)
        return mse_idx, qjl_sgn, v_idx

    def attention(
        self,
        queries: torch.Tensor,
        scale: Optional[float] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute attention over the entire compressed KV cache.

        This is the pure-PyTorch reference implementation.  It decompresses
        block by block, computes softmax attention, and returns the weighted
        value sum.  A fused Triton kernel will replace this in Week 2.

        Parameters
        ----------
        queries : torch.Tensor, shape (batch, n_heads, q_len, head_dim)
        scale : float, optional
            Attention scale.  Default: 1/√head_dim.
        causal : bool
            If True, apply causal masking.

        Returns
        -------
        output : torch.Tensor, shape (batch, n_heads, q_len, head_dim)
        """
        if not self._cache:
            raise ValueError("No compressed KV blocks in cache. Call append() first.")

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        batch, n_heads, q_len, d = queries.shape

        # Rotate queries once (shared rotation matrix)
        Pi = self.key_quantizer.mse_quantizer.Pi
        q_rot = queries @ Pi.T  # (batch, n_heads, q_len, d)

        # Collect all scores and values
        all_scores = []
        all_values = []
        kv_offset = 0

        for block in self._cache:
            kv_len = block.seq_len
            mse_indices, qjl_signs, val_indices = self._unpack_block(block)

            # --- Key scores via two-stage IP estimator ---
            # Expand q_rot for broadcasting
            # q_rot: (B, H, Q, D),  block entries: (B, H, KV, D)
            q_exp = q_rot.unsqueeze(3).expand(-1, -1, -1, kv_len, -1)

            # MSE part
            k_hat = self.key_quantizer.mse_quantizer.codebook.dequantize(
                mse_indices
            )  # (B, H, KV, D)
            k_hat_exp = k_hat.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
            mse_dot = (q_exp * k_hat_exp).sum(dim=-1)  # (B, H, Q, KV)

            # QJL correction
            S = self.key_quantizer.S
            sign_vals = qjl_signs.float() * 2.0 - 1.0  # (B, H, KV, D)
            sign_exp = sign_vals.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
            correction = self.key_quantizer.qjl_scale * (
                q_exp.abs() * sign_exp * S
            ).sum(dim=-1)

            # Combine with key norms
            k_norms = block.key_norms.unsqueeze(2).expand(-1, -1, q_len, -1)
            scores = k_norms * (mse_dot + correction) * scale  # (B, H, Q, KV)

            if causal:
                # Build causal mask: query pos i can attend to kv pos j if kv_offset + j <= current_q_offset + i
                # For simplicity, assume queries start after all previous KV
                q_pos = torch.arange(q_len, device=queries.device) + self._total_tokens - q_len
                kv_pos = torch.arange(kv_len, device=queries.device) + kv_offset
                mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # (Q, KV)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            all_scores.append(scores)

            # --- Values: decompress ---
            v_hat = self.value_quantizer.codebook.dequantize(val_indices)
            v_reconst = self.value_quantizer.unrotate(v_hat)
            v_reconst = v_reconst * block.val_norms.unsqueeze(-1)
            all_values.append(v_reconst)

            kv_offset += kv_len

        # Concatenate across all blocks
        all_scores = torch.cat(all_scores, dim=-1)  # (B, H, Q, total_KV)
        all_values = torch.cat(all_values, dim=2)    # (B, H, total_KV, D)

        # Softmax + weighted sum
        attn_weights = torch.softmax(all_scores, dim=-1)  # (B, H, Q, total_KV)
        output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, all_values)

        return output

    # ------------------------------------------------------------------
    # Compression ratio calculation
    # ------------------------------------------------------------------

    def compression_ratio(self, original_dtype: torch.dtype = torch.float16) -> float:
        """Estimate the compression ratio achieved.

        Original: each KV element is stored in `original_dtype` (e.g. 16 bits).
        Compressed: each element is `key_bits` or `value_bits` bits, plus norms.
        """
        if not self._cache:
            return 1.0

        orig_bits = torch.finfo(original_dtype).bits
        # Per token per head: 2 * head_dim * orig_bits (key + value)
        original_per_token = 2 * self.head_dim * orig_bits

        # Compressed: keys = (key_bits * head_dim + 32) bits per token
        #            (the 32 is for the float32 norm)
        # Values: (value_bits * head_dim + 32) bits per token
        compressed_per_token = (
            self.key_bits * self.head_dim + 32  # key indices + norm
            + self.value_bits * self.head_dim + 32  # value indices + norm
        )

        return original_per_token / compressed_per_token

    def actual_memory_bytes(self) -> int:
        """Compute actual memory used by the compressed cache in bytes."""
        total = 0
        for block in self._cache:
            total += block.mse_indices.nelement() * block.mse_indices.element_size()
            total += block.qjl_signs.nelement() * block.qjl_signs.element_size()
            total += block.key_norms.nelement() * block.key_norms.element_size()
            total += block.val_indices.nelement() * block.val_indices.element_size()
            total += block.val_norms.nelement() * block.val_norms.element_size()
        return total

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "QuashKVEngine":
        self.device = torch.device(device)
        self.key_quantizer.to(device)
        self.value_quantizer.to(device)
        for block in self._cache:
            block.mse_indices = block.mse_indices.to(device)
            block.qjl_signs = block.qjl_signs.to(device)
            block.key_norms = block.key_norms.to(device)
            block.val_indices = block.val_indices.to(device)
            block.val_norms = block.val_norms.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"QuashKVEngine(head_dim={self.head_dim}, key_bits={self.key_bits}, "
            f"value_bits={self.value_bits}, cached_tokens={self._total_tokens}, "
            f"blocks={len(self._cache)}, device={self.device})"
        )
