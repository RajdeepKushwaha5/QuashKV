"""
vLLM attention backend integration for quashkv.

Provides a custom vLLM attention backend that stores KV cache in
TurboQuant-compressed form, decompressing on-the-fly during attention.

Architecture
------------
vLLM has two integration surfaces:
  1. **AttentionBackend** — declares cache shapes, dtypes, and metadata.
  2. **AttentionImpl.forward()** — runs the actual attention computation,
     receiving the KV cache tensor and query/key/value for the current step.

Our approach:
  - Allocate a standard paged KV cache (vLLM manages block tables).
  - On each forward pass:
      a. Compress the new K/V before writing to the cache pages.
      b. During attention, decompress stored blocks on-the-fly.
  - The compression/decompression uses QuashKV's fused kernels (Triton on
    GPU, PyTorch fallback on CPU).

Requirements
------------
  - vLLM >= 0.6 (v1 engine with AttentionBackend registry)
  - GPU with Triton support (for production speed; CPU works for testing)

Usage
-----
    from vllm import LLM, SamplingParams

    # Register the QuashKV backend before creating the LLM
    from quashkv.integrations.vllm_backend import register_quashkv_backend
    register_quashkv_backend()

    llm = LLM(
        model="meta-llama/Llama-3.2-1B",
        # Tell vLLM to use our backend (via environment or config)
        # Note: exact mechanism depends on vLLM version
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..engine import QuashKVEngine
from ..triton_kernels.compress import fused_compress_mse, fused_compress_ip
from ..triton_kernels.decompress import fused_decompress_mse
from ..triton_kernels.attention import fused_quantized_attention, CompressedBlock

# ---------------------------------------------------------------------------
# Try importing vLLM — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
    from vllm.v1.attention.backends.registry import (
        register_backend,
        AttentionBackendEnum,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# ---------------------------------------------------------------------------
# Compressed KV page manager
# ---------------------------------------------------------------------------

@dataclass
class QuashKVPageConfig:
    """Configuration for compressed KV pages in vLLM."""
    key_bits: int = 3       # 2-bit MSE + 1-bit QJL
    value_bits: int = 3     # 3-bit MSE
    block_size: int = 16    # tokens per page block
    seed: int = 42


class QuashKVPageManager:
    """Manages compressed KV storage for a single attention layer.

    Each 'page' stores compressed indices + norms for block_size tokens.
    This runs alongside vLLM's block allocator — we maintain a parallel
    compressed representation.

    Parameters
    ----------
    num_kv_heads : int
    head_size : int
    config : QuashKVPageConfig
    device : torch.device
    layer_idx : int
        Used to offset the random seed per layer.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        config: QuashKVPageConfig,
        device: torch.device,
        layer_idx: int = 0,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.config = config
        self.device = device

        # Per-layer engine for compression/decompression
        self.engine = QuashKVEngine(
            head_dim=head_size,
            key_bits=config.key_bits,
            value_bits=config.value_bits,
            seed=config.seed + layer_idx,
            device=device,
        )

    def compress_and_store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Compress new KV tokens and append to the engine cache.

        Parameters
        ----------
        keys : (batch, num_kv_heads, seq_len, head_size)
        values : (batch, num_kv_heads, seq_len, head_size)
        """
        self.engine.append(keys, values)

    def get_compressed_blocks(self) -> list[CompressedBlock]:
        """Return compressed blocks in the format expected by fused attention."""
        blocks = []
        for blk in self.engine._cache:
            mse, qjl, val = self.engine._unpack_block(blk)
            blocks.append(CompressedBlock(
                mse_indices=mse,
                qjl_signs=qjl,
                key_norms=blk.key_norms,
                val_indices=val,
                val_norms=blk.val_norms,
                seq_len=blk.seq_len,
            ))
        return blocks

    def attention(
        self,
        queries: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention over the compressed cache.

        Uses fused quantized attention with online softmax.

        Parameters
        ----------
        queries : (batch, num_kv_heads, q_len, head_size)
        scale : float, optional

        Returns
        -------
        output : (batch, num_kv_heads, q_len, head_size)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_size)

        blocks = self.get_compressed_blocks()
        if not blocks:
            raise ValueError("No compressed KV blocks stored yet")

        return fused_quantized_attention(
            queries=queries,
            blocks=blocks,
            Pi=self.engine.key_quantizer.mse_quantizer.Pi,
            key_centroids=self.engine.key_quantizer.mse_quantizer.codebook.centroids,
            val_centroids=self.engine.value_quantizer.codebook.centroids,
            S=self.engine.key_quantizer.S,
            qjl_scale=self.engine.key_quantizer.qjl_scale,
            scale=scale,
            total_kv_tokens=self.engine.total_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.engine.total_tokens

    def clear(self) -> None:
        self.engine.clear()

    def compression_ratio(self) -> float:
        return self.engine.compression_ratio()


# ---------------------------------------------------------------------------
# vLLM-compatible forward function (standalone, no vLLM dependency)
# ---------------------------------------------------------------------------

def quashkv_attention_forward(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    page_manager: QuashKVPageManager,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Combined compress-and-attend for a single layer step.

    This is the function that would be called from vLLM's AttentionImpl.forward().
    It compresses the new KV, stores them, and computes attention.

    Parameters
    ----------
    queries : (batch, num_heads, q_len, head_size)
    keys : (batch, num_kv_heads, new_kv_len, head_size)
    values : (batch, num_kv_heads, new_kv_len, head_size)
    page_manager : QuashKVPageManager for this layer
    scale : float, optional

    Returns
    -------
    output : (batch, num_heads, q_len, head_size)
    """
    # 1. Compress and store
    page_manager.compress_and_store(keys, values)

    # 2. Attend over entire compressed history
    return page_manager.attention(queries, scale=scale)


# ---------------------------------------------------------------------------
# Multi-layer manager (convenience wrapper)
# ---------------------------------------------------------------------------

class QuashKVModelManager:
    """Manages compressed KV caches for all layers of a model.

    This is the top-level entry point for vLLM integration.
    Create one per model, it manages one QuashKVPageManager per layer.

    Parameters
    ----------
    num_layers : int
    num_kv_heads : int
    head_size : int
    config : QuashKVPageConfig, optional
    device : torch.device
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        config: Optional[QuashKVPageConfig] = None,
        device: str | torch.device = "cpu",
    ):
        self.num_layers = num_layers
        self.config = config or QuashKVPageConfig()
        self.device = torch.device(device)

        self.layers: list[QuashKVPageManager] = [
            QuashKVPageManager(
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                config=self.config,
                device=self.device,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]

    def forward_layer(
        self,
        layer_idx: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run compressed attention for one layer."""
        return quashkv_attention_forward(
            queries, keys, values, self.layers[layer_idx], scale,
        )

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

    def stats(self) -> dict:
        """Return per-layer compression statistics."""
        return {
            "num_layers": self.num_layers,
            "tokens_per_layer": [l.total_tokens for l in self.layers],
            "compression_ratios": [
                l.compression_ratio() for l in self.layers
            ],
            "config": {
                "key_bits": self.config.key_bits,
                "value_bits": self.config.value_bits,
                "block_size": self.config.block_size,
            },
        }

    def total_compressed_bytes(self) -> int:
        return sum(l.engine.actual_memory_bytes() for l in self.layers)


# ---------------------------------------------------------------------------
# vLLM backend registration (only when vLLM is installed)
# ---------------------------------------------------------------------------

def register_quashkv_backend() -> None:
    """Register QuashKV as a vLLM attention backend.

    Call this before creating an LLM instance. Requires vLLM >= 0.6.

    Raises
    ------
    ImportError
        If vLLM is not installed.
    """
    if not HAS_VLLM:
        raise ImportError(
            "vLLM is not installed. Install it with: pip install vllm>=0.6"
        )
    # Registration would use:
    #   @register_backend(AttentionBackendEnum.CUSTOM)
    #   class QuashKVAttentionBackend(AttentionBackend): ...
    #
    # The full implementation requires GPU testing with vLLM's
    # attention metadata builders. The QuashKVModelManager above
    # provides all the compression/attention logic — the backend
    # class just wires it into vLLM's dispatch.
    raise NotImplementedError(
        "Full vLLM backend registration requires GPU testing. "
        "Use QuashKVModelManager directly for now, or QuashKVCache "
        "from quashkv.integrations.hf_cache for HuggingFace models."
    )
