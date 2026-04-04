"""
HuggingFace model integration for quashkv.

Provides a wrapper that hooks into a HuggingFace causal LM and compresses
the KV cache on the fly, enabling end-to-end evaluation of TurboQuant
compression on real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..engine import QuashKVEngine


@dataclass
class QuashKVCacheConfig:
    """Configuration for compressed KV cache."""
    key_bits: int = 3       # 2-bit MSE + 1-bit QJL
    value_bits: int = 3     # 3-bit MSE
    seed: int = 42


class QuashKVCache:
    """Drop-in replacement for HuggingFace DynamicCache that compresses KV pairs.

    This wraps one QuashKVEngine per attention layer. During generation,
    new KV pairs are compressed on append and decompressed during attention.

    Usage with HuggingFace:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from quashkv.integrations.hf_cache import QuashKVCache

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        cache = QuashKVCache.from_model(model)
        inputs = tokenizer("Hello", return_tensors="pt")
        outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        config: Optional[QuashKVCacheConfig] = None,
        device: str | torch.device = "cpu",
    ):
        self.config = config or QuashKVCacheConfig()
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.device = torch.device(device)

        # One engine per layer
        self.engines: list[QuashKVEngine] = []
        for layer_idx in range(num_layers):
            engine = QuashKVEngine(
                head_dim=head_dim,
                key_bits=self.config.key_bits,
                value_bits=self.config.value_bits,
                seed=self.config.seed + layer_idx,
                device=device,
            )
            self.engines.append(engine)

    @classmethod
    def from_model(
        cls,
        model: "PreTrainedModel",
        config: Optional[QuashKVCacheConfig] = None,
        device: Optional[str | torch.device] = None,
    ) -> "QuashKVCache":
        """Create a QuashKVCache from a HuggingFace model.

        Automatically detects num_layers and head_dim from the model config.
        """
        model_config = model.config
        num_layers = getattr(model_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Cannot detect num_hidden_layers from model config")

        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            hidden_size = model_config.hidden_size
            num_heads = model_config.num_attention_heads
            head_dim = hidden_size // num_heads

        if device is None:
            device = next(model.parameters()).device

        return cls(num_layers=num_layers, head_dim=head_dim, config=config, device=device)

    def _decompress_engine(
        self, engine: QuashKVEngine
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress all blocks in an engine into full-precision KV tensors.

        Note: Key reconstruction uses only the MSE centroid values (QJL signs
        are discarded).  This is correct for standard dot-product attention —
        the QJL correction is only needed in the fused quantized-attention path
        where scores are computed without explicit key reconstruction.
        """
        all_keys = []
        all_values = []
        for block in engine._cache:
            mse_indices, _, val_indices = engine._unpack_block(block)

            k_hat = engine.key_quantizer.mse_quantizer.codebook.dequantize(
                mse_indices
            )
            k_unit = engine.key_quantizer.mse_quantizer.unrotate(k_hat)
            all_keys.append(k_unit * block.key_norms.unsqueeze(-1))

            v_hat = engine.value_quantizer.codebook.dequantize(val_indices)
            v_unit = engine.value_quantizer.unrotate(v_hat)
            all_values.append(v_unit * block.val_norms.unsqueeze(-1))

        return (torch.cat(all_keys, dim=2), torch.cat(all_values, dim=2))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states and return the full (decompressed) KV for attention.

        This matches the HuggingFace Cache.update() interface.

        Parameters
        ----------
        key_states : (batch, n_kv_heads, new_seq_len, head_dim)
        value_states : (batch, n_kv_heads, new_seq_len, head_dim)
        layer_idx : int

        Returns
        -------
        full_keys : (batch, n_kv_heads, total_seq_len, head_dim)
        full_values : (batch, n_kv_heads, total_seq_len, head_dim)
        """
        engine = self.engines[layer_idx]

        # Compress and store the new tokens
        engine.append(key_states, value_states)

        # Decompress entire cache for standard attention
        # (In production, we'd use fused quantized attention instead)
        return self._decompress_engine(engine)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.engines[layer_idx].total_tokens

    def get_max_cache_length(self) -> Optional[int]:
        return None  # Unlimited

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Allow indexing like past_key_values[layer_idx]."""
        engine = self.engines[layer_idx]
        if not engine._cache:
            return (torch.empty(0), torch.empty(0))
        return self._decompress_engine(engine)

    def __len__(self) -> int:
        return self.num_layers

    def __iter__(self):
        for i in range(self.num_layers):
            yield self[i]

    def reset(self) -> None:
        """Clear all compressed caches."""
        for engine in self.engines:
            engine.clear()

    def compression_stats(self) -> dict:
        """Return compression statistics across all layers."""
        total_original = 0
        total_compressed = 0
        tokens_per_layer = []
        for engine in self.engines:
            tokens_per_layer.append(engine.total_tokens)
            if engine.total_tokens > 0:
                ratio = engine.compression_ratio()
                # Approximate memory
                d = engine.head_dim
                per_token = 2 * d * 16  # fp16 key + value
                compressed = (engine.key_bits * d + 32 + engine.value_bits * d + 32)
                total_original += engine.total_tokens * per_token
                total_compressed += engine.total_tokens * compressed

        return {
            "num_layers": self.num_layers,
            "tokens_per_layer": tokens_per_layer,
            "total_original_bits": total_original,
            "total_compressed_bits": total_compressed,
            "overall_compression_ratio": (
                total_original / total_compressed if total_compressed > 0 else 0
            ),
        }
