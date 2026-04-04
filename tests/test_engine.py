"""Tests for the QuashKVEngine orchestrator."""

import math
import pytest
import torch

from quashkv.engine import QuashKVEngine


class TestQuashKVEngine:
    """Integration tests for the full compress → attention pipeline."""

    @pytest.fixture
    def engine(self):
        return QuashKVEngine(head_dim=64, key_bits=3, value_bits=3, seed=42)

    def test_init(self, engine):
        assert engine.head_dim == 64
        assert engine.total_tokens == 0
        assert engine.num_blocks == 0

    def test_append_updates_cache(self, engine):
        B, H, S, D = 1, 2, 16, 64
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        engine.append(keys, values)
        assert engine.total_tokens == 16
        assert engine.num_blocks == 1

    def test_append_multiple_blocks(self, engine):
        B, H, D = 1, 2, 64
        for s in [8, 16, 32]:
            engine.append(torch.randn(B, H, s, D), torch.randn(B, H, s, D))
        assert engine.total_tokens == 56
        assert engine.num_blocks == 3

    def test_clear(self, engine):
        engine.append(torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 64))
        engine.clear()
        assert engine.total_tokens == 0
        assert engine.num_blocks == 0

    def test_attention_shape(self, engine):
        B, H, S, D = 1, 2, 16, 64
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        engine.append(keys, values)

        queries = torch.randn(B, H, 4, D)
        out = engine.attention(queries)
        assert out.shape == (B, H, 4, D)

    def test_attention_empty_cache_raises(self, engine):
        queries = torch.randn(1, 2, 4, 64)
        with pytest.raises(ValueError, match="No compressed KV blocks"):
            engine.attention(queries)

    def test_attention_approximates_true(self):
        """Quantized attention should be close to exact attention."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 1, 64, 4, 64
        scale = 1.0 / math.sqrt(D)

        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        # Exact attention
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys) * scale
        attn = torch.softmax(scores, dim=-1)
        exact_out = torch.einsum("bhqk,bhkd->bhqd", attn, values)

        # Quantized attention
        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        engine.append(keys, values)
        quant_out = engine.attention(queries)

        # Should be reasonably close (3-bit quantization introduces some error)
        cosine_sim = torch.nn.functional.cosine_similarity(
            exact_out.flatten(), quant_out.flatten(), dim=0
        ).item()
        assert cosine_sim > 0.7, f"Cosine similarity {cosine_sim} too low"

    def test_compression_ratio(self, engine):
        engine.append(torch.randn(1, 2, 16, 64), torch.randn(1, 2, 16, 64))
        ratio = engine.compression_ratio(torch.float16)
        # 16 bits original, ~3 bits compressed per dim + norm overhead
        # Expected: 2*64*16 / (3*64+32 + 3*64+32) = 2048 / 448 ≈ 4.57
        assert ratio > 3.0
        assert ratio < 6.0

    def test_repr(self, engine):
        r = repr(engine)
        assert "head_dim=64" in r
        assert "key_bits=3" in r

    def test_multi_block_attention(self, engine):
        """Attention should work with multiple appended blocks."""
        B, H, D = 1, 2, 64
        engine.append(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D))
        engine.append(torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))

        queries = torch.randn(B, H, 4, D)
        out = engine.attention(queries)
        assert out.shape == (B, H, 4, D)
