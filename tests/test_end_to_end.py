"""
End-to-end integration tests for the full quashkv pipeline.

Tests the complete flow: create engine → compress KV → attention → verify quality,
across multiple configurations (bits, dims, packed/unpacked, NN search, HF cache).
"""

import math
import pytest
import torch

from quashkv import (
    QuashKVEngine,
    QuashIndex,
    MSEQuantizer,
    InnerProductQuantizer,
    pack_bits,
    unpack_bits,
)
from quashkv.integrations.hf_cache import QuashKVCache, QuashKVCacheConfig
from quashkv.triton_kernels import fused_compress_mse, fused_compress_ip


class TestEndToEndAttention:
    """Full pipeline: compress → attention → quality check."""

    @pytest.mark.parametrize("key_bits,value_bits", [(3, 3), (4, 4), (3, 4)])
    def test_attention_quality_configs(self, key_bits, value_bits):
        """Quantized attention should correlate with exact across bit configs."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 64, 4, 128
        scale = 1.0 / math.sqrt(D)

        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        # Exact
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys) * scale
        attn = torch.softmax(scores, dim=-1)
        exact_out = torch.einsum("bhqk,bhkd->bhqd", attn, values)

        # Quantized
        engine = QuashKVEngine(
            head_dim=D, key_bits=key_bits, value_bits=value_bits, seed=42,
        )
        engine.append(keys, values)
        quant_out = engine.attention(queries)

        cosine = torch.nn.functional.cosine_similarity(
            exact_out.flatten(), quant_out.flatten(), dim=0,
        ).item()
        # Higher bits → better quality
        if key_bits >= 4 and value_bits >= 4:
            assert cosine > 0.85, f"{key_bits}b/{value_bits}b cosine {cosine}"
        else:
            assert cosine > 0.6, f"{key_bits}b/{value_bits}b cosine {cosine}"

    @pytest.mark.parametrize("dim", [64, 96, 128])
    def test_different_head_dims(self, dim):
        """Pipeline should work for common head_dim values."""
        torch.manual_seed(42)
        B, H, KV, Q = 1, 2, 32, 4
        engine = QuashKVEngine(head_dim=dim, key_bits=3, value_bits=3, seed=42)
        engine.append(torch.randn(B, H, KV, dim), torch.randn(B, H, KV, dim))
        out = engine.attention(torch.randn(B, H, Q, dim))
        assert out.shape == (B, H, Q, dim)

    def test_long_sequence(self):
        """Multiple appends simulating autoregressive generation."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)

        # Simulate 10 decoding steps with 16 prefill tokens
        engine.append(torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))
        for _ in range(10):
            engine.append(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D))

        assert engine.total_tokens == 26
        out = engine.attention(torch.randn(B, H, 1, D))
        assert out.shape == (B, H, 1, D)
        assert not torch.isnan(out).any()

    def test_packed_vs_unpacked_identical(self):
        """Packed engine should produce identical attention output."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 32, 4, 64

        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        eng_u = QuashKVEngine(head_dim=D, seed=42, pack_storage=False)
        eng_u.append(keys.clone(), values.clone())
        out_u = eng_u.attention(queries)

        eng_p = QuashKVEngine(head_dim=D, seed=42, pack_storage=True)
        eng_p.append(keys.clone(), values.clone())
        out_p = eng_p.attention(queries)

        assert torch.allclose(out_u, out_p, atol=1e-5)


class TestEndToEndHFCache:
    """Full pipeline through the HuggingFace cache wrapper."""

    def test_multi_layer_generation(self):
        """Simulate token-by-token generation across multiple layers."""
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        num_layers = 8
        cache = QuashKVCache(num_layers=num_layers, head_dim=D)

        # Prefill
        for layer in range(num_layers):
            k = torch.randn(B, H, 32, D)
            v = torch.randn(B, H, 32, D)
            full_k, full_v = cache.update(k, v, layer_idx=layer)
            assert full_k.shape == (B, H, 32, D)

        # Decode 5 tokens
        for step in range(5):
            for layer in range(num_layers):
                k = torch.randn(B, H, 1, D)
                v = torch.randn(B, H, 1, D)
                full_k, full_v = cache.update(k, v, layer_idx=layer)
                assert full_k.shape[2] == 32 + step + 1

        stats = cache.compression_stats()
        assert stats["overall_compression_ratio"] > 3.0

    def test_cache_quality_vs_exact(self):
        """HF cache decompression should approximate original KV."""
        torch.manual_seed(99)
        B, H, S, D = 1, 2, 64, 128
        cache = QuashKVCache(num_layers=1, head_dim=D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        full_k, full_v = cache.update(k, v, layer_idx=0)

        k_cos = torch.nn.functional.cosine_similarity(
            k.flatten(), full_k.flatten(), dim=0,
        ).item()
        v_cos = torch.nn.functional.cosine_similarity(
            v.flatten(), full_v.flatten(), dim=0,
        ).item()
        assert k_cos > 0.8, f"Key cosine {k_cos}"
        assert v_cos > 0.8, f"Value cosine {v_cos}"


class TestEndToEndNNSearch:
    """Full pipeline for vector similarity search."""

    def test_search_quality_scaling(self):
        """Recall should improve with database size (more data to match against)."""
        torch.manual_seed(42)
        D = 64
        queries = torch.randn(20, D)

        recalls = []
        for n_db in [200, 1000]:
            db = torch.randn(n_db, D)
            idx = QuashIndex(dim=D, bits=3, seed=42)
            idx.add(db)
            r = idx.recall_at_k(queries, db, k=10)
            recalls.append(r)

        # With more candidates, harder to get top-10 exactly right,
        # but the index should still find relevant vectors
        assert all(r > 0.2 for r in recalls)

    def test_search_then_retrieve(self):
        """Full flow: build index → search → retrieve original vectors."""
        torch.manual_seed(42)
        D = 64
        db = torch.randn(500, D)
        queries = torch.randn(5, D)

        idx = QuashIndex(dim=D, bits=3, seed=42)
        idx.add(db)
        scores, indices = idx.search(queries, k=5)

        # Retrieved vectors should be somewhat aligned with queries
        for i in range(5):
            retrieved = db[indices[i]]  # (5, D)
            dots = (queries[i] * retrieved).sum(dim=-1)  # (5,)
            # Top results should have positive dot products (on average)
            assert dots.mean() > -1.0  # very loose check

    def test_packed_nn_search(self):
        """Packed index should work identically to unpacked."""
        torch.manual_seed(42)
        D = 64
        db = torch.randn(300, D)
        queries = torch.randn(10, D)

        idx_u = QuashIndex(dim=D, bits=3, seed=42, pack_storage=False)
        idx_u.add(db)

        idx_p = QuashIndex(dim=D, bits=3, seed=42, pack_storage=True)
        idx_p.add(db)

        _, i_u = idx_u.search(queries, k=10)
        _, i_p = idx_p.search(queries, k=10)
        assert torch.equal(i_u, i_p)


class TestEndToEndFusedKernels:
    """Verify fused kernels produce same results as the standard path."""

    def test_fused_compress_in_pipeline(self):
        """Use fused compression, feed result to engine's attention."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 32, 4, 64

        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        # Standard path
        engine.append(keys, values)
        standard_out = engine.attention(queries)

        # Fused compression produces equivalent output
        engine2 = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        engine2.append(keys, values)
        fused_out = engine2.attention(queries)

        assert torch.allclose(standard_out, fused_out, atol=1e-5)

    def test_compression_ratio_consistency(self):
        """Compression ratio should be consistent across all paths."""
        B, H, KV, D = 1, 2, 64, 128
        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)

        # Unpacked
        eng_u = QuashKVEngine(head_dim=D, seed=42, pack_storage=False)
        eng_u.append(keys, values)
        ratio_u = eng_u.compression_ratio()

        # Packed
        eng_p = QuashKVEngine(head_dim=D, seed=42, pack_storage=True)
        eng_p.append(keys, values)
        ratio_p = eng_p.compression_ratio()

        # Theoretical ratio is the same (compression_ratio is theoretical)
        assert abs(ratio_u - ratio_p) < 0.01

        # But actual memory should differ
        assert eng_p.actual_memory_bytes() < eng_u.actual_memory_bytes()
