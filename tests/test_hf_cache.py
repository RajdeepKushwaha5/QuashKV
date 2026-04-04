"""Tests for the HuggingFace cache integration."""

import math
import pytest
import torch

from quashkv.integrations.hf_cache import QuashKVCache, QuashKVCacheConfig


class TestQuashKVCacheConfig:
    def test_defaults(self):
        cfg = QuashKVCacheConfig()
        assert cfg.key_bits == 3
        assert cfg.value_bits == 3
        assert cfg.seed == 42

    def test_custom(self):
        cfg = QuashKVCacheConfig(key_bits=4, value_bits=4, seed=99)
        assert cfg.key_bits == 4
        assert cfg.value_bits == 4
        assert cfg.seed == 99


class TestQuashKVCache:
    """Functional tests for the HF-compatible compressed KV cache."""

    @pytest.fixture
    def cache(self):
        return QuashKVCache(num_layers=4, head_dim=64)

    # ---- init / basic properties ----

    def test_init_defaults(self, cache):
        assert cache.num_layers == 4
        assert cache.head_dim == 64
        assert len(cache.engines) == 4

    def test_init_custom_config(self):
        cfg = QuashKVCacheConfig(key_bits=4, value_bits=4, seed=7)
        c = QuashKVCache(num_layers=2, head_dim=128, config=cfg)
        assert c.engines[0].key_bits == 4
        assert c.engines[0].value_bits == 4

    def test_len(self, cache):
        assert len(cache) == 4

    def test_seed_offset_per_layer(self, cache):
        """Each layer engine gets a unique seed (base + layer_idx)."""
        seeds = [e.seed for e in cache.engines]
        assert seeds == [42, 43, 44, 45]

    # ---- update / roundtrip ----

    def test_update_returns_correct_shapes(self, cache):
        B, H, S, D = 1, 2, 16, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        full_k, full_v = cache.update(k, v, layer_idx=0)
        assert full_k.shape == (B, H, S, D)
        assert full_v.shape == (B, H, S, D)

    def test_update_accumulates(self, cache):
        B, H, D = 1, 2, 64
        k1 = torch.randn(B, H, 8, D)
        v1 = torch.randn(B, H, 8, D)
        cache.update(k1, v1, layer_idx=0)

        k2 = torch.randn(B, H, 4, D)
        v2 = torch.randn(B, H, 4, D)
        full_k, full_v = cache.update(k2, v2, layer_idx=0)
        # Should have 8 + 4 = 12 tokens
        assert full_k.shape == (B, H, 12, D)
        assert full_v.shape == (B, H, 12, D)

    def test_update_different_layers(self, cache):
        B, H, D = 1, 2, 64
        for layer in range(4):
            s = 8 * (layer + 1)
            cache.update(torch.randn(B, H, s, D), torch.randn(B, H, s, D), layer_idx=layer)

        assert cache.get_seq_length(0) == 8
        assert cache.get_seq_length(1) == 16
        assert cache.get_seq_length(2) == 24
        assert cache.get_seq_length(3) == 32

    def test_update_roundtrip_quality(self):
        """Decompressed KV from update() should approximate the original."""
        torch.manual_seed(123)
        B, H, S, D = 1, 1, 32, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        cache = QuashKVCache(num_layers=1, head_dim=D)
        full_k, full_v = cache.update(k, v, layer_idx=0)

        k_cos = torch.nn.functional.cosine_similarity(
            k.flatten(), full_k.flatten(), dim=0
        ).item()
        v_cos = torch.nn.functional.cosine_similarity(
            v.flatten(), full_v.flatten(), dim=0
        ).item()
        # 3-bit quantization should preserve reasonable quality
        assert k_cos > 0.75, f"Key cosine {k_cos} too low"
        assert v_cos > 0.75, f"Value cosine {v_cos} too low"

    # ---- __getitem__ / __iter__ ----

    def test_getitem_empty_layer(self, cache):
        k, v = cache[0]
        assert k.numel() == 0
        assert v.numel() == 0

    def test_getitem_after_update(self, cache):
        B, H, S, D = 1, 2, 16, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        cache.update(k, v, layer_idx=1)

        got_k, got_v = cache[1]
        assert got_k.shape == (B, H, S, D)
        assert got_v.shape == (B, H, S, D)

    def test_iter(self, cache):
        B, H, D = 1, 2, 64
        for i in range(4):
            cache.update(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D), layer_idx=i)
        layers = list(cache)
        assert len(layers) == 4
        for k, v in layers:
            assert k.shape == (B, H, 8, D)

    # ---- seq length / max length ----

    def test_get_seq_length_default(self, cache):
        assert cache.get_seq_length() == 0

    def test_get_seq_length_specific_layer(self, cache):
        cache.update(torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64), layer_idx=2)
        assert cache.get_seq_length(2) == 10
        assert cache.get_seq_length(0) == 0

    def test_get_max_cache_length(self, cache):
        assert cache.get_max_cache_length() is None

    # ---- reset ----

    def test_reset(self, cache):
        B, H, D = 1, 2, 64
        for i in range(4):
            cache.update(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D), layer_idx=i)
        cache.reset()
        for i in range(4):
            assert cache.get_seq_length(i) == 0

    # ---- compression_stats ----

    def test_compression_stats_empty(self, cache):
        stats = cache.compression_stats()
        assert stats["num_layers"] == 4
        assert stats["overall_compression_ratio"] == 0
        assert stats["tokens_per_layer"] == [0, 0, 0, 0]

    def test_compression_stats_nonempty(self, cache):
        B, H, D = 1, 2, 64
        for i in range(4):
            cache.update(torch.randn(B, H, 16, D), torch.randn(B, H, 16, D), layer_idx=i)
        stats = cache.compression_stats()
        assert stats["num_layers"] == 4
        assert all(t == 16 for t in stats["tokens_per_layer"])
        assert stats["overall_compression_ratio"] > 3.0
        assert stats["total_original_bits"] > stats["total_compressed_bits"]


class TestQuashKVCachePackedMode:
    """Tests that QuashKVCache works correctly when engines use pack_storage."""

    @pytest.fixture
    def packed_cache(self):
        cache = QuashKVCache(num_layers=2, head_dim=64)
        for eng in cache.engines:
            eng.pack_storage = True
        return cache

    def test_update_packed_shapes(self, packed_cache):
        B, H, S, D = 1, 2, 16, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        full_k, full_v = packed_cache.update(k, v, layer_idx=0)
        assert full_k.shape == (B, H, S, D)
        assert full_v.shape == (B, H, S, D)

    def test_packed_saves_memory(self, packed_cache):
        """Packed storage should use fewer bytes than unpacked."""
        B, H, S, D = 1, 2, 32, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Unpacked engine for comparison
        unpacked = QuashKVCache(num_layers=1, head_dim=64)
        unpacked.update(k, v, layer_idx=0)
        unpacked_bytes = unpacked.engines[0].actual_memory_bytes()

        packed_cache.update(k, v, layer_idx=0)
        packed_bytes = packed_cache.engines[0].actual_memory_bytes()

        assert packed_bytes < unpacked_bytes, (
            f"Packed ({packed_bytes}) should be < unpacked ({unpacked_bytes})"
        )

    def test_packed_getitem_roundtrip(self, packed_cache):
        B, H, S, D = 1, 2, 16, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        packed_cache.update(k, v, layer_idx=0)
        got_k, got_v = packed_cache[0]
        assert got_k.shape == (B, H, S, D)
        assert got_v.shape == (B, H, S, D)

    def test_packed_quality(self):
        """Packed path should produce same quality as unpacked."""
        torch.manual_seed(77)
        B, H, S, D = 1, 1, 32, 64
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        unpacked = QuashKVCache(num_layers=1, head_dim=D, config=QuashKVCacheConfig(seed=42))
        packed = QuashKVCache(num_layers=1, head_dim=D, config=QuashKVCacheConfig(seed=42))
        for eng in packed.engines:
            eng.pack_storage = True

        uk, uv = unpacked.update(k.clone(), v.clone(), layer_idx=0)
        pk, pv = packed.update(k.clone(), v.clone(), layer_idx=0)

        # Packed and unpacked should give identical results (pack/unpack is lossless)
        assert torch.allclose(uk, pk, atol=1e-5), "Packed keys differ from unpacked"
        assert torch.allclose(uv, pv, atol=1e-5), "Packed values differ from unpacked"
