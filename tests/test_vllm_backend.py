"""Tests for the vLLM integration module (QuashKVPageManager, QuashKVModelManager).

These tests exercise the compression/attention logic WITHOUT requiring
vLLM to be installed — they test the standalone components.
"""

import math
import pytest
import torch

from quashkv.integrations.vllm_backend import (
    QuashKVPageConfig,
    QuashKVPageManager,
    QuashKVModelManager,
    quashkv_attention_forward,
    register_quashkv_backend,
    HAS_VLLM,
)


class TestQuashKVPageConfig:
    def test_defaults(self):
        cfg = QuashKVPageConfig()
        assert cfg.key_bits == 3
        assert cfg.value_bits == 3
        assert cfg.block_size == 16
        assert cfg.seed == 42

    def test_custom(self):
        cfg = QuashKVPageConfig(key_bits=4, value_bits=4, block_size=32, seed=7)
        assert cfg.key_bits == 4
        assert cfg.block_size == 32


class TestQuashKVPageManager:
    @pytest.fixture
    def pm(self):
        return QuashKVPageManager(
            num_kv_heads=4, head_size=64,
            config=QuashKVPageConfig(), device=torch.device("cpu"),
        )

    def test_init(self, pm):
        assert pm.total_tokens == 0
        assert pm.num_kv_heads == 4
        assert pm.head_size == 64

    def test_compress_and_store(self, pm):
        B, H, S, D = 1, 4, 16, 64
        pm.compress_and_store(torch.randn(B, H, S, D), torch.randn(B, H, S, D))
        assert pm.total_tokens == 16

    def test_compress_multiple(self, pm):
        B, H, D = 1, 4, 64
        pm.compress_and_store(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D))
        pm.compress_and_store(torch.randn(B, H, 4, D), torch.randn(B, H, 4, D))
        assert pm.total_tokens == 12

    def test_attention_shape(self, pm):
        B, H, D = 1, 4, 64
        pm.compress_and_store(torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))
        queries = torch.randn(B, H, 4, D)
        out = pm.attention(queries)
        assert out.shape == (B, H, 4, D)

    def test_attention_empty_raises(self, pm):
        with pytest.raises(ValueError, match="No compressed"):
            pm.attention(torch.randn(1, 4, 1, 64))

    def test_attention_quality(self):
        """Attention output should approximate exact attention."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 32, 4, 64
        scale = 1.0 / math.sqrt(D)

        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        # Exact attention
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys) * scale
        exact = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), values)

        # Compressed attention
        pm = QuashKVPageManager(
            num_kv_heads=H, head_size=D,
            config=QuashKVPageConfig(seed=42), device=torch.device("cpu"),
        )
        pm.compress_and_store(keys, values)
        compressed = pm.attention(queries)

        cosine = torch.nn.functional.cosine_similarity(
            exact.flatten(), compressed.flatten(), dim=0,
        ).item()
        assert cosine > 0.6, f"Cosine {cosine} too low"

    def test_clear(self, pm):
        pm.compress_and_store(torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
        pm.clear()
        assert pm.total_tokens == 0

    def test_compression_ratio(self, pm):
        pm.compress_and_store(torch.randn(1, 4, 16, 64), torch.randn(1, 4, 16, 64))
        ratio = pm.compression_ratio()
        assert ratio > 3.0

    def test_get_compressed_blocks(self, pm):
        pm.compress_and_store(torch.randn(1, 4, 16, 64), torch.randn(1, 4, 16, 64))
        blocks = pm.get_compressed_blocks()
        assert len(blocks) == 1
        assert blocks[0].seq_len == 16

    def test_layer_seed_offset(self):
        """Different layer_idx should produce different rotations."""
        pm0 = QuashKVPageManager(
            num_kv_heads=2, head_size=64,
            config=QuashKVPageConfig(seed=42), device=torch.device("cpu"),
            layer_idx=0,
        )
        pm1 = QuashKVPageManager(
            num_kv_heads=2, head_size=64,
            config=QuashKVPageConfig(seed=42), device=torch.device("cpu"),
            layer_idx=1,
        )
        assert pm0.engine.seed != pm1.engine.seed


class TestQuashKVAttentionForward:
    def test_basic(self):
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 16, 4, 64
        pm = QuashKVPageManager(
            num_kv_heads=H, head_size=D,
            config=QuashKVPageConfig(seed=42), device=torch.device("cpu"),
        )
        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        out = quashkv_attention_forward(queries, keys, values, pm)
        assert out.shape == (B, H, Q, D)
        assert pm.total_tokens == KV


class TestQuashKVModelManager:
    @pytest.fixture
    def manager(self):
        return QuashKVModelManager(
            num_layers=4, num_kv_heads=8, head_size=64,
        )

    def test_init(self, manager):
        assert manager.num_layers == 4
        assert len(manager.layers) == 4

    def test_forward_layer(self, manager):
        B, H, KV, Q, D = 1, 8, 16, 1, 64
        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        out = manager.forward_layer(0, queries, keys, values)
        assert out.shape == (B, H, Q, D)

    def test_multi_layer_generation(self, manager):
        """Simulate token-by-token generation across all layers."""
        B, H, D = 1, 8, 64

        # Prefill
        for layer in range(4):
            k = torch.randn(B, H, 32, D)
            v = torch.randn(B, H, 32, D)
            q = torch.randn(B, H, 32, D)
            manager.forward_layer(layer, q, k, v)

        # Decode 5 tokens
        for step in range(5):
            for layer in range(4):
                k = torch.randn(B, H, 1, D)
                v = torch.randn(B, H, 1, D)
                q = torch.randn(B, H, 1, D)
                out = manager.forward_layer(layer, q, k, v)
                assert out.shape == (B, H, 1, D)

        stats = manager.stats()
        assert stats["tokens_per_layer"] == [37, 37, 37, 37]

    def test_clear(self, manager):
        B, H, D = 1, 8, 64
        for layer in range(4):
            manager.forward_layer(
                layer,
                torch.randn(B, H, 4, D),
                torch.randn(B, H, 16, D),
                torch.randn(B, H, 16, D),
            )
        manager.clear()
        for layer in manager.layers:
            assert layer.total_tokens == 0

    def test_stats(self, manager):
        stats = manager.stats()
        assert stats["num_layers"] == 4
        assert stats["config"]["key_bits"] == 3

    def test_total_compressed_bytes(self, manager):
        B, H, D = 1, 8, 64
        manager.forward_layer(
            0, torch.randn(B, H, 4, D),
            torch.randn(B, H, 16, D),
            torch.randn(B, H, 16, D),
        )
        assert manager.total_compressed_bytes() > 0

    def test_custom_config(self):
        cfg = QuashKVPageConfig(key_bits=4, value_bits=4, seed=99)
        m = QuashKVModelManager(
            num_layers=2, num_kv_heads=4, head_size=128, config=cfg,
        )
        assert m.layers[0].engine.key_bits == 4
        assert m.layers[0].engine.value_bits == 4


class TestRegisterBackend:
    def test_no_vllm_raises(self):
        """Without vLLM installed, register should raise."""
        if HAS_VLLM:
            pytest.skip("vLLM is installed")
        with pytest.raises(ImportError, match="vLLM is not installed"):
            register_quashkv_backend()
