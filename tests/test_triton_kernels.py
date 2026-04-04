"""Tests for fused Triton kernel fallbacks (PyTorch reference implementations)."""

import math
import pytest
import torch

from quashkv.codebook import LloydMaxCodebook
from quashkv.quantizer import MSEQuantizer, InnerProductQuantizer
from quashkv.engine import QuashKVEngine
from quashkv.triton_kernels.compress import fused_compress_mse, fused_compress_ip
from quashkv.triton_kernels.decompress import fused_decompress_mse, fused_decompress_ip
from quashkv.triton_kernels.attention import (
    fused_quantized_attention,
    CompressedBlock,
)


class TestFusedCompressMSE:
    """Verify fused_compress_mse matches MSEQuantizer.compress."""

    @pytest.fixture
    def setup(self):
        d, bits, seed = 64, 3, 42
        q = MSEQuantizer(d=d, bits=bits, seed=seed)
        return q, q.Pi, q.codebook.boundaries

    def test_shapes(self, setup):
        q, Pi, bounds = setup
        x = torch.randn(2, 4, 16, 64)
        indices, norms = fused_compress_mse(x, Pi, bounds)
        assert indices.shape == (2, 4, 16, 64)
        assert norms.shape == (2, 4, 16)
        assert indices.dtype == torch.uint8

    def test_matches_quantizer(self, setup):
        """Fused compression should produce identical output to the quantizer."""
        q, Pi, bounds = setup
        torch.manual_seed(0)
        x = torch.randn(1, 2, 32, 64)

        idx_fused, norms_fused = fused_compress_mse(x, Pi, bounds)
        idx_ref, norms_ref = q.compress(x)

        # Norms should be identical
        assert torch.allclose(norms_fused, norms_ref, atol=1e-6)
        # Indices should match (bucketize vs boundary loop may differ at exact boundaries,
        # but should match for random data with very high probability)
        agreement = (idx_fused == idx_ref).float().mean().item()
        assert agreement > 0.99, f"Index agreement {agreement:.4f} too low"

    def test_2d_input(self, setup):
        """Should work on (N, D) vectors for NN search."""
        q, Pi, bounds = setup
        x = torch.randn(100, 64)
        indices, norms = fused_compress_mse(x, Pi, bounds)
        assert indices.shape == (100, 64)
        assert norms.shape == (100,)


class TestFusedCompressIP:
    """Verify fused_compress_ip matches InnerProductQuantizer.compress."""

    @pytest.fixture
    def setup(self):
        d, bits, seed = 64, 3, 42
        q = InnerProductQuantizer(d=d, total_bits=bits, seed=seed)
        return q

    def test_shapes(self, setup):
        q = setup
        x = torch.randn(1, 2, 16, 64)
        mse_idx, qjl, norms = fused_compress_ip(
            x, q.mse_quantizer.Pi, q.mse_quantizer.codebook.boundaries,
            q.S, q.mse_quantizer.codebook.centroids,
        )
        assert mse_idx.shape == (1, 2, 16, 64)
        assert qjl.shape == (1, 2, 16, 64)
        assert norms.shape == (1, 2, 16)

    def test_matches_quantizer(self, setup):
        q = setup
        torch.manual_seed(0)
        x = torch.randn(1, 2, 32, 64)

        mse_fused, qjl_fused, norms_fused = fused_compress_ip(
            x, q.mse_quantizer.Pi, q.mse_quantizer.codebook.boundaries,
            q.S, q.mse_quantizer.codebook.centroids,
        )
        mse_ref, qjl_ref, norms_ref = q.compress(x)

        assert torch.allclose(norms_fused, norms_ref, atol=1e-6)
        mse_agree = (mse_fused == mse_ref).float().mean().item()
        qjl_agree = (qjl_fused == qjl_ref).float().mean().item()
        assert mse_agree > 0.99, f"MSE index agreement {mse_agree:.4f} too low"
        assert qjl_agree > 0.99, f"QJL sign agreement {qjl_agree:.4f} too low"


class TestFusedDecompressMSE:
    def test_roundtrip(self):
        """Compress then decompress should approximate the original."""
        torch.manual_seed(42)
        d, bits = 64, 3
        q = MSEQuantizer(d=d, bits=bits, seed=42)
        x = torch.randn(1, 2, 16, d)

        indices, norms = q.compress(x)
        x_hat = fused_decompress_mse(
            indices, norms, q.Pi, q.codebook.centroids,
        )
        assert x_hat.shape == x.shape

        # Same as MSEQuantizer.decompress
        x_ref = q.decompress(indices, norms)
        assert torch.allclose(x_hat, x_ref, atol=1e-5)

    def test_shapes(self):
        d = 64
        codebook = LloydMaxCodebook(d, bits=3)
        Pi = torch.eye(d)
        indices = torch.zeros(10, d, dtype=torch.uint8)
        norms = torch.ones(10)
        x_hat = fused_decompress_mse(indices, norms, Pi, codebook.centroids)
        assert x_hat.shape == (10, d)


class TestFusedDecompressIP:
    def test_shapes(self):
        d = 64
        q = InnerProductQuantizer(d=d, total_bits=3, seed=42)
        x = torch.randn(1, 2, 16, d)
        mse_idx, qjl, norms = q.compress(x)
        x_hat = fused_decompress_ip(
            mse_idx, qjl, norms,
            q.mse_quantizer.Pi,
            q.mse_quantizer.codebook.centroids,
            q.S,
            q.qjl_scale,
        )
        assert x_hat.shape == x.shape

    def test_quality(self):
        """IP decompress should produce vectors close to originals."""
        torch.manual_seed(99)
        d = 64
        q = InnerProductQuantizer(d=d, total_bits=3, seed=42)
        x = torch.randn(1, 1, 32, d)
        mse_idx, qjl, norms = q.compress(x)
        x_hat = fused_decompress_ip(
            mse_idx, qjl, norms,
            q.mse_quantizer.Pi,
            q.mse_quantizer.codebook.centroids,
            q.S,
            q.qjl_scale,
        )

        cosine = torch.nn.functional.cosine_similarity(
            x.flatten(), x_hat.flatten(), dim=0
        ).item()
        assert cosine > 0.7, f"Cosine {cosine} too low"


class TestFusedQuantizedAttention:
    """Test fused attention against the engine's naive implementation."""

    def _make_blocks(self, engine):
        """Convert engine cache to CompressedBlock list for the fused kernel."""
        blocks = []
        for blk in engine._cache:
            mse, qjl, val = engine._unpack_block(blk)
            blocks.append(CompressedBlock(
                mse_indices=mse,
                qjl_signs=qjl,
                key_norms=blk.key_norms,
                val_indices=val,
                val_norms=blk.val_norms,
                seq_len=blk.seq_len,
            ))
        return blocks

    def test_matches_engine_attention(self):
        """Fused attention should produce same output as engine.attention."""
        torch.manual_seed(42)
        B, H, KV, Q, D = 1, 2, 32, 4, 64

        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        keys = torch.randn(B, H, KV, D)
        values = torch.randn(B, H, KV, D)
        queries = torch.randn(B, H, Q, D)

        engine.append(keys, values)
        engine_out = engine.attention(queries)

        blocks = self._make_blocks(engine)
        fused_out = fused_quantized_attention(
            queries=queries,
            blocks=blocks,
            Pi=engine.key_quantizer.mse_quantizer.Pi,
            key_centroids=engine.key_quantizer.mse_quantizer.codebook.centroids,
            val_centroids=engine.value_quantizer.codebook.centroids,
            S=engine.key_quantizer.S,
            qjl_scale=engine.key_quantizer.qjl_scale,
            total_kv_tokens=engine.total_tokens,
        )

        assert fused_out.shape == engine_out.shape
        # Online softmax and naive softmax should give same result
        assert torch.allclose(fused_out, engine_out, atol=1e-4), (
            f"Max diff: {(fused_out - engine_out).abs().max():.6f}"
        )

    def test_multi_block(self):
        """Fused attention should handle multiple KV blocks."""
        torch.manual_seed(77)
        B, H, D = 1, 2, 64
        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        engine.append(torch.randn(B, H, 16, D), torch.randn(B, H, 16, D))
        engine.append(torch.randn(B, H, 8, D), torch.randn(B, H, 8, D))

        queries = torch.randn(B, H, 4, D)
        engine_out = engine.attention(queries)

        blocks = self._make_blocks(engine)
        fused_out = fused_quantized_attention(
            queries=queries,
            blocks=blocks,
            Pi=engine.key_quantizer.mse_quantizer.Pi,
            key_centroids=engine.key_quantizer.mse_quantizer.codebook.centroids,
            val_centroids=engine.value_quantizer.codebook.centroids,
            S=engine.key_quantizer.S,
            qjl_scale=engine.key_quantizer.qjl_scale,
            total_kv_tokens=engine.total_tokens,
        )

        assert torch.allclose(fused_out, engine_out, atol=1e-4), (
            f"Max diff: {(fused_out - engine_out).abs().max():.6f}"
        )

    def test_output_shape(self):
        torch.manual_seed(42)
        B, H, KV, Q, D = 2, 4, 16, 8, 64
        engine = QuashKVEngine(head_dim=D, key_bits=3, value_bits=3, seed=42)
        engine.append(torch.randn(B, H, KV, D), torch.randn(B, H, KV, D))

        queries = torch.randn(B, H, Q, D)
        blocks = self._make_blocks(engine)
        out = fused_quantized_attention(
            queries=queries,
            blocks=blocks,
            Pi=engine.key_quantizer.mse_quantizer.Pi,
            key_centroids=engine.key_quantizer.mse_quantizer.codebook.centroids,
            val_centroids=engine.value_quantizer.codebook.centroids,
            S=engine.key_quantizer.S,
            qjl_scale=engine.key_quantizer.qjl_scale,
            total_kv_tokens=engine.total_tokens,
        )
        assert out.shape == (B, H, Q, D)
