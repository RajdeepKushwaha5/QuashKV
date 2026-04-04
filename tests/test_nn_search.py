"""Tests for the QuashIndex nearest-neighbor search module."""

import pytest
import torch

from quashkv.nn_search.index import QuashIndex


class TestQuashIndexInit:
    def test_defaults(self):
        idx = QuashIndex(dim=64)
        assert idx.dim == 64
        assert idx.bits == 3
        assert idx.n_vectors == 0
        assert len(idx) == 0

    def test_custom(self):
        idx = QuashIndex(dim=128, bits=4, seed=99, pack_storage=True)
        assert idx.dim == 128
        assert idx.bits == 4
        assert idx.pack_storage is True

    def test_repr(self):
        idx = QuashIndex(dim=64)
        r = repr(idx)
        assert "dim=64" in r
        assert "n_vectors=0" in r


class TestQuashIndexAdd:
    def test_add_single_batch(self):
        idx = QuashIndex(dim=64, seed=42)
        vecs = torch.randn(100, 64)
        idx.add(vecs)
        assert idx.n_vectors == 100

    def test_add_incremental(self):
        idx = QuashIndex(dim=64, seed=42)
        idx.add(torch.randn(50, 64))
        idx.add(torch.randn(30, 64))
        assert idx.n_vectors == 80

    def test_add_wrong_dim_raises(self):
        idx = QuashIndex(dim=64)
        with pytest.raises(ValueError, match="Expected"):
            idx.add(torch.randn(10, 32))

    def test_add_wrong_ndim_raises(self):
        idx = QuashIndex(dim=64)
        with pytest.raises(ValueError, match="Expected"):
            idx.add(torch.randn(64))  # 1D

    def test_add_packed(self):
        idx = QuashIndex(dim=64, seed=42, pack_storage=True)
        idx.add(torch.randn(100, 64))
        assert idx.n_vectors == 100

    def test_reset(self):
        idx = QuashIndex(dim=64, seed=42)
        idx.add(torch.randn(50, 64))
        idx.reset()
        assert idx.n_vectors == 0


class TestQuashIndexSearch:
    @pytest.fixture
    def index_and_db(self):
        """Create an index and return (index, database, queries)."""
        torch.manual_seed(42)
        dim = 64
        n_db = 500
        n_queries = 10

        db = torch.randn(n_db, dim)
        queries = torch.randn(n_queries, dim)

        idx = QuashIndex(dim=dim, bits=3, seed=42)
        idx.add(db)
        return idx, db, queries

    def test_search_shapes(self, index_and_db):
        idx, db, queries = index_and_db
        scores, indices = idx.search(queries, k=5)
        assert scores.shape == (10, 5)
        assert indices.shape == (10, 5)

    def test_search_k_clamped(self, index_and_db):
        idx, db, queries = index_and_db
        # k larger than n_vectors → clamp
        scores, indices = idx.search(queries, k=9999)
        assert scores.shape == (10, 500)

    def test_search_empty_raises(self):
        idx = QuashIndex(dim=64)
        with pytest.raises(ValueError, match="empty"):
            idx.search(torch.randn(5, 64))

    def test_search_scores_descending(self, index_and_db):
        idx, db, queries = index_and_db
        scores, _ = idx.search(queries, k=10)
        for i in range(scores.shape[0]):
            diffs = scores[i, :-1] - scores[i, 1:]
            assert (diffs >= -1e-6).all(), "Scores not descending"

    def test_search_indices_valid(self, index_and_db):
        idx, db, queries = index_and_db
        _, indices = idx.search(queries, k=10)
        assert (indices >= 0).all()
        assert (indices < 500).all()


class TestQuashIndexRecall:
    def test_recall_reasonable(self):
        """Approximate search should have non-trivial recall at 3 bits."""
        torch.manual_seed(123)
        dim = 64
        n_db = 1000
        db = torch.randn(n_db, dim)
        queries = torch.randn(20, dim)

        idx = QuashIndex(dim=dim, bits=3, seed=42)
        idx.add(db)

        recall = idx.recall_at_k(queries, db, k=10)
        # At 3-bit with d=64, expect decent recall (>30%)
        assert recall > 0.3, f"Recall@10 = {recall:.3f} too low"

    def test_higher_bits_better_recall(self):
        """4-bit index should have higher recall than 3-bit."""
        torch.manual_seed(77)
        dim = 64
        n_db = 500
        db = torch.randn(n_db, dim)
        queries = torch.randn(20, dim)

        idx3 = QuashIndex(dim=dim, bits=3, seed=42)
        idx3.add(db)
        r3 = idx3.recall_at_k(queries, db, k=10)

        idx4 = QuashIndex(dim=dim, bits=4, seed=42)
        idx4.add(db)
        r4 = idx4.recall_at_k(queries, db, k=10)

        assert r4 >= r3, f"4-bit recall ({r4:.3f}) should be >= 3-bit ({r3:.3f})"


class TestQuashIndexBruteForce:
    def test_brute_force_shapes(self):
        db = torch.randn(100, 64)
        queries = torch.randn(5, 64)
        scores, indices = QuashIndex.brute_force(queries, db, k=10)
        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_brute_force_correct(self):
        """Top-1 brute force should match manual computation."""
        torch.manual_seed(0)
        db = torch.randn(50, 32)
        q = torch.randn(1, 32)
        _, idx = QuashIndex.brute_force(q, db, k=1)
        # Manual
        expected = (q @ db.T).argmax(dim=1)
        assert idx[0, 0] == expected[0]


class TestQuashIndexPacked:
    def test_packed_search_matches_unpacked(self):
        """Pack/unpack is lossless, so results should match exactly."""
        torch.manual_seed(42)
        dim = 64
        db = torch.randn(200, dim)
        queries = torch.randn(5, dim)

        unpacked = QuashIndex(dim=dim, bits=3, seed=42, pack_storage=False)
        unpacked.add(db)

        packed = QuashIndex(dim=dim, bits=3, seed=42, pack_storage=True)
        packed.add(db)

        s_u, i_u = unpacked.search(queries, k=10)
        s_p, i_p = packed.search(queries, k=10)

        assert torch.equal(i_u, i_p), "Packed and unpacked return different indices"
        assert torch.allclose(s_u, s_p, atol=1e-4), "Packed and unpacked scores differ"

    def test_packed_saves_memory(self):
        dim = 64
        db = torch.randn(200, dim)

        unpacked = QuashIndex(dim=dim, bits=3, seed=42, pack_storage=False)
        unpacked.add(db)

        packed = QuashIndex(dim=dim, bits=3, seed=42, pack_storage=True)
        packed.add(db)

        assert packed.memory_bytes() < unpacked.memory_bytes()

    def test_packed_compression_ratio(self):
        dim = 128
        db = torch.randn(1000, dim)

        idx = QuashIndex(dim=dim, bits=3, seed=42, pack_storage=True)
        idx.add(db)

        ratio = idx.compression_ratio(torch.float32)
        # float32 = 4 bytes/elem, 3-bit packed ≈ 0.375 bytes/elem + norms
        assert ratio > 3.0, f"Expected ratio > 3, got {ratio:.2f}"


class TestQuashIndexMemory:
    def test_memory_bytes_empty(self):
        idx = QuashIndex(dim=64)
        assert idx.memory_bytes() == 0

    def test_memory_bytes_nonempty(self):
        idx = QuashIndex(dim=64, seed=42)
        idx.add(torch.randn(100, 64))
        assert idx.memory_bytes() > 0

    def test_compression_ratio_empty(self):
        idx = QuashIndex(dim=64)
        assert idx.compression_ratio() == 1.0

    def test_compression_ratio_nonempty(self):
        idx = QuashIndex(dim=64, seed=42)
        idx.add(torch.randn(100, 64))
        ratio = idx.compression_ratio()
        # uint8 indices (1 byte each) vs float32 (4 bytes) → ratio ~2x
        # (without packing; with packing it's higher)
        assert ratio > 1.0
