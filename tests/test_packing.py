"""Tests for bit-packing utilities."""

import pytest
import torch

from quashkv.packing import pack_bits, unpack_bits, packed_size


class TestPackUnpack:
    """Verify pack/unpack roundtrip for all supported bit widths."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_roundtrip_exact(self, bits: int):
        d = 128
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (16, d), dtype=torch.uint8)
        packed = pack_bits(indices, bits)
        unpacked = unpack_bits(packed, bits, d)
        assert torch.equal(indices, unpacked)

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_roundtrip_batched(self, bits: int):
        d = 64
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (2, 4, 32, d), dtype=torch.uint8)
        packed = pack_bits(indices, bits)
        unpacked = unpack_bits(packed, bits, d)
        assert torch.equal(indices, unpacked)

    @pytest.mark.parametrize("bits,expected_ratio", [
        (1, 8),   # 8x compression
        (2, 4),   # 4x compression
        (4, 2),   # 2x compression
    ])
    def test_packed_size_reduction(self, bits: int, expected_ratio: int):
        d = 128
        indices = torch.randint(0, (1 << bits), (8, d), dtype=torch.uint8)
        packed = pack_bits(indices, bits)
        assert packed.shape[-1] == d // expected_ratio

    def test_3bit_packed_size(self):
        d = 128  # 128 * 3 = 384 bits = 48 bytes
        indices = torch.randint(0, 8, (4, d), dtype=torch.uint8)
        packed = pack_bits(indices, 3)
        assert packed.shape[-1] == 48

    def test_packed_size_function(self):
        assert packed_size(128, 1) == 16
        assert packed_size(128, 2) == 32
        assert packed_size(128, 3) == 48
        assert packed_size(128, 4) == 64

    @pytest.mark.parametrize("d", [63, 65, 100, 127, 129])
    def test_non_aligned_dimensions(self, d: int):
        """Test with dimensions that don't align perfectly to byte boundaries."""
        for bits in [1, 2, 3, 4]:
            max_val = (1 << bits) - 1
            indices = torch.randint(0, max_val + 1, (4, d), dtype=torch.uint8)
            packed = pack_bits(indices, bits)
            unpacked = unpack_bits(packed, bits, d)
            assert torch.equal(indices, unpacked), f"Failed for d={d}, bits={bits}"

    def test_8bit_passthrough(self):
        indices = torch.randint(0, 256, (4, 128), dtype=torch.uint8)
        packed = pack_bits(indices, 8)
        assert torch.equal(packed, indices)
        unpacked = unpack_bits(packed, 8, 128)
        assert torch.equal(unpacked, indices)
