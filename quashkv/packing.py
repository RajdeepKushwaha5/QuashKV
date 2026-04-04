"""
Bit-packing utilities for quashkv.

Quantized indices are 1-4 bits each, but stored as uint8 by the codebook.
These utilities pack multiple indices into single bytes for actual memory
savings, and unpack them for decompression.

Packing layout (little-endian within each byte):
  1-bit: 8 values per byte
  2-bit: 4 values per byte
  3-bit: 8 values per 3 bytes (24-bit groups)
  4-bit: 2 values per byte

The last dimension is the one that gets packed.
"""

from __future__ import annotations

import torch


def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low-bit indices into bytes.

    Parameters
    ----------
    indices : torch.Tensor (uint8), shape (..., d)
        Values in range [0, 2**bits - 1].
    bits : int
        Bit width (1, 2, 3, or 4).

    Returns
    -------
    packed : torch.Tensor (uint8), shape (..., ceil(d * bits / 8))
    """
    if bits == 8:
        return indices

    *leading, d = indices.shape
    flat = indices.reshape(-1, d)
    n_rows = flat.shape[0]

    if bits == 1:
        # 8 values per byte
        pad = (8 - d % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        reshaped = flat.reshape(n_rows, -1, 8)
        shifts = torch.arange(8, device=indices.device, dtype=torch.uint8)
        packed = (reshaped << shifts).sum(dim=-1).to(torch.uint8)

    elif bits == 2:
        # 4 values per byte
        pad = (4 - d % 4) % 4
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        reshaped = flat.reshape(n_rows, -1, 4)
        shifts = torch.arange(0, 8, 2, device=indices.device, dtype=torch.uint8)
        packed = (reshaped << shifts).sum(dim=-1).to(torch.uint8)

    elif bits == 3:
        # 8 values → 3 bytes (24 bits)
        pad = (8 - d % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        d_padded = flat.shape[-1]
        n_groups = d_padded // 8
        reshaped = flat.reshape(n_rows, n_groups, 8).to(torch.int32)

        # Pack 8 x 3-bit values into 24 bits then split into 3 bytes
        packed_list = []
        for g in range(n_groups):
            vals = reshaped[:, g, :]  # (n_rows, 8)
            word = torch.zeros(n_rows, dtype=torch.int32, device=indices.device)
            for i in range(8):
                word = word | (vals[:, i] << (i * 3))
            # Split 24-bit word into 3 bytes
            packed_list.append((word & 0xFF).to(torch.uint8))
            packed_list.append(((word >> 8) & 0xFF).to(torch.uint8))
            packed_list.append(((word >> 16) & 0xFF).to(torch.uint8))
        packed = torch.stack(packed_list, dim=-1)

    elif bits == 4:
        # 2 values per byte
        pad = d % 2
        if pad:
            flat = torch.nn.functional.pad(flat, (0, 1))
        reshaped = flat.reshape(n_rows, -1, 2)
        packed = (reshaped[:, :, 0] | (reshaped[:, :, 1] << 4)).to(torch.uint8)

    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return packed.reshape(*leading, -1)


def unpack_bits(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bytes back into low-bit indices.

    Parameters
    ----------
    packed : torch.Tensor (uint8), shape (..., packed_d)
    bits : int
        Original bit width.
    d : int
        Original last dimension size (before packing).

    Returns
    -------
    indices : torch.Tensor (uint8), shape (..., d)
    """
    if bits == 8:
        return packed

    *leading, packed_d = packed.shape
    flat = packed.reshape(-1, packed_d)
    n_rows = flat.shape[0]
    mask = (1 << bits) - 1

    if bits == 1:
        unpacked = []
        for byte_idx in range(packed_d):
            byte_val = flat[:, byte_idx].to(torch.int32)
            for bit in range(8):
                unpacked.append(((byte_val >> bit) & 1).to(torch.uint8))
        result = torch.stack(unpacked, dim=-1)[:, :d]

    elif bits == 2:
        unpacked = []
        for byte_idx in range(packed_d):
            byte_val = flat[:, byte_idx].to(torch.int32)
            for shift in range(0, 8, 2):
                unpacked.append(((byte_val >> shift) & mask).to(torch.uint8))
        result = torch.stack(unpacked, dim=-1)[:, :d]

    elif bits == 3:
        n_groups = packed_d // 3
        result_list = []
        for g in range(n_groups):
            b0 = flat[:, g * 3].to(torch.int32)
            b1 = flat[:, g * 3 + 1].to(torch.int32)
            b2 = flat[:, g * 3 + 2].to(torch.int32)
            word = b0 | (b1 << 8) | (b2 << 16)
            for i in range(8):
                result_list.append(((word >> (i * 3)) & mask).to(torch.uint8))
        result = torch.stack(result_list, dim=-1)[:, :d]

    elif bits == 4:
        unpacked = []
        for byte_idx in range(packed_d):
            byte_val = flat[:, byte_idx].to(torch.int32)
            unpacked.append((byte_val & mask).to(torch.uint8))
            unpacked.append(((byte_val >> 4) & mask).to(torch.uint8))
        result = torch.stack(unpacked, dim=-1)[:, :d]

    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return result.reshape(*leading, d)


def packed_size(d: int, bits: int) -> int:
    """Compute the packed byte size for d elements at given bit width."""
    total_bits = d * bits
    return (total_bits + 7) // 8
