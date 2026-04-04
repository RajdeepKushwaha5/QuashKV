"""
Triton kernel availability detection and dispatch utilities.

Provides a clean auto-fallback: Triton kernels when available,
PyTorch reference implementations otherwise.
"""

from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from .compress import fused_compress_mse, fused_compress_ip
from .decompress import fused_decompress_mse, fused_decompress_ip
from .attention import fused_quantized_attention

__all__ = [
    "HAS_TRITON",
    "fused_compress_mse",
    "fused_compress_ip",
    "fused_decompress_mse",
    "fused_decompress_ip",
    "fused_quantized_attention",
]
