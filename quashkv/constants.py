"""Shared constants and configuration for quashkv."""

import math

# ---------------------------------------------------------------------------
# Tile / block sizes (tuned for Triton on A100/H100; multiples of 16 for MMA)
# ---------------------------------------------------------------------------
BLOCK_Q = 16        # Query tile rows per thread block in attention
BLOCK_KV = 64       # KV block size in attention loop
BLOCK_S = 64        # Block size for compress / decompress kernels

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
SUPPORTED_BITS = {1, 2, 3, 4}
DEFAULT_TOTAL_BITS = 3      # 2-bit MSE + 1-bit QJL for keys, 3-bit MSE for values
DEFAULT_SEED = 42

# Correction scale for QJL inner product estimator: sqrt(pi/2) / d
# This is set per-engine since it depends on head_dim.
QJL_COEFF = math.sqrt(math.pi / 2)

# Lloyd-Max solver parameters
LLOYD_MAX_ITER = 300
LLOYD_MAX_TOL = 1e-12
LLOYD_MAX_RANGE_SIGMAS = 4.0    # search range = ±(this * sigma) of Gaussian
