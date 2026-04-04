"""
Lloyd-Max optimal scalar quantizer for quashkv.

After random rotation, each coordinate of a unit-norm vector follows a scaled
Beta distribution that converges to N(0, 1/d) in high dimensions.  We solve
Lloyd-Max (iterative continuous 1-D k-means against this known PDF) to find
optimal centroids and decision boundaries.

The codebook is tiny (at most 16 entries for 4-bit) and computed once at init
time on CPU.  It is then reused for all quantize / dequantize calls.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from scipy import integrate, special

from .constants import (
    LLOYD_MAX_ITER,
    LLOYD_MAX_TOL,
    LLOYD_MAX_RANGE_SIGMAS,
    SUPPORTED_BITS,
)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float, sigma: float) -> float:
    """Standard Gaussian PDF with given sigma."""
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(
        -x * x / (2 * sigma * sigma)
    )


def _beta_pdf(x: float, d: int) -> float:
    """Exact Beta-type PDF for a coordinate of a random unit vector in R^d.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}
    for x in [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    log_coeff = (
        special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    if exponent == 0:
        return math.exp(log_coeff)
    return math.exp(log_coeff + exponent * math.log(1.0 - x * x))


# ---------------------------------------------------------------------------
# Lloyd-Max solver
# ---------------------------------------------------------------------------

def solve_lloyd_max(
    d: int,
    bits: int,
    *,
    use_exact_pdf: bool = False,
    max_iter: int = LLOYD_MAX_ITER,
    tol: float = LLOYD_MAX_TOL,
) -> tuple[list[float], list[float]]:
    """Solve continuous 1-D k-means (Lloyd-Max) for the coordinate PDF.

    Parameters
    ----------
    d : int
        Vector dimension. Determines sigma = 1/sqrt(d) for the Gaussian
        approximation (or the exact Beta PDF shape).
    bits : int
        Number of quantization bits.  Produces 2**bits centroids.
    use_exact_pdf : bool
        If True, use the exact Beta-type PDF from Lemma 1 of the paper.
        If False (default), use the Gaussian approximation N(0, 1/d), which
        is accurate for d >= 64.
    max_iter : int
        Maximum Lloyd-Max iterations.
    tol : float
        Convergence tolerance on max centroid shift.

    Returns
    -------
    centroids : list[float]
        Sorted optimal centroid values, length 2**bits.
    boundaries : list[float]
        Decision boundaries (midpoints), length 2**bits - 1.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be in {SUPPORTED_BITS}, got {bits}")

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)

    if use_exact_pdf:
        pdf = lambda x: _beta_pdf(x, d)
        lo, hi = -1.0, 1.0
    else:
        pdf = lambda x: _gaussian_pdf(x, sigma)
        lo, hi = -LLOYD_MAX_RANGE_SIGMAS * sigma, LLOYD_MAX_RANGE_SIGMAS * sigma

    # Initialize centroids uniformly in [lo, hi]
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for iteration in range(max_iter):
        # Boundaries = midpoints of consecutive centroids
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        # Voronoi cell edges: extend to ±∞ effectively
        edges = [lo * 5] + boundaries + [hi * 5]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b, limit=100)
            den, _ = integrate.quad(pdf, a, b, limit=100)
            if den > 1e-15:
                new_centroids.append(num / den)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    # Final boundaries
    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return centroids, boundaries


# ---------------------------------------------------------------------------
# Precomputed MSE cost (for verification against paper Table in Theorem 1)
# ---------------------------------------------------------------------------

def compute_mse_cost(
    d: int, bits: int, *, use_exact_pdf: bool = False
) -> float:
    """Compute the per-coordinate MSE cost C(f_X, b) for the optimal quantizer.

    Returns d * C(f_X, b), which equals D_mse for unit-norm vectors.
    """
    centroids, boundaries = solve_lloyd_max(d, bits, use_exact_pdf=use_exact_pdf)
    sigma = 1.0 / math.sqrt(d)

    if use_exact_pdf:
        pdf = lambda x: _beta_pdf(x, d)
        lo, hi = -1.0, 1.0
    else:
        pdf = lambda x: _gaussian_pdf(x, sigma)
        lo = -LLOYD_MAX_RANGE_SIGMAS * sigma
        hi = LLOYD_MAX_RANGE_SIGMAS * sigma

    n_levels = 1 << bits
    edges = [lo * 5] + boundaries + [hi * 5]

    total_cost = 0.0
    for i in range(n_levels):
        a, b_edge = edges[i], edges[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b_edge, limit=100)
        total_cost += val

    return d * total_cost  # D_mse = d * C(f_X, b)


# ---------------------------------------------------------------------------
# Codebook class
# ---------------------------------------------------------------------------

class LloydMaxCodebook:
    """Pre-solved Lloyd-Max codebook for a given (dimension, bits) pair.

    Attributes
    ----------
    d : int
        Dimension the codebook was computed for.
    bits : int
        Quantization bit-width.
    n_levels : int
        Number of quantization levels (2**bits).
    centroids : torch.Tensor
        Sorted centroid values, shape (n_levels,), float32.
    boundaries : torch.Tensor
        Decision boundaries, shape (n_levels - 1,), float32.
    """

    def __init__(self, d: int, bits: int, *, use_exact_pdf: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits

        centroids_list, boundaries_list = solve_lloyd_max(
            d, bits, use_exact_pdf=use_exact_pdf
        )
        self.centroids = torch.tensor(centroids_list, dtype=torch.float32)
        self.boundaries = torch.tensor(boundaries_list, dtype=torch.float32)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map each value in x to the index of its nearest centroid.

        Parameters
        ----------
        x : torch.Tensor
            Arbitrary shape, values should be in the range covered by centroids.

        Returns
        -------
        indices : torch.Tensor (same shape, dtype=torch.uint8)
        """
        centroids = self.centroids.to(x.device)
        # x: (...,)  centroids: (n_levels,)
        diffs = x.unsqueeze(-1) - centroids  # (..., n_levels)
        return diffs.abs().argmin(dim=-1).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up centroid values for given indices.

        Parameters
        ----------
        indices : torch.Tensor (uint8 or long)

        Returns
        -------
        values : torch.Tensor (float32, same shape as indices)
        """
        return self.centroids.to(indices.device)[indices.long()]

    def quantize_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize using boundary comparisons (faster, matches GPU kernel logic).

        Instead of computing distance to all centroids, compare against sorted
        boundaries.  This is O(n_levels) comparisons via chained conditionals,
        which maps well to GPU predication.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        indices : torch.Tensor (uint8)
        """
        bounds = self.boundaries.to(x.device)
        # Start with index 0, increment for each boundary exceeded
        idx = torch.zeros_like(x, dtype=torch.int32)
        for i in range(len(bounds)):
            idx = idx + (x > bounds[i]).int()
        return idx.to(torch.uint8)

    def mse_cost(self) -> float:
        """Compute the theoretical D_mse for this codebook (unit-norm vectors)."""
        return compute_mse_cost(self.d, self.bits)

    def to(self, device: str | torch.device) -> "LloydMaxCodebook":
        """Move tensors to a device (returns self for chaining)."""
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"LloydMaxCodebook(d={self.d}, bits={self.bits}, "
            f"levels={self.n_levels}, "
            f"centroids=[{', '.join(f'{c:.6f}' for c in self.centroids.tolist())}])"
        )


# ---------------------------------------------------------------------------
# Convenience: precompute codebooks for common configs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def get_codebook(d: int, bits: int) -> LloydMaxCodebook:
    """Cached codebook factory."""
    return LloydMaxCodebook(d, bits)
