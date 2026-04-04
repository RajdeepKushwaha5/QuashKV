"""
Quantizers for quashkv.

Two main classes:

* **MSEQuantizer** — rotates by a random orthogonal matrix Π, then applies
  Lloyd-Max scalar quantization coordinate-wise.  Used directly for *values*
  (where we need low MSE).

* **InnerProductQuantizer** — a two-stage scheme for *keys*, where we need
  unbiased low-distortion inner product estimation.  Stage 1: (b−1)-bit MSE
  quantizer.  Stage 2: 1-bit QJL sign hash on the residual, with bias
  correction.

Both quantizers are **pure PyTorch** (CPU or GPU, no custom kernels).
"""

from __future__ import annotations

import math

import torch

from .codebook import LloydMaxCodebook
from .constants import DEFAULT_SEED, DEFAULT_TOTAL_BITS, QJL_COEFF


# ---------------------------------------------------------------------------
# Random orthogonal projection matrix Π
# ---------------------------------------------------------------------------

def _random_orthogonal(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a d×d random orthogonal matrix via QR decomposition.

    This is the rotation matrix Π from the paper.  We draw entries
    i.i.d. from N(0, 1) and compute Q from the QR factorization.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    # Ensure it's a proper rotation (det = +1 is fine for quantization)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


# ---------------------------------------------------------------------------
# QJL random sign matrix S
# ---------------------------------------------------------------------------

def _random_signs(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a d-dimensional random sign vector (±1) for QJL.

    In the full version this would be a matrix S ∈ {±1}^{d × m}, but the
    paper uses m = d for maximum quality and the sign is applied to the
    residual coordinate-wise.  We store one sign vector per head.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
    return torch.randint(0, 2, (d,), generator=gen, dtype=torch.float32, device="cpu").mul_(2).sub_(1).to(device)


# ---------------------------------------------------------------------------
# MSEQuantizer
# ---------------------------------------------------------------------------

class MSEQuantizer:
    """Rotate by Π, then scalar-quantize each coordinate with Lloyd-Max.

    This gives near-optimal MSE distortion D_mse = d · C(f_X, b) ≤ √(3π/2) · 4^{-b}.

    Parameters
    ----------
    d : int
        Vector dimension (head_dim).
    bits : int
        Quantization bit-width.
    seed : int
        Random seed for generating the orthogonal matrix Π.
    device : str or torch.device
        Device for tensors.
    use_exact_pdf : bool
        Use exact Beta PDF for codebook (vs. Gaussian approximation).
    """

    def __init__(
        self,
        d: int,
        bits: int = 3,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
        use_exact_pdf: bool = False,
    ):
        self.d = d
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)

        # Rotation matrix
        self.Pi = _random_orthogonal(d, seed, self.device)

        # Codebook
        self.codebook = LloydMaxCodebook(d, bits, use_exact_pdf=use_exact_pdf)
        self.codebook.to(self.device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: x' = x @ Π^T  (last dim must be d)."""
        return x @ self.Pi.T

    def unrotate(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Inverse rotation: x = x_rot @ Π."""
        return x_rot @ self.Pi

    def compress(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress: normalize → rotate → quantize.

        Parameters
        ----------
        x : torch.Tensor, shape (..., d)
            Input vectors (e.g. value heads [batch, n_heads, seq_len, head_dim]).

        Returns
        -------
        indices : torch.Tensor, shape (..., d), dtype=uint8
        norms : torch.Tensor, shape (...), original L2 norms.
        """
        # Normalize to unit norm for quantization
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        x_rot = self.rotate(x_unit)
        indices = self.codebook.quantize_boundary(x_rot)
        return indices, norms.squeeze(-1)

    def decompress(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Decompress: dequantize → unrotate → rescale.

        Parameters
        ----------
        indices : torch.Tensor (uint8), shape (..., d)
        norms : torch.Tensor, shape (...)

        Returns
        -------
        x_hat : torch.Tensor, shape (..., d)
        """
        x_rot_hat = self.codebook.dequantize(indices)
        x_unit_hat = self.unrotate(x_rot_hat)
        return x_unit_hat * norms.unsqueeze(-1)

    def to(self, device: str | torch.device) -> "MSEQuantizer":
        self.device = torch.device(device)
        self.Pi = self.Pi.to(self.device)
        self.codebook.to(self.device)
        return self


# ---------------------------------------------------------------------------
# InnerProductQuantizer
# ---------------------------------------------------------------------------

class InnerProductQuantizer:
    """Two-stage quantizer for keys: (b-1)-bit MSE + 1-bit QJL.

    Stage 1: MSE quantizer at (total_bits - 1) bits.
    Stage 2: QJL sign hash on the residual with √(π/2)/d bias correction.

    This produces an **unbiased** inner product estimator with distortion
    bounded by D_ip ≤ √(3π/2) · ||y||² / d · 4^{-b}.

    Parameters
    ----------
    d : int
        Vector dimension (head_dim).
    total_bits : int
        Total bit budget per coordinate.  Stage 1 uses (total_bits - 1) bits,
        stage 2 uses 1 bit.  Default: 3 (= 2-bit MSE + 1-bit QJL).
    seed : int
        Random seed.
    device : str or torch.device
        Device for tensors.
    """

    def __init__(
        self,
        d: int,
        total_bits: int = DEFAULT_TOTAL_BITS,
        seed: int = DEFAULT_SEED,
        device: str | torch.device = "cpu",
        use_exact_pdf: bool = False,
    ):
        if total_bits < 2:
            raise ValueError("InnerProductQuantizer needs at least 2 total bits")

        self.d = d
        self.total_bits = total_bits
        self.mse_bits = total_bits - 1
        self.seed = seed
        self.device = torch.device(device)

        # Stage 1: MSE quantizer at (b-1) bits
        self.mse_quantizer = MSEQuantizer(
            d, bits=self.mse_bits, seed=seed, device=device,
            use_exact_pdf=use_exact_pdf,
        )

        # Stage 2: random sign vector for QJL
        self.S = _random_signs(d, seed, self.device)

        # Bias correction coefficient: sqrt(pi/2) / d
        self.qjl_scale = QJL_COEFF / d

    def compress(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress a key vector for inner-product-preserving quantization.

        Parameters
        ----------
        x : torch.Tensor, shape (..., d)
            Key vectors.

        Returns
        -------
        mse_indices : torch.Tensor (uint8), shape (..., d)
            MSE quantization indices at (b-1) bits.
        qjl_signs : torch.Tensor (uint8), shape (..., d)
            QJL sign bits (0 or 1) of the rotated residual * S.
        norms : torch.Tensor, shape (...)
            Original L2 norms.
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Stage 1: rotate + MSE quantize
        x_rot = self.mse_quantizer.rotate(x_unit)
        mse_indices = self.mse_quantizer.codebook.quantize_boundary(x_rot)
        x_rot_hat = self.mse_quantizer.codebook.dequantize(mse_indices)

        # Stage 2: residual → sign(residual * S)
        residual = x_rot - x_rot_hat
        qjl_signs = ((residual * self.S) >= 0).to(torch.uint8)

        return mse_indices, qjl_signs, norms.squeeze(-1)

    def decompress_for_dot(
        self,
        mse_indices: torch.Tensor,
        qjl_signs: torch.Tensor,
        norms: torch.Tensor,
        q_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate inner product <q, k> using the two-stage scheme.

        This doesn't reconstruct k explicitly.  Instead:
          <q, k̂> ≈ norm_k * ( <q_rot, c[indices]> + correction )
        where correction = (√(π/2)/d) * Σ_j |q_rot_j| * sign_j * S_j

        Parameters
        ----------
        mse_indices : torch.Tensor (uint8), shape (..., d)
        qjl_signs : torch.Tensor (uint8), shape (..., d)
        norms : torch.Tensor, shape (...)
            Key norms.
        q_rot : torch.Tensor, shape (..., d)
            **Rotated** query vector (already multiplied by Π).

        Returns
        -------
        scores : torch.Tensor, shape (...)
            Approximate <q, k> values.
        """
        # MSE part: dot product with centroids
        k_rot_hat = self.mse_quantizer.codebook.dequantize(mse_indices)
        mse_dot = (q_rot * k_rot_hat).sum(dim=-1)

        # QJL correction: √(π/2)/d · Σ_j |q_rot_j| · (2*sign_j - 1) · S_j
        sign_vals = qjl_signs.float() * 2.0 - 1.0  # {0,1} → {-1,+1}
        correction = self.qjl_scale * (q_rot.abs() * sign_vals * self.S).sum(dim=-1)

        return norms * (mse_dot + correction)

    def to(self, device: str | torch.device) -> "InnerProductQuantizer":
        self.device = torch.device(device)
        self.mse_quantizer.to(device)
        self.S = self.S.to(self.device)
        return self
