"""
TurboQuant vector search index.

Compresses a database of vectors using the TurboQuant inner-product
quantizer, then supports approximate maximum inner-product search (MIPS).

Key advantages over PQ / IVF-PQ (from TurboQuant paper, Section 4.4):
  - Near-zero indexing time (no k-means training on the database)
  - Theoretical distortion bound: D_ip ≤ √(3π/2) · ||y||²/d · 4^{-b}
  - Unbiased: E[<y, x̃>] = <y, x>
"""

from __future__ import annotations

import torch

from ..quantizer import InnerProductQuantizer
from ..packing import pack_bits, unpack_bits


class QuashIndex:
    """Approximate nearest-neighbor search via TurboQuant compression.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    bits : int
        Total bits per coordinate.  Default 3 (2-bit MSE + 1-bit QJL).
    seed : int
        Random seed for rotation matrix and QJL signs.
    device : str or torch.device
        Device for tensors.
    pack_storage : bool
        If True, pack quantized indices for memory savings.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        seed: int = 42,
        device: str | torch.device = "cpu",
        pack_storage: bool = False,
    ):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = torch.device(device)
        self.pack_storage = pack_storage

        self.quantizer = InnerProductQuantizer(
            d=dim, total_bits=bits, seed=seed, device=device,
        )

        # Compressed storage
        self._mse_indices: torch.Tensor | None = None  # (N, D) or packed
        self._qjl_signs: torch.Tensor | None = None    # (N, D) or packed
        self._norms: torch.Tensor | None = None         # (N,) float32
        self._n_vectors: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    def __len__(self) -> int:
        return self._n_vectors

    # ------------------------------------------------------------------
    # Add vectors
    # ------------------------------------------------------------------

    def add(self, vectors: torch.Tensor) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        vectors : (N, D) float tensor
        """
        if vectors.dim() != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected (N, {self.dim}) tensor, got shape {tuple(vectors.shape)}"
            )

        x = vectors.to(self.device)
        mse_idx, qjl_sgn, norms = self.quantizer.compress(x)

        if self.pack_storage:
            mse_bits = self.bits - 1
            mse_idx = pack_bits(mse_idx, mse_bits)
            qjl_sgn = pack_bits(qjl_sgn, 1)

        if self._mse_indices is None:
            self._mse_indices = mse_idx
            self._qjl_signs = qjl_sgn
            self._norms = norms
        else:
            self._mse_indices = torch.cat([self._mse_indices, mse_idx], dim=0)
            self._qjl_signs = torch.cat([self._qjl_signs, qjl_sgn], dim=0)
            self._norms = torch.cat([self._norms, norms], dim=0)

        self._n_vectors += vectors.shape[0]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _get_unpacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mse_indices, qjl_signs) as (N, D) uint8."""
        assert self._mse_indices is not None
        if not self.pack_storage:
            return self._mse_indices, self._qjl_signs
        mse = unpack_bits(self._mse_indices, self.bits - 1, self.dim)
        qjl = unpack_bits(self._qjl_signs, 1, self.dim)
        return mse, qjl

    def search(self, queries: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximate maximum inner-product search.

        Uses the TurboQuant two-stage IP estimator:
        score = ||x|| · ( <q_rot, centroid(x)> + QJL_correction )

        Parameters
        ----------
        queries : (M, D) float tensor
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        scores : (M, k) approximate inner products (descending)
        indices : (M, k) database vector indices
        """
        if self._n_vectors == 0:
            raise ValueError("Index is empty. Call add() first.")

        k = min(k, self._n_vectors)
        queries = queries.to(self.device)
        mse_idx, qjl_sgn = self._get_unpacked()

        # Rotate queries into the compressed coordinate frame
        Pi = self.quantizer.mse_quantizer.Pi  # (D, D)
        q_rot = queries @ Pi.T  # (M, D)

        # --- MSE dot products ---
        centroids = self.quantizer.mse_quantizer.codebook.dequantize(mse_idx)  # (N, D)
        mse_dots = q_rot @ centroids.T  # (M, N)

        # --- QJL correction ---
        S = self.quantizer.S  # (D,)
        sign_vals = qjl_sgn.float() * 2.0 - 1.0  # (N, D)
        correction = self.quantizer.qjl_scale * (
            (q_rot.abs() * S) @ sign_vals.T
        )  # (M, N)

        # Combine with stored norms
        scores = self._norms.unsqueeze(0) * (mse_dots + correction)  # (M, N)

        return torch.topk(scores, k, dim=1)

    # ------------------------------------------------------------------
    # Ground truth & evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def brute_force(
        queries: torch.Tensor, database: torch.Tensor, k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact brute-force MIPS (for ground truth).

        Parameters
        ----------
        queries : (M, D) float
        database : (N, D) float
        k : int

        Returns
        -------
        scores : (M, k)
        indices : (M, k)
        """
        k = min(k, database.shape[0])
        scores = queries @ database.T  # (M, N)
        return torch.topk(scores, k, dim=1)

    def recall_at_k(
        self,
        queries: torch.Tensor,
        database: torch.Tensor,
        k: int = 10,
    ) -> float:
        """Compute recall@k against exact brute-force.

        Parameters
        ----------
        queries : (M, D) float
        database : (N, D) float — the original vectors added via add()
        k : int

        Returns
        -------
        float : mean recall@k over all queries (0.0 to 1.0)
        """
        _, approx_idx = self.search(queries, k)
        _, exact_idx = self.brute_force(queries, database, k)

        # Vectorized set-intersection via sorting + comparison
        total_hits = 0
        m = queries.shape[0]
        for i in range(m):
            approx_set = set(approx_idx[i].tolist())
            exact_set = set(exact_idx[i].tolist())
            total_hits += len(approx_set & exact_set)

        return total_hits / (m * k)

    # ------------------------------------------------------------------
    # Memory / compression stats
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Actual memory consumed by compressed index (bytes)."""
        if self._n_vectors == 0:
            return 0
        total = self._mse_indices.nelement() * self._mse_indices.element_size()
        total += self._qjl_signs.nelement() * self._qjl_signs.element_size()
        total += self._norms.nelement() * self._norms.element_size()
        return total

    def compression_ratio(self, original_dtype: torch.dtype = torch.float32) -> float:
        """Ratio of original float storage to compressed storage."""
        if self._n_vectors == 0:
            return 1.0
        element_bytes = torch.finfo(original_dtype).bits // 8
        original = self._n_vectors * self.dim * element_bytes
        compressed = self.memory_bytes()
        return original / compressed if compressed > 0 else float("inf")

    def reset(self) -> None:
        """Clear all stored vectors."""
        self._mse_indices = None
        self._qjl_signs = None
        self._norms = None
        self._n_vectors = 0

    def __repr__(self) -> str:
        return (
            f"QuashIndex(dim={self.dim}, bits={self.bits}, "
            f"n_vectors={self._n_vectors}, "
            f"pack_storage={self.pack_storage}, device={self.device})"
        )
