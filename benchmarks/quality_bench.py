"""
Quality benchmark for quashkv.

Measures:
  - MSE distortion for value quantization at various bit widths
  - Inner product distortion for key quantization
  - Attention output cosine similarity (quantized vs exact)
  - Compression ratio

Runs on CPU with synthetic data — no GPU or model required.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from quashkv import QuashKVEngine, MSEQuantizer, InnerProductQuantizer
from quashkv.codebook import compute_mse_cost


@dataclass
class BenchmarkResult:
    head_dim: int
    bits: int
    theoretical_mse: float
    empirical_mse: float
    ip_bias: float
    ip_mse: float
    attn_cosine_sim: float
    compression_ratio: float
    compress_time_ms: float
    attention_time_ms: float


def benchmark_quality(
    head_dim: int = 128,
    key_bits: int = 3,
    value_bits: int = 3,
    n_vectors: int = 1024,
    n_queries: int = 16,
    n_heads: int = 4,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a full quality benchmark on synthetic data."""
    torch.manual_seed(seed)

    # --- Theoretical MSE ---
    theoretical_mse = compute_mse_cost(head_dim, value_bits)

    # --- Empirical MSE (values) ---
    val_q = MSEQuantizer(d=head_dim, bits=value_bits, seed=seed)
    vecs = torch.randn(n_vectors, head_dim)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    indices, norms = val_q.compress(vecs)
    vecs_hat = val_q.decompress(indices, norms)
    empirical_mse = ((vecs - vecs_hat) ** 2).sum(dim=-1).mean().item()

    # --- Inner product quality (keys) ---
    ip_q = InnerProductQuantizer(d=head_dim, total_bits=key_bits, seed=seed)
    keys = torch.randn(n_vectors, head_dim)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    query = torch.randn(head_dim)
    query = query / query.norm()

    true_ips = (keys * query).sum(dim=-1)
    mse_idx, qjl_signs, k_norms = ip_q.compress(keys)
    q_rot = ip_q.mse_quantizer.rotate(query.unsqueeze(0)).expand(n_vectors, -1)
    est_ips = ip_q.decompress_for_dot(mse_idx, qjl_signs, k_norms, q_rot)

    ip_bias = (est_ips - true_ips).mean().item()
    ip_mse = ((est_ips - true_ips) ** 2).mean().item()

    # --- Full attention quality ---
    B, H, S, D = 1, n_heads, n_vectors, head_dim
    Q_len = n_queries
    scale = 1.0 / math.sqrt(D)

    all_keys = torch.randn(B, H, S, D)
    all_values = torch.randn(B, H, S, D)
    queries = torch.randn(B, H, Q_len, D)

    # Exact attention
    scores_exact = torch.einsum("bhqd,bhkd->bhqk", queries, all_keys) * scale
    attn_exact = torch.softmax(scores_exact, dim=-1)
    out_exact = torch.einsum("bhqk,bhkd->bhqd", attn_exact, all_values)

    # Quantized attention via engine
    engine = QuashKVEngine(
        head_dim=D, key_bits=key_bits, value_bits=value_bits, seed=seed
    )

    t0 = time.perf_counter()
    engine.append(all_keys, all_values)
    compress_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_quant = engine.attention(queries)
    attention_time = (time.perf_counter() - t0) * 1000

    cosine_sim = torch.nn.functional.cosine_similarity(
        out_exact.flatten(), out_quant.flatten(), dim=0
    ).item()

    compression_ratio = engine.compression_ratio()

    return BenchmarkResult(
        head_dim=head_dim,
        bits=key_bits,
        theoretical_mse=theoretical_mse,
        empirical_mse=empirical_mse,
        ip_bias=ip_bias,
        ip_mse=ip_mse,
        attn_cosine_sim=cosine_sim,
        compression_ratio=compression_ratio,
        compress_time_ms=compress_time,
        attention_time_ms=attention_time,
    )


def run_all_benchmarks():
    """Run benchmarks across bit widths and dimensions, print results."""
    print("=" * 90)
    print(f"{'dim':>5} {'bits':>5} {'theory_mse':>11} {'emp_mse':>11} "
          f"{'ip_bias':>10} {'ip_mse':>10} {'attn_cos':>10} {'ratio':>7} "
          f"{'comp_ms':>9} {'attn_ms':>9}")
    print("=" * 90)

    for head_dim in [64, 128]:
        for bits in [2, 3, 4]:
            r = benchmark_quality(head_dim=head_dim, key_bits=bits, value_bits=bits)
            print(
                f"{r.head_dim:>5} {r.bits:>5} {r.theoretical_mse:>11.6f} "
                f"{r.empirical_mse:>11.6f} {r.ip_bias:>10.6f} {r.ip_mse:>10.6f} "
                f"{r.attn_cosine_sim:>10.4f} {r.compression_ratio:>7.2f}x "
                f"{r.compress_time_ms:>8.1f} {r.attention_time_ms:>8.1f}"
            )

    print("=" * 90)


if __name__ == "__main__":
    run_all_benchmarks()
