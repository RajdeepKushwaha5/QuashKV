"""
NN search benchmark for quashkv.

Benchmarks QuashIndex against brute-force across:
  - Database sizes (1K → 100K)
  - Dimensionalities (64, 128, 256, 512)
  - Bit widths (2, 3, 4)
  - Packed vs unpacked storage

Measures: recall@1, recall@10, recall@100, index time, search time,
compression ratio, and memory usage.

Paper claim (Section 4.4): TurboQuant indexing is ~180,000x faster
than Product Quantization because it needs NO training on the database.

Run:
    python -m benchmarks.nn_search_bench
    python -m benchmarks.nn_search_bench --large  # 100K vectors
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import torch

from quashkv.nn_search import QuashIndex


@dataclass
class SearchBenchResult:
    n_db: int
    n_queries: int
    dim: int
    bits: int
    packed: bool
    index_time_ms: float
    search_time_ms: float
    brute_force_ms: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    memory_bytes: int
    compression_ratio: float


def _timer_ms(fn, warmup: int = 1, repeats: int = 5) -> float:
    """Median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench_search(
    n_db: int = 10_000,
    n_queries: int = 100,
    dim: int = 128,
    bits: int = 3,
    packed: bool = False,
    seed: int = 42,
) -> SearchBenchResult:
    """Run a single NN search benchmark configuration."""
    torch.manual_seed(seed)
    db = torch.randn(n_db, dim)
    queries = torch.randn(n_queries, dim)

    # --- Index time ---
    idx = QuashIndex(dim=dim, bits=bits, seed=seed, pack_storage=packed)
    t0 = time.perf_counter()
    idx.add(db)
    index_ms = (time.perf_counter() - t0) * 1000

    # --- Search time (k=100 to cover recall@1/10/100) ---
    k_max = min(100, n_db)
    search_ms = _timer_ms(lambda: idx.search(queries, k=k_max))

    # --- Brute force time ---
    brute_ms = _timer_ms(lambda: QuashIndex.brute_force(queries, db, k=k_max))

    # --- Recall at various k ---
    _, approx_idx = idx.search(queries, k=k_max)
    _, exact_idx = QuashIndex.brute_force(queries, db, k=k_max)

    def _recall_at_k(k: int) -> float:
        hits = 0
        for i in range(n_queries):
            approx_set = set(approx_idx[i, :k].tolist())
            exact_set = set(exact_idx[i, :k].tolist())
            hits += len(approx_set & exact_set)
        return hits / (n_queries * k)

    r1 = _recall_at_k(1)
    r10 = _recall_at_k(min(10, k_max))
    r100 = _recall_at_k(k_max)

    return SearchBenchResult(
        n_db=n_db,
        n_queries=n_queries,
        dim=dim,
        bits=bits,
        packed=packed,
        index_time_ms=index_ms,
        search_time_ms=search_ms,
        brute_force_ms=brute_ms,
        recall_at_1=r1,
        recall_at_10=r10,
        recall_at_100=r100,
        memory_bytes=idx.memory_bytes(),
        compression_ratio=idx.compression_ratio(),
    )


def main():
    parser = argparse.ArgumentParser(description="QuashKV NN Search Benchmark")
    parser.add_argument("--large", action="store_true", help="Include 100K vector tests")
    args = parser.parse_args()

    print("=" * 80)
    print("QuashKV NN Search Benchmark")
    print(f"PyTorch {torch.__version__} | CPU")
    print("=" * 80)

    # --- Bit-width sweep ---
    print("\n--- Bit-width sweep (N=10K, D=128) ---")
    header = f"{'bits':>4} {'R@1':>6} {'R@10':>6} {'R@100':>6} {'idx_ms':>8} {'search_ms':>10} {'brute_ms':>10} {'ratio':>6}"
    print(header)
    print("-" * len(header))
    for bits in [2, 3, 4]:
        r = bench_search(n_db=10_000, dim=128, bits=bits)
        print(
            f"{bits:>4} {r.recall_at_1:>6.3f} {r.recall_at_10:>6.3f} "
            f"{r.recall_at_100:>6.3f} {r.index_time_ms:>8.1f} "
            f"{r.search_time_ms:>10.1f} {r.brute_force_ms:>10.1f} "
            f"{r.compression_ratio:>6.1f}x"
        )

    # --- Dimension sweep ---
    print("\n--- Dimension sweep (N=10K, 3-bit) ---")
    header = f"{'dim':>4} {'R@1':>6} {'R@10':>6} {'idx_ms':>8} {'search_ms':>10} {'brute_ms':>10} {'mem_MB':>8}"
    print(header)
    print("-" * len(header))
    for dim in [64, 128, 256]:
        r = bench_search(n_db=10_000, dim=dim, bits=3)
        mem_mb = r.memory_bytes / (1024 * 1024)
        print(
            f"{dim:>4} {r.recall_at_1:>6.3f} {r.recall_at_10:>6.3f} "
            f"{r.index_time_ms:>8.1f} {r.search_time_ms:>10.1f} "
            f"{r.brute_force_ms:>10.1f} {mem_mb:>8.2f}"
        )

    # --- Packed vs unpacked ---
    print("\n--- Packed vs Unpacked (N=10K, D=128, 3-bit) ---")
    for packed in [False, True]:
        r = bench_search(n_db=10_000, dim=128, bits=3, packed=packed)
        label = "packed" if packed else "unpacked"
        mem_mb = r.memory_bytes / (1024 * 1024)
        print(
            f"  {label:>8}: R@10={r.recall_at_10:.3f}  "
            f"search={r.search_time_ms:.1f}ms  mem={mem_mb:.2f}MB  "
            f"ratio={r.compression_ratio:.1f}x"
        )

    # --- Scale sweep ---
    if args.large:
        print("\n--- Scale sweep (D=128, 3-bit) ---")
        header = f"{'N':>8} {'R@10':>6} {'idx_ms':>10} {'search_ms':>10} {'brute_ms':>10}"
        print(header)
        print("-" * len(header))
        for n_db in [1_000, 10_000, 50_000, 100_000]:
            r = bench_search(n_db=n_db, dim=128, bits=3)
            print(
                f"{n_db:>8} {r.recall_at_10:>6.3f} "
                f"{r.index_time_ms:>10.1f} {r.search_time_ms:>10.1f} "
                f"{r.brute_force_ms:>10.1f}"
            )
    else:
        print("\n(Use --large to include 50K-100K scale tests)")

    # --- Key insight: indexing time ---
    print("\n--- Indexing Time Advantage ---")
    print("TurboQuant/QuashKV requires NO training on the database.")
    print("Index time is pure compression time (rotate + quantize per vector).")
    for n in [1_000, 10_000]:
        r = bench_search(n_db=n, dim=128, bits=3)
        per_vec_us = r.index_time_ms * 1000 / n
        print(f"  N={n:>6}: {r.index_time_ms:.1f}ms total, {per_vec_us:.1f}µs/vec")

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
