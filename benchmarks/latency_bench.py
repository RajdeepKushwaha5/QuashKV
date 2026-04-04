"""
Latency benchmark for quashkv operations.

Measures wall-clock time for:
  1. Compression (MSE and IP quantizer)
  2. Decompression (MSE and IP dequantizer)
  3. Attention over compressed cache
  4. NN search (add + search)
  5. Fused kernels vs standard path

Run:
    python -m benchmarks.latency_bench
"""

from __future__ import annotations

import time
import sys
import torch

from quashkv import QuashKVEngine, QuashIndex
from quashkv.integrations.hf_cache import QuashKVCache
from quashkv.triton_kernels import HAS_TRITON
from quashkv.triton_kernels.compress import fused_compress_mse, fused_compress_ip
from quashkv.triton_kernels.attention import (
    fused_quantized_attention,
    CompressedBlock,
)


def _timer(fn, warmup: int = 2, repeats: int = 10) -> float:
    """Time a function, returning median time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


def bench_compression(B: int, H: int, S: int, D: int, bits: int) -> dict:
    """Benchmark compression latency."""
    engine = QuashKVEngine(head_dim=D, key_bits=bits, value_bits=bits, seed=42)
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)

    def compress_keys():
        engine.compress_keys(keys)

    def compress_values():
        engine.compress_values(values)

    def full_append():
        engine.clear()
        engine.append(keys, values)

    # Fused MSE compression
    Pi_v = engine.value_quantizer.Pi
    bounds_v = engine.value_quantizer.codebook.boundaries

    def fused_mse():
        fused_compress_mse(values, Pi_v, bounds_v)

    return {
        "key_compress_ms": _timer(compress_keys),
        "value_compress_ms": _timer(compress_values),
        "full_append_ms": _timer(full_append),
        "fused_mse_compress_ms": _timer(fused_mse),
    }


def bench_attention(B: int, H: int, KV: int, Q: int, D: int, bits: int) -> dict:
    """Benchmark attention latency."""
    engine = QuashKVEngine(head_dim=D, key_bits=bits, value_bits=bits, seed=42)
    # Append in chunks to be realistic
    chunk_size = min(64, KV)
    for start in range(0, KV, chunk_size):
        end = min(start + chunk_size, KV)
        s = end - start
        engine.append(torch.randn(B, H, s, D), torch.randn(B, H, s, D))

    queries = torch.randn(B, H, Q, D)

    def engine_attention():
        engine.attention(queries)

    # Fused attention
    blocks = []
    for blk in engine._cache:
        mse, qjl, val = engine._unpack_block(blk)
        blocks.append(CompressedBlock(
            mse_indices=mse, qjl_signs=qjl,
            key_norms=blk.key_norms, val_indices=val,
            val_norms=blk.val_norms, seq_len=blk.seq_len,
        ))

    def fused_attn():
        fused_quantized_attention(
            queries, blocks,
            Pi=engine.key_quantizer.mse_quantizer.Pi,
            key_centroids=engine.key_quantizer.mse_quantizer.codebook.centroids,
            val_centroids=engine.value_quantizer.codebook.centroids,
            S=engine.key_quantizer.S,
            qjl_scale=engine.key_quantizer.qjl_scale,
            total_kv_tokens=engine.total_tokens,
        )

    return {
        "engine_attention_ms": _timer(engine_attention),
        "fused_attention_ms": _timer(fused_attn),
    }


def bench_nn_search(N: int, D: int, M: int, bits: int) -> dict:
    """Benchmark NN search latency."""
    db = torch.randn(N, D)
    queries = torch.randn(M, D)

    # Indexing time
    idx = QuashIndex(dim=D, bits=bits, seed=42)
    t0 = time.perf_counter()
    idx.add(db)
    index_ms = (time.perf_counter() - t0) * 1000

    def search():
        idx.search(queries, k=10)

    def brute():
        QuashIndex.brute_force(queries, db, k=10)

    return {
        "index_time_ms": index_ms,
        "search_ms": _timer(search),
        "brute_force_ms": _timer(brute),
        "speedup": _timer(brute) / max(_timer(search), 0.001),
    }


def bench_hf_cache(num_layers: int, B: int, H: int, D: int, prefill: int, decode_steps: int) -> dict:
    """Benchmark HF cache update latency."""
    cache = QuashKVCache(num_layers=num_layers, head_dim=D)

    def full_generation():
        cache.reset()
        # Prefill
        for layer in range(num_layers):
            k = torch.randn(B, H, prefill, D)
            v = torch.randn(B, H, prefill, D)
            cache.update(k, v, layer_idx=layer)
        # Decode
        for step in range(decode_steps):
            for layer in range(num_layers):
                k = torch.randn(B, H, 1, D)
                v = torch.randn(B, H, 1, D)
                cache.update(k, v, layer_idx=layer)

    return {
        "full_generation_ms": _timer(full_generation, warmup=1, repeats=3),
    }


def main():
    print("=" * 70)
    print("QuashKV Latency Benchmark")
    print(f"Device: CPU | Triton: {HAS_TRITON} | PyTorch: {torch.__version__}")
    print("=" * 70)

    configs = [
        {"B": 1, "H": 4, "S": 64, "D": 128, "bits": 3, "label": "B=1 H=4 S=64 D=128 3-bit"},
        {"B": 1, "H": 8, "S": 128, "D": 128, "bits": 3, "label": "B=1 H=8 S=128 D=128 3-bit"},
        {"B": 1, "H": 4, "S": 64, "D": 128, "bits": 4, "label": "B=1 H=4 S=64 D=128 4-bit"},
    ]

    print("\n--- Compression Latency ---")
    for cfg in configs:
        r = bench_compression(cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["bits"])
        print(f"  {cfg['label']}:")
        print(f"    Key compress:       {r['key_compress_ms']:8.2f} ms")
        print(f"    Value compress:     {r['value_compress_ms']:8.2f} ms")
        print(f"    Full append:        {r['full_append_ms']:8.2f} ms")
        print(f"    Fused MSE compress: {r['fused_mse_compress_ms']:8.2f} ms")

    print("\n--- Attention Latency ---")
    attn_configs = [
        {"B": 1, "H": 4, "KV": 64, "Q": 1, "D": 128, "bits": 3, "label": "KV=64 Q=1"},
        {"B": 1, "H": 4, "KV": 256, "Q": 1, "D": 128, "bits": 3, "label": "KV=256 Q=1"},
        {"B": 1, "H": 4, "KV": 256, "Q": 16, "D": 128, "bits": 3, "label": "KV=256 Q=16"},
    ]
    for cfg in attn_configs:
        r = bench_attention(cfg["B"], cfg["H"], cfg["KV"], cfg["Q"], cfg["D"], cfg["bits"])
        print(f"  {cfg['label']}:")
        print(f"    Engine attention: {r['engine_attention_ms']:8.2f} ms")
        print(f"    Fused attention:  {r['fused_attention_ms']:8.2f} ms")

    print("\n--- NN Search Latency ---")
    nn_configs = [
        {"N": 1000, "D": 64, "M": 10, "bits": 3, "label": "N=1K D=64"},
        {"N": 10000, "D": 128, "M": 10, "bits": 3, "label": "N=10K D=128"},
    ]
    for cfg in nn_configs:
        r = bench_nn_search(cfg["N"], cfg["D"], cfg["M"], cfg["bits"])
        print(f"  {cfg['label']}:")
        print(f"    Index time:    {r['index_time_ms']:8.2f} ms")
        print(f"    Search time:   {r['search_ms']:8.2f} ms")
        print(f"    Brute force:   {r['brute_force_ms']:8.2f} ms")

    print("\n--- HF Cache Simulation ---")
    r = bench_hf_cache(num_layers=4, B=1, H=4, D=128, prefill=32, decode_steps=10)
    print(f"  4-layer prefill=32 decode=10: {r['full_generation_ms']:.2f} ms")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
