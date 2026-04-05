"""
Latency benchmark for quashkv operations.

Measures wall-clock time for:
  1. Compression (MSE and IP quantizer)
  2. Decompression (MSE and IP dequantizer)
  3. Attention over compressed cache
  4. NN search (add + search)
  5. Fused Triton kernels vs PyTorch fallback (GPU only)

Run (CPU):
    python benchmarks/latency_bench.py

Run (GPU):
    python benchmarks/latency_bench.py --device cuda

Run (Triton vs PyTorch comparison):
    python benchmarks/latency_bench.py --device cuda --triton-compare
"""

from __future__ import annotations

import argparse
import time
import sys
import torch

from quashkv import QuashKVEngine, QuashIndex
from quashkv.integrations.hf_cache import QuashKVCache
from quashkv.triton_kernels import HAS_TRITON
from quashkv.triton_kernels.compress import (
    fused_compress_mse, fused_compress_ip,
    _compress_mse_pytorch, _compress_ip_pytorch,
)
from quashkv.triton_kernels.decompress import (
    fused_decompress_mse, fused_decompress_ip,
    _decompress_mse_pytorch, _decompress_ip_pytorch,
)
from quashkv.triton_kernels.attention import (
    fused_quantized_attention,
    _fused_attention_pytorch,
    CompressedBlock,
)


def _timer(fn, warmup: int = 2, repeats: int = 10, device: str = "cpu") -> float:
    """Time a function, returning median time in milliseconds."""
    use_cuda = device != "cpu" and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
        if use_cuda:
            torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


def bench_compression(B: int, H: int, S: int, D: int, bits: int, device: str = "cpu") -> dict:
    """Benchmark compression latency."""
    engine = QuashKVEngine(head_dim=D, key_bits=bits, value_bits=bits, seed=42, device=device)
    keys = torch.randn(B, H, S, D, device=device)
    values = torch.randn(B, H, S, D, device=device)

    def compress_keys():
        engine.compress_keys(keys)

    def compress_values():
        engine.compress_values(values)

    def full_append():
        engine.clear()
        engine.append(keys, values)

    # Fused MSE compression
    Pi_v = engine.value_quantizer.Pi.to(device)
    bounds_v = engine.value_quantizer.codebook.boundaries.to(device)

    def fused_mse():
        fused_compress_mse(values, Pi_v, bounds_v)

    return {
        "key_compress_ms": _timer(compress_keys, device=device),
        "value_compress_ms": _timer(compress_values, device=device),
        "full_append_ms": _timer(full_append, device=device),
        "fused_mse_compress_ms": _timer(fused_mse, device=device),
    }


def bench_attention(B: int, H: int, KV: int, Q: int, D: int, bits: int, device: str = "cpu") -> dict:
    """Benchmark attention latency."""
    engine = QuashKVEngine(head_dim=D, key_bits=bits, value_bits=bits, seed=42, device=device)
    # Append in chunks to be realistic
    chunk_size = min(64, KV)
    for start in range(0, KV, chunk_size):
        end = min(start + chunk_size, KV)
        s = end - start
        engine.append(torch.randn(B, H, s, D, device=device),
                      torch.randn(B, H, s, D, device=device))

    queries = torch.randn(B, H, Q, D, device=device)

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

    Pi = engine.key_quantizer.mse_quantizer.Pi.to(device)
    key_cents = engine.key_quantizer.mse_quantizer.codebook.centroids.to(device)
    val_cents = engine.value_quantizer.codebook.centroids.to(device)
    S = engine.key_quantizer.S.to(device)

    def fused_attn():
        fused_quantized_attention(
            queries, blocks,
            Pi=Pi,
            key_centroids=key_cents,
            val_centroids=val_cents,
            S=S,
            qjl_scale=engine.key_quantizer.qjl_scale,
            total_kv_tokens=engine.total_tokens,
        )

    return {
        "engine_attention_ms": _timer(engine_attention, device=device),
        "fused_attention_ms": _timer(fused_attn, device=device),
    }


def bench_nn_search(N: int, D: int, M: int, bits: int, device: str = "cpu") -> dict:
    """Benchmark NN search latency."""
    db = torch.randn(N, D, device=device)
    queries = torch.randn(M, D, device=device)

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
        "search_ms": _timer(search, device=device),
        "brute_force_ms": _timer(brute, device=device),
        "speedup": _timer(brute, device=device) / max(_timer(search, device=device), 0.001),
    }


def bench_hf_cache(num_layers: int, B: int, H: int, D: int, prefill: int, decode_steps: int, device: str = "cpu") -> dict:
    """Benchmark HF cache update latency."""
    cache = QuashKVCache(num_layers=num_layers, head_dim=D, device=device)

    def full_generation():
        cache.reset()
        # Prefill
        for layer in range(num_layers):
            k = torch.randn(B, H, prefill, D, device=device)
            v = torch.randn(B, H, prefill, D, device=device)
            cache.update(k, v, layer_idx=layer)
        # Decode
        for step in range(decode_steps):
            for layer in range(num_layers):
                k = torch.randn(B, H, 1, D, device=device)
                v = torch.randn(B, H, 1, D, device=device)
                cache.update(k, v, layer_idx=layer)

    return {
        "full_generation_ms": _timer(full_generation, warmup=1, repeats=3, device=device),
    }


def bench_triton_vs_pytorch(device: str = "cuda") -> dict:
    """Direct Triton kernel vs PyTorch fallback comparison (GPU only).

    Forces the PyTorch path even on CUDA to get a fair comparison.
    """
    import math

    results = {}
    D = 128
    bits = 3

    for S in [64, 256, 1024, 4096]:
        tag = f"S={S}"
        B, H = 1, 8
        engine = QuashKVEngine(head_dim=D, key_bits=bits, value_bits=bits, seed=42, device=device)

        keys = torch.randn(B, H, S, D, device=device)
        values = torch.randn(B, H, S, D, device=device)
        engine.append(keys, values)

        queries = torch.randn(B, H, 4, D, device=device)

        blocks = []
        for blk in engine._cache:
            mse, qjl, val = engine._unpack_block(blk)
            blocks.append(CompressedBlock(
                mse_indices=mse, qjl_signs=qjl,
                key_norms=blk.key_norms, val_indices=val,
                val_norms=blk.val_norms, seq_len=blk.seq_len,
            ))

        Pi = engine.key_quantizer.mse_quantizer.Pi.to(device)
        key_cents = engine.key_quantizer.mse_quantizer.codebook.centroids.to(device)
        val_cents = engine.value_quantizer.codebook.centroids.to(device)
        Sv = engine.key_quantizer.S.to(device)
        qjl_s = engine.key_quantizer.qjl_scale
        scale = 1.0 / math.sqrt(D)

        # PyTorch path (force)
        def pytorch_attn():
            _fused_attention_pytorch(
                queries, blocks, Pi, key_cents, val_cents,
                Sv, qjl_s, scale, causal=False,
            )

        # Triton path (auto-dispatch on CUDA)
        def triton_attn():
            fused_quantized_attention(
                queries, blocks, Pi=Pi,
                key_centroids=key_cents, val_centroids=val_cents,
                S=Sv, qjl_scale=qjl_s,
                total_kv_tokens=engine.total_tokens,
            )

        pt_ms = _timer(pytorch_attn, warmup=3, repeats=20, device=device)
        tr_ms = _timer(triton_attn, warmup=3, repeats=20, device=device)

        results[tag] = {
            "pytorch_ms": pt_ms,
            "triton_ms": tr_ms,
            "speedup": pt_ms / max(tr_ms, 0.001),
        }

        # Also bench compress
        Pi_v = engine.value_quantizer.Pi.to(device)
        bounds = engine.value_quantizer.codebook.boundaries.to(device)

        def pytorch_compress():
            _compress_mse_pytorch(values, Pi_v, bounds)

        def triton_compress():
            fused_compress_mse(values, Pi_v, bounds)

        pt_c = _timer(pytorch_compress, warmup=3, repeats=20, device=device)
        tr_c = _timer(triton_compress, warmup=3, repeats=20, device=device)
        results[tag]["compress_pytorch_ms"] = pt_c
        results[tag]["compress_triton_ms"] = tr_c
        results[tag]["compress_speedup"] = pt_c / max(tr_c, 0.001)

        del engine

    return results


def main():
    parser = argparse.ArgumentParser(description="QuashKV Latency Benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device to benchmark on")
    parser.add_argument("--triton-compare", action="store_true",
                        help="Run Triton vs PyTorch head-to-head (GPU only)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    print("=" * 70)
    print("QuashKV Latency Benchmark")
    print(f"Device: {device} ({gpu_name}) | Triton: {HAS_TRITON} | PyTorch: {torch.__version__}")
    print("=" * 70)

    configs = [
        {"B": 1, "H": 4, "S": 64, "D": 128, "bits": 3, "label": "B=1 H=4 S=64 D=128 3-bit"},
        {"B": 1, "H": 8, "S": 128, "D": 128, "bits": 3, "label": "B=1 H=8 S=128 D=128 3-bit"},
        {"B": 1, "H": 4, "S": 64, "D": 128, "bits": 4, "label": "B=1 H=4 S=64 D=128 4-bit"},
    ]

    print("\n--- Compression Latency ---")
    for cfg in configs:
        r = bench_compression(cfg["B"], cfg["H"], cfg["S"], cfg["D"], cfg["bits"], device)
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
    if device == "cuda":
        attn_configs.extend([
            {"B": 1, "H": 8, "KV": 1024, "Q": 1, "D": 128, "bits": 3, "label": "KV=1K Q=1 H=8"},
            {"B": 1, "H": 8, "KV": 4096, "Q": 1, "D": 128, "bits": 3, "label": "KV=4K Q=1 H=8"},
        ])

    for cfg in attn_configs:
        r = bench_attention(cfg["B"], cfg["H"], cfg["KV"], cfg["Q"], cfg["D"], cfg["bits"], device)
        print(f"  {cfg['label']}:")
        print(f"    Engine attention: {r['engine_attention_ms']:8.2f} ms")
        print(f"    Fused attention:  {r['fused_attention_ms']:8.2f} ms")

    print("\n--- NN Search Latency ---")
    nn_configs = [
        {"N": 1000, "D": 64, "M": 10, "bits": 3, "label": "N=1K D=64"},
        {"N": 10000, "D": 128, "M": 10, "bits": 3, "label": "N=10K D=128"},
    ]
    for cfg in nn_configs:
        r = bench_nn_search(cfg["N"], cfg["D"], cfg["M"], cfg["bits"], device)
        print(f"  {cfg['label']}:")
        print(f"    Index time:    {r['index_time_ms']:8.2f} ms")
        print(f"    Search time:   {r['search_ms']:8.2f} ms")
        print(f"    Brute force:   {r['brute_force_ms']:8.2f} ms")

    print("\n--- HF Cache Simulation ---")
    r = bench_hf_cache(num_layers=4, B=1, H=4, D=128, prefill=32, decode_steps=10, device=device)
    print(f"  4-layer prefill=32 decode=10: {r['full_generation_ms']:.2f} ms")

    # Triton vs PyTorch head-to-head
    if args.triton_compare and device == "cuda" and HAS_TRITON:
        print("\n" + "=" * 70)
        print("Triton vs PyTorch Head-to-Head (D=128, H=8, 3-bit)")
        print("=" * 70)
        results = bench_triton_vs_pytorch(device)
        print(f"  {'Seq':>6} | {'PyTorch':>10} | {'Triton':>10} | {'Speedup':>8}"
              f" | {'PT Comp':>10} | {'TR Comp':>10} | {'Comp Spd':>8}")
        print(f"  {'-' * 75}")
        for tag, r in results.items():
            print(f"  {tag:>6} | {r['pytorch_ms']:>9.2f}ms"
                  f" | {r['triton_ms']:>9.2f}ms"
                  f" | {r['speedup']:>7.2f}x"
                  f" | {r['compress_pytorch_ms']:>9.2f}ms"
                  f" | {r['compress_triton_ms']:>9.2f}ms"
                  f" | {r['compress_speedup']:>7.2f}x")
    elif args.triton_compare and device != "cuda":
        print("\n  --triton-compare requires --device cuda")

    print("\n" + "=" * 70)
    print("Benchmark complete.")



if __name__ == "__main__":
    main()
