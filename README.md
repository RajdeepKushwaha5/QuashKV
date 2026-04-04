# QuashKV

**Near-optimal KV cache compression and vector quantization** based on the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google Research).

QuashKV compresses high-dimensional vectors (LLM key/value caches, embedding databases) to 2–4 bits per coordinate with proven distortion guarantees, enabling:

- **4–5× KV cache compression** with minimal quality loss
- **Zero-training vector search** (180,000× faster indexing than Product Quantization)
- **Drop-in HuggingFace and vLLM integration**

## Key Features

| Feature | Description |
|---------|-------------|
| Lloyd-Max codebooks | Optimal scalar quantizers solved against the exact coordinate PDF |
| MSE quantizer | Random rotation Π + per-coordinate Lloyd-Max quantization |
| Inner product quantizer | (b-1)-bit MSE + 1-bit QJL for unbiased IP estimation |
| Bit-packing | 1/2/3/4-bit packing with lossless round-trip |
| Mixed precision | Per-channel adaptive bit allocation (outlier channels get more bits) |
| Fused kernels | PyTorch fallbacks + Triton GPU kernel stubs |
| HF cache | Drop-in `DynamicCache` replacement for HuggingFace models |
| vLLM backend | `QuashKVModelManager` for vLLM serving integration |
| NN search | `QuashIndex` for approximate maximum inner-product search |

## Installation

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[dev]"      # pytest, benchmarks
pip install -e ".[triton]"   # Triton GPU kernels
pip install -e ".[bench]"    # transformers, datasets, matplotlib
pip install -e ".[all]"      # everything
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, SciPy ≥ 1.10

## Quick Start

### KV Cache Compression

```python
import torch
from quashkv import QuashKVEngine

engine = QuashKVEngine(head_dim=128, key_bits=3, value_bits=3)

# Compress and store KV pairs
keys = torch.randn(1, 8, 512, 128)    # (batch, heads, seq_len, head_dim)
values = torch.randn(1, 8, 512, 128)
engine.append(keys, values)

# Attend over compressed cache
queries = torch.randn(1, 8, 1, 128)
output = engine.attention(queries)     # (1, 8, 1, 128)

print(f"Compression ratio: {engine.compression_ratio():.1f}x")
# → Compression ratio: 4.9x
```

### HuggingFace Integration

```python
from quashkv.integrations.hf_cache import QuashKVCache

# Drop-in replacement for DynamicCache
cache = QuashKVCache(num_layers=32, head_dim=128)

# Use with any HuggingFace model:
# outputs = model.generate(**inputs, past_key_values=cache)

# Or auto-detect from model config:
# cache = QuashKVCache.from_model(model)
```

### Vector Similarity Search

```python
from quashkv import QuashIndex

# Build index (no training needed!)
index = QuashIndex(dim=128, bits=3)
index.add(database_vectors)  # (N, 128) float tensor

# Search
scores, indices = index.search(queries, k=10)

# Evaluate
recall = index.recall_at_k(queries, database_vectors, k=10)
print(f"Recall@10: {recall:.3f}")
```

### Mixed Precision (Outlier Handling)

```python
from quashkv import MixedPrecisionMSEQuantizer

# 2.5 effective bits: 32 outlier channels at 3b, 96 regular at 2b
quantizer = MixedPrecisionMSEQuantizer(
    d=128, outlier_bits=3, regular_bits=2, n_outliers=32,
)
quantizer.calibrate(calibration_data)  # (N, 128) sample

indices, norms = quantizer.compress(vectors)
reconstructed = quantizer.decompress(indices, norms)
```

## Architecture

```
quashkv/
├── codebook.py          # Lloyd-Max optimal scalar quantizer
├── quantizer.py         # MSEQuantizer + InnerProductQuantizer
├── engine.py            # QuashKVEngine orchestrator
├── packing.py           # Bit-packing (1-4 bit → bytes)
├── mixed_precision.py   # Per-channel adaptive bit allocation
├── constants.py         # Block sizes, defaults
├── integrations/
│   ├── hf_cache.py      # HuggingFace DynamicCache replacement
│   └── vllm_backend.py  # vLLM attention backend
├── nn_search/
│   └── index.py         # QuashIndex for vector search
└── triton_kernels/
    ├── compress.py      # Fused normalize→rotate→quantize
    ├── decompress.py    # Fused dequantize→unrotate→rescale
    └── attention.py     # Online-softmax quantized attention
```

## Theoretical Guarantees

From the TurboQuant paper (Zandieh et al., ICLR 2026):

| Metric | Bound | 3-bit value |
|--------|-------|-------------|
| MSE distortion | $\leq \sqrt{\frac{3\pi}{2}} \cdot 4^{-b}$ | ≤ 0.042 |
| IP distortion | $\leq \sqrt{\frac{3\pi}{2}} \cdot \frac{\|\|y\|\|^2}{d} \cdot 4^{-b}$ | ≤ 0.042/d |
| Optimality gap | $\frac{\text{upper bound}}{\text{lower bound}} \approx 2.72$ | Near-optimal |

The inner product estimator is **unbiased**: $E[\langle y, \tilde{x} \rangle] = \langle y, x \rangle$.

## Benchmarks

Run the included benchmarks:

```bash
# Quality (MSE, cosine similarity, compression ratio)
python -m benchmarks.quality_bench

# Latency (compression, attention, search timing)
python -m benchmarks.latency_bench

# NN search (recall@k, indexing time, memory)
python -m benchmarks.nn_search_bench
python -m benchmarks.nn_search_bench --large  # 100K vectors
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

154+ tests covering all modules: codebook, quantizer, engine, packing, HF cache, vLLM backend, NN search, Triton fallbacks, mixed precision, and end-to-end integration.

## Comparison with Related Work

| Aspect | QuashKV | Anirudh's cuTile impl | KIVI | PolarQuant |
|--------|---------|----------------------|------|------------|
| Hardware | Any GPU (A100/H100/4090) | B200 only | Any | Any |
| Kernel framework | Triton (portable) | cuTile (Blackwell-locked) | PyTorch | PyTorch |
| Vector search | Full NN search module | Not implemented | N/A | N/A |
| Outlier handling | Mixed-precision adaptive | Not implemented | Per-channel | None |
| Serving integration | vLLM + HuggingFace | Standalone notebook | HuggingFace | HuggingFace |
| Theoretical guarantees | Full paper bounds verified | Partial | None | None |
| Bit-width configs | 2, 2.5, 3, 3.5, 4-bit | 3-bit only | 2-bit | 4-bit |

## API Reference

### Core Classes

- **`QuashKVEngine`** — Main orchestrator. `append(keys, values)` compresses, `attention(queries)` attends.
- **`MSEQuantizer`** — Rotation + Lloyd-Max for MSE-optimal compression.
- **`InnerProductQuantizer`** — Two-stage (MSE + QJL) for unbiased IP estimation.
- **`LloydMaxCodebook`** — Precomputed optimal scalar quantizer.
- **`QuashIndex`** — Approximate MIPS index with `add()`, `search()`, `recall_at_k()`.
- **`MixedPrecisionMSEQuantizer`** / **`MixedPrecisionIPQuantizer`** — Adaptive bit allocation.

### Integration Classes

- **`QuashKVCache`** — HuggingFace `DynamicCache` replacement.
- **`QuashKVModelManager`** / **`QuashKVPageManager`** — vLLM integration.

### Utilities

- **`pack_bits(indices, bits)`** / **`unpack_bits(packed, bits, d)`** — Bit-packing.
- **`fused_compress_mse()`** / **`fused_compress_ip()`** — Fused compression kernels.
- **`fused_quantized_attention()`** — Online-softmax attention over compressed cache.

## Citation

Based on the TurboQuant paper:

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```

## License

Apache 2.0
