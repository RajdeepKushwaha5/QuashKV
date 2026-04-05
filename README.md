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
| Fused kernels | PyTorch fallbacks + Triton GPU kernels (compress, decompress, fused attention) |
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

## Real-Model Validation

Validated on real LLM KV caches using a Tesla T4 GPU (Kaggle). Both models use head_dim=64.

### 3-bit Compression (Key: IP Quantizer, Value: MSE Quantizer)

| Model | Layers | KV Heads | Key Cosine | Value Cosine | Attention Cosine |
|-------|--------|----------|-----------|-------------|-----------------|
| Qwen/Qwen2.5-0.5B | 24 | 2 (GQA) | 0.9409 | 0.9838 | 0.8869 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 22 | 4 (GQA) | 0.9420 | 0.9836 | 0.9237 |

### Bit-Width Sweep

| Bits | Key Cosine (Qwen / TinyLlama) | Value Cosine (Qwen / TinyLlama) | Compression |
|------|-------------------------------|--------------------------------|-------------|
| 2-bit | 0.7988 / 0.8011 | 0.9421 / 0.9417 | 8.0x |
| 3-bit | 0.9409 / 0.9420 | 0.9838 / 0.9836 | 5.3x |
| 4-bit | 0.9835 / 0.9837 | 0.9956 / 0.9956 | 4.0x |

### Memory Savings (3-bit, packed)

| Metric | Value |
|--------|-------|
| Compression ratio | 4.57x |
| Memory saved | 78.1% |

### Long-Sequence Stability (TinyLlama 1.1B)

| Sequence Length | Key Cosine | Value Cosine | Original | Compressed |
|----------------|-----------|-------------|----------|------------|
| 122 tokens | 0.9420 | 0.9836 | 2.6 MB | 0.6 MB |
| 1,040 tokens | 0.9420 | 0.9835 | 22.3 MB | 5.0 MB |

Quality is **perfectly stable** — identical cosine similarity at 8.5x longer sequences. Compression ratio holds at 4.57x regardless of sequence length.

> Key cosine similarity (not raw MSE) is the correct quality metric for keys because the IP quantizer preserves direction. Value MSE stays well within the theoretical bound (0.002 vs 0.035).

See [`notebooks/kaggle_validation.ipynb`](notebooks/kaggle_validation.ipynb) for the full reproducible notebook.

## A100 GPU Benchmarks

All numbers collected on an NVIDIA A100-SXM4-80GB, PyTorch 2.8.0+cu128, Triton enabled.

### Fused Attention Speedup

The fused Triton attention kernel (online softmax, no KV materialization) vs. the engine's standard decompress-then-attend path:

| KV Length | Engine (naive) | Fused Attention | Speedup |
|-----------|---------------|-----------------|---------|
| 64 | 0.40 ms | 0.20 ms | **2.0×** |
| 256 | 1.06 ms | 0.24 ms | **4.4×** |
| 1,024 | 3.71 ms | 0.50 ms | **7.4×** |
| 4,096 | 14.65 ms | 1.55 ms | **9.4×** |

Fused compression is also ~2× faster: 0.13 ms (fused) vs. 0.27 ms (standard) at seq_len=256.

### WikiText-2 Perplexity — TinyLlama 1.1B (22 layers, head_dim=64)

Baseline perplexity: **6.97** (4096 tokens, sliding window, no compression)

| Bits | Key Cosine | Value Cosine | Compressed PPL | Δ PPL | Compression | Memory Saved |
|------|-----------|-------------|---------------|-------|-------------|--------------|
| 2 | 0.8011 | 0.9411 | 31.56 | +24.59 | 6.40× | 84.4% |
| 3 | 0.9411 | 0.9831 | **7.09** | **+0.12** | 4.57× | 78.1% |
| 4 | 0.9832 | 0.9954 | **6.97** | **+0.00** | 3.56× | 71.9% |

> 2-bit degrades badly on TinyLlama (head_dim=64) because each coordinate carries more information. With Mistral 7B (head_dim=128), 2-bit is far more usable — see below.

### WikiText-2 Perplexity — Mistral 7B (32 layers, head_dim=128)

Baseline perplexity: **5.06** (4096 tokens, fp16, no compression)

| Bits | Key Cosine | Value Cosine | Compressed PPL | Δ PPL | Compression | Memory Saved |
|------|-----------|-------------|---------------|-------|-------------|--------------|
| 2 | 0.7997 | 0.9403 | 5.88 | +0.82 | 7.11× | 85.9% |
| 3 | 0.9404 | 0.9829 | **5.10** | **+0.04** | 4.92× | 79.7% |
| 4 | 0.9829 | 0.9953 | **5.07** | **+0.01** | 3.76× | 73.4% |

> **3-bit compression adds only +0.04 perplexity** (5.06 → 5.10) while saving 80% memory — effectively lossless. 4-bit is indistinguishable from baseline (+0.01). Even 2-bit only adds +0.82 with 86% memory savings.

### End-to-End Generation — TinyLlama 1.1B (A100)

Side-by-side text generation comparing standard (uncompressed) vs. compressed KV cache:

| Metric | Standard | 3-bit Compressed | 2-bit Compressed |
|--------|----------|------------------|------------------|
| Decode speed | 65 tok/s | 3.3 tok/s | 3.4 tok/s |
| KV cache size | 2.4 MB | 0.5 MB | 0.4 MB |
| Compression | 1.0× | **4.6×** | **6.4×** |
| Memory saved | — | **78%** | **84%** |

**3-bit sample** (prompt: *"In a small village nestled between mountains,"*):

> **Standard:** *a young woman named Lily lives a simple life. She works as a cook in a local inn, and her days are filled with the sounds of birds chirping and the rustling of leaves in the wind…*
>
> **3-bit compressed:** *a young woman named Lily had a secret. She had been raised by her grandmother, who had been a kind and loving woman, but Lily had never known her mother…*

3-bit outputs are fluent and coherent — the autoregressive path diverges but produces equally valid continuations. 2-bit (1-bit MSE + 1-bit QJL for keys) shows visible quality degradation with repetitive tokens, confirming that **3-bit is the practical sweet spot** for generation quality.

### End-to-End Generation — Mistral 7B (A100)

| Metric | Standard | 3-bit Compressed |
|--------|----------|------------------|
| Decode speed | 47 tok/s | 2.3 tok/s |
| KV cache size | 13.8 MB | 2.8 MB |
| Compression | 1.0× | **4.9×** |
| Memory saved | — | **80%** |

**Sample output** (prompt: *"The key insight behind transformer models is that"*):

> **Standard:** *the order of words in a sentence doesn't matter as much as the relationships between words. This is a powerful idea, but it's also a bit of a double-edged sword…*
>
> **3-bit compressed:** *the order of words in a sentence doesn't matter as much as the relationships between words. This is a powerful idea, and it's the reason why transformer models can be trained on a relatively small amount of data…*

**28% token match** on this prompt — the first 25 tokens are identical. Mistral 7B's larger head_dim (128 vs 64) means the compression noise is proportionally smaller, producing higher-quality compressed output than TinyLlama.

The compressed decode speed reflects per-token decompress→forward→recompress overhead; the fused Triton kernel eliminates decompression entirely for attention-bound workloads (9.4× speedup benchmarked separately).

```bash
# Run generation demo
python benchmarks/generation_demo.py
python benchmarks/generation_demo.py --bits 2      # 2-bit aggressive
python benchmarks/generation_demo.py --model mistralai/Mistral-7B-Instruct-v0.3 --dtype float16
```

### Run Benchmarks

```bash
# Latency (GPU required, --triton-compare for head-to-head)
python benchmarks/latency_bench.py --device cuda --triton-compare

# Perplexity (downloads model + WikiText-2)
python benchmarks/perplexity_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits 2 3 4 --max-tokens 2048
python benchmarks/perplexity_eval.py --model mistralai/Mistral-7B-v0.3 --bits 2 3 4 --max-tokens 4096 --dtype float16

# Quality (MSE, cosine similarity, compression ratio)
python -m benchmarks.quality_bench

# NN search (recall@k, indexing time, memory)
python -m benchmarks.nn_search_bench
python -m benchmarks.nn_search_bench --large  # 100K vectors
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

175 tests covering all modules: codebook, quantizer, engine, packing, HF cache, vLLM backend, NN search, Triton fallbacks, mixed precision, and end-to-end integration.

## Comparison with Related Work

### Feature Matrix

| Aspect | QuashKV | Anirudh's cuTile impl | KIVI | PolarQuant |
|--------|---------|----------------------|------|------------|
| Hardware | Any GPU (A100/H100/4090) | B200 only | Any | Any |
| Kernel framework | PyTorch + Triton kernels | cuTile (Blackwell-locked) | PyTorch | PyTorch |
| End-to-end generation | ✅ Side-by-side demo | ✅ Qwen 2.5 demo | ❌ | ❌ |
| Compressed perplexity eval | ✅ WikiText-2 (2 models) | ❌ | ❌ | ❌ |
| Vector search | Full NN search module | ❌ | N/A | N/A |
| Outlier handling | Mixed-precision adaptive | ❌ | Per-channel | None |
| Serving integration | vLLM + HuggingFace | Standalone notebook | HuggingFace | HuggingFace |
| Theoretical guarantees | Full paper bounds verified | Partial | None | None |
| Bit-width configs | 2, 2.5, 3, 3.5, 4-bit | 3-bit only | 2-bit | 4-bit |
| Multi-model validation | TinyLlama, Mistral 7B | Qwen 2.5-1.5B only | Llama | Llama |
| Test coverage | 175 tests | ~5 tests | Minimal | Minimal |
| Lines of code | ~4,090 | ~2,000 | — | — |

### Where cuTile Is Stronger (and why)

**Raw kernel performance.** Anirudh's fused attention kernel runs on NVIDIA B200 with cuTile — a Blackwell-native DSL that compiles directly to Tensor Core instructions, hardware TMA (Tensor Memory Accelerator) prefetch, and TMEM→MMA paths that don't exist on older architectures. His generation demo hits **144.7 tok/s** on B200 vs our 47 tok/s on A100 for a comparable model size. This isn't close.

Specific Blackwell advantages cuTile exploits that we can't touch:
- **Hardware TMA prefetch** with latency hints (`latency=2`, `latency=4`) — overlaps memory loads with Tensor Core compute at the hardware level
- **Block swizzling** for L2 cache line optimization (~6% latency reduction at 16K tokens)
- **Native exp2 with flush-to-zero** — Blackwell's fast-path softmax in hardware
- **Approximate division** with `rounding_mode=APPROX` — hardware reciprocal instead of software division
- **Pi matrix resident in shared memory** (32KB) — stays loaded for the entire fused attention pass, zero reloads

These are real hardware-level wins that Triton on Ampere/Hopper simply cannot replicate. cuTile generates code that's purpose-built for Blackwell's memory hierarchy.

**On-chip V decompression.** His fused kernel decompresses value indices → centroids → un-rotates via Pi → weighted accumulation in a single pass without ever writing decompressed values to HBM. Our Triton fused attention kernel operates on compressed KV directly and avoids the decompress round-trip, but we don't have the same level of on-chip data reuse because Triton's programming model is more constrained than cuTile's explicit tile-level control.

### Where QuashKV Is Stronger (and why)

**1. Portability.** cuTile is locked to Blackwell (B200/B100). QuashKV runs on any GPU with Triton support — A100, H100, L40S, RTX 4090, even older V100s for the PyTorch fallback path. This matters because:
- Most researchers and startups don't have B200 access
- Most cloud GPU instances are still A100/H100
- Reproducibility requires hardware accessibility

**2. Completeness of the paper.** We implement *every* component from the TurboQuant paper:
- MSE quantizer (Algorithm 1) ✅
- Inner product quantizer with QJL (Algorithm 2) ✅
- Mixed-precision / outlier channel splitting (Section 4.2) ✅
- Approximate nearest neighbor search (Section 5) ✅
- Full theoretical bound verification ✅

cuTile implements the core compress/decompress/attention pipeline (which is the most important part) but doesn't cover NN search, mixed precision, or multi-bitwidth configurations.

**3. Compressed perplexity — the metric that actually matters.** Cosine similarity (cuTile reports ~0.985) tells you reconstruction quality, but it doesn't tell you whether the model's output is degraded. We measured actual WikiText-2 perplexity with compressed KV cache:
- Mistral 7B 3-bit: **+0.04 perplexity** (5.06 → 5.10) — effectively lossless
- TinyLlama 3-bit: **+0.12 perplexity** (6.97 → 7.09)
- Mistral 7B 4-bit: **+0.01 perplexity** — indistinguishable from baseline

No other TurboQuant implementation has published compressed perplexity numbers. This is the strongest evidence that the theoretical guarantees translate to real model quality.

**4. Multi-model validation.** cuTile tested on Qwen 2.5-1.5B only. We validated on two architecturally different models (TinyLlama head_dim=64, Mistral 7B head_dim=128), which revealed that **2-bit quality depends heavily on head_dim** — a finding you can't discover from a single model.

**5. Production integration.** QuashKV provides drop-in replacements for HuggingFace's `DynamicCache` and a vLLM attention backend. cuTile's demo is a standalone notebook. If you want to actually deploy compressed KV in a serving pipeline, QuashKV is plug-and-play.

**6. Flexible bit-widths.** 2, 2.5, 3, 3.5, and 4-bit configurations, with per-channel mixed precision that allocates extra bits to outlier channels. cuTile is hardcoded to 3-bit (2-bit MSE + 1-bit QJL for keys, 3-bit MSE for values).

**7. Test coverage.** 175 tests covering every module end-to-end. This isn't about vanity — it means any researcher can fork the repo, change something, and know immediately if they broke the math.

### Where We're Honest About Limitations

**1. Decode speed.** Our compressed generation runs at 2.3–3.4 tok/s (decompress → forward → recompress per token) vs 47–65 tok/s uncompressed. cuTile's fused kernel avoids this by operating directly on compressed data during generation. Our fused Triton attention kernel (benchmarked at 9.4× speedup) isn't yet wired into the generation loop — it's benchmarked standalone. **This is our biggest gap.**

**2. Kernel throughput.** Triton generates good GPU code, but it can't match cuTile's Blackwell-specific optimizations. On equivalent hardware, cuTile would be faster for the core attention+decompress path. We trade peak performance for portability.

**3. Scale.** We validated on TinyLlama 1.1B and Mistral 7B. cuTile's Qwen 2.5-1.5B is comparable in size. Neither implementation has been tested on truly large models (70B+, MoE architectures) where per-layer error accumulation across 80+ layers is the real challenge.

**4. No benchmark on matching hardware.** We can't do an apples-to-apples latency comparison because we don't have B200 access. Our A100 numbers and their B200 numbers measure different things.

### Why We Built This

**The short answer:** Anirudh built the right thing on hardware nobody has. We built the version anyone can use.

TurboQuant is a beautiful paper — near-optimal vector quantization with provable guarantees, elegant math (random rotation → independent coordinates → Lloyd-Max is optimal), and practical KV cache compression. When we saw Anirudh's implementation, three things stood out:

1. **B200-only is a dealbreaker for most people.** The paper's algorithms are hardware-agnostic. The math works on any GPU — why should the implementation be locked to one chip that costs $30K+ and has limited cloud availability?

2. **The paper has more to offer than KV compression.** TurboQuant's NN search application (180,000× faster indexing than Product Quantization) and mixed-precision adaptive quantization are significant contributions that deserved implementation.

3. **Benchmarks need to go deeper.** Cosine similarity and generation demos are great proof-of-concepts, but the research community needs **compressed perplexity** — the actual metric that determines whether you'd deploy this in production. We measured it, and the results (+0.04 PPL at 3-bit on 7B) are strong enough to publish.

The implementations are complementary. If you have a B200 and want maximum throughput, use cuTile. If you want to understand, experiment with, or deploy TurboQuant on accessible hardware, use QuashKV.

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
