# I Implemented the Full TurboQuant Paper on Commodity GPUs

**TL;DR:** I built [QuashKV](https://github.com/RajdeepKushwaha5/QuashKV) — a complete, portable implementation of TurboQuant (ICLR 2026, Google Research) that compresses LLM KV caches to 2–4 bits with mathematically proven near-optimal distortion. It runs on any GPU (A100, H100, 4090), includes fused Triton kernels, NN search, mixed precision, and HuggingFace integration. Validated on TinyLlama 1.1B and Mistral 7B with real perplexity and generation benchmarks.

---

## Why This Matters

Large language models are memory-bound at inference time. The KV cache — the key-value pairs stored during generation — grows linearly with sequence length and can consume tens of gigabytes for long contexts. Compressing this cache is one of the most impactful optimizations for LLM serving.

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) by Zandieh et al. (Google Research, ICLR 2026) provides a theoretically grounded solution: compress each coordinate to 2–4 bits using Lloyd-Max optimal scalar quantizers, with proven **near-optimal distortion** — only 2.72× worse than the information-theoretic lower bound.

I built **QuashKV**, a complete implementation that makes these guarantees real on commodity hardware.

## What QuashKV Does

| Component | What it does |
|-----------|-------------|
| **Lloyd-Max codebooks** | Precomputed optimal scalar quantizers for the exact coordinate PDF |
| **MSE quantizer** | Random rotation Π + per-coordinate Lloyd-Max for values |
| **Inner product quantizer** | (b-1)-bit MSE + 1-bit QJL hash for unbiased key IP estimation |
| **Fused Triton kernels** | Compress, decompress, and fused attention — no KV materialization |
| **Mixed precision** | Per-channel adaptive bit allocation (outlier channels get more bits) |
| **NN search** | Zero-training approximate MIPS with 180,000× faster indexing than PQ |
| **HuggingFace cache** | Drop-in `DynamicCache` replacement |
| **vLLM backend** | `QuashKVModelManager` architectural scaffolding (not yet registered) |

## Benchmark Results (A100-SXM4-80GB)

### Fused Attention Speedup

The fused Triton kernel reads compressed KV directly — no decompression step:

| KV Length | Standard Path | Fused Kernel | Speedup |
|-----------|--------------|-------------|---------|
| 64 | 0.40 ms | 0.20 ms | **2.0×** |
| 256 | 1.06 ms | 0.24 ms | **4.4×** |
| 1,024 | 3.71 ms | 0.50 ms | **7.4×** |
| 4,096 | 14.65 ms | 1.55 ms | **9.4×** |

### WikiText-2 Perplexity

**TinyLlama 1.1B** (baseline: 6.97):
| Bits | Compressed PPL | Δ PPL | Compression | Memory Saved |
|------|---------------|-------|-------------|-------------|
| 2 | 31.56 | +24.59 | 6.40× | 84.4% |
| 3 | **7.09** | **+0.12** | 4.57× | 78.1% |
| 4 | **6.97** | **+0.00** | 3.56× | 71.9% |

**Mistral 7B** (baseline: 5.06):
| Bits | Compressed PPL | Δ PPL | Compression | Memory Saved |
|------|---------------|-------|-------------|-------------|
| 2 | 5.88 | +0.82 | 7.11× | 85.9% |
| 3 | **5.10** | **+0.04** | 4.92× | 79.7% |
| 4 | **5.07** | **+0.01** | 3.76× | 73.4% |

**3-bit adds only +0.04 perplexity on Mistral 7B while saving 80% memory.** 4-bit is effectively indistinguishable from baseline. Note: 2-bit degrades badly on TinyLlama (head_dim=64) but works well on Mistral 7B (head_dim=128) — larger dimensions tolerate aggressive quantization better.

### End-to-End Generation

Side-by-side compressed vs. uncompressed text generation:

**TinyLlama 1.1B (3-bit):**
| Metric | Standard | Compressed |
|--------|----------|-----------|
| Speed | 65 tok/s | 3.3 tok/s |
| KV size | 2.4 MB | 0.5 MB |
| Compression | 1.0× | **4.6×** |

**Mistral 7B (3-bit):**
| Metric | Standard | Compressed |
|--------|----------|-----------|
| Speed | 47 tok/s | 2.3 tok/s |
| KV size | 13.8 MB | 2.8 MB |
| Compression | 1.0× | **4.9×** |

Sample output (Mistral 7B, prompt: *"The key insight behind transformer models is that"*):

> **Standard:** *the order of words in a sentence doesn't matter as much as the relationships between words. This is a powerful idea, but it's also a bit of a double-edged sword…*
>
> **3-bit compressed:** *the order of words in a sentence doesn't matter as much as the relationships between words. This is a powerful idea, and it's the reason why transformer models can be trained on a relatively small amount of data…*

The first 25 tokens are **identical**. Compression noise causes the autoregressive path to diverge, but both outputs are fluent and coherent.

## How It Compares to cuTile

The other known implementation is [Anirudh's cuTile version](https://github.com/DevTechJr/turboquant_cutile) — built on NVIDIA's Blackwell-specific cuTile framework, targeting the B200 GPU.

### Where cuTile wins

**Raw speed.** cuTile compiles to Blackwell Tensor Core instructions with hardware TMA prefetch, block swizzling, native exp2 fast-paths, and Pi-matrix resident in shared memory. His fused attention kernel hits **144.7 tok/s** on B200 vs our 47 tok/s on A100. The hardware gap is real — cuTile exploits Blackwell memory hierarchy features (TMEM→MMA, approximate reciprocal) that Triton on Ampere simply can't access.

**On-chip V decompression.** His fused kernel decompresses values entirely in SRAM — indices → centroids → un-rotate via Pi → weighted sum, never touching HBM. That's the theoretical ideal for bandwidth-bound attention.

### Where QuashKV wins

**Portability.** Runs on A100, H100, L40S, RTX 4090 — any GPU with Triton support. Most researchers and cloud instances don't have B200 access. The paper's math is hardware-agnostic; the implementation should be too.

**Compressed perplexity — the metric that matters.** cuTile reports ~0.985 cosine similarity, but cosine doesn't tell you if the model's output quality degrades. We measured actual WikiText-2 perplexity with compressed KV: Mistral 7B 3-bit adds only **+0.04 perplexity** while saving 80% memory. No other implementation has published this number.

**Paper completeness.** We implement every component: MSE quantizer, IP quantizer with QJL, mixed-precision outlier handling, NN search (180,000× faster indexing than PQ), and flexible 2/2.5/3/3.5/4-bit configs. cuTile covers the core pipeline at 3-bit only.

**Production integration.** Drop-in `DynamicCache` replacement for HuggingFace. vLLM backend is architecturally implemented but not yet registered — cuTile's demo is a standalone notebook.

**Multi-model validation.** Tested on TinyLlama (head_dim=64) and Mistral 7B (head_dim=128), revealing that 2-bit quality depends heavily on head_dim — invisible from a single-model test.

### Honest limitations

**Our decode speed is slow** — 2.3–3.4 tok/s with per-token decompress→forward→recompress. The fused Triton kernel (9.4× speedup) isn't yet wired into the generation loop. This is our biggest gap.

**We can't benchmark on matching hardware.** A100 vs B200 numbers measure different things.

### Why build it?

Anirudh built the right thing on hardware nobody has. We built the version anyone can use. The implementations are complementary — B200 + max throughput → cuTile; accessible hardware + experiments + deployment → QuashKV.

## Architecture

```
quashkv/
├── codebook.py          # Lloyd-Max optimal scalar quantizer
├── quantizer.py         # MSEQuantizer + InnerProductQuantizer
├── engine.py            # QuashKVEngine orchestrator
├── packing.py           # Bit-packing (1-4 bit → bytes)
├── mixed_precision.py   # Per-channel adaptive bit allocation
├── integrations/
│   ├── hf_cache.py      # HuggingFace DynamicCache replacement
│   └── vllm_backend.py  # vLLM backend scaffolding (not yet registered)
├── nn_search/
│   └── index.py         # QuashIndex for vector search
└── triton_kernels/
    ├── compress.py      # Fused normalize→rotate→quantize
    ├── decompress.py    # Fused dequantize→unrotate→rescale
    └── attention.py     # Online-softmax quantized attention
```

16 source files, ~3,280 lines of code, 175 tests.

## Theoretical Guarantees (Verified)

From the paper:

- **MSE distortion:** ≤ √(3π/2) · 4^{-b} — verified via cosine similarity matching theoretical predictions
- **IP estimator:** Unbiased — E[⟨y, x̃⟩] = ⟨y, x⟩ — verified experimentally
- **Optimality gap:** upper/lower bound ratio ≈ 2.72 — near-optimal

## Quick Start

```bash
pip install -e ".[triton]"
```

```python
from quashkv import QuashKVEngine

engine = QuashKVEngine(head_dim=128, key_bits=3, value_bits=3)
engine.append(keys, values)           # compress & store
output = engine.attention(queries)    # attend over compressed cache

print(f"Compression: {engine.compression_ratio():.1f}x")
# → Compression: 4.9x
```

## What I Learned

1. **The rotation matrix is everything.** Random orthogonal rotation before quantization makes coordinates independent and approximately Gaussian — this is what enables per-coordinate Lloyd-Max to be near-optimal.

2. **QJL is elegant.** A single random sign bit provides an unbiased inner product estimator. The proof is beautiful: the sign of ⟨s, Πx⟩ corrects the bias from MSE quantization.

3. **3-bit is the sweet spot.** 2-bit works for memory saving but degrades generation quality. 4-bit is nearly lossless but less impressive compression. 3-bit gives 4.6–4.9× compression with coherent output.

4. **Transformers 5.x broke everything.** The HuggingFace DynamicCache API changed significantly between 4.x and 5.x. Custom cache wrappers need to use `DynamicCache.update()` — it's the only stable method.

5. **Triton is a game-changer for accessibility.** Writing fused kernels in Triton took days instead of weeks compared to CUDA. The 9.4× attention speedup runs on any GPU with Triton support.

## Links

- **GitHub:** [github.com/RajdeepKushwaha5/QuashKV](https://github.com/RajdeepKushwaha5/QuashKV)
- **Paper:** [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- **Benchmarks:** Reproducible scripts in `benchmarks/` directory

---

*Built as an independent implementation of TurboQuant (ICLR 2026). Not affiliated with Google Research.*
