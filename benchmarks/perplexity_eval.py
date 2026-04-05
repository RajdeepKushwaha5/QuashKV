#!/usr/bin/env python3
"""
QuashKV perplexity & quality evaluation.

Measures:
  1. Baseline perplexity on WikiText-2 (no compression)
  2. Per-layer KV cache compression quality at various bit widths
  3. Compression ratio and memory savings

Usage (RunPod A100):
    python benchmarks/perplexity_eval.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --bits 2 3 4

    python benchmarks/perplexity_eval.py \\
        --model meta-llama/Llama-3.1-8B \\
        --bits 3 \\
        --max-tokens 4096
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure quashkv is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quashkv.engine import QuashKVEngine
from quashkv.integrations.hf_cache import QuashKVCache, QuashKVCacheConfig
from quashkv.triton_kernels import HAS_TRITON


# ======================================================================
# Helpers
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="QuashKV perplexity evaluation")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4],
                   help="Bit widths to evaluate (key_bits = value_bits)")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max tokens from WikiText-2 to process")
    p.add_argument("--stride", type=int, default=512,
                   help="Stride for sliding-window perplexity")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default=None,
                   help="Device (default: cuda if available, else cpu)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compressed-ppl", action="store_true",
                   help="Measure actual perplexity with compressed KV cache")
    return p.parse_args()


def load_wikitext2(tokenizer, max_tokens=None):
    """Load and tokenize WikiText-2 test set."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    ids = encodings.input_ids
    if max_tokens and ids.size(1) > max_tokens:
        ids = ids[:, :max_tokens]
    return ids


# ======================================================================
# Baseline perplexity (no compression)
# ======================================================================

def baseline_perplexity(model, input_ids, max_length, stride, device):
    """Standard sliding-window perplexity on WikiText-2."""
    seq_len = input_ids.size(1)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        target = chunk.clone()
        trg_len = end - prev_end
        if trg_len < chunk.size(1):
            target[:, :-trg_len] = -100

        with torch.no_grad():
            loss = model(chunk, labels=target).loss

        n = (target != -100).sum().item()
        total_nll += loss.item() * n
        total_tokens += n
        prev_end = end

        if end == seq_len:
            break

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ======================================================================
# KV cache extraction
# ======================================================================

def extract_kv_layers(outputs):
    """Extract list of (keys, values) from model outputs.

    Handles DynamicCache (transformers >=4.36), tuple-based cache, and
    the (key, value, None) format from transformers 5.x.
    """
    kv = outputs.past_key_values
    if kv is None:
        return []

    # DynamicCache with explicit key_cache / value_cache attributes
    if hasattr(kv, "key_cache"):
        return [
            (kv.key_cache[i], kv.value_cache[i])
            for i in range(len(kv.key_cache))
        ]

    # Tuple / list of layer entries
    layers = []
    for item in kv:
        if isinstance(item, (list, tuple)):
            layers.append((item[0], item[1]))
    return layers


# ======================================================================
# Per-layer KV quality
# ======================================================================

def evaluate_kv_quality(model, input_ids, bits, seed, max_length, device):
    """Compress and decompress KV cache, measure quality per layer."""
    chunk = input_ids[:, :max_length].to(device)

    with torch.no_grad():
        outputs = model(chunk, use_cache=True)

    layers = extract_kv_layers(outputs)
    if not layers:
        print("  WARNING: could not extract KV cache from model outputs")
        return []

    results = []
    for layer_idx, (keys, values) in enumerate(layers):
        head_dim = keys.shape[-1]

        engine = QuashKVEngine(
            head_dim=head_dim,
            key_bits=bits,
            value_bits=bits,
            seed=seed + layer_idx,
            device=str(keys.device),
        )

        kf = keys.float()
        vf = values.float()
        engine.append(kf, vf)

        # Decompress
        block = engine._cache[0]
        mse_idx, _, val_idx = engine._unpack_block(block)

        # Key reconstruction (MSE centroid only; QJL used for scoring)
        k_hat = engine.key_quantizer.mse_quantizer.codebook.dequantize(mse_idx)
        k_unrot = engine.key_quantizer.mse_quantizer.unrotate(k_hat)
        k_rec = k_unrot * block.key_norms.unsqueeze(-1)

        # Value reconstruction
        v_hat = engine.value_quantizer.codebook.dequantize(val_idx)
        v_unrot = engine.value_quantizer.unrotate(v_hat)
        v_rec = v_unrot * block.val_norms.unsqueeze(-1)

        kcos = F.cosine_similarity(kf.flatten(), k_rec.flatten(), dim=0).item()
        vcos = F.cosine_similarity(vf.flatten(), v_rec.flatten(), dim=0).item()

        results.append({
            "layer": layer_idx,
            "key_cos": kcos,
            "val_cos": vcos,
            "key_mse": (kf - k_rec).pow(2).mean().item(),
            "val_mse": (vf - v_rec).pow(2).mean().item(),
        })

        # Free GPU memory
        del engine, block, mse_idx, val_idx
        del k_hat, k_unrot, k_rec, v_hat, v_unrot, v_rec
        if keys.is_cuda:
            torch.cuda.empty_cache()

    return results


# ======================================================================
# Robust KV extraction (transformers 4.x / 5.x compatible)
# ======================================================================

def _extract_kv_pairs(past_key_values, num_layers):
    """Extract list of (key, value) tensors from model's past_key_values.

    Works with DynamicCache (4.36+), transformers 5.x, and legacy tuples.
    """
    # Method 1: key_cache / value_cache lists (transformers 4.36-4.x)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        kc = past_key_values.key_cache
        vc = past_key_values.value_cache
        if isinstance(kc, list) and len(kc) > 0:
            return [(kc[i], vc[i]) for i in range(num_layers)]

    # Method 2: to_legacy_cache()
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [(legacy[i][0], legacy[i][1]) for i in range(num_layers)]

    # Method 3: iteration
    try:
        pairs = []
        for item in past_key_values:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
            else:
                k, v = item
                pairs.append((k, v))
        if len(pairs) == num_layers:
            return pairs
    except (TypeError, ValueError):
        pass

    # Method 4: direct subscript
    try:
        return [(past_key_values[i][0], past_key_values[i][1])
                for i in range(num_layers)]
    except (TypeError, KeyError, IndexError):
        pass

    raise RuntimeError(
        f"Cannot extract KV pairs from {type(past_key_values).__name__}. "
        f"Attrs: {[a for a in dir(past_key_values) if not a.startswith('__')]}"
    )


# ======================================================================
# Compressed perplexity
# ======================================================================

@torch.no_grad()
def compressed_perplexity(model, input_ids, bits, seed, max_length, stride, device):
    """Measure perplexity with compressed KV cache.

    For each sliding window, the prefix (context) KV cache is compressed
    and decompressed before evaluating the loss on the remaining tokens.
    This measures the actual impact of KV compression on model quality.
    """
    from transformers import DynamicCache

    seq_len = input_ids.size(1)
    num_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    model_dtype = next(model.parameters()).dtype

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        trg_len = end - prev_end    # tokens to evaluate (matches baseline)
        ctx_len = chunk.size(1) - trg_len  # context tokens (to compress)

        if ctx_len <= 0:
            # No context to compress — use normal forward (matches baseline)
            target = chunk.clone()
            loss = model(chunk, labels=target).loss
            n = (target != -100).sum().item()
            total_nll += loss.item() * n
            total_tokens += n
        else:
            # Step 1: forward on prefix to get KV cache
            prefix = chunk[:, :ctx_len]
            eval_tokens = chunk[:, ctx_len:]

            prefix_out = model(prefix, use_cache=True)
            kv_pairs = _extract_kv_pairs(prefix_out.past_key_values, num_layers)
            del prefix_out

            # Step 2: compress and decompress KV
            config = QuashKVCacheConfig(key_bits=bits, value_bits=bits, seed=seed)
            cache = QuashKVCache(
                num_layers=num_layers, head_dim=head_dim,
                config=config, device=str(device),
            )
            for layer_idx, (k, v) in enumerate(kv_pairs):
                cache.update(k.float(), v.float(), layer_idx=layer_idx)
            del kv_pairs

            # Step 3: build real DynamicCache from decompressed KV
            hf_cache = DynamicCache()
            for layer_idx in range(num_layers):
                k_dec, v_dec = cache[layer_idx]
                hf_cache.update(
                    k_dec.to(model_dtype), v_dec.to(model_dtype), layer_idx
                )
            del cache

            # Step 4: forward eval tokens with compressed cache
            labels = eval_tokens.clone()
            outputs = model(
                eval_tokens, past_key_values=hf_cache, labels=labels
            )
            n = (labels != -100).sum().item()
            total_nll += outputs.loss.item() * n
            total_tokens += n

            del hf_cache, outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prev_end = end
        if end == seq_len:
            break

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ======================================================================
# Printing
# ======================================================================

def print_quality_table(results, bits, head_dim):
    """Pretty-print per-layer quality and compression ratio."""
    print(f"\n{'=' * 65}")
    print(f"  KV Cache Quality — {bits}-bit compression")
    print(f"{'=' * 65}")
    print(f"  {'Layer':>5} | {'Key Cos':>8} | {'Val Cos':>8}"
          f" | {'Key MSE':>10} | {'Val MSE':>10}")
    print(f"  {'-' * 55}")

    for r in results:
        print(f"  {r['layer']:>5} | {r['key_cos']:>8.4f} | {r['val_cos']:>8.4f}"
              f" | {r['key_mse']:>10.6f} | {r['val_mse']:>10.6f}")

    avg_k = sum(r["key_cos"] for r in results) / len(results)
    avg_v = sum(r["val_cos"] for r in results) / len(results)
    print(f"  {'-' * 55}")
    print(f"  {'AVG':>5} | {avg_k:>8.4f} | {avg_v:>8.4f}")

    # Compression ratio (per-token, both KV)
    orig_bits = 2 * head_dim * 16                        # fp16 key + value
    comp_bits = (bits * head_dim + 32) * 2               # indices + norms × 2
    ratio = orig_bits / comp_bits
    pct_saved = (1 - 1 / ratio) * 100
    print(f"\n  Compression ratio: {ratio:.2f}x  ({pct_saved:.1f}% memory saved)")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Model:      {args.model}")
    print(f"  Bit widths: {args.bits}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Device:     {args.device}")
    print(f"  Triton:     {'available' if HAS_TRITON else 'not available'}")
    print()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # ---- Load model ----
    print("  Loading model & tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype_map[args.dtype],
            device_map=args.device,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map=args.device,
        )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Load data ----
    print("  Loading WikiText-2 test set...")
    input_ids = load_wikitext2(tokenizer, max_tokens=args.max_tokens)
    print(f"  {input_ids.size(1)} tokens\n")

    # ---- 1. Baseline perplexity ----
    print(f"{'=' * 65}")
    print("  Baseline Perplexity (no compression)")
    print(f"{'=' * 65}")
    max_len = min(
        2048,
        getattr(model.config, "max_position_embeddings", 2048),
    )
    t0 = time.time()
    ppl = baseline_perplexity(model, input_ids, max_len, args.stride, args.device)
    print(f"  Perplexity: {ppl:.2f}  ({time.time() - t0:.1f}s)")

    # ---- 2. KV quality at each bit width ----
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads

    for bits in args.bits:
        t0 = time.time()
        results = evaluate_kv_quality(
            model, input_ids,
            bits=bits,
            seed=args.seed,
            max_length=min(args.max_tokens, max_len),
            device=args.device,
        )
        if results:
            print_quality_table(results, bits, head_dim)
            print(f"  Eval time: {time.time() - t0:.1f}s")

    # ---- 3. Compressed perplexity (optional) ----
    if args.compressed_ppl:
        print(f"\n{'=' * 65}")
        print("  Compressed Perplexity (KV cache compressed per window)")
        print(f"{'=' * 65}")
        for bits in args.bits:
            t0 = time.time()
            try:
                cppl = compressed_perplexity(
                    model, input_ids,
                    bits=bits,
                    seed=args.seed,
                    max_length=max_len,
                    stride=args.stride,
                    device=args.device,
                )
                elapsed = time.time() - t0
                delta = cppl - ppl
                print(f"  {bits}-bit:  {cppl:.2f}  "
                      f"(Δ = +{delta:.2f} vs baseline {ppl:.2f})  "
                      f"({elapsed:.1f}s)")
            except Exception as e:
                print(f"  {bits}-bit:  ERROR — {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 65}")
    print("  Done!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
