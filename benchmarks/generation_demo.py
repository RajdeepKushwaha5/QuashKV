#!/usr/bin/env python3
"""
QuashKV End-to-End Generation Demo.

Loads a real model, generates text with both uncompressed and compressed
KV caches, and compares output quality, speed, and memory usage.

Usage (RunPod A100):
    python benchmarks/generation_demo.py
    python benchmarks/generation_demo.py --model mistralai/Mistral-7B-v0.3 --dtype float16
    python benchmarks/generation_demo.py --bits 2 --max-new-tokens 100
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quashkv.integrations.hf_cache import QuashKVCache, QuashKVCacheConfig
from quashkv.triton_kernels import HAS_TRITON


# ======================================================================
# Args
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="QuashKV generation demo")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--bits", type=int, default=3,
                   help="Bit width for key and value compression")
    p.add_argument("--max-new-tokens", type=int, default=100,
                   help="Number of tokens to generate per prompt")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ======================================================================
# Memory helpers
# ======================================================================

def gpu_memory_mb():
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def gpu_peak_memory_mb():
    """Return peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


# ======================================================================
# Standard generation (no compression)
# ======================================================================

@torch.no_grad()
def generate_standard(model, input_ids, max_new_tokens, temperature=1.0):
    """Generate tokens using the model's native KV cache (no compression)."""
    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    mem_before = gpu_memory_mb()
    t_start = time.perf_counter()

    generated = input_ids.to(device)
    past_key_values = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            # Prefill: full prompt
            outputs = model(input_ids=generated, use_cache=True)
        else:
            # Decode: single new token
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == model.config.eos_token_id:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    mem_after = gpu_peak_memory_mb()
    n_generated = generated.shape[1] - input_ids.shape[1]

    return {
        "token_ids": generated,
        "n_generated": n_generated,
        "elapsed": elapsed,
        "tok_per_sec": n_generated / elapsed if elapsed > 0 else 0,
        "peak_memory_mb": mem_after,
        "kv_memory_mb": mem_after - mem_before,
    }


# ======================================================================
# KV extraction helper (transformers version-compatible)
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

    # Method 2: to_legacy_cache() (some transformers versions)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [(legacy[i][0], legacy[i][1]) for i in range(num_layers)]

    # Method 3: iteration
    try:
        pairs = []
        for item in past_key_values:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
            elif hasattr(item, "key") and hasattr(item, "value"):
                pairs.append((item.key, item.value))
            elif hasattr(item, "key_state") and hasattr(item, "value_state"):
                pairs.append((item.key_state, item.value_state))
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

    # Method 5: probe internal attributes
    for attr in ("_cache", "cache", "_data", "layers"):
        if hasattr(past_key_values, attr):
            internal = getattr(past_key_values, attr)
            if isinstance(internal, (list, tuple)) and len(internal) >= num_layers:
                pairs = []
                for item in internal:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        pairs.append((item[0], item[1]))
                    elif hasattr(item, "key") and hasattr(item, "value"):
                        pairs.append((item.key, item.value))
                    elif hasattr(item, "keys") and hasattr(item, "values"):
                        pairs.append((item.keys, item.values))
                if len(pairs) == num_layers:
                    return pairs

    raise RuntimeError(
        f"Cannot extract KV pairs from {type(past_key_values).__name__}. "
        f"Attrs: {[a for a in dir(past_key_values) if not a.startswith('__')]}"
    )


# ======================================================================
# Compressed generation (QuashKV)
# ======================================================================

@torch.no_grad()
def generate_compressed(model, input_ids, max_new_tokens, bits, seed,
                        temperature=1.0):
    """Generate tokens using QuashKV compressed KV cache.

    Strategy: manual token-by-token generation with our custom cache.
    On each forward pass, the model computes fresh K, V from the current
    token. We intercept via a hook — but the simplest approach is:

    1. Prefill: run full prompt through the model, extract KV cache
    2. Compress the KV cache into QuashKVCache
    3. Decode loop: for each new token, run model forward, compress
       new KV, decompress full cache for attention
    """
    device = next(model.parameters()).device
    model_config = model.config
    num_layers = model_config.num_hidden_layers

    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_config.hidden_size // model_config.num_attention_heads

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    mem_before = gpu_memory_mb()
    t_start = time.perf_counter()

    # --- Prefill: get KV cache from full prompt ---
    prompt = input_ids.to(device)
    outputs = model(input_ids=prompt, use_cache=True)
    last_logits = outputs.logits[:, -1, :]

    # Extract native KV pairs (version-compatible)
    native_kv = outputs.past_key_values
    kv_pairs = _extract_kv_pairs(native_kv, num_layers)

    # Build compressed cache
    config = QuashKVCacheConfig(key_bits=bits, value_bits=bits, seed=seed)
    cache = QuashKVCache(
        num_layers=num_layers,
        head_dim=head_dim,
        config=config,
        device=str(device),
    )

    # Compress the prefill KV
    for layer_idx, (k, v) in enumerate(kv_pairs):
        cache.update(k.float(), v.float(), layer_idx=layer_idx)

    # Delete native KV to free memory
    del native_kv, kv_pairs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_prefill = time.perf_counter() - t_start

    # --- Decode loop ---
    generated = prompt.clone()

    t_prefill = time.perf_counter() - t_start

    # Now decode token by token
    for step in range(max_new_tokens):
        # Sample next token from logits
        if temperature <= 0:
            next_token = last_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == model_config.eos_token_id:
            break

        # Forward pass with just the new token
        # We need to provide the decompressed KV as past_key_values
        # Use tuple-of-tuples format which is universally supported
        model_dtype = next(model.parameters()).dtype

        past_kv_tuple = tuple(
            (k.to(model_dtype), v.to(model_dtype))
            for layer_idx in range(num_layers)
            for k, v in [cache[layer_idx]]
        )

        outputs = model(
            input_ids=next_token,
            past_key_values=past_kv_tuple,
            use_cache=True,
        )
        last_logits = outputs.logits[:, -1, :]

        # Extract the new KV pair (only the new single token) and compress
        new_kv = outputs.past_key_values
        new_kv_pairs = _extract_kv_pairs(new_kv, num_layers)

        for layer_idx, (nk, nv) in enumerate(new_kv_pairs):
            # Cache already has old tokens; only compress the last (new) one
            cache.update(nk[:, :, -1:, :].float(), nv[:, :, -1:, :].float(),
                         layer_idx=layer_idx)

        del past_kv_tuple, outputs, new_kv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    n_generated = generated.shape[1] - input_ids.shape[1]

    # Compression stats
    stats = cache.compression_stats()

    return {
        "token_ids": generated,
        "n_generated": n_generated,
        "elapsed": elapsed,
        "prefill_time": t_prefill,
        "tok_per_sec": n_generated / elapsed if elapsed > 0 else 0,
        "peak_memory_mb": gpu_peak_memory_mb(),
        "kv_memory_mb": gpu_peak_memory_mb() - mem_before,
        "compression_ratio": stats["overall_compression_ratio"],
        "cache_stats": stats,
    }


# ======================================================================
# Output comparison
# ======================================================================

def compare_outputs(tokens_std, tokens_cmp, prompt_len):
    """Compare generation overlap between standard and compressed."""
    gen_std = tokens_std[0, prompt_len:].tolist()
    gen_cmp = tokens_cmp[0, prompt_len:].tolist()

    min_len = min(len(gen_std), len(gen_cmp))
    if min_len == 0:
        return {"match_rate": 0.0, "first_divergence": 0}

    matches = sum(1 for a, b in zip(gen_std[:min_len], gen_cmp[:min_len]) if a == b)
    first_diff = next((i for i, (a, b) in enumerate(zip(gen_std, gen_cmp)) if a != b),
                      min_len)

    return {
        "match_rate": matches / min_len,
        "first_divergence": first_diff,
        "overlap_tokens": min_len,
    }


# ======================================================================
# Main
# ======================================================================

PROMPTS = [
    "The future of artificial intelligence is",
    "In a small village nestled between mountains,",
    "The key insight behind transformer models is that",
]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("  QuashKV End-to-End Generation Demo")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Compression:    {args.bits}-bit (key: {args.bits-1}b MSE + 1b QJL, "
          f"value: {args.bits}b MSE)")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Device:         {args.device}")
    print(f"  Triton:         {'available' if HAS_TRITON else 'not available'}")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
    )
    model.eval()
    model_mem = gpu_memory_mb()
    print(f"  Loaded in {time.time() - t0:.1f}s  "
          f"(model memory: {model_mem:.0f} MB)")
    print()

    # ---- Generate for each prompt ----
    for i, prompt in enumerate(PROMPTS):
        print("=" * 70)
        print(f"  Prompt {i+1}/{len(PROMPTS)}")
        print("=" * 70)
        print(f'  "{prompt}"')
        print()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = input_ids.shape[1]

        # --- Standard (uncompressed) generation ---
        print("  [1/2] Standard generation (no compression)...")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        result_std = generate_standard(
            model, input_ids, args.max_new_tokens, temperature=0,
        )

        text_std = tokenizer.decode(
            result_std["token_ids"][0, prompt_len:], skip_special_tokens=True
        )
        print(f"    Tokens: {result_std['n_generated']}  |  "
              f"Time: {result_std['elapsed']:.2f}s  |  "
              f"Speed: {result_std['tok_per_sec']:.1f} tok/s  |  "
              f"Peak mem: {result_std['peak_memory_mb']:.0f} MB")

        # --- Compressed generation ---
        print(f"  [2/2] Compressed generation ({args.bits}-bit QuashKV)...")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        result_cmp = generate_compressed(
            model, input_ids, args.max_new_tokens,
            bits=args.bits, seed=args.seed, temperature=0,
        )

        text_cmp = tokenizer.decode(
            result_cmp["token_ids"][0, prompt_len:], skip_special_tokens=True
        )
        print(f"    Tokens: {result_cmp['n_generated']}  |  "
              f"Time: {result_cmp['elapsed']:.2f}s  |  "
              f"Speed: {result_cmp['tok_per_sec']:.1f} tok/s  |  "
              f"Compression: {result_cmp['compression_ratio']:.1f}x")

        # --- Comparison ---
        comp = compare_outputs(
            result_std["token_ids"], result_cmp["token_ids"], prompt_len
        )

        print()
        print("  --- Standard Output ---")
        print(f"  {text_std[:300]}")
        print()
        print(f"  --- Compressed Output ({args.bits}-bit) ---")
        print(f"  {text_cmp[:300]}")
        print()
        print(f"  Token match rate: {comp['match_rate']:.1%}  "
              f"(first divergence at token {comp['first_divergence']})")
        print()

    # ---- Summary ----
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)
    bits = args.bits
    seq_len = prompt_len + args.max_new_tokens

    fp16_kv_mb = (2 * n_layers * n_kv_heads * seq_len * head_dim * 2) / (1024 * 1024)
    compressed_kv_mb = fp16_kv_mb / result_cmp["compression_ratio"] if result_cmp["compression_ratio"] > 0 else fp16_kv_mb

    print(f"  Model:              {args.model}")
    print(f"  Layers:             {n_layers}")
    print(f"  KV heads:           {n_kv_heads}")
    print(f"  Head dim:           {head_dim}")
    print(f"  Sequence length:    {seq_len} tokens")
    print(f"  Compression:        {bits}-bit → {result_cmp['compression_ratio']:.1f}x")
    print(f"  FP16 KV cache:      {fp16_kv_mb:.1f} MB")
    print(f"  Compressed KV:      {compressed_kv_mb:.1f} MB")
    print(f"  Memory saved:       {fp16_kv_mb - compressed_kv_mb:.1f} MB "
          f"({(1 - compressed_kv_mb / fp16_kv_mb) * 100:.0f}%)")
    print()
    print(f"  Standard speed:     {result_std['tok_per_sec']:.1f} tok/s")
    print(f"  Compressed speed:   {result_cmp['tok_per_sec']:.1f} tok/s")
    print()
    print("=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
