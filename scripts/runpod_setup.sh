#!/bin/bash
# ==========================================================
# QuashKV RunPod A100 Setup & Benchmark Script
#
# Usage:
#   1. Create a RunPod pod with:
#        - GPU: A100 80GB (or A100 40GB)
#        - Template: RunPod PyTorch 2.x
#        - Disk: 50GB
#
#   2. SSH into the pod or use the web terminal
#
#   3. Run this script:
#        bash scripts/runpod_setup.sh
#
#   4. Or run steps manually (see sections below)
# ==========================================================

set -e

echo "=========================================="
echo "  QuashKV RunPod Setup"
echo "=========================================="

# --- 1. Clone repo ---
if [ ! -d "QuashKV" ]; then
    echo "[1/5] Cloning QuashKV..."
    git clone https://github.com/RajdeepKushwaha5/QuashKV.git
else
    echo "[1/5] QuashKV already cloned, pulling latest..."
    cd QuashKV && git pull && cd ..
fi

cd QuashKV

# --- 2. Install dependencies ---
echo "[2/5] Installing dependencies..."
pip install -e ".[triton]" 2>&1 | tail -5
pip install datasets transformers accelerate 2>&1 | tail -5

# --- 3. Verify setup ---
echo "[3/5] Verifying setup..."
python -c "
import torch
import triton
from quashkv.triton_kernels import HAS_TRITON
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'  Triton:   {HAS_TRITON}')
print(f'  GPU Mem:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- 4. Run tests on GPU ---
echo "[4/5] Running tests..."
python -m pytest tests/ -q --tb=short 2>&1 | tail -5

# --- 5. Run benchmarks ---
echo "[5/5] Running benchmarks..."
echo ""
echo "=========================================="
echo "  Step A: Latency benchmark (Triton vs PyTorch)"
echo "=========================================="
python benchmarks/latency_bench.py --device cuda --triton-compare

echo ""
echo "=========================================="
echo "  Step B: Perplexity eval — TinyLlama (quick sanity check)"
echo "=========================================="
python benchmarks/perplexity_eval.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --bits 2 3 4 \
    --max-tokens 2048

echo ""
echo "=========================================="
echo "  Step C: Perplexity eval — Mistral 7B (the real test)"
echo "=========================================="
python benchmarks/perplexity_eval.py \
    --model mistralai/Mistral-7B-v0.3 \
    --bits 2 3 4 \
    --max-tokens 4096 \
    --dtype float16

echo ""
echo "=========================================="
echo "  ALL DONE! Copy the output above for README."
echo "=========================================="
