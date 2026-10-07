#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Gevva v2 Pipeline Queue ==="

# Step 1: Ensure e2b GGUF is exported
if [ ! -f "ckpt/gevva-v2-e2b/best/best-q4_k_m.gguf" ]; then
    echo "[$(date)] Step 1/3: Exporting Gevva v2 e2b to Q4_K_M GGUF..."
    uv run python scripts/export_gevva_v2_gguf.py \
        --model-dir ckpt/gevva-v2-e2b/best \
        --convert \
        --quant Q4_K_M
else
    echo "[$(date)] Step 1/3: Gevva v2 e2b GGUF already exported at ckpt/gevva-v2-e2b/best/best-q4_k_m.gguf."
fi

# Step 2: Launch e4b training with mandatory QAT and PagedAdamW8bit
echo "[$(date)] Step 2/3: Launching Gevva v2 e4b training with mandatory QAT and PagedAdamW8bit..."
uv run python train_gevva_v2.py \
    --model-id google/gemma-4-E4B-it \
    --data data/gevva_v2_train.jsonl \
    --out-dir ckpt/gevva-v2-e4b \
    --target-quant q4_k_m \
    --qat-bits 4 \
    --qat-group-size 32 \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 1.5e-5 \
    --use-8bit-adam

# Step 3: Export e4b to Q4_K_M GGUF
echo "[$(date)] Step 3/3: Exporting Gevva v2 e4b to Q4_K_M GGUF..."
uv run python scripts/export_gevva_v2_gguf.py \
    --model-dir ckpt/gevva-v2-e4b/best \
    --convert \
    --quant Q4_K_M

echo "[$(date)] === Gevva v2 e2b & e4b Full Pipeline Complete! ==="
