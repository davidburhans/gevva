#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

E2B_PID="${1:-3730494}"

echo "=== Gevva v2 Automated Pipeline Queue ==="
echo "Monitoring running e2b PID: $E2B_PID..."

while kill -0 "$E2B_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Gevva v2 e2b training completed. Checking best checkpoint..."
if [ ! -d "ckpt/gevva-v2-e2b/best" ]; then
    echo "ERROR: ckpt/gevva-v2-e2b/best not found! Aborting pipeline."
    exit 1
fi

echo "[$(date)] Step 1/3: Exporting Gevva v2 e2b to Q4_K_M GGUF..."
uv run python scripts/export_gevva_v2_gguf.py \
    --model-dir ckpt/gevva-v2-e2b/best \
    --convert \
    --quant Q4_K_M

echo "[$(date)] Step 2/3: Launching Gevva v2 e4b training with mandatory QAT..."
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
    --lr 1.5e-5

echo "[$(date)] Step 3/3: Exporting Gevva v2 e4b to Q4_K_M GGUF..."
uv run python scripts/export_gevva_v2_gguf.py \
    --model-dir ckpt/gevva-v2-e4b/best \
    --convert \
    --quant Q4_K_M

echo "[$(date)] === Gevva v2 e2b & e4b Full Pipeline Complete! ==="
