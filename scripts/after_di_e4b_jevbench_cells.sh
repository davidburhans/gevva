#!/usr/bin/env bash
# after_di_e4b_jevbench_cells.sh - Fills the two missing JevBench comparison cells
# once the e2b Decision Index run releases the GPU:
#   1. e4b flagship @ T=1.0 (never measured raw)
#   2. e4b r3 @ T=1.6 (never measured at the calibrated temp)
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[pipeline] e4b-jevbench-cells watcher armed $(date)" >> "$LOG"

# Wait for the e2b Decision Index stage to complete (last stage of the e2b chain)
while ! grep -q "decision index e2b done" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[pipeline] GPU free; running e4b jevbench cells $(date)" >> "$LOG"

uv run python scripts/eval_jevbench_public.py \
  --model-path ckpt/gevva-e4b-flagship/best --temperature 1.0 \
  --out results/jevbench_e4b_flagship_t10.json \
  > results/jevbench_e4b_flagship_t10.log 2>&1 || echo "[pipeline] WARN flagship t1.0 failed" >> "$LOG"

uv run python scripts/eval_jevbench_public.py \
  --model-path ckpt/gevva-e4b-phase5-r3/best --temperature 1.6 \
  --out results/jevbench_e4b_r3_t16.json \
  > results/jevbench_e4b_r3_t16.log 2>&1 || echo "[pipeline] WARN r3 t1.6 failed" >> "$LOG"

echo "[pipeline] e4b jevbench cells done $(date)" >> "$LOG"
