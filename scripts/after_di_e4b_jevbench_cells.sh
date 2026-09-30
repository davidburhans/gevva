#!/usr/bin/env bash
# after_di_e4b_jevbench_cells.sh - Post-Decision-Index GPU queue (user order, 2026-09-29):
#   1. e4b flagship @ T=1.0   (raw-vs-raw comparison with r3; user priority)
#   2. e4b r3 @ T=1.6         (calibrated cell)
#   3. RAGTruth verdicts (fixed parser: added-rows.jsonl.gz + bool labels)
#      for e2b-1.1 and e4b-r3
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[pipeline] cells watcher v2 armed $(date)" >> "$LOG"

while ! grep -q "decision index e2b done" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[pipeline] GPU free; running comparison cells $(date)" >> "$LOG"

uv run python scripts/eval_jevbench_public.py \
  --model-path ckpt/gevva-e4b-flagship/best --temperature 1.0 \
  --out results/jevbench_e4b_flagship_t10.json \
  > results/jevbench_e4b_flagship_t10.log 2>&1 || echo "[pipeline] WARN flagship t1.0 failed" >> "$LOG"
echo "[pipeline] flagship@1.0 done $(date)" >> "$LOG"

uv run python scripts/eval_jevbench_public.py \
  --model-path ckpt/gevva-e4b-phase5-r3/best --temperature 1.6 \
  --out results/jevbench_e4b_r3_t16.json \
  > results/jevbench_e4b_r3_t16.log 2>&1 || echo "[pipeline] WARN r3 t1.6 failed" >> "$LOG"
echo "[pipeline] r3@1.6 done $(date)" >> "$LOG"

uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_e2b_ragtruth.json > results/phase5_e2b_ragtruth.log 2>&1 || echo "[pipeline] WARN e2b ragtruth failed" >> "$LOG"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-phase5-r3/best \
  --out results/phase5_r3_ragtruth.json > results/phase5_r3_ragtruth.log 2>&1 || echo "[pipeline] WARN r3 ragtruth failed" >> "$LOG"
echo "[pipeline] ragtruth verdicts (rerun) done $(date)" >> "$LOG"
