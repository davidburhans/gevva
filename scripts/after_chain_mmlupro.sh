#!/usr/bin/env bash
# after_chain_mmlupro.sh - deferred K-robustness A/B once the v2 chain completes:
# phase5 (cross-option OFF) vs xopt (cross-option ON) vs e2b-cal on MMLU-Pro.
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/night_calibration.log
while ! grep -q "\[night-v2\] chain complete" "$LOG" 2>/dev/null; do sleep 300; done
echo "[mmlupro-deferred] running $(date)" >> "$LOG"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_mmlupro.json > results/phase5_mmlupro.log 2>&1 || echo "[mmlupro-deferred] WARN phase5" >> "$LOG"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_mmlupro.json > results/xopt_mmlupro.log 2>&1 || echo "[mmlupro-deferred] WARN xopt" >> "$LOG"
echo "[mmlupro-deferred] done $(date)" >> "$LOG"
