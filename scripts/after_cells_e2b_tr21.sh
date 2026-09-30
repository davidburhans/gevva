#!/usr/bin/env bash
# after_cells_e2b_tr21.sh - When comparison cells finish, launch the e2b listwise
# objective A/B: IDENTICAL mixture/seed/base as the e2b 1.1 round, ONLY
# --cross-option-weight 0.5 restored (the e2b champion lineage config that the
# 1.1 round dropped). Any delta vs ckpt/gevva-e2b-phase5 = objective effect.
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[tr21] watcher armed $(date)" >> "$LOG"
while ! grep -q "\[cells\] all done" "$LOG" 2>/dev/null; do
  sleep 60
done
echo "[tr21] launching e2b listwise round $(date)" >> "$LOG"
nohup uv run python finetune.py \
  --data data/train_phase5_e2b.jsonl \
  --base-model ckpt/gevva-e2b \
  --out-dir ckpt/gevva-e2b-phase5-xopt \
  --full-fine-tune --use-8bit-adam \
  --epochs 2 --lr 3.0e-6 --head-lr 2.5e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.5 --served-dist-weight 1.0 --cross-option-weight 0.5 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 44 \
  --checkpoint-interval 1000 --keep-step-checkpoints 2 --min-free-gb 40 \
  --resume-from auto \
  > results/phase5_e2b_xopt_train.log 2>&1 &
echo "[tr21] e2b listwise round pid $! $(date)" >> "$LOG"
