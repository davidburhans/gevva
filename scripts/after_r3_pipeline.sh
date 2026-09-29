#!/usr/bin/env bash
# after_r3_pipeline.sh - Chains the post-r3 sequence automatically:
#   1. Wait for e4b r3 training to finish
#   2. Quick JevBench on r3 best at T=1.0 ONLY (user directive 2026-09-29: raw
#      temperature, no benchmaxed sweep)
#   3. Quick gate-set eval on r3 best (7,015 items; non-fatal)
#   4. Launch the e2b 1.1 round (the Hopper run) with resume machinery
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_r3_pipeline.log
echo "[pipeline] armed $(date)" >> "$LOG"

# 1. Wait for r3 completion marker
while ! grep -q "Fine-Tuning Finished" results/phase5_e4b_r3_train.log 2>/dev/null; do
  sleep 120
done
echo "[pipeline] r3 finished $(date)" >> "$LOG"
sleep 30  # let final checkpoint writes flush

# 2. JevBench at T=1.0 only (preserve any prior best.json artifact)
[ -f results/jevbench_public_best.json ] && cp results/jevbench_public_best.json "results/jevbench_public_best_$(date +%s).json.bak"
uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e4b-phase5-r3/best --temperature 1.0 \
  > results/phase5_r3_jevbench_t10.log 2>&1 || echo "[pipeline] WARN: jevbench t1.0 failed" >> "$LOG"
echo "[pipeline] jevbench T=1.0 done $(date)" >> "$LOG"

# 3. Gate-set eval (non-fatal; ~12 min)
uv run python scripts/eval_gate_set.py \
  --model-path ckpt/gevva-e4b-phase5-r3/best \
  --out results/phase5_r3_gate_eval.json \
  > results/phase5_r3_gate_eval.log 2>&1 || echo "[pipeline] WARN: gate eval failed" >> "$LOG"
echo "[pipeline] gate eval done $(date)" >> "$LOG"

# 4. Launch the e2b 1.1 round (Hopper run): continual from the e2b champion,
#    anchor-asserted mixture (39k ANLI R1-R3 + 18k WANLI + 15k RAGTruth, all
#    fresh to e2b; 817 multimodal replay rows), resume machinery armed.
nohup uv run python finetune.py \
  --data data/train_phase5_e2b.jsonl \
  --base-model ckpt/gevva-e2b \
  --out-dir ckpt/gevva-e2b-phase5 \
  --full-fine-tune --use-8bit-adam \
  --epochs 2 --lr 3.0e-6 --head-lr 2.5e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.5 --served-dist-weight 1.0 --cross-option-weight 0.0 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 44 \
  --checkpoint-interval 1000 --keep-step-checkpoints 2 --min-free-gb 40 \
  --resume-from auto \
  > results/phase5_e2b_train.log 2>&1 &
echo "[pipeline] e2b 1.1 round launched pid $! $(date)" >> "$LOG"
