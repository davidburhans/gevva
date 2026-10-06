#!/usr/bin/env bash
# night_calibration_chain.sh v2 - with GPU guards + abort-on-failure.
# v1 lesson (2026-09-30): a root-owned llama-server held 10GB all night; the
# chain cascaded through every stage on OOM. v2 waits for a free GPU before
# every stage and aborts hard on training failure instead of cascading.
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/night_calibration.log
echo "[night-v2] armed $(date)" >> "$LOG"
log() { echo "[night-v2] $1 $(date)" >> "$LOG"; }

wait_gpu_free() {  # poll until <2000 MiB used; abort marker after 30 min
  for i in $(seq 1 60); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${USED:-99999}" -lt 2000 ] && return 0
    sleep 30
  done
  log "ABORTED: GPU still busy (${USED} MiB) after 30 min - foreign process holding memory"
  exit 1
}

# 1. xopt gates (xopt training completed under v1)
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_gate_eval.json > results/xopt_gate_eval.log 2>&1 || log "WARN xopt gate eval"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_mmlupro.json > results/xopt_mmlupro.log 2>&1 || log "WARN xopt mmlupro"
log "xopt evals done"

# 2. e2b-cal (fresh restart; v1 run died pre-checkpoint)
wait_gpu_free
log "launching e2b-cal"
uv run python finetune.py \
  --data data/train_cal_e2b.jsonl --base-model ckpt/gevva-e2b-phase5-xopt/best --out-dir ckpt/gevva-e2b-cal \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 2.5e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.8 --served-dist-weight 1.0 --cross-option-weight 0.5 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 45 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > results/cal_e2b_train.log 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" results/cal_e2b_train.log; then
  log "ABORTED: e2b-cal training failed rc=$RC (see results/cal_e2b_train.log)"
  exit 1
fi
log "e2b-cal training done"

# 3. e2b-cal battery
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-cal/best \
  --out results/cal_e2b_gate_eval.json > results/cal_e2b_gate_eval.log 2>&1 || log "WARN e2b-cal gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-cal/best --served-framing \
  --out results/cal_e2b_ragtruth_served.json > results/cal_e2b_ragtruth.log 2>&1 || log "WARN e2b-cal ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-cal/best \
  --out results/cal_e2b_mmlupro.json > results/cal_e2b_mmlupro.log 2>&1 || log "WARN e2b-cal mmlupro"
log "e2b-cal evals done"

# 4. e4b-cal
wait_gpu_free
log "launching e4b-cal"
uv run python finetune.py \
  --data data/train_cal_e4b.jsonl --base-model ckpt/gevva-e4b-phase5-r3/best --out-dir ckpt/gevva-e4b-cal \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 1.8e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.8 --served-dist-weight 1.0 --cross-option-weight 0.5 --nli-aux-weight 0.20 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 46 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > results/cal_e4b_train.log 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" results/cal_e4b_train.log; then
  log "ABORTED: e4b-cal training failed rc=$RC (see results/cal_e4b_train.log)"
  exit 1
fi
log "e4b-cal training done"

# 5. e4b-cal battery + report
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e4b-cal/best \
  --out results/cal_e4b_gate_eval.json > results/cal_e4b_gate_eval.log 2>&1 || log "WARN e4b-cal gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-cal/best --served-framing \
  --out results/cal_e4b_ragtruth_served.json > results/cal_e4b_ragtruth.log 2>&1 || log "WARN e4b-cal ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e4b-cal/best \
  --out results/cal_e4b_mmlupro.json > results/cal_e4b_mmlupro.log 2>&1 || log "WARN e4b-cal mmlupro"
log "e4b-cal evals done"
log "chain complete"
