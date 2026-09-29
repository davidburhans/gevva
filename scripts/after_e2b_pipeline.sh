#!/usr/bin/env bash
# after_e2b_pipeline.sh - Chains the post-e2b evaluation sequence:
#   1. Wait for e2b 1.1 training to finish
#   2. Quick gate-set eval (floor/medium/hard + ANLI neutral recall)
#   3. RAGTruth response-level F1 (TR-06 verdict, both models owed)
#   4. SDK parity post-hoc McNemar gate (e2b-1.1 vs champion)
#   5. JevBench T=1.0 (raw, no sweep)
#   6. Full Decision Index 0.2 on e2b-1.1 (user priority: e2b first)
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[pipeline] armed $(date)" >> "$LOG"

while ! grep -q "Fine-Tuning Finished" results/phase5_e2b_train.log 2>/dev/null; do
  sleep 120
done
echo "[pipeline] e2b training finished $(date)" >> "$LOG"
sleep 30

uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_e2b_gate_eval.json > results/phase5_e2b_gate_eval.log 2>&1 || echo "[pipeline] WARN gate eval failed" >> "$LOG"
echo "[pipeline] gate eval done $(date)" >> "$LOG"

uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_e2b_ragtruth.json > results/phase5_e2b_ragtruth.log 2>&1 || echo "[pipeline] WARN ragtruth failed" >> "$LOG"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-phase5-r3/best \
  --out results/phase5_r3_ragtruth.json > results/phase5_r3_ragtruth.log 2>&1 || echo "[pipeline] WARN e4b ragtruth failed" >> "$LOG"
echo "[pipeline] ragtruth verdicts done $(date)" >> "$LOG"

[ -f results/jevbench_public_best.json ] && cp results/jevbench_public_best.json "results/jevbench_public_best_$(date +%s).json.bak"
uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e2b-phase5/best --temperature 1.0 \
  > results/phase5_e2b_jevbench_t10.log 2>&1 || echo "[pipeline] WARN jevbench failed" >> "$LOG"
echo "[pipeline] jevbench T=1.0 done $(date)" >> "$LOG"

# Full Decision Index on e2b-1.1 (the headline measurement; e4b rerun comes after)
uv run python scripts/run_decision_index_eval.py --models e2b-phase5 \
  > results/decision_index_e2b_phase5.log 2>&1 || echo "[pipeline] WARN decision index failed" >> "$LOG"
echo "[pipeline] decision index e2b done $(date)" >> "$LOG"
