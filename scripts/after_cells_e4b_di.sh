#!/usr/bin/env bash
# after_cells_e4b_di.sh - Queue the e4b 1.1 Decision Index rerun after the
# comparison cells finish (definitive undertraining-fix measurement).
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[pipeline] e4b-DI watcher armed $(date)" >> "$LOG"
while ! grep -q "ragtruth verdicts (rerun) done" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[pipeline] launching e4b Decision Index rerun $(date)" >> "$LOG"
uv run python scripts/run_decision_index_eval.py --models e4b-phase5-r3 \
  > results/decision_index_e4b_phase5_r3.log 2>&1 || echo "[pipeline] WARN e4b DI failed" >> "$LOG"
echo "[pipeline] e4b DI done $(date)" >> "$LOG"
