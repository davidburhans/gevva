#!/usr/bin/env bash
# run_comparison_cells.sh - The four quick comparison cells + RAGTruth verdicts (GPU free).
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/after_e2b_pipeline.log
echo "[cells] starting $(date)" >> "$LOG"
uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e4b-flagship/best --temperature 1.0 --out results/jevbench_e4b_flagship_t10.json > results/jevbench_e4b_flagship_t10.log 2>&1 || echo "[cells] WARN flagship t1.0" >> "$LOG"
uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e4b-phase5-r3/best --temperature 1.6 --out results/jevbench_e4b_r3_t16.json > results/jevbench_e4b_r3_t16.log 2>&1 || echo "[cells] WARN r3 t1.6" >> "$LOG"
uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e2b --temperature 1.0 --out results/jevbench_e2b_champion_t10.json > results/jevbench_e2b_champion_t10.log 2>&1 || echo "[cells] WARN e2b champ t1.0" >> "$LOG"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-phase5/best --out results/phase5_e2b_ragtruth.json > results/phase5_e2b_ragtruth.log 2>&1 || echo "[cells] WARN e2b ragtruth" >> "$LOG"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-phase5-r3/best --out results/phase5_r3_ragtruth.json > results/phase5_r3_ragtruth.log 2>&1 || echo "[cells] WARN r3 ragtruth" >> "$LOG"
echo "[cells] all done $(date)" >> "$LOG"
