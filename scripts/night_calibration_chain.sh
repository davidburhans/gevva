#!/usr/bin/env bash
# night_calibration_chain.sh - Overnight calibration program (2026-09-29 -> 09-30)
#
#   1. Wait for the e2b listwise (xopt) round to finish
#   2. Evaluate xopt: gate set + MMLU-Pro K-robustness (the listwise gate)
#   3. Train e2b-cal: served-framing RAGTruth + SDK parity + fresh anchors,
#      1 epoch, Brier 0.8 (calibration emphasis)
#   4. Evaluate e2b-cal: gate + RAGTruth SERVED framing (F1 @ 0.5, no fitted
#      parameters - the industry-facing number) + MMLU-Pro
#   5. Train e4b-cal (same recipe from r3/best)
#   6. Evaluate e4b-cal (same battery)
#   7. Write the morning report
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/night_calibration.log
echo "[night] armed $(date)" >> "$LOG"

log() { echo "[night] $1 $(date)" >> "$LOG"; }

# 1. Wait for xopt
while ! grep -q "Fine-Tuning Finished" results/phase5_e2b_xopt_train.log 2>/dev/null; do
  sleep 180
done
log "xopt finished"

# 2. xopt gates
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_gate_eval.json > results/xopt_gate_eval.log 2>&1 || log "WARN xopt gate eval"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_mmlupro.json > results/xopt_mmlupro.log 2>&1 || log "WARN xopt mmlupro"
log "xopt evals done"

# 3. e2b-cal (continue from xopt best if present, else phase5 best)
BASE="ckpt/gevva-e2b-phase5-xopt/best"
[ -d "$BASE" ] || BASE="ckpt/gevva-e2b-phase5/best"
log "launching e2b-cal from $BASE"
uv run python finetune.py \
  --data data/train_cal_e2b.jsonl --base-model "$BASE" --out-dir ckpt/gevva-e2b-cal \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 2.5e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.8 --served-dist-weight 1.0 --cross-option-weight 0.5 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 45 \
  --checkpoint-interval 1000 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > results/cal_e2b_train.log 2>&1 || log "WARN e2b-cal training rc"
log "e2b-cal training done"

# 4. e2b-cal battery
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-cal/best \
  --out results/cal_e2b_gate_eval.json > results/cal_e2b_gate_eval.log 2>&1 || log "WARN e2b-cal gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-cal/best --served-framing \
  --out results/cal_e2b_ragtruth_served.json > results/cal_e2b_ragtruth.log 2>&1 || log "WARN e2b-cal ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-cal/best \
  --out results/cal_e2b_mmlupro.json > results/cal_e2b_mmlupro.log 2>&1 || log "WARN e2b-cal mmlupro"
log "e2b-cal evals done"

# 5. e4b-cal
log "launching e4b-cal from ckpt/gevva-e4b-phase5-r3/best"
uv run python finetune.py \
  --data data/train_cal_e4b.jsonl --base-model ckpt/gevva-e4b-phase5-r3/best --out-dir ckpt/gevva-e4b-cal \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 1.8e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.8 --served-dist-weight 1.0 --cross-option-weight 0.5 --nli-aux-weight 0.20 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 46 \
  --checkpoint-interval 1000 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > results/cal_e4b_train.log 2>&1 || log "WARN e4b-cal training rc"
log "e4b-cal training done"

# 6. e4b-cal battery
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e4b-cal/best \
  --out results/cal_e4b_gate_eval.json > results/cal_e4b_gate_eval.log 2>&1 || log "WARN e4b-cal gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-cal/best --served-framing \
  --out results/cal_e4b_ragtruth_served.json > results/cal_e4b_ragtruth.log 2>&1 || log "WARN e4b-cal ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e4b-cal/best \
  --out results/cal_e4b_mmlupro.json > results/cal_e4b_mmlupro.log 2>&1 || log "WARN e4b-cal mmlupro"
log "e4b-cal evals done"

# 7. Morning report
python3 - <<'PYEOF' > results/night_calibration_report.md 2>>results/night_calibration.log
import json
from pathlib import Path

def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

lines = ["# Night Calibration Report (auto-generated)\n"]
for name, gate, rag, mm in [
    ("e2b 1.1 (baseline)", "results/phase5_e2b_gate_eval.json", None, None),
    ("e2b 1.1+xopt", "results/xopt_gate_eval.json", None, "results/xopt_mmlupro.json"),
    ("e2b cal", "results/cal_e2b_gate_eval.json", "results/cal_e2b_ragtruth_served.json", "results/cal_e2b_mmlupro.json"),
    ("e4b r3 (baseline)", "results/phase5_r3_gate_eval.json", None, None),
    ("e4b cal", "results/cal_e4b_gate_eval.json", "results/cal_e4b_ragtruth_served.json", "results/cal_e4b_mmlupro.json"),
]:
    g = load(gate)
    if not g:
        continue
    lines.append(f"## {name}\n")
    lines.append("| metric | value |")
    lines.append("| :--- | :--- |")
    for sl, v in g["by_slice"].items():
        lines.append(f"| gate {sl} | {v['acc']:.4f} (n={v['n']}) |")
    a = g.get("anli", {})
    lines.append(f"| ANLI acc / neutral preds | {a.get('accuracy', 0):.4f} / {a.get('neutral_predictions')} |")
    r = load(rag) if rag else None
    if r:
        lines.append(f"| RAGTruth served F1 @0.5 | {r['f1']:.4f} (prec {r['precision']}, rec {r['recall']}, yes-rate {r['yes_rate']}) |")
    m = load(mm) if mm else None
    if m:
        lines.append(f"| MMLU-Pro sample acc | {m['accuracy']:.4f} (n={m['n']}) |")
    lines.append("")
print("\n".join(lines))
PYEOF
log "morning report written -> results/night_calibration_report.md"
