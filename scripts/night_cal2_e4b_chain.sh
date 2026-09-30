#!/usr/bin/env bash
# night_cal2_e4b_chain.sh - e4b round-2 calibration (same v3 discipline:
# timestamped logs, hard aborts, committed before arming).
set -u
cd /home/dave/workspaces/nli-cross-encoder
TS=$(date +%Y%m%d_%H%M)
LOG="results/night_cal2_e4b_${TS}.log"
TRAIN_LOG="results/cal2_e4b_train_${TS}.log"

log() { echo "[cal2e4b-${TS}] $1 $(date)" >> "$LOG"; }

wait_gpu_free() {
  for i in $(seq 1 180); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${USED:-99999}" -lt 2000 ] && return 0
    sleep 10
  done
  log "ABORTED: GPU never freed (${USED} MiB after 30 min)"
  exit 1
}

# Gate on the e2b JevBench cell if still running (single-GPU serialization)
wait_gpu_free
log "chain start"

# 1. e4b-cal2 training (r3 base, v2 slice; hyperparams from e4b round-1)
uv run python finetune.py \
  --data data/train_cal2_e4b.jsonl --base-model ckpt/gevva-e4b-phase5-r3/best --out-dir ckpt/gevva-e4b-cal2 \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 1.8e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.4 --served-dist-weight 1.0 --cross-option-weight 0.0 --nli-aux-weight 0.20 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 47 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > "$TRAIN_LOG" 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" "$TRAIN_LOG"; then
  log "ABORTED: e4b-cal2 training failed rc=$RC (see $TRAIN_LOG)"
  exit 1
fi
log "e4b-cal2 training done"

# 2. Battery: gate + ragtruth served + mmlupro + paired McNemar vs r3
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e4b-cal2/best \
  --out results/cal2_e4b_gate_eval.json > results/cal2_e4b_gate_eval.log 2>&1 || log "WARN cal2 gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-cal2/best --served-framing \
  --out results/cal2_e4b_ragtruth_served.json > results/cal2_e4b_ragtruth_served.log 2>&1 || log "WARN cal2 ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e4b-cal2/best \
  --out results/cal2_e4b_mmlupro.json > results/cal2_e4b_mmlupro.log 2>&1 || log "WARN cal2 mmlupro"
uv run python scripts/mcnemar_gate_paired.py \
  --model-a ckpt/gevva-e4b-phase5-r3/best --model-b ckpt/gevva-e4b-cal2/best \
  --out results/mcnemar_r3_vs_cal2_e4b.json > results/mcnemar_r3_vs_cal2_e4b.log 2>&1 || log "WARN mcnemar"
log "battery done"

# 3. Validated report
python3 - "$TS" <<'PYEOF' > "results/night_cal2_e4b_report_${TS}.md"
import json, sys
from pathlib import Path
ts = sys.argv[1]
paths = {"gate": "results/cal2_e4b_gate_eval.json",
         "rag": "results/cal2_e4b_ragtruth_served.json",
         "mm": "results/cal2_e4b_mmlupro.json",
         "mcnemar": "results/mcnemar_r3_vs_cal2_e4b.json"}
missing = [f"{k}: {p}" for k, p in paths.items() if not Path(p).exists()]
if missing:
    print("# REPORT INCOMPLETE - missing inputs:\n")
    for m in missing: print(f"- {m}")
    sys.exit(1)

g = json.loads(Path(paths["gate"]).read_text())
r = json.loads(Path(paths["rag"]).read_text())
m = json.loads(Path(paths["mm"]).read_text())
mc = json.loads(Path(paths["mcnemar"]).read_text())
for tag, d in (("gate", g), ("rag", r), ("mm", m)):
    if d.get("n", 0) == 0: sys.exit(f"REPORT ABORT: {paths[tag]} n=0")

print(f"# e4b Cal-Round-2 Report (validated, {ts})\n")
print("| metric | value |")
print("| :--- | :--- |")
for sl, v in g["by_slice"].items():
    print(f"| gate {sl} | {v['acc']:.4f} (n={v['n']}) |")
a = g.get("anli", {})
print(f"| ANLI acc / neutral preds | {a.get('accuracy', 0):.4f} / {a.get('neutral_predictions')} |")
print(f"| RAGTruth served F1 @0.5 | {r['f1']:.4f} (prec {r['precision']}, rec {r['recall']}, yes {r['yes_rate']}) |")
print(f"| MMLU-Pro (engine framing) | {m['accuracy']:.4f} (n={m['n']}) |")
print(f"| McNemar hard vs r3 | delta {mc['slices']['hard']['delta_pp']}pp, p={mc['slices']['hard']['mcnemar_p']} |")
print(f"| McNemar ALL vs r3 | delta {mc['slices']['ALL']['delta_pp']}pp, p={mc['slices']['ALL']['mcnemar_p']} |")
print("\n## Comparison (RAGTruth served F1@0.5)")
print("| model | F1 | yes_rate |")
print("| :--- | :--- | :--- |")
print("| e4b v1 | 0.3668 | - |")
print("| e4b r3 (cal-1, rejected) | 0.2048 | 0.664 |")
print(f"| e4b cal-2 | {r['f1']} | {r['yes_rate']} |")
print(f"| e2b cal-2 | 0.6972 | 0.2819 |")
PYEOF
RC=$?
if [ $RC -ne 0 ]; then
  log "ABORT: report generation failed rc=$RC"
  exit 1
fi
log "report written: results/night_cal2_e4b_report_${TS}.md"
log "chain complete"
