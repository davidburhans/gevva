#!/usr/bin/env bash
# night_cal2_chain.sh - Round-2 calibration chain (v3 design: timestamped logs,
# hard aborts, committed before arming - review rules 2/5 and m3).
set -u
cd /home/dave/workspaces/nli-cross-encoder
TS=$(date +%Y%m%d_%H%M)
LOG="results/night_cal2_${TS}.log"
TRAIN_LOG="results/cal2_e2b_train_${TS}.log"

log() { echo "[cal2-${TS}] $1 $(date)" >> "$LOG"; }

wait_gpu_free() {
  for i in $(seq 1 180); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${USED:-99999}" -lt 2000 ] && return 0
    sleep 10
  done
  log "ABORTED: GPU never freed (${USED} MiB after 30 min)"
  exit 1
}

# 0. Preflight
wait_gpu_free
log "chain start"

# 1. Data build (idempotent, asserting)
uv run python scripts/build_ragtruth_served_v2.py --seed 47 >> "$LOG" 2>&1 || { log "ABORTED: v2 data build failed"; exit 1; }
uv run python scripts/compile_cal2.py --seed 47 >> "$LOG" 2>&1 || { log "ABORTED: mixture compile failed"; exit 1; }
log "data v2 + mixture compiled"

# 2. Train e2b-cal2 (phase5 base, brier 0.4, cross-option 0.0)
wait_gpu_free
log "launching e2b-cal2 training"
uv run python finetune.py \
  --data data/train_cal2_e2b.jsonl --base-model ckpt/gevva-e2b-phase5/best --out-dir ckpt/gevva-e2b-cal2 \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 2.5e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.4 --served-dist-weight 1.0 --cross-option-weight 0.0 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 47 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > "$TRAIN_LOG" 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" "$TRAIN_LOG"; then
  log "ABORTED: e2b-cal2 training failed rc=$RC (see $TRAIN_LOG)"
  exit 1
fi
log "e2b-cal2 training done"

# 3. Battery: gate + ragtruth served + mmlupro (cal2 + the A/B pair)
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-cal2/best \
  --out results/cal2_e2b_gate_eval.json > results/cal2_e2b_gate_eval.log 2>&1 || log "WARN cal2 gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-cal2/best --served-framing \
  --out results/cal2_e2b_ragtruth_served.json > results/cal2_e2b_ragtruth_served.log 2>&1 || log "WARN cal2 ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-cal2/best \
  --out results/cal2_e2b_mmlupro.json > results/cal2_e2b_mmlupro.log 2>&1 || log "WARN cal2 mmlupro"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_mmlupro.json > results/phase5_mmlupro.log 2>&1 || log "WARN phase5 mmlupro"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_mmlupro.json > results/xopt_mmlupro.log 2>&1 || log "WARN xopt mmlupro"
log "battery done"

# 4. Validated report (hard-fails on missing/empty inputs)
python3 - "$TS" <<'PYEOF' > "results/night_cal2_report_${TS}.md"
import json, sys
from pathlib import Path
ts = sys.argv[1]
EXPECTED = {
    "e2b 1.1 (phase5 base)": {
        "gate": "results/phase5_e2b_gate_eval.json",
        "rag": None,
        "mm": "results/phase5_mmlupro.json"},
    "e2b 1.1+xopt": {
        "gate": "results/xopt_gate_eval.json",
        "rag": None,
        "mm": "results/xopt_mmlupro.json"},
    "e2b cal-round-1 (rejected)": {
        "gate": "results/cal_e2b_gate_eval.json",
        "rag": "results/cal_e2b_ragtruth_served.json",
        "mm": "results/cal_e2b_mmlupro.json"},
    "e2b cal-round-2": {
        "gate": "results/cal2_e2b_gate_eval.json",
        "rag": "results/cal2_e2b_ragtruth_served.json",
        "mm": "results/cal2_e2b_mmlupro.json"},
}
missing = [f"{n}: {p}" for n, d in EXPECTED.items() for k, p in d.items() if p and not Path(p).exists()]
if missing:
    print("# REPORT INCOMPLETE - missing inputs:\n")
    for m in missing: print(f"- {m}")
    sys.exit(1)

print(f"# Cal-Round-2 Report (validated, {ts})\n")
for name, d in EXPECTED.items():
    g = json.loads(Path(d["gate"]).read_text())
    if g.get("n", 0) == 0: sys.exit(f"REPORT ABORT: {d['gate']} n=0")
    print(f"## {name}\n")
    print("| metric | value |")
    print("| :--- | :--- |")
    for sl, v in g["by_slice"].items():
        print(f"| gate {sl} | {v['acc']:.4f} (n={v['n']}) |")
    a = g.get("anli", {})
    print(f"| ANLI acc / neutral preds | {a.get('accuracy', 0):.4f} / {a.get('neutral_predictions')} |")
    if d["rag"]:
        r = json.loads(Path(d["rag"]).read_text())
        if r.get("n", 0) == 0: sys.exit(f"REPORT ABORT: {d['rag']} n=0")
        print(f"| RAGTruth served F1 @0.5 | {r['f1']:.4f} (prec {r['precision']}, rec {r['recall']}, yes {r['yes_rate']}) |")
    if d["mm"] and Path(d["mm"]).exists():
        m = json.loads(Path(d["mm"]).read_text())
        if m.get("n", 0) == 0: sys.exit(f"REPORT ABORT: {d['mm']} n=0")
        print(f"| MMLU-Pro (engine framing) | {m['accuracy']:.4f} (n={m['n']}) |")
    print()
print("## TR-21 listwise A/B (first valid measurement; engine framing)")
a = json.loads(Path("results/phase5_mmlupro.json").read_text())
b = json.loads(Path("results/xopt_mmlupro.json").read_text())
print(f"- phase5 (cross-option 0.0): {a['accuracy']:.4f}")
print(f"- xopt    (cross-option 0.5): {b['accuracy']:.4f}")
print("\n## Round-1 rejection criteria replay")
try:
    r2 = json.loads(Path("results/cal2_e2b_ragtruth_served.json").read_text())
    r1 = json.loads(Path("results/cal_e2b_ragtruth_served.json").read_text())
    p5 = json.loads(Path("results/phase5_e2b_gate_eval.json").read_text())
    verdict = "PASS" if (r2["f1"] > r1["f1"] + 0.05 and r2["yes_rate"] < 0.45) else "REVIEW"
    print(f"- cal2 F1 {r2['f1']} vs cal1 {r1['f1']} | yes_rate {r2['yes_rate']} -> {verdict}")
except KeyError as e:
    sys.exit(f"REPORT ABORT: missing field {e}")
PYEOF
RC=$?
if [ $RC -ne 0 ]; then
  log "ABORT: report generation failed rc=$RC"
  exit 1
fi
log "report written: results/night_cal2_report_${TS}.md"
log "chain complete"
