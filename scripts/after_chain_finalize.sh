#!/usr/bin/env bash
# after_chain_finalize.sh - Validated finalization (review C1/C2): fixed MMLU-Pro
# A/B, then a morning report that HARD-FAILS on missing/empty inputs.
set -u
cd /home/dave/workspaces/nli-cross-encoder
LOG=results/night_calibration.log
while ! grep -q "\[night-v2\] chain complete" "$LOG" 2>/dev/null; do sleep 300; done
echo "[finalize] running $(date)" >> "$LOG"

uv run python scripts/audit_ragtruth_leakage.py >> "$LOG" 2>&1 || echo "[finalize] ABORT: RAGTruth leakage!" >> "$LOG"

uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5/best \
  --out results/phase5_mmlupro.json > results/phase5_mmlupro.log 2>&1 || echo "[finalize] WARN phase5 mmlupro" >> "$LOG"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-phase5-xopt/best \
  --out results/xopt_mmlupro.json > results/xopt_mmlupro.log 2>&1 || echo "[finalize] WARN xopt mmlupro" >> "$LOG"

python3 - <<'PYEOF' > results/night_calibration_report.md
import json, sys
from pathlib import Path

EXPECTED = {
    "e2b 1.1 (baseline)": ("results/phase5_e2b_gate_eval.json", None, None),
    "e2b 1.1+xopt": ("results/xopt_gate_eval.json", None, "results/xopt_mmlupro.json"),
    "e2b cal": ("results/cal_e2b_gate_eval.json", "results/cal_e2b_ragtruth_served.json", "results/cal_e2b_mmlupro.json"),
    "e4b r3 (baseline)": ("results/phase5_r3_gate_eval.json", None, None),
    "e4b cal": ("results/cal_e4b_gate_eval.json", "results/cal_e4b_ragtruth_served.json", "results/cal_e4b_mmlupro.json"),
}
missing = []
for name, paths in EXPECTED.items():
    for p in paths:
        if p and not Path(p).exists():
            missing.append(f"{name}: {p}")
if missing:
    print("# REPORT INCOMPLETE - missing inputs:\n")
    for m in missing:
        print(f"- {m}")
    sys.exit(1)

print("# Night Calibration Report (validated)\n")
for name, (gate, rag, mm) in EXPECTED.items():
    g = json.loads(Path(gate).read_text())
    if g.get("n", 0) == 0:
        sys.exit(f"REPORT ABORT: {gate} has n=0")
    print(f"## {name}\n")
    print("| metric | value |")
    print("| :--- | :--- |")
    for sl, v in g["by_slice"].items():
        print(f"| gate {sl} | {v['acc']:.4f} (n={v['n']}) |")
    a = g.get("anli", {})
    print(f"| ANLI acc / neutral preds | {a.get('accuracy', 0):.4f} / {a.get('neutral_predictions')} |")
    if rag:
        r = json.loads(Path(rag).read_text())
        print(f"| RAGTruth served F1 @0.5 | {r['f1']:.4f} (prec {r['precision']}, rec {r['recall']}) |")
    if mm and Path(mm).exists():
        m = json.loads(Path(mm).read_text())
        print(f"| MMLU-Pro sample acc | {m['accuracy']:.4f} (n={m['n']}) |")
    print()
print("## MMLU-Pro listwise A/B (phase5 vs xopt)")
a = json.loads(Path("results/phase5_mmlupro.json").read_text())
b = json.loads(Path("results/xopt_mmlupro.json").read_text())
print(f"- phase5 (cross-option 0.0): {a['accuracy']:.4f}")
print(f"- xopt    (cross-option 0.5): {b['accuracy']:.4f}")
PYEOF
RC=$?
if [ $RC -ne 0 ]; then
  echo "[finalize] ABORT: report generation failed rc=$RC (missing/empty inputs)" >> "$LOG"
else
  echo "[finalize] report written $(date)" >> "$LOG"
fi
