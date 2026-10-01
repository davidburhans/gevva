#!/usr/bin/env bash
# night_round4_chain.sh - Distillation round 4: committee generation -> compile ->
# e2b-distill -> battery -> e4b-distill -> battery -> validated report.
# v4 discipline: timestamped logs, GPU waits, hard aborts, asserting builders.
set -u
cd /home/dave/workspaces/nli-cross-encoder
TS=$(date +%Y%m%d_%H%M)
LOG="results/night_round4_${TS}.log"

log() { echo "[r4-${TS}] $1 $(date)" >> "$LOG"; }

wait_gpu_free() {
  for i in $(seq 1 360); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${USED:-99999}" -lt 2000 ] && return 0
    sleep 10
  done
  log "ABORTED: GPU never freed (${USED} MiB after 60 min)"
  exit 1
}

# 1. Committee generation (teacher gemma-4-31b-q4 + cascade panel via llama-swap 8080).
#    Resumable: --resume-run auto reuses the DB + checkpoint on restart.
log "generation start (cascade panel: qwen-3.6-27b-q4, gpt-oss-120b, deepseek-v4-flash-q3)"
uv run python generate_sdk_synthetic_data.py \
  --out-dir data/round4 --samples-per-mode 2000 --resume-run auto \
  > "results/round4_generation_${TS}.log" 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f data/round4/sdk_synthetic_train.jsonl ]; then
  log "ABORTED: generation failed rc=$RC (see results/round4_generation_${TS}.log; resumable)"
  exit 1
fi
log "generation done: $(wc -l < data/round4/sdk_synthetic_train.jsonl) train rows"

# 2. Compile distill mixtures (asserting builder)
uv run python scripts/compile_distill.py --dose 0.10 --seed 48 \
  --out data/train_distill_e2b.jsonl >> "$LOG" 2>&1 || { log "ABORTED: e2b compile failed"; exit 1; }
uv run python scripts/compile_distill.py --dose 0.10 --seed 48 \
  --base data/train_cal2_e4b.jsonl --out data/train_distill_e4b.jsonl >> "$LOG" 2>&1 \
  || { log "ABORTED: e4b compile failed"; exit 1; }
log "mixtures compiled"

# 3. e2b-distill (cal2 base + distill slice)
wait_gpu_free
log "launching e2b-distill training"
uv run python finetune.py \
  --data data/train_distill_e2b.jsonl --base-model ckpt/gevva-e2b-cal2/best --out-dir ckpt/gevva-e2b-distill \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 2.5e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.4 --served-dist-weight 1.0 --cross-option-weight 0.0 --nli-aux-weight 0.25 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 48 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > "results/distill_e2b_train_${TS}.log" 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" "results/distill_e2b_train_${TS}.log"; then
  log "ABORTED: e2b-distill training failed rc=$RC"
  exit 1
fi
log "e2b-distill training done"

# 4. e2b battery + paired McNemar vs cal2
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e2b-distill/best \
  --out results/distill_e2b_gate_eval.json > results/distill_e2b_gate_eval.log 2>&1 || log "WARN distill gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e2b-distill/best --served-framing \
  --out results/distill_e2b_ragtruth_served.json > results/distill_e2b_ragtruth_served.log 2>&1 || log "WARN distill ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e2b-distill/best \
  --out results/distill_e2b_mmlupro.json > results/distill_e2b_mmlupro.log 2>&1 || log "WARN distill mmlupro"
uv run python scripts/mcnemar_gate_paired.py \
  --model-a ckpt/gevva-e2b-cal2/best --model-b ckpt/gevva-e2b-distill/best \
  --out results/mcnemar_cal2_vs_distill_e2b.json > results/mcnemar_cal2_vs_distill_e2b.log 2>&1 || log "WARN mcnemar"
log "e2b-distill battery done"

# 5. e4b-distill (cal2 base + distill slice)
wait_gpu_free
log "launching e4b-distill training"
uv run python finetune.py \
  --data data/train_distill_e4b.jsonl --base-model ckpt/gevva-e4b-cal2/best --out-dir ckpt/gevva-e4b-distill \
  --full-fine-tune --use-8bit-adam --epochs 1 --lr 1.8e-6 --head-lr 2.0e-5 \
  --grad-accum 16 --token-bucketing --max-tokens-per-batch 2048 --max-length 1024 \
  --brier-weight 0.4 --served-dist-weight 1.0 --cross-option-weight 0.0 --nli-aux-weight 0.20 \
  --decision-temp 1.0 --val-ratio 0.02 --label-convention ours --image-root . --seed 48 \
  --checkpoint-interval 500 --keep-step-checkpoints 2 --min-free-gb 40 --resume-from auto \
  > "results/distill_e4b_train_${TS}.log" 2>&1
RC=$?
if [ $RC -ne 0 ] || ! grep -q "Fine-Tuning Finished" "results/distill_e4b_train_${TS}.log"; then
  log "ABORTED: e4b-distill training failed rc=$RC"
  exit 1
fi
log "e4b-distill training done"

# 6. e4b battery + paired McNemar vs cal2
wait_gpu_free
uv run python scripts/eval_gate_set.py --model-path ckpt/gevva-e4b-distill/best \
  --out results/distill_e4b_gate_eval.json > results/distill_e4b_gate_eval.log 2>&1 || log "WARN distill gate"
uv run python scripts/eval_ragtruth.py --model-path ckpt/gevva-e4b-distill/best --served-framing \
  --out results/distill_e4b_ragtruth_served.json > results/distill_e4b_ragtruth_served.log 2>&1 || log "WARN distill ragtruth"
uv run python scripts/eval_mmlupro_sample.py --model-path ckpt/gevva-e4b-distill/best \
  --out results/distill_e4b_mmlupro.json > results/distill_e4b_mmlupro.log 2>&1 || log "WARN distill mmlupro"
uv run python scripts/mcnemar_gate_paired.py \
  --model-a ckpt/gevva-e4b-cal2/best --model-b ckpt/gevva-e4b-distill/best \
  --out results/mcnemar_cal2_vs_distill_e4b.json > results/mcnemar_cal2_vs_distill_e4b.log 2>&1 || log "WARN mcnemar"
log "e4b-distill battery done"

# 7. Validated report
python3 - "$TS" <<'PYEOF' > "results/night_round4_report_${TS}.md"
import json, sys
from pathlib import Path
ts = sys.argv[1]
MODELS = {
    "e2b": {"base_gate": "results/cal2_e2b_gate_eval.json",
            "gate": "results/distill_e2b_gate_eval.json",
            "rag": "results/distill_e2b_ragtruth_served.json",
            "mm": "results/distill_e2b_mmlupro.json",
            "mc": "results/mcnemar_cal2_vs_distill_e2b.json",
            "base_rag": "results/cal2_e2b_ragtruth_served.json"},
    "e4b": {"base_gate": "results/cal2_e4b_gate_eval.json",
            "gate": "results/distill_e4b_gate_eval.json",
            "rag": "results/distill_e4b_ragtruth_served.json",
            "mm": "results/distill_e4b_mmlupro.json",
            "mc": "results/mcnemar_cal2_vs_distill_e4b.json",
            "base_rag": "results/cal2_e4b_ragtruth_served.json"},
}
missing = [f"{m}.{k}: {p}" for m, d in MODELS.items() for k, p in d.items() if not Path(p).exists()]
if missing:
    print("# REPORT INCOMPLETE - missing inputs:\n")
    for x in missing: print(f"- {x}")
    sys.exit(1)
print(f"# Round-4 Distillation Report (validated, {ts})\n")
for m, d in MODELS.items():
    g = json.loads(Path(d["gate"]).read_text())
    b = json.loads(Path(d["base_gate"]).read_text())
    r = json.loads(Path(d["rag"]).read_text())
    br = json.loads(Path(d["base_rag"]).read_text())
    mm = json.loads(Path(d["mm"]).read_text())
    mc = json.loads(Path(d["mc"]).read_text())
    for tag, dd in (("gate", g), ("rag", r), ("mm", mm)):
        if dd.get("n", 0) == 0: sys.exit(f"REPORT ABORT: {m} {tag} n=0")
    print(f"## {m}-distill vs {m}-cal2\n")
    print("| metric | cal2 | distill |")
    print("| :--- | :--- | :--- |")
    for sl in g["by_slice"]:
        print(f"| gate {sl} | {b['by_slice'][sl]['acc']:.4f} | {g['by_slice'][sl]['acc']:.4f} |")
    a = b.get("anli", {}); a2 = g.get("anli", {})
    print(f"| ANLI acc | {a.get('accuracy', 0):.4f} | {a2.get('accuracy', 0):.4f} |")
    print(f"| RAGTruth served F1@0.5 | {br['f1']:.4f} | {r['f1']:.4f} |")
    print(f"| MMLU-Pro (engine) | - | {mm['accuracy']:.4f} |")
    for sl in ("hard", "ALL"):
        s = mc["slices"][sl]
        print(f"| McNemar {sl} | - | {s['delta_pp']}pp (p={s['mcnemar_p']}) |")
    print()
print("\n## Verdict rule (pre-registered)")
print("Keep distill lineage only if: RAGTruth F1 not worse by >0.02, AND no McNemar-significant")
print("gate regression (p<0.01), AND ANLI neutral predictions stable. Otherwise cal2 stands.")
PYEOF
RC=$?
if [ $RC -ne 0 ]; then
  log "ABORT: report generation failed rc=$RC"
  exit 1
fi
log "report written: results/night_round4_report_${TS}.md"
log "chain complete"
