#!/usr/bin/env bash
# night_di_chain.sh - Full Decision Index runs for the e2b champion decision
# (cal2 vs phase5), the primary instrument per AGENTS.md. v4 discipline:
# timestamped logs, GPU waits, hard aborts, progress watchdog (review M6:
# the previous DI run died silently at 16%).
set -u
cd /home/dave/workspaces/nli-cross-encoder
TS=$(date +%Y%m%d_%H%M)
LOG="results/night_di_${TS}.log"

log() { echo "[di-${TS}] $1 $(date)" >> "$LOG"; }

wait_gpu_free() {
  for i in $(seq 1 180); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${USED:-99999}" -lt 2000 ] && return 0
    sleep 10
  done
  log "ABORTED: GPU never freed (${USED} MiB after 30 min)"
  exit 1
}

watch_progress() {  # $1 = model alias, $2 = run dir; logs while pipeline runs
  local alias="$1" run_dir="$2"
  while pgrep -f "decision_index pipeline.*${alias}" > /dev/null 2>&1; do
    if [ -f "${run_dir}/status.json" ]; then
      DONE=$(python3 -c "import json;d=json.load(open('${run_dir}/status.json'));print(d.get('completed','?'))" 2>/dev/null || echo "?")
      log "${alias} progress: ${DONE} items"
    fi
    sleep 1800
  done
}

run_di() {  # $1 = alias
  local alias="$1"
  local run_dir="runs/gevva-${alias}-0.2"
  wait_gpu_free
  log "launching DI for ${alias}"
  uv run python scripts/run_decision_index_eval.py --models "${alias}" \
    >> "results/di_${alias}_${TS}.log" 2>&1 &
  local di_pid=$!
  watch_progress "${alias}" "${run_dir}" &
  local watch_pid=$!
  wait $di_pid
  local rc=$?
  kill $watch_pid 2>/dev/null
  if [ $rc -ne 0 ] || [ ! -f "${run_dir}/scores.json" ]; then
    log "ABORTED: DI ${alias} failed rc=$rc, scores.json missing (see results/di_${alias}_${TS}.log)"
    exit 1
  fi
  log "DI ${alias} complete: $(python3 -c "import json;print(json.load(open('${run_dir}/scores.json')).get('score','?'))" 2>/dev/null || echo 'score parse failed')"
}

run_di e2b-cal2
run_di e2b-phase5

log "both DI runs complete; champion comparison ready"
log "chain complete"
