# SESSION HAND-OFF: NLI Cross-Encoder Pipeline (2026-09-20)

> Read this + AGENTS.md + docs/EVALUATION_PROTOCOL.md + PROGRESS.md §8-9 to resume
> with full context. This document assumes ZERO prior session knowledge.

## 1. Mission (owner-set, pre-registered)

Remediate all audit findings with scientific discipline → retrain (two-stage: quick
shakedown → stage2 ~370K flagship) → honest head-to-head vs Jev/OpenJEV/Laya → if
same-size SOTA gate fires (McNemar p<0.05 wins vs openjev-2B on most rows + calib/latency
edge), replicate the pipeline on Qwen3.5-0.8B vs openjev-0.8B (C+ plan: self-train
openjev-0.8B from their public train.py — their HF weights for 0.8B/2B DO NOT EXIST).

**Frozen protocol**: docs/EVALUATION_PROTOCOL.md. Gates: synthetic A/B (arm A clean vs
arm B +≤12.5% validated synthetic; McNemar p<0.05 + ECE non-regression; tonight's run is
the SINGLE-JUDGE PILOT — flagship mixing requires the multi-judge gate), same-size SOTA,
MDE ≈1.5pp at n=3113. Owner rules: commit before launching agents; fresh sub-agents for
heavy work; ground every assumption; no unverifiable headline claims.

## 2. Live state at hand-off

| Item | State |
| :--- | :--- |
| GPU processes | NONE (all stopped; GPU ~2GB idle) |
| Committee run | STOPPED at judge 1: `run_20260919_223701` in `data/validation_metrics.db`; ~2,700+ ok qwen-3.6 verdicts persisted (non-ok purged; auto-purge now also at judge start) |
| Night chain | NOT LAUNCHED — gated on the two in-flight workers (below) |
| Git | HEAD `7244c90` + hand-off commit; clean tree except workers' in-flight edits |
| Tests | 33/33 across 4 suites (tests/test_{validator_committee,sdk_parity,night_stats,data_hygiene}.py) |

**In-flight workers** (fresh context, async):
- `210329bc` (worker): fix `scripts/tonight_chain.py` main() control flow (failure →
  alert + committee resume; success → committee resume for judges 2-4 + watchdog re-arm;
  exactly one launch per path) + `train_cross_encoder.py` head-restore numeric guard +
  test_sha fingerprint in test artifacts.
- `e49dbcef` (worker): doc amendments — protocol MDE/multiplicity/cap-label/two-stage
  gate; README+PROGRESS claims sweep (strike quoted-baseline ratios, year 2026, A4
  split counts, GLiNER2 grounded status).

**On worker completion**: verify their claims (py_compile + 4 test suites + read diffs),
fix anything broken, `git add -A && git commit`, THEN launch the chain:
```bash
cd /home/dave/workspaces/nli-cross-encoder
nohup uv run python -u scripts/tonight_chain.py > results/tonight_chain_driver.log 2>&1 &
echo $! > results/tonight_chain.pid
```

## 3. Chain stages (tonight_chain.py — what it does autonomously)

ensure committee (resume run_20260919_223701, liveness-verified 45s) → wait judge 1
(7,515 ok, liveness-checked, ≤3 auto-restarts) → pause+purge all run non-ok → unload →
GPU drain-wait (<2.5GB) → recompile (sanitized vision premises) → train arm A (clean
39K) → export validated synthetic (10% val holdback) → arms → train arm B → **synthetic
pilot gate** → resume committee (judges 2-4: qwen-3.8-125b-q3 → qwen-3.8-125b-q4 →
deepseek-v4-flash-q3 LAST) with watchdog. Failure anywhere → alert file + committee
resumed in `finally` (validation never dies).

## 4. Judge order decision (owner)

qwen-3.6-27b-q4 (running/done) → qwen-3.8-125b-q3 → qwen-3.8-125b-q4 → deepseek-v4-flash-q3
(slowest = last). Resume uses explicit `--resume-run run_20260919_223701` (auto-match
would fail: validator order changed vs the stored run row).

## 5. Validator fixes baked in today (grounded by live A/B measurement)

- `enable_thinking=false` for qwen-3.6-27b-q4 (hidden reasoning truncated the JSON
  budget → 89% parse_error; off = 3.2s vs 20.3s, 100% parse). DeepSeek keeps vendor
  default (thinking-off breaks its batch completeness).
- `--validator-timeout 600` (180s caused deepseek timeouts), retry-once for offline
  batches, purge non-ok at judge start + run scope, resume quality warnings.
- Single-judge overrides: confidence = quorum fraction, severity escalates when judges
  failed (user CRITICAL finding).

## 6. Key artifacts

| Artifact | Path |
| :--- | :--- |
| Protocol (frozen) | docs/EVALUATION_PROTOCOL.md |
| Council memo (0.8B baseline C+ plan) | docs/council_memo_08b_baseline.md |
| Round-1 audit (22 items) + G1-G5 grounding | PROGRESS.md §8-9, research/reports/08 |
| Clean dataset (39,494/4,670/3,113 + manifest) | data/{train,val,test}.jsonl, dataset_manifest.json |
| Committee checkpoint/resume | data/validation_metrics.db, data/sdk_synthetic_raw.jsonl |
| Disagreement review queue (post-run) | data/sdk_synthetic_disagreements.jsonl |
| Gate verdict (post-arms) | results/gate_decision.json |
| Metrics DB analyses | `sqlite3 data/validation_metrics.db "SELECT * FROM judge_performance"` |

## 7. Open items / deferred (do NOT lose)

1. **A7 image-into-collator** (multimodal rows train text-only; marker string preserved
   in `premise_markers` field for the fix). Top of backlog before stage2 if possible.
2. W4A16 vision RoPE buffer restoration (user HIGH — W4A16 path only, arms unaffected).
3. INT4 NaN/INT_MIN packing guard (export_w4a16.py).
4. forward_packed cross-sequence attention (128K path — not used by shakedown).
5. SDK delimiter-spoofing sanitization (grade/rerank serving hardening).
6. openjev-0.8B self-training per council memo C+ (~4-7 GPU-h, after SOTA gate) +
   harness gap: foreign-checkpoint per-item evaluator for McNemar pairing.
7. H1 (proper-scoring loss λ sweep {0.25,0.5,1.0}) + H2 (temperature scaling) —
   registered, unimplemented.
8. Local baselines: openjev-4B v1/v2 weights EXIST (prefetch in eval_baselines);
   openjev-0.8B/2B ABSENT (quoted forever); Jev API-only (quoted forever); Laya adapter
   TODO (vendor XNLI prompt unpublished — label any row "our mapping documented").

## 8. Incident history (all root-caused, fixes in HEAD)

1. Validator parse flood (89%): hidden reasoning truncated token budget → thinking-off
   per-model policy + 3072 tokens + rationale maxLength 240.
2. DeepSeek timeout flood: 180s too short → 600s knob + retry-once + purge-and-rerun.
3. Premature queue firing (3× GPU contention + train_A OOM): stale driver watched a
   dead pid; tonight_chain kills it at startup and run_night_queue.py is retired.
4. Resume counted failed rows as done: purge at judge start + quality warnings.
5. pkill/kill footguns: always kill the resolved python child pid (uv wrapper does not
   forward signals); never pkill -f with a pattern contained in your own command line.
