# Pre-Registered Evaluation & Decision Protocol

> **Frozen 2026-09-20, before any shakedown/flagship results exist.** Any deviation
> must be documented in PROGRESS.md §8/§9 *with the reason it was impossible to
> anticipate*. This document exists so no result can move the goalposts.

## 1. Data & Splits

- Compiler: `data_pipeline.py` (contamination guards on: pair-key dedup, pair-disjoint
  haystack holdout, XNLI train-split ingestion, `dataset_manifest.json` per compile).
- **train** — fitting only. **val** — checkpoint selection only. **test** — reported
  metrics only, evaluated exactly once per trained artifact on the *reloaded* best
  checkpoint (`train_cross_encoder.py --test-file`).
- External untouched sets (never in any compile): MNLI validation_mismatched,
  XNLI test. Reported alongside the internal test split.
- Synthetic rows (committee-validated, `sdk_synthetic_*.jsonl`) are staged in
  `data/staged/` and mixed into arms by `scripts/build_arms.py` — never auto-ingested
  by the compiler, so the A/B comparison stays under experimental control.

## 2. Shakedown (pilot — never quoted as a headline)

- Arm A: clean quick-mode train set (39,494 rows). Arm B: clean + ≤12.5%
  committee-validated synthetic (capped, seeded sample).
- Recipe (both arms identical): Gemma-4 E2B base, BF16 LoRA r=64 α=128 all-projections,
  `--target-quant none`, 3 epochs, batch 8 × grad-accum 4, lr 1.5e-4 cosine, seed 42.
- Comparison: paired per-item McNemar test on shared test ids (`scripts/gate_decision.py`).

## 3. Synthetic-Data Gate (pre-registered)

Two-stage gate with strictly separated authority:

- **Pilot gate (single judge)** — directional evidence ONLY. It can never authorize
  flagship mixing; it exists to kill weak curricula before multi-judge labeling
  budget is spent.
- **Binding gate (multi-judge)** — computed on **≥ 2 judges' consensus labels**;
  this stage alone can authorize flagship mixing.

Binding-gate PASS iff **all** hold: acc_B > acc_A **and** McNemar p < 0.05
(Holm-corrected across the registered gate family, §6) **and** ECE_B ≤ ECE_A + 0.01
(15-bin ECE; the 0.01 tolerance must exceed 2× the standard error of the ECE
difference, otherwise a bootstrap CI on the ECE difference is required instead).
PASS → stage2 flagship mixes the same validated synthetic rows under the same
12.5% cap — an a priori cap (arbitrary budget, not tuned). FAIL → flagship is
clean-only, and the synthetic curriculum goes back to the drawing board (reported
either way).

- **Statistical power**: at n=3,113 test rows the McNemar gate's minimum detectable
  effect is ~1.5 percentage points (80% power); a FAIL is only binding for effects
  ≥ 1.5pp.
- **Multiplicity policy**: the synthetic gate is one member of the registered gate
  family {synthetic gate, H1 lambda arms, H2, SOTA rows}; Holm correction applies
  across that family, with a cap of 2 re-gate attempts per hypothesis (re-testing
  the same test split after a FAIL counts as an attempt).
- **Gate artifacts** must record the ECE bin count (15) and whether the verdict
  came from the pilot (single-judge) or binding (multi-judge) stage.

## 4. Flagship (stage2)

- `--stage2` compile (~370K pairs) + gated synthetic mix; same LoRA recipe; longer
  schedule if loss has not plateaued (documented, decided **before** looking at test).
- Test evaluation: one shot, reloaded best checkpoint.

## 5. Final Benchmark Comparison vs Jev / OpenJEV / Laya

- **Local runs only** for any model we can execute (openjev-2B/4B, Laya if weights are
  public, Jev if pip-runnable) under ONE frozen protocol: n≥1000 seeded-shuffled
  slices (seed 0), same prompt templates, same scoring rule for all models, per-item
  logs retained. Anything not locally runnable stays in the table labeled
  **"quoted from publication — protocol/hardware unknown"** and is excluded from all
  ratio claims.
- **Both** scoring variants of ours are reported (raw entailment = the protocol
  competitors document; margin = our deployment variant). Headline = raw entailment.
- Statistics: Wilson 95% CIs on every proportion; McNemar (paired, per-item) for any
  A-vs-B claim; no claim from n<1000; ECE with bin count stated, never compared
  cross-dataset; latency = single canonical E2E batch-1 measurement (200 samples,
  20 warmup, CUDA sync) on stated hardware, never divided by quoted numbers.
- Symmetric annotation: every table marks wins **and** losses; no percentage-point
  delta quoted without its CI.

## 6. Pre-Registered Gates

1. **Synthetic gate** — §3 above.
2. **SOTA gate (0.8B phase trigger)** — statistically significant (McNemar p<0.05,
   n≥1000) wins vs openjev-2B on most capability rows **plus** a calibration or
   latency edge. openjev-4B results reported honestly regardless. Gate met →
   Qwen 3.5 0.8B replication vs openjev-0.8B (same protocol, same gates).
3. **Multiplicity policy (gate family-wide)** — the registered gate family is
   {synthetic gate (§3), H1 lambda arms, H2, SOTA rows}; p-values are
   Holm-corrected across that family, and each hypothesis is capped at 2 re-gate
   attempts (re-testing the same test split after a FAIL counts as an attempt).
4. **Iteration rule** — pipeline improvements land only via this loop: hypothesis →
   shakedown arm → gate → adopt/reject. No post-hoc cherry-picking of arms.

## 7. Reporting Standards

- Every number carries: n, split, seed, scoring rule, artifact path.
- Pilot results are labeled "pilot" everywhere they appear.
- Losses, regressions, and gate FAILs are reported in README/PROGRESS with the same
  prominence as wins.
- This protocol supersedes any claim written before 2026-09-20 that conflicts with it
  (see PROGRESS.md §8 audit).
