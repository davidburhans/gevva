# JevBench Shortcoming Remediation Plan

> Written 2026-09-22, grounded in the public-231 scored run (`ce-stage3-v2`) and the
> axis decomposition. Baseline: **Score ≈ 47.9 (unofficial)** — Intelligence 53.5,
> Calibration 19.6, Speed 100, Cost ~50. Board targets: Laya 70.1 · open-alternative-jev
> 69.8 · system-one-open 68.9 · GLiNER2 53.0.

## Evidence: where the points are lost

| Weakness | Measured | Axis impact | Root cause (evidence) |
|---|---|---|---|
| Option-distribution calibration | hard-tier ECE **0.4021** on renormalized P(entailment) | Calibration 19.6 (vs leaders 72–83) | **Train-serving parity gap**: serving renormalizes per-option P(ent) into a distribution; training only ever optimizes per-pair 3-class CE. The raw 3-class softmax is excellent (ECE 0.066) — the *served artifact* was never trained. |
| Hard tier | 30.9% public (111) | Intelligence 53.5 | tradeoff 0/6, multi_hop 22%, long_policy 26%, ambiguous 14%, probability 30%, temporal_numeric 33%. Genuine capability gap + possible convention mismatches (must audit — tradeoff 0/6 smells like the haystack_embedded/counterfactual pattern: a *labelable* defect). |
| Judge & rubric | judge_hard 41%, adequacy 42%, policy 50% | Intelligence | Rubric-following/answer-judging under-served in curriculum. |
| Confirmation asymmetry | score_gold 50% / choice_gold 62% vs neg/alt 91–97% | Intelligence (gold halves) | Training mix under-weights *verifying correct options* vs rejecting wrong ones. |

Fixed levers (no action): Speed axis 100 (28 ms p50, fastest on the board); the
mapping is frozen (pre-registered; we train under it, never change it).

## Phase 0 — Audit & instrument (CPU + ~1h GPU) — *before any training*

1. **Error audit of weak families** (the haystack_embedded lesson): read every wrong
   tradeoff/ambiguous/multi_hop/judge_hard public item; classify each as
   (a) model error, (b) convention/format mismatch, (c) ambiguous gold. Deliverable:
   `docs/jevbench-error-audit.md` with counts. Gate: if >30% of a family's errors are
   (b)/(c), fix data/mapping-side understanding first — don't train against noise.
2. **Local JevBench harness**: fold the adapter mapping (premise/option/hypothesis
   construction + renormalization) into `eval_system1_suite.py` so every checkpoint
   reports public-231 accuracy-by-family + renormalized-ECE without the private runbook.
   Add Wilson CIs (n=111 hard is ±8.7pp).
3. **P1 checkpoint balanced re-eval** (GPU, ~10 min): the 87.94% is a degenerate-slice
   number (0 neutral support); know what P1 actually contributes before deciding reuse.
4. **Declare v2's context budget 16384** in the mapping doc pre-registration
   (trained length; covers all items — the default 4096 under-serves it).

## Phase 1 — Train the served distribution (the +18-point lever) — 1 training round

**Goal:** hard-tier renormalized ECE 0.40 → ≤0.20 (Calibration axis → ~60+).

1. **Distribution-loss objective**: for grouped decision items, compute the served
   artifact in-loop — P(ent_i) per option → renormalize → proper scoring rule
   (CE + Brier on the option simplex) against gold. Reuses the existing grouped
   collator + P1 mixture infrastructure; new loss is ~50 lines in the trainer.
2. **Data**: existing grouped sets (typed-decisions 82K, hard-decisions 4.1K) +
   NLI anchors for replay. No new generation needed for this phase.
3. **Train**: warm-start from v2 (`--warm-start`), checkpoint/resume recipe.
4. **Gate (pre-registered)**: public-231 ECE ≤ 0.20 AND no regression on
   data/test.jsonl accuracy (McNemar, α=0.05) AND raw 3-class NLI ECE ≤ 0.08.

## Phase 2 — Hard-tier curriculum (the +3–5-point lever) — generation + 1 round

**Goal:** hard 30.9% → ≥50%, judge ≥55% (Intelligence → 63+).

1. **Scale the weak-family generators** (`generate_hard_decisions_synthetic.py`
   already emits exactly these families) through the 4-judge committee
   (crash-safe, resumable — built). Target ~20–30K validated rows, weighted
   toward tradeoff/multi_hop/ambiguous/long_policy per Phase 0 audit.
2. **Balanced gold:alt exposure** to fix confirmation asymmetry (1:1 gold/alt
   in grouped items).
3. **Gate** (A/B, McNemar on public-231 hard tier, α=0.05): arm B (+hard data)
   must beat arm A (Phase-1 checkpoint) with no ECE regression. Precedent: the
   stage-2 synthetic gate FAILED — that gate was flagship-NLI accuracy; this one
   gates hard-tier capability, and synthetic hard rows only enter through it.

## Phase 3 — Judge & rubric slice

judge_hard/adequacy/policy: rubric-grounded grading data via the teacher
(gemma-4-31b-q4 + committee) — answer-vs-rubric pairs with graded golds; reuse
the SDK grade() parity slice format. Smaller volume (~5–10K), same gates.

## Phase 4 — Score & publish

- Re-run public-231 with per-family artifact; request official 534-item scoring
  from the maintainers (or submit the row).
- Realistic landing zone: **66–70** (open-alternative-jev / system-one-open
  territory) with Speed 100 and honest cost basis; stretch 72+ (Laya) if Phase 3
  lands.
- Then: HF upload with the honest model card (leading with NLI + 128K + speed +
  calibration; JevBench row linked).

## Parallel engineering debt (small, non-blocking)

- 10 GiB GPU pin diagnosis (`TCE_MEM_DEBUG=1`, next occurrence)
- v2 temperature refit on ECE objective (test ECE 0.070 → target ≤0.05)
- `test.jsonl` embedded-con relabel + restate all runs (5 min, per-item logs ready)
- A2/A5 external-claims gate for the internal suite (already partially satisfied
  by the full-set JevBench-style reruns)

## Sequencing & budget

| Phase | Wall time | GPU |
|---|---|---|
| 0 Audit & instrument | ~0.5 day | ~1 h |
| 1 Distribution loss | ~1 day (code + 4–6 h train) | 1 round |
| 2 Hard-tier data + train | ~1.5 days (committee generation overnight + 4–6 h train) | 1 round |
| 3 Judge slice | ~1 day | 1 round |
| 4 Score & publish | ~0.5 day | — |

Every phase reuses proven machinery: warm-start recipe, checkpoint/resume,
committee pipeline, A/B gates, per-item McNemar. No architecture changes.
