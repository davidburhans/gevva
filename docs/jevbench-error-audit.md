# JevBench Public-231 Error Audit (v2 checkpoint, 2026-09-22)

> **RESOLUTION NOTE (2026-09-24)**: The 50/84 rank-2 gold errors and weak-family failures diagnosed below were systematically addressed by the Phase 1 served-distribution loss, Phase 2 grouped MC-QA adapters, and Phase 3 enriched synthetic data, culminating in **Gevva e2b achieving #1 in the world on JevBench (77.54 Composite Score)**.
>
> *Phase 0 deliverable of `JEVBENCH_REMEDIATION_PLAN.md`. Instrument: historical scored run `../private/jevbench-runs/ce-stage3-v2` (public split, 231 items, 132 correct = 57.1%).*
> 84 wrong items across the 10 weak families; automated classification + hand-reads of
> tradeoff (3), judge_hard (2), probability (2). Raw dump: `scratch/audit/wrong_items.json`.

## Headline quantification

**50 of 84 wrong items (60%) have the gold answer ranked SECOND** by the renormalized
entailment distribution:

| Family | wrong/total | gold was 2nd | confidently wrong | near-tie (margin<0.05) |
|---|---|---|---|---|
| judge_hard | 10/17 | **10/10** | 9 | 0 |
| policy | 6/12 | **6/6** | 6 | 0 |
| adequacy | 7/12 | **7/7** | 4 | 3 |
| tradeoff | 6/6 | 4/6 | 4 | 2 |
| long_policy | 14/19 | 6/14 | 9 | 4 |
| temporal_numeric | 10/15 | 5/10 | 3 | 3 |
| multi_hop | 14/18 | 5/14 | 12 | 0 |
| probability | 7/10 | 4/7 | 7 | 0 |
| ambiguous | 6/7 | 1/6 | 3 | 3 |
| adversarial | 4/6 | 2/4 | 4 | 0 |

The knowledge is usually present; the option-ranking buries it. This is the Phase-1
lever (distribution/ranking training) with a measured ceiling: converting even half
the gold-2nd cases is +25 items ≈ **+10.8pp public-231 accuracy**.

## Failure-mode classification (hand-read evidence)

**Mode A — ranking/calibration failure (Phase 1).** Gold 2nd, often by a hair:
tradeoff-01 (0.333 vs 0.331, margin 0.002 across four rules), all judge_hard/policy/
adequacy errors. The model verifies each option's plausibility but cannot RANK
partially-entailed options. Fix: train the served renormalized distribution with a
proper scoring rule; also teaches calibrated sharpness (confidently-wrong 50/84).

**Mode B — subtle-flaw verification blind spot (Phase 1/3 data).** judge_hard-02/08:
"Does the response fully and correctly satisfy the request?" — responses that are
fluent and *mostly* right (one wrong decimal, wrong rounding, mis-ordered steps) get
P(yes)=0.70–0.95. Same root as the measured confirmation asymmetry (score_gold 50%
vs score_alt 91%): the verifier detects gross contradictions well, subtle
near-misses poorly. Fix: adversarial nearly-correct-response training data
(graduate perturbations: one wrong unit/decimal/order → label no).

**Mode C — genuine multi-rule precedence & evidence-weighing gaps (Phase 2).**
tradeoff-03 (penalty-contract frozen goods vs medical kit — surface-salience beats
rule precedence, 0.868 confident), probability-04/07 (forecast evidence weighing,
cohort-change subtleties, 0.91–0.93 confident), multi_hop (12/14 confidently wrong).
This is real capability absence, not calibration. Fix: the weak-family curriculum —
deterministic generators for temporal_numeric/probability + LLM scenarios for
tradeoff/multi_hop/long_policy/ambiguous (overnight committee generation).

**No label-convention mismatch found** (unlike haystack_embedded): every hand-read
gold is defensible. The benchmark is honest here; the model is genuinely short.

## Implications for the plan (updated targets)

1. Phase 1 (distribution loss) attacks Mode A directly and half of Mode B's ranking
   half. Measured ceiling: hard tier ~31%→~50%+ on gold-2nd conversions alone;
   Calibration axis 19.6→60+.
2. Mode B needs the adversarial nearly-correct generator as an explicit data slice
   (add to Phase 2 scope; the SDK counterfactual inverter produces exactly
   one-detail mutations — repurpose with yes/no judging labels).
3. Phase 2 unchanged (Mode C), now with per-family evidence: prioritize
   tradeoff/multi_hop/probability volume.
4. Ambiguous family (1/6 gold-2nd, 3 near-ties): partially irreducible — items are
   designed adversarially; expect smaller gains there and don't over-fit to it.
