# Adversarial Review: IMPROVEMENT_OPPORTUNITIES.md

> **Reviewer scope**: Full read of the 950-line roadmap plus code verification against the repository on 2026-09-28.
> **Stated goal under test**: Gevva models should out-perform competition one size class up (e2b vs 4B field, e4b vs ~8B field) after the next training round.
> **Tone policy**: understated over overstated. Every number below is either sourced from the repo or labeled as an estimate.
> **Verdict in one line**: The diagnostic sections (§2–§6, §10–§11) are strong and evidence-based; the projection arithmetic (§17) and several "fix" mechanisms do not survive adversarial scrutiny, and the executive summary overstates implementation status. Corrected, the plan is still worth executing — with the gates and honest ranges added by this review.

---

## 1. What Survives Scrutiny (keep, unchanged)

1. **KV-invariance prefix caching (OPT-01)** — causal-attention argument is correct; implemented in `gemma4_cross_encoder.py` (`predict_candidates`, `predict_candidates_logits`) with unit tests (`tests/test_prefix_kv_cache.py`). The redundant-encoding diagnosis (POP909 = 62% of suite runtime) is the single highest-leverage finding in the document.
2. **MQA vs DeepSpeed Ulysses incompatibility (§9)** — the `num_key_value_heads = 1` divisibility argument is correct; Ring Attention as the alternative is the right call.
3. **RAGTruth lexical-overlap diagnosis (Insight 1 / ENG-01)** — the on-repo adversarial test (formatting fixed, `p_true = 1 − p_ent` → 0.0% F1) is exactly the kind of negative result that should drive planning. The conclusion "training, not engine tricks" is correct.
4. **System 1 / System 2 scoping (Insight 7, §15)** — declining GSM8K/CRUXEval/Chess as out of scope is sound strategy, not a dodge.
5. **EXP-01 pre-registration structure (§13)** — fixed arms, pre-registered gates, McNemar testing: good practice.
6. **ACOS independence-multiplication math (Insight 2)** — 0.90^64 ≈ 0.0011 is correct and well applied.

---

## 2. Findings

Severity: **C** = changes decisions or invalidates projections; **M** = wrong/unsupported claim, must correct; **m** = consistency/hygiene.

### F-01 (C) — Phase A books gains the document itself refutes
- §5 ENG-01 states plainly: *"Formatting alone does NOT solve RAGTruth … yielded 0.0% F1 … requires dedicated sentence-level counterfactual fine-tuning (TR-06), not superficial engine tricks."* Yet §17 Phase A credits ENG-01 with RAGTruth 0% → 35% skill (**+1.10 index points**) as a *zero-training* fix. Self-contradiction. The +1.10 must move to Phase B behind TR-06 with a wide range.
- The ANLI row books **+0.90** via "TR-01: zero-temperature margin scoring" under a "Zero Training Required" header. TR-01 is a *training* intervention, and temperature changes do not alter argmax ranking (F-03). Miscategorized twice over.
- The MMLU-Pro row books **+0.35** via ENG-03 (F-03: rank-invariant no-op for accuracy).
- **Net effect**: of the claimed +3.20 Phase A points, roughly +2.35 are unsupported. Reviewed Phase A: **+0.3 to +0.9**.

### F-02 (C) — Double counting between Phase B and Phase C
- Phase B's ContractNLI row already credits "TR-03 & TR-15" (de-sliding) for 52.4% → 66%. Phase C separately books **+0.70** index points for "Attention De-Sliding". The same intervention is counted twice. Additionally, Phase C books +0.40 (API-Bank) and +0.30 (ContractNLI) for *latency* reductions as accuracy index deltas; the Decision Index skill metrics for those benchmarks are accuracy-based, so latency alone should book **0** unless a specific latency-sensitive quality objective is identified (RouterBench is the candidate — verify).

### F-03 (M) — ENG-03 temperature scaling is a mathematical no-op for top-1 accuracy
Dividing every candidate's logits by a shared temperature $T(K)$ is a monotone transformation: argmax, ranking, and top-1 accuracy are unchanged. The stated rationale ("contracts distractor variance") is backwards — affine scaling multiplies score mean and dispersion together, leaving $\mu/\sigma$ and hence $P(\text{gold ranks first})$ invariant. Temperature only matters where *absolute* probabilities are compared to thresholds. The genuine lever for large-$K$ robustness is training-time grouped/listwise exposure to $K$ distractors, which §11 Insight 3 already names ("group-atomic ranking loss"). **Required change**: recategorize ENG-03 as (small) threshold calibration; move the accuracy claim to a new listwise training initiative (now TR-21 in the revised roadmap).

### F-04 (M) — ForecastBench projection ignores the discrimination bound — **RESOLVED BY MEASUREMENT (2026-09-28)**
AUC audit on run artifacts (n = 10,139 resolved ForecastBench items, base rate 0.353):
- **e4b: AUC 0.585** — raw predictions (Brier 0.2380) are *worse than the base rate* (0.2283). 5-fold CV isotonic bound: **Brier 0.2189 → 12.4% skill**. The pre-review "28%+" is unreachable for e4b; its weakness is discrimination (capability), not calibration. The pre-review root-cause story ("saturated sigmoid, p≈0.05/0.95") was empirically wrong — outputs cluster 0.4–0.6.
- **e2b: AUC 0.761**, underconfident — CV isotonic: **Brier 0.1814 → 27.4% skill** (from 17.9%), a near-free +0.2 index-point win booked in §17.4.
Bookings updated in §5 ENG-05, §11 Insight 11, §17 Phase A. Production calibration must use held-out, temporally valid resolved questions and be refit on a schedule.

### F-05 (M) — Executive summary overstates implementation status
Code audit 2026-09-28 (`grep` across `gemma4_cross_encoder.py`, `train_cross_encoder.py`, `finetune.py`, `gevva/`):
- Implemented: **OPT-01** (prefix KV), **OPT-03** (adaptive OOM bisection), partially **OPT-11** (adaptive suffix chunking in `_eval_suffix_chunk_adaptive`).
- **Not implemented anywhere in the repo**: OPT-02 (token-budget batching), OPT-05 (OOS routing), OPT-06 (de-sliding — model still runs with native sliding windows), OPT-07 (attentive pooling — `forward` still does last-token pooling at `gemma4_cross_encoder.py:222–235`), OPT-08/ENG-04 (conformal), ENG-05 (isotonic), TR-14 (plateau controller — no `min_lr`/extension logic in `finetune.py`).
The exec summary says these are "Resolved by …". **Required change**: reserve "resolved" for implemented-and-measured; label the rest "Proposed". A roadmap whose summary misstates status will misallocate the next training round.

### F-06 (M) — Omitted negative evidence: the synthetic-inclusion gate FAILED
`results/gate_decision.json`: pre-registered A/B (n = 3,113 paired evals), Arm B (synthetic blend) +0.58pp accuracy, **McNemar p = 0.182 → gate failed**. The roadmap's SYN-01…07 propose 135k+ synthetic pairs across seven generators without citing this. Also, the arms on disk (`data/armA_train.jsonl` 39,494 rows / `armB_train.jsonl` 39,975 rows, 1.2% synthetic) do **not** match the EXP-01 spec in §13 (100k pairs, 65% medium) — EXP-01 as described has not run. **Required change**: cite the failed gate in §13/§14; treat synthetic gains as unproven until each SYN pipeline passes its own McNemar gate; reconcile or regenerate arms.

### F-07 (M) — e4b ANLI skill 0.00% — **INVESTIGATED & RESOLVED (2026-09-28): curriculum gap, not an engine bug**
- Hypothesis "engine/framing bug" was wrong; the payloads and adapter path are identical between runs (same `payload_sha256` set). The evidence:
- **Neutral-class collapse.** From `runs/gevva-e4b-0.2/results.jsonl` vs gold (n = 3,200; A=entailment, B=neutral, C=contradiction):

| | pred A | pred B | pred C | recall |
| :--- | ---: | ---: | ---: | ---: |
| **e4b** gold A | 775 | 37 | 258 | 0.724 |
| **e4b** gold B | 571 | **31** | 466 | **0.029** |
| **e4b** gold C | 643 | 14 | 405 | 0.381 |
| **e2b** gold B | 253 | 285 | 530 | 0.267 |

  e4b predicted "neutral" 82 times out of 3,200 (2.6%). With near-zero recall on one of three balanced classes, macro-F1 (31.01%) lands *below* the 33.2% random baseline → skill clamps to 0. (This also identifies the previously unexplained "31.0%" baseline quoted in TR-01: it is e4b's raw ANLI macro-F1.)
- **Why: the e4b training mixture contained zero adversarial-NLI rows.** `data/train_e4b_overnight.jsonl` (107,771 rows) has no ANLI/WANLI/CAD source of any name; its only NLI anchor is `anchor_foundational_nli` (33,703 SNLI/MNLI-style rows, balanced incl. 12,000 easy topic-unrelated neutrals). e2b's `train_phase3_enriched.jsonl` included 3,795 balanced ANLI anchor rows across its multi-stage training. ANLI's neutral class is *adversarially related but unwarranted* — a pattern easy-SNLI neutrals do not teach. e4b also shows mushier confidence (mean max-prob 0.41 vs e2b 0.45), consistent with the single-epoch LR-clamped head (Insight 9).
- **Remedy**: already prescribed — TR-01 adversarial anchor replay (ANLI R1–R3, WANLI, CAD) in the e4b Phase 5 mixture, which was absent from its first training round. Expected recovery to ≥ e2b level (44.7% raw ≈ 17 skill ≈ **+0.3–0.5 index points**), booked in Phase B (§17). No engine change required.

### F-08 (M) — Train/serve parity gap for de-sliding
If OPT-06 trains with full attention but the W4A16 compressed export (`ckpt/gemma-4-e2b-nli-w4a16*`, served via compressed-tensors/vLLM) runs with native sliding windows from the model config, served behavior will mismatch training — the exact class of bug the project's own "Train-Serving Parity" rule prohibits. **Required change**: the de-sliding gate must include verification on the *quantized export path*, not only the BF16 engine (`sliding_window=None` propagated to export config, plus a served-distribution parity test; `tests/test_served_distribution_loss.py` is the right harness to extend).

### F-09 (M) — OPT-07 breaks checkpoint compatibility and has no acceptance gate
Attentive pooling replaces the input to the score head; existing champion checkpoints cannot warm-start the new head as-is (the doc's two-stage warmup trains the head from scratch). The doc prescribes a warmup protocol but no acceptance criteria or rollback. **Required change**: gate vs the last-token baseline on the stratified anchor set — accept only if ΔMacro-F1 ≥ +1.0pp with no ECE regression; keep the last-token path as a runtime fallback so a failed ablation costs nothing.

### F-10 (M) — τ_attn "entropy invariance" is mislabeled and unproven
The quoted values (1.0 / 1.15 / 1.38) are the *scaling factors* $\tau$, not entropies; attention entropy is not "1.0" at L=512 (that would be near-uniform). $\sqrt{\ln L / \ln 512}$ is a plausible heuristic under random-key assumptions, not settled math, and it stacks with YaRN's own softmax temperature $t = 1 + 0.1\ln s$ — two overlapping damping knobs with no ablation isolating either. **Required change**: present as an ablatable heuristic (optionally a learned per-layer scalar initialized at the formula), gated by the short-context regression check in F-08's gate set.

### F-11 (M) — Phase B baselines look copy-pasted from e2b — **RESOLVED (2026-09-28, all e4b baselines re-verified from `runs/gevva-e4b-0.2/benchmark-summary.json`)**
Verified e4b figures: HoVer 57.77 (≈ e2b's 57.97 — no scale advantage without multi-hop training), ContractNLI 54.47 (was misquoted 52.40), MMLU-Pro 36.41 raw / ≈28.5 skill (was misquoted 19.70 — the pre-review Phase A even "projected" e4b to 28.50, a value it already had), RAGTruth 36.68 raw F1 (0.00 is only the skill clamp below the 41.13 random baseline), Home Appliance 15.00 ✓, MuSR 58.64 raw / 34.24 skill ✓, ANLI 31.01 ✓. §17 Phase B table updated with verified values; the "Home Appliance (Cat 40)" mislabel fixed to Cat 9.

### F-12 (m) — Request-count inconsistency
§1 origin line says **151,476**; §10 and the scorecard say **151,034** (twice, as completed counts). Standardize on the completed count and note the delta (442 requests) if it is real (e.g., retried/voided items).

### F-13 (m) — "At or below random chance" misclassifies ChessBench
ChessBench scored 11.76% against an 8.2% chance baseline — *above* chance. Three of the four named benchmarks are below chance; one is above. Reword.

### F-14 (m) — Layer-ratio inconsistency
"5 out of every 6 layers" (≈83.3%) vs "28 of 35 (80%)" for e2b. Both cannot be exact. Read `config.json` from each checkpoint and quote exact per-model `layer_types` counts.

### F-15 (m) — Landmark-sink implementation gap
Pooling over $[H_{\text{landmark}}, H_{\text{hyp}}]$ requires hidden states at landmark positions *inside the cached prefix*; prefix-KV serving does not retain them by default. Requires a hidden-state capture hook during stage-1 prefill (stash 32 vectors per 128K request). Not mentioned in the doc; affects the SYN-07 gate design.

### F-16 (m) — YaRN formula as written mixes conventions
Linear-YaRN interpolates frequencies as $(1-\gamma) + \gamma/s$; NTK-aware applies the $d/(d-2)$ exponent to the *base*. The doc's $\gamma_i \cdot s^{-d/(d-2)}$ inside the ramp is a hybrid of the two. Also $L_{\text{base}} = 8{,}192$ while Stage-1 training runs at 4,096 — reconcile which is the true base. Verify against a reference YaRN implementation before coding.

### F-17 (m) — Conformal "empty set = OOD" is a heuristic, not a guarantee
Under the given cumulative-scoring construction, an empty set arises when $p_{(1)} > \hat{q}_\alpha$ — i.e., the model is more confident than the calibration distribution. That correlates with OOD but also occurs under calibration drift or aggressive $\alpha$. Phrase as a flag, not a certified anomaly detector. Separately: **conformal abstention must be disabled (forced singleton) in leaderboard mode**, or suite accuracy will be penalized for the safety feature. The doc never reconciles the two operating modes.

### F-18 (M) — ENG-02 per-benchmark threshold fitting may violate suite rules
The doc itself quotes the upstream rule: *"Do not adapt the prompt per benchmark; refuse instead."* Fitting $\tau$ on ACOS validation cases to maximize the ACOS suite metric is per-benchmark adaptation in spirit even if prompts are untouched. **Required change**: confirm permitted practice with the suite maintainers in writing, or default to a *global* class-prior-adjusted threshold policy fitted on generic calibration data. Book ENG-02 conservatively until resolved.

### F-19 (M) — There is no e2b plan, but e2b is the stated priority
"Beat the next size up" for e2b means the 4B field. The only 4B model ahead of e2b is **Hopper (30.01 vs 26.79)**. §17 projects only e4b. **Required change**: add an explicit e2b trajectory with gap accounting (see revised §17): engine fixes that apply to both models, curriculum gains, and the specific Hopper delta to clear.

### F-20 (M) — The 8B-class goal is not honestly framed
For e4b, "next size up" competition is the ~8B class (Jev proprietary: 51.67 skill). Current e4b: 29.88. Reviewed best-case 1.1 plan: ~+5 → ~35. The gap is ~17 points and will not close in one release. The pre-review projection (36.88%, presented adjacent to a table containing 8B/27B entries) invites the reading that 1.1 is 8B-competitive. **Required change**: state the gap plainly; make 8B-class parity a multi-release objective with named levers (listwise training at scale, teacher distillation from gemma-4-26B with better-than-75% consensus gates, 128K long-context training) and interim milestones.

### F-21 (m) — Estimates presented as measurements; cost optimism
"~90×" (exec summary) exceeds the document's own best computed case (77× POP909). Label all speedups *computed estimates* until the 2.5h-suite rerun measures them. 8×H100 SXM at "\$18–24/hr" is below current market (~\$24–35/hr depending on provider); the \$250–380 budget should be restated as \$300–560 to avoid a mid-run stop.

### F-22 (m) — Hyperbole
"WORLD CHAMPION", "#1 open model in the world", "decisively crushed", "massive +3.09 leap", "rock-solid stability", "zero negative impact". Absolutes invite refutation and make the document a weaker planning artifact. Replaced throughout with bounded, sourced statements. (The underlying results are good; they do not need the adjectives.)

### F-23 (m) — Internal target conflicts
TR-03 targets ContractNLI "62–66%" while TR-15 targets "70%+" for the same benchmark; Phase B books 66%. ANLI baselines appear as 31.0% (TR-01), 44.74% (§10, e2b), and 0.00% skill (e4b, §17). Harmonize on one set of sourced baselines and gate targets as ranges.

---

## 3. Corrected Projection (what the plan is worth after fixes)

| Phase | Pre-review claim | Reviewed range | What changed |
| :--- | :---: | :---: | :--- |
| A: engine/protocol fixes | +3.20 | **+0.3 … +0.9** | RAGTruth/ANLI/MMLU-Pro rows unsupported (F-01, F-03, F-04, F-07); ENG-02 gated on rules check (F-18) |
| B: curriculum + multi-epoch | +2.40 | **+1.5 … +3.5** | Synthetic reliance discounted by failed gate (F-06); baselines verified (F-11); targets as gated ranges (F-23) |
| C: latency + de-sliding | +1.40 | **+0.0 … +0.4** | De-sliding accuracy counted once (F-02); latency-only deltas zeroed pending latency-sensitive metric verification |
| **Total (e4b)** | **36.88** | **≈ 31.7 … 34.7** | Ranges are gate-contingent, not promises |

**e2b (added by this review)**: engine fixes apply equally (+0.3–0.9); a Phase-B-style curriculum round for e2b (+1.5–3.0, same gates) → **≈ 28.6 … 30.7**. Clearing Hopper (30.01) is plausible but not assured; the honest statement is "narrowly clear to clearly clear, gated on EXP-01 + TR-06 style fixes landing".

**Goal-gap honesty**: e4b vs 8B-class (51.67) remains a ~17-point gap after 1.1. Levers that plausibly move multiples of points per release: listwise K-distractor training at scale (TR-21), teacher distillation with raised consensus gates, 128K long-context training (TR-19/SYN-07), and the pooling/de-sliding architecture pair if their ablations pass.

---

## 4. Required Actions Before the Next Training Round

1. Fix §17 arithmetic per F-01/F-02/F-04 (done in the revised roadmap, 2026-09-28).
2. Relabel implementation status everywhere (done; see roadmap §18 status log).
3. ~~Investigate e4b ANLI = 0.00% as a probable engine bug~~ **Done (2026-09-28)**: curriculum gap, not a bug — e4b's training mixture had zero ANLI/adversarial-NLI rows; neutral recall 2.9%. Fix = TR-01 in the Phase 5 mixture; booked in Phase B (§17). See F-07 above.
4. ~~Measure ForecastBench AUC before booking ENG-05 uplift~~ **Done (2026-09-28)**: e4b AUC 0.585 → calibration ceiling 12.4% skill; e2b AUC 0.761 → 27.4%. Bookings updated.
5. Written confirmation (or safe default) on per-benchmark threshold legality (F-18).
6. Run EXP-01 exactly as pre-registered (the arms on disk are a different, earlier experiment) — it gates the entire Phase B mixture design.
7. De-sliding and attentive pooling land only behind ablation gates with rollback (F-08, F-09), including quantized-export parity.
8. Re-baseline every e4b figure in §17 from run artifacts (F-11).
9. Standardize request counts, layer counts, category IDs, and benchmark baselines (F-12–F-14, F-23).
