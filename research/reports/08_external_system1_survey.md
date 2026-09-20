# External System-1 Survey: von-1.0 & GLiNER2 (2026-09-20)

> Grounding note: von-1.0 facts cite `https://huggingface.co/wfzyx/von-1.0` (README,
> config.json, calibration.json fetched directly). GLiNER2 facts cite
> `https://huggingface.co/fastino/gliner2-large-v1` and
> `github.com/fastino-ai/GLiNER2` (README). Items marked **[pending G2]** in the
> first draft have since been **grounded (task G2, §4)** — the classification API is
> primary-source verified, no longer assumed.

## 1. wfzyx/von-1.0 — what it is (grounded)

A 395M-param ModernBERT (hidden 1024, 28 layers, 16 heads, 2048 ctx) non-autoregressive
decision model, Apache-2.0, `pip install von-sdk`. Three decision primitives:
`decide` (K-way choice with labeled/described options), `judge` (calibrated binary
probability), `rate` (ordinal expected value over ordered levels). Post-training:
composite **CE + 0.5·Brier** loss (branded "RLCD"), then **post-hoc temperature
scaling** (T=1.1692, NLL-fit on held-out logits, shipped as `calibration.json`).
Training corpus: **250K class-balanced examples from ANLI R1-3 + WANLI + MNLI + SNLI**
— a strict subset of our Stage-1 curriculum. Benchmark: 78-case `jabr/classifier-benchmark`
(macro 93.5% vs GLiNER2 78.5% vs Jev API 97.2%).

### Adopted into our program

1. **H1 externally validated** — von's headline calibration method is a narrow instance
   of report-04's Composite Proper Scoring Loss (our H1: CE + λ·proper-scoring).
   Registered λ sweep: {0.25, 0.5, 1.0} (von uses 0.5). Gated identically (test ECE at
   statistically-equal accuracy).
2. **H2 (NEW): post-hoc temperature scaling** — fit T on *validation* logits by NLL
   after training; store in checkpoint (`calibration.json`); apply inside SDK `predict`.
   Argmax-invariant, so the gate is pure ECE/Brier improvement with accuracy unchanged.
   Nearly free; von demonstrates the shipping pattern.
3. **Benchmark roster**: von-1.0 added as a LOCAL measured baseline (adapter: grounded
   decide/probabilities → our label order). GLiNER2 added and now **grounded
   (task G2)** — classification API verified (`classify_text` with
   `include_confidence=True`: top-1 label + confidence only, no full distribution,
   so ECE/Brier rows are N/A in published tables).
4. **Backlog (design, not tonight)**: ordinal `rate`-style expected-value output for
   rubric grading (SDK product idea); dual positive/negative criteria framing for
   binary verification (RAG-hallucination data augmentation); option-marker joint
   attention (single-pass multi-option scoring — architecture research); adding our
   model to the public `jabr/classifier-benchmark` for ecosystem position.

### Honesty observation

Von's README claims to surpass "published commercial alternatives" while its own table
shows Jev at 97.2% vs von 93.5% — the same claim-laundering pattern our 2026-09-20
audit removed from OUR README. Also note: their benchmark is n=78 self-published
(smoke-scale); we will not adopt it as a headline suite, and any von/GLiNER2 rows we
publish will be our own local measurements at n≥1000 under our frozen protocol.

## 2. GLiNER2 — what it is (grounded)

Schema-conditioned span/boundary extraction family (DeBERTa-v3 encoders):
`gliner2-large-v1` = 340M DeBERTa-v3-large, span architecture, Apache-2.0, loaded via
`GLiNER2.from_pretrained(...)`; local inference needs the `gliner2[local]` extra.
Strengths: NER/structured-extraction/classification in one forward pass, CPU-first.
On von's independent decision suite it scored 78.5% macro — extraction-first models
are not calibrated decision engines.

### Program relevance

- Included in our local baseline roster for completeness (**grounded (task G2)** —
  the refuse-ungrounded gate was lifted after source verification; ECE/Brier rows
  for GLiNER2 are N/A in published tables).
- Architectural lesson: schema-in-input multi-task conditioning is attractive for
  serving many decision heads from one encoder, but decision calibration (our
  differentiator) is not its strength. No training-pipeline change adopted.

## 3. Registration summary

| Item | Type | Status |
| :--- | :--- | :--- |
| H1: proper-scoring composite loss (λ sweep) | training hypothesis | REGISTERED (externally validated by von's method) |
| H2: post-hoc temperature scaling | calibration hypothesis | REGISTERED (von-grounded shipping pattern) |
| von-1.0 local baseline | benchmark | ADDED (grounded adapter + tests) |
| GLiNER2 local baseline | benchmark | ADDED, GROUNDED (task G2) |
| Ordinal rate() primitive; negation-framing augmentation; option-marker attention; jabr suite | backlog ideas | LOGGED |

## 4. Grounding addendum (task G1-G5, 2026-09-20 late evening)

**G1 — von training internals (grounded, run.log + marker_calibration.json)**:
actual run = 65,991 operational-decision samples (16K Noul, 9,998 Score ordinal,
19,993 Choice routing from Banking77+Emotion, 20K adversarial core), 1 epoch,
effective batch 64, 4-GPU DDP, 985 steps, then temperature fit. Composite loss
verbatim: L = L_CE + 0.5 * L_Brier. Learning rate: UNVERIFIED (absent from log).
README internally inconsistent on temperature (Key Capabilities: T=1.0367;
Training section + calibration.json: T=1.1692). The 250K "ANLI+WANLI+MNLI+SNLI"
README claim does not match the run.log mixture (which is decision-trajectory
heavy) - vendor docs conflict with their own artifacts.

**G2 — GLiNER2 classification API (grounded)**: `GLiNER2.from_pretrained(repo)` +
`model.classify_text(text, {"field": [labels]}, include_confidence=True)` ->
`{"field": {"label": str, "confidence": float}}` (runtime.py classify_text).
Vendor exposes ONLY top-1 label + confidence (no full distribution). Implemented;
adapter spreads remainder uniformly — accuracy rows exact, ECE/Brier rows for
GLiNER2 must be N/A in published tables.

**G3 — openjev subfolders (grounded via HF tree API)**: `qwen3.5-4b-nli` EXISTS,
`qwen3.5-4b-nli-v2` EXISTS, **`qwen3.5-0.8b-nli` ABSENT, `qwen3.5-2b-nli` ABSENT**.
Consequence: 0.8B/2B quoted rows can never become local rows (weights not
published). The planned Qwen3.5-0.8B vs openjev-0.8B phase must either train
openjev-0.8B ourselves from their public train.py (200K-pair full FT) or compare
against quoted numbers with that caveat - DECISION DEFERRED TO SOTA-GATE TIME.

**G4 — Laya XNLI protocol (partially grounded)**: XNLI ran on their harness
(English 0.860) but the per-example wording is unpublished. Grounded generic
format: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
with choice options "k: description", noul wording verbatim. Any Laya NLI row we
produce will be labeled "vendor XNLI prompt unpublished; our mapping documented".

**G5 — jabr suite (grounded)**: v1 = 8 tasks / 78 cases (`bench/cases.py`), v2 =
49 tasks / 869 cases. Adding a backend is documented (`bench/backends/base.py`
contract: predict_choice/predict_noul/predict_score). jabr's own numbers for
von-1.0.1 (0.923/0.930) are LOWER than von's README claim for von-1.0 (0.935) -
third-party vs self-published discrepancy, consistent with our audit findings.
