---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags: [laya, system-one, calibrated-decisions, rlcd, classification, routing, scoring, guardrails, moderation, reinforcement-learning, commercial-use]
---

<p align="center">
  <img src="assets/logo-lockup.png" alt="Laya" width="330" />
</p>

**Multilingual, non-autoregressive System 1 decision model.** Give it a **state** (text, email,
ticket, or JSON) and **typed questions**; it returns typed answers with probabilities in a
single forward pass — 33 ms — across 100+ languages. Trained with reinforcement learning against
strictly proper scoring rules (**RLCD**), so reporting honest probabilities is the only way to
maximise reward. It never generates text, so there is nothing to parse and nothing to
hallucinate.

<p align="center">
  <img src="assets/laya_vs_jev_full.png" alt="Laya versus TypeSafe Jev: accuracy, every application workflow, all 51 languages, speed, calibration and routing cost" width="100%" />
</p>

**This repo holds all three checkpoints** and is the hub for the family. The English checkpoint
is at the repo root; the other two are subfolders, and only the one you ask for is downloaded:

```python
import laya

laya.load("convaiinnovations/laya")                            # English
laya.load("convaiinnovations/laya", subfolder="multilingual")  # 100+ languages
laya.load("convaiinnovations/laya", subfolder="typed-decisions")
```

| checkpoint | encoder | params | context | use it for |
|---|---|---|---|---|
| **`convaiinnovations/laya`** (this repo) | ModernBERT-large | 421M | 512 | English |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages, ~2x faster |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

| question type | returns |
|---|---|
| `choice` | selected option, probability per option, confidence |
| `score` | expected level on your ordinal rubric, distribution, confidence |
| `noul` | calibrated probability P(true) |

## Quickstart

```bash
pip install laya
```

```python
import laya

agent = laya.load("convaiinnovations/laya")
result = agent.predict(
    {"subject": "Duplicate charge on invoice 4411",
     "body": "We were billed twice for March. Please refund the duplicate."},
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?",
                 "criteria": ["not urgent", "soon", "blocking"]},
     "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}},
)
print(result["answers"]["department"]["choice"])
```

### Routing between the three checkpoints

```python
from laya import Router

router = Router()                       # lazy-loads only what a request needs
router.predict({"body": "I was charged twice"}, questions)          # -> laya
router.predict({"body": "मुझसे दो बार शुल्क लिया गया"}, questions)   # -> laya-multilingual
router.predict(state, questions, model="typed-decisions")           # explicit
```

**Preload for a server or a demo.** A cold checkpoint build costs seconds; detection costs
microseconds. At the default `max_loaded=1`, alternating languages rebuilds a model on every
request — measured at a 7.4 s median on CPU and 10.3 s on a T4.

```python
router = Router(preload=True)                # every checkpoint resident, routing is free
router.attach("english", existing_agent)     # reuse one you already built
```

On a mixed workload this is worth up to **4.8×** at 50% non-English traffic.

> **If `laya.load()` hangs:** `transformers` probes for TensorFlow at import, and when TF is
> installed its abseil runtime can deadlock model construction. Run with `USE_TF=0`.

## Architecture

- **Backbone** ModernBERT-large (395M, bidirectional, fully fine-tuned) + a decision head
  trained from scratch: 2 transformer layers, an option-marker scorer, and an act/escalate head.
  421M total.
- **Option markers** every option is scored at its own `[MASK]` token, then softmaxed over that
  question's options. The answer space is defined at request time, so new schemas need no
  retraining.
- **Budget** 512 tokens per question (question + options + state).
- **Batching** every question in a call is answered in one forward pass.

## Training

**RLCD (Reinforcement Learning for Calibrated Decisions).** The policy reports a distribution;
exploration adds zero-mean Gaussian noise to the logits; the reward is a strictly proper scoring
rule (log + spherical, plus ranked probability score for ordinal questions). Expected reward is
maximised only by reporting honest probabilities. Updates are REINFORCE with a group-mean
baseline (GRPO-style). Multi-turn conversations use TD(λ=1.0) over prefix slices.

7,313 updates, 1 epoch, ~1.96 h. Fitted temperatures `[1.637, 1.251, 1.983]` with
per-option-count scaling.

## Benchmarks

Measured on a Tesla T4; every checkpoint answered byte-identical questions in the same run.

### Speed

| questions/call | `laya` | `laya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

103–332 questions/sec batched. TypeSafe Jev has been independently measured at 236–276 ms p50
([AbdelStark](https://github.com/AbdelStark/jev-benchmarks),
[nibzard](https://github.com/nibzard/decision-model-benchmark)), so Laya answers a single
question roughly **6–7× faster**.

### Laya (with routing) vs Jev

Every Laya figure is what `Router().predict(...)` returns — the checkpoint the router selects
for that input. Jev figures are **third-party published, never measured here** (no TypeSafe API
access); sample sizes and prompts differ.

| | Jev 1.13.0 | Laya (routed) | |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.727 | **0.766** | +0.039 |
| AG News, 4 labels | 0.910 | **0.950** | +0.040 |
| DAIR Emotion, 6 labels | 0.480 | **0.595** | +0.115 |
| ECE *(lower better)* | 0.246 | **0.081** | 3× better |
| p50 latency, 1 question | 236–276 ms | **32.8 ms** | 7.8× faster |
| Languages usable | *no published benchmark* | **45 of 51** | — |
| Weights | closed API | **Apache 2.0** | — |
| Cost | $0.042 / 1M tokens | **$0 self-hosted** | — |

On DAIR Emotion, Jev assigned zero probability to the true label on 16% of examples.

Full detail — every workflow, all 51 languages:
[BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)

### typed-decisions, all three checkpoints

400 cases, 2,000 decisions, four workflows — measured here.

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| **`laya-typed-decisions`** | **0.766** | 0.471 | **0.062** | 0.213 | **0.242** |
| `laya` | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher ceiling* | *0.735* | | | | |
| *majority class* | *0.461* | | | | |

The fine-tuned checkpoint clears the teacher ceiling and wins all four workflows: invoice
processing 0.804, security incidents 0.766, customer service 0.764, agent-trace observability
0.730. By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

The base checkpoints sit below the majority-class baseline here — the capability on this
benchmark comes from fine-tuning, which is what the
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
is for.

### English tasks

| task | `laya` | `laya-multilingual` | |
|---|---|---|---|
| AG News | **0.947** | 0.937 | in training mix |
| BoolQ | **0.830** | 0.787 | in training mix |
| DAIR Emotion | **0.573** | 0.513 | held out |
| prompt-injections | **0.698** | 0.578 | held out, n=116 |
| SST-5 (ordinal) | 0.372 | 0.282 | held out |

### Languages — use `laya-multilingual` outside English

Across 51 languages on MASSIVE intent (20 options, random = 0.050), this checkpoint
macro-averages **0.227** with macro ECE **0.733**, clearing 3× random on only 23 of 51.
Khmer scores **0.000 accuracy at 0.952 confidence**.

| | `laya` | `laya-multilingual` |
|---|---|---|
| MASSIVE intent, English | **0.783** | 0.657 |
| MASSIVE intent, 13 others | 0.306 | **0.451** |
| XNLI, English | **0.860** | 0.843 |
| XNLI, 14 others | 0.521 | **0.731** |

The confidence score gives no warning when the input is unreadable, so the choice has to be made
before the forward pass — that is what `Router` is for.

## Limits

- **Near chance on typed-decisions zero-shot** — 0.362 here and 0.352 for multilingual, against a
  0.318 random and 0.461 majority-class baseline. The 0.766 belongs to the checkpoint fine-tuned
  on that benchmark's own training split. Laya is a fast base to specialise, not a zero-shot
  decision engine.
- **Keep `choice` questions under ~20 options.** Options share a fixed `head_max_len` budget
  (192 tokens here), so a very large label space leaves only a few tokens per label and
  accuracy falls off sharply. Split into a coarse choice then a fine one.
- Ordinal `score` questions are the weakest primitive (SST-5 0.372).
- Ships over-confident: refitting one temperature per (question type, option count) moves mean
  ECE **0.466 → 0.081**. Do this on your own data before trusting the probabilities.
- English only. Use `laya-multilingual` for anything else.

## Links

- **GitHub** https://github.com/NandhaKishorM/laya
- **PyPI** https://pypi.org/project/laya/
- **Demo** https://huggingface.co/spaces/convaiinnovations/laya-demo
- **Write-up** [dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

Apache 2.0 · Convai Innovations
