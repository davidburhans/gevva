---
license: mit
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-classification
library_name: transformers
tags:
- nli
- cross-encoder
- qwen3.5
- reranker
- text-classification
language:
- en
---

# openjev — Qwen3.5 trained as jev model

Zero-shot Doom, played by the cross-encoder from the text state (hypotheses "the nearest enemy is left of / right of /
exactly on the crosshair" → turn left / turn right / attack; 11 kills on average, oracle 18.8, ~57 ms per decision):

<video controls src="https://huggingface.co/AlexWortega/openjev/resolve/main/videos/doom_zs_position.mp4" width="720"></video>

From pixels (the frame goes in as image tokens through the Qwen3.5 vision tower, hypotheses about the monster's
x-position; 5.2 kills, still zero-shot):

<video controls src="https://huggingface.co/AlexWortega/openjev/resolve/main/videos/doom_vision_zeroshot_pixels.mp4" width="720"></video>

Zero-shot multiple-choice accuracy against Jev and Terra (their numbers read off their published chart; Chess and
GSM8K-k-choice are our constructions):

![radar](assets/radar_openjev.png)

A Qwen3.5 decoder turned into a 3-way NLI cross-encoder in the spirit of
[dleemiller's "NLI cross encoders: ways to use them"](https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use).
**Every number in the main tables is zero-shot**: the cross-encoder is applied as-is (argmax entailment over the
options, or entailment vs a reference answer) with no task-specific training. `dleemiller/ModernCE-large-nli`
(ModernBERT-large, 395M) is the reference throughout.

Checkpoint in this repo: `qwen3.5-4b-nli/` (Qwen3.5-4B backbone, 3 labels). Code: `modeling_openjev.py`, `code/`.

## Architecture

**Cross-encoder.** `Qwen3_5ForSequenceClassification`: the Qwen3.5 text backbone (Gated DeltaNet + gated attention,
decoder-only) followed by a linear `score` head over the hidden state of the *last non-pad token*. The pair is fed as a
single string (template stored in `config.nli_template`):

```
Premise: {premise}
Hypothesis: {hypothesis}
```

Three labels in dleemiller's order: `0 = contradiction, 1 = entailment, 2 = neutral`. Right padding; `pad_token_id`
lives on `config.get_text_config()`. The vision tower that ships with Qwen3.5 checkpoints is kept in the weights but is
not used for text. Loss for the NLI head: cross-entropy over the 3 classes (full fine-tune of the text backbone for
0.8B/2B/4B).

`modeling_openjev.py` also ships `LatentMLPHead` (a probe on the frozen latent, soft-BCE loss) — it is *not* zero-shot
and its numbers are confined to the appendix at the end.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo, sub = "AlexWortega/openjev", "qwen3.5-4b-nli"
tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=sub, dtype=torch.bfloat16).cuda().eval()
tok.padding_side = "right"
model.config.get_text_config().pad_token_id = tok.pad_token_id

pairs = [("A man is playing a guitar on stage.", "Someone is making music."),
         ("A man is playing a guitar on stage.", "The room is silent.")]
texts = [model.config.nli_template.format(premise=p, hypothesis=h) for p, h in pairs]
enc = tok(texts, padding=True, return_tensors="pt").to("cuda")
probs = model(**enc).logits.float().softmax(-1)          # [contradiction, entailment, neutral]
```

`sentence_transformers.CrossEncoder` also loads it (`CrossEncoder(repo, subfolder=sub)` on recent versions) but note the
tokenizer has no pair separator: build the `Premise/Hypothesis` string yourself as above.

With the helper module:

```python
from modeling_openjev import OpenJevCrossEncoder, LatentMLPHead
ce = OpenJevCrossEncoder("AlexWortega/openjev", subfolder="qwen3.5-4b-nli")
ce.predict(pairs)                                   # probs
ce.rerank("Which gas do plants absorb?", ["CO2", "O2", "N2", "He"])   # zero-shot option index
ce.grade("What is 2+2?", reference="4", candidate="4")                   # entailment / contradiction / neutral
```

## Results (zero-shot)

Protocols follow the blog: **rerank without reference** (#3): premise = question, hypothesis = `The correct answer is: {option}`
(WinoGrande: the sentence with the blank filled; HellaSwag: the ending), pick argmax P(entailment).
**Grading with reference** (#6): premise = question + `Reference answer: {gold}`, hypothesis = `Answer: {option}`;
predict entailment iff the option is the gold one. Chess is our synthetic set (random position, 4 SAN moves, one legal);
GSM8K k-choice = gold final answer + numeric distractors; GSM8K best-of-4 uses candidates sampled from Qwen3.5-4B.

### NLI sanity

| model | MNLI-m | MNLI-mm |
|---|---|---|
| openjev-0.8B | 0.869 | 0.874 |
| openjev-2B | 0.886 | 0.889 |
| openjev-4B | 0.904 | 0.907 |
| Qwen3.5-2B head-only | 0.829 | 0.841 |
| ModernCE-large-nli (reference) | 0.909 | 0.921 |

### Multiple choice, rerank without reference (zero-shot entailment)

| model | GPQA-diamond | MMLU | ARC-Easy | ARC-Challenge | WinoGrande | Chess (4 moves, 1 legal) | HellaSwag | GSM8K 4-choice | GSM8K 10-choice |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.250 | 0.250 | 0.250 | 0.250 | 0.500 | 0.250 | 0.250 | 0.250 | 0.100 |
| openjev-0.8B | 0.207 | 0.351 | 0.555 | 0.375 | 0.504 | 0.280 | - | - | - |
| openjev-2B | 0.237 | 0.394 | 0.629 | 0.491 | 0.534 | 0.324 | - | - | - |
| openjev-4B | 0.273 | 0.472 | 0.769 | 0.592 | 0.586 | 0.240 | 0.317 | 0.392 | 0.175 |
| Qwen3.5-2B head-only | 0.253 | 0.378 | 0.596 | 0.497 | 0.544 | 0.334 | - | - | - |
| ModernCE-large-nli (reference) | 0.242 | 0.354 | 0.607 | 0.416 | 0.569 | 0.260 | - | - | - |

### Multiple choice, grading with reference (accuracy / F1)

| model | GPQA-diamond | MMLU | ARC-Easy | ARC-Challenge | WinoGrande | Chess (4 moves, 1 legal) | HellaSwag | GSM8K 4-choice | GSM8K 10-choice |
|---|---|---|---|---|---|---|---|---|---|
| openjev-0.8B | 0.939 / 0.891 | 0.951 / 0.909 | 0.974 / 0.951 | 0.958 / 0.922 | 0.866 / 0.877 | 0.996 / 0.992 | - | - | - |
| openjev-2B | 0.956 / 0.918 | 0.968 / 0.940 | 0.984 / 0.970 | 0.972 / 0.947 | 0.828 / 0.852 | 0.997 / 0.994 | - | - | - |
| openjev-4B | 0.968 / 0.940 | 0.974 / 0.949 | 0.993 / 0.986 | 0.987 / 0.975 | 0.843 / 0.863 | 0.997 / 0.993 | 0.980 / 0.960 | 0.995 / 0.990 | 0.993 / 0.967 |
| Qwen3.5-2B head-only | 0.932 / 0.865 | 0.897 / 0.807 | 0.938 / 0.887 | 0.932 / 0.869 | 0.716 / 0.761 | 0.929 / 0.876 | - | - | - |
| ModernCE-large-nli (reference) | 0.912 / 0.846 | 0.953 / 0.912 | 0.969 / 0.941 | 0.964 / 0.931 | 0.916 / 0.916 | 0.975 / 0.948 | - | - | - |

### GSM8K, 200 test questions, candidates from Qwen3.5-4B (greedy + 4 samples)

| model | greedy | pass@1 | maj@4 | NLI rerank@4 | oracle@4 | grading acc / F1 |
|---|---|---|---|---|---|---|
| openjev-0.8B | 0.880 | 0.859 | 0.915 | 0.865 | 0.940 | 0.986 / 0.992 |
| openjev-2B | 0.880 | 0.859 | 0.915 | 0.855 | 0.940 | 0.996 / 0.998 |
| openjev-4B | 0.880 | 0.859 | 0.915 | 0.845 | 0.940 | 0.996 / 0.998 |
| Qwen3.5-2B head-only | 0.880 | 0.859 | 0.915 | 0.885 | 0.940 | 0.582 / 0.681 |
| ModernCE-large-nli (reference) | 0.880 | 0.859 | 0.915 | 0.850 | 0.940 | 0.984 / 0.991 |

### Few-shot (5 solved examples in the premise, hypothesis `Answer: {option}`; MMLU on a 2k subsample)

| model | MMLU-2k 0-shot | MMLU-2k 5-shot | GPQA 0-shot | GPQA 5-shot | MMLU grading 0/5-shot | GPQA grading 0/5-shot |
|---|---|---|---|---|---|---|
| openjev-0.8B | 0.350 | 0.355 | 0.207 | 0.207 | 0.952 / 0.917 | 0.939 / 0.937 |
| openjev-2B | 0.389 | 0.393 | 0.237 | 0.212 | 0.968 / 0.947 | 0.956 / 0.957 |
| openjev-4B | 0.474 | 0.521 | 0.273 | 0.278 | 0.974 / 0.974 | 0.968 / 0.968 |
| Qwen3.5-2B head-only | 0.370 | 0.405 | 0.253 | 0.273 | 0.901 / 0.885 | 0.932 / 0.909 |
| ModernCE-large-nli (reference) | 0.336 | 0.326 | 0.242 | 0.247 | 0.948 / 0.887 | 0.912 / 0.891 |

Demos do not help the entailment head (it is not an in-context learner); only the 4B gains a few points on MMLU.

### Radar vs Jev and Terra

![radar](assets/radar_openjev.png)

All three series are zero-shot. Jev / Terra values are read off their published chart (approximate). Chess and GSM8K k-choice are our own constructions and may differ from theirs. Polygon area is not an aggregate score.

## Real-time demos (zero-shot)

The cross-encoder as a game policy with no training at all: the game state is the premise, the actions are the hypotheses, the action with the highest P(entailment) is taken every frame.

**Flappy Bird** (`code/flappy.py`, game at 15 fps, one 4B forward per option per frame, ~55–65 ms per decision with `flash-linear-attention` kernels; 6 episodes, 900 frames max = 28 pipes). Premise = the state as a sentence (height, velocity, next gap, offset from the gap centre). Only the hypothesis wording changes; each hypothesis is a statement about the state bound to an action:

| hypotheses (→ action) | pipes mean | pipes max |
|---|---|---|
| `The correct action is: flap` / `… do nothing` | 0.0 | 0 |
| `The bird should flap now.` / `The bird should not flap now.` | 0.0 | 0 |
| same, with a rule of thumb added to the premise | 17.7 | 28 |
| `The bird is below the centre of the gap.` → flap / `… above …` → do nothing | 27.5 | 28 |
| `… below the centre of the gap or falling fast.` / `… above …` | 0.3 | 2 |
| `The offset relative to the gap centre is negative.` → flap / `… positive.` → do nothing | **28.0** | 28 |
| (oracle heuristic) | 24.5 | 28 |
| (random) | 0.0 | 0 |

Asking the model to *name the action* fails; asking it to *verify a statement about the state* and binding that statement to an action gives a perfect game (28/28, above the heuristic oracle). Videos: `videos/flappy_nli.mp4`.

**Doom, ViZDoom "Defend the Center"** (`code/doom.py`, turn left / turn right / attack, one decision per 4 tics = 114 ms budget, 5 episodes):

| input | hypotheses (→ action) | kills mean | kills max | decision latency |
|---|---|---|---|---|
| - | (random) | 1.0 | 2 | - |
| labels buffer | (oracle heuristic) | 18.8 | 22 | - |
| text: enemies from the labels buffer (name, offset from crosshair, distance) | `The correct action is: turn left / turn right / attack` | 1.0 | 1 | 60 ms |
| text, same state | `The nearest enemy is to the left of the crosshair.` → turn left / `… to the right of …` → turn right / `… exactly on the crosshair.` → attack | **11.0** | 16 | 57 ms |
| text, same state | same + `There is no enemy in view.` → turn left | 10.2 | 14 | 58 ms |
| **pixels**: the 320×240 frame as image tokens through the Qwen3.5 vision tower; hypotheses about the monster's x-position (left / a little left / centre / a little right / right / none) → turn left / attack / turn right | zero-shot NLI | 5.2 | 8 | 101 ms |
| pixels, 3 coarse position hypotheses | zero-shot NLI | 3.4 | 6 | 94 ms |
| pixels, positions as % of screen width | zero-shot NLI | 3.4 | 6 | 102 ms |

The pixel variant is the only one that needs no game-state parser at all: the frame goes in as image tokens (the vision tower was never fine-tuned; the NLI head only ever saw text), the hypotheses describe where the monster is, and each hypothesis is bound to an action. 5.2 kills vs 1.0 random and 18.8 for the oracle, ~100 ms per decision. With the text state and state-statement hypotheses the zero-shot head reaches 11.0 kills (oracle 18.8). Videos: `videos/doom_zs_position.mp4` (text state, position hypotheses), `videos/doom_nli.mp4` (text state, action hypotheses), `videos/doom_vision_zeroshot_pixels.mp4` (pixels).

## Files

* `qwen3.5-4b-nli/` — the 4B cross-encoder (transformers format, `id2label` in dleemiller order).
* `modeling_openjev.py` — `OpenJevCrossEncoder`, `LatentMLPHead`, `soft_bce`.
* `code/` — training (`train.py`), evaluation (`eval.py`, `latent_mlp.py`, `summarize.py`, `radar.py`), demos (`flappy.py`, `flappy_video.py`, `doom.py`, `sweep_flappy.sh`).
* `results/` — raw JSON for every table above. `assets/`, `videos/`.

Reference model: [dleemiller/ModernCE-large-nli](https://huggingface.co/dleemiller/ModernCE-large-nli). License: MIT (model weights inherit the Qwen3.5 license).

---

## Appendix — not zero-shot: linear/MLP probes on the frozen latent

For reference only. A small head `Linear(d, 512) → GELU → Dropout(0.1) → Linear(512, 1)` on the pooled last-token latent of the *frozen* cross-encoder, one scalar per (question, option), trained with a soft BCE loss (`BCEWithLogits(logit, y·(1−ε) + (1−y)·ε)`, ε = 0.1, positives re-weighted by (1−p)/p) on a few thousand labelled questions per task; argmax within each question at inference. `raw` = the same Qwen3.5 backbone without the NLI fine-tune.

| backbone | GPQA-diamond | MMLU | ARC-Easy | ARC-Challenge | WinoGrande | Chess (4 moves, 1 legal) | HellaSwag | GSM8K 4-choice | GSM8K 10-choice |
|---|---|---|---|---|---|---|---|---|---|
| openjev-0.8B zero-shot | 0.207 | 0.351 | 0.555 | 0.375 | 0.504 | 0.280 | - | - | - |
| openjev-0.8B + MLP | 0.217 | 0.415 | 0.771 | 0.557 | 0.567 | 0.770 | - | - | - |
| openjev-2B zero-shot | 0.237 | 0.394 | 0.629 | 0.491 | 0.534 | 0.324 | - | - | - |
| openjev-2B + MLP | 0.268 | 0.495 | 0.859 | 0.701 | 0.622 | 0.796 | - | - | - |
| raw Qwen3.5-2B + MLP | 0.273 | 0.485 | 0.856 | 0.663 | 0.592 | 0.778 | - | - | - |
| openjev-4B zero-shot | 0.273 | 0.472 | 0.769 | 0.592 | 0.586 | 0.240 | 0.317 | 0.392 | 0.175 |
| openjev-4B + MLP | 0.338 | 0.597 | 0.924 | 0.847 | 0.718 | 0.804 | 0.873 | 0.577 | 0.371 |
| raw Qwen3.5-4B + MLP | 0.303 | 0.589 | 0.920 | 0.798 | 0.695 | 0.854 | - | - | - |

Soft-target ε (0 / 0.1 / 0.2) and appending the 3 NLI logits to the latent change nothing beyond noise; a joint head over all tasks is 1–4 pts behind per-task heads.

Data scaling of the 4B head (fraction of a larger labelled pool): MMLU 0.581 → 0.582 → 0.593 (0.1 / 0.3 / 1.0), WinoGrande 0.692 → 0.721 → 0.722, Chess 0.790 → 0.816 → 0.882. More data helps only where the pool matches the test distribution.

With such a probe the game policies reach oracle level (Flappy 28/28 pipes, Doom 16.0 kills mean vs oracle 18.8); raw JSON in `results/` (`latent_mlp_*.json`, `flappy_4b*.json`, `doom_4b.json`).
