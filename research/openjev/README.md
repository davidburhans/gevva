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
- image-text-to-text
language:
- en
---


# openjev — Qwen3.5 trained as jev model

<video controls src="https://huggingface.co/AlexWortega/openjev/resolve/main/videos/v2/doom_vision_v2.mp4" width="720"></video>

<video controls src="https://huggingface.co/AlexWortega/openjev/resolve/main/videos/minecraft_chain.mp4" width="720"></video>

**openjev** is Qwen3.5 turned into a *jev* model: a single cross-encoder that reads a premise and a hypothesis and
answers with entailment, contradiction or neutral. That one primitive is enough to rerank answers, grade them against a
reference, guard content, and play games in real time: hand it the game state and a few statements about it, and the
argmax entailment is the move. Nothing is trained per task.

## openjev-4B v2: text, images and agents

The new 4B checkpoint (`qwen3.5-4b-nli-v2/`) reads images as well as text and was trained on a much larger and harder
mixture. It is strictly zero-shot on everything shown here.

* Doom straight from the pixels (first video): **10.4 kills** per episode, twice the v1 model (5.2); random play gets 1.
* Crafts an **iron pickaxe from nothing in real Minecraft** (second video): 11 milestones in ~22 decisions, driven by a
  backward-chaining scaffold where the jev model only checks statements about the inventory and the world.
* Much stronger on adversarial NLI (ANLI r3 0.42 → 0.63, WANLI 0.63 → 0.77) and on image claims (0.52 → 0.84),
  better reranking (ARC-Challenge 0.59 → 0.72, MMLU 0.47 → 0.53), same MNLI (0.91).

Doom from the text state (v2, 11 kills per episode; a perfect-information bot gets 18.8):

<video controls src="https://huggingface.co/AlexWortega/openjev/resolve/main/videos/v2/doom_position_v2.mp4" width="720"></video>

![radar](assets/radar_openjev.png)

Bigger jev: Qwen3.5-35B-A3B (MoE) as the backbone (`qwen3.5-35b-a3b-nli/`). Zero-shot, and with the backbone frozen plus
a small MLP head on the last-token latent (`mlp_heads_35b/`, one head per task, loadable with `LatentMLPHead.load`):

![radar 35B](assets/radar_openjev_35b.png)

## What's inside

* `qwen3.5-4b-nli-v2/` — **recommended**: the 4B v2 jev checkpoint, text + images.
* `qwen3.5-4b-nli/` — the original 4B jev checkpoint (text).
* `qwen3.5-35b-a3b-nli/` — the 35B-A3B MoE jev checkpoint (load with `modeling_qwen35_moe_seqcls.py`).
* All checkpoints: `Qwen3_5ForSequenceClassification`, 3 labels `contradiction`, `entailment`, `neutral`, last-token
  pooling, trained with plain cross-entropy over the three classes.
* `modeling_openjev.py` — `OpenJevCrossEncoder`: `predict`, `rerank`, `grade`, `latents`; `LatentMLPHead` for the per-task heads.
* `modeling_qwen35_moe_seqcls.py` — `Qwen3_5MoeForSequenceClassification` for the 35B-A3B backbone.
* `mlp_heads_35b/<task>/` — `head.pt` + `norm.npz` + `meta.json`, the 35B latent + MLP heads behind the second radar.
* `code/` — everything used here: the trainer and data mixture builder, the evaluation harness, Flappy Bird, Doom
  (text and pixels), the Minecraft scaffold and bot, the radar.
* `videos/` — Flappy Bird, Doom and Minecraft replays; `results/` — raw JSON for every run and the full report.

## Use it

```python
from modeling_openjev import OpenJevCrossEncoder
jev = OpenJevCrossEncoder("AlexWortega/openjev", subfolder="qwen3.5-4b-nli-v2")

jev.predict([("The bird is 0.05 below the centre of the gap.", "The bird is below the centre of the gap.")])
# -> [[contradiction, entailment, neutral]] probabilities

jev.rerank("Which gas do plants absorb during photosynthesis?", ["oxygen", "carbon dioxide", "nitrogen"])
# -> index of the option with the highest entailment
```

Or with plain transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-4b-nli-v2")
model = AutoModelForSequenceClassification.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-4b-nli-v2")
text = model.config.nli_template.format(premise="...", hypothesis="...")
```

Images go inside the premise as `<|vision_start|><|image_pad|>…<|vision_end|>` with `pixel_values` / `image_grid_thw`
from the Qwen3.5 image processor; see `code/doom_vision.py` and `code/eval_image_nli.py`.

Reference point: [dleemiller's NLI cross-encoders](https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use). Licence MIT.
