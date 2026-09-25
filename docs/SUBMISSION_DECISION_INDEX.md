# Run Request / Submission: Gevva e2b and Gevva e4b on Jev Decision Index

**Target Repository**: [https://github.com/apolinario/decision-index](https://github.com/apolinario/decision-index)  
**Space**: [https://huggingface.co/spaces/multimodalart/jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index)  
**Issue Title**: `Run request: Gevva e2b and Gevva e4b (Gemma 4 System 1 Cross-Encoders)`

---

## Issue / Submission Body

Hi Pedro (@apolinario / @multimodalart),

We would like to request adding **Gevva e2b** and **Gevva e4b** to the **Jev Decision Index** (Decision Index 0.2 suite). 

We know the leaderboard runs all entrants through the frozen 120,615-request suite on a standardized RTX 6000 Ada to keep latency and throughput strictly comparable across models. We have built and verified a clean in-process engine adapter compatible with `decision_index.engines.base.Engine`.

---

### 1. Models & Weights

Both models are fully open source (Apache 2.0) sequence classification cross-encoders built on Google's Gemma 4:

* **`davidburhans/gevva-e2b`**: [https://huggingface.co/davidburhans/gevva-e2b](https://huggingface.co/davidburhans/gevva-e2b)
  * Backbone: `google/gemma-4-E2B-it` (2.3B params, 18 layers, 128K context window).
  * Architecture: Gemma 4 backbone with Gemma4RMSNorm pooled projection and a 3-class calibrated classification head (`Contradiction: 0, Entailment: 1, Neutral: 2`).
  * Latency: ~16.5 ms on RTX 5090.
* **`davidburhans/gevva-e4b`**: [https://huggingface.co/davidburhans/gevva-e4b](https://huggingface.co/davidburhans/gevva-e4b)
  * Backbone: `google/gemma-4-E4B-it` (4.5B params, 42 layers, 128K context window).
  * Architecture: Full 4.5B backbone with Gemma4RMSNorm pooled projection and 3-class calibrated linear score head.
  * Latency: ~17.8 ms on RTX 5090.
  * Performance: 84.0% ARC-Challenge, 92.0% ARC-Easy, 75.0% WinoGrande, 59.0% MMLU, 76.6% JevBench.

---

### 2. Readout & Mechanism

* **Non-Autoregressive System 1**: Unlike generative LLMs that output tokens sequentially, Gevva scores options in a single forward pass.
* **Option Scoring**: For `choice` questions, each candidate is evaluated against the state/instructions as an NLI pair. The raw score is the margin $z_{\text{entailment}} - z_{\text{contradiction}}$ (or calibrated probabilities), normalized via softmax across the candidate keys so that the output probability distribution sums to 1.0.
* **Binary Decisions**: For `noul` questions, the true probability is the normalized entailment confidence $\frac{P(\text{entailment})}{P(\text{entailment}) + P(\text{contradiction})}$.
* **Zero Thinking / Generation Tokens**: Pure feed-forward classification.

---

### 3. Engine Adapter Code (`decision_index/engines/gevva_engine.py`)

Here is the clean adapter conforming to `decision_index.engines.base.Engine` and verified against `validate()`:

```python
import json
import numpy as np
import torch
from decision_index.engines.base import Engine, text

def _softmax(logits, temperature=1.0):
    arr = np.array(logits, dtype=np.float64) / max(temperature, 1e-4)
    exp = np.exp(arr - np.max(arr))
    probs = exp / np.sum(exp)
    return (probs / np.sum(probs)).tolist()

class GevvaEngine(Engine):
    name = "gevva"
    latency = "Device-synchronized in-process request wall time including NLI formatting and cross-encoder inference."

    def __init__(self, model="davidburhans/gevva-e4b", device=None, dtype="bfloat16", max_length=4096, temperature=1.0, **options):
        super().__init__(**options)
        from gevva import load
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = float(temperature)
        self.model_id = model
        self.model = load(model, device=self.device)
        self.provenance = {
            "kind": "gevva_cross_encoder",
            "model": model,
            "device": self.device,
            "dtype": dtype,
            "temperature": self.temperature,
            "architecture": "Gemma 4 Multimodal System 1 Decision Engine",
        }

    def warmup(self):
        warm = {"warmup": {"type": "choice", "instructions": "Which color is named?", "criteria": {"red": "red", "blue": "blue"}}}
        self("The color is red.", warm)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, state, questions):
        state_str = text(state) if state not in ("", None, {}, []) else ""
        answers = {}

        for q_id, q_dict in questions.items():
            q_type = q_dict.get("type", "choice")
            instructions = text(q_dict.get("instructions", ""))
            criteria = q_dict.get("criteria", {})

            if q_type == "choice":
                keys = list(criteria.keys())
                premise = f"{state_str}\n\nQuestion: {instructions}" if (state_str and instructions) else (state_str or instructions)
                options = [f"{k}: {text(criteria[k])}" if text(criteria[k]) and text(criteria[k]).lower() != str(k).lower() else str(k) for k in keys]
                
                # Rerank via Gevva margin scoring
                result = self.model.rerank(premise=premise, options=options, hyp_format="The correct answer is: {}", scoring="margin")
                probs_list = _softmax(result.scores, temperature=self.temperature)
                
                answers[q_id] = {
                    "type": "choice",
                    "choice": keys[int(np.argmax(probs_list))],
                    "probabilities": {k: float(p) for k, p in zip(keys, probs_list)},
                }
            elif q_type == "noul":
                premise = state_str if state_str else instructions
                hypothesis = instructions if state_str else "The statement is true."
                probs = self.model.predict([(premise, hypothesis)])[0]
                p_con, p_ent, _ = probs[0], probs[1], probs[2]
                p_true = float(p_ent / (p_ent + p_con + 1e-6))
                answers[q_id] = {
                    "type": "noul",
                    "noul": max(0.0, min(1.0, p_true)),
                }

        return {"answers": answers}
```

---

### 4. Running the Evaluation

With `gevva` installed (`pip install gevva`), you can launch the pipeline directly:

```bash
pip install gevva

# Run Gevva e4b
python -m decision_index pipeline \
    --engine gevva \
    --option model=davidburhans/gevva-e4b \
    --out runs/gevva-e4b

# Run Gevva e2b
python -m decision_index pipeline \
    --engine gevva \
    --option model=davidburhans/gevva-e2b \
    --out runs/gevva-e2b
```

We would be thrilled to submit a Pull Request adding `gevva_engine.py` to `decision_index/engines/` if you prefer!

Thank you so much for maintaining the Jev Decision Index and creating such an open, standardized arena for the community.
