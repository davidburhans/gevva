# Research Report 09: Survey of SOTA System 1 Decision Engines & Novel Ideas

> **Provenance & Attribution Notice**: This report compiles architectural, algorithmic, and data curation findings from 5 state-of-the-art open-source decision-engine and fast-inference projects. All code patterns, loss formulations, and benchmarks cited herein are documented with explicit attribution to their respective authors, organizations, and open-source licenses.

---

## 1. Project-by-Project Deep Dive & Attribution

### 1.1 TheoLeeCJ/SemIf
* **Attribution & Creators:** Theo Lee ([@TheoLeeCJ](https://github.com/TheoLeeCJ)), formerly known as `OpenJev` ([openjev.com](https://openjev.com)).
* **Repository & License:** [`https://github.com/TheoLeeCJ/SemIf`](https://github.com/TheoLeeCJ/SemIf) | **MIT License**.
* **Key Architecture & Objective:**
  * Non-autoregressive categorical decision readout directly from frozen causal language model weights (`Qwen/Qwen3.5-4B`, `MiniCPM5-2B`).
  * Evaluates structured prompts formatted with single-token answer slots (e.g. letters `'A'`, `'B'`, `'C'`). Uses `logits_to_keep=1` to compute only the final token's vocabulary logits, slicing out:
    $$\mathbf{z}_{\text{options}} = \mathbf{z}_{\text{vocab}}[\text{slot\_tokens}], \quad \mathbf{p} = \text{Softmax}(\mathbf{z}_{\text{options}})$$
  * **Shared-State Prefix Reuse (`shared.py`):** Prefills a long document/state once with `use_cache=True`, replicates the KV cache across $M$ criteria via native cache reordering (`cache.reorder_cache(torch.zeros(M, dtype=torch.long))`), and evaluates all criteria in a single parallel suffix pass.
* **Latency & Throughput:**
  * 5.21× faster than autoregressive JSON generation (1.02s vs 5.33s on RTX 3090).
  * Prefix cache replication achieves **20.03 decisions/s** (vs 2.33 fresh decisions/s) — an **8.6× throughput boost**.
* **Actionable Adoption for Gemma 4 Cross-Encoder:**
  * Adopt shared prefix caching in `gemma4_cross_encoder.py` for evaluating multiple hypotheses against a single 128K document context.

---

### 1.2 Bespoke-Nimble-9B
* **Attribution & Creators:** Bespoke Labs (Kartikay Khandelwal and the Bespoke Labs engineering team; [@bespokelabsai](https://github.com/bespokelabsai)).
* **Repository & License:** [`bespokelabs/Bespoke-Nimble-9B`](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | [`bespokelabsai/nimble`](https://github.com/bespokelabsai/nimble) | **Apache 2.0 License**.
* **Key Architecture & Objective:**
  * LoRA adapter ($\text{rank}=16, \alpha=32, \text{dropout}=0.05$) on `Qwen/Qwen3.5-9B`.
  * Supports three core primitives: `boolean` ("noul"), `enum` ("choice"), and `score` (ordered rubric levels).
  * Implements continuous expected value readout for rubric scoring:
    $$\mathbb{E}[\text{score}] = \sum_{i} v_i \cdot P(y = v_i)$$
  * Gathers candidate logits using `torch.gather` on pre-indexed candidate token matrices, masking invalid options with $-\infty$.
* **Data Curation:**
  * Multi-teacher contrastive synthetic generation: GPT-5.6 + Claude Sonnet 5 with strict automated rejection filtering (2,676 high-quality decision rows).
* **Performance:**
  * **91.96%** aggregate accuracy on held-out evaluations (100% on boolean tasks, 0.2757 Score Absolute Error).
* **Actionable Adoption for Gemma 4 Cross-Encoder:**
  * Implement continuous expected value rubric scoring in `gemma4_cross_encoder.py:grade()`: $\mathbb{E}[S] = \sum v_i P(v_i)$.

---

### 1.3 Mapika/decider
* **Attribution & Creators:** Mapika ([@Mapika](https://github.com/Mapika)).
* **Repository & License:** [`https://github.com/Mapika/decider`](https://github.com/Mapika/decider) | [`Mapika/decider-2b`](https://huggingface.co/Mapika/decider-2b) | **Apache 2.0 License**.
* **Key Architecture & Objective:**
  * High-throughput decision family (`0.8B`, `2B`, `35B-A3B MoE`, `2B-Vision` multimodal model).
  * **Selective Slot Projection (`decider/model.py`):** Projects hidden states at answer slots against only the weight vectors of declared options ($K \le 255$) rather than full vocabulary ($V \approx 152\text{K}$), reducing head latency to ~4ms with CUDA graphs.
  * **Dual Layout & Schema Cache:** Trains 50% State-First and 50% Schema-First (`Question ... Options ... Context`). Schema-First enables pre-compiling static routing schemas into persistent KV caches.
* **Data Curation & Abstention (`none_augment`):**
  * Appends explicit abstain/unverifiable options (75% negative control where gold target is preserved, 25% adversarial replacement where all options are replaced with distractors from unrelated task families and gold target switches to abstain).
* **Training & Loss:**
  * Composite proper-scoring loss: Cross-Entropy + $\lambda_{\text{Brier}} \cdot \text{Brier}$.
  * Deterministic token-bucket batching (`batches_by_tokens`): constrains ragged sequences to ~24 fixed $(B, T)$ shapes, eliminating CUDA graph recompilations.
* **Actionable Adoption for Gemma 4 Cross-Encoder:**
  * Integrate `none_augment` into `generate_sdk_synthetic_data.py` to train robust zero-shot abstention and neutral-boundary recognition.
  * Implement deterministic token-bucket batching in `train_cross_encoder.py` for ragged sequences up to 128K tokens.

---

### 1.4 TianyuCodings/NanoJev
* **Attribution & Creators:** Tianyu Chen ([@TianyuCodings](https://github.com/TianyuCodings), [@C-Tianyu](https://huggingface.co/C-Tianyu)).
* **Repository & License:** [`https://github.com/TianyuCodings/NanoJev`](https://github.com/TianyuCodings/NanoJev) | [`C-Tianyu/NanoJev`](https://huggingface.co/C-Tianyu/NanoJev) | **MIT License**.
* **Key Architecture & Objective:**
  * 0.6B parallel decision model on `Qwen/Qwen3-0.6B`.
  * **Candidate Set-Attention Head (`scripts/train_toy_decisions.py`):** Encodes each candidate continuation into pooled representations, injects option cardinality embedding $\log K$, and runs multi-head self-attention across the candidate set to model competitive option dynamics.
* **Calibrated Reinforcement Learning (RLCD):**
  * Unbiased Paired Brier Policy Gradient: samples $M$ actions with replacement ($A_1 \dots A_M \sim \mathbf{p}$), computing advantage relative to a leave-one-out control baseline:
    $$\text{Advantage}_i = \text{LocalReward}_i - \text{ControlBaseline}_i$$
    $$\mathcal{L}_{\text{PG}} = - \sum_{i=1}^M \text{Advantage}_i \cdot \log p(A_i)$$
* **Actionable Adoption for Gemma 4 Cross-Encoder:**
  * Set-Attention head over candidate representations in `gemma4_cross_encoder.py:rerank()`.
  * Paired Brier Policy Gradient for post-training calibration.

---

### 1.5 sabeel111/OpenSourceJev
* **Attribution & Creators:** Sabeel K. ([@sabeel111](https://github.com/sabeel111)).
* **Repository & License:** [`https://github.com/sabeel111/OpenSourceJev`](https://github.com/sabeel111/OpenSourceJev) | **MIT License**.
* **Key Architecture & Objective:**
  * Ultra-lightweight edge System 1 runner wrapping native `llama.cpp` through Python `ctypes`.
  * **Multi-Token Candidate Scoring with KV Trimming:** Evaluates multi-token candidate phrases by stepping single tokens and using `model.trim_kv(prefix_len)` to roll back the KV cache to the prefix boundary without recomputing the context.
  * **Length-Normalized Scoring:**
    $$\text{Score}(\mathbf{w}) = \frac{\sum_{t=1}^L \log P(w_t \mid \text{prefix}, w_{<t})}{L^\alpha}, \quad \alpha \approx 0.7$$
  * **Empirical Temperature Calibration:** Fits scalar temperature $T \approx 9.47$ on BoolQ, dropping binary ECE from 0.20 to 0.09.
* **Actionable Adoption for Gemma 4 Cross-Encoder:**
  * Length-normalized scoring for verbose candidate hypotheses.
  * Explicit post-hoc validation temperature fitting ($T^*$) stored in `calibration.json`.

---

## 2. Synthesis of High-Impact Techniques for Our Project

| Technique | Source Project | Target File in Our Repo | Primary Benefit |
|---|---|---|---|
| **Abstention Augmentation (`none_augment`)** | `Mapika/decider` | `generate_sdk_synthetic_data.py` | Eliminates forced false positives when no candidate option is supported by context. |
| **Cross-Family Distractor Partitioning** | `Mapika/decider` | `generate_sdk_synthetic_data.py` | Guarantees hard negatives are drawn from distinct semantic families, avoiding false negative collisions. |
| **Token-Bucket Batching (`batches_by_tokens`)** | `Mapika/decider` | `train_cross_encoder.py`, `finetune.py` | Eliminates CUDA graph recompilations and memory fragmentation across ragged 128K context windows. |
| **Expected Value Rubric Scoring ($\mathbb{E}[\text{score}]$)** | `Bespoke-Nimble-9B` | `gemma4_cross_encoder.py` | Provides smooth, continuous rubric scores alongside discrete argmax levels. |
| **Candidate Set-Attention with $\log K$ Cardinality** | `TianyuCodings/NanoJev` | `gemma4_cross_encoder.py` | Allows candidate hypotheses to compete through self-attention, calibrating multi-option choice distributions. |
| **Post-Hoc NLL Temperature Scaling ($T^*$)** | `sabeel111/OpenSourceJev` | `finetune.py`, `gemma4_cross_encoder.py` | Reduces ECE with zero regression to classification accuracy. |

