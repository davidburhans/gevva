# Gevva v2 Architecture & Implementation Specification: Native Zero-Patch System 1 Engine for llama.cpp

> **Target Version**: Gevva v2 (`gevva-v2-e2b` and `gevva-v2-e4b`)  
> **Status**: Approved Planning & Preparation (Scheduled for training upon completion of Gevva 1.1)  
> **Compatibility Target**: Unmodified `ggml-org/llama.cpp` (`llama-server /v1/systemone` endpoint)  
> **Base Foundation Models**: `google/gemma-4-E2B-it` (~2.3B parameters) and `google/gemma-4-E4B-it` (~4.5B parameters)  
> **Primary Evaluation Instrument**: Decision Index 0.2 (44 benchmarks, 151,034 requests)  
> **Secondary Leaderboard Check**: JevBench Public (111 hard-tier calibration items)  

---

## 1. Executive Summary & Paradigm Evolution

### The Problem in Gevva 1.0 / 1.1
Gevva v1 and v1.1 established industry-leading accuracy on sub-3B decision models (#1 sub-3B on Decision Index 0.2 with 26.79 Balanced Skill, #1 on JevBench with 77.54 Composite). However, v1's architectural design relied on a **custom 3-class sequence classification head** (`Gemma4RMSNorm` + `nn.Linear(hidden_size, 3)`):
1. **Requires C++ Server Patches**: Stock `llama.cpp` (`llama-server`) only builds causal language modeling graphs for `LLM_ARCH_GEMMA4` (`src/models/gemma4.cpp`), lacking the sequence pooling and classification head loader required by v1.
2. **$K$-Pass Pairwise Inefficiency**: In multi-option tasks (e.g. POP909 with $K=129$, API-Bank with $K=53$), evaluating $K$ separate (Premise, Hypothesis) pairs consumed massive GPU time and memory, requiring custom prefix KV-caching (`predict_candidates`) to remain tractable.

### The Gevva v2 Breakthrough
**Gevva v2 transitions to a native Causal Decision Engine that runs 100% out-of-the-box in unmodified `llama.cpp` (`llama-server`) without editing or recompiling a single line of C++ code.**

```
Gevva v1.x (Sequence Classification Cross-Encoder):
  [Context + Question] x [Candidate_k] --> 3-class Head --> P(ent) - P(con) --> Softmax
  Requires: Custom Python runtime / C++ patches to llama.cpp

Gevva v2 (Native Causal Decision Engine):
  [Context + Question + All K Options [A]... [B]... [C]...] --> Native Gemma 4 LM Head
  Inference: 1 forward pass in llama-server /v1/systemone
  Debiasing: 2-variant reverse averaging built into llama.cpp
  Compatibility: 100% Zero-Patch Stock llama.cpp via `lev` Decision Protocol
```

---

## 2. Lessons Learned from Gevva 1.0 & 1.1: What v2 Leverages

Gevva v2 does not start from scratch. It directly inherits the hard-won empirical insights from Gevva 1.0 and 1.1:

### Lesson 1: Foundational NLI Anchors Are Strictly Load-Bearing (EXP-01 / F-07)
* In experiment EXP-01 (2026-09-28), pruning foundational NLI anchors in favor of pure synthetic decision data caused catastrophic silent collapse: e4b's neutral recall fell to **2.9%**, and accuracy dropped across floor (−11.7pp), medium (−13.9pp), and hard (−5.2pp) tiers ($p \approx 0$).
* **v2 Policy**: Foundational NLI data (SNLI, MNLI, ANLI R1–R3, WANLI, FEVER) is **never discarded**. In v2, every foundational NLI sample is automatically restructured into decision verification tasks, preserving deep semantic entailment reasoning while adopting the causal format.

### Lesson 2: Positional Bias & Option Ordering Must Be Countered at Both Train and Serving Time
* In causal LMs, models naturally exhibit primacy (favoring option `A`) or recency bias.
* **v2 Policy**:
  - **Serving Time**: By adopting the `lev` protocol in `llama.cpp`, `llama-server` automatically performs bidirectional 2-variant evaluation (forward $[0 \dots K-1]$ and reverse $[K-1 \dots 0]$), averaging the resulting probabilities to cancel position bias.
  - **Training Time**: Full random option permutations ($\pi \sim S_K$) applied on every epoch, combined with a symmetric Jensen-Shannon divergence consistency loss.

### Lesson 3: Train-Serving Parity (Rule 4 in AGENTS.md)
* Models degrade when benchmark interactions differ syntactically from training prompts.
* **v2 Policy**: The training formatting matches `llama-server`'s `systemone` Jinja template bit-for-bit, including turn tokens, option bracket formatting (`[A]`, `[AA]`), and question headers.

### Lesson 4: Option Scalability (The $K=255$ Contract)
* Benchmark catalogs (e.g. tool routing, API-Bank, enterprise dispatch) frequently feature 100–200+ options. A 52-letter alphabet (`A..Z, a..z`) crashes on $K > 52$.
* **v2 Policy**: Adopt the two-letter alphanumeric code sequence (`A..Z` then `AA..ZZ`). Gemma 4's 256K tokenizer provides **679 clean single-token codes**, allowing full support up to the `llama.cpp` ceiling of **255 options**.

---

## 3. The Native `llama.cpp` Protocol Contract

### Protocol: `COMMON_DECISION_TYPE_LEV`
In `llama.cpp` (`tools/server/server-decision.cpp`), the decision engine dispatches on `{arch}.decision.type`. Gevva v2 declares:
```
gemma4.decision.type = "lev"
```

### Label Mapping & Single-Token Verification
`llama-server` generates candidate labels using the following loop:
1. Codes `A` through `Z` (indices 0–25).
2. Codes `AA` through `ZZ` (indices 26–254).
3. Verifies each code is a single token in the model's vocabulary.

**Empirical Verification on Gemma 4 (`gemma-4-E2B-it`)**:
- Total codes tested: 702
- Single-token codes available: **679**
- Single-letter examples: `'A'` (Token ID `236776`), `'B'` (Token ID `236799`), `'C'` (Token ID `236780`), `'D'` (Token ID `236796`)
- Two-letter examples: `'AA'` (Token ID `8686`), `'AB'`, `'AC'`...
- Capacity: **255 options fully satisfied with zero token splitting**.

### Turn Structure & Jinja `systemone` Template
Gevva v2 embeds the following chat template into the GGUF metadata under `systemone`:

```jinja
<start_of_turn>user
{% for image in images %}{{ image }}{% endfor %}
{% if images %}The image shows the input visual scene.

{% endif %}
State:
{{ state if state is string else state | tojson(indent=2) }}

Question: {{ instructions }}
{% if type == 'score' %}Rate along the ordered levels below (lowest first).
{% endif %}
Options:
{% for o in options %}
[{{ o.label }}] {{ o.key }}{% if o.description %}: {{ o.description }}{% endif %}
{% endfor %}

Answer with the option code only.<end_of_turn>
<start_of_turn>model
```

At inference time, `llama-server` evaluates the prompt and reads the output logits at the single decision position following `<start_of_turn>model\n`:
$$P(\text{option}_i) = \text{softmax}\left(\frac{\text{logit}(\text{label}_i)}{T}\right)$$

---

## 4. Master Data Mixture & Compilation for Gevva v2

Gevva v2 converts all existing high-value assets compiled during v1.0, Phase 2–5, and Round 4c into the unified `lev` format:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Gevva v2 Master Curriculum Mixture                   │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Enterprise Typed Decisions (25% of mixture)                         │
│    • n4ze3m/typed-decisions-synth (25,859 questions, 149 workflows)    │
│    • Soft teacher labels (DeepSeek V4.1 Flash, 3-pass self-consistent) │
│                                                                        │
│ 2. Foundational NLI Verification Anchors (25% of mixture)              │
│    • SNLI, MNLI, ANLI R1–R3, WANLI, FEVER                              │
│    • Restructured into binary (noul) & multi-hypothesis choice pairs   │
│    • Load-bearing anchor: guarantees deep semantic reasoning           │
│                                                                        │
│ 3. SDK-Parity Synthetic Data (25% of mixture)                          │
│    • Rounds 1, 2, 3, 4, 4b, 4c from generate_sdk_synthetic_data.py     │
│    • Tool routing (API-Bank, ToolBench, Gorrila)                       │
│    • RAG hallucination guardrails (RAGTruth, factual attribution)      │
│    • Rubric grading & code diff triage                                 │
│                                                                        │
│ 4. Long-Context & Document Grounding (15% of mixture)                  │
│    • DocNLI multi-page contracts and reports (up to 32K context)       │
│    • Synthetic needle-in-a-haystack verification (up to 128K context)  │
│                                                                        │
│ 5. Multimodal & Multilingual Foundations (10% of mixture)              │
│    • DocVQA, ChartQA, UI screen actions (SigLIP visual tokens)         │
│    • XNLI (15 languages) and Flores-200 multilingual decision routing  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Methodology & Objective Functions

### 1. Active-Set Target-Restricted Cross-Entropy Loss
Unlike generative language modeling where loss is computed over the entire 256,000-token vocabulary, Gevva v2 restricts the classification objective to the **active option codes** $\mathcal{C}_{\text{active}} = \{c_0, c_1, \dots, c_{K-1}\}$:

$$\mathcal{L}_{\text{active}}(\mathbf{z}, y) = -\log \frac{\exp(z_{c_y} / \tau)}{\sum_{j=0}^{K-1} \exp(z_{c_j} / \tau)}$$

This aligns the gradient update exactly with the inference-time softmax normalization in `llama-server`.

### 2. Epoch-Level Permutation Augmentation
For every training example with $K$ options:
1. Sample a random permutation $\pi \in \mathcal{S}_K$.
2. Permute the option order: $\text{options}' = [\text{options}_{\pi(0)}, \dots, \text{options}_{\pi(K-1)}]$.
3. Assign option codes in linear order: $c_i = \text{codes}[i]$.
4. Set gold target label: $y' = \pi^{-1}(y_{\text{gold}})$.

This completely destroys any spurious correlation between option position and correctness.

### 3. Dual-Permutation Symmetric Regularization
For 20% of batches, evaluate both the forward permutation $\pi$ and its reverse $\pi^{\text{rev}}$:
$$\mathcal{L}_{\text{symm}} = D_{\text{JS}}\left(P_{\pi}(\text{options}) \;\parallel\; P_{\pi^{\text{rev}}}(\text{options})\right)$$
This explicitly forces the attention heads to represent options symmetrically regardless of presentation order.

### 4. Soft Label Distillation (Brier / KL Divergence)
Where teacher soft probabilities $\mathbf{q}$ exist (`typed-decisions-synth` DeepSeek V4.1 Flash labels):
$$\mathcal{L}_{\text{distill}} = \text{KL}(\mathbf{q} \parallel P_{\text{active}}) + \lambda_{\text{brier}} \sum_{j=0}^{K-1} (P_j - q_j)^2$$
Ensures top-tier calibration without post-hoc temperature scaling distortion.

### 5. Mandatory In-Loop Quantization-Aware Training (QAT) by Default

In Gevva v1.x, a silent regression occurred when an audit flipped `--qat` to opt-in (`default=False`), causing training chains to train in unquantized `bfloat16` without an assertion failure.

**Gevva v2 Architectural Rule: QAT is Non-Negotiable and Active by Default.**

1. **Target Quantization Formats**:
   - Primary: `Q4_K_M` (k-quants for `llama.cpp`) and `w4a16` (INT4 Group-32 symmetric).
   - In-loop fake-quantization simulator: MultiQuant Straight-Through Estimator (STE) parameterized over linear layers in `self_attn` and `mlp`.
2. **Hard Fail-Fast Invariant**:
   In `train_gevva_v2.py`:
   ```python
   # Hard fail-fast: training must never execute in float precision without explicit override
   if not getattr(args, "qat", True):
       if not getattr(args, "allow_unquantized_experimental_run", False):
           raise RuntimeError(
               "FATAL: Gevva v2 requires Quantization-Aware Training (QAT) by contract. "
               "To train in float, you must explicitly pass --allow-unquantized-experimental-run."
           )
   ```
3. **Artifact Provenance & Deployment Gate**:
   - Checkpoint artifact `qat_config.json` must record `"qat_applied": true` and `"target_quant": "q4_k_m"` / `"w4a16"`.
   - The GGUF exporter (`scripts/export_gevva_v2_gguf.py`) must verify `qat_applied: true` before packing.
4. **Preserved Components**:
   - `lm_head` (causal decision projection) is kept in 16-bit to preserve uncorrupted logit calibration on the decision token.
   - Rotary embeddings (RoPE) and layer norms remain FP32/BF16.

---

## 6. GGUF Export & llama.cpp Deployment

### Standalone Exporter (`scripts/export_gevva_v2_gguf.py`)
The export pipeline packages the fine-tuned Gemma 4 causal weights directly using `gguf-py`:

```python
import gguf

writer = gguf.GGUFWriter(output_path, "gemma4")
writer.add_name("Gevva v2 e2b System 1 Decision Engine")
writer.add_architecture("gemma4")
writer.add_decision_type("lev")  # Triggers COMMON_DECISION_TYPE_LEV in llama.cpp

# Calibrated temperature scaling
writer.add_decision_temperature("choice.small", 0.85)  # K <= 8
writer.add_decision_temperature("choice.mid",   1.10)  # K <= 26
writer.add_decision_temperature("choice.large", 1.35)  # K <= 255
writer.add_decision_temperature("noul",         1.55)
writer.add_decision_temperature("score",        0.90)

# Embedded systemone template
writer.add_chat_template([{"name": "systemone", "template": SYSTEMONE_TEMPLATE}])
```

### Zero-Patch Serving Command
Once exported, the model runs immediately with official `llama.cpp` binaries:
```bash
# Pure text decision serving
llama-server -m gevva-v2-e2b-Q4_K_M.gguf --port 8080

# Multimodal decision serving (with SigLIP vision companion)
llama-server -m gevva-v2-e2b-Q4_K_M.gguf --mmproj mmproj-gevva-v2-f16.gguf --port 8080
```

---

## 7. Execution Checklist (Ready for Completion of Gevva 1.1)

- [x] **Architecture Verification**: Confirmed `gemma4.decision.type = "lev"` maps to `COMMON_DECISION_TYPE_LEV` in `llama.cpp`.
- [x] **Vocabulary Coverage**: Probed Gemma 4 tokenizer — 679 clean single-token codes in `A..Z` + `AA..ZZ` range (exceeds 255-option limit).
- [x] **Debiasing Strategy**: Established dual-level order invariance (serving 2-pass reverse averaging + training permutation loss).
- [ ] **Mandatory QAT Engine**: `train_gevva_v2.py` with in-loop STE fake-quantization enabled by default (`--qat` default `True`) and hard assertion gates.
- [ ] **Data Pipeline Script**: `scripts/compile_gevva_v2_mixture.py` (converts v1/1.1 datasets into `lev` format).
- [ ] **Training Execution**: Run `train_gevva_v2.py` on `gevva-v2-e2b` and `gevva-v2-e4b`.
- [ ] **GGUF Packaging Tool**: `scripts/export_gevva_v2_gguf.py` (native GGUF metadata writer).
- [ ] **Verification Harness**: `scripts/test_gevva_v2_systemone.py` (end-to-end HTTP test against `llama-server /v1/systemone`).
