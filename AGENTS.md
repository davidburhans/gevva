# Agent Instructions & Project Context: Multimodal 128K NLI Cross-Encoder

> **Rule for All Antigravity Agent Sessions**: This repository implements a large-context, multilingual, vision-enabled NLI cross-encoder / System 1 decision engine based on Google's Gemma 4 models. Follow the guidelines and architectural decisions documented below when inspecting code, proposing modifications, generating data, or training models.

---

## 1. Project Mission & Overview

This project builds a **large-context (128K)**, **multilingual (100+ languages)**, **vision-enabled NLI cross-encoder** based on Google's lightweight multimodal foundation models:
- **`google/gemma-4-E2B`** (~2.3B effective parameters, edge-optimized)
- **`google/gemma-4-E4B`** (~4.5B effective parameters, high accuracy)

### What We Are Building
A high-throughput, non-autoregressive **System 1 Decision Engine** (inspired by [TypeSafe AI Jev](http://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Convai Laya](https://huggingface.co/convaiinnovations/laya)). Rather than generating tokens autoregressively, the model evaluates input pairs in a single forward pass (~25–40 ms) and outputs calibrated probability distributions over three standard states:

$$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

### Key Applications
1. **RAG Hallucination Detection & Attribution**: Verifying extracted answers against full retrieved documents (up to 128K tokens).
2. **Visual Entailment & Multimodal Guardrails**: Checking visual scenes, UI states, document PDFs, and charts against factual claims.
3. **Zero-Shot Intent & Tool Routing**: Instant routing without prompt-engineering or text-generation latency.
4. **Automated Response Evaluation & Grading**: Comparing outputs against ground-truth rubrics.

---

## 2. Hardware Environment & Constraints

Always respect these hardware specifications when planning batch sizes, sequence lengths, and precision:

- **GPU**: 1x NVIDIA GeForce RTX 5090 (32,607 MiB VRAM, Blackwell architecture)
- **CPU**: AMD Ryzen 9 9950X3D (16 physical cores, 32 threads)
- **Host Memory**: 128 GB RAM (~60+ GB available)
- **Storage**: Fast NVMe SSD with 600+ GB available space
- **Python Environment**:
  - Package Manager: `uv` (`/home/dave/.local/bin/uv`)
  - Target Python: Python 3.12 (`/home/dave/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`)

---

## 3. Core Architectural Decisions

### Label Convention
We strictly adopt the standard `dleemiller/ModernCE-large-nli` and `AlexWortega/openjev` label indexing:
```python
ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {"contradiction": 0, "entailment": 1, "neutral": 2}
```
*(Note: Stanford SNLI / NYU MNLI natively use 0=entailment, 1=neutral, 2=contradiction; always map with `NATIVE2OURS = {0: 1, 1: 2, 2: 0}` when ingesting standard datasets).*

### Input Framing & Multimodal Formatting
```
Premise: <|vision_start|><|image_pad|>*N<|vision_end|> {text_premise}
Hypothesis: {hypothesis}
```
- For **text-only** inputs, the `<|vision_start|>...<|vision_end|>` block is omitted.
- For **multimodal** inputs, the image placeholder is embedded inside the premise. The hypothesis remains a declarative claim evaluating the visual/textual evidence.

### Pooling & Classification Head
- **Padding**: Right-padding (`tokenizer.padding_side = "right"`).
- **Pooling**: Last non-pad token pooling over the backbone's `last_hidden_state`:
  ```python
  last_token_idx = attention_mask.sum(dim=1) - 1
  pooled = last_hidden_state[torch.arange(batch_size), last_token_idx]
  logits = score_head(pooled)  # (batch_size, 3)
  ```
- **Vision Tower Optimization**:
  - The `gemma4_vision` tower is frozen (`torch.no_grad()`) during sequence classification training.
  - Patch projection is executed in FP32 to avoid cuDNN bf16 latency stalls.

---

## 4. Repository Structure

```
/home/dave/workspaces/nli-cross-encoder/
├── README.md                      # Primary project overview, quickstart & benchmarks
├── RESOURCES.md                   # Initial reference links & foundation models
├── AGENTS.md                      # This file (session memory, rules, and guidelines)
├── PROGRESS.md                    # Live state tracking, benchmark logs, and runbook
├── gemma4_cross_encoder.py        # Core model, 100% Jev/OpenJEV API, W4A16 loader & inference engine
├── finetune.py                    # Turnkey custom data fine-tuning engine (auto-detects formats & columns)
├── export_w4a16.py                # Production INT4 Group-32 exporter (compressed-tensors layout)
├── eval_downstream_decisions.py   # Latency, tool routing, hallucination & multilingual eval
├── eval_openjev_benchmarks.py     # Direct head-to-head capability benchmark suite vs Jev/OpenJEV/Laya
├── data_pipeline.py               # Multi-source dataset compiler (NLI, vision, multilingual)
├── train_cross_encoder.py         # PyTorch training & calibration loop with in-loop QAT
├── docs/
│   └── CUSTOM_FINETUNING_GUIDE.md # User-facing custom fine-tuning guide & recipes
├── ckpt/
│   ├── gemma-4-e2b-nli-stage1/    # Baseline BF16 LoRA adapter
│   ├── gemma-4-e2b-nli-qat-stage1/# In-loop 4-bit QAT LoRA adapter
│   └── gemma-4-e2b-nli-w4a16/     # Production standalone W4A16 model (7.04 GB, 14.3ms latency)
├── results/                       # Benchmark outputs and comparative JSON logs
└── research/
    ├── openjev/                   # Downloaded reference scripts from AlexWortega/openjev
    ├── laya/                      # Reference implementations from Convai Laya
    ├── adapters/                  # Data adapters & collators
    └── reports/                   # Technical deep dives (Reports 01 through 05)
```

---

## 5. Training Curriculum & Data Mixtures

1. **Stage 1: Core Text & Visual NLI (Context up to 4K)**
   - Text NLI: SNLI, MNLI, ANLI (R1, R2, R3), WANLI, FEVER.
   - Multimodal NLI: SNLI-VE, converted VQA v2, GQA, DocVQA, spatial coordinate claims.
   - Multilingual: XNLI (15 languages), Flores-200.
2. **Stage 2: Mid-Context & Document Grounding (Context up to 32K)**
   - DocNLI multi-page document pairs.
   - Medium synthetic haystack verification (retrieved passage validation).
3. **Stage 3: Full 128K Needle-in-a-Haystack & Enterprise Verification (Context up to 128K)**
   - Long synthetic haystack (evidence injected vs dropped/neutral vs corrupted/contradiction).
   - Long-context RAG citation verification and repository-level code diff claims.

---

## 6. Custom Data Fine-Tuning Engine

To adapt the model to downstream domain tasks (e.g. enterprise RAG hallucination guardrails, proprietary tool routing, or domain-specific search reranking), use `finetune.py`:

```bash
# 1-line fine-tuning with auto-detection of column names & label formats:
uv run python finetune.py --data my_data.jsonl --out-dir ./ckpt/my_domain_model

# Continual adaptation starting from our pre-trained NLI checkpoint:
uv run python finetune.py \
    --data my_data.csv \
    --adapter ./ckpt/gemma-4-e2b-nli-stage1/best \
    --out-dir ./ckpt/my_domain_finetuned \
    --epochs 3
```

Key features:
- **Format Auto-Detection**: Accepts `.jsonl`, `.csv`, `.tsv`, or `.json`.
- **Column Auto-Mapping**: Automatically detects context (`context`, `premise`, `document`), claim (`hypothesis`, `claim`, `query`), and target (`label`, `verdict`, `gold`).
- **Flexible Labels**: Normalizes strings (`"supports"`, `"refutes"`, `"unverifiable"`) or integers (`0`, `1`, `2`).
- **Stratified Auto-Split**: Splits train/val automatically if no separate validation file is provided.

---

## 7. Runbook for Future Agent Sessions

### Environment Activation
Always run Python commands via `uv` or the configured virtual environment:
```bash
uv run python <script.py>
# Or inside a project virtualenv
source .venv/bin/activate
```

### Running Benchmarks
- Run the direct capability comparison against Jev / OpenJEV / Laya:
  ```bash
  uv run python eval_openjev_benchmarks.py --limit 100
  ```
- Run downstream System 1 decisions evaluation:
  ```bash
  uv run python eval_downstream_decisions.py --model-path ./ckpt/gemma-4-e2b-nli-w4a16
  ```

### Guiding Principles for Contributions
1. **Never generate text when classification suffices**: Keep the model strictly non-autoregressive.
2. **Calibration is paramount**: Use strictly proper scoring rules or soft BCE with Brier score monitoring to ensure confidence matches empirical accuracy.
3. **Preserve modularity**: Keep data curation adapters, model architecture definitions, and training runners decoupled.
4. **Enforce Train-Serving Parity**: Every method exposed in the public SDK (`gemma4_cross_encoder.py` — `rerank`, `grade`, tool routing, RAG hallucination checks) must have an explicit, calibrated data generation slice in `generate_sdk_synthetic_data.py`. Never evaluate the model on interaction patterns that have zero representation in the training curriculum.


