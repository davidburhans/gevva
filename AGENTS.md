# Agent Instructions & Project Context: Gevva Multimodal 128K System 1 Decision Engine

> **Rule for All Antigravity Agent Sessions**: This repository implements **Gevva**, a large-context (128K), multilingual (100+ languages), vision-enabled System 1 decision engine and NLI cross-encoder based on Google's Gemma 4 models. **Primary evaluation instrument: Decision Index 0.2** (44 benchmarks, 151,034 requests, ±0.25pp resolution) — Gevva e2b scores 26.79 Balanced Skill (highest sub-3B open entrant) and Gevva e4b scores 29.88. **Secondary/leaderboard check: JevBench Public** (Gevva e2b: 77.54 Composite, #1 at publication; note the hard tier is n=111 with a ±9.3pp CI — it cannot resolve development-scale deltas and must never be used as a training-round gate). Follow the guidelines and architectural decisions documented below when inspecting code, proposing modifications, generating data, or training models.

---

## 1. Project Mission & Overview

This project builds **Gevva**, a family of **large-context (128K)**, **multilingual (100+ languages)**, **vision-enabled System 1 decision engines** based on Google's lightweight multimodal foundation models:
- **`Gevva e2b`** (built on `google/gemma-4-E2B-it`, ~2.3B effective parameters; Decision Index 26.79 Balanced Skill, JevBench 77.54)
- **`Gevva e4b`** (built on `google/gemma-4-E4B-it`, ~4.5B effective parameters, deep reasoning model)

### What We Are Building
A high-throughput, non-autoregressive **System 1 Decision Engine** (inspired by Daniel Kahneman's System 1 cognitive model, [TypeSafe AI Jev](http://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Convai Laya](https://huggingface.co/convaiinnovations/laya)). Rather than generating tokens autoregressively, the model evaluates input pairs in a single forward pass (~14.3–16.5 ms) and outputs calibrated probability distributions over three standard states:

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
├── pyproject.toml                 # Package configuration (gevva 1.0.0, CLI entrypoint)
├── gevva/                         # Gevva Python SDK package (from gevva import GevvaCrossEncoder, load)
│   ├── __init__.py                # Top-level exports and load() helper
│   └── cli.py                     # CLI entrypoint (gevva predict, rerank, grade, finetune, eval)
├── gemma4_cross_encoder.py        # Core model, 100% Jev/OpenJEV API, W4A16 loader & inference engine
├── finetune.py                    # Turnkey custom data fine-tuning engine (auto-detects formats & columns)
├── export_w4a16.py                # Production INT4 Group-32 exporter (compressed-tensors layout)
├── eval_downstream_decisions.py   # Latency, tool routing, hallucination & multilingual eval
├── eval_openjev_benchmarks.py     # Direct head-to-head capability benchmark suite vs Jev/OpenJEV/Laya
├── data_pipeline.py               # Multi-source dataset compiler (NLI, vision, multilingual)
├── train_cross_encoder.py         # PyTorch training & calibration loop with in-loop QAT
├── generate_sdk_synthetic_data.py # SDK-parity synthetic data engine (teacher + 4-judge committee)
├── validator_committee.py         # Multi-judge consensus, disagreement queue, checkpoint/resume
├── validation_metrics_db.py       # SQLite judge metrics DB (idempotent verdicts, judge_performance view)
├── llm_client.py                  # OpenAI-compatible llama-server client (GBNF-constrained JSON)
├── nli_labels.py                  # Shared label enum (0=contradiction, 1=entailment, 2=neutral)
├── tests/                         # Test suite: .venv/bin/python -m unittest discover tests (95 tests)
├── docs/                          # Guides, model cards, and methodology protocols
│   ├── CUSTOM_FINETUNING_GUIDE.md # User-facing custom fine-tuning guide & recipes
│   ├── EVALUATION_PROTOCOL.md     # Pre-registered evaluation protocol & JevBench gate outcomes
│   ├── HUGGINGFACE_MODEL_CARD.md  # Official HuggingFace model card for Gevva e2b
│   ├── IMPROVEMENT_OPPORTUNITIES.md # Architectural & curriculum optimizations (Prefix KV, DST)
│   ├── JEVBENCH_REMEDIATION_PLAN.md # Remediation plan (completed: #1 JevBench 77.54)
│   └── jevbench-error-audit.md    # Error audit documentation
├── ckpt/
│   ├── gevva-e2b/                 # Champion e2b checkpoint (Gevva 1.1 distill -> gevva-e2b-distill/best)
│   ├── gevva-e4b/                 # Champion e4b checkpoint (Gevva 1.1 distill -> gevva-e4b-distill/best)
│   └── gemma-4-e2b-nli-w4a16/     # Production standalone W4A16 model (7.04 GB, 14.3ms latency)
├── results/                       # Benchmark outputs and comparative JSON logs
└── research/                      # Reference implementations, adapters, and deep-dive reports
```

---

## 5. Training Curriculum & Data Mixtures

1. **Stage 1: Core Text, Visual & Decision NLI (Context up to 4K)**
   - Text NLI: SNLI, MNLI, ANLI (R1, R2, R3), WANLI, FEVER.
   - Multimodal NLI: SNLI-VE, converted VQA v2, GQA, DocVQA, spatial coordinate claims.
   - Multilingual: XNLI (15 languages), Flores-200.
   - SDK-Aligned Decisions: 8,515 consensus-filtered pairs (tool routing, search rerank, cloze, grading, RAG).
2. **Stage 2: Mid-Context, Enterprise Decisions & Document Grounding (Context up to 32K)**
   - System 1 Decisions: `n4ze3m/typed-decisions-synth` (25,859 enterprise questions, 149 workflows, MIT license, Muhammed Nazeem 2026) with DeepSeek V4.1 Flash calibrated soft labels.
   - DocNLI multi-page document pairs (average 800–3,500 tokens).
   - Medium synthetic haystack verification (retrieved passage validation up to 32K).
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
- **Primary — Decision Index** (development gate; per-benchmark resolution ±0.25–1.7pp):
  ```bash
  # Quick paired gate (7,015 items: SNLI/MNLI floor, FEVER/QNLI medium, ANLI R1-R3 + ContractNLI hard)
  uv run python scripts/run_exp01.py   # or evaluate a ckpt directly on data/exp01_gate_eval.jsonl
  # Full 44-benchmark suite (~151k requests)
  uv run python scripts/run_decision_index_eval.py --model-path <ckpt>
  ```
- Run downstream System 1 decisions evaluation:
  ```bash
  uv run python eval_downstream_decisions.py --model-path ckpt/gevva-e2b
  ```
- Run direct capability comparison against Jev / OpenJEV / Laya:
  ```bash
  uv run python eval_openjev_benchmarks.py --limit 100
  ```
- **Secondary — JevBench Public** (leaderboard/positioning check only; hard tier n=111, ±9.3pp CI — never a training-round gate):
  ```bash
  uv run python scripts/eval_jevbench_public.py --model-path ckpt/gevva-e2b --temperature 1.6
  ```

### Guiding Principles for Contributions
1. **Never generate text when classification suffices**: Keep the model strictly non-autoregressive.
2. **Calibration is paramount**: Use strictly proper scoring rules or soft BCE with Brier score monitoring to ensure confidence matches empirical accuracy.
3. **Preserve modularity**: Keep data curation adapters, model architecture definitions, and training runners decoupled.
4. **Enforce Train-Serving Parity**: Every method exposed in the public SDK (`gemma4_cross_encoder.py` — `rerank`, `grade`, tool routing, RAG hallucination checks) must have an explicit, calibrated data generation slice in `generate_sdk_synthetic_data.py`. Never evaluate the model on interaction patterns that have zero representation in the training curriculum.
5. **Anchor-asserted mixtures** (EXP-01 + F-07, 2026-09-28): foundational-NLI anchors are load-bearing at every difficulty tier, and a training round with zero adversarial-NLI rows silently collapsed e4b's ANLI neutral recall to 2.9%. Every mixture compile must assert anchor presence (see `scripts/compile_phase5_e4b.py`) and record its manifest.
6. **Measure with the right instrument**: Decision Index is the primary development gate (151k requests, real diagnostic power). JevBench is a leaderboard check with a ±9.3pp hard-tier CI at n=111 — cite it for positioning, never for training decisions.


