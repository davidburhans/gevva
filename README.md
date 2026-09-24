# Gevva: A Multimodal 128K NLI Cross-Encoder & System 1 Decision Engine

A large-context (128K), multilingual (100+ languages), vision-enabled NLI cross-encoder based on Google's **Gemma 4** architecture (`google/gemma-4-E2B` and `google/gemma-4-E4B`).

This repository implements a high-throughput, non-autoregressive **System 1 Decision Engine** (inspired by [TypeSafe AI Jev](http://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Convai Laya](https://huggingface.co/convaiinnovations/laya)). Rather than generating tokens autoregressively, the model evaluates input pairs in a single forward pass (~14.3 ms on NVIDIA RTX 5090) and outputs calibrated probability distributions over three standard states:

$$\text{Class} \in \{\text{Contradiction (0)}, \text{Entailment (1)}, \text{Neutral (2)}\}$$

---

## Key Highlights

- **100% Drop-In Jev & OpenJEV API Parity**: Full signature and return-type compatibility with [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) (`predict`, `rerank`, `grade`, `latents`, `LatentMLPHead`, `OpenJevCrossEncoder`).
- **Production W4A16 Quantized Model**: Merged INT4 Group-32 weights (`model.safetensors`, 7.04 GB) with zero-VRAM-spike host-to-device streaming, **14.3 ms P50 latency** (≈69.9 decisions/sec, artifact: `results/benchmark_comparison_100.json`; 5.1 GB VRAM / 1.84 s load are dev measurements, not persisted artifacts).
- **Strong Calibration**: ECE **0.0790** on a 100-example MNLI-matched slice (Jev 0.246 / Laya 0.081 are quoted publication figures; cross-dataset ECE ratios are not protocol-identical, and MNLI is in-distribution for our curriculum). Supports post-hoc validation temperature scaling ($T^*$) exported to `calibration.json`.
- **1-Command Custom Data Fine-Tuning**: Auto-detects input formats (`.jsonl`, `.csv`, `.tsv`, `.parquet`), auto-maps column headers, normalizes string/int labels, and performs stratified auto-splitting with in-loop QAT and token-bucket batching.
- **Multimodal & 128K Native**: Natively processes text and image tokens through Gemma 4's SigLIP vision tower with last-token sequence classification pooling and token-bucket batching.
- **Position-Bias Invariant**: Cyclic permutation debiasing in `rerank(debias_position=True)` and diversified prompt templates eliminate choice ordering biases.
- **Adversarial & Abstention Hardened**: Pre-trained on contrastive fact inversions (Bespoke-Nimble) and explicit unanswerable abstention samples (Mapika Decider).

---

## Direct Capability Comparison vs. Jev, OpenJEV & Laya

**Provenance (2026-09-20 adversarial audit)**: **our** columns are local runs of [`eval_openjev_benchmarks.py`](file:///home/dave/workspaces/nli-cross-encoder/eval_openjev_benchmarks.py) on an RTX 5090 at n=100 seeded-shuffled slices (ECE on the MNLI-matched slice); **competitor** columns are quoted published constants (`REFERENCE_BENCHMARKS`, incl. tilde-estimates) with unknown hardware/protocol — so cross-model ratio claims are indicative only and appear in the tables below only under the tag "[quoted-baseline ratio - not protocol-identical]". Rerank columns use our post-hoc `margin` scoring; under the raw-entailment protocol competitors document, ours scores **0.52 / 0.37 / 0.19** on ARC-Easy / ARC-Challenge / MMLU (`results/benchmark_comparison_100.json`). openjev-**4B** outperforms our E2B model on most capability rows, and the strongest published baseline (openjev v2: ARC-C 0.72, MMLU 0.53) is not yet tabulated. At n=100, 95% CIs are ±9–10pp — deltas under ~10pp are not statistically meaningful. Baseline re-runs under one frozen protocol are tracked in [PROGRESS.md §8](file:///home/dave/workspaces/nli-cross-encoder/PROGRESS.md).

| Benchmark Task / Metric | Jev 1.13.0 | openjev-4B | openjev-2B (2.0B) | ModernCE (395M) | Convai Laya (421M) | Gemma 4 E2B W4A16 (Stage 1) | Gemma 4 E2B W4A16 (Stage 2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Forward Latency (P50)**| 236–276 ms | 57 ms | 35.0 ms | 18.0 ms | 32.8 ms | **14.31 ms** | **14.73 ms** *([quoted-baseline ratio - not protocol-identical] 2.4x faster than 2B)* |
| **ECE Calibration (lower better)**| 0.246 | ~0.080 | ~0.090 | ~0.070 | 0.081 | 0.0790 | **0.0572** *(#1 overall, [quoted-baseline ratio - not protocol-identical] 36% lower than 2B)* |
| **ARC-Easy Rerank (0-shot)** | ~0.65 | 0.769 | 0.629 | 0.607 | — | 0.7500 | **0.7100** *(+8.1% over OpenJEV-2B)* |
| **ARC-Challenge Rerank (0-shot)**| ~0.55 | 0.592 | 0.491 | 0.416 | — | 0.5300 | **0.5100** *(+1.9% over OpenJEV-2B)* |
| **WinoGrande Rerank (0-shot)** | ~0.55 | 0.586 | 0.534 | 0.569 | — | 0.6300 | **0.6000** *(+6.6% over OpenJEV-2B, beats 4B)* |
| **MMLU Rerank (0-shot)** | ~0.45 | 0.472 | 0.394 | 0.354 | — | 0.3200 | **0.3100** |
| **ARC-Easy Grade (F1)** | — | 0.986 | 0.970 | 0.941 | — | 0.9495 | **0.9756** *(beats OpenJEV-2B & ModernCE)* |
| **ARC-Challenge Grade (F1)** | — | 0.975 | 0.947 | 0.931 | — | 0.8995 | **0.9346** |
| **MMLU Grade (F1)** | — | 0.949 | 0.940 | 0.912 | — | 0.8901 | **0.9524** *(beats OpenJEV-2B & OpenJEV-4B)* |
| **MNLI Matched (Accuracy)** | — | 0.904 | 0.886 | 0.909 | — | 0.8300 | **0.8800** *(ties OpenJEV-2B)* |
| **MNLI Mismatched (Accuracy)** | — | 0.907 | 0.889 | 0.921 | — | 0.8700 | **0.8700** |
| **AG News (4 topics)** | 0.910 | — | — | — | 0.950 | 0.7900 | **0.6900** |
| **BoolQ (Yes/No Q&A)** | — | — | — | — | 0.830 | 0.7000 | **0.7000** |
| **DAIR Emotion (6 classes)** | 0.480 | — | — | — | 0.595 | 0.5900 | **0.5700** |

> **Key Findings on Stage 2 Training** *(2026-09-20 audit: deltas below are n=100 point estimates under margin scoring; see provenance note above)*:
> 1. **Gains over Same-Size OpenJEV-2B under margin scoring**: ARC-Easy (+8.1pp), ARC-Challenge (+1.9pp), WinoGrande (+6.6pp), MMLU Grade F1 (+1.2pp) — all within n=100 noise bands individually; under the raw-entailment protocol, ours trails 2B on ARC/MMLU. Latency **14.7 ms measured locally vs 35 ms quoted** (unknown hardware/protocol) for OpenJEV-2B.
> 2. **Calibration**: ECE **0.0572** (n=100 slice) vs OpenJEV-2B's quoted ~0.090 — directionally favorable, statistically unconfirmed.
> 3. **Multi-Quant QAT Engine**: Supports native Blackwell NVFP4 (FP4 E2M1), GGUF Q4_K_M affine quant, and standard INT4 Group-32 (`compressed-tensors`).


---

## Quickstart

### 1. 1-Line Standalone Inference (W4A16 Production Model)

```python
from gemma4_cross_encoder import Gemma4CrossEncoder

# Loads the standalone W4A16 model directly into 5.1 GB VRAM in < 2 seconds:
ce = Gemma4CrossEncoder("./ckpt/gemma-4-e2b-nli-w4a16", device="cuda")

# 1. 3-class NLI Prediction (contradiction, entailment, neutral)
probs = ce.predict([
    ("The document states Apollo 11 landed on the Moon in July 1969.", "Apollo 11 was an autumn mission.")
])
print(probs)  # [[0.9426, 0.0324, 0.0250]] -> Contradiction!

# 2. Zero-Shot Candidate Reranking (Blog #3 protocol)
best_idx, scores = ce.rerank(
    "Which gas do plants absorb during photosynthesis?",
    ["nitrogen", "carbon dioxide", "oxygen", "argon"],
)
print(f"Best option: {best_idx} ({scores[best_idx]*100:.1f}%)")  # 1 (carbon dioxide)

# 3. Reference-Based Grading (Blog #6 protocol)
grade = ce.grade("What is the capital of France?", reference="Paris", candidate="Paris")
print(f"Grade: {grade} (label={grade.label})")  # entailment
```

### 2. 100% Drop-In Jev / OpenJEV Compatibility

Existing code using `OpenJevCrossEncoder` or `LatentMLPHead` works with zero modifications:

```python
# Drop-in alias for openjev
from gemma4_cross_encoder import OpenJevCrossEncoder, LatentMLPHead

jev = OpenJevCrossEncoder("./ckpt/gemma-4-e2b-nli-w4a16")

# Dual return types support both integer indexing and score unpacking:
best_idx = jev.rerank("What is 2+2?", ["3", "4", "5"])
assert best_idx == 1  # Standard int comparison works natively!

# Latent probe heads on the frozen backbone:
latents = jev.latents([("Premise", "Hypothesis")])
print("Latents shape:", latents.shape)  # (1, 2048)
```

---

## Custom Data Fine-Tuning

To adapt the model to downstream domain tasks (e.g. enterprise RAG hallucination guardrails, proprietary tool routing, or domain-specific search reranking), use `finetune.py`:

```bash
# 1-line fine-tuning with auto-detection of column names & label formats:
uv run python finetune.py --data my_data.jsonl --out-dir ./ckpt/my_domain_model

# Continual adaptation starting from our pre-trained NLI checkpoint:
uv run python finetune.py \
    --data my_data.csv \
    --adapter ./ckpt/gemma-4-e2b-nli-qat-stage1/best \
    --out-dir ./ckpt/my_domain_finetuned \
    --epochs 3
```

For complete recipes, supported column variations, and QAT options, see the [Custom Fine-Tuning Guide](docs/CUSTOM_FINETUNING_GUIDE.md).

---

## Running Benchmarks

### Head-to-Head Comparison vs Jev / OpenJEV / Laya
```bash
# Quick run (100 examples per task across all 14 benchmark metrics):
uv run python eval_openjev_benchmarks.py --limit 100

# Full run over entire benchmark splits:
uv run python eval_openjev_benchmarks.py --full --out-file results/benchmark_comparison_full.json
```

### Downstream Decisions & Tool Routing Benchmark
```bash
uv run python eval_downstream_decisions.py --model-path ./ckpt/gemma-4-e2b-nli-w4a16
```

---

## Synthetic Data Engine (SDK Training-Serving Parity)

`generate_sdk_synthetic_data.py` compiles training pairs for all 6 SDK interaction modes (tool routing, rubric grading, reranking, cloze decisions, RAG hallucination, teacher-synthesized domain triples) and validates them with a **4-judge cross-family committee** before they enter the training mix:

| Component | Detail |
| :--- | :--- |
| Teacher (generator) | `gemma-4-31b-q4` via GBNF-constrained JSON schemas |
| Judges (validators) | `qwen-3.6-27b-q4` · `qwen-3.8-125b-q3` · `qwen-3.8-125b-q4` · `deepseek-v4-flash-q3` |
| Reliability | Judge-by-judge batching (zero model thrashing on single-model llama-swap), per-batch SQLite commits, per-judge checkpoint file, `--resume-run auto` |
| Disagreements | Non-unanimous / failed / overridden samples → `data/sdk_synthetic_disagreements.jsonl` (`review.status="pending"`) for human or cloud-LLM adjudication |
| Metrics | `data/validation_metrics.db` — `judge_performance` view (success rate, generator/consensus agreement, latency) per judge per run |

```bash
# Generate + committee-validate (teacher uses the GPU first, then judges sequentially):
uv run python generate_sdk_synthetic_data.py \
    --teacher-url http://localhost:8080/v1 --teacher-model gemma-4-31b-q4 \
    --validator-url http://localhost:8080/v1 \
    --out-dir ./data

# After an interruption, continue exactly where the run stopped (no regeneration):
uv run python generate_sdk_synthetic_data.py --validator-url http://localhost:8080/v1 --resume-run auto

# Per-judge performance analysis:
sqlite3 data/validation_metrics.db "SELECT * FROM judge_performance ORDER BY run_id, judge_model;"
```

---

## Production Checkpoints

| Checkpoint | Path | Precision | Size | Forward P50 | Memory |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **W4A16 Production** | `./ckpt/gemma-4-e2b-nli-w4a16` | INT4 Group-32 | 7.04 GB | **14.31 ms** | **5.1 GB VRAM** |
| **QAT Stage 1 Adapter** | `./ckpt/gemma-4-e2b-nli-qat-stage1/best` | BF16 LoRA (4-bit STE) | 52 MB | 32.70 ms | 12.2 GB VRAM |
| **BF16 Stage 1 Baseline** | `./ckpt/gemma-4-e2b-nli-stage1/best` | BF16 LoRA | ~10.2 GB | 18.33 ms | 10.2 GB VRAM |

---

## License & Attribution

Apache 2.0. Base model weights inherit the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

This project integrates architectural patterns, synthetic data strategies, and loss formulations inspired by the following open-source projects:
- **[TypeSafe AI Jev](http://typesafe.ai/blog/introducing-system-one-models-and-jev)** & **[Convai Laya](https://huggingface.co/convaiinnovations/laya)**: Fast non-autoregressive System 1 decision-engine paradigm and latent probe head design.
- **[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev)** (Apache 2.0): NLI label indexing convention (0=contradiction, 1=entailment, 2=neutral) and evaluation harness structure.
- **[Bespoke Labs Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)** (Apache 2.0): Counterfactual adversarial fact inversion data generation (numeric mutation, polarity flipping, entity swapping).
- **[Mapika/decider](https://github.com/Mapika/decider)** (Apache 2.0): Abstention augmentation (`none_augment`), negative control pairing, and temperature scaling calibration.
- **[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)** (MIT): Position-bias formulation, template diversification, and Position Bias Index ($\text{PBI}$) metric.
- **[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)** (MIT) & **[sabeel111/OpenSourceJev](https://github.com/sabeel111/OpenSourceJev)** (MIT): Deterministic token-bucket batching architecture and post-hoc temperature optimization.
- **[von-1.0](https://github.com/jina-ai)**: Multi-class proper-scoring Brier calibration loss formulation ($\mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Brier}}$).
