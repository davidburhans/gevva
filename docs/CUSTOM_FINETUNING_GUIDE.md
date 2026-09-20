# Custom Data Fine-Tuning Guide: Gemma 4 NLI Cross-Encoder

This guide provides step-by-step instructions for fine-tuning the **Gemma 4 NLI Cross-Encoder / System 1 Decision Engine** on domain-specific datasets.

The fine-tuning pipeline is designed for **zero boilerplate**: it automatically detects input file formats, column headers, and label formats, performs stratified train/validation splits, and saves lightweight LoRA adapter checkpoints (~21 MB).

---

## 1. Quickstart (1-Line Fine-Tuning)

### CLI
```bash
# Fine-tune starting from base model (auto-detects columns and splits 85/15 train/val)
python finetune.py --data my_data.jsonl --out-dir ./ckpt/my_domain_model

# Or continually fine-tune starting from our pre-trained NLI checkpoint
python finetune.py \
    --data my_data.csv \
    --adapter ./ckpt/gemma-4-e2b-nli-stage1/best \
    --out-dir ./ckpt/my_domain_finetuned \
    --epochs 3
```

### Python API
```python
from finetune import finetune_custom_data

results = finetune_custom_data(
    train_data="my_dataset.jsonl",
    output_dir="./ckpt/my_domain_model",
    adapter_path="./ckpt/gemma-4-e2b-nli-stage1/best",  # optional: resume from NLI checkpoint
    epochs=3,
    batch_size=8,
    lr=2e-4,
)

print(f"Fine-tuning complete! Best accuracy: {results['best_accuracy']*100:.2f}%")
```

---

## 2. Supported Data Formats

The engine supports `.jsonl`, `.csv`, `.tsv`, and `.json`.

### Column Auto-Detection
The engine automatically detects column names matching any common conventions:

| Field | Recognized Column Names (Case-Insensitive) |
| :--- | :--- |
| **Premise / Context** | `premise`, `context`, `document`, `passage`, `evidence`, `text_a`, `prompt`, `reference` |
| **Hypothesis / Claim** | `hypothesis`, `claim`, `assertion`, `candidate`, `response`, `text_b`, `query`, `answer` |
| **Label / Verdict** | `label`, `gold`, `target`, `annotation`, `class`, `ground_truth`, `verdict` |

### Label Auto-Mapping
Labels can be integers or strings:

| Class | ID | Recognized String Formats |
| :--- | :---: | :--- |
| **Contradiction** | `0` | `"contradiction"`, `"contradict"`, `"refutes"`, `"refuted"`, `"false"`, `"no"`, `"negative"`, `"0"` |
| **Entailment** | `1` | `"entailment"`, `"entails"`, `"supports"`, `"supported"`, `"true"`, `"yes"`, `"positive"`, `"1"` |
| **Neutral** | `2` | `"neutral"`, `"unverifiable"`, `"unknown"`, `"not_enough_info"`, `"nei"`, `"neither"`, `"2"` |

---

## 3. Real-World Task Recipes

### Recipe 1: RAG Hallucination Detection & Citation Grounding
Verify whether extracted answers or generated claims are faithful to retrieved documents.

**Dataset (`rag_grounding.jsonl`)**:
```json
{"context": "Patient was diagnosed with Type 2 diabetes in 2021 and prescribed Metformin 500mg daily.", "claim": "Patient is taking Metformin for diabetes management.", "verdict": "supports"}
{"context": "Patient was diagnosed with Type 2 diabetes in 2021 and prescribed Metformin 500mg daily.", "claim": "Patient was diagnosed with Type 1 diabetes in childhood.", "verdict": "refutes"}
{"context": "Patient was diagnosed with Type 2 diabetes in 2021 and prescribed Metformin 500mg daily.", "claim": "Patient has a family history of hypertension.", "verdict": "unverifiable"}
```

**Run Fine-Tuning**:
```bash
python finetune.py \
    --data rag_grounding.jsonl \
    --adapter ./ckpt/gemma-4-e2b-nli-stage1/best \
    --out-dir ./ckpt/medical_rag_verifier \
    --epochs 3 \
    --lr 1e-4
```

---

### Recipe 2: Zero-Shot Intent & Tool Routing
Route user prompts directly to candidate tools/APIs in a single forward pass (~25 ms).

**Dataset (`tool_routing.csv`)**:
```csv
prompt,candidate_tool,class
"Send $200 to Dave for utilities","Execute financial wire transfer or P2P payment",entailment
"Send $200 to Dave for utilities","Query weather forecast API",contradiction
"What is the forecast for tomorrow in Berlin?","Query weather forecast API",entailment
"What is the forecast for tomorrow in Berlin?","Search web news articles",neutral
```

**Run Fine-Tuning**:
```bash
python finetune.py \
    --data tool_routing.csv \
    --adapter ./ckpt/gemma-4-e2b-nli-stage1/best \
    --out-dir ./ckpt/agent_router \
    --epochs 3
```

---

### Recipe 3: Multi-Choice Candidate Reranking
Rank multiple candidates (e.g., search passages, code snippets, or product recommendations).

**Inference with `Gemma4CrossEncoder` (1-Line Loading)**:
```python
from gemma4_cross_encoder import Gemma4CrossEncoder

# 1-line initialization (automatically detects base model, LoRA adapter, and head weights)
encoder = Gemma4CrossEncoder("./ckpt/my_domain_model/best")

# Candidate reranking in a single batched pass (~20 ms)
query = "How to handle out-of-memory errors in PyTorch gradient checkpointing?"
passages = [
    "PyTorch gradient checkpointing trades compute for memory by recalculating activations.",
    "The capital of Washington state is Olympia.",
    "TensorBoard visualizes training loss curves over time."
]

best_idx, scores = encoder.rerank(
    premise=f"Query: {query}",
    options=passages,
    hyp_format="Relevant documentation: {}",
)

print(f"Top Candidate ({scores[best_idx]*100:.1f}% confidence): {passages[best_idx]}")
```

---

## 4. Hardware Sizing & Recommended Settings

The fine-tuning engine is optimized for consumer GPUs:

| Hardware | Max Length | Batch Size | Grad Accum | Effective Batch Size | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 5090 (32GB)** | 512 | 4 | 8 | 32 | ~11.6 GB |
| **NVIDIA RTX 4090 / 3090 (24GB)** | 512 | 4 | 8 | 32 | ~11.6 GB |
| **NVIDIA RTX 4080 / 16GB GPU** | 256 | 2 | 16 | 32 | ~9.5 GB |

---

## 5. Output Artifacts

Each fine-tuning run saves the following in `<output_dir>/best/`:
1. `adapter_model.safetensors` (~21 MB): Trained LoRA parameters + classification head (`score`).
2. `head_weights.pt`: Explicit serialized state dictionaries for `score` and `norm`.
3. `adapter_config.json`: LoRA configuration matching text decoder projections.
4. `tokenizer.json` & `tokenizer_config.json`: Gemma 4 256K vocabulary tokenizer.
5. `eval_report.json`: Per-class precision, recall, F1, accuracy, and Brier calibration score.

---

## 6. Quantization-Aware Training (QAT) & W4A16 Deployment

To produce a model tailored for low-memory, high-throughput edge deployment:

### Step 1: Train with In-Loop 4-Bit Group QAT
```bash
python finetune.py \
    --data my_data.jsonl \
    --adapter ./ckpt/gemma-4-e2b-nli-stage1/best \
    --out-dir ./ckpt/my_domain_qat \
    --qat \
    --qat-bits 4 \
    --qat-group-size 32 \
    --epochs 2
```

### Step 2: Export to W4A16 `compressed-tensors` Format
Merge the LoRA adapter and pack target linear layers into INT4 Group-32 format:
```bash
python export_w4a16.py \
    --base-model google/gemma-4-E2B \
    --adapter-path ./ckpt/my_domain_qat/best \
    --output-dir ./ckpt/my_domain_w4a16 \
    --group-size 32
```
This produces a unified `model.safetensors` and `quantization_config.json` compatible with vLLM, TensorRT-LLM, and ExLlamaV2/Marlin runtimes with ~4x memory reduction and zero accuracy degradation.

---

## 7. Advanced SOTA Training Techniques

### Deterministic Token-Bucket Batching
Variable-length sequences cause severe padding overhead and GPU memory spikes. Pass `--use-token-bucketing` to group sequences into discrete length buckets ($64 \le L \le 131,072$) and bound total tokens per batch:
```bash
python finetune.py \
    --data my_data.jsonl \
    --use-token-bucketing \
    --max-tokens-per-batch 8192 \
    --out-dir ./ckpt/my_domain_bucketed
```

### Post-Hoc Temperature Calibration ($T^*$)
At the conclusion of fine-tuning, the engine automatically fits scalar temperature $T^*$ on validation logits using L-BFGS to minimize validation NLL without modifying $\arg\max$ predictions. It exports `calibration.json` to the output checkpoint directory:
```json
{
  "optimal_temperature": 1.1420,
  "val_ece_before": 0.0572,
  "val_ece_after": 0.0241,
  "val_brier_before": 0.2315,
  "val_brier_after": 0.2189
}
```
`Gemma4CrossEncoder` automatically detects and applies `calibration.json` at inference time.

### Multi-Class Proper-Scoring Brier Loss ($\lambda \cdot \text{Brier}$)
In addition to standard cross-entropy, penalize uncalibrated confidence probabilities directly during training via `--brier-weight 0.5`:
```bash
python finetune.py \
    --data my_data.jsonl \
    --brier-weight 0.5 \
    --out-dir ./ckpt/my_domain_calibrated
```

### Multimodal Fine-Tuning
If your dataset contains visual evidence (e.g. document images, UI screenshots, or camera frames), specify the image path in the `image` column:
```json
{"premise": "Customer invoice dated 2026-03-15.", "image": "images/invoice_001.jpg", "hypothesis": "Total due is $1,240.50.", "label": "entailment"}
```
`CustomNLICollator` automatically loads the image with PIL, extracts SigLIP visual patch features via `Gemma4ImageProcessorPil`, dynamically allocates soft tokens in `input_ids`, and passes `pixel_values` and `image_position_ids` directly to the model.
