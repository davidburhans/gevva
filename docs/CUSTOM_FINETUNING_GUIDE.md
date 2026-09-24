# Custom Data Fine-Tuning Guide: Gevva Decision Engine

This guide provides step-by-step instructions for fine-tuning the **Gevva System 1 Decision Engine** on domain-specific datasets.

The fine-tuning pipeline is designed for **zero boilerplate**:
- **Format Auto-Detection**: Supports `.jsonl`, `.csv`, `.tsv`, `.parquet`, and `.json`.
- **Column Auto-Mapping**: Automatically detects premise, hypothesis, and label fields regardless of naming conventions.
- **Label Normalization**: Accepts raw integer IDs (`0`, `1`, `2`) or natural language strings (`"supports"`, `"refutes"`, `"unverifiable"`).
- **Stratified Auto-Splits**: Automatically creates balanced train/val splits when a separate validation set is not provided.
- **Dual Training Paradigms**: Supports both fast Parameter-Efficient Fine-Tuning (LoRA) and high-accuracy Full Fine-Tuning (FFT).

---

## 1. Quickstart (1-Line Fine-Tuning)

### Using the Gevva CLI
```bash
# 1-line fine-tuning starting from base Gemma 4 E2B-it (auto-detects columns and splits 85/15)
gevva finetune --data my_data.jsonl --out-dir ./ckpt/my_domain_gevva

# Continually adapt starting from the champion Gevva e2b checkpoint
gevva finetune \
    --data my_data.jsonl \
    --base-model ckpt/gevva-e2b \
    --out-dir ./ckpt/my_domain_finetuned \
    --full-fine-tune \
    --epochs 2
```

### Python API
```python
from finetune import finetune_custom_data

results = finetune_custom_data(
    train_data="my_dataset.jsonl",
    output_dir="./ckpt/my_domain_model",
    base_model_id="ckpt/gevva-e2b",
    full_fine_tune=True,
    epochs=2,
    batch_size=4,
    grad_accum=8,
    lr=1e-5,
)

print(f"Fine-tuning complete! Best accuracy: {results['best_accuracy']*100:.2f}%")
```

---

## 2. Supported Data Formats & Conventions

### Recognized Column Names
The engine inspects header keys case-insensitively and maps them automatically:

| Field | Recognized Column Names |
| :--- | :--- |
| **Premise / Context** | `premise`, `context`, `document`, `passage`, `evidence`, `text_a`, `prompt`, `reference` |
| **Hypothesis / Claim** | `hypothesis`, `claim`, `assertion`, `candidate`, `response`, `text_b`, `query`, `answer` |
| **Label / Target** | `label`, `gold`, `target`, `annotation`, `class`, `ground_truth`, `verdict` |
| **Multimodal Image** | `image`, `image_path`, `img`, `visual_evidence`, `photo` |

### Label Schema Normalization

| Class | ID | Recognized String Formats |
| :--- | :---: | :--- |
| **Contradiction** | `0` | `"contradiction"`, `"contradict"`, `"refutes"`, `"refuted"`, `"false"`, `"no"`, `"negative"`, `"0"` |
| **Entailment** | `1` | `"entailment"`, `"entails"`, `"supports"`, `"supported"`, `"true"`, `"yes"`, `"positive"`, `"1"` |
| **Neutral** | `2` | `"neutral"`, `"unverifiable"`, `"unknown"`, `"not_enough_info"`, `"nei"`, `"neither"`, `"2"` |

---

## 3. Real-World Task Recipes

### Recipe 1: RAG Hallucination Detection & Document Grounding
Verify whether LLM-generated claims are faithful to source reference documentation.

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
    --base-model ckpt/gevva-e2b \
    --out-dir ./ckpt/medical_rag_verifier \
    --full-fine-tune \
    --epochs 2 \
    --lr 1e-5
```

---

### Recipe 2: Zero-Shot Intent & Tool Routing
Route user requests to the correct agent tool or microservice in a single forward pass (~16 ms).

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
    --base-model ckpt/gevva-e2b \
    --out-dir ./ckpt/agent_router \
    --epochs 3
```

---

### Recipe 3: Multi-Choice Candidate Reranking
Rank candidate search passages or documents given a query.

**Inference with `Gevva`**:
```python
import gevva

encoder = gevva.load("./ckpt/my_domain_model/best")

query = "How to handle out-of-memory errors during PyTorch training?"
passages = [
    "PyTorch gradient checkpointing trades compute for memory by recomputing activations during backward pass.",
    "The capital of Washington state is Olympia.",
    "TensorBoard visualizes training loss curves over training epochs."
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

The fine-tuning engine is optimized for modern NVIDIA GPUs:

| Hardware | Mode | Max Length | Batch Size | Grad Accum | Effective Batch Size | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 5090 (32GB)** | Full Fine-Tune (FFT) | 2,048 | 4 | 8 | 32 | ~24.5 GB |
| **NVIDIA RTX 5090 (32GB)** | LoRA Adapter | 2,048 | 8 | 4 | 32 | ~14.2 GB |
| **NVIDIA RTX 4090 / 3090 (24GB)**| LoRA Adapter | 1,024 | 4 | 8 | 32 | ~12.5 GB |
| **NVIDIA RTX 4080 (16GB)** | LoRA Adapter | 512 | 2 | 16 | 32 | ~9.8 GB |

---

## 5. Output Checkpoint Artifacts

Each fine-tuning run outputs the following in `<output_dir>/best/`:
1. `model.safetensors` (for Full Fine-Tuning) or `adapter_model.safetensors` (for LoRA).
2. `config.json` & `tokenizer.json`: Gemma 4 tokenizer and sequence classification config.
3. `head_weights.pt`: Explicit serialized tensors for classification score and layer norm.
4. `calibration.json`: Fitted temperature scaling factor ($T^*$) and validation ECE / Brier metrics.
5. `eval_report.json`: Per-class accuracy, precision, recall, and F1.

---

## 6. Advanced Training Options

### Full Fine-Tuning vs. LoRA
- **LoRA (Default)**: Trains low-rank adapters on attention and MLP projections. Very fast, uses minimal disk space (~21 MB adapter), ideal for modest domain datasets (<10K samples).
- **Full Fine-Tuning (`--full-fine-tune`)**: Keeps embedding tables and vision tower frozen while tuning all transformer backbone layers. Achieved **#1 in the world on JevBench (77.54)** by enabling deep semantic co-adaptation. Recommended when fine-tuning on ≥20K examples.

### Multi-Class Proper-Scoring Brier Loss
Penalize uncalibrated probabilities directly in the loss function via `--brier-weight 0.5`:
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Brier}}$$
This prevents the model from generating overconfident predictions on ambiguous inputs.

### Post-Hoc Temperature Calibration
The fine-tuning engine automatically computes the optimal scalar temperature ($T^*$) on validation logits using L-BFGS to minimize negative log-likelihood (NLL). When loading the model with `gevva.load()`, this temperature is applied automatically at inference time.

### Multimodal Vision Fine-Tuning
To train on images alongside text, simply populate an `image` column in your JSONL/CSV with local image paths. The data collator automatically processes patches through SigLIP and embeds visual tokens inside the premise.
