# Inclusion Request: Gevva e2b and Gevva e4b on JevBench

**Target Repository**: [https://github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (or [Benchmark Heaven](https://benchmarkheaven.com/jev-models))  
**Issue Title**: `Inclusion Request: Gevva e2b and Gevva e4b (Gemma 4 System 1 Cross-Encoders)`

---

## Issue / Submission Body

Hi Florian (@fstandhartinger),

We would love to request the inclusion of **Gevva e2b** and **Gevva e4b** on the official JevBench leaderboard (and Benchmark Heaven). 

Both models are fully open-source (Apache 2.0) System 1 decision engines and cross-encoders built on Google's lightweight Gemma 4 foundation models. They evaluate bounded decision rubrics and classification states in a single forward pass without autoregressive token generation.

We evaluated both models locally against the exact frozen `jevbench/datasets/public` 231-item split using `scripts/eval_jevbench_public.py` (which imports the task loader, mapping helpers, and composite scoring functions directly from the `jevbench` repository).

---

### 1. Model Summary & Hugging Face Checkpoints

| Model | Base Architecture | Trainable Parameters | Context | Weights & Model Card |
| :--- | :--- | :---: | :---: | :--- |
| **Gevva e2b** | `google/gemma-4-E2B-it` | ~2.3B (18 layers) | 128K | [huggingface.co/davidburhans/gevva-e2b](https://huggingface.co/davidburhans/gevva-e2b) |
| **Gevva e4b** | `google/gemma-4-E4B-it` | ~4.5B (42 layers) | 128K | [huggingface.co/davidburhans/gevva-e4b](https://huggingface.co/davidburhans/gevva-e4b) |

- **GitHub Repository**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)
- **PyPI SDK**: `pip install gevva`
- **Interactive Space**: [huggingface.co/spaces/davidburhans/gevva-demo](https://huggingface.co/spaces/davidburhans/gevva-demo)
- **License**: Apache 2.0

---

### 2. Evaluated Scores on JevBench Public-231

Evaluated under the official `composite_v12.py` geometric mean across the four standard axes:

$$\text{JevBench Score} = \left(\text{Intelligence} \times \text{Calibration} \times \text{Speed} \times \text{Cost}\right)^{1/4}$$

| Metric / Axis | Gevva e2b (2.3B) | Gevva e4b (4.5B) | Current #1 (Jev 1.13.0) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 71.43% (165/231) | **76.62%** (177/231) | ~68% |
| **Easy Tier Accuracy** | 100.0% (48/48) | **100.0%** (48/48) | 100.0% |
| **Standard Tier Accuracy** | 88.89% (64/72) | **94.44%** (68/72) | 88.9% |
| **Hard Tier Accuracy** | 47.75% (53/111) | **54.95%** (61/111) | 48.6% |
| **Hard Renormalized ECE** | **0.0655** | 0.0863 | 0.174 |
| **Intelligence Axis** | 73.91 | **79.07** | 85.7 |
| **Calibration Axis** | **86.90** | 82.74 | 82.7 |
| **Speed Axis (GPU adjusted)** | **86.86** ($p_{50}=19.2$ ms) | 84.13 ($p_{50}=31.0$ ms) | 83.3 ($p_{50}=650$ ms) |
| **Cost Axis ($0.0149 tariff)** | 64.80 | 64.80 | 52.0 ($0.0399 tariff) |
| **JevBench Composite (GPU)** | **77.54** | **77.28** | 74.4 |
| **JevBench Composite (API)** | 77.54 | **78.76** | 74.4 |

---

### 3. How to Reproduce / Evaluate Independently

You can verify and evaluate both models locally with two commands using the public checkpoints:

```bash
# 1. Install gevva SDK
pip install gevva

# 2. Evaluate Gevva e2b
python -c "
from gevva import load
model = load('davidburhans/gevva-e2b')
# Rerank, Grade, or evaluate directly
"

# 3. Or using the jevbench runner directly with the cross_encoder_local adapter:
jevbench run \
    --tasks datasets/public \
    --adapter cross_encoder_local \
    --endpoint davidburhans/gevva-e2b \
    --results results/gevva_e2b.jsonl
```

If you prefer to run the evaluations on your own RunPod / benchmarking hardware (RTX PRO 4500 / H100) to keep the latency normalization strictly consistent with the other entrants, we are completely on board!

Let us know if you need any additional metadata, artifacts, or adjustments. Thank you for building such an essential benchmark for the System 1 ecosystem!
