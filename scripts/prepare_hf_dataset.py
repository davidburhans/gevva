#!/usr/bin/env python3
"""scripts/prepare_hf_dataset.py
=================================
Prepares and packages the complete Gevva training, benchmark, and multimodal
decision datasets for publication on Hugging Face Hub: `davidburhans/gevva-decisions`.

Datasets included:
1. `flagship_train.jsonl` (Phase 3 Enriched master mixture, #1 Global JevBench: 77.54)
2. `multimodal_train.jsonl` (Phase 4 Vision grounding mixture, 88.8% visual accuracy)
3. `sdk_decisions_train.jsonl` & `sdk_decisions_val.jsonl` (Committee-verified tool routing, reranking, grading)
4. `weak_family_remediation.jsonl` (Targeted multi-hop, policy precedence, temporal arithmetic)
5. `test_decisions.jsonl` (Held-out 2,947-item downstream decision benchmark)
6. `test_multimodal.jsonl` (Held-out 249-item visual grounding benchmark)
7. `images/` (Synthesized image assets for invoices/tables, charts/graphs, and spatial scenes)
"""

import json
import os
import shutil
from pathlib import Path


def prepare_dataset_staging(staging_dir: str = "data/hf_dataset_staging"):
    staging_path = Path(staging_dir)
    data_dir = staging_path / "data"
    images_dir = staging_path / "images"

    staging_path.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Staging directory created at: {staging_path}")

    # 1. Copy flagship Phase-3 enriched master mixture
    src_flagship = "data/train_phase3_enriched.jsonl"
    dst_flagship = data_dir / "flagship_train.jsonl"
    print(f"Copying flagship training data ({src_flagship} -> {dst_flagship})...")
    shutil.copy2(src_flagship, dst_flagship)

    # 2. Normalize and copy multimodal Phase-4 mixture
    src_p4 = "data/train_phase4_mixture.jsonl"
    dst_p4 = data_dir / "multimodal_train.jsonl"
    print(f"Normalizing image paths for multimodal training data ({src_p4} -> {dst_p4})...")
    with open(src_p4, "r", encoding="utf-8") as f_in, open(dst_p4, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            if "image" in row and row["image"]:
                img_path = str(row["image"])
                if "images/" in img_path:
                    row["image"] = "images/" + img_path.split("images/")[-1]
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3. Copy SDK-parity synthetic decision data
    src_sdk_tr = "data/staged/sdk_synthetic_train.jsonl"
    dst_sdk_tr = data_dir / "sdk_decisions_train.jsonl"
    print(f"Copying SDK decisions train ({src_sdk_tr} -> {dst_sdk_tr})...")
    shutil.copy2(src_sdk_tr, dst_sdk_tr)

    src_sdk_val = "data/staged/sdk_synthetic_val.jsonl"
    dst_sdk_val = data_dir / "sdk_decisions_val.jsonl"
    print(f"Copying SDK decisions val ({src_sdk_val} -> {dst_sdk_val})...")
    shutil.copy2(src_sdk_val, dst_sdk_val)

    # 4. Copy targeted weak-family remediation
    src_weak = "data/staged/phase4/synth_phase4_remediation.jsonl"
    dst_weak = data_dir / "weak_family_remediation.jsonl"
    print(f"Copying weak family remediation ({src_weak} -> {dst_weak})...")
    shutil.copy2(src_weak, dst_weak)

    # 5. Copy held-out benchmarks
    src_test = "data/test.jsonl"
    dst_test = data_dir / "test_decisions.jsonl"
    print(f"Copying downstream decision test benchmark ({src_test} -> {dst_test})...")
    shutil.copy2(src_test, dst_test)

    src_mm_val = "data/visual_synth/val.jsonl"
    dst_mm_val = data_dir / "test_multimodal.jsonl"
    print(f"Normalizing and copying multimodal validation set ({src_mm_val} -> {dst_mm_val})...")
    with open(src_mm_val, "r", encoding="utf-8") as f_in, open(dst_mm_val, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            if "image" in row and row["image"]:
                img_path = str(row["image"])
                if "images/" in img_path:
                    row["image"] = "images/" + img_path.split("images/")[-1]
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. Copy visual synthetic images
    src_imgs_dir = Path("data/visual_synth/images")
    if src_imgs_dir.exists():
        print(f"Copying visual assets from {src_imgs_dir} to {images_dir}...")
        img_count = 0
        for f in src_imgs_dir.glob("*.*"):
            shutil.copy2(f, images_dir / f.name)
            img_count += 1
        print(f"Copied {img_count} image assets.")

    # 7. Write README.md (Dataset Card)
    readme_content = """---
license: apache-2.0
language:
- en
- fr
- es
- de
- zh
- ar
- hi
- ru
- sw
- vi
- multilingual
task_categories:
- text-classification
- feature-extraction
- visual-question-answering
tags:
- gevva
- cross-encoder
- nli
- system1
- decision-engine
- tool-routing
- search-reranking
- multimodal
- visual-entailment
- synthetic-data
- jevbench
size_categories:
- 100K<n<1M
configs:
- config_name: flagship
  data_files: "data/flagship_train.jsonl"
- config_name: multimodal
  data_files: "data/multimodal_train.jsonl"
- config_name: sdk_decisions
  data_files:
    - split: train
      path: "data/sdk_decisions_train.jsonl"
    - split: validation
      path: "data/sdk_decisions_val.jsonl"
- config_name: remediation
  data_files: "data/weak_family_remediation.jsonl"
- config_name: benchmarks
  data_files:
    - split: test_decisions
      path: "data/test_decisions.jsonl"
    - split: test_multimodal
      path: "data/test_multimodal.jsonl"
---

# ⚡ Gevva Decisions: Complete Training Curriculum & Benchmark Suite

Official training mixtures, committee-verified synthetic datasets, and held-out benchmarks used to train and evaluate the **Gevva** family of large-context (128K), multimodal System 1 decision engines (**#1 Global on JevBench: 77.54 Composite Score**).

- **Associated Models**:
  - Flagship: [`davidburhans/gevva-e2b`](https://huggingface.co/davidburhans/gevva-e2b) (Default `main` branch)
  - Multimodal: [`davidburhans/gevva-e2b` (branch `multimodal`)](https://huggingface.co/davidburhans/gevva-e2b/tree/multimodal)
- **Codebase & Reproduction Scripts**: [https://github.com/davidburhans/gevva](https://github.com/davidburhans/gevva)

---

## 📦 Dataset Configurations & Splits

| Configuration | File | Records | Description | Primary Role |
| :--- | :--- | :---: | :--- | :--- |
| **`flagship`** | `data/flagship_train.jsonl` | 145,000+ | Master Phase-3 enriched curriculum | Reproduces **#1 Global JevBench Champion (77.54)** |
| **`multimodal`** | `data/multimodal_train.jsonl` | 65,472 | Phase-4 multimodal + anchor replay | Trains vision grounding (**88.8%** accuracy, 96.4% tables) |
| **`sdk_decisions`** | `data/sdk_decisions_train.jsonl` / `val` | 8,515 | 4-judge committee-verified SDK pairs | Tool routing, candidate reranking, rubric grading |
| **`remediation`** | `data/weak_family_remediation.jsonl` | 7,600 | Targeted weak-family reasoning pairs | Complex policy precedence, arithmetic, multi-hop |
| **`benchmarks`** | `data/test_decisions.jsonl` | 2,947 | Held-out decision benchmark | Generalization testing & downstream decision scoring |
| **`benchmarks`** | `data/test_multimodal.jsonl` | 249 | Held-out visual grounding benchmark | Evaluates tables, charts, invoices, and spatial scenes |

---

## 🛠️ How to Load via Hugging Face `datasets`

```python
from datasets import load_dataset

# 1. Load the flagship master training mixture
flagship_ds = load_dataset("davidburhans/gevva-decisions", "flagship")
print(flagship_ds["train"][0])

# 2. Load SDK-parity decision pairs (tool routing, search reranking, grading)
sdk_ds = load_dataset("davidburhans/gevva-decisions", "sdk_decisions")
print(f"SDK Train: {len(sdk_ds['train'])}, SDK Val: {len(sdk_ds['validation'])}")

# 3. Load held-out test benchmarks
benchmarks = load_dataset("davidburhans/gevva-decisions", "benchmarks")
print(f"Decisions Test: {len(benchmarks['test_decisions'])}, Multimodal Test: {len(benchmarks['test_multimodal'])}")
```

---

## 🔬 Dataset Schema & Record Format

All records follow the unified cross-encoder / decision format:

```json
{
  "premise": "Customer account balance is $450.00 with pending refund request for $45.00.",
  "hypothesis": "Process a refund of $45.00 to original payment method.",
  "label": 1,
  "source": "sdk_synthetic_tool_routing",
  "group_id": 1042,
  "language": "en"
}
```

### Label Conventions:
Following the standard `ModernCE` / `openjev` specification:
- `0`: **Contradiction** (Reject candidate / refuted claim)
- `1`: **Entailment** (Select candidate / verified claim)
- `2`: **Neutral** (Unverifiable / irrelevant candidate)

---

## 📜 Citation & License

This dataset is released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

```bibtex
@dataset{gevva_decisions_2026,
  author = {Burhans, Dave and Contributors},
  title = {Gevva Decisions: Master Training Curriculum & Benchmark Suite for Multimodal System 1 Engines},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/davidburhans/gevva-decisions}
}
```
"""
    readme_path = staging_path / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"README.md written at: {readme_path}")
    print("Staging complete!")


if __name__ == "__main__":
    prepare_dataset_staging()
