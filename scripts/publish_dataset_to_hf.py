#!/usr/bin/env python3
"""scripts/publish_dataset_to_hf.py - Hugging Face Hub publisher for Gevva multimodal datasets.

Converts multi-image and multimodal decision datasets into Parquet with native
Hugging Face `datasets.Image` features (viewable directly on the HF Hub web UI),
generates standard Dataset Card README metadata, and pushes to Hugging Face Hub.

Usage:
  # Dry-run validation & local Parquet build:
  uv run python scripts/publish_dataset_to_hf.py \
      --data-dir data/multimodal_multi_image \
      --dry-run

  # Publish directly to Hugging Face Hub:
  uv run python scripts/publish_dataset_to_hf.py \
      --data-dir data/multimodal_multi_image \
      --repo-id davidburhans/gevva-multimodal-decisions \
      --private False
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import datasets
from datasets import ClassLabel, Features, Sequence, Value
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import ID2LABEL, LABEL2ID  # noqa: E402


def generate_dataset_card(
    repo_id: str,
    train_count: int,
    val_count: int,
    task_families: List[str],
) -> str:
    """Generates standard YAML frontmatter + Markdown dataset card for Hugging Face Hub."""
    families_str = "\n".join(f"- `{f}`" for f in sorted(set(task_families)))
    return f"""---
license: mit
task_categories:
  - zero-shot-classification
  - visual-question-answering
  - text-classification
language:
  - en
tags:
  - nli
  - cross-encoder
  - multimodal
  - multi-image
  - counterfactual
  - decision-engine
  - gemma-4
  - qwen-image
size_categories:
  - 1K<n<10K
viewer: true
---

# {repo_id}

This dataset contains high-quality, verified **multimodal and multi-image decision pairs** designed for training and benchmarking non-autoregressive **System 1 Decision Engines** and **NLI Cross-Encoders** (such as [Gevva](https://github.com/davidburhans/nli-cross-encoder)).

## Key Features

- **Multi-Image Support**: Contains paired and sequence images (e.g. Image 1 vs Image 2 Before/After states, cross-chart comparisons, UI workflows).
- **Exact Ground-Truth Logic**: Every sample includes calibrated claims categorized into 3 standard states:
  - `contradiction` (0)
  - `entailment` (1)
  - `neutral` (2)
- **Hard Adversarial Distractors**: Near-miss counterfactuals where a single visual dependency is mutated, preventing superficial keyword matching.
- **Committee-Verified**: Cleaned and validated through multi-judge consensus with zero label ambiguity.

## Dataset Structure

- **Train Samples**: {train_count:,}
- **Validation Samples**: {val_count:,}

### Task Families Included
{families_str}

## Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique sample identifier |
| `premise` | `string` | Contextual text premise referencing the images |
| `hypothesis` | `string` | The claim evaluated against visual & textual evidence |
| `label` | `ClassLabel` | `0: contradiction`, `1: entailment`, `2: neutral` |
| `task_family` | `string` | Category of visual decision task |
| `images` | `Sequence(Image)` | List of PIL images embedded in the Parquet dataset |
| `num_images` | `int32` | Number of images associated with this sample |
| `metadata` | `string` | JSON-encoded provenance metadata |

## Usage with Hugging Face `datasets`

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset["train"][0])
```

## Citation & Attribution

If you use this dataset in your research or applications, please cite:

```bibtex
@misc{{gevva2026multimodal,
  author = {{David Burhans and Contributors}},
  title = {{Gevva Multimodal Decision & Multi-Image NLI Dataset}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face Hub}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```
"""


def load_split_records(file_path: Path, image_root: Path) -> List[Dict[str, Any]]:
    """Loads JSONL records and attaches resolved PIL images."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # Resolve image paths
            raw_imgs = data.get("images")
            if raw_imgs is None:
                single = data.get("image")
                raw_imgs = single if isinstance(single, list) else ([single] if single else [])
            elif isinstance(raw_imgs, str):
                raw_imgs = [raw_imgs]

            pil_images = []
            for img_name in raw_imgs:
                p = Path(img_name)
                if not p.is_absolute():
                    p = image_root / p
                if not p.exists():
                    p_alt = file_path.parent / img_name
                    if p_alt.exists():
                        p = p_alt
                
                if p.exists():
                    try:
                        pil_images.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        print(f"Warning: Failed to load image {p}: {e}")
                else:
                    print(f"Warning: Image file not found: {p}")

            # Normalize label
            raw_label = data.get("label", 1)
            if isinstance(raw_label, str):
                label_id = LABEL2ID.get(raw_label.lower(), 1)
            else:
                label_id = int(raw_label)

            records.append({
                "id": str(data.get("id", f"{file_path.stem}_{line_num}")),
                "premise": str(data.get("premise", "")),
                "hypothesis": str(data.get("hypothesis", "")),
                "label": label_id,
                "task_family": str(data.get("task_family", data.get("family", "multimodal_decision"))),
                "images": pil_images,
                "num_images": len(pil_images),
                "metadata": json.dumps(data.get("metadata", {})),
            })
    return records


def build_and_publish_dataset(
    data_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    private: bool = False,
    dry_run: bool = False,
) -> None:
    data_dir = data_dir.resolve()
    image_root = data_dir / "images"
    if not image_root.exists():
        image_root = data_dir

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        # Check for fallback single file
        for fallback in ["data.jsonl", "dataset.jsonl", "mixture.jsonl"]:
            if (data_dir / fallback).exists():
                train_file = data_dir / fallback
                break

    if not train_file.exists():
        raise FileNotFoundError(f"No train.jsonl found in {data_dir}")

    print(f"Loading train split from {train_file}...")
    train_records = load_split_records(train_file, image_root)
    print(f"Loaded {len(train_records):,} train records.")

    val_records = []
    if val_file.exists():
        print(f"Loading val split from {val_file}...")
        val_records = load_split_records(val_file, image_root)
        print(f"Loaded {len(val_records):,} val records.")

    features = Features({
        "id": Value("string"),
        "premise": Value("string"),
        "hypothesis": Value("string"),
        "label": ClassLabel(names=["contradiction", "entailment", "neutral"]),
        "task_family": Value("string"),
        "images": Sequence(datasets.Image()),
        "num_images": Value("int32"),
        "metadata": Value("string"),
    })

    task_families = [r["task_family"] for r in train_records] + [r["task_family"] for r in val_records]

    splits = {}
    splits["train"] = datasets.Dataset.from_list(train_records, features=features)
    if val_records:
        splits["validation"] = datasets.Dataset.from_list(val_records, features=features)

    dataset_dict = datasets.DatasetDict(splits)
    print(f"Constructed DatasetDict:\n{dataset_dict}")

    # Generate Dataset Card README
    card_content = generate_dataset_card(
        repo_id=repo_id,
        train_count=len(train_records),
        val_count=len(val_records),
        task_families=task_families,
    )
    readme_path = data_dir / "README.md"
    readme_path.write_text(card_content, encoding="utf-8")
    print(f"Wrote Hugging Face Dataset Card to {readme_path}")

    # Save local Parquet cache
    parquet_dir = data_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ds in dataset_dict.items():
        out_p = parquet_dir / f"{split_name}.parquet"
        ds.to_parquet(str(out_p))
        print(f"Saved local Parquet: {out_p} ({out_p.stat().st_size / (1024**2):.2f} MB)")

    if dry_run:
        print("[dry-run] Dataset validated and Parquet files generated successfully. Skipping HF upload.")
        return

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id} (private={private})...")
        dataset_dict.push_to_hub(repo_id, private=private)
        # Upload the README.md card
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"SUCCESS: Published to https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Gevva Multimodal Dataset to Hugging Face Hub")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing train.jsonl and images/")
    parser.add_argument("--repo-id", default="davidburhans/gevva-multimodal-decisions", help="Hugging Face repo ID")
    parser.add_argument("--push-to-hub", action="store_true", help="Push directly to Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Make Hugging Face repo private")
    parser.add_argument("--dry-run", action="store_true", help="Build local Parquet and validate schema without uploading")
    args = parser.parse_args()

    build_and_publish_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        private=args.private,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
