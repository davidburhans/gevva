#!/usr/bin/env python3
"""scripts/compile_stage2.py - Compiles the Flagship Stage 2 Training & Validation Dataset.

Mixture composition:
1. Stage 1 Clean Baseline (data/train.jsonl)
2. Staged SDK Synthetic Consensus (data/staged/sdk_synthetic_train.jsonl)
   - Filtered by 2-judge unanimous consensus (kappa=0.8391, 89.4% agreement).
   - Disagreements remain 100% quarantined in data/staged/sdk_synthetic_disagreements.jsonl.
3. Typed Decisions Synthetic (n4ze3m/typed-decisions-synth)
   - 2,000 enterprise cases (approx 18,000 NLI pairs).
   - DeepSeek V4.1 Flash soft calibrated probabilities.
   - MIT License, Muhammed Nazeem (2026).
4. Zero-Contamination Enforcement:
   - Strict holdout of all premises and pairs present in data/test.jsonl (n=3,113).

Outputs:
  data/stage2_train.jsonl
  data/stage2_val.jsonl
  data/stage2_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.adapters.typed_decisions_adapter import convert_typed_decisions_to_nli


def _pair_key(r: Dict[str, Any]) -> Tuple[str, str]:
    return (r.get("premise", "").strip(), r.get("hypothesis", "").strip())


def compile_stage2(
    data_dir: str = "data",
    typed_cases_train: int = 2000,
    typed_cases_val: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    data_path = Path(data_dir)

    print("=" * 65)
    print("STAGE 2 FLAGSHIP DATASET COMPILER")
    print("=" * 65)

    # 1. Load Golden Test Set for Contamination Guard
    test_file = data_path / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Golden test file {test_file} not found!")

    with open(test_file, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f if line.strip()]

    test_pairs: Set[Tuple[str, str]] = {_pair_key(r) for r in test_rows}
    test_premises: Set[str] = {r.get("premise", "").strip() for r in test_rows}
    test_sha = hashlib.sha256(open(test_file, "rb").read()).hexdigest()[:16]
    print(f"Loaded {len(test_rows):,} test rows (SHA: {test_sha}) for contamination guard.")

    # 2. Ingest Train Components
    train_rows: List[Dict[str, Any]] = []

    # 2a. Stage 1 Clean Baseline
    base_train_file = data_path / "train.jsonl"
    if base_train_file.exists():
        with open(base_train_file, "r", encoding="utf-8") as f:
            base_train = [json.loads(line) for line in f if line.strip()]
        train_rows.extend(base_train)
        print(f"  [1/3] Clean Baseline Train: {len(base_train):,} rows")
    else:
        print(f"  [1/3] Warning: {base_train_file} not found!")

    # 2b. Staged SDK Synthetic Consensus
    sdk_train_file = data_path / "staged" / "sdk_synthetic_train.jsonl"
    if sdk_train_file.exists():
        with open(sdk_train_file, "r", encoding="utf-8") as f:
            sdk_train = [json.loads(line) for line in f if line.strip()]
        train_rows.extend(sdk_train)
        print(f"  [2/3] SDK Consensus Synthetic: {len(sdk_train):,} rows")
    else:
        print(f"  [2/3] Warning: {sdk_train_file} not found!")

    # 2c. Typed Decisions Synthetic (n4ze3m/typed-decisions-synth)
    print(f"  [3/3] Loading Typed Decisions Synthetic ({typed_cases_train:,} cases)...")
    try:
        td_train = convert_typed_decisions_to_nli(
            split="train", max_cases=typed_cases_train, seed=seed
        )
        train_rows.extend(td_train)
        print(f"        Loaded {len(td_train):,} pairs from Typed Decisions (MIT License, Muhammed Nazeem 2026)")
    except Exception as e:
        print(f"        Failed to load Typed Decisions train: {e}")
        td_train = []

    # 3. Ingest Val Components
    val_rows: List[Dict[str, Any]] = []

    # 3a. Base Val
    base_val_file = data_path / "val.jsonl"
    if base_val_file.exists():
        with open(base_val_file, "r", encoding="utf-8") as f:
            base_val = [json.loads(line) for line in f if line.strip()]
        val_rows.extend(base_val)
        print(f"  [Val 1/3] Base Val: {len(base_val):,} rows")

    # 3b. Staged SDK Val
    sdk_val_file = data_path / "staged" / "sdk_synthetic_val.jsonl"
    if sdk_val_file.exists():
        with open(sdk_val_file, "r", encoding="utf-8") as f:
            sdk_val = [json.loads(line) for line in f if line.strip()]
        val_rows.extend(sdk_val)
        print(f"  [Val 2/3] SDK Consensus Val: {len(sdk_val):,} rows")

    # 3c. Typed Decisions Val
    print(f"  [Val 3/3] Loading Typed Decisions Val ({typed_cases_val:,} cases)...")
    try:
        td_val = convert_typed_decisions_to_nli(
            split="validation", max_cases=typed_cases_val, seed=seed + 1
        )
        val_rows.extend(td_val)
        print(f"        Loaded {len(td_val):,} pairs from Typed Decisions Val")
    except Exception as e:
        print(f"        Failed to load Typed Decisions val: {e}")
        td_val = []

    # 4. Strict Contamination & Deduplication Filtering
    print("\nFiltering and deduplicating...")
    clean_train: List[Dict[str, Any]] = []
    seen_train_pairs: Set[Tuple[str, str]] = set()
    n_test_filtered_train = 0
    n_internal_dup_train = 0

    for r in train_rows:
        pair = _pair_key(r)
        # Contamination check: exact pair or premise in test
        if pair in test_pairs or (r.get("source") != "typed_decisions_synth" and r.get("premise", "").strip() in test_premises):
            n_test_filtered_train += 1
            continue
        if pair in seen_train_pairs:
            n_internal_dup_train += 1
            continue
        seen_train_pairs.add(pair)
        clean_train.append(r)

    clean_val: List[Dict[str, Any]] = []
    seen_val_pairs: Set[Tuple[str, str]] = set()
    n_test_filtered_val = 0
    n_train_leak_val = 0
    n_internal_dup_val = 0

    for r in val_rows:
        pair = _pair_key(r)
        if pair in test_pairs or (r.get("source") != "typed_decisions_synth" and r.get("premise", "").strip() in test_premises):
            n_test_filtered_val += 1
            continue
        if pair in seen_train_pairs:
            n_train_leak_val += 1
            continue
        if pair in seen_val_pairs:
            n_internal_dup_val += 1
            continue
        seen_val_pairs.add(pair)
        clean_val.append(r)

    print(f"  Train: {len(train_rows):,} -> {len(clean_train):,} clean "
          f"(dropped {n_test_filtered_train} test overlaps, {n_internal_dup_train} internal dups)")
    print(f"  Val:   {len(val_rows):,} -> {len(clean_val):,} clean "
          f"(dropped {n_test_filtered_val} test overlaps, {n_train_leak_val} train overlaps, {n_internal_dup_val} internal dups)")

    # 5. Deterministic Shuffle
    rng.shuffle(clean_train)
    rng.shuffle(clean_val)

    # 6. Save Stage 2 Artifacts
    out_train = data_path / "stage2_train.jsonl"
    out_val = data_path / "stage2_val.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for r in clean_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for r in clean_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_dist = dict(Counter(r["label"] for r in clean_train))
    val_dist = dict(Counter(r["label"] for r in clean_val))
    train_sources = dict(Counter(r.get("source", "unknown") for r in clean_train))

    manifest = {
        "dataset_name": "gemma4-nli-stage2-flagship",
        "seed": seed,
        "n_train": len(clean_train),
        "n_val": len(clean_val),
        "n_test": len(test_rows),
        "test_sha": test_sha,
        "train_label_distribution": train_dist,
        "val_label_distribution": val_dist,
        "train_sources": train_sources,
        "components": {
            "clean_baseline_train": len(base_train) if base_train_file.exists() else 0,
            "sdk_consensus_synthetic_train": len(sdk_train) if sdk_train_file.exists() else 0,
            "typed_decisions_synthetic_train": len(td_train),
            "clean_baseline_val": len(base_val) if base_val_file.exists() else 0,
            "sdk_consensus_synthetic_val": len(sdk_val) if sdk_val_file.exists() else 0,
            "typed_decisions_synthetic_val": len(td_val),
        },
    }

    out_manifest = data_path / "stage2_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print(f"STAGE 2 COMPILED SUCCESSFULLY:")
    print(f"  Train: {len(clean_train):,} rows -> {out_train}")
    print(f"  Val:   {len(clean_val):,} rows -> {out_val}")
    print(f"  Test:  {len(test_rows):,} rows -> {test_file} (UNTOUCHED)")
    print(f"  Manifest: {out_manifest}")
    print(f"  Train Labels: Contradiction(0)={train_dist.get(0, 0):,}, "
          f"Entailment(1)={train_dist.get(1, 0):,}, Neutral(2)={train_dist.get(2, 0):,}")
    print("=" * 65)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Compile Stage 2 flagship dataset")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--typed-cases-train", type=int, default=2000)
    parser.add_argument("--typed-cases-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compile_stage2(
        data_dir=args.data_dir,
        typed_cases_train=args.typed_cases_train,
        typed_cases_val=args.typed_cases_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
