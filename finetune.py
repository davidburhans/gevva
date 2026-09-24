#!/usr/bin/env python3
"""finetune.py
=============
Turnkey custom data fine-tuning engine for the Gevva System 1 Decision Engine.

Allows users to fine-tune on domain-specific datasets (RAG verification, tool routing,
domain NLI, search reranking) with zero boilerplate.

Features:
- Auto-detects input formats: JSONL, CSV, TSV, Parquet, or JSON.
- Auto-maps flexible column names:
    * Premise: premise, context, document, text_a, passage, prompt, evidence
    * Hypothesis: hypothesis, claim, question, text_b, candidate, response, query
    * Label: label, gold, target, annotation, class, ground_truth
- Auto-normalizes flexible label representations:
    * Contradiction (0): "contradiction", "contradict", "refutes", "refuted", "false", "no", "0"
    * Entailment (1): "entailment", "entails", "supports", "supported", "true", "yes", "1"
    * Neutral (2): "neutral", "unverifiable", "unknown", "not_enough_info", "nei", "2"
- Auto-splits train / validation (stratified) if separate validation file is not provided.
- Continual Fine-Tuning: Can start from raw base (`google/gemma-4-E2B-it`) OR continue
  from pre-trained Gevva checkpoints (`davidburhans/gevva-e2b` or local directory).
- Full Fine-Tuning (`--full-fine-tune`) and LoRA adapter modes.
- CLI and Python API (`finetune_custom_data(...)`).

Usage:
    # Fine-tune starting from base model
    python finetune.py --data my_data.jsonl --out-dir ./my_model

    # Continually fine-tune starting from Gevva e2b champion on HuggingFace
    python finetune.py --data my_data.csv --base-model davidburhans/gevva-e2b --full-fine-tune --epochs 2

    # Python API
    from finetune import finetune_custom_data
    finetune_custom_data("my_data.jsonl", output_dir="./my_model")
"""

from __future__ import annotations

import argparse
import os

# WHY: match train_cross_encoder's allocator config; long grouped batches at
# 2048 tokens fragment without expandable segments (review F10).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    ID2LABEL,
    LABEL2ID,
    DEFAULT_NLI_TEMPLATE,
    Gemma4ForSequenceClassification,
    apply_quantization_aware_training,
    tokenize_nli_pair_safe,
)
from research.adapters.grouped_decision_collator import (
    GroupedDecisionCollator,
    GroupedTokenBucketBatchSampler,
    compute_cross_option_loss,
    compute_served_distribution_loss,
)

# -----------------------------------------------------------------------------
# Column & Label Normalization Constants
# -----------------------------------------------------------------------------
PREMISE_CANDIDATE_KEYS = [
    "premise", "context", "document", "passage", "evidence", "text_a",
    "prompt", "source_text", "background", "reference"
]
HYPOTHESIS_CANDIDATE_KEYS = [
    "hypothesis", "claim", "assertion", "candidate", "response", "text_b",
    "query", "target_text", "statement", "answer"
]
LABEL_CANDIDATE_KEYS = [
    "label", "gold", "target", "annotation", "class", "ground_truth", "verdict"
]

LABEL_STRING_MAP = {
    # Contradiction (0)
    "contradiction": CONTRADICTION,
    "contradict": CONTRADICTION,
    "contradicts": CONTRADICTION,
    "contradictory": CONTRADICTION,
    "refutes": CONTRADICTION,
    "refuted": CONTRADICTION,
    "false": CONTRADICTION,
    "no": CONTRADICTION,
    "negative": CONTRADICTION,
    "0": CONTRADICTION,
    0: CONTRADICTION,
    # Entailment (1)
    "entailment": ENTAILMENT,
    "entails": ENTAILMENT,
    "entail": ENTAILMENT,
    "supports": ENTAILMENT,
    "supported": ENTAILMENT,
    "true": ENTAILMENT,
    "yes": ENTAILMENT,
    "positive": ENTAILMENT,
    "1": ENTAILMENT,
    1: ENTAILMENT,
    # Neutral (2)
    "neutral": NEUTRAL,
    "neutrality": NEUTRAL,
    "unverifiable": NEUTRAL,
    "unknown": NEUTRAL,
    "not_enough_info": NEUTRAL,
    "not enough info": NEUTRAL,
    "nei": NEUTRAL,
    "neither": NEUTRAL,
    "2": NEUTRAL,
    2: NEUTRAL,
}


# -----------------------------------------------------------------------------
# Data Ingestion & Auto-Detection
# -----------------------------------------------------------------------------
def detect_columns(sample_dict: Dict[str, Any]) -> Tuple[str, str, str]:
    """Finds premise, hypothesis, and label keys from candidate names."""
    premise_key = None
    hyp_key = None
    label_key = None

    lower_keys = {k.lower(): k for k in sample_dict.keys()}

    for cand in PREMISE_CANDIDATE_KEYS:
        if cand in lower_keys:
            premise_key = lower_keys[cand]
            break

    for cand in HYPOTHESIS_CANDIDATE_KEYS:
        if cand in lower_keys:
            hyp_key = lower_keys[cand]
            break

    for cand in LABEL_CANDIDATE_KEYS:
        if cand in lower_keys:
            label_key = lower_keys[cand]
            break

    if not premise_key:
        raise ValueError(
            f"Could not identify a premise/context column in fields: {list(sample_dict.keys())}. "
            f"Expected one of {PREMISE_CANDIDATE_KEYS}"
        )
    if not hyp_key:
        raise ValueError(
            f"Could not identify a hypothesis/claim column in fields: {list(sample_dict.keys())}. "
            f"Expected one of {HYPOTHESIS_CANDIDATE_KEYS}"
        )
    if not label_key:
        raise ValueError(
            f"Could not identify a label column in fields: {list(sample_dict.keys())}. "
            f"Expected one of {LABEL_CANDIDATE_KEYS}"
        )

    return premise_key, hyp_key, label_key


def parse_raw_label(raw_val: Any, convention: str = "ours") -> Optional[int]:
    """Maps raw label (string, int, or float) to 0, 1, or 2.

    Conventions:
    - 'ours' (default): 0=contradiction, 1=entailment, 2=neutral (OpenJEV / ModernCE convention).
    - 'native': 0=entailment, 1=neutral, 2=contradiction (Stanford SNLI / NYU MNLI convention).
    """
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, np.integer)):
        iv = int(raw_val)
        if iv in (0, 1, 2):
            if convention == "native":
                return {0: 1, 1: 2, 2: 0}[iv]
            return iv
    str_val = str(raw_val).strip().lower()
    return LABEL_STRING_MAP.get(str_val, None)


def load_custom_file(file_path: str, convention: str = "ours") -> List[Dict[str, Any]]:
    """Loads records from JSONL, CSV, TSV, Parquet, or JSON file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Custom data file does not exist: {file_path}")

    suffix = p.suffix.lower()
    rows = []

    if suffix in (".jsonl", ".ndjson"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    elif suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "data" in data:
                rows = data["data"]
            else:
                raise ValueError("JSON file must be a list of records or contain a 'data' list.")

    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for r in reader:
                rows.append(dict(r))

    elif suffix in (".parquet", ".pq"):
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            rows = df.to_dict(orient="records")
        except ImportError:
            raise ImportError(f"Reading parquet files requires 'pandas' and 'pyarrow'. Please install them or provide .jsonl/.csv.")

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Please provide .jsonl, .csv, .tsv, .parquet, or .json")

    if not rows:
        raise ValueError(f"File {file_path} contained 0 rows.")

    # Detect keys and normalize
    p_key, h_key, l_key = detect_columns(rows[0])
    print(f"Loaded {len(rows):,} raw rows from {file_path}")
    print(f"  Detected mapping: Premise='{p_key}', Hypothesis='{h_key}', Label='{l_key}' (convention: {convention})")

    normalized_data = []
    skipped_labels = Counter()

    for idx, r in enumerate(rows):
        premise = str(r.get(p_key, "")).strip()
        hypothesis = str(r.get(h_key, "")).strip()
        raw_label = r.get(l_key)
        norm_label = parse_raw_label(raw_label, convention=convention)

        if not premise or not hypothesis:
            continue
        if norm_label is None:
            skipped_labels[str(raw_label)] += 1
            continue

        item = {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": norm_label,
            "source": r.get("source", "custom_dataset"),
        }
        if "image" in r and r["image"]:
            item["image"] = r["image"]
        if "group_id" in r and r["group_id"] is not None:
            item["group_id"] = str(r["group_id"])
        if "is_gold" in r:
            item["is_gold"] = bool(r["is_gold"])
        if "soft_target" in r:
            item["soft_target"] = float(r["soft_target"])
        elif "teacher_prob" in r:
            item["soft_target"] = float(r["teacher_prob"])
        elif "soft_labels" in r:
            item["soft_labels"] = r["soft_labels"]
        normalized_data.append(item)

    if skipped_labels:
        print(f"  Warning: Skipped {sum(skipped_labels.values())} rows with unmappable labels: {dict(skipped_labels)}")
    print(f"  Valid normalized samples: {len(normalized_data):,}")

    label_counts = Counter(r["label"] for r in normalized_data)
    for l_id, count in sorted(label_counts.items()):
        pct = count / len(normalized_data) * 100
        print(f"    - [{l_id}] {ID2LABEL[l_id]}: {count:,} ({pct:.1f}%)")

    return normalized_data


def stratified_split(
    data: List[Dict[str, Any]], val_ratio: float = 0.15, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits data into train and validation sets while preserving label proportions.
    If items contain group_id, guarantees that all items belonging to the same group_id
    land in the SAME split, preventing train/val data leakage."""
    random.seed(seed)

    # Check if dataset has grouped items. The "-1" sentinel marks un-grouped
    # rows (same convention as the collator/sampler) - treating it as a real
    # group key made ALL anchor rows split all-or-nothing, recreating the
    # zero-neutral-validation bug (adversarial review F05, 2026-09-22).
    def _is_real_group(gid) -> bool:
        return gid is not None and str(gid) not in ("", "-1")

    has_groups = any(_is_real_group(r.get("group_id")) for r in data)
    if has_groups:
        groups = defaultdict(list)
        singletons = []
        for r in data:
            gid = r.get("group_id")
            if _is_real_group(gid):
                groups[str(gid)].append(r)
            else:
                singletons.append(r)

        all_units = list(groups.values()) + [[s] for s in singletons]
        random.shuffle(all_units)

        # WHY: a plain group shuffle can starve minority classes in the val split -
        # the P1 run drew 13,022 val rows with ZERO neutral support because every
        # neutral-bearing group landed in train (2026-09-21 audit). Stratify unit
        # selection by each group's majority label so every class present in the
        # data contributes ~val_ratio of its units (and at least one) to val.
        def _majority_label(unit: List[Dict[str, Any]]) -> Any:
            return Counter(r.get("label") for r in unit).most_common(1)[0][0]

        strata: Dict[Any, List[List[Dict[str, Any]]]] = defaultdict(list)
        for unit in all_units:
            strata[_majority_label(unit)].append(unit)

        val_units: List[List[Dict[str, Any]]] = []
        train_units: List[List[Dict[str, Any]]] = []
        for _, units in sorted(strata.items(), key=lambda kv: str(kv[0])):  # deterministic order
            k_val = max(1, int(round(len(units) * val_ratio)))
            val_units.extend(units[:k_val])
            train_units.extend(units[k_val:])

        train_set = [item for unit in train_units for item in unit]
        val_set = [item for unit in val_units for item in unit]

        random.shuffle(train_set)
        random.shuffle(val_set)
        print(f"Group-aware split: {len(train_units):,} train groups ({len(train_set):,} items), {len(val_units):,} val groups ({len(val_set):,} items).")
        return train_set, val_set

    # Standard pair-level stratified split
    seen = set()
    deduped = []
    dropped_dupes = 0
    for r in data:
        k = (str(r.get("premise", "")).strip(), str(r.get("hypothesis", "")).strip())
        if k in seen:
            dropped_dupes += 1
            continue
        seen.add(k)
        deduped.append(r)
    if dropped_dupes:
        print(f"Stratified split: dropped {dropped_dupes} duplicate rows before splitting to prevent train/val leakage.")

    by_label = defaultdict(list)
    for r in deduped:
        by_label[r["label"]].append(r)

    train_set, val_set = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        if len(items) <= 1:
            train_set.extend(items)
            continue
        n_val = max(1, int(len(items) * val_ratio))
        if n_val >= len(items):
            n_val = len(items) - 1
        val_set.extend(items[:n_val])
        train_set.extend(items[n_val:])

    random.shuffle(train_set)
    random.shuffle(val_set)
    return train_set, val_set


# -----------------------------------------------------------------------------
# PyTorch Dataset & Collator
# -----------------------------------------------------------------------------
class CustomNLIDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class CustomNLICollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        template: str = DEFAULT_NLI_TEMPLATE,
        pad_to_multiple_of: int = 8,
        image_processor: Optional[Any] = None,
        image_root: str = "./data",
        is_gemma: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.pad_to_multiple_of = pad_to_multiple_of
        self.image_root = image_root
        self.tokenizer.padding_side = "right"
        if image_processor is None and is_gemma:
            try:
                self.image_processor = Gemma4ImageProcessorPil()
            except Exception:
                self.image_processor = None
        else:
            self.image_processor = image_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = []
        pixel_values_list = []
        image_pos_ids_list = []

        for r in batch:
            img_field = r.get("image")
            resolved_path = None
            if img_field and self.image_processor is not None:
                p_with_root = os.path.join(self.image_root, img_field)
                if os.path.exists(p_with_root):
                    resolved_path = p_with_root
                elif os.path.exists(img_field):
                    resolved_path = img_field

            if resolved_path is not None:
                img = Image.open(resolved_path).convert("RGB")
                feat = self.image_processor(img, return_tensors="pt")
                n_soft = int(feat["num_soft_tokens_per_image"][0])
                ids = tokenize_nli_pair_safe(
                    tokenizer=self.tokenizer,
                    premise=r["premise"],
                    hypothesis=r["hypothesis"],
                    max_length=self.max_length,
                    image_soft_tokens=n_soft,
                )
                pixel_values_list.append(feat["pixel_values"][0])
                image_pos_ids_list.append(feat["image_position_ids"][0])
            else:
                ids = tokenize_nli_pair_safe(
                    tokenizer=self.tokenizer,
                    premise=r["premise"],
                    hypothesis=r["hypothesis"],
                    max_length=self.max_length,
                    image_soft_tokens=0,
                )
            batch_input_ids.append(ids)

        labels = [r["label"] for r in batch]
        sources = [r.get("source", "custom") for r in batch]

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of > 0 and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_ids = []
        attn_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        res = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sources": sources,
        }
        if pixel_values_list:
            res["pixel_values"] = torch.stack(pixel_values_list)
            res["image_position_ids"] = torch.stack(image_pos_ids_list)

        return res


# -----------------------------------------------------------------------------
# Token-Bucket Batching (Mapika/decider Style)
# -----------------------------------------------------------------------------
# Attribution: Deterministic token-bucket batching inspired by Mapika/decider (Apache 2.0 License).
# Reference: https://github.com/Mapika/decider

class TokenBucketBatchSampler(Sampler[List[int]]):
    """Batches variable-length sequences into discrete token-budget buckets.

    Attribution:
        Deterministic token-bucket batching inspired by Mapika/decider (Apache 2.0 License).
        Constrains ragged sequences into fixed bucket shapes, eliminating CUDA graph recompilations
        and GPU memory fragmentation across sequence lengths up to 128K tokens.
    """
    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int = 4096,
        bucket_boundaries: Optional[List[int]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if bucket_boundaries is None:
            self.bucket_boundaries = [
                64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
            ]
        else:
            self.bucket_boundaries = sorted(bucket_boundaries)

        self.bucket_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            assigned_bucket = self.bucket_boundaries[-1]
            for b in self.bucket_boundaries:
                if b >= length:
                    assigned_bucket = b
                    break
            self.bucket_indices[assigned_bucket].append(idx)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []

        for bucket_cap, indices in self.bucket_indices.items():
            if not indices:
                continue
            cur_indices = list(indices)
            if self.shuffle:
                rng.shuffle(cur_indices)

            batch_size = max(1, self.max_tokens_per_batch // bucket_cap)
            for i in range(0, len(cur_indices), batch_size):
                batches.append(cur_indices[i : i + batch_size])

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for bucket_cap, indices in self.bucket_indices.items():
            if not indices:
                continue
            batch_size = max(1, self.max_tokens_per_batch // bucket_cap)
            total += (len(indices) + batch_size - 1) // batch_size
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SkipPrefixBatchSampler(Sampler[List[int]]):
    """Wraps a deterministic batch sampler and yields only batches after the first `skip`.

    Used for exact mid-epoch training resume: the base sampler (same seed + epoch)
    regenerates the identical batch sequence, and this wrapper replays only the
    not-yet-trained tail without re-running the collator on completed batches.

    Example:
        >>> base = TokenBucketBatchSampler([8, 8, 512], max_tokens_per_batch=512, shuffle=True, seed=0)
        >>> tail = SkipPrefixBatchSampler(base, skip=1)
        >>> len(tail) == len(base) - 1
        True
    """

    def __init__(self, base_sampler: Sampler[List[int]], skip: int):
        if skip < 0:
            raise ValueError(f"skip must be >= 0, got {skip}")
        self.base_sampler = base_sampler
        self.skip = skip

    def __iter__(self):
        iterator = iter(self.base_sampler)
        for _ in range(self.skip):
            next(iterator)
        for batch in iterator:
            yield batch

    def __len__(self) -> int:
        return max(0, len(self.base_sampler) - self.skip)


# -----------------------------------------------------------------------------
# Post-Hoc Validation Temperature Calibration (OpenSourceJev Style)
# -----------------------------------------------------------------------------
# Attribution: Post-hoc validation temperature scaling inspired by
# sabeel111/OpenSourceJev (MIT License) and Guo et al. (2017).
# Reference: https://github.com/sabeel111/OpenSourceJev

def compute_multiclass_ece(probs: np.ndarray, golds: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error over multiclass softmax probabilities."""
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    corrects = (preds == golds).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(golds)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        bin_lo = bin_boundaries[i]
        bin_hi = bin_boundaries[i + 1]
        mask = (confs > bin_lo) & (confs <= bin_hi) if i > 0 else (confs >= bin_lo) & (confs <= bin_hi)
        if np.sum(mask) == 0:
            continue
        bin_acc = np.mean(corrects[mask])
        bin_conf = np.mean(confs[mask])
        ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature_scaling(
    logits: np.ndarray,
    golds: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Fits post-hoc validation temperature scaling T* by minimizing cross-entropy NLL.

    Attribution:
        Post-hoc validation temperature scaling inspired by
        sabeel111/OpenSourceJev (MIT License) and Guo et al. (2017).
        Reference: https://github.com/sabeel111/OpenSourceJev

    Returns:
        (optimal_temperature, ece_before, ece_after, brier_before, brier_after)
    """
    from scipy.optimize import minimize_scalar

    probs_before = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs_before = probs_before / np.sum(probs_before, axis=-1, keepdims=True)
    ece_before = compute_multiclass_ece(probs_before, golds)

    N = len(golds)
    one_hot = np.zeros_like(probs_before)
    one_hot[np.arange(N), golds] = 1.0
    brier_before = float(np.mean(np.sum((probs_before - one_hot) ** 2, axis=-1)))

    def nll_obj(temp: float) -> float:
        t = max(temp, 1e-4)
        scaled_logits = logits / t
        max_z = np.max(scaled_logits, axis=-1, keepdims=True)
        log_denom = max_z.squeeze(-1) + np.log(np.sum(np.exp(scaled_logits - max_z), axis=-1))
        gold_logits = scaled_logits[np.arange(N), golds]
        nll = np.mean(log_denom - gold_logits)
        return float(nll)

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    t_opt = float(res.x) if res.success else 1.0

    scaled_logits = logits / max(t_opt, 1e-4)
    probs_after = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probs_after = probs_after / np.sum(probs_after, axis=-1, keepdims=True)
    ece_after = compute_multiclass_ece(probs_after, golds)
    brier_after = float(np.mean(np.sum((probs_after - one_hot) ** 2, axis=-1)))

    return t_opt, ece_before, ece_after, brier_before, brier_after



def compute_brier_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes multi-class Brier calibration loss.

    Formula:
        probs = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
        brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))

    Attribution:
        Proper-scoring Brier calibration loss inspired by von-1.0, research report 08,
        and Brier (1950).
    """
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))


# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def should_save_best(
    dec_acc, anchor_acc, best_dec_acc, best_anchor_acc, tol: float = 0.02
) -> bool:
    """G2 selection rule (regression review 2026-09-22): decision accuracy is the
    primary metric ONLY within an anchor-accuracy floor - an epoch that improves
    served decisions while regressing classic NLI behavior must not ship.

    Example:
        >>> should_save_best(0.80, 0.90, 0.78, 0.92)   # NLI within 2pp: save
        True
        >>> should_save_best(0.82, 0.85, 0.78, 0.92)   # NLI -7pp: reject
        False
    """
    primary = dec_acc if dec_acc is not None else anchor_acc
    best_primary = best_dec_acc if dec_acc is not None else best_anchor_acc
    if primary is None or best_primary is None:
        return True  # nothing comparable yet (first epoch always saves)
    if primary <= best_primary:
        return False
    if anchor_acc is not None and best_anchor_acc is not None:
        if anchor_acc < best_anchor_acc - tol:
            return False  # served win bought with an NLI regression: reject
    return True


@torch.no_grad()
def evaluate_dataset(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds, all_probs, all_logits, all_golds, all_sources = [], [], [], [], []
    all_group_ids, all_is_gold = [], []
    gid_offset = 0  # batch-local group ids -> globally unique (review round-2 N8)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        sources = batch["sources"]
        pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
        image_position_ids = batch["image_position_ids"].to(device) if "image_position_ids" in batch else None

        if "group_ids" in batch and batch["group_ids"] is not None:
            # Review round-2 N8: the collator remaps group ids to BATCH-LOCAL ints
            # (0..G-1 per __call__), so concatenating batches makes every batch's
            # local-id-0 group merge into one mega-group. Offset them to be
            # globally unique across the evaluation pass; -1 sentinels pass through.
            gids = batch["group_ids"].cpu().numpy().copy()
            pos = gids >= 0
            if pos.any():
                local_max = int(gids[pos].max())
                gids[pos] += gid_offset
                gid_offset += local_max + 1
            all_group_ids.append(gids)
            if "is_gold" in batch and batch["is_gold"] is not None:
                all_is_gold.append(batch["is_gold"].cpu().numpy())
            else:
                all_is_gold.append(np.zeros(len(labels)))

        amp_device = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
            )
            logits = outputs.logits.float().cpu().numpy()
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        all_logits.append(logits)
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=-1))
        all_golds.append(labels.numpy())
        all_sources.extend(sources)

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 3))
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 3))
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,))
    golds = np.concatenate(all_golds, axis=0) if all_golds else np.empty((0,))

    acc = float(np.mean(preds == golds)) if len(golds) > 0 else 0.0

    # Decision accuracy on the SERVED artifact: group argmax over renormalized
    # P(entailment) - the exact distribution clients receive (review F07: the old
    # z_ent - z_con margin metric selects checkpoints for a different ranking).
    decision_acc = None
    if all_group_ids:
        cat_gids = np.concatenate(all_group_ids, axis=0)
        cat_is_gold = np.concatenate(all_is_gold, axis=0)
        valid_g = cat_gids >= 0
        if np.any(valid_g):
            p_ent_eval = probs[:, ENTAILMENT]
            u_gids = np.unique(cat_gids[valid_g])
            corr_dec = 0
            n_dec = 0
            for ug in u_gids:
                u_idx = np.where(cat_gids == ug)[0]
                if len(u_idx) > 1:
                    served = p_ent_eval[u_idx]
                    if served.sum() > 0:
                        pred_best = u_idx[np.argmax(served)]
                        if cat_is_gold[pred_best] > 0.5:
                            corr_dec += 1
                        n_dec += 1
            if n_dec > 0:
                decision_acc = round(corr_dec / n_dec, 4)

    # Brier score
    N = len(golds)
    if N > 0:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), golds] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
        ece = compute_multiclass_ece(probs, golds)
    else:
        brier = 0.0
        ece = 0.0

    # Per-class metrics
    per_class = {}
    for c_id in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        c_name = ID2LABEL[c_id]
        gold_c = golds == c_id
        pred_c = preds == c_id
        tp = np.sum(gold_c & pred_c)
        fp = np.sum(~gold_c & pred_c)
        fn = np.sum(gold_c & ~pred_c)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(gold_c)),
        }

    res = {
        "accuracy": round(acc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "n_samples": len(golds),
        "per_class": per_class,
        "logits": logits,
        "golds": golds,
    }
    if decision_acc is not None:
        res["decision_accuracy"] = decision_acc
    # G2: anchor rows (ungrouped classic-NLI) accuracy - the regression floor for
    # best-checkpoint selection when decision_accuracy is primary.
    if all_group_ids:
        cat_gids_a = np.concatenate(all_group_ids, axis=0)
        anchor_mask = cat_gids_a == -1
        if anchor_mask.any():
            res["anchor_accuracy"] = round(float(np.mean(preds[anchor_mask] == golds[anchor_mask])), 4)
    return res


# -----------------------------------------------------------------------------
# Core Fine-Tuning Routine
# -----------------------------------------------------------------------------
def finetune_custom_data(
    train_data: Union[str, List[Dict[str, Any]]],
    val_data: Optional[Union[str, List[Dict[str, Any]]]] = None,
    output_dir: str = "./ckpt/custom_model",
    base_model_id: str = "google/gemma-4-E2B",
    adapter_path: Optional[str] = None,
    head_weights: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
    lora_r: int = 64,
    lora_alpha: int = 128,
    label_smoothing: float = 0.05,
    brier_weight: float = 0.0,
    cross_option_weight: float = 1.0,
    nli_aux_weight: float = 0.15,
    served_dist_weight: float = 0.0,
    decision_temp: float = 1.0,
    val_ratio: float = 0.15,
    start_epoch: int = 0,
    full_fine_tune: bool = False,
    qat: bool = False,  # A8: QAT must be opt-in; silent nvfp4 simulation corrupted train/serve parity
    target_quant: str = "w4a16",  # A8: default matches export_w4a16.py (INT4 group-32 compressed-tensors)
    qat_bits: int = 4,
    qat_group_size: int = 32,
    use_token_bucketing: bool = False,
    max_tokens_per_batch: int = 2048,
    image_root: str = "./data",
    image_processor: Optional[Any] = None,
    label_convention: str = "ours",
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """High-level Python API to fine-tune Gemma 4 Cross-Encoder on custom data.

    Args:
        train_data: File path (.jsonl, .csv, .tsv, .parquet, .json) or in-memory list of dicts.
        val_data: Optional separate validation file or list. If None, auto-splits train_data.
        output_dir: Directory to save best checkpoint, adapters, and eval reports.
        base_model_id: HuggingFace foundation model ID (default: google/gemma-4-E2B).
        adapter_path: Optional pre-trained LoRA adapter to resume / continue training from.
        epochs: Number of fine-tuning epochs.
        batch_size: Per-device batch size.
        grad_accum: Number of gradient accumulation steps (effective batch size = batch_size * grad_accum).
        lr: Peak learning rate.
        max_length: Maximum sequence token length.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        label_smoothing: Label smoothing coefficient for calibration.
        brier_weight: Weight for multi-class Brier calibration loss (default: 0.0, recommended: 0.5).
        val_ratio: Validation split fraction when val_data is not provided.
        label_convention: 'ours' (0=contradiction, 1=entailment, 2=neutral) or 'native' (0=entailment, 1=neutral, 2=contradiction).
        seed: Random seed.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        Dictionary containing training and final evaluation metrics.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Gemma 4 NLI Cross-Encoder: Custom Data Fine-Tuning")
    print("=" * 65)
    print(f"Base Model:       {base_model_id}")
    if adapter_path:
        print(f"Starting Adapter: {adapter_path} (Continual Fine-Tuning)")
    print(f"Output Directory: {output_dir}")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Ingest Data
    if isinstance(train_data, str):
        raw_records = load_custom_file(train_data, convention=label_convention)
    else:
        raw_records = train_data

    if val_data is not None:
        if isinstance(val_data, str):
            val_records = load_custom_file(val_data, convention=label_convention)
        else:
            val_records = val_data
        train_records = raw_records
    else:
        print(f"Auto-splitting records into train ({1-val_ratio:.0%}) and validation ({val_ratio:.0%})...")
        train_records, val_records = stratified_split(raw_records, val_ratio=val_ratio, seed=seed)

    print(f"Dataset ready: {len(train_records):,} train samples, {len(val_records):,} validation samples.")

    # 2. Tokenizer
    tokenizer_source = adapter_path if adapter_path and os.path.exists(adapter_path) else base_model_id
    print(f"Loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Architecture
    print(f"Initializing base architecture {base_model_id}...")
    config = AutoConfig.from_pretrained(base_model_id)
    config.num_labels = 3
    base_model = Gemma4ForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    base_model.freeze_vision_tower(freeze_adapter=False)

    def _get_raw_model(m):
        if hasattr(m, "peft_config") or type(m).__name__ == "PeftModel":
            if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                return m.base_model.model
        if hasattr(m, "score"):
            return m
        return m

    def _load_into_module(mod, state):
        if not isinstance(state, dict):
            w = state
        elif "weight" in state:
            w = state["weight"]
        elif "modules_to_save.default.weight" in state:
            w = state["modules_to_save.default.weight"]
        elif "default.weight" in state:
            w = state["default.weight"]
        elif "original_module.weight" in state:
            w = state["original_module.weight"]
        else:
            w = next(iter(state.values()))
        if hasattr(mod, "modules_to_save") and "default" in mod.modules_to_save:
            mod.modules_to_save["default"].weight.data.copy_(w)
            if hasattr(mod, "original_module") and hasattr(mod.original_module, "weight"):
                mod.original_module.weight.data.copy_(w)
        elif hasattr(mod, "weight"):
            mod.weight.data.copy_(w)
        else:
            mod.load_state_dict({"weight": w}, strict=False)

    if full_fine_tune:
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading and merging existing LoRA weights from {adapter_path} for full fine-tuning...")
            peft_m = PeftModel.from_pretrained(base_model, adapter_path)
            head_weights_path = os.path.join(adapter_path, "head_weights.pt")
            if os.path.exists(head_weights_path):
                hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
                raw = _get_raw_model(peft_m)
                if "score" in hw and hasattr(raw, "score"):
                    _load_into_module(raw.score, hw["score"])
                if "norm" in hw and hasattr(raw, "norm"):
                    _load_into_module(raw.norm, hw["norm"])
            model = peft_m.merge_and_unload()
        else:
            model = base_model
            head_weights_path = head_weights if (head_weights and os.path.exists(head_weights)) else os.path.join(base_model_id, "head_weights.pt")
            if os.path.exists(head_weights_path):
                print(f"Loading head weights from {head_weights_path}...")
                hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
                raw = _get_raw_model(model)
                if "score" in hw and hasattr(raw, "score"):
                    _load_into_module(raw.score, hw["score"])
                if "norm" in hw and hasattr(raw, "norm"):
                    _load_into_module(raw.norm, hw["norm"])

        # Unfreeze all parameters first
        for p in model.parameters():
            p.requires_grad = True

        # Freeze non-transformer components (vision tower, audio tower, and massive token/per-layer embedding tables)
        if hasattr(model, "freeze_vision_tower"):
            model.freeze_vision_tower(freeze_adapter=False)
        if hasattr(model, "model") and hasattr(model.model, "audio_tower"):
            for p in model.model.audio_tower.parameters():
                p.requires_grad = False
        lm = getattr(model, "model", model)
        if hasattr(lm, "language_model"):
            lm = lm.language_model
        if hasattr(lm, "embed_tokens") and hasattr(lm.embed_tokens, "weight"):
            lm.embed_tokens.weight.requires_grad = False
        if hasattr(lm, "embed_tokens_per_layer") and hasattr(lm.embed_tokens_per_layer, "weight"):
            lm.embed_tokens_per_layer.weight.requires_grad = False

        trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_cnt = sum(p.numel() for p in model.parameters())
        print(f"Full Fine-Tuning mode: Unfrozen {trainable_cnt:,} / {total_cnt:,} parameters ({trainable_cnt/total_cnt*100:.2f}%) across all transformer layers.")
    elif adapter_path and os.path.exists(adapter_path):
        print(f"Loading existing LoRA weights from {adapter_path} for continual fine-tuning...")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        # WHY (Phase-1 OOM, 2026-09-22): PEFT's from_pretrained(is_trainable=True)
        # re-enables requires_grad on base params, silently un-freezing the vision
        # tower that was frozen above - a batch of 8 multimodal rows then spiked
        # +18 GiB (tower forward with grad) and OOM'd training. Re-freeze AFTER the
        # wrap, matching train_cross_encoder's working order. The FrozenVision
        # model subclass additionally enforces no_grad/eval at call time.
        model.base_model.model.freeze_vision_tower(freeze_adapter=False)
        head_weights_path = os.path.join(adapter_path, "head_weights.pt")
        if os.path.exists(head_weights_path):
            print(f"Restoring classification head from {head_weights_path}...")
            hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
            raw = _get_raw_model(model)

            if "score" in hw and hasattr(raw, "score"):
                _load_into_module(raw.score, hw["score"])
            if "norm" in hw and hasattr(raw, "norm"):
                _load_into_module(raw.norm, hw["norm"])
    else:
        print(f"Configuring new LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=r".*language_model.*(q|k|v|o|gate|up|down)_proj",
            modules_to_save=["score", "norm"],
            lora_dropout=0.05,
            bias="none",
            task_type=None,
        )
        model = get_peft_model(base_model, lora_cfg)

    if qat:
        from gemma4_cross_encoder import apply_quantization_aware_training
        print(f"Applying Quantization-Aware Training (QAT): Format='{target_quant}' (group_size={qat_group_size})...")
        model = apply_quantization_aware_training(
            model,
            quant_format=target_quant,
            num_bits=qat_bits,
            group_size=qat_group_size,
        )

    def _get_lm(m):
        curr = m
        for _ in range(4):
            if hasattr(curr, "language_model"):
                return curr.language_model
            if hasattr(curr, "model") and hasattr(curr.model, "language_model"):
                return curr.model.language_model
            if hasattr(curr, "base_model"):
                curr = curr.base_model
            else:
                break
        return getattr(curr, "language_model", None)

    # Enable gradient checkpointing to safely fit in RTX 5090 VRAM
    raw_lm = _get_lm(model)
    if raw_lm is not None and hasattr(raw_lm, "gradient_checkpointing_enable"):
        raw_lm.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Enabled gradient checkpointing on language model backbone.")

    model.to(device)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {tr:,} || all params: {tot:,} || trainable%: {tr/tot*100:.4f}")

    # 4. DataLoaders
    is_gemma = "gemma" in base_model_id.lower() or "gevva" in base_model_id.lower()
    if image_processor is None and is_gemma:
        try:
            image_processor = Gemma4ImageProcessorPil()
        except Exception:
            image_processor = None

    has_groups = any(r.get("group_id") is not None for r in train_records)
    has_val_groups = any(r.get("group_id") is not None for r in val_records)

    if has_groups or has_val_groups:
        collator = GroupedDecisionCollator(
            tokenizer,
            max_length=max_length,
            image_processor=image_processor,
            image_root=image_root,
        )
    else:
        collator = CustomNLICollator(
            tokenizer,
            max_length=max_length,
            image_processor=image_processor,
            image_root=image_root,
            is_gemma=is_gemma,
        )

    use_pin = (device != "cpu" and torch.cuda.is_available())

    def _token_lengths(records):
        """Tokenizer-accurate capped lengths; the chars//4 heuristic undercounts
        dense synthetic text 2-4x and broke the token budget (stage-3 OOM, 2026-09-21)."""
        premises = [str(r.get("premise", "")) for r in records]
        hypotheses = [str(r.get("hypothesis", "")) for r in records]
        n_p = [len(x) for x in tokenizer(premises, add_special_tokens=False)["input_ids"]]
        n_h = [len(x) for x in tokenizer(hypotheses, add_special_tokens=False)["input_ids"]]
        return [min(max_length, 32 + p + h) for p, h in zip(n_p, n_h)]

    if use_token_bucketing:
        train_lengths = _token_lengths(train_records)
        if has_groups:
            print(f"Using Grouped Token-Bucket Batching (P1 cross-option, max_tokens_per_batch={max_tokens_per_batch})...")
            train_gids = [r.get("group_id") for r in train_records]
            train_sampler = GroupedTokenBucketBatchSampler(
                train_lengths, group_ids=train_gids, max_tokens_per_batch=max_tokens_per_batch, shuffle=True, seed=seed
            )
        else:
            print(f"Using Deterministic Token-Bucket Batching (max_tokens_per_batch={max_tokens_per_batch})...")
            train_sampler = TokenBucketBatchSampler(
                train_lengths, max_tokens_per_batch=max_tokens_per_batch, shuffle=True, seed=seed
            )
        train_loader = DataLoader(
            CustomNLIDataset(train_records),
            batch_sampler=train_sampler,
            collate_fn=collator,
            pin_memory=use_pin,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            CustomNLIDataset(train_records),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=use_pin,
        )
    val_records_dataset = CustomNLIDataset(val_records)
    if use_token_bucketing and has_val_groups:
        # Review F07: val groups must stay batch-atomic or the decision metric
        # (and best-checkpoint selection) scores partial option sets.
        # WHY 2048 (not max_tokens_per_batch=4096): validation is inference-only,
        # so it doesn't need the training budget. A smaller cap leaves more VRAM
        # headroom for the allocator's cached pool, and group atomicity is
        # preserved (a group's own footprint still overrides the budget).
        val_sampler = GroupedTokenBucketBatchSampler(
            _token_lengths(val_records),
            group_ids=[r.get("group_id") for r in val_records],
            max_tokens_per_batch=min(2048, max_tokens_per_batch),
            shuffle=False,
        )
        val_loader = DataLoader(
            val_records_dataset,
            batch_sampler=val_sampler,
            collate_fn=collator,
            pin_memory=use_pin,
        )
    else:
        val_loader = DataLoader(
            val_records_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            pin_memory=use_pin,
        )

    # 5. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = max(5, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 6. Training Loop
    print(f"\nBeginning training: {epochs} epoch(s) (starting at epoch {start_epoch+1}), {len(train_loader)} batches/epoch, {total_steps} optimizer updates")
    best_val_acc = 0.0
    best_dec_acc = None  # G2: primary metric (served decision accuracy)
    best_anchor_acc = None  # G2: regression floor (classic-NLI anchor accuracy)
    prev_report_path = os.path.join(output_dir, "best", "eval_report.json")
    if os.path.exists(prev_report_path):
        try:
            with open(prev_report_path) as f:
                prev_rep = json.load(f)
            best_dec_acc = prev_rep.get("decision_accuracy")
            best_anchor_acc = prev_rep.get("anchor_accuracy")
            best_val_acc = best_dec_acc if best_dec_acc is not None else prev_rep.get("accuracy", 0.0)
            print(f"Loaded existing best baseline from {prev_report_path}: Decision Acc = {best_dec_acc*100 if best_dec_acc is not None else 'N/A'}%")
        except Exception as e:
            print(f"Warning: could not load existing best eval report: {e}")
    best_metrics = {}
    global_step = 0
    t_start = time.time()

    for epoch_idx in range(epochs):
        epoch = start_epoch + epoch_idx
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_xopt_loss = 0.0
        epoch_brier_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
            image_position_ids = batch["image_position_ids"].to(device) if "image_position_ids" in batch else None

            amp_device = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                )
                ce_loss = loss_fn(outputs.logits, labels)

                # Grouped losses. Two independent objectives on grouped items:
                # (a) served-distribution loss (Phase 1): CE on the EXACT served
                #     artifact (renormalized P(entailment)) - train-serving parity;
                # (b) cross-option margin loss (P1): softmax CE over z_ent - z_con.
                # Weights are independent; the NLI aux CE applies whenever ANY
                # grouped term is active (previously it silently vanished when
                # cross_option_weight == 0 - adversarial review F03, 2026-09-22).
                served_loss, served_meta = None, {"served_groups": 0}
                xopt_loss, xopt_meta = None, {"n_groups": 0}
                if "group_ids" in batch:
                    group_ids = batch["group_ids"].to(device)
                    is_gold = batch["is_gold"].to(device)
                    soft_targets = batch.get("soft_targets")
                    if soft_targets is not None:
                        soft_targets = soft_targets.to(device)
                    if served_dist_weight > 0.0:
                        served_loss, served_meta = compute_served_distribution_loss(
                            logits=outputs.logits,
                            group_ids=group_ids,
                            is_gold=is_gold,
                            soft_targets=soft_targets,
                        )
                    if cross_option_weight > 0.0:
                        scores = outputs.logits[:, ENTAILMENT] - outputs.logits[:, CONTRADICTION]
                        xopt_loss, xopt_meta = compute_cross_option_loss(
                            scores=scores,
                            group_ids=group_ids,
                            is_gold=is_gold,
                            soft_targets=soft_targets,
                            temperature=decision_temp,
                        )

                grouped_terms = []
                if served_loss is not None and served_meta.get("served_groups", 0) > 0:
                    grouped_terms.append(served_dist_weight * served_loss)
                if xopt_loss is not None and xopt_meta.get("n_groups", 0) > 0:
                    grouped_terms.append(cross_option_weight * xopt_loss)
                if grouped_terms:
                    total_loss = sum(grouped_terms) + nli_aux_weight * ce_loss
                else:
                    total_loss = ce_loss

                if brier_weight > 0.0:
                    probs = torch.softmax(outputs.logits, dim=-1)
                    one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
                    brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
                    total_loss = total_loss + brier_weight * brier_loss
                else:
                    brier_loss = None
                loss = total_loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            if xopt_loss is not None:
                epoch_xopt_loss += xopt_loss.item()
            if brier_loss is not None:
                epoch_brier_loss += brier_loss.item()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % max(1, (total_steps // 10)) == 0:
                    curr_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / (step + 1)
                    speed = (step + 1) * batch_size / (time.time() - t0)
                    extra_msg = ""
                    if has_groups and cross_option_weight > 0.0:
                        extra_msg += f" | XOpt: {epoch_xopt_loss / (step + 1):.4f}"
                    if brier_weight > 0.0:
                        extra_msg += f" | Brier: {epoch_brier_loss / (step + 1):.4f}"
                    print(
                        f"Epoch {epoch+1}/{start_epoch+epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f}{extra_msg} | LR: {curr_lr:.2e} | Speed: {speed:.1f} samples/s",
                        flush=True,
                    )

        # Validation at epoch end
        # WHY empty_cache (Phase-1 OOM, 2026-09-22): the training loop's caching
        # allocator keeps its peak (~29.6 GiB) RESERVED-but-cached when the epoch
        # ends. Validation runs inference-only (no_grad), so it needs headroom
        # the allocator refuses to give back without an explicit release - the
        # 2nd OOM died exactly at Epoch-1 validation with 27.35 GiB live tensors
        # and only 127 MiB free. The model can't be unloaded (validation needs
        # it), but empty_cache() returns the cached pool to the OS first.
        torch.cuda.empty_cache()
        print(f"\n--- Validation (Epoch {epoch+1}/{start_epoch+epochs}) ---", flush=True)
        val_metrics = evaluate_dataset(model, val_loader, device)
        acc = val_metrics["accuracy"]
        dec_acc = val_metrics.get("decision_accuracy")
        dec_msg = f" | Decision Acc: {dec_acc*100:.2f}%" if dec_acc is not None else ""
        print(f"Accuracy: {acc*100:.2f}%{dec_msg} | Brier Score: {val_metrics['brier']:.4f}")
        for cls_name, m in val_metrics["per_class"].items():
            print(f"  [{cls_name:13s}] Prec: {m['precision']:.3f} | Rec: {m['recall']:.3f} | F1: {m['f1']:.3f} (n={m['support']})")

        # G2 (regression review 2026-09-22): snapshot EVERY epoch so a bad selection
        # can never destroy the good epoch's weights; best/ selection uses dec_acc
        # as primary ONLY within an anchor-accuracy floor (should_save_best).
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)

        raw_model_epoch = _get_raw_model(model)
        def _clean_state_ep(mod):
            if hasattr(mod, "modules_to_save") and "default" in mod.modules_to_save:
                return mod.modules_to_save["default"].state_dict()
            return mod.state_dict()
        _head_epoch = {"score": _clean_state_ep(raw_model_epoch.score)}
        if hasattr(raw_model_epoch, "norm"):
            _head_epoch["norm"] = _clean_state_ep(raw_model_epoch.norm)
        torch.save(_head_epoch, os.path.join(epoch_dir, "head_weights.pt"))
        clean_epoch_metrics = {k: v for k, v in val_metrics.items() if k not in ("logits", "golds")}
        with open(os.path.join(epoch_dir, "eval_report.json"), "w") as f:
            json.dump(clean_epoch_metrics, f, indent=2)
        print(f"-> Epoch {epoch+1} snapshot saved to {epoch_dir}")

        anchor_acc = val_metrics.get("anchor_accuracy")
        if should_save_best(dec_acc, anchor_acc, best_dec_acc, best_anchor_acc):
            best_dec_acc = dec_acc if dec_acc is not None else anchor_acc
            best_anchor_acc = anchor_acc if anchor_acc is not None else (best_anchor_acc or 0.0)
            best_metrics = val_metrics
            best_dir = os.path.join(output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            metric_name = "decision accuracy" if dec_acc is not None else "accuracy"
            best_val_acc = (dec_acc if dec_acc is not None else anchor_acc) or best_val_acc
            print(f"-> New best {metric_name} ({best_val_acc*100:.2f}%)! Saving checkpoint to {best_dir}...")
            model.save_pretrained(best_dir)
            # Explicitly save classification head and norm weights (unwrapping any PEFT wrappers)
            raw_model = _get_raw_model(model)
            def clean_state(mod):
                if hasattr(mod, "modules_to_save") and "default" in mod.modules_to_save:
                    return mod.modules_to_save["default"].state_dict()
                return mod.state_dict()

            head_dict = {
                "score": clean_state(raw_model.score),
            }
            if hasattr(raw_model, "norm"):
                head_dict["norm"] = clean_state(raw_model.norm)
            torch.save(head_dict, os.path.join(best_dir, "head_weights.pt"))
            tokenizer.save_pretrained(best_dir)

            # Fit and save post-hoc validation temperature scaling T*
            if "logits" in val_metrics and "golds" in val_metrics and len(val_metrics["golds"]) > 0:
                t_opt, ece_bef, ece_aft, br_bef, br_aft = fit_temperature_scaling(
                    val_metrics["logits"], val_metrics["golds"]
                )
                # G8 (regression review 2026-09-22): ship T*=1.0. The NLL fit targets the
                # 3-class distribution on an E/C-dominated slice, not the served
                # renormalized distribution the gate measures - and v2's fit
                # WORSENED val ECE (0.043->0.052) while P1's degenerate slice fit
                # T*=0.57. The fitted value is preserved as diagnostics; refit
                # offline on served-ECE if the T=1 gate leg lands marginal.
                calib_dict = {
                    "optimal_temperature": 1.0,
                    "fitted_temperature_nll": round(float(t_opt), 4),
                    "val_ece_before": round(float(ece_bef), 4),
                    "val_ece_after": round(float(ece_aft), 4),
                    "val_brier_before": round(float(br_bef), 4),
                    "val_brier_after": round(float(br_aft), 4),
                    "attribution": "Post-hoc validation temperature scaling inspired by sabeel111/OpenSourceJev (MIT License) and Guo et al. (2017)",
                    "ship_note": "optimal_temperature pinned to 1.0 (training temperature, F08 parity); fitted_temperature_nll is diagnostic only",
                }
                with open(os.path.join(best_dir, "calibration.json"), "w") as f:
                    json.dump(calib_dict, f, indent=2)
                print(f"-> Fitted NLL temperature T* = {t_opt:.4f} (diagnostic; shipping T=1.0; ECE fit: {ece_bef:.4f} -> {ece_aft:.4f})")

            report_dict = {k: v for k, v in val_metrics.items() if k not in ("logits", "golds")}
            with open(os.path.join(best_dir, "eval_report.json"), "w") as f:
                json.dump(report_dict, f, indent=2)
            # A8: the trained quantization format must be recoverable from the artifact.
            qat_provenance = {
                "qat_applied": bool(qat),
                "target_quant": target_quant if qat else "none",
                "qat_bits": qat_bits,
                "qat_group_size": qat_group_size,
                "note": "w4a16 = export_w4a16.py INT4 group-32 compressed-tensors layout",
            }
            with open(os.path.join(best_dir, "qat_config.json"), "w") as f:
                json.dump(qat_provenance, f, indent=2)

        torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"Fine-Tuning Finished in {total_time/60:.1f} minutes!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved Checkpoint:        {os.path.join(output_dir, 'best')}")
    print("=" * 65)

    return {
        "best_accuracy": best_val_acc,
        "best_metrics": best_metrics,
        "total_time_seconds": total_time,
        "output_dir": output_dir,
    }


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 Cross-Encoder on custom data.")
    parser.add_argument("--data", required=True, help="Path to custom training dataset (.jsonl, .csv, .tsv, or .json)")
    parser.add_argument("--val-data", default=None, help="Optional separate validation file")
    parser.add_argument("--out-dir", default="./ckpt/custom_finetuned", help="Directory to save fine-tuned model")
    parser.add_argument("--base-model", default="google/gemma-4-E2B", help="Base HuggingFace model path")
    parser.add_argument("--adapter", default=None, help="Optional path to existing LoRA adapter checkpoint to resume from")
    parser.add_argument("--head-weights", default=None, help="Optional path to existing head_weights.pt to warm-start classification head")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio if no val-data is provided")
    parser.add_argument("--qat", action="store_true", default=False,
                        help="Enable Quantization-Aware Training (QAT) - opt-in (audit A8; was silently on)")
    parser.add_argument("--no-qat", action="store_false", dest="qat", help="Disable QAT and train in full precision")
    parser.add_argument(
        "--target-quant",
        default="w4a16",
        choices=["nvfp4", "w4a16", "q4_k_m"],
        help="Target quantization format for QAT (default w4a16 = the export_w4a16.py production format; "
             "nvfp4 for native Blackwell FP4; q4_k_m for GGUF)",
    )
    parser.add_argument("--qat-bits", type=int, default=4, help="QAT weight bit-width (default: 4)")
    parser.add_argument("--qat-group-size", type=int, default=32, help="QAT group size (default: 32)")
    parser.add_argument("--token-bucketing", action="store_true", help="Enable deterministic token-bucket batching (decider style)")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048, help="Token budget per batch when using token bucketing")
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=0.0,
        help="Weight for multi-class Brier calibration loss (default: 0.0, recommended: 0.5 based on von-1.0 and research report 08)",
    )
    parser.add_argument(
        "--label-convention",
        default="ours",
        choices=["ours", "native"],
        help="Label convention for integer labels: 'ours' (0=contradiction, 1=entailment, 2=neutral) or 'native' (0=entailment, 1=neutral, 2=contradiction)",
    )
    parser.add_argument("--image-root", default="./data", help="Root directory for multimodal images")
    parser.add_argument(
        "--cross-option-weight",
        type=float,
        default=1.0,
        help="Weight for P1 cross-option softmax loss across competing candidate options (default: 1.0)",
    )
    parser.add_argument(
        "--served-dist-weight",
        type=float,
        default=0.0,
        help="Weight of the served-distribution loss: CE on renormalized P(entailment) over options - Phase-1 train-serving parity (2026-09-22)",
    )
    parser.add_argument(
        "--nli-aux-weight",
        type=float,
        default=0.15,
        help="Weight for auxiliary pointwise 3-class NLI anchor loss (default: 0.15)",
    )
    parser.add_argument(
        "--decision-temp",
        type=float,
        default=1.0,
        help="Temperature for cross-option softmax competition (default: 1.0)",
    )
    parser.add_argument(
        "--full-fine-tune",
        action="store_true",
        default=False,
        help="Train the full transformer backbone (freezing embedding tables & vision tower) instead of LoRA",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Starting epoch index (0-indexed, default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    finetune_custom_data(
        train_data=args.data,
        val_data=args.val_data,
        output_dir=args.out_dir,
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        head_weights=args.head_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        brier_weight=args.brier_weight,
        cross_option_weight=args.cross_option_weight,
        nli_aux_weight=args.nli_aux_weight,
        served_dist_weight=args.served_dist_weight,
        decision_temp=args.decision_temp,
        val_ratio=args.val_ratio,
        start_epoch=args.start_epoch,
        label_convention=args.label_convention,
        full_fine_tune=args.full_fine_tune,
        qat=args.qat,
        target_quant=args.target_quant,
        qat_bits=args.qat_bits,
        qat_group_size=args.qat_group_size,
        use_token_bucketing=args.token_bucketing,
        max_tokens_per_batch=args.max_tokens_per_batch,
        image_root=args.image_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
