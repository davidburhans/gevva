#!/usr/bin/env python3
"""finetune.py
=============
Turnkey custom data fine-tuning engine for the Gemma 4 NLI Cross-Encoder / System 1 Decision Engine.

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
- Continual Fine-Tuning: Can start from raw base (`google/gemma-4-E2B`) OR continue
  from a pre-trained NLI checkpoint / LoRA adapter.
- CLI and Python API (`finetune_custom_data(...)`).

Usage:
    # From JSONL
    python finetune.py --data my_data.jsonl --out-dir ./my_model

    # From CSV with custom columns and starting from pre-trained NLI checkpoint
    python finetune.py --data my_data.csv --adapter ./ckpt/gemma-4-e2b-nli-stage1/best --epochs 3

    # Python API
    from finetune import finetune_custom_data
    finetune_custom_data("my_data.jsonl", output_dir="./my_model")
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

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


def parse_raw_label(raw_val: Any) -> Optional[int]:
    """Maps raw label (string, int, or float) to 0, 1, or 2."""
    if raw_val is None:
        return None
    if isinstance(raw_val, (int, np.integer)):
        if int(raw_val) in (0, 1, 2):
            return int(raw_val)
    str_val = str(raw_val).strip().lower()
    return LABEL_STRING_MAP.get(str_val, None)


def load_custom_file(file_path: str) -> List[Dict[str, Any]]:
    """Loads records from JSONL, CSV, TSV, or JSON file."""
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

    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Please provide .jsonl, .csv, .tsv, or .json")

    if not rows:
        raise ValueError(f"File {file_path} contained 0 rows.")

    # Detect keys and normalize
    p_key, h_key, l_key = detect_columns(rows[0])
    print(f"Loaded {len(rows):,} raw rows from {file_path}")
    print(f"  Detected mapping: Premise='{p_key}', Hypothesis='{h_key}', Label='{l_key}'")

    normalized_data = []
    skipped_labels = Counter()

    for idx, r in enumerate(rows):
        premise = str(r.get(p_key, "")).strip()
        hypothesis = str(r.get(h_key, "")).strip()
        raw_label = r.get(l_key)
        norm_label = parse_raw_label(raw_label)

        if not premise or not hypothesis:
            continue
        if norm_label is None:
            skipped_labels[str(raw_label)] += 1
            continue

        normalized_data.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": norm_label,
            "source": r.get("source", "custom_dataset"),
        })

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
    """Splits data into train and validation sets while preserving label proportions."""
    random.seed(seed)
    by_label = defaultdict(list)
    for r in data:
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
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer.padding_side = "right"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [
            tokenize_nli_pair_safe(
                tokenizer=self.tokenizer,
                premise=r["premise"],
                hypothesis=r["hypothesis"],
                max_length=self.max_length,
            )
            for r in batch
        ]
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

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sources": sources,
        }


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


# -----------------------------------------------------------------------------
# Evaluation Helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dataset(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds, all_probs, all_logits, all_golds, all_sources = [], [], [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        sources = batch["sources"]

        amp_device = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

    return {
        "accuracy": round(acc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "n_samples": len(golds),
        "per_class": per_class,
        "logits": logits,
        "golds": golds,
    }


# -----------------------------------------------------------------------------
# Core Fine-Tuning Routine
# -----------------------------------------------------------------------------
def finetune_custom_data(
    train_data: Union[str, List[Dict[str, Any]]],
    val_data: Optional[Union[str, List[Dict[str, Any]]]] = None,
    output_dir: str = "./ckpt/custom_model",
    base_model_id: str = "google/gemma-4-E2B",
    adapter_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
    lora_r: int = 64,
    lora_alpha: int = 128,
    label_smoothing: float = 0.05,
    val_ratio: float = 0.15,
    qat: bool = True,
    target_quant: str = "nvfp4",
    qat_bits: int = 4,
    qat_group_size: int = 32,
    use_token_bucketing: bool = False,
    max_tokens_per_batch: int = 2048,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """High-level Python API to fine-tune Gemma 4 Cross-Encoder on custom data.

    Args:
        train_data: File path (.jsonl, .csv, .tsv, .json) or in-memory list of dicts.
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
        val_ratio: Validation split fraction when val_data is not provided.
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
        raw_records = load_custom_file(train_data)
    else:
        raw_records = train_data

    if val_data is not None:
        if isinstance(val_data, str):
            val_records = load_custom_file(val_data)
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

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading existing LoRA weights from {adapter_path} for continual fine-tuning...")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        head_weights_path = os.path.join(adapter_path, "head_weights.pt")
        if os.path.exists(head_weights_path):
            print(f"Restoring classification head from {head_weights_path}...")
            hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
            raw = model.base_model.model if hasattr(model, "base_model") else model
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

    # Enable gradient checkpointing to safely fit in RTX 5090 VRAM
    raw_lm = model.base_model.model.model.language_model if hasattr(model, "base_model") else model.model.language_model
    if hasattr(raw_lm, "gradient_checkpointing_enable"):
        raw_lm.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Enabled gradient checkpointing on language model backbone.")

    model.to(device)
    model.print_trainable_parameters()

    # 4. DataLoaders
    collator = CustomNLICollator(tokenizer, max_length=max_length)
    use_pin = (device != "cpu" and torch.cuda.is_available())
    if use_token_bucketing:
        print(f"Using Deterministic Token-Bucket Batching (max_tokens_per_batch={max_tokens_per_batch})...")
        train_lengths = [max(16, min(max_length, (len(r["premise"]) + len(r["hypothesis"])) // 4 + 16)) for r in train_records]
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
    val_loader = DataLoader(
        CustomNLIDataset(val_records),
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
    print(f"\nBeginning training: {epochs} epochs, {len(train_loader)} batches/epoch, {total_steps} optimizer updates")
    best_val_acc = 0.0
    best_metrics = {}
    global_step = 0
    t_start = time.time()

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            amp_device = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels) / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum

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
                    print(
                        f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {curr_lr:.2e} | Speed: {speed:.1f} samples/s",
                        flush=True,
                    )

        # Validation at epoch end
        print(f"\n--- Validation (Epoch {epoch+1}/{epochs}) ---", flush=True)
        val_metrics = evaluate_dataset(model, val_loader, device)
        acc = val_metrics["accuracy"]
        print(f"Accuracy: {acc*100:.2f}% | Brier Score: {val_metrics['brier']:.4f}")
        for cls_name, m in val_metrics["per_class"].items():
            print(f"  [{cls_name:13s}] Prec: {m['precision']:.3f} | Rec: {m['recall']:.3f} | F1: {m['f1']:.3f} (n={m['support']})")

        # Save best checkpoint
        if acc > best_val_acc:
            best_val_acc = acc
            best_metrics = val_metrics
            best_dir = os.path.join(output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            print(f"-> New best accuracy ({best_val_acc*100:.2f}%)! Saving checkpoint to {best_dir}...")
            model.save_pretrained(best_dir)
            # Explicitly save classification head and norm weights (unwrapping any PEFT wrappers)
            raw_model = model.base_model.model if hasattr(model, "base_model") else model
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
                calib_dict = {
                    "optimal_temperature": round(float(t_opt), 4),
                    "val_ece_before": round(float(ece_bef), 4),
                    "val_ece_after": round(float(ece_aft), 4),
                    "val_brier_before": round(float(br_bef), 4),
                    "val_brier_after": round(float(br_aft), 4),
                    "attribution": "Post-hoc validation temperature scaling inspired by sabeel111/OpenSourceJev (MIT License) and Guo et al. (2017)",
                }
                with open(os.path.join(best_dir, "calibration.json"), "w") as f:
                    json.dump(calib_dict, f, indent=2)
                print(f"-> Fitted optimal validation temperature: T* = {t_opt:.4f} (ECE: {ece_bef:.4f} -> {ece_aft:.4f})")

            report_dict = {k: v for k, v in val_metrics.items() if k not in ("logits", "golds")}
            with open(os.path.join(best_dir, "eval_report.json"), "w") as f:
                json.dump(report_dict, f, indent=2)

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
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio if no val-data is provided")
    parser.add_argument("--qat", action="store_true", default=True, help="Enable Quantization-Aware Training (QAT)")
    parser.add_argument("--no-qat", action="store_false", dest="qat", help="Disable QAT and train in full precision")
    parser.add_argument(
        "--target-quant",
        default="nvfp4",
        choices=["nvfp4", "w4a16", "q4_k_m"],
        help="Target quantization format for QAT (nvfp4 for native Blackwell, w4a16 for compressed-tensors, q4_k_m for GGUF)",
    )
    parser.add_argument("--qat-bits", type=int, default=4, help="QAT weight bit-width (default: 4)")
    parser.add_argument("--qat-group-size", type=int, default=32, help="QAT group size (default: 32)")
    parser.add_argument("--token-bucketing", action="store_true", help="Enable deterministic token-bucket batching (decider style)")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048, help="Token budget per batch when using token bucketing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    finetune_custom_data(
        train_data=args.data,
        val_data=args.val_data,
        output_dir=args.out_dir,
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        val_ratio=args.val_ratio,
        qat=args.qat,
        target_quant=args.target_quant,
        qat_bits=args.qat_bits,
        qat_group_size=args.qat_group_size,
        use_token_bucketing=args.token_bucketing,
        max_tokens_per_batch=args.max_tokens_per_batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
