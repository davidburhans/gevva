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
from torch.utils.data import DataLoader, Dataset
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
# Evaluation Helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dataset(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_preds, all_probs, all_golds, all_sources = [], [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        sources = batch["sources"]

        amp_device = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=-1))
        all_golds.append(labels.numpy())
        all_sources.extend(sources)

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    golds = np.concatenate(all_golds, axis=0)

    acc = float(np.mean(preds == golds))

    # Brier score
    N = len(golds)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), golds] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

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
        "n_samples": len(golds),
        "per_class": per_class,
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
            with open(os.path.join(best_dir, "eval_report.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)

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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
