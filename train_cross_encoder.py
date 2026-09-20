#!/usr/bin/env python3
"""train_cross_encoder.py
=========================
Fine-tuning and calibration engine for Gemma 4 (E2B / E4B) NLI Cross-Encoder.

Key Features:
- Backbone: google/gemma-4-E2B or google/gemma-4-E4B
- Custom Gemma4ForSequenceClassification with normalized linear score head
- Vision tower frozen with torch.no_grad()
- LoRA adapters on text attention and MLP projections
- Soft Cross-Entropy / Brier calibration objective
- Metrics: 3-class accuracy, per-source breakdown, and Expected Calibration Error (ECE)

Usage:
    python train_cross_encoder.py --model google/gemma-4-E2B --data-dir ./data --out-dir ./ckpt/gemma-4-e2b-nli --epochs 3 --lr 2e-4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma4Config,
    get_cosine_schedule_with_warmup,
)
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
from finetune import TokenBucketBatchSampler, fit_temperature_scaling, compute_brier_loss

# -----------------------------------------------------------------------------
# Calibration Metrics
# -----------------------------------------------------------------------------
def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Computes Expected Calibration Error (ECE) and Brier Score."""
    N = len(labels)
    if N == 0:
        return {"ece": 0.0, "brier": 0.0}

    # Brier score: mean squared error between probability vector and one-hot gold
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # ECE on the top predicted class
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {"ece": float(ece), "brier": brier}


# -----------------------------------------------------------------------------
# Dataset & Dynamic Collator
# -----------------------------------------------------------------------------
class NLIDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                if max_samples and len(self.rows) >= max_samples:
                    break
        print(f"Loaded {len(self.rows):,} rows from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class DataCollatorNLI:
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
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
        sources = [r["source"] for r in batch]
        ids = [r.get("id", f"row_{i}") for i, r in enumerate(batch)]

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
            "ids": ids,
        }
        if pixel_values_list:
            res["pixel_values"] = torch.stack(pixel_values_list)
            res["image_position_ids"] = torch.stack(image_pos_ids_list)

        return res


# -----------------------------------------------------------------------------
# Training Runner
# -----------------------------------------------------------------------------
def train_cross_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. Tokenizer
    print(f"Loading tokenizer for {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        print("Falling back to local / default tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Model Configuration & Instantiation
    print(f"Initializing {args.model} cross-encoder...")
    is_gemma = "gemma" in args.model.lower()
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 3
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.pad_token_id = pad_id
    if hasattr(config, "get_text_config"):
        config.get_text_config().pad_token_id = pad_id

    if is_gemma:
        model = Gemma4ForSequenceClassification.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
        )
        model.freeze_vision_tower(freeze_adapter=False)
        lora_target = (
            r".*language_model.*(q|k|v|o|gate|up|down)_proj"
            if getattr(args, "lora_all_projections", True)
            else r".*language_model.*self_attn\.(q|k|v|o)_proj"
        )
        modules_to_save = ["score", "norm"]
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_id
            if hasattr(model.config, "get_text_config"):
                model.config.get_text_config().pad_token_id = pad_id
        # Freeze visual tower if present (e.g. Qwen3.5 vision)
        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                p.requires_grad = False
            print("Frozen Qwen visual parameters.")
        elif hasattr(getattr(model, "model", None), "visual"):
            for p in model.model.visual.parameters():
                p.requires_grad = False
            print("Frozen Qwen model.visual parameters.")
        lora_target = (
            r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
            if getattr(args, "lora_all_projections", True)
            else r".*(q_proj|k_proj|v_proj|o_proj).*"
        )
        modules_to_save = ["score"]

    # Apply LoRA if requested
    if args.lora:
        print(f"Applying LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha}) to projections: {lora_target}...")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target,
            modules_to_save=modules_to_save,
            lora_dropout=0.05,
            bias="none",
            task_type=None,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    target_quant = getattr(args, "target_quant", "none")
    if getattr(args, "qat", False) or (target_quant and target_quant != "none"):
        from gemma4_cross_encoder import apply_quantization_aware_training
        fmt = target_quant if target_quant != "none" else "w4a16"
        print(f"Applying Quantization-Aware Training (QAT): Format='{fmt}' (group_size={args.qat_group_size})...")
        model = apply_quantization_aware_training(
            model,
            quant_format=fmt,
            num_bits=args.qat_bits,
            group_size=args.qat_group_size,
        )

    # Enable gradient checkpointing to safely fit in RTX 5090 VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Enabled gradient checkpointing on model.")
    else:
        raw_lm = getattr(model, "base_model", model)
        if hasattr(raw_lm, "gradient_checkpointing_enable"):
            raw_lm.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on raw backbone.")

    model.to(device)

    # 3. Datasets & Dataloaders
    # WHY: --train-file lets the A/B shakedown arms swap training mixes without
    # duplicating loaders; val stays the selection split, test the report-only split.
    train_path = args.train_file or os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    train_ds = NLIDataset(train_path, max_samples=args.max_train_samples)
    val_ds = NLIDataset(val_path, max_samples=args.max_val_samples)
    image_processor = Gemma4ImageProcessorPil() if is_gemma else None
    collator = DataCollatorNLI(
        tokenizer,
        max_length=args.max_length,
        image_processor=image_processor,
        image_root=args.data_dir,
        is_gemma=is_gemma,
    )
    if getattr(args, "token_bucketing", False):
        print(f"Using Deterministic Token-Bucket Batching (max_tokens_per_batch={getattr(args, 'max_tokens_per_batch', 2048)})...")
        train_lengths = [max(16, min(args.max_length, (len(r["premise"]) + len(r["hypothesis"])) // 4 + 16)) for r in train_ds.rows]
        train_sampler = TokenBucketBatchSampler(
            train_lengths, max_tokens_per_batch=getattr(args, "max_tokens_per_batch", 2048), shuffle=True, seed=args.seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # 4. Optimizer & LR Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 5. Training Loop
    print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} batches/epoch, {total_steps} total update steps")
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_brier_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
            image_position_ids = batch["image_position_ids"].to(device) if "image_position_ids" in batch else None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                )
                ce_loss = loss_fn(outputs.logits, labels)
                if getattr(args, "brier_weight", 0.0) > 0.0:
                    probs = torch.softmax(outputs.logits, dim=-1)
                    one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
                    brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
                    total_loss = ce_loss + args.brier_weight * brier_loss
                else:
                    brier_loss = None
                    total_loss = ce_loss
                loss = total_loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            if brier_loss is not None:
                epoch_brier_loss += brier_loss.item()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    brier_msg = f" | Brier: {epoch_brier_loss / (step + 1):.4f}" if getattr(args, "brier_weight", 0.0) > 0.0 else ""
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {epoch_loss / (step + 1):.4f}{brier_msg} | LR: {lr:.2e} | "
                        f"Speed: {(step + 1) * args.batch_size / (time.time() - t0):.1f} samples/s",
                        flush=True,
                    )

        # Validation at epoch end
        print(f"\nRunning validation for Epoch {epoch+1}...")
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation Results: Acc={val_metrics['accuracy']:.4f} | ECE={val_metrics['ece']:.4f} | Brier={val_metrics['brier']:.4f}")
        for src, acc in val_metrics["by_source"].items():
            print(f"  - {src}: {acc:.4f}")

        # Checkpoint saving
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_dir = os.path.join(args.out_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            print(f"New best accuracy {best_val_acc:.4f}! Saving checkpoint to {save_dir}...")
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(save_dir)
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))
            # Explicitly save classification head and norm weights (unwrapping PEFT wrappers)
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
            torch.save(head_dict, os.path.join(save_dir, "head_weights.pt"))
            tokenizer.save_pretrained(save_dir)

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
                with open(os.path.join(save_dir, "calibration.json"), "w") as f:
                    json.dump(calib_dict, f, indent=2)
                print(f"-> Fitted optimal validation temperature: T* = {t_opt:.4f} (ECE: {ece_bef:.4f} -> {ece_aft:.4f})")

            clean_val_metrics = {k: v for k, v in val_metrics.items() if k not in ("logits", "golds")}
            with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
                json.dump(clean_val_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.4f}")
    print("=" * 60)

    # Config provenance (audit A8: the trained quant format & recipe must be recoverable).
    run_config = {"args": vars(args), "best_val_acc": best_val_acc,
                  "train_path": train_path, "val_path": val_path,
                  "test_file": args.test_file,
                  "qat_applied": bool(getattr(args, "qat", False) or getattr(args, "target_quant", "none") != "none"),
                  "target_quant": getattr(args, "target_quant", "none"),
                  "brier_weight": getattr(args, "brier_weight", 0.0)}
    for cfg_dir in (args.out_dir, os.path.join(args.out_dir, "best")):
        if os.path.isdir(cfg_dir):
            with open(os.path.join(cfg_dir, "train_config.json"), "w") as f:
                json.dump(run_config, f, indent=2)

    # Report-only test evaluation, exactly ONCE, on the RELOADED best checkpoint
    # (audit A1: test never participates in checkpoint selection).
    if args.test_file and os.path.exists(args.test_file):
        save_dir = os.path.join(args.out_dir, "best")
        if not os.path.isdir(save_dir):
            print("No best checkpoint saved; skipping test evaluation.")
            return
        print(f"Reloading best checkpoint from {save_dir} for one-shot test evaluation...")
        best_model = _reload_best_for_eval(args, tokenizer, save_dir)
        best_model.to(device)
        # Head-restore probe (WHY: a PEFT reload that silently drops the saved score
        # head scores near-chance; fail loudly BEFORE the one-shot test pass).
        probe = evaluate(best_model, val_loader, device)
        if probe["accuracy"] < 0.5 * max(best_val_acc, 1e-6):
            raise RuntimeError("reloaded checkpoint validation accuracy collapsed - PEFT head restore failed")
        # Test-set fingerprint: ties every reported number & row to the exact test bytes.
        test_sha = hashlib.sha256(open(args.test_file, 'rb').read()).hexdigest()[:16]
        test_ds = NLIDataset(args.test_file)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 collate_fn=collator, num_workers=2)
        test_out = evaluate(best_model, test_loader, device, return_items=True)
        items = test_out.pop("items")
        test_out["test_sha"] = test_sha
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_out, f, indent=2)
        with open(os.path.join(args.out_dir, "test_items.jsonl"), "w", encoding="utf-8") as f:
            for item in items:
                item["test_sha"] = test_sha
                f.write(json.dumps(item) + "\n")
        print(f"TEST (report-only, n={test_out['n_samples']}): "
              f"acc={test_out['accuracy']:.4f} ece={test_out['ece']:.4f} brier={test_out['brier']:.4f}")
        print(f"  per-item log: {os.path.join(args.out_dir, 'test_items.jsonl')}")


def _reload_best_for_eval(args, tokenizer, save_dir):
    """Rebuilds the base model and restores the best adapter + saved head modules."""
    from peft import PeftModel
    is_gemma = "gemma" in args.model.lower()
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 3
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.pad_token_id = pad_id
    if hasattr(config, "get_text_config"):
        config.get_text_config().pad_token_id = pad_id
    if is_gemma:
        base = Gemma4ForSequenceClassification.from_pretrained(
            args.model, config=config, torch_dtype=torch.bfloat16)
    else:
        from transformers import AutoModelForSequenceClassification
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model, config=config, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, save_dir)
    head_weights_path = os.path.join(save_dir, "head_weights.pt")
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
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, return_items: bool = False) -> Dict[str, Any]:
    model.eval()
    all_preds, all_probs, all_logits, all_golds, all_sources, all_ids = [], [], [], [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        sources = batch["sources"]
        all_ids.extend(batch["ids"])
        pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
        image_position_ids = batch["image_position_ids"].to(device) if "image_position_ids" in batch else None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
    calib = compute_calibration_metrics(probs, golds)

    # Breakdown by source
    by_source = defaultdict(list)
    for p, g, s in zip(preds, golds, all_sources):
        by_source[s].append(p == g)
    by_source_acc = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_source.items())}

    return {
        "accuracy": acc,
        "ece": calib["ece"],
        "brier": calib["brier"],
        "by_source": by_source_acc,
        "n_samples": len(golds),
        "label_dist": dict(Counter(preds.tolist())),
        "logits": logits,
        "golds": golds,
        **({"items": [
            {"id": i, "source": s, "gold": int(g), "pred": int(p),
             "confidence": float(max(pr)), "correct": bool(p == g)}
            for i, s, g, p, pr in zip(all_ids, all_sources, golds, preds, probs)
        ]} if return_items else {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B", help="Base Gemma 4 model path")
    parser.add_argument("--data-dir", default="./data", help="Path to compiled dataset directory")
    parser.add_argument("--train-file", default=None, help="Override train.jsonl path (A/B arm datasets)")
    parser.add_argument("--test-file", default=None, help="Report-only test jsonl; evaluated once on the reloaded best checkpoint")
    parser.add_argument("--out-dir", default="./ckpt/gemma-4-e2b-nli", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lora", action="store_true", default=True, help="Use LoRA fine-tuning")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha scaling (default: 128)")
    parser.add_argument("--lora-all-projections", action="store_true", default=True, help="Target all 7 linear projections")
    parser.add_argument("--token-bucketing", action="store_true", help="Enable deterministic token-bucket batching (decider style)")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048, help="Token budget per batch when using token bucketing")
    parser.add_argument("--qat", action="store_true", default=False, help="Enable Quantization-Aware Training (QAT)")
    parser.add_argument(
        "--target-quant",
        default="nvfp4",
        choices=["nvfp4", "w4a16", "q4_k_m", "none"],
        help="Target quantization format for QAT (default: nvfp4 for native Blackwell FP4; w4a16 for compressed-tensors; q4_k_m for GGUF)",
    )
    parser.add_argument("--qat-bits", type=int, default=4, help="QAT weight bit-width (default: 4)")
    parser.add_argument("--qat-group-size", type=int, default=32, help="QAT group size (default: 32)")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=0.0,
        help="Weight for multi-class Brier calibration loss (default: 0.0, recommended: 0.5 based on von-1.0 and research report 08)",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_cross_encoder(args)
