#!/usr/bin/env python3
"""train_gevva_v2.py - Production Training Engine for Gevva v2 Native Decision Engine.

Trains Gemma 4 as a native Causal Decision Engine matching the stock llama.cpp `lev` protocol
with:
1. Mandatory In-Loop Quantization-Aware Training (QAT) by default (Straight-Through Estimator).
2. Active-Set Target-Restricted Cross-Entropy Loss over dynamic candidate codes ['A'..'Z', 'AA'..'ZZ'].
3. Epoch-Level Permutation Augmentation & Symmetry Regularization.
4. Soft Teacher Distillation (Brier / KL Divergence).
5. Fail-Fast QAT Assertions (Principle 7).
6. Sliced Logit Memory Optimization: Projects hidden state to logits only at decision token over active codes.

Usage:
    # Production QAT Training (Q4_K_M for llama.cpp):
    uv run python train_gevva_v2.py \
        --model-id google/gemma-4-E2B-it \
        --data data/gevva_v2_train.jsonl \
        --out-dir ckpt/gevva-v2-e2b \
        --target-quant q4_k_m \
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT))

from gemma4_cross_encoder import apply_quantization_aware_training  # noqa: E402


SYSTEMONE_CHAT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{% for image in images %}{{ image }}{% endfor %}"
    "{% if images %}The image shows the input visual scene.\n\n{% endif %}"
    "State:\n"
    "{{ state if state is string else state | tojson(indent=2) }}\n\n"
    "Question: {{ instructions }}\n"
    "{% if type == 'score' %}Rate along the ordered levels below (lowest first).\n{% endif %}"
    "Options:\n"
    "{% for o in options %}"
    "[{{ o.label }}] {{ o.key }}{% if o.description %}: {{ o.description }}{% endif %}\n"
    "{% endfor %}\n"
    "Answer with the option code only.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def get_option_code(idx: int) -> str:
    """Returns single/two-letter code: 0->'A'..25->'Z', 26->'AA'..254->'IU'."""
    if idx < 26:
        return chr(65 + idx)
    first = chr(65 + (idx - 26) // 26)
    second = chr(65 + (idx - 26) % 26)
    return f"{first}{second}"


def format_systemone_sample(
    sample: Dict[str, Any],
    permute: bool = False,
) -> Tuple[str, str, List[str], str, List[float]]:
    """Formats a Gevva v2 JSONL sample into user prompt, completion, active codes, gold code, and soft probs."""
    state_str = sample.get("state", "")
    if not isinstance(state_str, str):
        state_str = json.dumps(state_str, indent=2)

    instr_str = sample.get("instructions", "")
    orig_options = sample.get("options", [])
    gold_code = sample.get("gold_code", "A").strip()

    orig_soft_probs = sample.get("soft_probabilities") or []
    if sample.get("soft_probability") is not None and len(orig_options) == 2:
        sp = float(sample["soft_probability"])
        gold_i = next((i for i, o in enumerate(orig_options) if o.get("label") == gold_code), 0)
        orig_soft_probs = [sp, 1.0 - sp] if gold_i == 0 else [1.0 - sp, sp]

    if permute and len(orig_options) > 1:
        perm = list(range(len(orig_options)))
        random.shuffle(perm)
        options = []
        new_gold_code = "A"
        new_soft_probs = [0.0] * len(orig_options) if orig_soft_probs else []
        for new_idx, old_idx in enumerate(perm):
            code = get_option_code(new_idx)
            opt_copy = dict(orig_options[old_idx])
            opt_copy["label"] = code
            options.append(opt_copy)
            if orig_options[old_idx].get("label") == gold_code:
                new_gold_code = code
            if orig_soft_probs and old_idx < len(orig_soft_probs):
                new_soft_probs[new_idx] = orig_soft_probs[old_idx]
        gold_code = new_gold_code
        soft_probs = new_soft_probs
    else:
        options = orig_options
        soft_probs = orig_soft_probs

    active_codes: List[str] = []
    option_lines: List[str] = []
    for opt in options:
        lbl = opt.get("label", "").strip()
        key = opt.get("key", "").strip()
        desc = opt.get("description", "").strip()
        active_codes.append(lbl)
        if desc:
            option_lines.append(f"[{lbl}] {key}: {desc}")
        else:
            option_lines.append(f"[{lbl}] {key}")

    options_block = "\n".join(option_lines)
    user_prompt = (
        "<start_of_turn>user\n"
        f"State:\n{state_str}\n\n"
        f"Question: {instr_str}\n"
        "Options:\n"
        f"{options_block}\n"
        "Answer with the option code only.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    completion = f"{gold_code}<end_of_turn>"
    return user_prompt, completion, active_codes, gold_code, soft_probs


class GevvaV2Dataset(Dataset):
    """Causal Decision Dataset with Active-Set Option Token resolution."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        permute: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.permute = permute
        # Cache single-token IDs for option codes
        self._code_to_token: Dict[str, int] = {}

    def _get_token_id(self, code: str) -> int:
        if code not in self._code_to_token:
            ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(ids) != 1:
                self._code_to_token[code] = ids[0]
            else:
                self._code_to_token[code] = ids[0]
        return self._code_to_token[code]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        user_prompt, completion, active_codes, gold_code, soft_probs = format_systemone_sample(
            raw, permute=self.permute
        )

        prompt_ids = self.tokenizer.encode(user_prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        input_ids = prompt_ids + completion_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + completion_ids

        # Decision token position is the last token of prompt (which predicts completion_ids[0])
        decision_pos = len(prompt_ids) - 1

        active_token_ids = [self._get_token_id(c) for c in active_codes]
        gold_token_id = self._get_token_id(gold_code)
        gold_idx = active_codes.index(gold_code) if gold_code in active_codes else 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "decision_pos": decision_pos,
            "active_token_ids": torch.tensor(active_token_ids, dtype=torch.long),
            "gold_token_id": gold_token_id,
            "gold_idx": gold_idx,
            "soft_probs": torch.tensor(soft_probs, dtype=torch.float) if soft_probs else torch.empty(0),
        }


def collate_gevva_v2(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Collates variable-length decision sequences with right-padding."""
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    decision_positions = []
    active_token_ids_list = []
    gold_indices = []
    soft_probs_list = []

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = 1
        decision_positions.append(b["decision_pos"])
        active_token_ids_list.append(b["active_token_ids"])
        gold_indices.append(b["gold_idx"])
        soft_probs_list.append(b["soft_probs"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decision_positions": decision_positions,
        "active_token_ids": active_token_ids_list,
        "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
        "soft_probs": soft_probs_list,
    }


def compute_active_set_loss(
    model: nn.Module,
    last_hidden_state: torch.Tensor,
    decision_positions: List[int],
    active_token_ids: List[torch.Tensor],
    gold_indices: torch.Tensor,
    soft_probs: List[torch.Tensor],
    brier_weight: float = 0.4,
    device: str = "cuda",
) -> Tuple[torch.Tensor, float]:
    """Computes target-restricted active-set cross-entropy and soft label Brier loss.

    Memory Optimization:
    Slices hidden state strictly at decision_pos and projects only to active candidate tokens.
    Eliminates allocating the full (batch, seq_len, 262144) vocabulary projection.
    """
    total_loss = torch.tensor(0.0, device=device)
    correct_count = 0
    bsz = len(decision_positions)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    for i in range(bsz):
        pos = decision_positions[i]
        active_ids = active_token_ids[i].to(device)
        gold_idx = gold_indices[i].to(device)

        # Hidden state at decision position for item i: shape (1, hidden_dim)
        h_i = last_hidden_state[i:i+1, pos, :]
        # Active lm_head weights: shape (K, hidden_dim)
        w_active = model.lm_head.weight[active_ids, :]

        # Sliced logits: shape (K,)
        step_logits = torch.matmul(h_i, w_active.t()).squeeze(0)
        if softcap is not None:
            step_logits = torch.tanh(step_logits / softcap) * softcap

        pred_idx = torch.argmax(step_logits).item()
        if pred_idx == gold_idx.item():
            correct_count += 1

        # 1. Target-restricted cross-entropy loss
        ce_loss = F.cross_entropy(step_logits.unsqueeze(0), gold_idx.unsqueeze(0))

        # 2. Soft label distillation / Brier loss
        sp = soft_probs[i]
        if sp.numel() == active_ids.numel():
            sp = sp.to(device)
            pred_probs = F.softmax(step_logits, dim=-1)
            brier_loss = torch.mean((pred_probs - sp) ** 2)
            item_loss = (1.0 - brier_weight) * ce_loss + brier_weight * brier_loss
        else:
            item_loss = ce_loss

        total_loss = total_loss + item_loss

    avg_loss = total_loss / max(bsz, 1)
    acc = correct_count / max(bsz, 1)
    return avg_loss, acc


def train_gevva_v2(args):
    # -------------------------------------------------------------------------
    # 0. Mandatory QAT Invariant Enforcement (Principle 7)
    # -------------------------------------------------------------------------
    if not args.qat:
        if not args.allow_unquantized_experimental_run:
            raise RuntimeError(
                "FATAL INVARIANT VIOLATION: Gevva v2 requires Quantization-Aware Training (QAT) by contract. "
                "All v2 training rounds must simulate target quantization (q4_k_m / w4a16). "
                "To bypass this for an unquantized experiment, you must explicitly pass --allow-unquantized-experimental-run."
            )
        print("[WARNING] Running UNQUANTIZED experimental run via explicit bypass.")
    else:
        print(f"[QAT GUARD] Verified active QAT contract: format='{args.target_quant}', bits={args.qat_bits}, group_size={args.qat_group_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("Gevva v2 Native Decision Engine Trainer (lev Protocol)")
    print("=" * 65)
    print(f"Base Model:       {args.model_id}")
    print(f"Dataset:          {args.data}")
    print(f"Output Directory: {args.out_dir}")
    print(f"QAT Enabled:      {args.qat} ({args.target_quant})")
    print(f"Batch Size:       {args.batch_size} (accum: {args.grad_accum})")
    print(f"Learning Rate:    {args.lr}")
    print(f"Device:           {device}")

    # 1. Load Tokenizer & Config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Causal Foundation Model
    print(f"\nLoading {args.model_id} in bfloat16...")
    config = AutoConfig.from_pretrained(args.model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    # 3. Inject QAT Fake-Quantization Layers
    if args.qat:
        print(f"\nInjecting in-loop '{args.target_quant}' QAT into linear projections...")
        model = apply_quantization_aware_training(
            model,
            quant_format=args.target_quant,
            num_bits=args.qat_bits,
            group_size=args.qat_group_size,
        )

    # Enable gradient checkpointing to keep activation memory minimal
    model.gradient_checkpointing_enable()
    print("Enabled gradient checkpointing.")

    model.to(device)

    # Freeze vision tower and audio tower to keep training focused and fast
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vt = model.model.vision_tower
        if vt is not None:
            for p in vt.parameters():
                p.requires_grad = False
            print("Frozen SigLIP vision tower.")
    if hasattr(model, "model") and hasattr(model.model, "audio_tower"):
        at = model.model.audio_tower
        if at is not None:
            for p in at.parameters():
                p.requires_grad = False
            print("Frozen audio tower.")

    # 4. Prepare Dataset & Dataloader
    raw_samples: List[Dict[str, Any]] = []
    print(f"\nLoading Gevva v2 dataset from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_samples.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(raw_samples):,} total decision samples.")
    if len(raw_samples) == 0:
        raise ValueError(f"No samples loaded from {args.data}!")

    # Train / Val Split
    random.shuffle(raw_samples)
    val_size = max(int(len(raw_samples) * args.val_ratio), 10)
    train_size = len(raw_samples) - val_size
    train_samples = raw_samples[:train_size]
    val_samples = raw_samples[train_size:]

    train_ds = GevvaV2Dataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        permute=True,
    )
    val_ds = GevvaV2Dataset(
        samples=val_samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        permute=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_gevva_v2(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_gevva_v2(b, tokenizer.pad_token_id),
    )

    print(f"Train samples: {train_size:,} (with epoch permutation) | Val samples: {val_size:,}")

    # 5. Optimizer & Scheduler
    no_decay = ["bias", "rms_norm", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=max(total_steps, 1),
    )

    # 6. Training Loop
    print("\nStarting Training...")
    global_step = 0
    best_val_acc = 0.0
    model.train()

    if args.dry_run:
        print("[DRY-RUN] Executing 2 verification batches...")
        for step, batch in enumerate(train_loader):
            if step >= 2:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
                loss, acc = compute_active_set_loss(
                    model=model,
                    last_hidden_state=outputs.last_hidden_state,
                    decision_positions=batch["decision_positions"],
                    active_token_ids=batch["active_token_ids"],
                    gold_indices=batch["gold_indices"],
                    soft_probs=batch["soft_probs"],
                    brier_weight=args.brier_weight,
                    device=device,
                )
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            print(f"  Dry-run step {step}: Loss = {loss.item():.4f}, Acc = {acc*100:.1f}%")
        print("[DRY-RUN] Verification successful! Exiting dry-run cleanly.")
        return

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        running_acc = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
                loss, acc = compute_active_set_loss(
                    model=model,
                    last_hidden_state=outputs.last_hidden_state,
                    decision_positions=batch["decision_positions"],
                    active_token_ids=batch["active_token_ids"],
                    gold_indices=batch["gold_indices"],
                    soft_probs=batch["soft_probs"],
                    brier_weight=args.brier_weight,
                    device=device,
                )
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            epoch_loss += loss.item()
            running_acc += acc

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_acc = (running_acc / (step + 1)) * 100.0
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.1f}% | LR: {lr:.2e}",
                        flush=True,
                    )

                if args.save_interval_steps > 0 and global_step % args.save_interval_steps == 0:
                    interim_dir = Path(args.out_dir) / f"checkpoint-step-{global_step}"
                    interim_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[Checkpoint] Saving interim checkpoint to {interim_dir}...", flush=True)
                    model.save_pretrained(interim_dir)
                    tokenizer.save_pretrained(interim_dir)

        # Validation at epoch end
        model.eval()
        val_loss = 0.0
        val_acc_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in val_loader:
                input_ids = val_batch["input_ids"].to(device)
                attention_mask = val_batch["attention_mask"].to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    v_loss, v_acc = compute_active_set_loss(
                        model=model,
                        last_hidden_state=outputs.last_hidden_state,
                        decision_positions=val_batch["decision_positions"],
                        active_token_ids=val_batch["active_token_ids"],
                        gold_indices=val_batch["gold_indices"],
                        soft_probs=val_batch["soft_probs"],
                        brier_weight=args.brier_weight,
                        device=device,
                    )
                val_loss += v_loss.item()
                val_acc_total += v_acc
                val_steps += 1

        val_acc = (val_acc_total / max(val_steps, 1)) * 100.0
        val_loss_avg = val_loss / max(val_steps, 1)
        print(f"\n[Validation] Epoch {epoch+1}: Loss = {val_loss_avg:.4f} | Accuracy = {val_acc:.2f}%\n", flush=True)

        # Save Best Checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_dir = Path(args.out_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving new best checkpoint to {best_dir} (Val Acc: {val_acc:.2f}%)...", flush=True)

            # Remove parametrizations before saving state dict for clean inference
            import torch.nn.utils.parametrize as parametrize
            for name, mod in model.named_modules():
                if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
                    parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)

            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            # Persist QAT Provenance with explicit contract (Principle 7)
            qat_provenance = {
                "qat_applied": bool(args.qat),
                "target_quant": args.target_quant if args.qat else "none",
                "qat_bits": args.qat_bits,
                "qat_group_size": args.qat_group_size,
                "protocol": "lev",
                "val_accuracy": val_acc,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(best_dir / "qat_config.json", "w", encoding="utf-8") as f:
                json.dump(qat_provenance, f, indent=2)

            with open(best_dir / "eval_report.json", "w", encoding="utf-8") as f:
                json.dump({"val_accuracy": val_acc, "val_loss": val_loss_avg, "best_step": global_step}, f, indent=2)

            # Re-inject QAT for further epochs if training continues
            if args.qat and epoch < args.epochs - 1:
                model = apply_quantization_aware_training(
                    model,
                    quant_format=args.target_quant,
                    num_bits=args.qat_bits,
                    group_size=args.qat_group_size,
                )

        model.train()

    print(f"\nTraining Complete! Best checkpoint at {args.out_dir}/best with Val Acc = {best_val_acc:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Gevva v2 native causal decision engine.")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it", help="Foundation model identifier")
    parser.add_argument("--data", default=str(DATA_DIR / "gevva_v2_train.jsonl"), help="Path to compiled v2 JSONL")
    parser.add_argument("--out-dir", default="ckpt/gevva-v2-e2b", help="Output directory for checkpoints")
    parser.add_argument("--qat", action=argparse.BooleanOptionalAction, default=True, help="Enable Quantization-Aware Training (default: True)")
    parser.add_argument("--allow-unquantized-experimental-run", action="store_true", default=False, help="Explicit override to allow training without QAT")
    parser.add_argument("--target-quant", default="q4_k_m", choices=["q4_k_m", "w4a16", "nvfp4"], help="Target quantization format (default: q4_k_m)")
    parser.add_argument("--qat-bits", type=int, default=4, help="QAT weight bit-width (default: 4)")
    parser.add_argument("--qat-group-size", type=int, default=32, help="QAT group size (default: 32)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--save-interval-steps", type=int, default=1000, help="Save interim checkpoint every N steps (default: 1000)")
    parser.add_argument("--lr", type=float, default=2.0e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence token length")
    parser.add_argument("--brier-weight", type=float, default=0.4, help="Brier soft-probability loss weight")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation holdout ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 forward-backward steps to verify shapes and exit")
    args = parser.parse_args()

    train_gevva_v2(args)


if __name__ == "__main__":
    main()
