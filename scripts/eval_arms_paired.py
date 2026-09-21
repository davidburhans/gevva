#!/usr/bin/env python3
"""scripts/eval_arms_paired.py - Evaluates Arm A and Arm B on the exact same test split.

Ensures 100% paired coverage (all n=3,113 samples in data/test.jsonl) so the McNemar
paired test and ECE non-regression checks have full statistical power.

Usage:
  uv run python scripts/eval_arms_paired.py --test-file data/test.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gemma4_cross_encoder import Gemma4ForSequenceClassification, Gemma4ImageProcessorPil
from train_cross_encoder import DataCollatorNLI, NLIDataset, evaluate, _reload_best_for_eval


def eval_checkpoint(ckpt_dir: str, test_file: str, tokenizer, collator, device: str) -> None:
    items_path = os.path.join(ckpt_dir, "test_items.jsonl")
    if os.path.exists(items_path):
        with open(items_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        with open(test_file) as f:
            n_target = sum(1 for line in f if line.strip())
        if len(lines) == n_target:
            first_item = json.loads(lines[0])
            if isinstance(first_item.get("id"), str):
                print(f"  {ckpt_dir} already evaluated on {test_file} ({len(lines)} rows, valid string IDs); skipping.")
                return

    print(f"\nEvaluating {ckpt_dir} on {test_file}...")
    best_dir = os.path.join(ckpt_dir, "best") if os.path.exists(os.path.join(ckpt_dir, "best")) else ckpt_dir

    class DummyArgs:
        model = "google/gemma-4-E2B"
        qat = False
        target_quant = "none"

    model = _reload_best_for_eval(DummyArgs(), tokenizer, best_dir)
    model.to(device)
    model.eval()

    test_ds = NLIDataset(test_file)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collator)

    metrics = evaluate(model, test_loader, device, return_items=True)
    items = metrics.get("items", [])
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}% | ECE: {metrics['ece']:.4f} | Brier: {metrics['brier']:.4f}")

    import hashlib
    test_sha = hashlib.sha256(open(test_file, 'rb').read()).hexdigest()[:16]

    items_path = os.path.join(ckpt_dir, "test_items.jsonl")
    with open(items_path, "w", encoding="utf-8") as f:
        for it in items:
            it["test_sha"] = test_sha
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items):,} item evaluations to {items_path}.")

    metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
    summary = {k: v for k, v in metrics.items() if k not in ("logits", "golds", "items")}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--arm-a", default="ckpt/shakedown_A")
    parser.add_argument("--arm-b", default="ckpt/shakedown_B")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B", local_files_only=True)
    img_proc = Gemma4ImageProcessorPil()
    collator = DataCollatorNLI(tokenizer, max_length=512, image_processor=img_proc, image_root="data")

    eval_checkpoint(args.arm_a, args.test_file, tokenizer, collator, device)
    eval_checkpoint(args.arm_b, args.test_file, tokenizer, collator, device)

    # Run McNemar gate
    import subprocess
    cmd = [sys.executable, "scripts/gate_decision.py", "--arm-a", args.arm_a, "--arm-b", args.arm_b, "--out", "results/gate_decision.json"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
