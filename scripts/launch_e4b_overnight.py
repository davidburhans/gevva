#!/usr/bin/env python3
"""scripts/launch_e4b_overnight.py - Launches Flagship Gemma 4 E4B Full Fine-Tuning Run on Master Curriculum.

Master Unified Pipeline:
- Base Model: google/gemma-4-E4B-it (4.5B params, 42 layers, 128K context)
- Trainable Parameters: 3.98B transformer parameters across all 42 layers
- Dataset: data/train_master_full.jsonl (237,640 unique decontaminated pairs across 84 sources)
- Optimizer: bitsandbytes PagedAdamW8bit (8-bit paged optimizer)
- Differential Learning Rates:
  - Classification Head (score + norm): 1.0e-4 (rapid convergence from fresh initialization)
  - Transformer Backbone: 3.0e-6 (gentle representation adaptation, cosine decay with 10% warmup)
- Loss Function:
  - Served Distribution Loss: 1.0 (exact train-serving parity for all decision groups)
  - Cross-Option Loss: 0.0
  - Auxiliary Pointwise NLI Loss: 0.25 (3-class anchor regularizer)
  - Brier Calibration Loss: 0.5 (strictly proper scoring rule)
- Batching & Collator: Group-atomic batching via GroupedTokenBucketBatchSampler (max_tokens=2048, max_length=1024)
- Memory Profile on RTX 5090: Peak VRAM ~25.0 GB (< 32.6 GB limit, safe 7.5GB headroom)
- Runtime: 1 full epoch over 237,640 samples (~14,800 batches, ~925 optimizer updates) = ~9.4 hours
- Output: ckpt/gevva-e4b-flagship
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Launch Overnight Gemma 4 E4B Flagship Training on Master Curriculum")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=3.0e-6, help="Backbone learning rate (default: 3.0e-6)")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Classification head learning rate (default: 1e-4)")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence token length (default: 1024)")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048, help="Token budget per batch (default: 2048)")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--out-dir", default="ckpt/gevva-e4b-flagship", help="Output directory for checkpoint")
    args = parser.parse_args()

    data_file = REPO_ROOT / "data" / "train_master_full.jsonl"
    if not data_file.exists():
        print(f"Error: {data_file} not found. Run scripts/compile_full_master_mixture.py first.")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "finetune.py"),
        "--data", str(data_file),
        "--base-model", "google/gemma-4-E4B-it",
        "--out-dir", args.out_dir,
        "--full-fine-tune",
        "--use-8bit-adam",
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--head-lr", str(args.head_lr),
        "--grad-accum", str(args.grad_accum),
        "--token-bucketing",
        "--max-tokens-per-batch", str(args.max_tokens_per_batch),
        "--max-length", str(args.max_length),
        "--image-root", ".",
        "--brier-weight", "0.5",
        "--served-dist-weight", "1.0",
        "--cross-option-weight", "0.0",
        "--nli-aux-weight", "0.25",
        "--decision-temp", "1.0",
        "--val-ratio", "0.05",
    ]

    print("==================================================")
    print("LAUNCHING GEVVA E4B UNIFIED MASTER FLAGSHIP TRAINING")
    print(f"Curriculum: {data_file} (237,640 unique pairs)")
    print(f"Backbone LR: {args.lr:.2e} | Head LR: {args.head_lr:.2e} | Epochs: {args.epochs}")
    print(f"Command: {' '.join(cmd)}")
    print("==================================================")

    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
