#!/usr/bin/env python3
"""scripts/launch_phase3_epoch2.py - Launch Phase-3 Epoch 2 Training on E2B.

Continual Full Fine-Tuning from Epoch-1 best checkpoint:
- Base Model: ckpt/gemma-4-e2b-nli-phase3/best (Epoch 1 champion: 92.30% dec acc, 37.84% hard JevBench)
- Data: data/train_phase3_mixture.jsonl (231,553 pairs, stratified 85/15)
- Mode: Full Fine-Tuning (1.88B trainable transformer params, frozen embeddings & vision)
- Loss: served_dist 1.0 + NLI aux 0.15 + Brier 0.5 (cross_option 0.0)
- Optimizer: AdamW, peak lr=8e-6, cosine schedule, grad_accum=16
- Batching: token bucketing, max_tokens_per_batch=2048, max_length=2048
- Output: ckpt/gemma-4-e2b-nli-phase3 (saves epoch_2/ and updates best/ if improved)

Usage:
  uv run python scripts/launch_phase3_epoch2.py
  uv run python scripts/launch_phase3_epoch2.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "results" / "phase3_epoch2.pid"
LOG_FILE = REPO_ROOT / "results" / "phase3_epoch2_train.log"

TRAIN_CMD = [
    sys.executable, "finetune.py",
    "--data", "data/train_phase3_mixture.jsonl",
    "--base-model", "ckpt/gemma-4-e2b-nli-phase3/best",
    "--out-dir", "ckpt/gemma-4-e2b-nli-phase3",
    "--full-fine-tune",
    "--served-dist-weight", "1.0",
    "--cross-option-weight", "0.0",
    "--nli-aux-weight", "0.15",
    "--brier-weight", "0.5",
    "--epochs", "1",
    "--start-epoch", "1",
    "--lr", "8e-6",
    "--grad-accum", "16",
    "--token-bucketing",
    "--max-tokens-per-batch", "2048",
    "--max-length", "2048",
    "--seed", "42",
]


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 1 << 30


def check_preconditions() -> None:
    for path in (
        REPO_ROOT / "data" / "train_phase3_mixture.jsonl",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-phase3" / "best" / "model.safetensors",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-phase3" / "best" / "head_weights.pt",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-phase3" / "best" / "eval_report.json",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Precondition missing: {path.relative_to(REPO_ROOT)}")


def launch(dry_run: bool) -> None:
    check_preconditions()
    print("$ " + " ".join(TRAIN_CMD))
    if dry_run:
        print("[dry-run] Preconditions OK; not launching.")
        return

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(
        TRAIN_CMD,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"Launched Phase-3 Epoch 2 training: PID {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Phase-3 Epoch 2 training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-free-mib", type=int, default=4000)
    args = parser.parse_args()

    gpu = gpu_used_mib()
    if gpu >= args.gpu_free_mib and not args.dry_run:
        print(f"WARNING: GPU memory in use: {gpu} MiB (threshold: {args.gpu_free_mib} MiB). Aborting.")
        return 1

    launch(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
