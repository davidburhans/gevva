#!/usr/bin/env python3
"""scripts/launch_multimodal_training.py - Continual Full Fine-Tuning for Gevva Multimodal & Multi-Image.

Targets:
- Base: ckpt/gevva-e2b (Starts from our World Champion Gevva e2b checkpoint)
- Training Mode: Full Fine-Tuning (1.88B trainable parameters including embed_vision)
- Data: data/train_multimodal_mixture.jsonl (20,070 rows: 50.2% multimodal/multi-image, 49.8% anchor replay)
- Loss: served_dist 1.0 + cross_option 0.0 + NLI aux 0.25 + Brier 0.5
- Optimizer: AdamW, peak lr=2.5e-6, cosine schedule, grad_accum=16
- Token Bucketing: Dynamic token bucketing (max_tokens_per_batch=2048, max_length=2048)
- Output: ckpt/gevva-e2b-multimodal (Preserves ckpt/gevva-e2b intact)

Usage:
  # Check preconditions without launching:
  uv run python scripts/launch_multimodal_training.py --dry-run

  # Launch training on RTX 5090:
  uv run python scripts/launch_multimodal_training.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "results" / "multimodal_fft.pid"
LOG_FILE = REPO_ROOT / "results" / "multimodal_fft_train.log"

TRAIN_CMD = [
    sys.executable, "finetune.py",
    "--data", "data/train_multimodal_mixture.jsonl",
    "--base-model", "ckpt/gevva-e2b",
    "--out-dir", "ckpt/gevva-e2b-multimodal",
    "--full-fine-tune",
    "--val-ratio", "0.05",
    "--served-dist-weight", "1.0",
    "--cross-option-weight", "0.0",
    "--nli-aux-weight", "0.25",
    "--brier-weight", "0.5",
    "--epochs", "2",
    "--lr", "2.5e-6",
    "--grad-accum", "16",
    "--token-bucketing",
    "--max-tokens-per-batch", "2048",
    "--max-length", "2048",
    "--image-root", ".",
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
    data_path = REPO_ROOT / "data" / "train_multimodal_mixture.jsonl"
    model_path = REPO_ROOT / "ckpt" / "gevva-e2b"
    for path in (data_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"precondition missing: {path.relative_to(REPO_ROOT)}")


def launch(dry_run: bool) -> None:
    check_preconditions()
    print("$ " + " ".join(TRAIN_CMD))
    if dry_run:
        print("[dry-run] Preconditions verified successfully; multimodal mixture and champion model are staged. NOT launching training.")
        return

    used = gpu_used_mib()
    print(f"Current GPU VRAM in use: {used} MiB")

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(
        TRAIN_CMD,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"Launched Multimodal Full Fine-Tuning: pid {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Multimodal Full Fine-Tuning")
    parser.add_argument("--dry-run", action="store_true", help="Check preconditions without launching")
    args = parser.parse_args()

    launch(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
