#!/usr/bin/env python3
"""scripts/launch_phase4.py - Continual Full Fine-Tuning for Gevva Phase-4.

Targets Weak-Family Remediation & Long-Horizon Grounding:
- Base: ckpt/gevva-e2b (Starts from our World Champion Gevva e2b checkpoint)
- Training Mode: Full Fine-Tuning (1.88B trainable transformer parameters)
- Data: data/train_phase4_mixture.jsonl (62,600 rows: Phase-4 synthetic weak families + Phase-3 replay + clean NLI)
- Loss: served_dist 1.0 + cross_option 0.5 + NLI aux 0.15 + Brier 0.5
- Optimizer: AdamW, peak lr=1e-5, cosine schedule, grad_accum=16
- Output: ckpt/gevva-e2b-phase4

Usage:
  # Check preconditions without launching:
  uv run python scripts/launch_phase4.py --dry-run

  # Launch background training (when GPU is free):
  uv run python scripts/launch_phase4.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "results" / "phase4_fft.pid"
LOG_FILE = REPO_ROOT / "results" / "phase4_fft_train.log"

TRAIN_CMD = [
    sys.executable, "finetune.py",
    "--data", "data/train_phase4_mixture.jsonl",
    "--base-model", "ckpt/gevva-e2b",
    "--out-dir", "ckpt/gevva-e2b-phase4",
    "--full-fine-tune",
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
    data_path = REPO_ROOT / "data" / "train_phase4_mixture.jsonl"
    model_path = REPO_ROOT / "ckpt" / "gevva-e2b"
    for path in (data_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"precondition missing: {path.relative_to(REPO_ROOT)}")


def launch(dry_run: bool) -> None:
    check_preconditions()
    print("$ " + " ".join(TRAIN_CMD))
    if dry_run:
        print("[dry-run] Preconditions verified successfully; data and champion model are staged. NOT launching training.")
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
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"Launched Phase-4 Full Fine-Tuning: pid {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Phase-4 Remediation Fine-Tuning")
    parser.add_argument("--dry-run", action="store_true", help="Check preconditions without launching")
    parser.add_argument("--gpu-free-mib", type=int, default=4000, help="Max allowed GPU memory already in use")
    args = parser.parse_args()

    if args.dry_run:
        launch(dry_run=True)
        return 0

    gpu = gpu_used_mib()
    if gpu >= args.gpu_free_mib:
        print(f"WARNING: GPU memory in use: {gpu} MiB (threshold: {args.gpu_free_mib} MiB). Aborting to preserve GPU.")
        return 1

    launch(dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
