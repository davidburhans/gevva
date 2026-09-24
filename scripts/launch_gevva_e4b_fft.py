#!/usr/bin/env python3
"""scripts/launch_gevva_e4b_fft.py - Flagship Gevva e4b Full Fine-Tuning Launcher.

Architecture & Strategy:
- Model: google/gemma-4-E4B-it (4.5B params, 42 transformer layers, hidden 2560, intermediate 10240)
- Frozen Components: Embeddings (671M token + 2.82B per-layer) + Vision Tower (167M)
- Trainable Backbone: 4.28B transformer parameters across all 42 layers
- Mode: Full Fine-Tuning with Group-Atomic Cross-Option Ranking Loss
- Curriculum: data/train_phase3_enriched.jsonl (243,916 pairs, 41 sources)
- Loss: served_dist 1.0 + cross_option 0.5 + nli_aux 0.15 + brier 0.5
- Output: ckpt/gevva-e4b

Usage:
  uv run python scripts/launch_gevva_e4b_fft.py --dry-run
  uv run python scripts/launch_gevva_e4b_fft.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "results" / "gevva_e4b_train.pid"
LOG_FILE = REPO_ROOT / "results" / "gevva_e4b_train.log"

TRAIN_CMD = [
    sys.executable, "finetune.py",
    "--data", "data/train_phase3_enriched.jsonl",
    "--base-model", "google/gemma-4-E4B-it",
    "--out-dir", "ckpt/gevva-e4b",
    "--full-fine-tune",
    "--served-dist-weight", "1.0",
    "--cross-option-weight", "0.5",
    "--nli-aux-weight", "0.15",
    "--brier-weight", "0.5",
    "--epochs", "2",
    "--lr", "1.0e-5",
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
    data_path = REPO_ROOT / "data" / "train_phase3_enriched.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file missing: {data_path}")


def launch(dry_run: bool) -> None:
    check_preconditions()
    print("$ " + " ".join(TRAIN_CMD))
    if dry_run:
        print("[dry-run] Preconditions verified. Launcher is staged and ready for manual trigger.")
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
    print(f"Launched Gevva e4b training: pid {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Gevva e4b Full Fine-Tuning")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-free-mib", type=int, default=4000)
    args = parser.parse_args()

    gpu = gpu_used_mib()
    if gpu >= args.gpu_free_mib and not args.dry_run:
        print(f"WARNING: GPU memory in use: {gpu} MiB (threshold: {args.gpu_free_mib} MiB). Aborting.")
        return 1

    launch(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
