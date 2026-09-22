#!/usr/bin/env python3
"""scripts/launch_stage3_v2.py - Launch the production Stage 3 v2 training run.

Recipe (vs the v1 validation run):
- --warm-start ckpt/gemma-4-e2b-nli-stage2/best : continue from the flagship
  adapter instead of retraining core NLI from the base model (v1 regressed
  MNLI-m 86% -> 73% because a 12K-row mixture cannot relearn 52K-row skills).
- data/stage3_v2 : label-defect-filtered mixture (see its manifest.json).
- --checkpoint-interval 100 --resume-auto : interrupt insurance (<=8 min loss).

Modes:
  direct (default)  : preconditions checked, training launched immediately.
  --wait-for-chain  : poll until the running post-stage-2 chain exits AND the
                      GPU frees below --gpu-free-mib, then launch. Use this to
                      arm the production run the moment tonight's validation
                      run finishes (~01:00) without anyone awake.

Usage:
  uv run python scripts/launch_stage3_v2.py --dry-run
  uv run python scripts/launch_stage3_v2.py
  nohup uv run python scripts/launch_stage3_v2.py --wait-for-chain &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = REPO_ROOT / "results" / "post_stage2_chain_status.json"
PID_FILE = REPO_ROOT / "results" / "stage3_v2.pid"
LOG_FILE = REPO_ROOT / "results" / "stage3_v2_train.log"

TRAIN_CMD = [
    sys.executable, "train_cross_encoder.py",
    "--train-file", "data/stage3_v2/stage3_train.jsonl",
    "--val-file", "data/stage3_v2/stage3_val.jsonl",
    "--test-file", "data/test.jsonl",
    "--out-dir", "ckpt/gemma-4-e2b-nli-stage3-v2",
    "--warm-start", "ckpt/gemma-4-e2b-nli-stage2/best",
    "--brier-weight", "0.5",
    "--token-bucketing",
    "--max-tokens-per-batch", "8192",
    "--max-length", "16384",
    "--epochs", "2",
    "--lr", "5e-5",
    "--grad-accum", "8",
    "--log-interval", "20",
    "--checkpoint-interval", "100",
    "--resume-auto",
]


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 1 << 30  # unknown -> treat as busy


def chain_still_running() -> bool:
    if PID_FILE.parent.exists() and (REPO_ROOT / "results" / "post_stage2_chain.pid").exists():
        pid = (REPO_ROOT / "results" / "post_stage2_chain.pid").read_text().strip()
        alive = subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0
        return alive
    try:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("stage3_training", {}).get("state") == "running"


def check_preconditions() -> None:
    for path in (
        REPO_ROOT / "data" / "stage3_v2" / "stage3_train.jsonl",
        REPO_ROOT / "data" / "stage3_v2" / "stage3_val.jsonl",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-stage2" / "best" / "adapter_model.safetensors",
    ):
        if not path.exists():
            raise FileNotFoundError(f"precondition missing: {path.relative_to(REPO_ROOT)}")


def launch(dry_run: bool) -> None:
    check_preconditions()
    print("$ " + " ".join(TRAIN_CMD))
    if dry_run:
        print("[dry-run] preconditions OK; not launching.")
        return
    log = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(TRAIN_CMD, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"Launched stage-3 v2 training: pid {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch production Stage 3 v2 training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-chain", action="store_true",
                        help="Poll until the v1 chain exits and the GPU frees, then launch")
    parser.add_argument("--gpu-free-mib", type=int, default=4000,
                        help="GPU is considered free below this used-MiB threshold")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.wait_for_chain:
        print(f"Waiting for v1 chain to finish (poll {args.poll_seconds}s, gpu-free<{args.gpu_free_mib} MiB)...")
        while True:
            chain_busy = chain_still_running()
            gpu_busy = gpu_used_mib() >= args.gpu_free_mib
            if not chain_busy and not gpu_busy:
                print("Chain done and GPU free.")
                time.sleep(30)  # settle: let the chain's export stages grab the GPU if any
                launch(args.dry_run)
                return 0
            print(f"  chain_running={chain_busy} gpu_used={gpu_used_mib()} MiB - waiting")
            time.sleep(args.poll_seconds)
    launch(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
