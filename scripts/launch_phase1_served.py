#!/usr/bin/env python3
"""scripts/launch_phase1_served.py - Launch the Phase-1 served-distribution training.

Recipe (pre-registered, docs/JEVBENCH_REMEDIATION_PLAN.md Phase 1):
- data: the 91K grouped P1 mixture (typed decisions + hard synth + NLI anchors)
- warm start: stage-3 v2 best (best held-out NLI artifact, 85.67%)
- loss: served_dist 1.0 (the exact served artifact) + NLI aux 0.15 + Brier 0.5,
  cross-option margin loss ZEROED to isolate the hypothesis
- gate (tomorrow): public-231 renormalized ECE <= 0.20 AND no data/test.jsonl
  accuracy regression (McNemar) AND raw 3-class NLI ECE <= 0.08

Known gap: finetune.py has no mid-epoch resume (train_cross_encoder only);
accepted because this exact data+code path trained the P1 checkpoint without
incident, and epoch-end checkpoints still land.

Modes: direct (default) or --wait-for-generation (poll the Phase-2 generation
pid + GPU free, then launch - arms Phase 1 for the ~20:00 handoff tonight).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_PID_FILE = REPO_ROOT / "results" / "phase2_generation.pid"
PID_FILE = REPO_ROOT / "results" / "phase1_served.pid"
LOG_FILE = REPO_ROOT / "results" / "phase1_served_train.log"

TRAIN_CMD = [
    sys.executable, "finetune.py",
    "--data", "data/train_p1_mixture.jsonl",
    "--adapter", "ckpt/gemma-4-e2b-nli-stage3-v2/best",
    "--out-dir", "ckpt/gemma-4-e2b-nli-phase1-served",
    "--served-dist-weight", "1.0",
    "--cross-option-weight", "0.0",
    "--nli-aux-weight", "0.15",
    "--brier-weight", "0.5",
    "--epochs", "2",
    "--lr", "5e-5",
    "--grad-accum", "8",
    "--log-interval", "20",
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


def generation_running() -> bool:
    if not GEN_PID_FILE.exists():
        return False
    pid = GEN_PID_FILE.read_text().strip()
    return subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0


def check_preconditions() -> None:
    for path in (
        REPO_ROOT / "data" / "train_p1_mixture.jsonl",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-stage3-v2" / "best" / "adapter_model.safetensors",
        REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-stage3-v2" / "best" / "head_weights.pt",
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
    print(f"Launched Phase-1 training: pid {proc.pid}")
    print(f"Log: {LOG_FILE.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Phase-1 served-distribution training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-generation", action="store_true",
                        help="Poll until Phase-2 generation exits and the GPU frees, then launch")
    parser.add_argument("--gpu-free-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.wait_for_generation:
        print("Waiting for Phase-2 generation to finish...")
        while True:
            gen = generation_running()
            gpu = gpu_used_mib()
            if not gen and gpu < args.gpu_free_mib:
                print("Generation done and GPU free.")
                time.sleep(60)  # settle: let llama-swap unload completely
                launch(args.dry_run)
                return 0
            print(f"  generation_running={gen} gpu_used={gpu} MiB - waiting")
            time.sleep(args.poll_seconds)
    launch(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
