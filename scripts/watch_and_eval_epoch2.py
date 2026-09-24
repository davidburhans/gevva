#!/usr/bin/env python3
"""scripts/watch_and_eval_epoch2.py - Watch Phase 3 Epoch 2 and auto-evaluate upon completion.

Monitors the running Phase-3 Epoch 2 training process. When training completes:
1. Validates training completion and saves logs.
2. Runs the full evaluation suite (JevBench public, composite scores, deltas).
3. Evaluates on held-out test.jsonl (3,113 items) for accuracy and ECE.
4. Outputs final comparative summary across Phase 1, Phase 2, and Phase 3 (Epoch 1 & 2).

Usage:
  uv run python scripts/watch_and_eval_epoch2.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "results" / "phase3_epoch2.pid"
LOG_FILE = REPO_ROOT / "results" / "phase3_epoch2_train.log"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_training(pid: int) -> bool:
    logger.info(f"Monitoring Phase-3 Epoch 2 training process (PID {pid})...")
    start_t = time.time()
    last_log_size = 0

    while is_pid_alive(pid):
        time.sleep(30)
        if LOG_FILE.exists():
            size = LOG_FILE.stat().st_size
            if size != last_log_size:
                last_log_size = size
                # Show last line of progress
                try:
                    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                    for line in reversed(lines[-5:]):
                        if "Epoch" in line and "Step" in line:
                            logger.info(f"Progress: {line.strip()}")
                            break
                except Exception:
                    pass

    elapsed_m = (time.time() - start_t) / 60.0
    logger.info(f"Process {pid} exited after {elapsed_m:.1f} minutes.")
    return True


def run_evaluation() -> None:
    best_dir = REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-phase3" / "best"
    epoch2_dir = REPO_ROOT / "ckpt" / "gemma-4-e2b-nli-phase3" / "epoch_2"

    eval_target = best_dir if best_dir.exists() else epoch2_dir
    logger.info(f"Evaluating checkpoint: {eval_target}")

    cmd_eval = [
        sys.executable, "scripts/eval_phase3_epoch2_complete.py",
        "--model-path", str(eval_target),
        "--out-prefix", "results/jevbench_public_phase3_epoch2",
    ]
    logger.info(f"$ {' '.join(cmd_eval)}")
    subprocess.run(cmd_eval, check=True)


def main() -> None:
    if not PID_FILE.exists():
        logger.error(f"PID file {PID_FILE} does not exist!")
        sys.exit(1)

    pid = int(PID_FILE.read_text().strip())
    if is_pid_alive(pid):
        wait_for_training(pid)
    else:
        logger.info(f"Process {pid} is already finished.")

    run_evaluation()


if __name__ == "__main__":
    main()
