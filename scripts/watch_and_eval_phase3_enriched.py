#!/usr/bin/env python3
"""scripts/watch_and_eval_phase3_enriched.py - Automated Watchdog & Full Evaluation for Phase-3 Enriched FFT.

Monitors training of PID in results/phase3_enriched.pid.
Upon completion:
1. Runs JevBench public evaluation (231 items).
2. Computes official JevBench composite scores (raw and temperature-calibrated).
3. Evaluates on held-out test split (data/test.jsonl, 3,113 items).
4. Generates a multi-phase delta report: Phase 3 Epoch 2 -> Phase 3 Enriched.

Usage:
  uv run python scripts/watch_and_eval_phase3_enriched.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "jevbench"))

from jevbench.composite_v12 import intelligence, calibration, speed, cost, geometric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PID_FILE = REPO_ROOT / "results" / "phase3_enriched.pid"


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and evaluate Phase-3 Enriched FFT")
    parser.add_argument("--model-path", default="ckpt/gemma-4-e2b-nli-phase3-enriched/best", help="Model checkpoint path")
    parser.add_argument("--prev-results", default="results/jevbench_public_phase3_epoch2.json", help="Previous Phase-3 Epoch 2 results")
    parser.add_argument("--out-prefix", default="results/jevbench_public_phase3_enriched", help="Output prefix")
    args = parser.parse_args()

    # 1. Wait for training process to finish if running
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_pid_running(pid):
                logger.info(f"Detected running training process PID {pid}. Waiting for completion...")
                while is_pid_running(pid):
                    time.sleep(15)
                logger.info(f"Process {pid} completed.")
        except Exception as e:
            logger.warning(f"Error checking PID file: {e}")

    model_dir = Path(args.model_path)
    if not (model_dir / "model.safetensors").exists():
        logger.error(f"Checkpoint model.safetensors not found at {model_dir}!")
        sys.exit(1)

    logger.info(f"Proceeding with full evaluation of {model_dir}...")

    # 2. Run JevBench public evaluation
    jb_out = f"{args.out_prefix}.json"
    logger.info(f"Running JevBench public evaluation -> {jb_out}...")
    cmd_jb = [
        sys.executable, "scripts/eval_jevbench_public.py",
        "--model-path", str(model_dir),
        "--out", jb_out,
    ]
    subprocess.run(cmd_jb, check=True)

    with open(jb_out, "r", encoding="utf-8") as f:
        res = json.load(f)

    # 3. Compute official JevBench composite score
    tiers = {
        "easy": res["tiers"]["easy"]["accuracy"],
        "standard": res["tiers"]["standard"]["accuracy"],
        "hard": res["tiers"]["hard"]["accuracy"],
    }
    intel = intelligence(tiers)
    hard_ece = res.get("renormalized_ece_hard", 0.20)
    cal = calibration(hard_ece)
    p50_s = res["overall"]["p50_ms"] / 1000.0
    p95_s = res["overall"]["p95_ms"] / 1000.0
    spd_adj = speed(p50_s, p95_s, endpoint_kind="gpu")
    cst = cost(0.0149)  # standard open-weights tariff basis

    axes_adj = {"intelligence": intel, "calibration": cal, "speed": spd_adj, "cost": cst}
    score_adj = geometric(axes_adj)

    logger.info("\n==================== Official JevBench Composite Score (Phase-3 Enriched) ====================")
    logger.info(f"Composite JevBench Score (Adjusted): {score_adj:.2f}")
    logger.info(f"  - Intelligence Axis:  {intel:.2f} (Easy={tiers['easy']*100:.1f}%, Std={tiers['standard']*100:.1f}%, Hard={tiers['hard']*100:.1f}%)")
    logger.info(f"  - Calibration Axis:   {cal:.2f} (Hard ECE={hard_ece:.4f})")
    logger.info(f"  - Speed Axis (Adj):   {spd_adj:.2f} (p50={res['overall']['p50_ms']:.1f} ms, p95={res['overall']['p95_ms']:.1f} ms)")
    logger.info(f"  - Cost Axis:          {cst:.2f} ($0.0149 / 1k decisions)")

    # 4. Compare with Phase-3 Epoch 2 baseline
    prev_file = Path(args.prev_results)
    if prev_file.exists():
        with open(prev_file, "r", encoding="utf-8") as f:
            prev_res = json.load(f)
        p_hard = prev_res["tiers"]["hard"]["accuracy"]
        curr_hard = res["tiers"]["hard"]["accuracy"]
        logger.info("\n==================== Delta vs Phase-3 Epoch 2 Baseline ====================")
        logger.info(f"Hard Tier Accuracy: {p_hard*100:.1f}% -> {curr_hard*100:.1f}% (Delta: {(curr_hard - p_hard)*100:+.1f}pp)")
        logger.info(f"Hard Renormalized ECE: {prev_res.get('renormalized_ece_hard', 0.33):.4f} -> {hard_ece:.4f}")

        logger.info("\nFamily-Level Deltas:")
        for fam, d2 in res.get("families", {}).items():
            d1 = prev_res.get("families", {}).get(fam, {})
            acc1 = d1.get("accuracy", 0.0)
            acc2 = d2.get("accuracy", 0.0)
            n = d2.get("total", 0)
            delta = (acc2 - acc1) * 100
            flag = " [+]" if delta > 0 else (" [-]" if delta < 0 else " [=]")
            logger.info(f"  {flag} {fam:<20s} (n={n:2d}): {acc1*100:5.1f}% -> {acc2*100:5.1f}% ({delta:+5.1f}pp)")

    # 5. Held-out test split evaluation
    logger.info("\n--- Running held-out test split evaluation on data/test.jsonl ---")
    cmd_test = [
        sys.executable, "scripts/eval_test_split.py",
        "--model-path", str(model_dir),
    ]
    subprocess.run(cmd_test, check=True)

    # 6. Save summary JSON
    summary = {
        "model_path": str(model_dir),
        "composite_score": score_adj,
        "axes": axes_adj,
        "tiers": tiers,
        "hard_ece": hard_ece,
        "p50_ms": res["overall"]["p50_ms"],
        "p95_ms": res["overall"]["p95_ms"],
    }
    summary_path = f"{args.out_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved full evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
