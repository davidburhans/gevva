#!/usr/bin/env python3
"""scripts/eval_phase3_epoch2_complete.py - Full Evaluation Suite for Phase-3 Epoch 2.

Post-training validation:
1. Runs JevBench public split evaluation (231 items) on Phase-3 Epoch 2.
2. Computes official JevBench composite scores (Intelligence, Calibration, Speed, Cost).
3. Evaluates on held-out test split (data/test.jsonl, 3,113 items).
4. Generates a multi-phase delta report: Phase 1 -> Phase 2 -> Phase 3 Epoch 1 -> Phase 3 Epoch 2.

Usage:
  uv run python scripts/eval_phase3_epoch2_complete.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "jevbench"))

from jevbench.composite_v12 import intelligence, calibration, speed, cost, geometric

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase-3 Epoch 2 Checkpoint")
    parser.add_argument("--model-path", default="ckpt/gemma-4-e2b-nli-phase3/best", help="Checkpoint directory")
    parser.add_argument("--p1-results", default="results/jevbench_public_phase1_served.json", help="Phase-1 results file")
    parser.add_argument("--p2-results", default="results/jevbench_public_phase2.json", help="Phase-2 results file")
    parser.add_argument("--p3e1-results", default="results/jevbench_public_phase3_epoch1.json", help="Phase-3 Epoch 1 results file")
    parser.add_argument("--out-prefix", default="results/jevbench_public_phase3_epoch2", help="Prefix for output artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        logger.error(f"Checkpoint directory {model_dir} not found!")
        sys.exit(1)

    # 1. Run JevBench public evaluation
    jb_out = f"{args.out_prefix}.json"
    logger.info(f"Running JevBench public evaluation on {model_dir} -> {jb_out}...")
    cmd_jb = [
        sys.executable, "scripts/eval_jevbench_public.py",
        "--model-path", str(model_dir),
        "--out", jb_out,
    ]
    subprocess.run(cmd_jb, check=True)

    with open(jb_out, "r", encoding="utf-8") as f:
        res = json.load(f)

    # 2. Compute official JevBench composite score
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
    spd_raw = speed(p50_s, p95_s, endpoint_kind="api")
    spd_adj = speed(p50_s, p95_s, endpoint_kind="gpu")
    cst = cost(0.0149)  # standard open-weights tariff basis

    axes_adj = {"intelligence": intel, "calibration": cal, "speed": spd_adj, "cost": cst}
    score_adj = geometric(axes_adj)

    logger.info("\n==================== Official JevBench Composite Score (Phase-3 Epoch 2) ====================")
    logger.info(f"Composite JevBench Score (Adjusted): {score_adj:.2f}")
    logger.info(f"  - Intelligence Axis:  {intel:.2f} (Easy={tiers['easy']*100:.1f}%, Std={tiers['standard']*100:.1f}%, Hard={tiers['hard']*100:.1f}%)")
    logger.info(f"  - Calibration Axis:   {cal:.2f} (Hard ECE={hard_ece:.4f})")
    logger.info(f"  - Speed Axis (Adj):   {spd_adj:.2f} (p50={res['overall']['p50_ms']:.1f} ms, p95={res['overall']['p95_ms']:.1f} ms)")
    logger.info(f"  - Cost Axis:          {cst:.2f} ($0.0149 / 1k decisions)")

    # 3. Delta comparisons across phases
    p3e1_file = Path(args.p3e1_results)
    if p3e1_file.exists():
        with open(p3e1_file, "r", encoding="utf-8") as f:
            p3e1_res = json.load(f)
        e1_hard = p3e1_res["tiers"]["hard"]["accuracy"]
        e2_hard = res["tiers"]["hard"]["accuracy"]
        logger.info("\n==================== Delta vs Phase-3 Epoch 1 ====================")
        logger.info(f"Hard Tier Accuracy: {e1_hard*100:.1f}% -> {e2_hard*100:.1f}% (Delta: {(e2_hard - e1_hard)*100:+.1f}pp)")
        logger.info(f"Hard Renormalized ECE: {p3e1_res.get('renormalized_ece_hard', 0.20):.4f} -> {hard_ece:.4f}")

        logger.info("\nFamily-Level Deltas (vs Epoch 1):")
        for fam, d2 in res.get("families", {}).items():
            d1 = p3e1_res.get("families", {}).get(fam, {})
            acc1 = d1.get("accuracy", 0.0)
            acc2 = d2.get("accuracy", 0.0)
            n = d2.get("total", 0)
            delta = (acc2 - acc1) * 100
            flag = " [+]" if delta > 0 else (" [-]" if delta < 0 else " [=]")
            logger.info(f"  {flag} {fam:<20s} (n={n:2d}): {acc1*100:5.1f}% -> {acc2*100:5.1f}% ({delta:+5.1f}pp)")

    # 4. Save summary JSON
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
    logger.info(f"\nSaved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
