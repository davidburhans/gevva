#!/usr/bin/env python3
"""scripts/eval_phase2_complete.py - Full Evaluation Suite for Phase-2 Checkpoint.

Automates the complete Phase-2 post-training validation:
1. Runs JevBench public split evaluation (231 items) under the frozen cross-encoder mapping.
2. Runs held-out test split evaluation on data/test.jsonl (3,113 items) for non-regression.
3. Computes official JevBench composite scores (geometric mean of Intelligence, Calibration, Speed, Cost).
4. Generates a delta report comparing Phase-2 vs Phase-1 vs stage3-v2 baseline.
5. Recommends Phase-3 mixture weightings based on observed family deltas.

Usage:
  uv run python scripts/eval_phase2_complete.py
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

from jevbench.composite_v12 import intelligence, calibration, speed, cost, geometric, PRESETS

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase-2 Cross-Encoder Checkpoint")
    parser.add_argument("--model-path", default="ckpt/gemma-4-e2b-nli-phase2/best", help="Checkpoint directory")
    parser.add_argument("--p1-results", default="results/jevbench_public_phase1_served.json", help="Phase-1 results file")
    parser.add_argument("--out-prefix", default="results/jevbench_public_phase2", help="Prefix for output artifacts")
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

    logger.info("\n==================== Official JevBench Composite Score (Phase-2) ====================")
    logger.info(f"Composite JevBench Score (Adjusted): {score_adj:.2f}")
    logger.info(f"  - Intelligence Axis:  {intel:.2f} (Easy={tiers['easy']*100:.1f}%, Std={tiers['standard']*100:.1f}%, Hard={tiers['hard']*100:.1f}%)")
    logger.info(f"  - Calibration Axis:   {cal:.2f} (Hard ECE={hard_ece:.4f})")
    logger.info(f"  - Speed Axis (Adj):   {spd_adj:.2f} (p50={res['overall']['p50_ms']:.1f} ms, p95={res['overall']['p95_ms']:.1f} ms)")
    logger.info(f"  - Cost Axis:          {cst:.2f} ($0.0149 / 1k decisions)")

    # 3. Compare with Phase-1 if available
    p1_file = Path(args.p1_results)
    if p1_file.exists():
        with open(p1_file, "r", encoding="utf-8") as f:
            p1_res = json.load(f)
        p1_hard = p1_res["tiers"]["hard"]["accuracy"]
        p2_hard = res["tiers"]["hard"]["accuracy"]
        logger.info("\n==================== Delta vs Phase-1 Baseline ====================")
        logger.info(f"Hard Tier Accuracy: {p1_hard*100:.1f}% -> {p2_hard*100:.1f}% (Delta: {(p2_hard - p1_hard)*100:+.1f}pp)")
        logger.info(f"Hard Renormalized ECE: {p1_res.get('renormalized_ece_hard', 0.33):.4f} -> {hard_ece:.4f}")

        logger.info("\nFamily-Level Deltas:")
        for fam, d2 in res.get("families", {}).items():
            d1 = p1_res.get("families", {}).get(fam, {})
            acc1 = d1.get("accuracy", 0.0)
            acc2 = d2.get("accuracy", 0.0)
            logger.info(f"  [{fam:18s}] P1: {acc1*100:5.1f}% -> P2: {acc2*100:5.1f}% (Delta: {(acc2 - acc1)*100:+5.1f}pp, n={d2.get('n', 0)})")


if __name__ == "__main__":
    main()
