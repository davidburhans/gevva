#!/usr/bin/env python3
"""gate_decision.py - Pre-registered synthetic-data gate. NO post-hoc discretion.

PASS iff ALL hold on the report-only test split:
  1. acc_B > acc_A
  2. McNemar paired test p < 0.05 on discordant pairs (per-item logs, joined on id)
  3. ece_B <= ece_A + 0.01 (calibration non-regression)

Usage:
    uv run python scripts/gate_decision.py --arm-a ckpt/shakedown_A --arm-b ckpt/shakedown_B \
        --out results/gate_decision.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar p via chi2(1) with continuity correction.

    P(chi2_1 > x) = erfc(sqrt(x/2)). b/c = counts of A-right/B-wrong and A-wrong/B-right.

    Example: round(mcnemar_p(1, 9), 4) == 0.0269
    """
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(chi2 / 2.0))


def ece_from_items(items: List[Dict], n_bins: int = 15) -> float:
    """ECE over (confidence, correct) pairs with equal-width bins on [1/3, 1]."""
    lo, width = 1.0 / 3.0, (1.0 - 1.0 / 3.0) / n_bins
    total = len(items)
    ece = 0.0
    for k in range(n_bins):
        bin_lo = lo + k * width
        members = [i for i in items if bin_lo <= i["confidence"] < bin_lo + width
                   or (k == n_bins - 1 and i["confidence"] <= 1.0 and i["confidence"] >= bin_lo)]
        if not members:
            continue
        acc = sum(1 for i in members if i["correct"]) / len(members)
        conf = sum(i["confidence"] for i in members) / len(members)
        ece += len(members) / total * abs(acc - conf)
    return ece


def _load_items(arm_dir: str) -> Dict[str, Dict]:
    path = Path(arm_dir) / "test_items.jsonl"
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}


def evaluate_gate(arm_a_dir: str, arm_b_dir: str, alpha: float = 0.05,
                  ece_tolerance: float = 0.01) -> Dict:
    items_a = _load_items(arm_a_dir)
    items_b = _load_items(arm_b_dir)
    shared = sorted(set(items_a) & set(items_b))
    if len(shared) < 100:
        raise ValueError(f"paired test items too few: {len(shared)}")

    b = sum(1 for i in shared if items_a[i]["correct"] and not items_b[i]["correct"])
    c = sum(1 for i in shared if not items_a[i]["correct"] and items_b[i]["correct"])
    acc_a = sum(items_a[i]["correct"] for i in shared) / len(shared)
    acc_b = sum(items_b[i]["correct"] for i in shared) / len(shared)
    ece_a = ece_from_items([items_a[i] for i in shared])
    ece_b = ece_from_items([items_b[i] for i in shared])
    p_value = mcnemar_p(b, c)

    passed = bool(acc_b > acc_a and p_value < alpha and ece_b <= ece_a + ece_tolerance)
    return {
        "gate": "synthetic_data_inclusion",
        "decision_rule": "acc_B > acc_A AND McNemar p < 0.05 AND ece_B <= ece_A + 0.01",
        "passed": passed,
        "n_paired": len(shared),
        "acc_armA_clean": round(acc_a, 4),
        "acc_armB_synthetic": round(acc_b, 4),
        "ece_armA": round(ece_a, 4),
        "ece_armB": round(ece_b, 4),
        "discordant_aRight_bWrong": b,
        "discordant_aWrong_bRight": c,
        "mcnemar_p": round(p_value, 6),
        "alpha": alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", required=True, help="Dir with clean arm's test_items.jsonl")
    parser.add_argument("--arm-b", required=True, help="Dir with synthetic arm's test_items.jsonl")
    parser.add_argument("--out", default="results/gate_decision.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    verdict = evaluate_gate(args.arm_a, args.arm_b, alpha=args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    print("GATE:", "PASS - synthetic data joins the flagship mix" if verdict["passed"]
          else "FAIL - flagship trains clean-only")
    raise SystemExit(0 if verdict["passed"] else 3)


if __name__ == "__main__":
    main()
