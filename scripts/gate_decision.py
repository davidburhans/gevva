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
from typing import Dict, List, Tuple, Optional


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar p. Exact binomial when discordant counts are small (< 25),
    continuity-corrected chi2 otherwise (audit: chi2 breaks down on small counts).

    Example: round(mcnemar_p(1, 9), 4) == 0.0215  # exact binomial
    """
    if b + c == 0:
        return 1.0
    if b + c < 25:
        from math import comb
        n = b + c
        tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
        return min(1.0, 2.0 * tail / 2 ** n)
    chi2 = max(0, abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(chi2 / 2.0))


def ece_from_items(items: List[Dict], n_bins: int = 15) -> float:
    """ECE over (confidence, correct) pairs; confidences below 1/3 are clamped into
    the first bin (impossible for a 3-class softmax, defensive for other sources)."""
    lo, hi = 1.0 / 3.0, 1.0
    width = (hi - lo) / n_bins
    total = len(items)
    if total == 0:
        return 0.0
    ece = 0.0
    for k in range(n_bins):
        bin_lo, bin_hi = lo + k * width, lo + (k + 1) * width
        members = [i for i in items if (bin_lo <= i["confidence"] < bin_hi)
                   or (k == n_bins - 1 and bin_lo <= i["confidence"] <= hi)
                   or (k == 0 and i["confidence"] < lo)]
        if not members:
            continue
        acc = sum(1 for i in members if i["correct"]) / len(members)
        conf = sum(min(max(i["confidence"], lo), hi) for i in members) / len(members)
        ece += len(members) / total * abs(acc - conf)
    return ece


def _load_items(arm_dir: str) -> Dict[str, Dict]:
    path = Path(arm_dir) / "test_items.jsonl"
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}


def position_bias_index(preds: List[int], k_options: int = 3) -> float:
    """Computes Position Bias Index (PBI) across option choices.

    PBI = 1/K * sum_{k=0}^{K-1} | P(pred == k) - 1/K |

    Attribution:
        Position Bias Index formulation inspired by SOTA zero-shot decision engines
        (SemIf / decider). Measures deviation from uniform distribution across option positions.
    """
    if not preds or k_options <= 1:
        return 0.0
    n = len(preds)
    expected_prob = 1.0 / k_options
    counts = [sum(1 for p in preds if p == k) for k in range(k_options)]
    pbi = sum(abs(c / n - expected_prob) for c in counts) / k_options
    return float(pbi)


def evaluate_gate(arm_a_dir: str, arm_b_dir: str, alpha: float = 0.05,
                  ece_tolerance: float = 0.01, pbi_tolerance: float = 0.02,
                  expected_test_sha: Optional[str] = None) -> Dict:
    items_a = _load_items(arm_a_dir)
    items_b = _load_items(arm_b_dir)
    if expected_test_sha:
        for d, items in ((arm_a_dir, items_a), (arm_b_dir, items_b)):
            got = next(iter(items.values())).get("test_sha") if items else None
            if got and got != expected_test_sha:
                raise ValueError(f"{d}: test_sha mismatch ({got} != {expected_test_sha}) - "
                                 "arms were evaluated against different test files")
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

    # Position bias check across predicted choices (if present in test items)
    preds_a = [items_a[i]["pred"] for i in shared if "pred" in items_a[i]]
    preds_b = [items_b[i]["pred"] for i in shared if "pred" in items_b[i]]
    if len(preds_a) == len(shared) and len(preds_b) == len(shared):
        pbi_a = position_bias_index(preds_a, k_options=3)
        pbi_b = position_bias_index(preds_b, k_options=3)
        pbi_passed = bool(pbi_b <= pbi_a + pbi_tolerance)
    else:
        pbi_a = 0.0
        pbi_b = 0.0
        pbi_passed = True

    passed = bool(acc_b > acc_a and p_value < alpha and ece_b <= ece_a + ece_tolerance and pbi_passed)
    return {
        "gate": "synthetic_data_inclusion",
        "decision_rule": "acc_B > acc_A AND McNemar p < 0.05 AND ece_B <= ece_A + 0.01 AND pbi_B <= pbi_A + 0.02",
        "passed": passed,
        "n_paired": len(shared),
        "acc_armA_clean": round(acc_a, 4),
        "acc_armB_synthetic": round(acc_b, 4),
        "ece_armA": round(ece_a, 4),
        "ece_armB": round(ece_b, 4),
        "pbi_armA": round(pbi_a, 4),
        "pbi_armB": round(pbi_b, 4),
        "pbi_passed": pbi_passed,
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
