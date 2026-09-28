#!/usr/bin/env python3
"""run_exp01.py - EXP-01 driver: trains both arms, runs the paired gate eval,
and writes the pre-registered gate decision.

Stages (sequential, single GPU):
  1. Train Arm A (standard balanced curriculum)  -> ckpt/exp01_armA
  2. Train Arm B (aggressively pruned curriculum) -> ckpt/exp01_armB
  3. Paired evaluation on data/exp01_gate_eval.jsonl (floor/medium/hard slices)
  4. Gate decision -> results/exp01_gate_decision.json

Gates evaluated here (paired set): floor invariance + hard-tier lift with
McNemar. Harness gates (BANKING77 / When2Call / BFCL / JevBench Hard) are
recorded as pending and must be run via the existing harnesses afterwards.

Usage:
    nohup uv run python scripts/run_exp01.py > results/exp01_run.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    "armA": REPO_ROOT / "data" / "exp01_armA_train.jsonl",
    "armB": REPO_ROOT / "data" / "exp01_armB_train.jsonl",
}
CKPTS = {
    "armA": REPO_ROOT / "ckpt" / "exp01_armA",
    "armB": REPO_ROOT / "ckpt" / "exp01_armB",
}
GATE_EVAL = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"
DECISION_PATH = REPO_ROOT / "results" / "exp01_gate_decision.json"
TRAIN_LOGS = {
    "armA": REPO_ROOT / "results" / "exp01_train_armA.log",
    "armB": REPO_ROOT / "results" / "exp01_train_armB.log",
}

BASE_MODEL = "google/gemma-4-E2B-it"  # pre-registered in §13
COMMON_TRAIN_ARGS = [
    "--base-model", BASE_MODEL,
    "--full-fine-tune",
    "--use-8bit-adam",
    "--epochs", "2",
    "--lr", "3.0e-6",
    "--head-lr", "1.0e-4",
    "--grad-accum", "16",
    "--token-bucketing",
    "--max-tokens-per-batch", "2048",
    "--max-length", "1024",
    "--brier-weight", "0.5",
    "--served-dist-weight", "1.0",
    "--cross-option-weight", "0.0",
    "--nli-aux-weight", "0.25",
    "--decision-temp", "1.0",
    "--val-ratio", "0.03",
    "--label-convention", "ours",
    "--image-root", ".",
    "--seed", "42",
]


def _train_arm(arm: str) -> None:
    log_path = TRAIN_LOGS[arm]
    cmd = [
        sys.executable, str(REPO_ROOT / "finetune.py"),
        "--data", str(ARMS[arm]),
        "--out-dir", str(CKPTS[arm]),
        *COMMON_TRAIN_ARGS,
    ]
    print(f"[exp01] training {arm}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{arm} training failed (see {log_path})")


def _load_gate_rows() -> List[Dict]:
    rows = []
    with open(GATE_EVAL, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _evaluate_arm(arm: str, rows: List[Dict]) -> np.ndarray:
    """Returns predicted label per row for the arm's best checkpoint."""
    from gemma4_cross_encoder import Gemma4CrossEncoder

    best = CKPTS[arm] / "best"
    print(f"[exp01] evaluating {arm} on {len(rows)} gate rows", flush=True)
    enc = Gemma4CrossEncoder(model_path=str(best), device="cuda")
    pairs = [(r["premise"], r["hypothesis"]) for r in rows]
    probs = enc.predict(pairs)
    del enc
    import torch
    torch.cuda.empty_cache()
    return np.argmax(np.asarray(probs), axis=-1)


def _mcnemar_exact(better: int, worse: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _slice_report(rows: List[Dict], preds_a: np.ndarray, preds_b: np.ndarray) -> Dict:
    golds = np.array([int(r["label"]) for r in rows])
    report: Dict = {}
    slices = sorted({r["gate_slice"] for r in rows})
    for gate_slice in slices:
        idx = [i for i, r in enumerate(rows) if r["gate_slice"] == gate_slice]
        a_right = b_right = 0
        a_only = b_only = 0
        for i in idx:
            a_ok = bool(preds_a[i] == golds[i])
            b_ok = bool(preds_b[i] == golds[i])
            a_right += a_ok
            b_right += b_ok
            if a_ok and not b_ok:
                a_only += 1
            elif b_ok and not a_ok:
                b_only += 1
        report[gate_slice] = {
            "n": len(idx),
            "acc_armA": round(a_right / len(idx), 4),
            "acc_armB": round(b_right / len(idx), 4),
            "delta_pp": round(100 * (b_right - a_right) / len(idx), 2),
            "discordant_aRight_bWrong": a_only,
            "discordant_aWrong_bRight": b_only,
            "mcnemar_p": round(_mcnemar_exact(b_only, a_only), 6),
        }
        per_source = defaultdict(lambda: {"a": 0, "b": 0, "n": 0})
        for i in idx:
            per_source[rows[i]["source"]]["n"] += 1
            per_source[rows[i]["source"]]["a"] += int(preds_a[i] == golds[i])
            per_source[rows[i]["source"]]["b"] += int(preds_b[i] == golds[i])
        report[gate_slice]["per_source"] = {
            s: {"n": v["n"], "acc_armA": v["a"] / v["n"], "acc_armB": v["b"] / v["n"]}
            for s, v in sorted(per_source.items())
        }
    return report


def _gate_decision(report: Dict) -> Dict:
    floor = report.get("floor", {})
    hard = report.get("hard", {})
    floor_ratio = (
        floor["acc_armB"] / floor["acc_armA"] if floor.get("acc_armA") else 0.0
    )
    gates = {
        "gate1_floor_invariance": {
            "rule": "armB floor accuracy >= 99.5% of armA",
            "acc_armA": floor.get("acc_armA"),
            "acc_armB": floor.get("acc_armB"),
            "ratio": round(floor_ratio, 4),
            "passed": bool(floor_ratio >= 0.995),
        },
        "gate3_hard_lift": {
            "rule": "armB hard-slice accuracy > armA (McNemar p < 0.05)",
            "delta_pp": hard.get("delta_pp"),
            "mcnemar_p": hard.get("mcnemar_p"),
            "passed": bool(hard.get("delta_pp", 0) > 0 and hard.get("mcnemar_p", 1) < 0.05),
        },
        "harness_gates_pending": [
            "gate2_medium_lift: BANKING77 / When2Call / BFCL via engine harnesses",
            "gate3_jevbench_hard: scripts/eval_jevbench_public.py per arm",
        ],
    }
    paired_pass = gates["gate1_floor_invariance"]["passed"] and gates["gate3_hard_lift"]["passed"]
    return {
        "experiment": "EXP-01 curriculum pruning A/B (pre-registered §13, adapted per manifest)",
        "decision_rule_paired": "accept Arm B master curriculum iff gate1 AND gate3 pass; harness gates required before final acceptance",
        "paired_pass": paired_pass,
        "gates": gates,
    }


def main() -> None:
    for arm in ("armA", "armB"):
        if not (CKPTS[arm] / "best").exists():
            _train_arm(arm)
        else:
            print(f"[exp01] {arm} checkpoint exists; skipping training", flush=True)

    rows = _load_gate_rows()
    preds_a = _evaluate_arm("armA", rows)
    preds_b = _evaluate_arm("armB", rows)
    report = _slice_report(rows, preds_a, preds_b)
    decision = _gate_decision(report)
    decision["slices"] = report
    decision["train_args"] = " ".join(COMMON_TRAIN_ARGS)
    decision["base_model"] = BASE_MODEL

    with open(DECISION_PATH, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in decision.items() if k != "slices"}, indent=2))
    print(f"[exp01] decision written to {DECISION_PATH}")


if __name__ == "__main__":
    main()
