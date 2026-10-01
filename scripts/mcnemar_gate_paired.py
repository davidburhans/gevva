#!/usr/bin/env python3
"""mcnemar_gate_paired.py - Paired McNemar test between two checkpoints on the gate set.

Reviews (M5) require paired inference before any lineage decision: single-run
gate deltas inside +/-1.4pp noise are not evidence. This script loads both
models, predicts the identical gate items in the identical framing as
eval_gate_set.py, and reports per-slice exact-McNemar (two-sided) plus the
discordant-pair counts that drive it.

Usage:
    uv run python scripts/mcnemar_gate_paired.py \
        --model-a ckpt/gevva-e2b-phase5/best --model-b ckpt/gevva-e2b-cal2/best \
        --out results/mcnemar_phase5_vs_cal2.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_EVAL = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs (binomial tail)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="Reference (champion) checkpoint")
    parser.add_argument("--model-b", required=True, help="Challenger checkpoint")
    parser.add_argument("--gate-file", default=str(GATE_EVAL))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from gemma4_cross_encoder import Gemma4CrossEncoder

    rows = [json.loads(line) for line in open(args.gate_file, encoding="utf-8")]
    if len(rows) < 200:
        raise SystemExit(f"ABORT: gate file suspiciously small ({len(rows)} rows)")
    pairs = [(r["premise"], r["hypothesis"]) for r in rows]
    golds = np.array([int(r["label"]) for r in rows])

    preds = {}
    for tag, path in (("a", args.model_a), ("b", args.model_b)):
        enc = Gemma4CrossEncoder(path, device=args.device)
        preds[tag] = np.argmax(np.asarray(enc.predict(pairs)), axis=-1)
        del enc

    slices = sorted({r.get("gate_slice", "ALL") for r in rows})
    report = {"model_a": args.model_a, "model_b": args.model_b, "n": len(rows), "slices": {}}
    for sl in slices + ["ALL"]:
        idx = list(range(len(rows))) if sl == "ALL" else [i for i, r in enumerate(rows) if r["gate_slice"] == sl]
        a_ok = preds["a"][idx] == golds[idx]
        b_ok = preds["b"][idx] == golds[idx]
        # b = A right B wrong; c = A wrong B right
        b = int((a_ok & ~b_ok).sum())
        c = int((~a_ok & b_ok).sum())
        p = exact_mcnemar_p(b, c)
        report["slices"][sl] = {
            "n": len(idx),
            "acc_a": round(float(a_ok.mean()), 4),
            "acc_b": round(float(b_ok.mean()), 4),
            "delta_pp": round(100 * float(b_ok.mean() - a_ok.mean()), 2),
            "a_right_b_wrong": b,
            "a_wrong_b_right": c,
            "mcnemar_p": round(p, 5),
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["slices"], indent=2))


if __name__ == "__main__":
    main()
