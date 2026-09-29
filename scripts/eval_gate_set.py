#!/usr/bin/env python3
"""eval_gate_set.py - Quick paired gate evaluation of a checkpoint on the EXP-01 gate set.

Evaluates data/exp01_gate_eval.jsonl (7,015 items: SNLI/MNLI floor, FEVER/QNLI/SciTail
medium, ANLI R1-R3 + ContractNLI hard) and writes per-slice/per-source accuracy plus
the ANLI neutral-recall check (F-07) to a JSON report.

Usage:
    uv run python scripts/eval_gate_set.py --model-path ckpt/X/best --out results/X_gate.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_EVAL = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gate-file", default=str(GATE_EVAL))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from gemma4_cross_encoder import Gemma4CrossEncoder

    rows = [json.loads(line) for line in open(args.gate_file, encoding="utf-8")]
    enc = Gemma4CrossEncoder(args.model_path, device=args.device)
    preds = np.argmax(np.asarray(enc.predict([(r["premise"], r["hypothesis"]) for r in rows])), axis=-1)
    golds = np.array([int(r["label"]) for r in rows])

    by_slice = defaultdict(lambda: [0, 0])
    by_src = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        ok = int(preds[i] == golds[i])
        by_slice[r["gate_slice"]][0] += ok
        by_slice[r["gate_slice"]][1] += 1
        by_src[r["source"]][0] += ok
        by_src[r["source"]][1] += 1

    anli_idx = [i for i, r in enumerate(rows) if r["source"].startswith("anli")]
    conf = Counter()
    for i in anli_idx:
        conf[(int(golds[i]), int(preds[i]))] += 1
    anli_recall = {
        str(g): conf.get((g, g), 0) / max(1, sum(conf.get((g, p), 0) for p in (0, 1, 2)))
        for g in (0, 1, 2)
    }

    report = {
        "model_path": args.model_path,
        "n": len(rows),
        "by_slice": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_slice.items())},
        "by_source": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_src.items())},
        "anli": {
            "n": len(anli_idx),
            "accuracy": sum(v for (g, p), v in conf.items() if g == p) / max(1, len(anli_idx)),
            "neutral_predictions": sum(v for (_, p), v in conf.items() if p == 2),
            "recall": anli_recall,
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("by_slice", "anli")}, indent=2))


if __name__ == "__main__":
    main()
