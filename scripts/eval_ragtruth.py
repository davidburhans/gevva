#!/usr/bin/env python3
"""eval_ragtruth.py - Response-level RAGTruth F1 (Decision Index Cat 59 metric) for a checkpoint.

Decision Index framing: each item is (prompt=premise, response=hypothesis); the
hallucination signal is P(contradiction) from the NLI head; F1 is computed on the
hallucinated class (Decision Index random baseline: 41.13% F1).

Usage:
    uv run python scripts/eval_ragtruth.py --model-path ckpt/X/best --out results/X_ragtruth.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE = REPO_ROOT / "suite-0.2" / "selected-rows.jsonl.gz"


def load_items() -> list:
    items = []
    with gzip.open(SUITE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_evaluation", {}).get("catalog_id") != 59:
                continue
            state = r.get("state", {})
            q = list(r["questions"].values())[0]
            prompt = state.get("prompt") if isinstance(state, dict) else None
            response = state.get("response") if isinstance(state, dict) else None
            if not prompt or not response:
                continue
            # 'noul' type: expected answer "yes" = hallucination present
            exp = (r.get("expected") or {}).get("q") or (r.get("gold") or {}).get("q")
            if exp not in ("yes", "no"):
                continue
            items.append({"premise": prompt, "hypothesis": response, "gold": 1 if exp == "yes" else 0})
    return items


def f1_on_positive(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": thr, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from gemma4_cross_encoder import Gemma4CrossEncoder

    items = load_items()
    enc = Gemma4CrossEncoder(args.model_path, device=args.device)
    probs = np.asarray(enc.predict([(i["premise"], i["hypothesis"]) for i in items]))
    y = np.array([i["gold"] for i in items])
    p_con = probs[:, 0]  # contradiction = hallucination signal

    best = max((f1_on_positive(y, p_con, t) for t in np.arange(0.1, 0.91, 0.05)),
               key=lambda m: m["f1"])
    report = {
        "model_path": args.model_path,
        "n": len(items),
        "base_rate_hallucinated": round(float(y.mean()), 4),
        "f1_at_0.5": f1_on_positive(y, p_con, 0.5),
        "f1_best_threshold": best,
        "baselines": {"e2b_v1_raw": 0.1556, "e4b_v1_raw": 0.3668, "random": 0.4113},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
