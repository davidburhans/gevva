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
import sys
from pathlib import Path

# Bootstrap repo root: the chain invokes this as `python scripts/eval_ragtruth.py`,
# which puts scripts/ (not the repo root) on sys.path - `research.adapters` then
# fails (2026-09-30 e2b-cal battery WARN).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_FILES = [
    REPO_ROOT / "suite-0.2" / "selected-rows.jsonl.gz",
    REPO_ROOT / "suite-0.2" / "added-rows.jsonl.gz",  # RAGTruth & later additions live here
]


def load_items() -> list:
    items = []
    for suite_path in SUITE_FILES:
        with gzip.open(suite_path, "rt", encoding="utf-8") as f:
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
                # 'noul' type: expected q is a bool (True = hallucination present) or yes/no string
                exp = (r.get("expected") or {}).get("q")
                if isinstance(exp, bool):
                    hallucinated = exp
                elif isinstance(exp, str):
                    hallucinated = exp.lower() == "yes"
                else:
                    continue
                items.append({"premise": prompt, "hypothesis": response, "gold": int(hallucinated)})
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
    parser.add_argument("--served-framing", action="store_true",
                        help="Evaluate the EXACT served path: premise=_format_state(state), hypothesis=noul instructions, "
                             "score = p_ent + 0.5*p_neu, decision at 0.5. The industry-facing metric.")
    args = parser.parse_args()

    from gemma4_cross_encoder import Gemma4CrossEncoder
    from research.adapters.gevva_decision_index_engine import _format_state

    items = load_items()
    if len(items) < 100:
        raise SystemExit(f"ABORT: loaded only {len(items)} RAGTruth items - suite shard moved?")
    enc = Gemma4CrossEncoder(args.model_path, device=args.device)

    if args.served_framing:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.build_ragtruth_served import RAGTRUTH_INSTRUCTIONS
        pairs = [(_format_state({"prompt": i["premise"], "response": i["hypothesis"]}),
                  RAGTRUTH_INSTRUCTIONS) for i in items]
        probs = np.asarray(enc.predict(pairs))
        y = np.array([i["gold"] for i in items])
        # hallucinated == the noul hypothesis is TRUE -> answer yes at p_true >= 0.5
        p_true = probs[:, 1] + 0.5 * probs[:, 2]
        pred = (p_true >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report = {
            "model_path": args.model_path,
            "framing": "served (adapter mapping, threshold 0.5, no fitted parameters)",
            "n": len(items),
            "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "yes_rate": round(float(pred.mean()), 4),
            "base_rate_hallucinated": round(float(y.mean()), 4),
            "baselines": {"random_f1": 0.4113, "e2b_v1_suite": 0.1556, "e4b_v1_suite": 0.3668},
        }
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

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
