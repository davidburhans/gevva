#!/usr/bin/env python3
"""build_ragtruth_served_v2.py - RAGTruth served slice, round 2 (anti-collapse).

Round 1 (data/ragtruth_served_train.jsonl) caused base-rate collapse: one
CONSTANT hypothesis string across all 15,090 rows let the model satisfy the
Brier-dominant objective by predicting the training prior (yes_rate 0.53 ==
train base rate 0.554) with zero item-level discrimination (best-F1 fell to
the always-yes bound 0.5177). Verified 2026-09-30, results/night_calibration_report.md.

Round 2 changes, all aimed at forcing premise<->response discrimination:
  1. Template diversity: 9 hypothesis phrasings (canonical served template
     weighted 2x so the eval-time string stays in-distribution).
  2. Eval-matched base rate: ~35% positive (was 55.4%), matching the suite's
     hallucination rate so 'fire at the prior' no longer helps.
  3. Halved dose: 7,545 rows (was 15,090) to limit shortcut pressure.

Premise construction is byte-identical to serving (_format_state import), same
as round 1. Leakage audited disjoint by scripts/audit_ragtruth_leakage.py.

Usage:
    uv run python scripts/build_ragtruth_served_v2.py [--seed 47]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.adapters.gevva_decision_index_engine import _format_state  # noqa: E402

CANONICAL = "The response contains content that is not supported by the context in the prompt."
TEMPLATES = [
    CANONICAL,  # served string, double weight
    CANONICAL,
    "The response includes claims that the prompt's context does not support.",
    "Parts of the response are not grounded in the provided context.",
    "The response states things that cannot be verified from the prompt alone.",
    "Some content in the response contradicts or goes beyond the given context.",
    "The response is not fully entailed by the information in the prompt.",
    "The response fabricates or assumes details absent from the context.",
    "The prompt does not provide evidence for everything the response claims.",
]
TARGET_POS = 2641
TARGET_NEG = 4904  # 35.0% / 65.0% of 7545, matching the suite base rate 0.349


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "ragtruth_train.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "ragtruth_served_train_v2.jsonl"))
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    pos, neg = [], []
    for line in open(args.raw):
        r = json.loads(line)
        m = re.match(r"Premise: (.*?)Hypothesis: (.*)", r.get("text", ""), re.S)
        if not m:
            continue
        premise = _format_state({"prompt": m.group(1).strip(), "response": m.group(2).strip()})
        row = {"premise": premise, "label": r["label"], "raw_text": r["text"]}
        (pos if r["label"] == 1 else neg).append(row)
    if len(pos) < TARGET_POS or len(neg) < TARGET_NEG:
        raise SystemExit(f"ABORT: pool too small pos={len(pos)} neg={len(neg)}")

    rng = random.Random(args.seed)
    rows = []
    for pool, n in ((pos, TARGET_POS), (neg, TARGET_NEG)):
        for r in rng.sample(pool, n):
            rows.append({
                "premise": r["premise"],
                "hypothesis": rng.choice(TEMPLATES),
                "label": r["label"],
                "source": "ragtruth_served_v2",
            })
    rng.shuffle(rows)

    # Assertions (review rule 1: no silent partial success)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    canon = sum(1 for r in rows if r["hypothesis"] == CANONICAL)
    uniq_hyp = len({r["hypothesis"] for r in rows})
    if len(rows) != TARGET_POS + TARGET_NEG or abs(n_pos / len(rows) - 0.35) > 0.01:
        raise SystemExit(f"ABORT: bad rebalance n={len(rows)} pos_rate={n_pos/len(rows):.3f}")
    if uniq_hyp < 8 or canon < 0.15 * len(rows):
        raise SystemExit(f"ABORT: template diversity broken uniq={uniq_hyp} canonical={canon}")

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "file": Path(args.out).name, "rows": len(rows), "pos": n_pos, "neg": len(rows) - n_pos,
        "pos_rate": round(n_pos / len(rows), 4), "unique_hypotheses": uniq_hyp,
        "canonical_share": round(canon / len(rows), 4), "seed": args.seed,
        "templates": len(TEMPLATES), "changes_vs_v1": [
            "9 hypothesis templates (v1: 1 constant)",
            f"{n_pos/len(rows):.0%} positive (v1: 55.4%)",
            f"{len(rows)} rows (v1: 15090)",
        ],
    }
    (REPO_ROOT / "data" / "ragtruth_served_v2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
