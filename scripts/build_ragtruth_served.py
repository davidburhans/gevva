#!/usr/bin/env python3
"""build_ragtruth_served.py - RAGTruth training slice in the EXACT served framing.

Train-serving parity (AGENTS.md principle 4): at Decision Index serving time,
RAGTruth (cat 59) reaches the engine as:
  premise   = _format_state({"prompt": ..., "response": ...})   -> "Prompt: ...\\n\\nResponse: ..."
  hypothesis = "The response contains content that is not supported by the context in the prompt."
  gold       = True (hallucinated) / False (clean)

Training on any other framing (e.g. premise=prompt, hypothesis=response) moves
probability scales without teaching the served decision. Labels map to our
convention: hallucinated -> 1 (entailment of the noul hypothesis), clean -> 0
(contradiction). Byte-identical formatting is guaranteed by importing the
engine's own _format_state.

Usage:
    uv run python scripts/build_ragtruth_served.py \
        [--raw data/ragtruth_train.jsonl --suite suite-0.2 --out data/ragtruth_served_train.jsonl]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from research.adapters.gevva_decision_index_engine import _format_state  # noqa: E402

RAGTRUTH_INSTRUCTIONS = (
    "The response contains content that is not supported by the context in the prompt."
)


def load_served_items(suite_dir: Path) -> list:
    """Rebuild the served (premise, hypothesis, gold) triples from the suite itself."""
    items = []
    for shard in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        path = suite_dir / shard
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("_evaluation", {}).get("catalog_id") != 59:
                    continue
                state = r.get("state", {})
                exp = (r.get("expected") or {}).get("q")
                if not isinstance(state, dict) or not isinstance(exp, bool):
                    continue
                items.append({
                    "premise": _format_state(state),
                    "hypothesis": RAGTRUTH_INSTRUCTIONS,
                    "hallucinated": exp,
                })
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(REPO_ROOT / "suite-0.2"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "ragtruth_served_train.jsonl"))
    args = parser.parse_args()

    items = load_served_items(Path(args.suite))
    # NOTE: the suite shard is the EVALUATION set - this script is for parity
    # verification only. Training rows come from the raw TRAIN split below.
    print(f"served-form verification items from suite: {len(items)}")

    # Training rows: raw train split (label 0=hallucinated, 1=supported) re-framed.
    raw_path = REPO_ROOT / "data" / "ragtruth_train.jsonl"
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"Premise: (.*?)Hypothesis: (.*)", r.get("text", ""), re.S)
            if not m:
                continue
            prompt, response = m.group(1).strip(), m.group(2).strip()
            hallucinated = int(r["label"]) == 0
            rows.append({
                "premise": _format_state({"prompt": prompt, "response": response}),
                "hypothesis": RAGTRUTH_INSTRUCTIONS,
                "label": 1 if hallucinated else 0,  # noul hypothesis entailed vs contradicted
                "source": "ragtruth_served_train",
            })

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_hall = sum(1 for r in rows if r["label"] == 1)
    print(f"wrote {len(rows)} served-framing rows -> {out} ({n_hall} hallucinated = {100*n_hall/max(1,len(rows)):.1f}%)")
    if rows:
        print("sample premise:", rows[0]["premise"][:120].replace("\n", " / "))


if __name__ == "__main__":
    main()
