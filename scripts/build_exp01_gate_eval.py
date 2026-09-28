#!/usr/bin/env python3
"""build_exp01_gate_eval.py - Builds the paired NLI gate evaluation set for EXP-01.

Slices (pre-registered in docs/IMPROVEMENT_OPPORTUNITIES.md §13, adapted 2026-09-28):
- floor   : SNLI validation + MNLI validation_matched (from data/test_v2.jsonl)
- medium  : FEVER dev + QNLI validation + SciTail validation (from data/test_v2.jsonl)
- hard    : ANLI R1-R3 test (suite catalog 12) + ContractNLI (suite catalog 11)

Labels use the project convention (0=contradiction, 1=entailment, 2=neutral).
ANLI native label_names are mapped via NATIVE2OURS per AGENTS.md.

Usage:
    uv run python scripts/build_exp01_gate_eval.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = REPO_ROOT / "suite-0.2" / "selected-rows.jsonl.gz"
TEST_V2_PATH = REPO_ROOT / "data" / "test_v2.jsonl"
OUT_PATH = REPO_ROOT / "data" / "exp01_gate_eval.jsonl"

# Suite criteria -> our label ids (0=contradiction, 1=entailment, 2=neutral)
ANLI_LETTER_TO_LABEL = {"A": 1, "B": 2, "C": 0}
CONTRACTNLI_NAME_TO_LABEL = {"Entailment": 1, "Contradiction": 0, "NotMentioned": 2}

FLOOR_SOURCES = {"snli_validation", "mnli_validation_matched"}
MEDIUM_SOURCES = {"fever_dev", "qnli_validation", "scitail_validation"}


def _parse_anli_instructions(text: str) -> Optional[Dict[str, str]]:
    """Split 'Premise: X\\nHypothesis: Y' into parts; None if malformed."""
    prem, hyp = None, None
    for line in text.split("\n"):
        if line.startswith("Premise: ") and prem is None:
            prem = line[len("Premise: "):]
        elif line.startswith("Hypothesis: ") and hyp is None:
            hyp = line[len("Hypothesis: "):]
    if prem is None or hyp is None:
        return None
    return {"premise": prem, "hypothesis": hyp}


def _anli_rows(record: Dict) -> List[Dict]:
    q = record["questions"].get("q1", {})
    parsed = _parse_anli_instructions(q.get("instructions", ""))
    gold_letter = (record.get("gold") or {}).get("q1")
    if not parsed or gold_letter not in ANLI_LETTER_TO_LABEL:
        return []
    return [{
        "premise": parsed["premise"],
        "hypothesis": parsed["hypothesis"],
        "label": ANLI_LETTER_TO_LABEL[gold_letter],
        "source": f"anli_suite_{record.get('split', 'test')}",
        "gate_slice": "hard",
    }]


def _contractnli_rows(record: Dict) -> List[Dict]:
    premise = record.get("state")
    if not isinstance(premise, str) or len(premise) < 200:
        return []
    rows: List[Dict] = []
    for qid, expected in (record.get("expected") or {}).items():
        q = record["questions"].get(qid, {})
        text = q.get("instructions", "")
        marker = "hypothesis:"
        if marker not in text.lower():
            continue
        hypothesis = text[text.lower().index(marker) + len(marker):].strip()
        if expected not in CONTRACTNLI_NAME_TO_LABEL:
            continue
        rows.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": CONTRACTNLI_NAME_TO_LABEL[expected],
            "source": "contractnli_suite",
            "gate_slice": "hard",
        })
    return rows


def _suite_rows() -> List[Dict]:
    rows: List[Dict] = []
    with gzip.open(SUITE_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cid = record.get("_evaluation", {}).get("catalog_id")
            if cid == 12:
                rows.extend(_anli_rows(record))
            elif cid == 11:
                rows.extend(_contractnli_rows(record))
    return rows


def _test_v2_rows() -> List[Dict]:
    rows: List[Dict] = []
    with open(TEST_V2_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            source = str(record.get("source", "?"))
            if source in FLOOR_SOURCES:
                gate_slice = "floor"
            elif source in MEDIUM_SOURCES:
                gate_slice = "medium"
            else:
                continue
            rows.append({
                "premise": record["premise"],
                "hypothesis": record["hypothesis"],
                "label": int(record["label"]),
                "source": source,
                "gate_slice": gate_slice,
            })
    return rows


def main() -> None:
    rows = _test_v2_rows() + _suite_rows()
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["gate_slice"], {})
        counts[row["gate_slice"]][row["source"]] = counts[row["gate_slice"]].get(row["source"], 0) + 1
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    for gate_slice, sources in sorted(counts.items()):
        total = sum(sources.values())
        print(f"  {gate_slice}: {total} items across {len(sources)} sources")
        for source, count in sorted(sources.items(), key=lambda kv: -kv[1]):
            print(f"    {source}: {count}")


if __name__ == "__main__":
    main()
