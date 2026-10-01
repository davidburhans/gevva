#!/usr/bin/env python3
"""audit_semantic_overlap.py - Token-shingle overlap between training mixture and DI suite.

Byte-hash audits cannot catch semantic near-duplicates when framings differ
(review C3 class; discovered again with the SDK val file 2026-09-30). This
script builds an inverted index over 5-word shingles of every suite state,
then scores each training row's premise for best shingle-Jaccard against any
suite state. High overlap (>=0.8) means the DI suite item was likely seen in
training - flag it per source.

CPU-only; safe to run alongside GPU inference.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKEN = re.compile(r"[a-z0-9]+")

def shingles(text: str, k: int = 5) -> set:
    toks = TOKEN.findall(text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}

def main() -> None:
    # 1. Index suite states by shingle
    index: dict[str, set[int]] = defaultdict(set)
    suite_texts: list[str] = []
    for shard in ("suite-0.2/selected-rows.jsonl.gz", "suite-0.2/added-rows.jsonl.gz"):
        with gzip.open(REPO / shard, "rt") as f:
            for line in f:
                r = json.loads(line)
                st = r.get("state", "")
                text = st if isinstance(st, str) else json.dumps(st, sort_keys=True)
                sid = len(suite_texts)
                suite_texts.append(text)
                for s in shingles(text):
                    index[s].add(sid)
    if len(suite_texts) < 10000:
        raise SystemExit(f"ABORT: suite index suspiciously small ({len(suite_texts)})")

    # 2. Score training rows (SDK + typed + ragtruth slices: the decision-task rows)
    TARGETS = ("sdk_", "typed_decisions", "ragtruth", "prometheus", "helpsteer")
    flagged = Counter()
    checked = Counter()
    top_hits: list[tuple[float, str, int]] = []
    for line in open(REPO / "data" / "train_cal2_e2b.jsonl"):
        r = json.loads(line)
        src = r.get("source", "")
        if not src.startswith(TARGETS):
            continue
        checked[src] += 1
        sh = shingles(r["premise"])
        cand: Counter[int] = Counter()
        for s in sh:
            for sid in index.get(s, ()):
                cand[sid] += 1
        if not cand:
            continue
        sid, inter = cand.most_common(1)[0]
        union = len(sh) + len(shingles(suite_texts[sid])) - inter
        j = inter / max(1, union)
        top_hits.append((j, src, sid))
        if j >= 0.8:
            flagged[src] += 1

    top_hits.sort(reverse=True)
    report = {
        "suite_states": len(suite_texts),
        "checked_rows": dict(checked),
        "flagged_ge_0.8": dict(flagged),
        "top_10_overlaps": [
            {"jaccard": round(j, 3), "source": s, "suite_state_head": suite_texts[i][:120]}
            for j, s, i in top_hits[:10]
        ],
    }
    out = REPO / "results" / "semantic_overlap_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    total_flag = sum(flagged.values())
    if total_flag > 50:
        print(f"\nWARNING: {total_flag} training rows are near-duplicates of suite states", file=sys.stderr)

if __name__ == "__main__":
    main()
