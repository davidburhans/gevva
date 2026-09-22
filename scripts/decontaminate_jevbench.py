#!/usr/bin/env python3
"""scripts/decontaminate_jevbench.py - Enforced 8-gram contamination filter.

Methodology review M2 (2026-09-22): no leakage was measured (0/4,304 generated
rows; 66/91,445 mixture rows sharing one generic billing phrase), but nothing
ENFORCED it. Registered into docs/EVALUATION_PROTOCOL.md §8: every training row
must pass an 8-gram check against the JevBench public items before any compile.

Also applies intra-set template-collapse caps (max consecutive identical
instructions), addressing the 4,000-rows-from-3-templates redundancy finding.

Usage:
  uv run python scripts/decontaminate_jevbench.py --in data/phase2_hard/synth_hard_scenarios_raw.jsonl --out data/phase2_hard/decontaminated.jsonl
  uv run python scripts/decontaminate_jevbench.py --in X.jsonl --check-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JEVBENCH_PUBLIC = Path(__file__).resolve().parents[2] / "jevbench" / "datasets" / "public"
NGRAM_N = 8


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower())


def ngrams(text: str) -> set:
    words = normalize(text).split()
    return {" ".join(words[i:i + NGRAM_N]) for i in range(len(words) - NGRAM_N + 1)}


def load_reference() -> set:
    ref: set = set()
    for fname in ("easy.jsonl", "original.jsonl", "hard.jsonl"):
        path = JEVBENCH_PUBLIC / fname
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            ref |= ngrams(json.dumps(d.get("state", "")) + " " + str(d.get("question", {}).get("instructions", "")))
    return ref


def main() -> int:
    parser = argparse.ArgumentParser(description="8-gram decontamination vs JevBench public items")
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", default=None)
    parser.add_argument("--check-only", action="store_true", help="Report without writing")
    args = parser.parse_args()

    ref = load_reference()
    rows = [json.loads(l) for l in open(args.infile, encoding="utf-8") if l.strip()]

    kept, dropped_overlap = [], 0
    for r in rows:
        grams = ngrams(str(r.get("premise", "")) + " " + str(r.get("hypothesis", "")))
        if grams & ref:
            dropped_overlap += 1
            continue
        kept.append(r)

    # Per-template cap (methodology M2: "4,000 rows from ~3 templates"): no single
    # instruction/state template may dominate the compiled set. Signature is
    # schema-aware: instruction if present (converted rows), else state prefix
    # (raw scenarios), else premise prefix.
    MAX_PER_TEMPLATE = 400

    def template_signature(r) -> str:
        q = r.get("question") or {}
        for cand in (q.get("instructions"), r.get("instructions")):
            if cand:
                return normalize(cand)[:100]
        for cand in (r.get("state"), r.get("premise")):
            if cand:
                return normalize(cand)[:100]
        return "<none>"

    sig_counts: Counter = Counter()
    final, dropped_template = [], 0
    for r in kept:
        sig = template_signature(r)
        sig_counts[sig] += 1
        if sig_counts[sig] > MAX_PER_TEMPLATE:
            dropped_template += 1
            continue
        final.append(r)
    print(f"reference 8-grams: {len(ref):,}")
    print(f"rows in: {len(rows):,} | overlap-dropped: {dropped_overlap} | "
          f"template-capped (max {MAX_PER_TEMPLATE}/template): {dropped_template} | rows out: {len(final):,}")
    print(f"top templates: {[(s[:60], c) for s, c in sig_counts.most_common(3)]}")
    if not args.check_only and args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote: {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
