#!/usr/bin/env python3
"""finish_round4_generation.py - Aggregate a partially-run committee from its checkpoint.

2026-09-30 round 4: judges 1+2 (qwen-3.6-27b-q4, gpt-oss-120b) completed all
10,015 verdicts; the deepseek arbiter was stopped after 380/3,578 contested
samples (~24h projected) because 2-1 majorities are review-flagged either way
and never enter training - the arbiter adds zero training rows. This finisher
rebuilds the committee from sdk_synthetic_validation_checkpoint.jsonl, runs the
standard aggregation (absent judges count as skipped, per the early-exit
semantics), and writes the standard train/val/disagreements artifacts with the
same split logic as the generator.

Usage:
    uv run python scripts/finish_round4_generation.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validator_committee import (  # noqa: E402
    JudgeVerdict,
    aggregate_committee_votes,
    write_disagreement_queue,
)

JUDGES = ["qwen-3.6-27b-q4", "gpt-oss-120b", "deepseek-v4-flash-q3"]
ROUND4 = REPO_ROOT / "data" / "round4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-overrides", action="store_true",
                        help="Unanimous overrides become training rows (policy amendment 2026-10-01)")
    args = parser.parse_args()

    samples = [json.loads(line) for line in open(ROUND4 / "sdk_synthetic_raw.jsonl")]
    ckpt = {r["id"]: r["votes"] for r in (json.loads(line) for line in
            open(ROUND4 / "sdk_synthetic_validation_checkpoint.jsonl"))}
    if len(samples) != len(ckpt):
        raise SystemExit(f"ABORT: checkpoint/sample mismatch {len(ckpt)} vs {len(samples)}")

    committee = []
    for s in samples:
        votes = {}
        for judge, v in ckpt[s["id"]].items():
            votes[judge] = JudgeVerdict(label=v["label"], rationale=v.get("rationale", ""),
                                        status=v.get("status", "ok"), latency_ms=0.0)
        committee.append(votes)

    result = aggregate_committee_votes(samples, committee, JUDGES,
                                       include_unanimous_overrides=args.include_overrides)
    stats = Counter(r.get("disagreement_type") or "unanimous_consensus" for r in result.validated)
    if stats.get("unanimous_consensus", 0) < 4000:
        raise SystemExit(f"ABORT: only {stats.get('unanimous_consensus', 0)} clean rows (<4000): {dict(stats)}")

    # Same split policy as the generator: shuffle with run seed, 90/10, floor rules.
    clean = [r for r in result.validated if not r.get("needs_review")]
    prov_ok = sum(1 for r in clean if r.get("committee_rationales"))
    if prov_ok < 0.95 * len(clean):
        raise SystemExit(f"ABORT: provenance missing on {len(clean) - prov_ok}/{len(clean)} clean rows")
    rng = random.Random(args.seed)
    rng.shuffle(clean)
    val_count = max(100, int(len(clean) * 0.1)) if len(clean) >= 1000 else max(1, min(int(len(clean) * 0.1), len(clean) // 2))
    train, val = clean[val_count:], clean[:val_count]

    (ROUND4 / "sdk_synthetic_train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train))
    (ROUND4 / "sdk_synthetic_val.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val))
    queued = write_disagreement_queue(str(ROUND4 / "sdk_synthetic_disagreements.jsonl"), result.review_rows)

    print(json.dumps({
        "raw": len(samples), "clean": len(clean), "train": len(train), "val": len(val),
        "review_queued": queued, "stats": dict(stats), "include_overrides": args.include_overrides,
        "label_source_counts": dict(Counter(r.get("label_source", "generator") for r in clean)),
        "note": "deepseek arbiter stopped at 380/3578 (adds no training rows; "
                "contested rows are review-flagged by policy either way)",
    }, indent=2))


if __name__ == "__main__":
    main()
