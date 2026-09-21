#!/usr/bin/env python3
"""scripts/export_final_consensus.py - Exports Multi-Judge Consensus Dataset.

Aggregates verdicts from Judge 1 (Qwen 3.6 27B) and Judge 2 (Qwen 3.8 125B) from
`data/validation_metrics.db`.
- Enforces strict 2-judge unanimous consensus for inclusion into training splits.
- Disagreements (e.g. 1-1 ties, QA-NLI syllogistic splits) are quarantined into
  `data/staged/sdk_synthetic_disagreements.jsonl`.
- Unanimous consensus samples are stratified into `sdk_synthetic_train.jsonl` (90%)
  and `sdk_synthetic_val.jsonl` (10%).

Usage:
  uv run python scripts/export_final_consensus.py --db data/validation_metrics.db --out-dir data/staged
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validation_metrics_db import ValidationMetricsDB
from validator_committee import (
    JudgeVerdict,
    aggregate_committee_votes,
    write_disagreement_queue,
)


def export_consensus(
    raw_path: str = "data/sdk_synthetic_raw.jsonl",
    db_path: str = "data/validation_metrics.db",
    out_dir: str = "data/staged",
    judges: list = None,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    if judges is None:
        judges = ["qwen-3.6-27b-q4", "qwen-3.8-125b-q3"]

    print("=" * 65)
    print("Exporting Consensus Synthetic Split (Strict Multi-Judge)")
    print(f"Judges: {', '.join(judges)}")
    print(f"Database: {db_path}")
    print("=" * 65)

    os.makedirs(out_dir, exist_ok=True)
    db = ValidationMetricsDB(db_path)

    # 1. Load raw samples
    samples = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples):,} raw candidate samples from {raw_path}.")

    sample_ids = [s["id"] for s in samples]

    # 2. Fetch persisted verdicts per judge
    committee = [{} for _ in samples]
    id_to_idx = {s["id"]: i for i, s in enumerate(samples)}

    for judge in judges:
        cur = db.conn.cursor()
        cur.execute(
            """
            SELECT sample_id, verdict_label, status, rationale, latency_ms
            FROM sample_verdicts
            WHERE judge_model = ? AND status = 'ok'
            """,
            (judge,),
        )
        rows = cur.fetchall()
        print(f"  [{judge}]: loaded {len(rows):,} ok verdicts from database.")
        for sid, lbl, status, rat, lat in rows:
            if sid in id_to_idx:
                idx = id_to_idx[sid]
                committee[idx][judge] = JudgeVerdict(
                    label=lbl,
                    status=status,
                    rationale=rat or "",
                    latency_ms=lat or 0.0,
                )

    # 3. Aggregate votes
    result = aggregate_committee_votes(samples, committee, judges)

    # 4. Strict filter: only keep samples where needs_review is False
    clean_samples = [s for s in result.validated if not s.get("needs_review", False)]
    flagged_samples = [s for s in result.validated if s.get("needs_review", False)]

    print(f"\nAggregation Results:")
    print(f"  Unanimous Consensus (Clean): {len(clean_samples):,} ({len(clean_samples)/len(samples)*100:.2f}%)")
    print(f"  Disagreements (Quarantined): {len(flagged_samples):,} ({len(flagged_samples)/len(samples)*100:.2f}%)")
    print(f"  Breakdown by outcome: {result.stats}")

    # 5. Write disagreement queue
    disagreements_path = os.path.join(out_dir, "sdk_synthetic_disagreements.jsonl")
    queued_count = write_disagreement_queue(disagreements_path, result.review_rows)
    print(f"  Wrote {queued_count:,} reviews to {disagreements_path}")

    # 6. Stratified train/val split of clean samples
    rng = random.Random(seed)
    rng.shuffle(clean_samples)

    n_val = max(1, int(len(clean_samples) * val_frac))
    val_samples = clean_samples[:n_val]
    train_samples = clean_samples[n_val:]

    train_path = os.path.join(out_dir, "sdk_synthetic_train.jsonl")
    val_path = os.path.join(out_dir, "sdk_synthetic_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nExport Completed Successfully:")
    print(f"  Clean Train Split: {train_path} ({len(train_samples):,} rows)")
    print(f"  Clean Val Split:   {val_path} ({len(val_samples):,} rows)")
    print(f"  Disagreements:     {disagreements_path} ({queued_count:,} rows)")

    train_dist = Counter(r["label"] for r in train_samples)
    train_srcs = Counter(r["source"] for r in train_samples)
    print(f"  Train Label Distribution: {dict(train_dist)}")
    print(f"  Train Sources: {dict(train_srcs)}")

    db.close()
    return {
        "total_samples": len(samples),
        "clean_samples": len(clean_samples),
        "disagreements": len(flagged_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Export Multi-Judge Consensus Synthetic Dataset.")
    parser.add_argument("--raw-path", default="data/sdk_synthetic_raw.jsonl")
    parser.add_argument("--db", default="data/validation_metrics.db")
    parser.add_argument("--out-dir", default="data/staged")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_consensus(
        raw_path=args.raw_path,
        db_path=args.db,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
