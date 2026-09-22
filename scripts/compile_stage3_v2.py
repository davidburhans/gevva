#!/usr/bin/env python3
"""scripts/compile_stage3_v2.py - Production Stage 3 v2 mixture compiler.

Differences from the v1 chain compile (2026-09-21 evening audit):
1. Filters the label-semantics defect out of the stage-2 replay slice:
   `haystack_embedded` rows with label=CONTRADICTION are unfalsifiable from
   the document (SNLI cross-caption convention does not survive haystack
   transplantation) and scored a strict reader WRONG (slice ceiling 57.9%).
2. Writes a full provenance manifest (audit A9 standard): input SHAs, filter
   counts, label distributions, compiler args.

The multi-resolution haystack files (4K-128K, sources haystack_{4k..128k}_*)
are reused verbatim - they are clean by construction (100% val across bands).

Usage:
  uv run python scripts/compile_stage3_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ID2LABEL  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()[:16]


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def drop_unfalsifiable_embedded(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Removes haystack_embedded CONTRADICTION rows (2026-09-21 label-semantics audit)."""
    kept = [r for r in rows if not (r.get("source") == "haystack_embedded" and r.get("label") == CONTRADICTION)]
    return kept, len(rows) - len(kept)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Stage 3 v2 training mixture (label-defect filtered)")
    parser.add_argument("--out-dir", default="data/stage3_v2")
    parser.add_argument("--haystack-train", default="data/stage3/stage3_haystack_train.jsonl")
    parser.add_argument("--haystack-val", default="data/stage3/stage3_haystack_val.jsonl")
    parser.add_argument("--replay-train", default="data/stage2_train.jsonl")
    parser.add_argument("--replay-val", default="data/stage2_val.jsonl")
    parser.add_argument("--replay-train-limit", type=int, default=10_000)
    parser.add_argument("--replay-val-limit", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    haystack_train = load_jsonl(REPO_ROOT / args.haystack_train)
    haystack_val = load_jsonl(REPO_ROOT / args.haystack_val)
    replay_train = load_jsonl(REPO_ROOT / args.replay_train, limit=args.replay_train_limit)
    replay_val = load_jsonl(REPO_ROOT / args.replay_val, limit=args.replay_val_limit)

    replay_train, dropped_train = drop_unfalsifiable_embedded(replay_train)
    replay_val, dropped_val = drop_unfalsifiable_embedded(replay_val)

    rng = random.Random(args.seed)
    train_rows = haystack_train + replay_train
    rng.shuffle(train_rows)
    val_rows = haystack_val + replay_val  # keep val in stable order for cross-run comparability

    train_path = out_dir / "stage3_train.jsonl"
    val_path = out_dir / "stage3_val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "compiler": "scripts/compile_stage3_v2.py",
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "inputs": {
            "haystack_train": {"path": args.haystack_train, "sha256_16": sha256_file(REPO_ROOT / args.haystack_train), "rows": len(haystack_train)},
            "haystack_val": {"path": args.haystack_val, "sha256_16": sha256_file(REPO_ROOT / args.haystack_val), "rows": len(haystack_val)},
            "replay_train": {"path": args.replay_train, "sha256_16": sha256_file(REPO_ROOT / args.replay_train), "rows_fetched": args.replay_train_limit},
            "replay_val": {"path": args.replay_val, "sha256_16": sha256_file(REPO_ROOT / args.replay_val), "rows_fetched": args.replay_val_limit},
        },
        "filter_rule": "drop source==haystack_embedded AND label==CONTRADICTION (unfalsifiable after haystack transplantation; 2026-09-21 audit)",
        "dropped_unfalsifiable": {"train": dropped_train, "val": dropped_val},
        "outputs": {
            "train": {"path": str(train_path), "rows": len(train_rows), "labels": dict(Counter(r["label"] for r in train_rows))},
            "val": {"path": str(val_path), "rows": len(val_rows), "labels": dict(Counter(r["label"] for r in val_rows))},
        },
        "label_names": {str(k): v for k, v in ID2LABEL.items()},
        "recipe_note": "launch with train_cross_encoder.py --warm-start ckpt/gemma-4-e2b-nli-stage2/best --resume-auto --checkpoint-interval 100",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    # Versioned provenance copy: data/ is gitignored, results/ is the artifact root.
    results_copy = REPO_ROOT / "results" / "stage3_v2_manifest.json"
    results_copy.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Stage 3 v2 compiled: {len(train_rows)} train / {len(val_rows)} val rows "
          f"(dropped {dropped_train} train + {dropped_val} val unfalsifiable embedded-contradiction rows)")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
