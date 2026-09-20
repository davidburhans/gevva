#!/usr/bin/env python3
"""build_arms.py - Builds the A/B shakedown training mixes (pre-registered ablation).

Arm A: clean compiled train set only.
Arm B: clean train set + committee-validated synthetic rows capped at
`--synthetic-frac` (default 0.125) of the combined pool, seeded-sampled when the
cap binds. Manifests record exact provenance for both arms.

Usage:
    uv run python scripts/build_arms.py --data-dir ./data --synthetic-frac 0.125
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline import _pair_key  # noqa: E402

ARM_MANIFEST = "arms_manifest.json"


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arms(data_dir: str, synthetic_frac: float, seed: int) -> Dict:
    """Writes armA/armB train files + manifest; returns the manifest dict."""
    train = _load_jsonl(os.path.join(data_dir, "train.jsonl"))
    sdk_path = os.path.join(data_dir, "staged", "sdk_synthetic_train.jsonl")
    synthetic = _load_jsonl(sdk_path) if os.path.exists(sdk_path) else []

    # Arm B: cap synthetic at synthetic_frac of the combined pool (pre-registered).
    n_synth = min(len(synthetic), int(synthetic_frac / (1 - synthetic_frac) * len(train)))
    rng = random.Random(seed)
    arm_b_synth = rng.sample(synthetic, n_synth) if n_synth < len(synthetic) else list(synthetic)

    # Defensive dedup: filter only the SYNTHETIC candidates against train keys
    # (never the train rows themselves - that would gut arm B to just the synth rows).
    seen_train = {_pair_key(r["premise"], r["hypothesis"]) for r in train}
    kept_synth, seen_synth = [], set()
    for r in arm_b_synth:
        k = _pair_key(r["premise"], r["hypothesis"])
        if k in seen_train or k in seen_synth:
            continue
        seen_synth.add(k)
        kept_synth.append(r)
    arm_a = list(train)
    arm_b = list(train) + kept_synth
    n_synth = len(kept_synth)

    _write_jsonl(os.path.join(data_dir, "armA_train.jsonl"), arm_a)
    _write_jsonl(os.path.join(data_dir, "armB_train.jsonl"), arm_b)

    manifest = {
        "seed": seed,
        "synthetic_frac_cap": synthetic_frac,
        "synthetic_available": len(synthetic),
        "armA_rows": len(arm_a),
        "armB_rows": len(arm_b),
        "armB_synthetic_rows": n_synth,
        "armB_synthetic_frac_actual": round(n_synth / max(1, len(arm_b)), 4),
    }
    with open(os.path.join(data_dir, ARM_MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def export_validated_synthetic(raw_path: str, db_path: str, out_path: str,
                               judge_model: str, min_label: int = 0) -> Dict:
    """Exports judge-validated synthetic rows (status=ok only) to a staged training file.

    Rows whose validation failed (offline/parse_error/missing) are excluded - they are
    review-queue material, not training data. Returns counts for the manifest.

    Example:
        stats = export_validated_synthetic("data/sdk_synthetic_raw.jsonl",
                                           "data/validation_metrics.db",
                                           "data/staged/sdk_synthetic_train.jsonl",
                                           "qwen-3.6-27b-q4")
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    verdicts = conn.execute(
        "SELECT sample_id, verdict_label FROM sample_verdicts"
        " WHERE judge_model = ? AND status = 'ok'", (judge_model,)).fetchall()
    conn.close()
    label_by_id = {r["sample_id"]: int(r["verdict_label"]) for r in verdicts}

    raw = _load_jsonl(raw_path) if os.path.exists(raw_path) else _load_jsonl(
        os.path.join(os.path.dirname(out_path), "sdk_synthetic_raw.jsonl"))
    kept, excluded = [], 0
    for row in raw:
        sid = row["id"]
        if sid not in label_by_id:
            excluded += 1
            continue
        kept.append({**row, "label": label_by_id[sid]})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_jsonl(out_path, kept)
    return {"validated_rows": len(kept), "excluded_rows": excluded, "judge_model": judge_model}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--synthetic-frac", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_arms(args.data_dir, args.synthetic_frac, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
