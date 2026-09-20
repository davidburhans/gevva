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
    test_path = os.path.join(data_dir, "test.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    test_rows = _load_jsonl(test_path) if os.path.exists(test_path) else []
    val_rows = _load_jsonl(val_path) if os.path.exists(val_path) else []

    sdk_path = os.path.join(data_dir, "staged", "sdk_synthetic_train.jsonl")
    synthetic = _load_jsonl(sdk_path) if os.path.exists(sdk_path) else []

    # Arm B: cap synthetic at synthetic_frac of the combined pool (pre-registered).
    n_synth = min(len(synthetic), int(synthetic_frac / (1 - synthetic_frac) * len(train)))
    rng = random.Random(seed)
    arm_b_synth = rng.sample(synthetic, n_synth) if n_synth < len(synthetic) else list(synthetic)

    # Defensive dedup & test leakage guard (ModernCE/OpenJEV hygiene audit):
    # filter only the SYNTHETIC candidates against seen_train | seen_test | seen_val
    # (never the train rows themselves - that would gut arm B to just the synth rows).
    seen_train = {_pair_key(r["premise"], r["hypothesis"]) for r in train}
    seen_test = {_pair_key(r["premise"], r["hypothesis"]) for r in test_rows}
    seen_val = {_pair_key(r["premise"], r["hypothesis"]) for r in val_rows}
    forbidden_keys = seen_train | seen_test | seen_val

    kept_synth, seen_synth = [], set()
    for r in arm_b_synth:
        k = _pair_key(r["premise"], r["hypothesis"])
        if k in forbidden_keys or k in seen_synth:
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


def export_validated_synthetic(raw_path: str, db_path: str, out_train_path: str,
                               out_val_path: str, run_id: str, judge_model: str,
                               val_frac: float = 0.1, seed: int = 42) -> Dict:
    """Exports judge-validated synthetic rows (status=ok only, run-scoped) with a
    seeded validation holdback.

    WHY the holdback (audit CRITICAL): training arms must never see rows that a later
    compile will place in sdk_synthetic_val (selection/test pools) - otherwise the
    forward dataset is contaminated by the ablation. Train rows -> out_train_path;
    holdback rows -> out_val_path (compiler val pool only).

    Example:
        stats = export_validated_synthetic("data/sdk_synthetic_raw.jsonl",
                                           "data/validation_metrics.db",
                                           "data/staged/sdk_synthetic_train.jsonl",
                                           "data/staged/sdk_synthetic_val.jsonl",
                                           run_id="run_20260919_223701", judge_model="qwen-3.6-27b-q4")
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    verdicts = conn.execute(
        "SELECT sample_id, verdict_label FROM sample_verdicts"
        " WHERE judge_model = ? AND status = 'ok' AND run_id = ?", (judge_model, run_id)).fetchall()
    conn.close()
    label_by_id = {r["sample_id"]: int(r["verdict_label"]) for r in verdicts}

    raw = _load_jsonl(raw_path) if os.path.exists(raw_path) else _load_jsonl(
        os.path.join(os.path.dirname(out_train_path), "sdk_synthetic_raw.jsonl"))
    validated, excluded = [], 0
    for row in raw:
        sid = row["id"]
        if sid not in label_by_id:
            excluded += 1
            continue
        validated.append({**row, "label": label_by_id[sid]})

    rng = random.Random(seed)
    rng.shuffle(validated)
    n_val = int(len(validated) * val_frac)
    val_rows, train_rows = validated[:n_val], validated[n_val:]

    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    _write_jsonl(out_train_path, train_rows)
    _write_jsonl(out_val_path, val_rows)
    return {"validated_rows": len(validated), "excluded_rows": excluded,
            "train_rows": len(train_rows), "val_holdback_rows": len(val_rows),
            "judge_model": judge_model, "run_id": run_id, "single_judge": True}


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
