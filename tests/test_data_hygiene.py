#!/usr/bin/env python3
"""tests/test_data_hygiene.py - Offline tests for train/benchmark contamination guards.

Run (single command, no GPU, no network):
    uv run python tests/test_data_hygiene.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline import _dedupe_split, _is_haystack_holdout, _pair_key  # noqa: E402


def _row(p: str, h: str) -> dict:
    return {"premise": p, "hypothesis": h, "label": 1}


def test_pair_key_normalizes_case_and_whitespace():
    assert _pair_key("The cat  sits.", "It naps.") == _pair_key("the cat sits.\n", "IT  naps.")
    assert _pair_key("a", "b") != _pair_key("a", "c")
    assert _pair_key("a", "b") != _pair_key("b", "a"), "pair order matters"


def test_holdout_is_deterministic_and_approximately_10pct():
    flags = [_is_haystack_holdout(f"premise {i}", f"hypothesis {i}") for i in range(5000)]
    assert flags == [_is_haystack_holdout(f"premise {i}", f"hypothesis {i}") for i in range(5000)]
    rate = sum(flags) / len(flags)
    assert 0.08 <= rate <= 0.12, f"holdout rate {rate:.3f} drifted from 10%"


def test_dedupe_split_drops_train_dupes_and_val_overlap():
    train = [_row("p1", "h1"), _row("p2", "h2"), _row("p1", "h1"), _row("P1,  h1!".replace(",", ""),"h1")]
    val = [_row("p3", "h3"), _row("p2", "h2")]
    clean_train, clean_val, dropped_train, dropped_val = _dedupe_split(train, val)
    assert len(clean_train) == 3 and dropped_train == 1, "intra-train duplicate must drop"
    assert len(clean_val) == 1 and dropped_val == 1, "val row duplicating train must drop"
    assert clean_val[0]["premise"] == "p3"


def test_eval_seeded_slice_is_deterministic_and_shuffled():
    import eval_openjev_benchmarks as ev
    from datasets import Dataset
    ds = Dataset.from_dict({"x": list(range(1000))})
    s1 = ev._seeded_slice(ds, 10)
    s2 = ev._seeded_slice(ds, 10)
    assert len(s1) == 10
    assert list(s1["x"]) == list(s2["x"]), "same seed must give identical slices"
    assert list(s1["x"]) != list(range(10)), "slice must be shuffled, not first-N"


def test_forbidden_keys_exclude_trained_pairs():
    import eval_openjev_benchmarks as ev
    with tempfile.TemporaryDirectory() as tmp:
        train = [_row("seen in train", "pair one"), _row("another", "pair two")]
        val = [_row("seen in val", "pair three")]
        (Path(tmp) / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
        (Path(tmp) / "val.jsonl").write_text(
            "\n".join(json.dumps(r) for r in val) + "\n", encoding="utf-8")
        ev._FORBIDDEN_PAIR_KEYS = None  # reset module cache between data_dirs
        forbidden = ev.load_forbidden_nli_keys(tmp)
        assert ev._pair_key("seen in train", "pair one") in forbidden
        assert ev._pair_key("SEEN IN TRAIN", "pair one") in forbidden, "normalization must match"
        assert ev._pair_key("unseen", "pair") not in forbidden
    ev._FORBIDDEN_PAIR_KEYS = None


def test_build_arms_prevents_test_leakage():
    from scripts.build_arms import build_arms
    with tempfile.TemporaryDirectory() as tmp:
        train = [_row(f"P_train_{i}", f"H_train_{i}") for i in range(10)]
        test = [_row("P_test", "H_test")]
        val = [_row("P_val", "H_val")]
        staged_synth = [
            _row("P_synth", "H_synth"),
            _row("P_test", "H_test"),  # LEAK ATTEMPT: must be excluded
            _row("P_val", "H_val"),    # VAL OVERLAP: must be excluded
        ]
        os.makedirs(Path(tmp) / "staged", exist_ok=True)
        (Path(tmp) / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
        (Path(tmp) / "test.jsonl").write_text(json.dumps(test[0]) + "\n", encoding="utf-8")
        (Path(tmp) / "val.jsonl").write_text(json.dumps(val[0]) + "\n", encoding="utf-8")
        (Path(tmp) / "staged" / "sdk_synthetic_train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in staged_synth) + "\n", encoding="utf-8"
        )

        build_arms(data_dir=tmp, synthetic_frac=0.5, seed=42)
        arm_b = [json.loads(line) for line in (Path(tmp) / "armB_train.jsonl").read_text(encoding="utf-8").splitlines() if line]
        keys = {_pair_key(r["premise"], r["hypothesis"]) for r in arm_b}
        assert _pair_key("P_test", "H_test") not in keys, "test set item leaked into armB!"
        assert _pair_key("P_val", "H_val") not in keys, "val set item leaked into armB!"
        assert _pair_key("P_synth", "H_synth") in keys


TESTS = [test_pair_key_normalizes_case_and_whitespace,
         test_holdout_is_deterministic_and_approximately_10pct,
         test_dedupe_split_drops_train_dupes_and_val_overlap,
         test_eval_seeded_slice_is_deterministic_and_shuffled,
         test_forbidden_keys_exclude_trained_pairs,
         test_build_arms_prevents_test_leakage]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            import traceback
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
