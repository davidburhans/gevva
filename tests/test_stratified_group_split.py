#!/usr/bin/env python3
"""tests/test_stratified_group_split.py - Group-aware split label-stratification tests.

Regression for the P1 eval incident (2026-09-21): the group-aware split shuffled
groups without label stratification, so every neutral-bearing group landed in
train and the validation slice received 13,022 rows with ZERO neutral support -
making the P1 checkpoint's 3-class ECE/Brier and per-neutral metrics meaningless.

Run: uv run python tests/test_stratified_group_split.py
"""

import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune import stratified_split  # noqa: E402


def _grouped_rows(n_binary_groups: int, n_neutral_groups: int, rows_per_group: int = 4):
    rows = []
    for g in range(n_binary_groups):
        # Choice-style mixed group: gold ent + rejected cons (like typed_decisions).
        for j in range(rows_per_group):
            rows.append({"group_id": f"g{g}", "label": 1 if j == 0 else 0,
                         "premise": f"p{g}", "hypothesis": f"h{g}-{j}"})
    for g in range(n_neutral_groups):
        for j in range(rows_per_group):
            rows.append({"group_id": f"neu{g}", "label": 2,
                         "premise": f"np{g}", "hypothesis": f"nh{g}-{j}"})
    return rows


def test_minority_class_always_reaches_val():
    """The P1 regression: neutral-only groups must contribute to validation."""
    rows = _grouped_rows(n_binary_groups=60, n_neutral_groups=2)
    _, val = stratified_split(rows, val_ratio=0.15, seed=42)
    val_labels = Counter(r["label"] for r in val)
    assert val_labels.get(2, 0) > 0, f"neutral starved from val: {dict(val_labels)}"


def test_groups_never_span_splits():
    rows = _grouped_rows(n_binary_groups=40, n_neutral_groups=3)
    train, val = stratified_split(rows, val_ratio=0.2, seed=7)
    train_gids = {r["group_id"] for r in train}
    val_gids = {r["group_id"] for r in val}
    assert not (train_gids & val_gids), "group leakage across splits"


def test_val_ratio_approximately_per_label():
    rows = _grouped_rows(n_binary_groups=100, n_neutral_groups=20)
    train, val = stratified_split(rows, val_ratio=0.15, seed=0)
    for lab, _name in ((0, "contradiction"), (1, "entailment"), (2, "neutral")):
        total = sum(1 for r in rows if r["label"] == lab)
        got = sum(1 for r in val if r["label"] == lab)
        # Groups are the atomic unit; allow granularity slack of one group (4 rows).
        assert 0 < got / total < 0.4, f"label {lab}: val share {got}/{total} far from ratio"


def test_split_is_deterministic_per_seed():
    rows = _grouped_rows(n_binary_groups=30, n_neutral_groups=4)
    a = stratified_split(rows, val_ratio=0.15, seed=11)
    b = stratified_split(rows, val_ratio=0.15, seed=11)
    assert [r["hypothesis"] for r in a[0]] == [r["hypothesis"] for r in b[0]]
    assert [r["hypothesis"] for r in a[1]] == [r["hypothesis"] for r in b[1]]


def test_ungrouped_path_still_stratifies():
    rows = [{"premise": f"p{i}", "hypothesis": f"h{i}", "label": i % 3} for i in range(300)]
    train, val = stratified_split(rows, val_ratio=0.15, seed=42)
    val_labels = Counter(r["label"] for r in val)
    assert all(val_labels.get(lab, 0) > 0 for lab in (0, 1, 2)), f"class missing from val: {dict(val_labels)}"


def test_minus1_sentinel_rows_are_singletons_not_one_mega_group():
    """Regression (review F05, 2026-09-22): ingestion stores anchor NLI rows with
    group_id="-1"; the grouped-split path treated the truthy "-1" as ONE group key,
    splitting all anchors all-or-nothing - recreating the zero-neutral-validation
    bug. "-1" must be a singleton exactly like a missing group_id."""
    rows = _grouped_rows(n_binary_groups=20, n_neutral_groups=3)
    # 60 anchor rows with the -1 sentinel, labels balanced across all 3 classes
    for i in range(60):
        rows.append({"group_id": "-1", "label": i % 3,
                     "premise": f"anchor-p{i}", "hypothesis": f"anchor-h{i}"})
    train, val = stratified_split(rows, val_ratio=0.15, seed=42)
    val_labels = Counter(r["label"] for r in val)
    assert all(val_labels.get(lab, 0) > 0 for lab in (0, 1, 2)), \
        f"anchor -1 sentinel broke class coverage in val: {dict(val_labels)}"
    assert any(r.get("group_id") == "-1" for r in train), "anchors must be able to reach train"


TESTS = [
    test_minority_class_always_reaches_val,
    test_groups_never_span_splits,
    test_val_ratio_approximately_per_label,
    test_split_is_deterministic_per_seed,
    test_ungrouped_path_still_stratifies,
    test_minus1_sentinel_rows_are_singletons_not_one_mega_group,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
