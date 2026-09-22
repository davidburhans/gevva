#!/usr/bin/env python3
"""tests/test_haystack_label_invariants.py - Label-semantics invariants for the
synthetic haystack generator.

Regression for the 2026-09-21 audit finding: haystack_embedded rows built from
SNLI-style CONTRADICTION source pairs are unfalsifiable once transplanted into
a haystack (the hypothesis is merely not stated), so the model was scored wrong
for correctly answering NEUTRAL. The generator now intercepts those pairs.

Run: uv run python tests/test_haystack_label_invariants.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import build_haystack_samples  # noqa: E402
from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL  # noqa: E402


def _make_rows(seed: int):
    pairs = (
        [(f"con-premise-{i} says 42 items exist", f"con-hyp-{i}", CONTRADICTION, "en") for i in range(20)]
        + [(f"ent-premise-{i} says 42 items exist", f"ent-hyp-{i}", ENTAILMENT, "en") for i in range(20)]
        + [(f"neu-premise-{i} says 42 items exist", f"neu-hyp-{i}", NEUTRAL, "en") for i in range(20)]
    )
    fillers = [f"filler paragraph {i} discussing entirely unrelated topic number {i}." for i in range(40)]
    return build_haystack_samples(pairs, fillers, n_samples=60, min_fillers=3, max_fillers=6, seed=seed)


def test_embedded_rows_never_carry_contradiction_labels():
    rows = _make_rows(seed=7)
    embedded = [r for r in rows if r["source"] == "haystack_embedded"]
    assert embedded, "expected some embedded rows"
    bad = [r["id"] for r in embedded if r["label"] == CONTRADICTION]
    assert not bad, f"embedded rows with unfalsifiable contradiction labels: {bad}"


def test_contradiction_source_pairs_are_never_embedded_as_needles():
    rows = _make_rows(seed=7)
    leaked = [r["id"] for r in rows if "con-premise-" in r["premise"]]
    assert not leaked, f"contradiction-orig needles still embedded/dropped into docs: {leaked}"


def test_embedded_needles_are_present_and_labels_preserved():
    rows = _make_rows(seed=11)
    embedded = [r for r in rows if r["source"] == "haystack_embedded"]
    for r in embedded:
        # The needle paragraph must actually be in the document (ent/neu prefixes only).
        has_ent = "ent-premise-" in r["premise"]
        has_neu = "neu-premise-" in r["premise"]
        assert has_ent or has_neu, f"{r['id']}: embedded doc lost its needle"
        expected = ENTAILMENT if has_ent else NEUTRAL
        assert r["label"] == expected, f"{r['id']}: label {r['label']} != source-pair label {expected}"


def test_dropped_needles_are_absent_and_neutral():
    rows = _make_rows(seed=11)
    dropped = [r for r in rows if r["source"] == "haystack_drop_neutral"]
    assert dropped, "expected some drop rows"
    for r in dropped:
        assert "-premise-" not in r["premise"], f"{r['id']}: dropped needle still present"
        assert r["label"] == NEUTRAL


def test_corrupted_rows_only_from_entailment_pairs():
    rows = _make_rows(seed=3)
    corrupted = [r for r in rows if r["source"] == "haystack_corrupted_con"]
    assert corrupted, "expected some corrupted rows"
    for r in corrupted:
        assert "ent-premise-" in r["premise"], (
            f"{r['id']}: corruption applied to a non-entailment needle (unverifiable contradiction)"
        )
        assert r["label"] == CONTRADICTION


def test_invariants_hold_across_seeds():
    for seed in (0, 1, 5, 42, 99):
        rows = _make_rows(seed=seed)
        embedded = [r for r in rows if r["source"] == "haystack_embedded"]
        assert all(r["label"] in (ENTAILMENT, NEUTRAL) for r in embedded), f"seed {seed} violated invariants"
        assert all("con-premise-" not in r["premise"] for r in rows), f"seed {seed} leaked con needles"


TESTS = [
    test_embedded_rows_never_carry_contradiction_labels,
    test_contradiction_source_pairs_are_never_embedded_as_needles,
    test_embedded_needles_are_present_and_labels_preserved,
    test_dropped_needles_are_absent_and_neutral,
    test_corrupted_rows_only_from_entailment_pairs,
    test_invariants_hold_across_seeds,
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
