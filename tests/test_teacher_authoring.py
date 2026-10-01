#!/usr/bin/env python3
"""tests/test_teacher_authoring.py - Teacher-authored generation gates (2026-10-01).

Run:
    uv run python tests/test_teacher_authoring.py

Covers: novel-batch acceptance, dedup against seen keys, the >=80% uniqueness
abort, and the include_unanimous_overrides aggregation amendment.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from teacher_authoring import author_mode_samples  # noqa: E402
from validator_committee import aggregate_committee_votes  # noqa: E402
from test_validator_committee import make_samples  # noqa: E402


class FakeAuthor:
    """Scripted teacher transport: yields `per_call` novel items per call, or repeats."""

    def __init__(self, mode: str, per_call: int, repeat_from: int = 10**9):
        self.calls = 0
        self.mode = mode
        self.per_call = per_call
        self.repeat_from = repeat_from  # start repeating earlier items after N novel ones

    def query_chat(self, system: str, user: str, **_):
        import json
        self.calls += 1
        items = []
        for i in range(self.per_call):
            n = self.calls * 1000 + i
            if n > self.repeat_from:  # degenerate: echo batch 1 forever
                n = i
            items.append({"premise": f"{self.mode} premise {n}",
                          "hypothesis": f"{self.mode} hypothesis {n}",
                          "label": n % 3})
        return json.dumps(items)


def test_authoring_produces_novel_samples() -> None:
    client = FakeAuthor("sdk_tool_routing", per_call=10)
    rows = author_mode_samples(client, "sdk_tool_routing", 25, rng=None)
    keys = {(r["premise"], r["hypothesis"]) for r in rows}
    assert len(rows) == 25 and len(keys) == 25, (len(rows), len(keys))
    assert all(r["source"] == "sdk_tool_routing" and r["label"] in (0, 1, 2) for r in rows)
    print("PASS  test_authoring_produces_novel_samples")


def test_authoring_dedups_against_seen_keys() -> None:
    client = FakeAuthor("sdk_rubric_grading", per_call=10)
    first = author_mode_samples(client, "sdk_rubric_grading", 20)
    seen = {(r["premise"], r["hypothesis"]) for r in first}
    second = author_mode_samples(client, "sdk_rubric_grading", 10, seen_keys=set(
        __import__("hashlib").sha256((p + "\x00" + h).encode()).hexdigest() for p, h in seen))
    both = first + second
    keys = {(r["premise"], r["hypothesis"]) for r in both}
    assert len(keys) == len(both), f"overlap: {len(both) - len(keys)}"
    print("PASS  test_authoring_dedups_against_seen_keys")


def test_authoring_aborts_on_degenerate_diversity() -> None:
    # Teacher echoes batch-1 content forever: uniqueness gate must abort loudly.
    client = FakeAuthor("sdk_cloze_reasoning", per_call=10, repeat_from=10)
    try:
        author_mode_samples(client, "sdk_cloze_reasoning", 50)
        assert False, "degenerate teacher must abort"
    except RuntimeError as e:
        assert "exhausted attempts" in str(e) or "uniqueness gate" in str(e), e
    print("PASS  test_authoring_aborts_on_degenerate_diversity")


def test_include_overrides_amendment() -> None:
    # Two judges unanimously flip the generator label: default reviews, amendment trains.
    samples = make_samples([1, 1])
    committee = [
        {"judge-a": _v(2), "judge-b": _v(2)},
        {"judge-a": _v(1), "judge-b": _v(1)},
    ]
    default = aggregate_committee_votes(samples, committee, ["judge-a", "judge-b"])
    assert all(r.get("needs_review") for r in default.validated if r.get("generator_label") != r.get("label"))
    amended = aggregate_committee_votes(samples, committee, ["judge-a", "judge-b"],
                                       include_unanimous_overrides=True)
    trained = [r for r in amended.validated if r["generator_label"] != r["label"]]
    assert len(trained) == 1 and not trained[0]["needs_review"], trained
    assert trained[0]["label_source"] == "committee_override"
    assert trained[0]["label"] == 2 and trained[0]["generator_label"] == 1
    print("PASS  test_include_overrides_amendment")


class _V:
    def __init__(self, label): self.label = label; self.status = "ok"; self.rationale = "r"; self.ok = True


def _v(label): return _V(label)


if __name__ == "__main__":
    test_authoring_produces_novel_samples()
    test_authoring_dedups_against_seen_keys()
    test_authoring_aborts_on_degenerate_diversity()
    test_include_overrides_amendment()
    print("\n4/4 teacher-authoring tests passed")
