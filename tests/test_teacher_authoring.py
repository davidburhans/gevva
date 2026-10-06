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


def test_robust_parser() -> None:
    from teacher_authoring import robust_parse_authored_samples

    # 1. Normal JSON object
    r1 = robust_parse_authored_samples('{"samples": [{"premise": "p1", "hypothesis": "h1", "label": 1}]}')
    assert len(r1) == 1 and r1[0]["premise"] == "p1"

    # 2. Single-quoted JSON object (the GBNF-unconstrained bug)
    r2 = robust_parse_authored_samples("{'samples': [{'premise': 'p2', 'hypothesis': 'h2', 'label': 0}]}")
    assert len(r2) == 1 and r2[0]["premise"] == "p2" and r2[0]["label"] == 0

    # 3. Trailing commas and markdown code blocks
    r3 = robust_parse_authored_samples("```json\n{'samples': [{'premise': 'p3', 'hypothesis': 'h3', 'label': 2},]}\n```")
    assert len(r3) == 1 and r3[0]["premise"] == "p3"

    # 4. Direct list format
    r4 = robust_parse_authored_samples('[{"premise": "p4", "hypothesis": "h4", "label": 1}]')
    assert len(r4) == 1 and r4[0]["premise"] == "p4"

    # 5. Conversational intro with regex recovery
    r5 = robust_parse_authored_samples("Sure, here are your requested samples:\n{'samples': [{'premise': 'p5', 'hypothesis': 'h5', 'label': 1}]}")
    assert len(r5) == 1 and r5[0]["premise"] == "p5"

    print("PASS  test_robust_parser")


def test_incremental_authoring() -> None:
    client = FakeAuthor("sdk_tool_routing", per_call=10)
    existing = [
        {"id": "sdk_tool_routing_000000", "premise": "sdk_tool_routing premise 1000",
         "hypothesis": "sdk_tool_routing hypothesis 1000", "label": 1, "source": "sdk_tool_routing"},
        {"id": "sdk_tool_routing_000001", "premise": "sdk_tool_routing premise 1001",
         "hypothesis": "sdk_tool_routing hypothesis 1001", "label": 2, "source": "sdk_tool_routing"},
    ]
    # Target 5, already have 2 -> should author 3 more
    new_rows = author_mode_samples(client, "sdk_tool_routing", 5, existing_samples=existing)
    assert len(new_rows) == 3, f"Expected 3 new rows, got {len(new_rows)}"
    assert new_rows[0]["id"] == "sdk_tool_routing_000002"

    # Target 2, already have 2 -> should return empty list
    skip_rows = author_mode_samples(client, "sdk_tool_routing", 2, existing_samples=existing)
    assert len(skip_rows) == 0, f"Expected 0 new rows, got {len(skip_rows)}"

    print("PASS  test_incremental_authoring")


if __name__ == "__main__":
    test_authoring_produces_novel_samples()
    test_authoring_dedups_against_seen_keys()
    test_authoring_aborts_on_degenerate_diversity()
    test_include_overrides_amendment()
    test_robust_parser()
    test_incremental_authoring()
    print("\n6/6 teacher-authoring tests passed")
