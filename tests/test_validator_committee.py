#!/usr/bin/env python3
"""tests/test_validator_committee.py - Offline tests for the validation committee.

Run (single command, no network, no GPU):
    uv run python tests/test_validator_committee.py

Covers: consensus/split/tie/failure aggregation policy, unanimous override controls,
judge-batching order (no llama-swap thrashing), SQLite metrics persistence and the
offline compile path (train/val split guard).
"""

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL  # noqa: E402
from validator_committee import (  # noqa: E402
    RunSpec,
    ValidationMetricsDB,
    aggregate_committee_votes,
    run_validator_committee,
    write_disagreement_queue,
)

JUDGES = ["judge-a", "judge-b", "judge-c"]


class EventLog:
    """Records (kind, model) tuples: load/batch/unload ordering across fake clients."""

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def record(self, kind: str, model: str) -> None:
        self.items.append((kind, model))


class FakeJudgeClient:
    """Scripted judge transport.

    verdict_by_premise maps PREMISE text -> verdict name; value None omits the sample
    from the batch response (STATUS_MISSING). `fail=True` simulates a transport failure.
    """

    def __init__(self, model: str, verdict_by_premise: Dict[str, Optional[str]],
                 log: EventLog, fail: bool = False):
        self.model = model
        self.verdict_by_premise = verdict_by_premise
        self.log = log
        self.fail = fail
        self.last_latency_ms = 42.0
        self.unload_count = 0
        log.record("load", model)

    def query_chat(self, system_prompt: str, user_prompt: str, **_: Any) -> Optional[str]:
        self.log.record("batch", self.model)
        if self.fail:
            return None
        verdicts = []
        for idx, premise, _hypo in re.findall(
                r"\[ID (\d+)\]\nPREMISE: (.*)\nHYPOTHESIS: (.*)\n", user_prompt):
            name = self.verdict_by_premise.get(premise)
            if name is not None:
                verdicts.append({"id": int(idx), "verdict": name,
                                 "rationale": f"{self.model} rationale"})
        return json.dumps(verdicts)

    def unload_model(self) -> bool:
        self.unload_count += 1
        self.log.record("unload", self.model)
        return True


def make_samples(labels: List[int]) -> List[Dict[str, Any]]:
    """Five-sample batch with distinct single-line premises p0..p4."""
    return [{"id": f"s{i:03d}", "premise": f"p{i}", "hypothesis": f"h{i}",
             "label": label, "source": "test_mode"}
            for i, label in enumerate(labels)]


def run_committee(samples: List[Dict[str, Any]], scripts: Dict[str, Dict[str, Optional[str]]],
                  log: EventLog, db: Optional[ValidationMetricsDB] = None,
                  run_id: str = "run_test", batch_size: int = 5,
                  fail: Optional[Dict[str, bool]] = None):
    fail = fail or {}
    def factory(model: str) -> FakeJudgeClient:
        return FakeJudgeClient(model, scripts.get(model, {}), log, fail=fail.get(model, False))
    return run_validator_committee("http://fake/v1", JUDGES, samples, db, run_id,
                                   batch_size=batch_size, client_factory=factory)


# -----------------------------------------------------------------------------
# Aggregation policy tests
# -----------------------------------------------------------------------------
def test_unanimous_consensus_is_clean():
    samples = make_samples([ENTAILMENT] * 5)
    agree = {f"p{i}": "entailment" for i in range(5)}
    committee = run_committee(samples, {j: agree for j in JUDGES}, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    assert len(result.review_rows) == 0, "unanimous consensus must not require review"
    assert result.stats == {"unanimous_consensus": 5}
    s = result.validated[0]
    assert s["label"] == ENTAILMENT and s["needs_review"] is False
    assert s["confidence"] == 1.0 and s["soft_labels"] == [0.0, 1.0, 0.0]
    assert s["generator_label"] == ENTAILMENT and s["disagreement_type"] is None


def test_majority_split_flips_label_and_reviews():
    samples = make_samples([ENTAILMENT])
    scripts = {"judge-a": {"p0": "contradiction"}, "judge-b": {"p0": "contradiction"},
               "judge-c": {"p0": "entailment"}}
    committee = run_committee(samples, scripts, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == CONTRADICTION, "2-of-3 majority must flip the label"
    assert s["soft_labels"] == [0.667, 0.333, 0.0]
    assert s["disagreement_type"] == "committee_split"
    assert s["validation_reason"] == "majority_committee_vote_2_of_3"
    assert len(result.review_rows) == 1
    review = result.review_rows[0]["review"]
    assert review["status"] == "pending" and review["severity"] == "high"
    assert review["generator_label"] == ENTAILMENT and review["final_label"] == CONTRADICTION
    assert set(review["votes"]) == set(JUDGES)


def test_tie_keeps_generator_label():
    samples = make_samples([ENTAILMENT])
    scripts = {"judge-a": {"p0": "contradiction"}, "judge-b": {"p0": "entailment"},
               "judge-c": {"p0": None}}  # judge-c omits the sample -> missing
    committee = run_committee(samples, scripts, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == ENTAILMENT, "1-1 tie with a failed judge must keep generator label"
    assert s["disagreement_type"] == "committee_split_tie" and s["needs_review"] is True


def test_all_judges_failing_keeps_generator_label():
    samples = make_samples([NEUTRAL])
    committee = run_committee(samples, {j: {} for j in JUDGES}, EventLog(),
                              fail={j: True for j in JUDGES})
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == NEUTRAL and s["confidence"] == 0.0
    assert s["disagreement_type"] == "no_judge_verdicts" and s["needs_review"] is True
    assert result.review_rows[0]["review"]["severity"] == "high"
    assert all(v["status"] == "offline" for v in result.review_rows[0]["review"]["votes"].values())


def test_unanimous_override_is_tracked_as_control():
    samples = make_samples([ENTAILMENT])
    neutral = {j: {"p0": "neutral"} for j in JUDGES}
    committee = run_committee(samples, neutral, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == NEUTRAL, "unanimous override must apply the flipped label"
    assert s["validation_reason"] == "committee_neutral_boundary_control"
    assert s["disagreement_type"] == "unanimous_label_override" and s["needs_review"] is True
    assert result.review_rows[0]["review"]["severity"] == "low"


def test_partial_failure_with_agreement_is_flagged():
    samples = make_samples([ENTAILMENT])
    scripts = {"judge-a": {"p0": "entailment"}, "judge-b": {"p0": "entailment"},
               "judge-c": {"p0": None}}
    committee = run_committee(samples, scripts, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == ENTAILMENT and s["confidence"] == 1.0
    assert s["disagreement_type"] == "partial_committee_failure"
    assert result.review_rows[0]["review"]["severity"] == "medium"


# -----------------------------------------------------------------------------
# Batching order (anti-thrashing) & DB persistence tests
# -----------------------------------------------------------------------------
def test_batches_grouped_per_judge_before_switching():
    samples = make_samples([ENTAILMENT] * 7)  # batch_size 5 -> 2 batches per judge
    log = EventLog()
    run_committee(samples, {j: {"p0": "entailment"} for j in JUDGES}, log, batch_size=5)
    expected = ([("load", "judge-a")] + [("batch", "judge-a")] * 2 + [("unload", "judge-a")]
                + [("load", "judge-b")] + [("batch", "judge-b")] * 2 + [("unload", "judge-b")]
                + [("load", "judge-c")] + [("batch", "judge-c")] * 2 + [("unload", "judge-c")])
    assert log.items == expected, f"judge switching must be strictly sequential, got {log.items}"


def test_metrics_db_persistence_and_view():
    with tempfile.TemporaryDirectory() as tmp:
        db = ValidationMetricsDB(str(Path(tmp) / "metrics.db"))
        samples = make_samples([ENTAILMENT, NEUTRAL, CONTRADICTION])
        scripts = {"judge-a": {"p0": "entailment", "p1": "neutral", "p2": "neutral"},
                   "judge-b": {"p0": "entailment", "p1": "neutral", "p2": "contradiction"},
                   "judge-c": {"p0": "entailment", "p1": "neutral", "p2": "contradiction"}}
        run_id = db.start_run(RunSpec("teacher-test", JUDGES, 3, 42))
        committee = run_committee(samples, scripts, EventLog(), db=db, run_id=run_id, batch_size=2)
        result = aggregate_committee_votes(samples, committee, JUDGES)
        db.record_final_labels([(run_id, *row) for row in result.final_rows])
        db.finish_run(run_id, "completed", total_samples=len(result.validated))

        run_row = db.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        assert run_row["status"] == "completed" and run_row["total_samples"] == 3
        n_verdicts = db.conn.execute("SELECT COUNT(*) FROM sample_verdicts").fetchone()[0]
        assert n_verdicts == 9, "3 samples x 3 judges must persist 9 verdict rows"
        n_batches = db.conn.execute("SELECT COUNT(*) FROM judge_batches").fetchone()[0]
        assert n_batches == 6, "batch_size=2 over 3 samples -> 2 batches per judge x 3 judges"
        perf = {r["judge_model"]: r for r in db.judge_summary(run_id)}
        assert perf["judge-a"]["success_rate"] == 1.0
        # judge-a dissents on p2 (neutral vs gen contradiction, losing 1-2 majority):
        # 2/3 agreement with both the generator labels and the committee finals.
        assert perf["judge-a"]["generator_agreement"] == round(2 / 3, 4)
        assert perf["judge-a"]["consensus_agreement"] == round(2 / 3, 4)
        assert perf["judge-c"]["generator_agreement"] == 1.0  # judges b & c match all generator labels
        assert perf["judge-c"]["consensus_agreement"] == 1.0  # judge-c always matches finals
        n_final = db.conn.execute("SELECT COUNT(*) FROM final_labels").fetchone()[0]
        assert n_final == 3
        db.close()


def test_disagreement_queue_writer():
    samples = make_samples([ENTAILMENT])
    scripts = {"judge-a": {"p0": "contradiction"}, "judge-b": {"p0": "contradiction"},
               "judge-c": {"p0": "entailment"}}
    committee = run_committee(samples, scripts, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    with tempfile.TemporaryDirectory() as tmp:
        queue_path = str(Path(tmp) / "nested" / "disagreements.jsonl")
        assert write_disagreement_queue(queue_path, result.review_rows) == 1
        rows = [json.loads(line) for line in open(queue_path, encoding="utf-8")]
        assert rows[0]["review"]["status"] == "pending"
        assert rows[0]["review"]["adjudicated_by"] is None
        assert rows[0]["generator_label"] == ENTAILMENT


def test_offline_compile_split_guard():
    """Regression: small runs previously produced val > train via max(100, 10%)."""
    from generate_sdk_synthetic_data import compile_sdk_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        counts = compile_sdk_synthetic_dataset(out_dir=tmp, samples_per_mode=25)
        train = [json.loads(l) for l in open(Path(tmp) / "sdk_synthetic_train.jsonl", encoding="utf-8")]
        val = [json.loads(l) for l in open(Path(tmp) / "sdk_synthetic_val.jsonl", encoding="utf-8")]
        total = sum(counts.values())
        assert len(train) + len(val) == total
        assert len(val) < len(train), "val split must stay smaller than train on small runs"
        assert len(val) == max(1, min(int(total * 0.1), total // 2))
        for row in train[:5]:
            assert {"id", "premise", "hypothesis", "label", "source"} <= set(row)


TESTS = [test_unanimous_consensus_is_clean,
         test_majority_split_flips_label_and_reviews,
         test_tie_keeps_generator_label,
         test_all_judges_failing_keeps_generator_label,
         test_unanimous_override_is_tracked_as_control,
         test_partial_failure_with_agreement_is_flagged,
         test_batches_grouped_per_judge_before_switching,
         test_metrics_db_persistence_and_view,
         test_disagreement_queue_writer,
         test_offline_compile_split_guard]


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
