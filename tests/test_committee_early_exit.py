#!/usr/bin/env python3
"""tests/test_committee_early_exit.py - Cascade committee tests (2026-09-30 round 4).

Run:
    uv run python tests/test_committee_early_exit.py

Covers the operator-directed early-exit cascade: judges run in order; once two
judges agree on a sample, later judges skip it; the tiebreaker only arbitrates
disagreements; aggregation counts only judges that ran (skipped != failed).
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_validator_committee import EventLog, FakeJudgeClient, make_samples  # noqa: E402
from validator_committee import aggregate_committee_votes, run_validator_committee  # noqa: E402
from validation_metrics_db import ValidationMetricsDB  # noqa: E402

JUDGES = ["judge-a", "judge-b", "judge-c"]


def run_cascade(samples, scripts, log, early_exit=True, run_id="run_early",
                batch_size=5, db=None):
    def factory(model: str) -> FakeJudgeClient:
        return FakeJudgeClient(model, scripts[model], log)
    return run_validator_committee(
        "http://fake/v1", JUDGES, samples, db, run_id,
        batch_size=batch_size, client_factory=factory, early_exit=early_exit)


def judge_loads(log: EventLog) -> List[str]:
    return [m for kind, m in log.items if kind == "load"]


def test_third_judge_skipped_when_first_two_agree() -> None:
    samples = make_samples([1, 1, 1])
    log = EventLog()
    scripts = {j: {"p0": "entailment", "p1": "entailment", "p2": "entailment"} for j in JUDGES}
    committee = run_cascade(samples, scripts, log)
    assert judge_loads(log) == ["judge-a", "judge-b"], judge_loads(log)
    # No judge-c verdicts at all: every sample settled by a/b agreement
    assert all("judge-c" not in votes for votes in committee)
    result = aggregate_committee_votes(samples, committee, JUDGES)
    # Settled 2/2 agreement is clean consensus, NOT "unanimous with failure"
    assert result.stats.get("unanimous_consensus", 0) == 3, result.stats
    assert not result.review_rows, result.review_rows
    print("PASS  test_third_judge_skipped_when_first_two_agree")


def test_third_judge_arbitrates_disagreement() -> None:
    samples = make_samples([1, 1, 1])
    log = EventLog()
    scripts = {
        "judge-a": {"p0": "entailment", "p1": "entailment", "p2": "entailment"},
        "judge-b": {"p0": "entailment", "p1": "neutral", "p2": "entailment"},
        "judge-c": {"p0": "neutral", "p1": "neutral", "p2": "neutral"},
    }
    committee = run_cascade(samples, scripts, log)
    assert judge_loads(log) == JUDGES, judge_loads(log)
    # p0 and p2 settled by a/b; only p1 went to judge-c
    assert "judge-c" in committee[1] and "judge-c" not in committee[0]
    assert "judge-c" not in committee[2]
    result = aggregate_committee_votes(samples, committee, JUDGES)
    # p1: b+c majority neutral overrides generator label 1, lands in review queue
    flipped = [r for r in result.review_rows
               if r.get("final_label", r.get("label")) != r["generator_label"]]
    assert flipped and "majority_committee_vote_2_of_3" in str(flipped[0]), result.review_rows
    print("PASS  test_third_judge_arbitrates_disagreement")


def test_failed_verdict_still_counts_as_ran() -> None:
    """A judge that ran and errored must not be treated as skipped."""
    samples = make_samples([1, 1])
    scripts = {j: {"p0": "entailment", "p1": "entailment"} for j in JUDGES}
    log = EventLog()
    # judge-b fails transport twice -> STATUS_OFFLINE entries present, not absent
    fail_client = FakeJudgeClient("judge-b", scripts["judge-b"], log, fail=True)
    fail_client.query_chat_orig = fail_client.query_chat

    def factory(model: str) -> FakeJudgeClient:
        if model == "judge-b":
            return fail_client
        return FakeJudgeClient(model, scripts[model], log)

    committee = run_validator_committee(
        "http://fake/v1", JUDGES, samples, None, "run_fail",
        batch_size=5, client_factory=factory, early_exit=True)
    assert "judge-b" in committee[0]  # present with non-OK status
    result = aggregate_committee_votes(samples, committee, JUDGES)
    # a failed judge is a failure, not a skip: unanimous-with-failure review expected.
    # (Without absent==skipped, settled samples would have been miscounted too.)
    assert result.stats.get("partial_committee_failure", 0) == 2, result.stats
    reasons = {r.get("validation_reason") for r in result.validated}
    assert reasons == {"unanimous_with_judge_failures"}, reasons
    assert len(result.review_rows) == 2, result.review_rows
    print("PASS  test_failed_verdict_still_counts_as_ran")


def test_db_resume_with_subset() -> None:
    """Early-exit + SQLite resume: persisted judge-a verdicts restore without reload."""
    with tempfile.TemporaryDirectory() as td:
        db = ValidationMetricsDB(str(Path(td) / "m.db"))
        db.start_run.__doc__  # noqa: B018 - presence check
        samples = make_samples([1, 1])
        scripts = {j: {"p0": "entailment", "p1": "entailment"} for j in JUDGES}
        log1 = EventLog()
        run_cascade(samples, scripts, log1, run_id="run_resume", db=db)
        # Re-run same run id: everything persisted -> no model loads at all
        log2 = EventLog()
        run_cascade(samples, scripts, log2, run_id="run_resume", db=db)
        assert judge_loads(log2) == [], judge_loads(log2)
        print("PASS  test_db_resume_with_subset")


def test_early_exit_disabled_runs_all() -> None:
    samples = make_samples([1, 1])
    scripts = {j: {"p0": "entailment", "p1": "entailment"} for j in JUDGES}
    log = EventLog()
    committee = run_cascade(samples, scripts, log, early_exit=False)
    assert judge_loads(log) == JUDGES
    assert all("judge-c" in votes for votes in committee)
    print("PASS  test_early_exit_disabled_runs_all")


if __name__ == "__main__":
    test_third_judge_skipped_when_first_two_agree()
    test_third_judge_arbitrates_disagreement()
    test_failed_verdict_still_counts_as_ran()
    test_db_resume_with_subset()
    test_early_exit_disabled_runs_all()
    print("\n5/5 cascade tests passed")
