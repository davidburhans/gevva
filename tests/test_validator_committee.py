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
from validation_metrics_db import RunSpec, ValidationMetricsDB  # noqa: E402
from validator_committee import (  # noqa: E402
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
        self.retry_delay = 0
        log.record("load", model)

    def query_chat(self, system_prompt: str, user_prompt: str, **_: Any) -> Optional[str]:
        self.log.record("batch", self.model)
        if self.fail:
            return None
        verdicts = []
        xml_matches = re.findall(
            r'<candidate id="(\d+)"><premise>(.*?)</premise><hypothesis>(.*?)</hypothesis></candidate>',
            user_prompt, re.DOTALL
        )
        matches = xml_matches if xml_matches else re.findall(
            r"\[ID (\d+)\]\nPREMISE: (.*)\nHYPOTHESIS: (.*)\n", user_prompt
        )
        for idx, premise, _hypo in matches:
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
                  fail: Optional[Dict[str, bool]] = None, judges: List[str] = JUDGES,
                  checkpoint_path: Optional[str] = None):
    fail = fail or {}
    def factory(model: str) -> FakeJudgeClient:
        return FakeJudgeClient(model, scripts.get(model, {}), log, fail=fail.get(model, False))
    return run_validator_committee("http://fake/v1", judges, samples, db, run_id,
                                   batch_size=batch_size, checkpoint_path=checkpoint_path,
                                   client_factory=factory)


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


def test_resume_skips_fully_persisted_judge_work():
    """A judge with all verdicts persisted must not even load its model on resume."""
    with tempfile.TemporaryDirectory() as tmp:
        db = ValidationMetricsDB(str(Path(tmp) / "metrics.db"))
        samples = make_samples([ENTAILMENT] * 4)
        scripts = {j: {f"p{i}": "entailment" for i in range(4)} for j in JUDGES[:2]}
        run_id = db.start_run(RunSpec("teacher-test", JUDGES[:2], 4, 42))
        run_committee(samples, scripts, EventLog(), db=db, run_id=run_id,
                      batch_size=2, judges=JUDGES[:2])
        # Simulate a crash right after judge-a finished: wipe judge-b's verdicts.
        db.conn.execute("DELETE FROM sample_verdicts WHERE judge_model = ?", ("judge-b",))
        db.conn.commit()

        log2 = EventLog()
        committee2 = run_committee(samples, scripts, log2, db=db, run_id=run_id,
                                   batch_size=2, judges=JUDGES[:2])
        assert [e for e in log2.items if e[1] == "judge-a"] == [], \
            "fully-persisted judge must be skipped without loading or batching"
        assert log2.items == [("load", "judge-b"), ("batch", "judge-b"),
                              ("batch", "judge-b"), ("unload", "judge-b")]
        assert db.count_verdicts(run_id) == 8, "resume must never duplicate verdict rows"
        for i, s in enumerate(samples):
            assert committee2[i]["judge-a"].label == ENTAILMENT, \
                "restored verdicts must match the original run"
        db.close()


def test_validation_checkpoint_rewritten_after_each_judge():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "validation_checkpoint.jsonl")
        samples = make_samples([ENTAILMENT] * 3)
        scripts = {j: {f"p{i}": "entailment" for i in range(3)} for j in JUDGES}
        run_committee(samples, scripts, EventLog(), batch_size=3, checkpoint_path=ckpt_path)
        rows = [json.loads(line) for line in open(ckpt_path, encoding="utf-8")]
        assert len(rows) == 3
        for row in rows:
            assert set(row["votes"]) == set(JUDGES)
            assert all(v["label"] == ENTAILMENT and v["status"] == "ok"
                       for v in row["votes"].values())
        assert not (Path(tmp) / "validation_checkpoint.jsonl.tmp").exists(), \
            "checkpoint writes must be atomic (tmp file renamed)"


def test_raw_checkpoint_survives_validation_crash():
    """Regression: generated data must hit disk BEFORE the committee stage runs."""
    import generate_sdk_synthetic_data as gen
    original_stage = gen.run_validation_committee_stage

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated llama-swap failure before the first batch")

    gen.run_validation_committee_stage = boom
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raised = False
            try:
                gen.compile_sdk_synthetic_dataset(out_dir=tmp, samples_per_mode=10,
                                                  validator_url="http://fake/v1")
            except RuntimeError:
                raised = True
            assert raised, "simulated committee failure must propagate"
            raw = [json.loads(line) for line in
                   open(Path(tmp) / "sdk_synthetic_raw.jsonl", encoding="utf-8")]
            assert len(raw) == 50, "all 5 offline modes x 10 samples must be checkpointed pre-validation"
            assert all({"id", "premise", "hypothesis", "label", "source"} <= set(r) for r in raw)
            assert not (Path(tmp) / "sdk_synthetic_train.jsonl").exists()
    finally:
        gen.run_validation_committee_stage = original_stage


def test_offline_batch_is_retried_once():
    """A transport blip must not permanently bake STATUS_OFFLINE for a batch."""
    class FlakyClient:
        def __init__(self):
            self.calls = 0
            self.last_latency_ms = 5.0
            self.retry_delay = 0.01
        def query_chat(self, system_prompt, user_prompt, **_):
            self.calls += 1
            if self.calls == 1:
                return None  # simulate llama-swap timeout
            return json.dumps([{"id": 0, "verdict": "entailment", "rationale": "ok"}])
        def unload_model(self):
            return True

    samples = make_samples([ENTAILMENT])
    flaky = FlakyClient()
    committee = run_validator_committee(
        "http://fake/v1", ["judge-a"], samples, None, "run_retry",
        batch_size=1, client_factory=lambda m: flaky)
    assert flaky.calls == 2, "offline batch must be retried exactly once"
    assert committee[0]["judge-a"].label == ENTAILMENT
    assert committee[0]["judge-a"].status == "ok"


def test_lone_judge_override_rejected_for_insufficient_quorum():
    """A single judge cannot override generator label (quorum >= 2 required)."""
    samples = make_samples([ENTAILMENT])
    scripts = {"judge-a": {"p0": "contradiction"}, "judge-b": {"p0": None}, "judge-c": {"p0": None}}
    committee = run_committee(samples, scripts, EventLog())
    result = aggregate_committee_votes(samples, committee, JUDGES)
    s = result.validated[0]
    assert s["label"] == ENTAILMENT, "lone judge must not override generator label"
    assert s["needs_review"] is True
    assert s["disagreement_type"] == "insufficient_quorum"
    assert result.review_rows[0]["review"]["severity"] == "high"


def test_relative_plurality_without_strict_majority_keeps_generator():
    """Relative plurality (e.g. 2 of 4) without strict majority (> 50%) keeps generator label."""
    from validator_committee import _resolve_sample, JudgeVerdict, STATUS_OK
    votes = {
        "j1": JudgeVerdict(CONTRADICTION, "c", STATUS_OK, 10.0),
        "j2": JudgeVerdict(CONTRADICTION, "c", STATUS_OK, 10.0),
        "j3": JudgeVerdict(ENTAILMENT, "e", STATUS_OK, 10.0),
        "j4": JudgeVerdict(NEUTRAL, "n", STATUS_OK, 10.0),
    }
    decision = _resolve_sample(ENTAILMENT, votes, ["j1", "j2", "j3", "j4"])
    assert decision.final_label == ENTAILMENT, "relative plurality 2-of-4 must not override generator"
    assert decision.needs_review is True
    assert decision.disagreement_type == "committee_split_tie"


def test_raw_checkpoint_truncation_prevented_without_force():
    """compile_sdk_synthetic_dataset must not wipe existing raw checkpoint without force=True."""
    from generate_sdk_synthetic_data import compile_sdk_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "sdk_synthetic_raw.jsonl"
        raw_path.write_text(json.dumps({"id": "dummy"}) + "\n", encoding="utf-8")
        try:
            compile_sdk_synthetic_dataset(out_dir=tmp, samples_per_mode=5, force=False)
            assert False, "must raise FileExistsError when raw_path already exists without force"
        except FileExistsError as e:
            assert "already exists" in str(e)


def test_compile_filters_out_needs_review_samples():
    """Samples flagged with needs_review=True must never enter train or val splits."""
    from generate_sdk_synthetic_data import compile_sdk_synthetic_dataset
    import generate_sdk_synthetic_data as gen

    original_stage = gen.run_validation_committee_stage

    def fake_committee(samples, *args, **kwargs):
        res = []
        for i, s in enumerate(samples):
            item = dict(s)
            if i % 2 == 1:
                item["needs_review"] = True
                item["disagreement_type"] = "committee_split"
            else:
                item["needs_review"] = False
                item["disagreement_type"] = None
            res.append(item)
        return res

    gen.run_validation_committee_stage = fake_committee
    try:
        with tempfile.TemporaryDirectory() as tmp:
            counts = compile_sdk_synthetic_dataset(out_dir=tmp, samples_per_mode=10, validator_url="http://fake/v1")
            train = [json.loads(line) for line in open(Path(tmp) / "sdk_synthetic_train.jsonl", encoding="utf-8")]
            val = [json.loads(line) for line in open(Path(tmp) / "sdk_synthetic_val.jsonl", encoding="utf-8")]
            all_split = train + val
            assert len(all_split) > 0
            assert all(not r.get("needs_review", False) for r in all_split), "no needs_review rows allowed in train/val!"
            assert len(all_split) == 25, f"expected 25 clean samples out of 50, got {len(all_split)}"
    finally:
        gen.run_validation_committee_stage = original_stage


TESTS = [test_unanimous_consensus_is_clean,
         test_majority_split_flips_label_and_reviews,
         test_tie_keeps_generator_label,
         test_all_judges_failing_keeps_generator_label,
         test_unanimous_override_is_tracked_as_control,
         test_partial_failure_with_agreement_is_flagged,
         test_batches_grouped_per_judge_before_switching,
         test_metrics_db_persistence_and_view,
         test_disagreement_queue_writer,
         test_resume_skips_fully_persisted_judge_work,
         test_offline_batch_is_retried_once,
         test_validation_checkpoint_rewritten_after_each_judge,
         test_raw_checkpoint_survives_validation_crash,
         test_offline_compile_split_guard,
         test_lone_judge_override_rejected_for_insufficient_quorum,
         test_relative_plurality_without_strict_majority_keeps_generator,
         test_raw_checkpoint_truncation_prevented_without_force,
         test_compile_filters_out_needs_review_samples]


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
