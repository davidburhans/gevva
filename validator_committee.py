#!/usr/bin/env python3
"""validator_committee.py - Multi-judge NLI consensus, disagreement queue & judge metrics DB.

Design goals:
1. NO model thrashing: llama-swap runs with `--models-max 1`, so validators run
   judge-outer / batch-inner; each judge is unloaded exactly once after its last batch.
2. Disagreement tracking: non-unanimous committees, judge failures and unanimous
   overrides go to a JSONL review queue (`review.status = "pending"`) for human / cloud-LLM
   adjudication.
3. Judge scoring: every verdict lands in SQLite (`judge_performance` view) for per-judge
   agreement / failure / latency analysis to tune future runs.

Usage example:
    from validator_committee import RunSpec, ValidationMetricsDB, ...
    db = ValidationMetricsDB("data/validation_metrics.db")
    run_id = db.start_run(RunSpec("gemma-4-31b-q4", ["qwen-3.6-27b-q4"], 10, 42))
    committee = run_validator_committee(url, judges, samples, db, run_id)
    result = aggregate_committee_votes(samples, committee, judges)
"""

import json
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llm_client import LLMEndpointClient
from nli_labels import CONTRADICTION, ENTAILMENT, ID2LABEL, LABEL2ID, NEUTRAL

BATCH_VERDICT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]},
            "rationale": {"type": "string"}
        },
        "required": ["id", "verdict", "rationale"]
    }
}

VALIDATOR_SYSTEM_PROMPT = """You are a rigorous, strictly logical NLI Verifier and Judge.
Given a PREMISE and a HYPOTHESIS, classify the logical relationship into exactly one category:
- ENTAILMENT: The premise provides sufficient evidence to guarantee that the hypothesis is definitely true.
- CONTRADICTION: The hypothesis directly conflicts with, disproves, or is mutually exclusive with facts in the premise.
- NEUTRAL: The hypothesis might be true or false, but the premise does NOT provide sufficient proof. (IMPORTANT: Missing information is NEUTRAL, never Contradiction).

Respond ONLY with valid JSON conforming to the schema."""

# Verdict status values recorded in sample_verdicts.status.
STATUS_OK = "ok"                # judge returned a parseable in-enum verdict
STATUS_MISSING = "missing"      # sample absent from the judge's batch response
STATUS_INVALID = "invalid"      # verdict string outside the label enum
STATUS_PARSE_ERROR = "parse_error"  # batch response was not valid JSON
STATUS_OFFLINE = "offline"      # transport-level failure / empty response


def utc_now() -> str:
    """UTC ISO-8601 timestamp for DB rows."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's raw verdict for one sample.

    `label` is None exactly when the judge failed to classify (status != ok);
    failed verdicts never participate in the majority tally.
    """
    label: Optional[int]
    rationale: str
    status: str
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.label in (CONTRADICTION, ENTAILMENT, NEUTRAL)


@dataclass
class RunSpec:
    """Parameters describing one pipeline invocation, persisted in the `runs` table."""
    teacher_model: str
    validator_models: List[str]
    samples_per_mode: int
    seed: int


@dataclass
class CommitteeDecision:
    """Aggregation outcome for a single sample."""
    final_label: int
    confidence: float
    soft_labels: List[float]
    reason: str
    disagreement_type: Optional[str]  # None => clean unanimous consensus
    needs_review: bool
    severity: str                     # "low" | "medium" | "high" | "none"


@dataclass
class AggregateResult:
    validated: List[Dict[str, Any]]
    review_rows: List[Dict[str, Any]]
    final_rows: List[Tuple]
    stats: Dict[str, int] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# SQLite Judge Metrics DB
# -----------------------------------------------------------------------------
# WHY: DDL lives in validator_schema.sql, reviewable independently of wrapper code (SRP).
DB_SCHEMA = (Path(__file__).parent / "validator_schema.sql").read_text(encoding="utf-8")


class ValidationMetricsDB:
    """Thin SQLite wrapper persisting judge verdicts & batch telemetry for later analysis.

    Inject an alternate `conn` (e.g. `:memory:`) in tests.

    Example:
        db = ValidationMetricsDB("data/validation_metrics.db")
        run_id = db.start_run(RunSpec("gemma-4-31b-q4", ["qwen-3.6-27b-q4"], 10, 42))
        db.finish_run(run_id, "completed", total_samples=60)
    """

    def __init__(self, db_path: str, conn: Optional[sqlite3.Connection] = None):
        self.db_path = db_path
        if conn is not None:
            self.conn = conn
        else:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.conn = sqlite3.connect(db_path)
        # WHY: name-addressable rows (sqlite3.Row) so analysis queries read r["judge_model"].
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "ValidationMetricsDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def start_run(self, spec: RunSpec, run_id: Optional[str] = None) -> str:
        """Inserts a running run row; returns the run_id (timestamp-based by default)."""
        rid = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.conn.execute(
            "INSERT INTO runs (run_id, started_at, teacher_model, validator_models,"
            " samples_per_mode, seed, status) VALUES (?,?,?,?,?,?, 'running')",
            (rid, utc_now(), spec.teacher_model, json.dumps(spec.validator_models),
             spec.samples_per_mode, spec.seed),
        )
        self.conn.commit()
        return rid

    def finish_run(self, run_id: str, status: str = "completed", total_samples: Optional[int] = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, total_samples = ? WHERE run_id = ?",
            (utc_now(), status, total_samples, run_id),
        )
        self.conn.commit()

    def record_batch(self, run_id: str, judge_model: str, batch_index: int,
                     batch_size: int, parse_ok: bool, latency_ms: float) -> None:
        self.conn.execute(
            "INSERT INTO judge_batches (run_id, judge_model, batch_index, batch_size,"
            " parse_ok, latency_ms, created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, judge_model, batch_index, batch_size, int(parse_ok), latency_ms, utc_now()),
        )
        self.conn.commit()

    def record_verdicts(self, rows: Sequence[Tuple]) -> None:
        self.conn.executemany(
            "INSERT INTO sample_verdicts (run_id, sample_id, source, judge_model,"
            " generator_label, verdict_label, status, rationale, agreed_with_generator,"
            " latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def record_final_labels(self, rows: Sequence[Tuple]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO final_labels (run_id, sample_id, generator_label,"
            " final_label, confidence, needs_review, disagreement_type) VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def judge_summary(self, run_id: str) -> List[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM judge_performance WHERE run_id = ? ORDER BY judge_model", (run_id,))
        return cur.fetchall()


# -----------------------------------------------------------------------------
# Single-Judge Batch Validation (raw verdicts only; flips happen at aggregation)
# -----------------------------------------------------------------------------
def validate_batch_consensus(
    client: Any,
    candidate_batch: List[Dict[str, Any]],
) -> List[JudgeVerdict]:
    """Validates a mini-batch of candidate pairs in ONE LLM call, GBNF-constrained via
    a strict JSON schema (llama-server compiles `response_format` into a GBNF grammar).

    Returns one JudgeVerdict per candidate; verdicts are RAW — generator-label flipping
    (adversarial / neutral-boundary controls) is deliberately deferred to committee
    aggregation so votes stay interpretable in the metrics DB.

    Example:
        verdicts = validate_batch_consensus(client, [{"id": "s1", "premise": "p",
                                                      "hypothesis": "h", "label": 1}])
    """
    if not candidate_batch:
        return []

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "nli_batch_verdicts", "strict": True, "schema": BATCH_VERDICT_SCHEMA},
    }
    user_prompt = "Pairs to classify:\n"
    for idx, c in enumerate(candidate_batch):
        user_prompt += f"[ID {idx}]\nPREMISE: {c['premise']}\nHYPOTHESIS: {c['hypothesis']}\n\n"
    user_prompt += "JSON Array Output:"

    response = client.query_chat(
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=2048,
        response_format=response_format,
    )
    latency = getattr(client, "last_latency_ms", None) or 0.0
    if not response:
        return [JudgeVerdict(None, "validator_offline_or_empty", STATUS_OFFLINE, latency) for _ in candidate_batch]

    try:
        verdicts = json.loads(response)
        return [_parse_verdict(idx, c, {int(v["id"]): v for v in _as_verdict_list(verdicts)}, latency)
                for idx, c in enumerate(candidate_batch)]
    except Exception as e:
        return [JudgeVerdict(None, f"parse_error_fallback: {e}", STATUS_PARSE_ERROR, latency)
                for _ in candidate_batch]


def _as_verdict_list(parsed: Any) -> List[Any]:
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"expected JSON array, got {type(parsed).__name__}")


def _parse_verdict(idx: int, candidate: Dict[str, Any],
                   verdict_map: Dict[int, Any], latency: float) -> JudgeVerdict:
    v = verdict_map.get(idx)
    if v is None:
        return JudgeVerdict(None, "missing_from_batch_response", STATUS_MISSING, latency)
    verdict_name = str(v.get("verdict", "")).lower().strip()
    label = LABEL2ID.get(verdict_name, -1)
    if label not in (CONTRADICTION, ENTAILMENT, NEUTRAL):
        return JudgeVerdict(None, f"invalid_verdict: {verdict_name!r}", STATUS_INVALID, latency)
    return JudgeVerdict(label, str(v.get("rationale", "")), STATUS_OK, latency)


# -----------------------------------------------------------------------------
# Committee Runner (judge-outer / batch-inner: zero model thrashing)
# -----------------------------------------------------------------------------
def run_validator_committee(
    validator_url: str,
    validator_models: List[str],
    samples: List[Dict[str, Any]],
    db: Optional[ValidationMetricsDB],
    run_id: str,
    batch_size: int = 5,
    timeout: int = 180,
    client_factory: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, JudgeVerdict]]:
    """Runs every judge over ALL samples before switching models.

    WHY: llama-swap serves `--models-max 1`; per-batch model switching would force a
    multi-GB reload for every 5 samples. Batches are grouped per judge and each judge
    is unloaded exactly once after its last batch.

    Returns a list aligned with `samples`: committee[i][judge] = JudgeVerdict.
    `client_factory` is an injection point for tests (fake LLM clients).

    Example:
        committee = run_validator_committee(url, judges, samples, db, run_id, batch_size=5)
    """
    factory = client_factory or (lambda m: LLMEndpointClient(base_url=validator_url, model=m, timeout=timeout))
    committee: List[Dict[str, JudgeVerdict]] = [dict() for _ in samples]
    for judge_idx, judge_model in enumerate(validator_models, 1):
        _run_single_judge(factory(judge_model), judge_model, judge_idx, len(validator_models),
                          samples, committee, db, run_id, batch_size)
    return committee


def _run_single_judge(
    client: Any,
    judge_model: str,
    judge_idx: int,
    num_judges: int,
    samples: List[Dict[str, Any]],
    committee: List[Dict[str, JudgeVerdict]],
    db: Optional[ValidationMetricsDB],
    run_id: str,
    batch_size: int,
) -> None:
    total_batches = (len(samples) + batch_size - 1) // batch_size
    print(f"\n[Judge {judge_idx}/{num_judges}] {judge_model}: validating ALL {len(samples)} "
          f"samples across {total_batches} batches before switching models (no llama-swap thrashing)")
    for b_idx in range(0, len(samples), batch_size):
        chunk = samples[b_idx:b_idx + batch_size]
        verdicts = validate_batch_consensus(client, chunk)
        latency = getattr(client, "last_latency_ms", None) or 0.0
        if db is not None:
            db.record_batch(run_id, judge_model, b_idx // batch_size, len(chunk),
                            all(v.ok for v in verdicts), latency)
            db.record_verdicts(_verdict_rows(run_id, judge_model, chunk, verdicts, latency))
        for offset, (sample, verdict) in enumerate(zip(chunk, verdicts)):
            committee[b_idx + offset][judge_model] = verdict
    print(f"  Unloading {judge_model} after its final batch to free 100% VRAM for the next judge...")
    client.unload_model()


def _verdict_rows(run_id: str, judge_model: str, chunk: List[Dict[str, Any]],
                  verdicts: List[JudgeVerdict], latency_ms: float) -> List[Tuple]:
    now = utc_now()
    return [
        (run_id, s["id"], s.get("source", "unknown"), judge_model, s["label"],
         v.label, v.status, v.rationale,
         None if v.label is None else int(v.label == s["label"]),
         latency_ms, now)
        for s, v in zip(chunk, verdicts)
    ]


# -----------------------------------------------------------------------------
# Committee Aggregation & Disagreement Detection
# -----------------------------------------------------------------------------
def aggregate_committee_votes(
    samples: List[Dict[str, Any]],
    committee: List[Dict[str, JudgeVerdict]],
    validator_models: List[str],
) -> AggregateResult:
    """Aggregates raw judge verdicts into final labels, soft distributions and a
    disagreement review queue.

    Policy (in priority order):
    - no successful verdicts            -> keep generator label, review (high)
    - tally tie (incl. 1-1 + failure)   -> keep generator label, review (high)
    - strict majority                   -> majority label, review (high)
    - unanimous, but some judge failed  -> keep label, review (medium)
    - unanimous override of generator   -> keep flipped label (control), review (low/medium)
    - unanimous, complete, matches gen  -> clean consensus, no review

    Example:
        result = aggregate_committee_votes(samples, committee, judges)
    """
    validated: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    final_rows: List[Tuple] = []
    stats: Counter = Counter()

    for sample, votes in zip(samples, committee):
        decision = _resolve_sample(sample["label"], votes, validator_models)
        enriched = _apply_decision(sample, votes, decision)
        validated.append(enriched)
        stats[decision.disagreement_type or "unanimous_consensus"] += 1
        if decision.needs_review:
            review_rows.append(_review_row(enriched, votes, decision))
        final_rows.append(_final_row(enriched, decision))

    return AggregateResult(validated, review_rows, final_rows, dict(stats))


def _resolve_sample(gen_label: int, votes: Dict[str, JudgeVerdict],
                    expected_judges: List[str]) -> CommitteeDecision:
    ok = {j: v for j, v in votes.items() if v.ok}
    failed = [j for j in expected_judges if j not in ok]
    num_ok = len(ok)

    if num_ok == 0:
        return CommitteeDecision(gen_label, 0.0, [1 / 3] * 3, "all_judges_failed",
                                 "no_judge_verdicts", True, "high")

    tally = Counter(v.label for v in ok.values())
    max_count = max(tally.values())
    leaders = [label for label, count in tally.items() if count == max_count]

    if len(leaders) > 1:
        return CommitteeDecision(gen_label, max_count / num_ok, _soft(tally, num_ok),
                                 "committee_split_tie_kept_generator_label",
                                 "committee_split_tie", True, "high")

    final = leaders[0]
    soft = _soft(tally, num_ok)
    if max_count == num_ok:  # unanimous among successful judges
        if final == gen_label:
            if failed:
                return CommitteeDecision(final, 1.0, soft, "unanimous_with_judge_failures",
                                         "partial_committee_failure", True, "medium")
            return CommitteeDecision(final, 1.0, soft, "unanimous_committee_consensus", None, False, "none")
        return _override_decision(gen_label, final, soft)

    return CommitteeDecision(final, max_count / num_ok, soft,
                             f"majority_committee_vote_{max_count}_of_{num_ok}",
                             "committee_split", True, "high")


def _override_decision(gen_label: int, final: int, soft: List[float]) -> CommitteeDecision:
    """Unanimous validator override of the generator label (control harvesting)."""
    if gen_label == ENTAILMENT and final == CONTRADICTION:
        return CommitteeDecision(final, 1.0, soft, "committee_adversarial_negative_control",
                                 "unanimous_label_override", True, "low")
    if gen_label == ENTAILMENT and final == NEUTRAL:
        return CommitteeDecision(final, 1.0, soft, "committee_neutral_boundary_control",
                                 "unanimous_label_override", True, "low")
    if gen_label == CONTRADICTION and final == NEUTRAL:
        return CommitteeDecision(final, 1.0, soft, "committee_relabelled_neutral",
                                 "unanimous_label_override", True, "low")
    return CommitteeDecision(final, 1.0, soft, "committee_unanimous_override",
                             "unanimous_label_override", True, "medium")


def _soft(tally: Counter, num_ok: int) -> List[float]:
    return [round(tally.get(label, 0) / num_ok, 3) for label in (CONTRADICTION, ENTAILMENT, NEUTRAL)]


def _apply_decision(sample: Dict[str, Any], votes: Dict[str, JudgeVerdict],
                    decision: CommitteeDecision) -> Dict[str, Any]:
    sample = dict(sample)
    # WHY: preserve the pre-aggregation label for review rows & the metrics DB
    # before it gets overwritten by the committee verdict.
    sample["generator_label"] = sample["label"]
    sample["label"] = decision.final_label
    sample["confidence"] = round(decision.confidence, 3)
    sample["soft_labels"] = decision.soft_labels
    sample["validation_reason"] = decision.reason
    sample["needs_review"] = decision.needs_review
    sample["disagreement_type"] = decision.disagreement_type
    sample["committee_rationales"] = {j: v.rationale for j, v in votes.items()}
    return sample


def _review_row(enriched: Dict[str, Any], votes: Dict[str, JudgeVerdict],
                decision: CommitteeDecision) -> Dict[str, Any]:
    row = dict(enriched)
    gen_label = row.get("generator_label")
    row["review"] = {
        "status": "pending",
        "disagreement_type": decision.disagreement_type,
        "severity": decision.severity,
        "generator_label": gen_label,
        "generator_label_name": ID2LABEL.get(gen_label, "unknown"),
        "final_label": decision.final_label,
        "final_label_name": ID2LABEL.get(decision.final_label, "unknown"),
        "votes": {
            judge: {"verdict": ID2LABEL.get(v.label), "status": v.status, "rationale": v.rationale}
            for judge, v in votes.items()
        },
        "adjudicated_by": None,
    }
    return row


def _final_row(enriched: Dict[str, Any], decision: CommitteeDecision) -> Tuple:
    # 6-tuple WITHOUT run_id; callers prepend run_id for record_final_labels().
    return (enriched["id"], enriched.get("generator_label", decision.final_label),
            decision.final_label, decision.confidence,
            int(decision.needs_review), decision.disagreement_type)


def write_disagreement_queue(path: str, review_rows: List[Dict[str, Any]]) -> int:
    """Writes pending-disagreement samples to a JSONL review queue for human/cloud adjudication.

    Example:
        write_disagreement_queue("data/sdk_synthetic_disagreements.jsonl", result.review_rows)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(review_rows)


def print_judge_summary(db: ValidationMetricsDB, run_id: str) -> None:
    """Prints the per-judge scoring table (also queryable via the judge_performance view)."""
    rows = db.judge_summary(run_id)
    print("\nJudge Performance (this run, persisted in metrics DB):")
    print(f"  {'judge':28s} {'ok/total':>10s} {'success':>8s} {'vs_gen':>8s} {'vs_final':>9s} {'ms/batch':>9s}")
    for r in rows:
        print(f"  {r['judge_model']:28s} {r['ok_votes']:>4d}/{r['total_verdicts']:<5d} "
              f"{(r['success_rate'] or 0.0):>8.4f} {(r['generator_agreement'] or 0.0):>8.4f} "
              f"{(r['consensus_agreement'] or 0.0):>9.4f} {(r['avg_latency_ms'] or 0.0):>9.1f}")
