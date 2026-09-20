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
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import LLMEndpointClient
from nli_labels import CONTRADICTION, ENTAILMENT, ID2LABEL, LABEL2ID, NEUTRAL
from validation_metrics_db import RunSpec, ValidationMetricsDB, utc_now  # noqa: F401 (re-export)

BATCH_VERDICT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]},
            "rationale": {"type": "string", "maxLength": 240}
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
# Single-Judge Batch Validation (raw verdicts only; flips happen at aggregation)
# -----------------------------------------------------------------------------
# Per-model thinking policy for batch validation (grounded by live A/B, 2026-09-20):
# - qwen-3.6-27b-q4: thinking emits ~9K chars of hidden reasoning that truncates the
#   token budget before the JSON completes (89% parse_error, 20.3s/batch). Off:
#   3.2s/batch, 100% parse.
# - deepseek-v4-flash-q3: thinking-OFF breaks batch completeness (returns 1 of 5
#   items); its default mode delivers full batches at ~82s. Leave at vendor default.
# - unknown models default to vendor behaviour; the watchdog alerts on failure rates.
THINKING_DISABLED_MODELS = {"qwen-3.6-27b-q4"}


def _chat_template_kwargs(client: Any) -> Optional[Dict[str, Any]]:
    model = str(getattr(client, "model", ""))
    if any(tag in model for tag in THINKING_DISABLED_MODELS):
        return {"enable_thinking": False}
    return None


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
        max_tokens=3072,
        response_format=response_format,
        # WHY (measured 2026-09-20): with default thinking, validators emit ~9K chars of
        # hidden reasoning that truncates the 2048-3072 token budget before the JSON
        # completes (89% parse_error). Thinking-off: 3.2s vs 20.3s per batch, 100% parse.
        # WHY: see THINKING_DISABLED_MODELS - hidden reasoning truncates the token
        # budget before the JSON completes on affected models (89% parse_error measured).
        chat_template_kwargs=_chat_template_kwargs(client),
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
@dataclass
class CommitteeContext:
    """Shared state threaded through the per-judge batch loops."""
    samples: List[Dict[str, Any]]
    committee: List[Dict[str, JudgeVerdict]]
    db: Optional[ValidationMetricsDB]
    run_id: str
    batch_size: int
    checkpoint_path: Optional[str] = None


def run_validator_committee(
    validator_url: str,
    validator_models: List[str],
    samples: List[Dict[str, Any]],
    db: Optional[ValidationMetricsDB],
    run_id: str,
    batch_size: int = 5,
    timeout: int = 600,
    checkpoint_path: Optional[str] = None,
    client_factory: Optional[Callable[[str], Any]] = None,
) -> List[Dict[str, JudgeVerdict]]:
    """Runs every judge over ALL samples before switching models (resume-aware).

    WHY no thrashing: llama-swap serves `--models-max 1`; batches are grouped per judge
    and each judge is unloaded exactly once after its last batch.
    WHY resumable: verdicts commit to SQLite per batch. On restart, judges with fully
    persisted samples are skipped WITHOUT loading the model, partially-persisted judges
    only re-run incomplete batches, and the checkpoint JSONL is rewritten after each
    judge completes.

    Returns a list aligned with `samples`: committee[i][judge] = JudgeVerdict.
    `client_factory` is an injection point for tests (fake LLM clients).

    Example:
        committee = run_validator_committee(url, judges, samples, db, run_id, batch_size=5)
    """
    factory = client_factory or (lambda m: LLMEndpointClient(base_url=validator_url, model=m, timeout=timeout))
    ctx = CommitteeContext(samples=samples, committee=[dict() for _ in samples],
                           db=db, run_id=run_id, batch_size=batch_size,
                           checkpoint_path=checkpoint_path)
    for judge_idx, judge_model in enumerate(validator_models, 1):
        _run_single_judge(factory, judge_model, judge_idx, len(validator_models), ctx)
    return ctx.committee


def _run_single_judge(factory: Callable[[str], Any], judge_model: str,
                      judge_idx: int, num_judges: int, ctx: CommitteeContext) -> None:
    if ctx.db is not None:
        # WHY: hygiene at judge start - failed rows (offline/parse) from a previous
        # run must be regenerated, never baked into the done-set (audit MEDIUM).
        purged = ctx.db.purge_non_ok(judge_model)
        if purged:
            log_line = f"  [{judge_model}] purged {purged} stale failed verdicts at resume"
            print(log_line, flush=True)
    done = (ctx.db.persisted_verdicts(ctx.run_id, judge_model, [s["id"] for s in ctx.samples])
            if ctx.db else {})
    pending = [i for i, s in enumerate(ctx.samples) if s["id"] not in done]
    if done:
        n_failed = sum(1 for r in done.values() if r["status"] != STATUS_OK)
        if n_failed:
            print(f"  WARNING: {n_failed}/{len(done)} persisted verdicts for {judge_model} are "
                  f"FAILURES (offline/parse_error) - purge rows or they count as done on resume")
    print(f"\n[Judge {judge_idx}/{num_judges}] {judge_model}: "
          f"{len(ctx.samples) - len(pending)}/{len(ctx.samples)} verdicts persisted, "
          f"{len(pending)} remaining - all batches run before switching models (no thrashing)")
    if not pending:
        _restore_persisted_verdicts(done, judge_model, ctx)
        _write_checkpoint(ctx)
        return
    # WHY: the client is constructed lazily so a fully-persisted judge never even loads.
    client = factory(judge_model)
    total_batches = (len(ctx.samples) + ctx.batch_size - 1) // ctx.batch_size
    for b_idx in range(0, len(ctx.samples), ctx.batch_size):
        _validate_batch_or_restore(client, judge_model, b_idx, total_batches, done, ctx)
    print(f"  Unloading {judge_model} after its final batch to free 100% VRAM for the next judge...")
    client.unload_model()
    _write_checkpoint(ctx)


def _validate_batch_or_restore(client: Any, judge_model: str, b_idx: int, total_batches: int,
                               done: Dict[str, Any], ctx: CommitteeContext) -> None:
    """Validates one batch via LLM, or restores it from the DB when already persisted."""
    chunk = ctx.samples[b_idx:b_idx + ctx.batch_size]
    if all(s["id"] in done for s in chunk):
        for offset, sample in enumerate(chunk):
            ctx.committee[b_idx + offset][judge_model] = _verdict_from_row(done[sample["id"]])
        return
    verdicts = validate_batch_consensus(client, chunk)
    # WHY: one bounded retry for transport-level timeouts - a llama-swap blip would
    # otherwise permanently bake STATUS_OFFLINE verdicts for a whole batch (audit
    # finding: no-retry silently shrinks real judge coverage on multi-day runs).
    if all(v.status == STATUS_OFFLINE for v in verdicts):
        time.sleep(30)
        verdicts = validate_batch_consensus(client, chunk)
    latency = getattr(client, "last_latency_ms", None) or 0.0
    if ctx.db is not None:
        ctx.db.record_batch(ctx.run_id, judge_model, b_idx // ctx.batch_size, len(chunk),
                            all(v.ok for v in verdicts), latency)
        ctx.db.record_verdicts(_verdict_rows(ctx.run_id, judge_model, chunk, verdicts, latency))
    for offset, (sample, verdict) in enumerate(zip(chunk, verdicts)):
        ctx.committee[b_idx + offset][judge_model] = verdict


def _restore_persisted_verdicts(done: Dict[str, Any], judge_model: str,
                                ctx: CommitteeContext) -> None:
    """Rebuilds committee verdicts entirely from DB rows (fully-done judge on resume)."""
    for i, sample in enumerate(ctx.samples):
        if sample["id"] in done:
            ctx.committee[i][judge_model] = _verdict_from_row(done[sample["id"]])


def _verdict_from_row(row: Any) -> JudgeVerdict:
    """Rebuilds a JudgeVerdict from a persisted sample_verdicts row (resume path)."""
    return JudgeVerdict(label=row["verdict_label"], rationale=row["rationale"] or "",
                        status=row["status"], latency_ms=row["latency_ms"] or 0.0)


def _write_checkpoint(ctx: CommitteeContext) -> None:
    """Atomically snapshots all verdicts so far (one JSON line per sample, tmp+rename)."""
    if not ctx.checkpoint_path:
        return
    tmp_path = ctx.checkpoint_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sample, votes in zip(ctx.samples, ctx.committee):
            row = {"id": sample["id"], "votes": {
                judge: {"label": v.label, "status": v.status, "rationale": v.rationale}
                for judge, v in votes.items()}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ctx.checkpoint_path)


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
        return _override_decision(gen_label, final, soft, num_ok, len(failed))

    return CommitteeDecision(final, max_count / num_ok, soft,
                             f"majority_committee_vote_{max_count}_of_{num_ok}",
                             "committee_split", True, "high")


def _override_decision(gen_label: int, final: int, soft: List[float], n_ok: int, n_failed: int) -> CommitteeDecision:
    """Unanimous validator override of the generator label (control harvesting).

    Audit fix (CRITICAL): a 'unanimous' override from an incomplete committee is NOT
    a low-severity control. If any judge failed, confidence reflects the reduced
    quorum and severity escalates (remaining judges may share a systematic bias).
    """
    confidence = round(n_ok / (n_ok + n_failed), 3) if (n_ok + n_failed) else 0.0
    degraded = n_failed > 0
    if gen_label == ENTAILMENT and final == CONTRADICTION:
        return CommitteeDecision(final, confidence, soft, "committee_adversarial_negative_control",
                                 "unanimous_label_override", True,
                                 "medium" if degraded else "low")
    if gen_label == ENTAILMENT and final == NEUTRAL:
        return CommitteeDecision(final, confidence, soft, "committee_neutral_boundary_control",
                                 "unanimous_label_override", True,
                                 "medium" if degraded else "low")
    if gen_label == CONTRADICTION and final == NEUTRAL:
        return CommitteeDecision(final, confidence, soft, "committee_relabelled_neutral",
                                 "unanimous_label_override", True,
                                 "medium" if degraded else "low")
    return CommitteeDecision(final, confidence, soft, "committee_unanimous_override",
                             "unanimous_label_override", True,
                             "high" if degraded else "medium")


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
