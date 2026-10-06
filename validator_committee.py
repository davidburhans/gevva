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
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import LLMEndpointClient
from nli_labels import CONTRADICTION, ENTAILMENT, ID2LABEL, LABEL2ID, NEUTRAL
from validation_metrics_db import RunSpec, ValidationMetricsDB, utc_now  # noqa: F401 (re-export)

BATCH_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
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
    },
    "required": ["verdicts"]
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
# - qwen-3.6-27b-q4 & qwen-3.8-125b-*: thinking emits ~9K chars of hidden reasoning
#   that truncates the token budget before JSON completes (160s/batch, parse errors).
#   Thinking-off: 10.6s/batch, 100% parse.
# - deepseek-v4-flash-q3: thinking-OFF breaks batch completeness (returns 1 of 5
#   items); its default mode delivers full batches at ~82s. Leave at vendor default.
# - gpt-oss-120b (2026-09-29, replaces the duplicated qwen-3.8-125b-q3 judge to break
#   the correlated Qwen bloc): reasoning channel is harmony-format; cap it at low
#   effort so it does not crowd the GBNF-constrained JSON budget.
# - unknown models default to vendor behaviour; the watchdog alerts on failure rates.
THINKING_DISABLED_MODELS = {"qwen-3.6-27b-q4", "qwen-3.8-125b-q3", "qwen-3.8-125b-q4", "qwen3.8-flash-next-iq3_s"}
MODEL_TEMPLATE_KWARGS = {
    "qwen-3.6-27b-q4": {"enable_thinking": False},
    "qwen-3.8-125b-q3": {"enable_thinking": False},
    "qwen-3.8-125b-q4": {"enable_thinking": False},
    "qwen3.8-flash-next-iq3_s": {"enable_thinking": False},
    "gpt-oss-120b": {"reasoning_effort": "low"},
}


def _chat_template_kwargs(client: Any) -> Optional[Dict[str, Any]]:
    model = str(getattr(client, "model", ""))
    for tag, kwargs in MODEL_TEMPLATE_KWARGS.items():
        if tag in model:
            return dict(kwargs)
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

    def _escape_xml(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    n_items = len(candidate_batch)
    batch_schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": n_items,
                "maxItems": n_items,
                "items": BATCH_VERDICT_SCHEMA["properties"]["verdicts"]["items"],
            }
        },
        "required": ["verdicts"]
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "nli_batch_verdicts", "strict": True, "schema": batch_schema},
    }
    user_prompt = (
        f"There are {n_items} candidate pairs to classify (IDs 0 to {n_items - 1}). "
        f"You MUST evaluate all {n_items} pairs and return a JSON object with a 'verdicts' array of exactly {n_items} objects, "
        f"one for each ID in order.\n\nPairs to classify:\n"
    )
    for idx, c in enumerate(candidate_batch):
        p_esc = _escape_xml(c.get("premise", ""))
        h_esc = _escape_xml(c.get("hypothesis", ""))
        user_prompt += f'<candidate id="{idx}"><premise>{p_esc}</premise><hypothesis>{h_esc}</hypothesis></candidate>\n\n'
    user_prompt += f"Output a JSON object with a 'verdicts' array of {n_items} objects:"

    t0 = time.monotonic()
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
    latency = (time.monotonic() - t0) * 1000.0
    if not response:
        return [JudgeVerdict(None, "validator_offline_or_empty", STATUS_OFFLINE, latency) for _ in candidate_batch]

    clean_resp = response.strip()
    if clean_resp.startswith("```"):
        clean_resp = re.sub(r"^```(?:json)?\s*", "", clean_resp)
        clean_resp = re.sub(r"\s*```$", "", clean_resp)
        clean_resp = clean_resp.strip()

    try:
        verdicts = json.loads(clean_resp)
        return [_parse_verdict(idx, c, {int(v["id"]): v for v in _as_verdict_list(verdicts)}, latency)
                for idx, c in enumerate(candidate_batch)]
    except Exception as e:
        m = re.search(r"(\[.*\]|\{.*\})", clean_resp, re.DOTALL)
        if m:
            try:
                verdicts = json.loads(m.group(0))
                return [_parse_verdict(idx, c, {int(v["id"]): v for v in _as_verdict_list(verdicts)}, latency)
                        for idx, c in enumerate(candidate_batch)]
            except Exception:
                pass
        return [JudgeVerdict(None, f"parse_error_fallback: {e}", STATUS_PARSE_ERROR, latency)
                for _ in candidate_batch]


def _as_verdict_list(parsed: Any) -> List[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "verdicts" in parsed and isinstance(parsed["verdicts"], list):
        return parsed["verdicts"]
    raise ValueError(f"expected JSON array or object with 'verdicts', got {type(parsed).__name__}")


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
    early_exit: bool = False,
) -> List[Dict[str, JudgeVerdict]]:
    """Runs every judge over ALL samples before switching models (resume-aware).

    WHY no thrashing: llama-swap serves `--models-max 1`; batches are grouped per judge
    and each judge is unloaded exactly once after its last batch.
    WHY resumable: verdicts commit to SQLite per batch. On restart, judges with fully
    persisted samples are skipped WITHOUT loading the model, partially-persisted judges
    only re-run incomplete batches, and the checkpoint JSONL is rewritten after each
    judge completes.

    early_exit=True (2026-09-30, operator directive: "don't run all validators if
    early ones agree"): once >= 2 judges have delivered agreeing OK verdicts for a
    sample, later judges skip it. Judge ORDER therefore matters - put cheap,
    cross-family judges first; the expensive judge only arbitrates disagreements.
    A sample settled by k<all judges keeps expected_judges=k in aggregation, so
    "unanimous" means unanimous-among-the-judges-that-ran.

    Returns a list aligned with `samples`: committee[i][judge] = JudgeVerdict.
    `client_factory` is an injection point for tests (fake LLM clients).

    Example:
        committee = run_validator_committee(url, judges, samples, db, run_id, batch_size=5)
    """
    def _default_client_factory(m: str) -> LLMEndpointClient:
        # Route Strata models directly to port 8089 if requested or matched
        if "qwen3.8-flash-next" in m or ":8089" in validator_url:
            url = "http://127.0.0.1:8089/v1" if "qwen3.8-flash-next" in m else validator_url
            return LLMEndpointClient(base_url=url, model=m, timeout=timeout)
        return LLMEndpointClient(base_url=validator_url, model=m, timeout=timeout)

    factory = client_factory or _default_client_factory
    ctx = CommitteeContext(samples=samples, committee=[dict() for _ in samples],
                           db=db, run_id=run_id, batch_size=batch_size,
                           checkpoint_path=checkpoint_path)
    settled: set = set()  # sample indices with >= 2 agreeing OK verdicts
    for judge_idx, judge_model in enumerate(validator_models, 1):
        if early_exit:
            subset = [i for i in range(len(samples)) if i not in settled]
            if not subset:
                print(f"\n[early-exit] all {len(samples)} samples settled by earlier judges; "
                      f"skipping {judge_model} entirely", flush=True)
                break
            skipped = len(samples) - len(subset)
            if skipped:
                print(f"\n[early-exit] {judge_model} arbitrates {len(subset)} unsettled samples "
                      f"({skipped} already settled by agreement)", flush=True)
        else:
            subset = None
        _run_single_judge(factory, judge_model, judge_idx, len(validator_models), ctx, subset_idx=subset)
        if early_exit:
            _mark_settled(ctx, judge_idx, validator_models, settled)
    return ctx.committee


def _mark_settled(ctx: "CommitteeContext", judge_idx: int,
                  validator_models: List[str], settled: set) -> None:
    """Marks samples where all judges run so far (>= 2) returned the same OK label."""
    ran = validator_models[:judge_idx]
    if len(ran) < 2:
        return
    for i, votes in enumerate(ctx.committee):
        if i in settled:
            continue
        ok_labels = [votes[j].label for j in ran if j in votes and votes[j].ok]
        if len(ok_labels) >= 2 and len(set(ok_labels)) == 1:
            settled.add(i)


# Models in llama-server configured with --parallel >= 2
PARALLEL_JUDGES = {
    "qwen-3.8-125b-q3": 2,
    "qwen-3.8-125b-q4": 2,
    "qwen-3.6-27b-q4": 2,
    "gpt-oss-120b": 2,
    "qwen3.8-flash-next-iq3_s": 2,
}


def _run_single_judge(factory: Callable[[str], Any], judge_model: str,
                      judge_idx: int, num_judges: int, ctx: CommitteeContext,
                      subset_idx: Optional[List[int]] = None) -> None:
    """Runs one judge over ALL samples (subset_idx=None) or only the given positions.

    The early-exit cascade passes subset_idx for later judges; batches, DB keys and
    checkpoint lines stay sample-id scoped, so resume semantics are unchanged for the
    samples that run. Skipped samples simply carry no verdict from this judge and
    aggregation counts only the judges that ran for them.
    """
    positions = list(subset_idx) if subset_idx is not None else list(range(len(ctx.samples)))
    view = [ctx.samples[i] for i in positions]
    if ctx.db is not None:
        # WHY: hygiene at judge start - failed rows (offline/parse) from a previous
        # run must be regenerated, never baked into the done-set (audit MEDIUM).
        purged = ctx.db.purge_non_ok(judge_model, run_id=ctx.run_id)
        if purged:
            log_line = f"  [{judge_model}] purged {purged} stale failed verdicts at resume (run={ctx.run_id})"
            print(log_line, flush=True)
    done = (ctx.db.persisted_verdicts(ctx.run_id, judge_model, [s["id"] for s in view])
            if ctx.db else {})
    pending = [i for i, s in enumerate(view) if s["id"] not in done]
    if done:
        n_failed = sum(1 for r in done.values() if r["status"] != STATUS_OK)
        if n_failed:
            print(f"  WARNING: {n_failed}/{len(done)} persisted verdicts for {judge_model} are "
                  f"FAILURES (offline/parse_error) - purge rows or they count as done on resume")
    print(f"\n[Judge {judge_idx}/{num_judges}] {judge_model}: "
          f"{len(view) - len(pending)}/{len(view)} verdicts persisted, "
          f"{len(pending)} remaining - all batches run before switching models (no thrashing)")
    if not pending:
        for i, sample in enumerate(view):
            if sample["id"] in done:
                ctx.committee[positions[i]][judge_model] = _verdict_from_row(done[sample["id"]])
        _write_checkpoint(ctx)
        return
    # WHY: the client is constructed lazily so a fully-persisted judge never even loads.
    client = factory(judge_model)
    total_batches = (len(view) + ctx.batch_size - 1) // ctx.batch_size
    concurrency = PARALLEL_JUDGES.get(judge_model, 1)

    batch_indices = list(range(0, len(view), ctx.batch_size))
    # Restore any fully persisted batches first (fast in-memory)
    for b_idx in batch_indices:
        chunk = view[b_idx:b_idx + ctx.batch_size]
        if all(s["id"] in done for s in chunk):
            for offset, sample in enumerate(chunk):
                ctx.committee[positions[b_idx + offset]][judge_model] = _verdict_from_row(done[sample["id"]])

    pending_batches = [
        b_idx for b_idx in batch_indices
        if not all(s["id"] in done for s in view[b_idx:b_idx + ctx.batch_size])
    ]

    if concurrency > 1 and len(pending_batches) > 1:
        import concurrent.futures

        def _worker(b_idx: int) -> None:
            _validate_batch_or_restore(client, judge_model, b_idx, total_batches, done, ctx,
                                       view=view, positions=positions)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(_worker, pending_batches))
    else:
        for b_idx in pending_batches:
            _validate_batch_or_restore(client, judge_model, b_idx, total_batches, done, ctx,
                                       view=view, positions=positions)

    print(f"  Unloading {judge_model} after its final batch to free 100% VRAM for the next judge...")
    client.unload_model()
    _write_checkpoint(ctx)


def _validate_batch_or_restore(client: Any, judge_model: str, b_idx: int, total_batches: int,
                               done: Dict[str, Any], ctx: CommitteeContext,
                               view: Optional[List[Dict[str, Any]]] = None,
                               positions: Optional[List[int]] = None) -> None:
    """Validates one batch via LLM, or restores it from the DB when already persisted.

    view/positions support the early-exit cascade: the batch slices the judge's view
    and writes verdicts back at the mapped committee positions (identity when None).
    """
    if view is None:
        view = ctx.samples
        positions = list(range(len(ctx.samples)))
    chunk = view[b_idx:b_idx + ctx.batch_size]
    chunk_pos = positions[b_idx:b_idx + ctx.batch_size]
    if all(s["id"] in done for s in chunk):
        for pos, sample in zip(chunk_pos, chunk):
            ctx.committee[pos][judge_model] = _verdict_from_row(done[sample["id"]])
        return
    verdicts = validate_batch_consensus(client, chunk)
    # WHY: one bounded retry for transport-level timeouts - a llama-swap blip would
    # otherwise permanently bake STATUS_OFFLINE verdicts for a whole batch (audit
    # finding: no-retry silently shrinks real judge coverage on multi-day runs).
    if all(v.status == STATUS_OFFLINE for v in verdicts):
        retry_delay = getattr(client, "retry_delay", 30)
        if retry_delay > 0:
            time.sleep(retry_delay)
        verdicts = validate_batch_consensus(client, chunk)
    latency = getattr(client, "last_latency_ms", None) or 0.0
    if ctx.db is not None:
        ctx.db.record_batch(ctx.run_id, judge_model, b_idx // ctx.batch_size, len(chunk),
                            all(v.ok for v in verdicts), latency)
        ctx.db.record_verdicts(_verdict_rows(ctx.run_id, judge_model, chunk, verdicts, latency))
    for pos, verdict in zip(chunk_pos, verdicts):
        ctx.committee[pos][judge_model] = verdict


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
    include_unanimous_overrides: bool = False,
) -> AggregateResult:
    """Aggregates raw judge verdicts into final labels, soft distributions and a
    disagreement review queue.

    include_unanimous_overrides=True (2026-10-01, operator-approved): unanimous
    committee overrides of the generator label become TRAINING rows (label_source=
    committee_override) instead of review-only controls. Rationale: two
    cross-family judges agreeing against the generator is a strong label; the
    committee is the authority this pipeline exists to distill. Default False
    preserves legacy behavior for existing callers/tests.

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
        # Early-exit cascade (2026-09-30): a judge absent from `votes` was
        # intentionally skipped after earlier judges agreed - it is NOT a failure.
        # Judges that ran and errored are present with non-OK status, so absence
        # unambiguously means skipped. Expected judges shrink accordingly, keeping
        # "unanimous" truthful per sample.
        expected = [j for j in validator_models if j in votes] or list(validator_models)
        decision = _resolve_sample(sample["label"], votes, expected)
        if include_unanimous_overrides and decision.disagreement_type == "unanimous_label_override":
            # 2026-10-01 policy amendment: unanimous cross-family override of the
            # generator label is authoritative - train on it (label_source stamped)
            # instead of routing to the review queue. Degraded quorum (any judge
            # failed) still reviews: severity medium stays out of training.
            if decision.severity == "low":
                decision = CommitteeDecision(
                    decision.final_label, decision.confidence, decision.soft_labels,
                    decision.reason, decision.disagreement_type, False, "none")
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

    # Strict majority requirement: max_count > num_ok / 2 (relative plurality or ties do not suffice)
    if len(leaders) > 1 or max_count <= num_ok / 2:
        return CommitteeDecision(gen_label, max_count / num_ok, _soft(tally, num_ok),
                                 "committee_split_tie_kept_generator_label",
                                 "committee_split_tie", True, "high")

    final = leaders[0]
    soft = _soft(tally, num_ok)

    # Minimum quorum for label override: require >= 2 agreeing judges before allowing override
    if final != gen_label and (num_ok < 2 or max_count < 2):
        return CommitteeDecision(gen_label, max_count / num_ok, soft,
                                 "insufficient_quorum_kept_generator_label",
                                 "insufficient_quorum", True, "high")

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
    if decision.disagreement_type == "unanimous_label_override" and not decision.needs_review:
        sample["label_source"] = "committee_override"
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
