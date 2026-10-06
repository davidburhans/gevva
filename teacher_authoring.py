#!/usr/bin/env python3
"""teacher_authoring.py - Teacher-authored SDK decision data (round 4 rework, 2026-10-01).

The template generators draw from closed pools (~330 unique pairs for 10k rows;
worst mode: 15 unique pairs / 2,000). Teacher authoring makes the LLM the AUTHOR
of novel (premise, hypothesis, label) scenarios per SDK mode, with:
  - per-mode authoring specs + format exemplars (derived from the templates)
  - explicit diversity directives and anti-repetition context
  - exact-dedup against all previously accepted rows (in-run and prior raw file)
  - uniqueness gates: batch acceptance requires >=70% novel, full run requires
    >=80% unique or the generator ABORTS (fail-closed, proven 2026-09-30)

The committee validates teacher rows exactly like template rows; labels the
teacher suggests are generator labels subject to consensus (and unanimous
override by the committee wins, policy as amended 2026-10-01).
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence

BATCH_SIZE = 5
MIN_NOVEL_PER_BATCH = 3  # accept a batch only if >=60% of items are new
MIN_UNIQUE_RATIO = 0.80  # full-run gate

TEACHER_AUTHOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "premise": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "label": {"type": "integer", "enum": [0, 1, 2]},
                },
                "required": ["premise", "hypothesis", "label"],
            },
        },
    },
    "required": ["samples"],
}


def _key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256((premise + "\x00" + hypothesis).encode("utf-8")).hexdigest()


def robust_parse_authored_samples(raw_text: str) -> List[Dict[str, Any]]:
    """Robust multi-tier parser for LLM-authored samples.

    Tiers:
    1. json.loads on outer object/array slice
    2. ast.literal_eval (handles single quotes, trailing commas, Python dict literals)
    3. Regex single-quote normalization -> json.loads
    4. Greedy regex extraction of individual {"premise": ..., "hypothesis": ..., "label": ...} dicts
    """
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()

    start_obj = clean.find("{")
    end_obj = clean.rfind("}")
    start_arr = clean.find("[")
    end_arr = clean.rfind("]")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj) and end_arr > start_arr:
        clean_slice = clean[start_arr : end_arr + 1]
    elif start_obj != -1 and end_obj > start_obj:
        clean_slice = clean[start_obj : end_obj + 1]
    else:
        clean_slice = clean

    # 1. Direct json.loads
    try:
        parsed = json.loads(clean_slice)
        if isinstance(parsed, dict):
            items = parsed.get("samples", parsed.get("data", []))
            if isinstance(items, list):
                return items
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 2. ast.literal_eval
    try:
        parsed = ast.literal_eval(clean_slice)
        if isinstance(parsed, dict):
            items = parsed.get("samples", parsed.get("data", []))
            if isinstance(items, list):
                return items
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 3. Single-quote normalization
    try:
        normalized = re.sub(r"(?<=[\{\[,:])\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*(?=[\}\],:])", r'"\1"', clean_slice)
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            items = parsed.get("samples", parsed.get("data", []))
            if isinstance(items, list):
                return items
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 4. Truncated stream repair: if output was cut off, salvage completed objects
    last_brace = clean.rfind("}")
    if last_brace != -1:
        prefix = clean[:last_brace + 1]
        for suffix in ("\n]}", "\n]"):
            candidate = prefix + suffix
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    items = parsed.get("samples", parsed.get("data", []))
                    if isinstance(items, list) and items:
                        return items
                elif isinstance(parsed, list) and parsed:
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    items = parsed.get("samples", parsed.get("data", []))
                    if isinstance(items, list) and items:
                        return items
                elif isinstance(parsed, list) and parsed:
                    return parsed
            except Exception:
                pass

    # 5. Greedy regex fallback for individual dicts
    extracted = []
    sample_pattern = re.compile(
        r'\{[^{}]*?["\']premise["\']\s*:\s*(["\'])(.*?)\1\s*,\s*["\']hypothesis["\']\s*:\s*(["\'])(.*?)\3\s*,\s*["\']label["\']\s*:\s*([012])[^{}]*?\}',
        re.DOTALL,
    )
    for m in sample_pattern.finditer(clean):
        p, h, l = m.group(2), m.group(4), int(m.group(5))
        extracted.append({"premise": p, "hypothesis": h, "label": l})
    if extracted:
        return extracted

    return []


# Per-mode authoring specs. `exemplars` come from the template generators
# (deterministic, seed 42) so format parity with serving is preserved.
MODE_AUTHOR_SPECS: Dict[str, Dict[str, Any]] = {
    "sdk_tool_routing": {
        "name": "Tool Routing",
        "instructions": (
            "Each sample is a user request (premise, 'User request: ...') and a claim about which "
            "tool should handle it (hypothesis). Label 1 = the tool claim is appropriate for the "
            "request; label 0 = a clearly wrong tool; label 2 = partially relevant/ambiguous. "
            "Vary: request domain (smart home, finance, coding, travel, health, email, calendar, "
            "files, weather, shopping...), phrasing style, and specificity. Make some cases subtle "
            "(two plausible tools, boundary of a tool's scope)."
        ),
        "exemplars": [
            ("User request: Why does it rain so much in the Pacific Northwest?",
             "The appropriate tool to handle this request is: Retrieve current meteorological data, "
             "hourly forecasts, and severe weather alerts for a location.", 2),
            ("User request: Adjust the smart thermostat in the living room to 72 degrees.",
             "This request should NOT be handled by: Search the codebase knowledge graph for "
             "functions, classes, and symbols matching a natural-language query.", 0),
        ],
    },
    "sdk_rubric_grading": {
        "name": "Rubric Grading",
        "instructions": (
            "Each sample pairs a grading rubric/answer key (premise) with a student or model answer "
            "(hypothesis). Label 1 = the answer meets the rubric; label 0 = clearly fails a stated "
            "criterion; label 2 = partially meets or the rubric does not cover it. Vary: subject "
            "(essays, math proofs, code review, lab reports, history, medicine...), rubric strictness, "
            "and failure mode (missing step, wrong unit, off-topic, overconfident)."
        ),
        "exemplars": [
            ("Rubric: A correct answer must (1) name the spill-cleanup reagent and (2) state the "
             "safety precaution for fumes.",
             "Use sodium bicarbonate to neutralize the acid spill; wear goggles and work in a "
             "fume hood.", 1),
        ],
    },
    "sdk_search_reranking": {
        "name": "Search Reranking",
        "instructions": (
            "Each sample is a query (premise, 'Query: ...') and a candidate document/passage "
            "(hypothesis). Label 1 = the passage directly answers the query; label 0 = topically "
            "related but does not answer (near-miss distractors encouraged); label 2 = partially "
            "answers or answers a related but different question. Vary: query type (factual, "
            "how-to, comparison, troubleshooting), domain, and distractor subtlety."
        ),
        "exemplars": [
            ("Query: How do I rotate AWS IAM access keys without breaking running services?",
             "IAM access keys can be rotated by creating a second key, updating applications, then "
             "deactivating the old key once no service reports failures.", 1),
        ],
    },
    "sdk_cloze_reasoning": {
        "name": "Cloze Decision",
        "instructions": (
            "Each sample is a fill-in question with context (premise, ending in a blank '____') and "
            "a proposed completion (hypothesis, 'The correct answer is: ...'). Label 1 = correct "
            "completion given the context; label 0 = clearly wrong (breaks logic/facts); label 2 = "
            "defensible alternative. Vary: domain, blank position, and whether the wrong option is "
            "locally plausible but globally incoherent."
        ),
        "exemplars": [
            ("Typical advertising regulatory bodies suggest that adverts must not: encourage "
             "_______, cause unnecessary ________ or _____, and must not cause _______ offence.",
             "The correct answer is: dangerous behaviour, fear, distress, and physical or mental "
             "harm-based offence.", 1),
        ],
    },
    "sdk_rag_hallucination": {
        "name": "RAG Hallucination",
        "instructions": (
            "Each sample is a retrieved context passage (premise, 'Context: ...') and a generated "
            "answer (hypothesis). Label 1 = the answer is fully supported by the context; label 0 = "
            "contains content NOT supported (fabricated entities, numbers, dates, or claims - subtle "
            "ones encouraged); label 2 = mostly supported with minor unverifiable additions. Vary: "
            "domain, passage length, and hallucination subtlety (wrong year, swapped entity, "
            "invented citation, overgeneralization)."
        ),
        "exemplars": [
            ("Context: The Eiffel Tower, completed in 1889 for the World's Fair, stands 330 meters "
             "tall and was the tallest man-made structure until 1930.",
             "The Eiffel Tower was completed in 1889 for the World's Fair and remained the tallest "
             "man-made structure in the world until the Chrysler Building surpassed it in 1930.", 1),
        ],
    },
}


def _authoring_prompt(spec: Dict[str, Any], n: int, avoid: Sequence[str]) -> tuple[str, str]:
    system = (
        "You are an expert training-data author for NLI cross-encoders and System 1 decision "
        "routing. You write NOVEL, diverse, concise, self-contained samples. Never repeat an example you "
        "have already produced. Labels use: 0=contradiction/clearly-wrong, 1=entailment/correct, "
        "2=neutral/ambiguous. Respond ONLY with valid JSON conforming to the schema."
    )
    ex = "\n".join(f"- P: {p}\n  H: {h}\n  label: {l}" for p, h, l in spec["exemplars"])
    if not avoid:
        avoid_block = "(none yet)"
    elif len(avoid) <= 20:
        avoid_block = "\n".join(f"- {a[:140]}" for a in avoid)
    else:
        # Sample up to 20 diverse premises: 10 historical + 10 most recent
        pool = avoid[:-10]
        sampled_hist = random.sample(pool, min(10, len(pool)))
        recent = avoid[-10:]
        avoid_block = "\n".join(f"- {a[:140]}" for a in (sampled_hist + recent))
    user = (
        f"Mode: {spec['name']}\n{spec['instructions']}\n\n"
        f"Format exemplars (match this structure exactly, do NOT copy their content):\n{ex}\n\n"
        f"Previously produced premises (do not duplicate or trivially rephrase):\n{avoid_block}\n\n"
        f"Author exactly {n} new samples as a JSON object: {{\"samples\": [{{\"premise\": str, \"hypothesis\": str, "
        f"\"label\": 0|1|2}}, ...]}}. Aim for a rough label balance across {n} samples. "
        f"Each premise must be concise (1-3 sentences). Each hypothesis must be concise (1-2 sentences). "
        f"Do not write lengthy essays or filler commentary."
    )
    return system, user


def author_mode_samples(
    client: Any,
    mode: str,
    n_target: int,
    seen_keys: Optional[set] = None,
    existing_samples: Optional[List[Dict[str, Any]]] = None,
    rng: Optional[random.Random] = None,
    on_batch_accepted: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Authors novel samples for one mode via the teacher LLM.

    Batches of BATCH_SIZE; a batch is accepted only if >=MIN_NOVEL_PER_BATCH items
    are exact-novel. Raises RuntimeError if the full-run uniqueness gate fails.
    If existing_samples are provided, only authors the remaining deficit to reach n_target.
    If on_batch_accepted is provided, it is invoked with each newly accepted batch.
    Returns the newly authored rows (or empty list if already at or above n_target).
    """
    spec = MODE_AUTHOR_SPECS[mode]
    seen: set = seen_keys if seen_keys is not None else set()
    rng = rng or random.Random(0)

    prior = list(existing_samples or [])
    for r in prior:
        p, h = r.get("premise", ""), r.get("hypothesis", "")
        if p and h:
            seen.add(_key(p, h))

    if len(prior) >= n_target:
        print(f"  [{mode}] already has {len(prior)} >= {n_target} target samples; skipping authoring.", flush=True)
        return []

    needed = n_target - len(prior)
    print(f"  [{mode}] authoring {needed} new samples (existing={len(prior)}, target={n_target})...", flush=True)

    newly_accepted: List[Dict[str, Any]] = []
    premises: List[str] = [r["premise"] for r in prior if "premise" in r]
    attempts = 0
    max_attempts = (needed // MIN_NOVEL_PER_BATCH) * 6 + 25

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "authored_samples", "strict": True, "schema": TEACHER_AUTHOR_SCHEMA},
    }

    while len(newly_accepted) < needed and attempts < max_attempts:
        attempts += 1
        n = min(BATCH_SIZE, needed - len(newly_accepted) + 2)  # slight overshoot for dedup losses
        system, user = _authoring_prompt(spec, n, premises)
        try:
            raw = client.query_chat(
                system, user,
                temperature=0.35,
                max_tokens=2048,
                response_format=response_format,
            )
            if not raw:
                print(f"  [{mode}] authoring batch empty response / timeout; retrying (attempt {attempts})", flush=True)
                continue
            items = robust_parse_authored_samples(raw)
            if not items:
                print(f"  [{mode}] authoring batch unparseable (len={len(raw)}); retrying (attempt {attempts})", flush=True)
                continue
        except Exception as exc:  # noqa: BLE001 - teacher hiccups must not kill the run
            print(f"  [{mode}] authoring batch request failed ({exc}); retrying (attempt {attempts})", flush=True)
            continue

        novel_in_batch = 0
        batch_added: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            p = str(it.get("premise", "")).strip()
            h = str(it.get("hypothesis", "")).strip()
            lab = it.get("label")
            try:
                lab = int(lab)
            except (ValueError, TypeError):
                continue
            if not p or not h or lab not in (0, 1, 2) or _key(p, h) in seen:
                continue
            seen.add(_key(p, h))
            novel_in_batch += 1
            idx = len(prior) + len(newly_accepted)
            row = {
                "id": f"{mode}_{idx:06d}",
                "premise": p, "hypothesis": h, "label": lab,
                "source": mode, "language": "en",
                "metadata": {"authored_by": "teacher", "mode": spec["name"]},
            }
            newly_accepted.append(row)
            batch_added.append(row)
            premises.append(p)
            if len(newly_accepted) >= needed:
                break

        if on_batch_accepted and batch_added:
            on_batch_accepted(batch_added)

        print(
            f"  [{mode}] progress: {len(prior) + len(newly_accepted)}/{n_target} "
            f"(+{novel_in_batch} new, batch_size={len(items)}, attempt={attempts})",
            flush=True,
        )

        if novel_in_batch < MIN_NOVEL_PER_BATCH and len(items) >= BATCH_SIZE:
            print(f"  [{mode}] low-novelty batch ({novel_in_batch}/{len(items)}); refreshing avoid-list", flush=True)

    if len(newly_accepted) < needed:
        raise RuntimeError(
            f"{mode}: teacher authoring exhausted attempts with {len(prior) + len(newly_accepted)}/{n_target} novel "
            f"samples (diversity too low or teacher failing - inspect server logs)"
        )

    all_mode = prior + newly_accepted[:needed]
    unique_ratio = len({_key(r['premise'], r['hypothesis']) for r in all_mode}) / len(all_mode)
    if unique_ratio < MIN_UNIQUE_RATIO:
        raise RuntimeError(f"{mode}: uniqueness gate failed ({unique_ratio:.2f} < {MIN_UNIQUE_RATIO})")
    return newly_accepted[:needed]
