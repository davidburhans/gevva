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

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence

BATCH_SIZE = 10
MIN_NOVEL_PER_BATCH = 7  # accept a batch only if >=70% of items are new
MIN_UNIQUE_RATIO = 0.80  # full-run gate


def _key(premise: str, hypothesis: str) -> str:
    return hashlib.sha256((premise + "\x00" + hypothesis).encode("utf-8")).hexdigest()


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
        "routing. You write NOVEL, diverse, self-contained samples. Never repeat an example you "
        "have already produced. Labels use: 0=contradiction/clearly-wrong, 1=entailment/correct, "
        "2=neutral/ambiguous."
    )
    ex = "\n".join(f"- P: {p}\n  H: {h}\n  label: {l}" for p, h, l in spec["exemplars"])
    avoid_block = "\n".join(f"- {a[:140]}" for a in avoid[-8:]) if avoid else "(none yet)"
    user = (
        f"Mode: {spec['name']}\n{spec['instructions']}\n\n"
        f"Format exemplars (match this structure exactly, do NOT copy their content):\n{ex}\n\n"
        f"Previously produced premises (do not duplicate or trivially rephrase):\n{avoid_block}\n\n"
        f"Author exactly {n} new samples as JSON: [{{\"premise\": str, \"hypothesis\": str, "
        f"\"label\": 0|1|2}}, ...]. Aim for a rough label balance across {n} samples. "
        f"Each premise must be self-contained and 1-4 sentences."
    )
    return system, user


def author_mode_samples(
    client: Any,
    mode: str,
    n_target: int,
    seen_keys: Optional[set] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Authors n_target novel samples for one mode via the teacher LLM.

    Batches of BATCH_SIZE; a batch is accepted only if >=MIN_NOVEL_PER_BATCH items
    are exact-novel. Raises RuntimeError if the full-run uniqueness gate fails.
    Returns rows shaped like the template generators (id/premise/hypothesis/label/source).
    """
    spec = MODE_AUTHOR_SPECS[mode]
    seen: set = seen_keys if seen_keys is not None else set()
    rng = rng or random.Random(0)
    accepted: List[Dict[str, Any]] = []
    premises: List[str] = []
    attempts = 0
    max_attempts = (n_target // MIN_NOVEL_PER_BATCH) * 6 + 10

    while len(accepted) < n_target and attempts < max_attempts:
        attempts += 1
        n = min(BATCH_SIZE, n_target - len(accepted) + 3)  # slight overshoot for dedup losses
        system, user = _authoring_prompt(spec, n, premises)
        try:
            raw = client.query_chat(
                system, user,
                response_format={"type": "json_object"},
            )
            import json
            items = json.loads(raw)
            if isinstance(items, dict):  # tolerate {"samples": [...]} envelopes
                items = items.get("samples", items.get("data", []))
        except Exception as exc:  # noqa: BLE001 - teacher hiccups must not kill the run
            print(f"  [{mode}] authoring batch failed ({exc}); retrying")
            continue

        novel_in_batch = 0
        for it in items:
            p = str(it.get("premise", "")).strip()
            h = str(it.get("hypothesis", "")).strip()
            lab = it.get("label")
            if not p or not h or lab not in (0, 1, 2) or _key(p, h) in seen:
                continue
            seen.add(_key(p, h))
            novel_in_batch += 1
            accepted.append({
                "id": f"{mode}_{len(accepted):06d}",
                "premise": p, "hypothesis": h, "label": int(lab),
                "source": mode, "language": "en",
                "metadata": {"authored_by": "teacher", "mode": spec["name"]},
            })
            premises.append(p)
        if novel_in_batch < MIN_NOVEL_PER_BATCH and len(items) >= BATCH_SIZE:
            print(f"  [{mode}] low-novelty batch ({novel_in_batch}/{len(items)}); re-prompting with "
                  f"stronger avoid-list")

    if len(accepted) < n_target:
        raise RuntimeError(
            f"{mode}: teacher authoring exhausted attempts with {len(accepted)}/{n_target} novel "
            f"samples (diversity too low or teacher failing - inspect server logs)")
    unique_ratio = len({_key(r['premise'], r['hypothesis']) for r in accepted}) / len(accepted)
    if unique_ratio < MIN_UNIQUE_RATIO:
        raise RuntimeError(f"{mode}: uniqueness gate failed ({unique_ratio:.2f} < {MIN_UNIQUE_RATIO})")
    return accepted[:n_target]  # trim batch overshoot; seen_keys keeps the extras out of later modes
