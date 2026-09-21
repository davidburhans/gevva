"""Typed Decisions Synthetic Dataset Adapter (Hmm / System One Format).

Ingests and converts `n4ze3m/typed-decisions-synth` into calibrated NLI cross-encoder
triplets for System 1 decision engine training (tool routing, triage, rubric grading,
and boolean policy verification).

Attribution & Provenance:
- Dataset: https://huggingface.co/datasets/n4ze3m/typed-decisions-synth
- Project: "Hmm: a small open model for typed decisions" (https://github.com/n4ze3m/hmm)
- Author: Muhammed Nazeem (2026)
- Model: DeepSeek V4.1 Flash (self-consistency verified, 3-pass teacher soft labels)
- License: MIT License
- Benchmark Isolation: The 4 evaluation workflows from `LocalLLaMA/typed-decisions`
  (customer service, invoice processing, security incidents, agent traces) were
  explicitly withheld by the dataset author, preventing downstream test contamination.

Format Mapping:
1. `noul` (Boolean / Policy verification, 10,192 questions):
   - Premise: Context state document + Question instruction
   - Hypothesis: Formatted assertion claim
   - Gold True  -> ENTAILMENT (1), Soft P(ent) = teacher_p
   - Gold False -> CONTRADICTION (0), Soft P(con) = 1 - teacher_p
2. `choice` (Multi-option selection / Tool routing, 10,179 questions):
   - Premise: Context state document + Question instruction
   - Hypothesis: "Selected option: {key}. {description}"
   - Gold Option -> ENTAILMENT (1)
   - Alternative Options -> CONTRADICTION (0) (hard negatives)
   - Teacher soft probabilities preserved for Brier calibration loss
3. `score` (Rubric level grading, 5,488 questions):
   - Premise: Context state document + Rubric instructions
   - Hypothesis: "Assigned rating level {idx}: {criteria[idx]}"
   - Gold Level -> ENTAILMENT (1)
   - Non-gold Levels -> CONTRADICTION (0)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional

from nli_labels import CONTRADICTION, ENTAILMENT

logger = logging.getLogger(__name__)

DATASET_NAME = "n4ze3m/typed-decisions-synth"
DATASET_LICENSE = "MIT"
DATASET_CITATION = (
    "@misc{nazeem2026hmm,\n"
    "  author = {Muhammed Nazeem},\n"
    "  title  = {Hmm: a small open model for typed decisions},\n"
    "  year   = {2026},\n"
    "  url    = {https://github.com/n4ze3m/hmm}\n"
    "}"
)

# Hypothesis templates to prevent stylistic overfitting (prioritizes serving format P2)
SERVING_CHOICE_TEMPLATE = "The correct answer is: {key}: {desc}"
SERVING_NOUL_TEMPLATE = "The correct answer is: {val}: {desc}"
SERVING_SCORE_TEMPLATE = "The correct answer is: {level}: {desc}"

CHOICE_TEMPLATES = [
    SERVING_CHOICE_TEMPLATE,
    SERVING_CHOICE_TEMPLATE,  # Doubled to give 50% serving parity
    "Selected option: {key}. {desc}",
    "Decision: {key} ({desc})",
    "Appropriate choice: {key}. {desc}",
    "Option: {key} - {desc}",
]

NOUL_TEMPLATES = [
    SERVING_NOUL_TEMPLATE,
    SERVING_NOUL_TEMPLATE,
    "Assessment: {instruction}",
    "Verification claim: {instruction}",
    "The following condition is met: {instruction}",
    "Confirmed: {instruction}",
    "{instruction}",
]

SCORE_TEMPLATES = [
    SERVING_SCORE_TEMPLATE,
    SERVING_SCORE_TEMPLATE,
    "Assigned rating level {level}: {desc}",
    "Score level {level}: {desc}",
    "Evaluation outcome: level {level} ({desc})",
    "Rubric rating {level}: {desc}",
]


def _format_state_premise(domain: str, state: Any, state_is_json: bool) -> str:
    """Formats the state context cleanly as premise text."""
    if state_is_json and isinstance(state, str):
        try:
            parsed = json.loads(state)
            state_text = json.dumps(parsed, indent=2)
        except Exception:
            state_text = state.strip()
    elif isinstance(state, dict):
        state_text = json.dumps(state, indent=2)
    else:
        state_text = str(state).strip()

    return f"Domain: {domain}\nContext:\n{state_text}"


def convert_case_to_nli_pairs(
    case: Dict[str, Any],
    rng: Optional[random.Random] = None,
    max_negatives_per_choice: int = 2,
    include_inverted_noul: bool = True,
    grouped: bool = False,
    serving_parity: bool = True,
) -> List[Dict[str, Any]]:
    """Converts a single Typed Decisions case into NLI cross-encoder pairs.

    Args:
        case: A raw dict from `n4ze3m/typed-decisions-synth`.
        rng: Random instance for template shuffling and distractor sampling.
        max_negatives_per_choice: Max hard negative options to sample per choice question.
        include_inverted_noul: Whether to generate refutation pairs for boolean questions.
        grouped: If True, extracts all K options with question-level group_id for P1 cross-option loss.
        serving_parity: If True, uses '{instructions}\\n\\n{state}' matching the JevBench / SDK serving layout.

    Returns:
        List of standardized NLI pair dicts.
    """
    if rng is None:
        rng = random.Random(42)

    state_id = case.get("state_id", "unknown")
    domain = case.get("domain", "general")
    state = case.get("state", "")
    state_is_json = bool(case.get("state_is_json", False))

    raw_questions = case.get("questions", "{}")
    raw_gold = case.get("gold", "{}")
    raw_teacher = case.get("teacher", "{}")

    questions: Dict[str, Any] = json.loads(raw_questions) if isinstance(raw_questions, str) else raw_questions
    gold: Dict[str, Any] = json.loads(raw_gold) if isinstance(raw_gold, str) else raw_gold
    teacher: Dict[str, Any] = json.loads(raw_teacher) if isinstance(raw_teacher, str) else raw_teacher

    if state_is_json and isinstance(state, str):
        try:
            parsed = json.loads(state)
            state_text = json.dumps(parsed, indent=2)
        except Exception:
            state_text = state.strip()
    elif isinstance(state, dict):
        state_text = json.dumps(state, indent=2)
    else:
        state_text = str(state).strip()

    base_premise = f"Domain: {domain}\nContext:\n{state_text}"
    pairs: List[Dict[str, Any]] = []
    pair_counter = 0

    for q_id, q_data in questions.items():
        q_type = q_data.get("type")
        instructions = q_data.get("instructions", "").strip()
        if not instructions:
            continue

        q_gold = gold.get(q_id)
        q_teacher = teacher.get(q_id, {})
        group_id_val = f"td_{state_id}_{q_id}"

        # Premise formatting: P2 serving-template parity
        if serving_parity:
            # 80% direct serving format, 20% domain-prefixed variant
            if rng.random() < 0.8:
                q_premise = f"{instructions}\n\n{state_text}"
            else:
                q_premise = f"Domain: {domain}\nContext:\n{state_text}\nQuestion: {instructions}"
        else:
            q_premise = f"{base_premise}\nQuestion: {instructions}"

        # ---------------------------------------------------------------------
        # 1. NOUL: Boolean / Policy Verification
        # ---------------------------------------------------------------------
        if q_type == "noul":
            if q_gold is None:
                continue
            is_true = bool(q_gold)

            # Extract teacher confidence if available
            p_true = float(q_teacher.get("noul", 1.0 if is_true else 0.0))
            p_true = max(0.0, min(1.0, p_true))

            criteria = q_data.get("criteria", {})
            true_desc = criteria.get("true", "Yes") if isinstance(criteria, dict) else "Yes"
            false_desc = criteria.get("false", "No") if isinstance(criteria, dict) else "No"

            if grouped:
                # Grouped mode: produce both Yes and No options for cross-option competition
                hyp_yes = SERVING_NOUL_TEMPLATE.format(val="yes", desc=true_desc) if rng.random() < 0.6 else f"Confirmed: {instructions}"
                hyp_no = SERVING_NOUL_TEMPLATE.format(val="no", desc=false_desc) if rng.random() < 0.6 else f"Verification claim (Negative): NOT ({instructions})"

                # Option 0: Yes
                pairs.append({
                    "id": f"typed_decisions_{state_id}_{q_id}_opt_yes",
                    "premise": q_premise,
                    "hypothesis": hyp_yes,
                    "label": ENTAILMENT if is_true else CONTRADICTION,
                    "group_id": group_id_val,
                    "is_gold": is_true,
                    "soft_target": p_true,
                    "soft_labels": [round(1.0 - p_true, 4), round(p_true, 4), 0.0],
                    "source": "typed_decisions_synth_noul_grouped",
                    "language": "en",
                    "image": "",
                    "metadata": {"state_id": state_id, "question_id": q_id, "q_type": "noul", "gold": is_true, "option": "yes", "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE}},
                })

                # Option 1: No
                pairs.append({
                    "id": f"typed_decisions_{state_id}_{q_id}_opt_no",
                    "premise": q_premise,
                    "hypothesis": hyp_no,
                    "label": CONTRADICTION if is_true else ENTAILMENT,
                    "group_id": group_id_val,
                    "is_gold": not is_true,
                    "soft_target": 1.0 - p_true,
                    "soft_labels": [round(p_true, 4), round(1.0 - p_true, 4), 0.0],
                    "source": "typed_decisions_synth_noul_grouped",
                    "language": "en",
                    "image": "",
                    "metadata": {"state_id": state_id, "question_id": q_id, "q_type": "noul", "gold": is_true, "option": "no", "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE}},
                })
            else:
                # Pointwise mode (backward-compatible)
                tpl = rng.choice(NOUL_TEMPLATES)
                hyp = tpl.format(instruction=instructions, val="yes" if is_true else "no", desc=true_desc if is_true else false_desc)
                label = ENTAILMENT if is_true else CONTRADICTION
                soft = [round(1.0 - p_true, 4), round(p_true, 4), 0.0]

                pairs.append({
                    "id": f"typed_decisions_{state_id}_{q_id}_pos_{pair_counter:02d}",
                    # WHY: q_premise unconditionally — the old
                    # `q_premise if serving_parity else base_premise` dropped the
                    # question instructions from the premise when serving_parity
                    # was False, while the hypothesis still referenced them.
                    "premise": q_premise,
                    "hypothesis": hyp,
                    "label": label,
                    "group_id": group_id_val,
                    "is_gold": True,
                    "soft_target": p_true if is_true else (1.0 - p_true),
                    "soft_labels": soft,
                    "source": "typed_decisions_synth_noul",
                    "language": "en",
                    "image": "",
                    "metadata": {
                        "state_id": state_id,
                        "domain": domain,
                        "question_id": q_id,
                        "q_type": "noul",
                        "gold": is_true,
                        "teacher_p": p_true,
                        "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                    },
                })
                pair_counter += 1

                if include_inverted_noul and rng.random() < 0.5:
                    inv_hyp = f"Verification claim (Negative): NOT ({instructions})"
                    inv_label = CONTRADICTION if is_true else ENTAILMENT
                    inv_soft = [soft[1], soft[0], 0.0]
                    pairs.append({
                        "id": f"typed_decisions_{state_id}_{q_id}_inv_{pair_counter:02d}",
                        # Same parity defect as the positive noul pair above.
                        "premise": q_premise,
                        "hypothesis": inv_hyp,
                        "label": inv_label,
                        "group_id": f"{group_id_val}_inv",
                        "is_gold": False,
                        "soft_target": 1.0 - p_true if is_true else p_true,
                        "soft_labels": inv_soft,
                        "source": "typed_decisions_synth_noul_inverted",
                        "language": "en",
                        "image": "",
                        "metadata": {
                            "state_id": state_id,
                            "domain": domain,
                            "question_id": q_id,
                            "q_type": "noul_inverted",
                            "gold": not is_true,
                            "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                        },
                    })
                    pair_counter += 1

        # ---------------------------------------------------------------------
        # 2. CHOICE: Multi-Option Selection & Routing
        # ---------------------------------------------------------------------
        elif q_type == "choice":
            criteria = q_data.get("criteria", {})
            if not isinstance(criteria, dict) or not criteria or q_gold is None:
                continue

            teacher_probs = q_teacher.get("probabilities", {}) if isinstance(q_teacher, dict) else {}
            gold_str = str(q_gold)

            if grouped:
                # Grouped mode: extract ALL options in criteria for full cross-option competition
                for opt_key, opt_desc in criteria.items():
                    opt_str = str(opt_key)
                    is_gold_opt = (opt_str == gold_str)
                    tpl = rng.choice(CHOICE_TEMPLATES)
                    hyp = tpl.format(key=opt_str, desc=opt_desc)

                    p_opt = float(teacher_probs.get(opt_str, 0.95 if is_gold_opt else 0.05 / max(1, len(criteria) - 1)))
                    p_opt = max(0.0, min(1.0, p_opt))
                    soft_opt = [round(max(0.0, 1.0 - p_opt), 4), round(p_opt, 4), 0.0]

                    pairs.append({
                        "id": f"typed_decisions_{state_id}_{q_id}_opt_{opt_str}",
                        "premise": q_premise,
                        "hypothesis": hyp,
                        "label": ENTAILMENT if is_gold_opt else CONTRADICTION,
                        "group_id": group_id_val,
                        "is_gold": is_gold_opt,
                        "soft_target": p_opt,
                        "soft_labels": soft_opt,
                        "source": "typed_decisions_synth_choice_grouped",
                        "language": "en",
                        "image": "",
                        "metadata": {
                            "state_id": state_id,
                            "domain": domain,
                            "question_id": q_id,
                            "q_type": "choice",
                            "option": opt_str,
                            "is_gold": is_gold_opt,
                            "teacher_prob": p_opt,
                            "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                        },
                    })
            else:
                # Pointwise mode: 1 gold + max_negatives_per_choice
                gold_desc = criteria.get(gold_str, criteria.get(q_gold, ""))
                tpl = rng.choice(CHOICE_TEMPLATES)
                hyp_gold = tpl.format(key=gold_str, desc=gold_desc)
                p_gold = float(teacher_probs.get(gold_str, 0.95))
                soft_gold = [round(max(0.0, 1.0 - p_gold), 4), round(p_gold, 4), 0.0]

                pairs.append({
                    "id": f"typed_decisions_{state_id}_{q_id}_gold_{pair_counter:02d}",
                    "premise": q_premise,
                    "hypothesis": hyp_gold,
                    "label": ENTAILMENT,
                    "group_id": group_id_val,
                    "is_gold": True,
                    "soft_target": p_gold,
                    "soft_labels": soft_gold,
                    "source": "typed_decisions_synth_choice_gold",
                    "language": "en",
                    "image": "",
                    "metadata": {
                        "state_id": state_id,
                        "domain": domain,
                        "question_id": q_id,
                        "q_type": "choice",
                        "option": gold_str,
                        "is_gold": True,
                        "teacher_prob": p_gold,
                        "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                    },
                })
                pair_counter += 1

                neg_keys = [k for k in criteria.keys() if str(k) != gold_str]
                if len(neg_keys) > max_negatives_per_choice:
                    neg_keys.sort(key=lambda k: float(teacher_probs.get(str(k), 0.0)), reverse=True)
                    sampled_negs = neg_keys[:max_negatives_per_choice]
                else:
                    sampled_negs = neg_keys

                for neg_key in sampled_negs:
                    neg_desc = criteria[neg_key]
                    tpl_neg = rng.choice(CHOICE_TEMPLATES)
                    hyp_neg = tpl_neg.format(key=neg_key, desc=neg_desc)
                    p_neg = float(teacher_probs.get(str(neg_key), 0.05))
                    soft_neg = [round(max(0.0, 1.0 - p_neg), 4), round(p_neg, 4), 0.0]

                    pairs.append({
                        "id": f"typed_decisions_{state_id}_{q_id}_neg_{pair_counter:02d}",
                        "premise": q_premise,
                        "hypothesis": hyp_neg,
                        "label": CONTRADICTION,
                        "group_id": group_id_val,
                        "is_gold": False,
                        "soft_target": p_neg,
                        "soft_labels": soft_neg,
                        "source": "typed_decisions_synth_choice_neg",
                        "language": "en",
                        "image": "",
                        "metadata": {
                            "state_id": state_id,
                            "domain": domain,
                            "question_id": q_id,
                            "q_type": "choice",
                            "option": str(neg_key),
                            "is_gold": False,
                            "teacher_prob": p_neg,
                            "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                        },
                    })
                    pair_counter += 1

        # ---------------------------------------------------------------------
        # 3. SCORE: Rubric Level Grading
        # ---------------------------------------------------------------------
        elif q_type == "score":
            criteria = q_data.get("criteria", [])
            if not isinstance(criteria, list) or not criteria or q_gold is None:
                continue

            teacher_probs = q_teacher.get("probabilities", {}) if isinstance(q_teacher, dict) else {}
            try:
                gold_idx = int(q_gold)
            except (ValueError, TypeError):
                continue

            if 0 <= gold_idx < len(criteria):
                if grouped:
                    # Grouped mode: extract all levels
                    for lvl_idx, lvl_desc in enumerate(criteria):
                        is_gold_lvl = (lvl_idx == gold_idx)
                        tpl = rng.choice(SCORE_TEMPLATES)
                        hyp_lvl = tpl.format(level=lvl_idx, desc=lvl_desc)
                        p_lvl = float(teacher_probs.get(str(lvl_idx), 0.90 if is_gold_lvl else 0.10 / max(1, len(criteria) - 1)))
                        soft_lvl = [round(max(0.0, 1.0 - p_lvl), 4), round(p_lvl, 4), 0.0]

                        pairs.append({
                            "id": f"typed_decisions_{state_id}_{q_id}_lvl_{lvl_idx}",
                            "premise": q_premise,
                            "hypothesis": hyp_lvl,
                            "label": ENTAILMENT if is_gold_lvl else CONTRADICTION,
                            "group_id": group_id_val,
                            "is_gold": is_gold_lvl,
                            "soft_target": p_lvl,
                            "soft_labels": soft_lvl,
                            "source": "typed_decisions_synth_score_grouped",
                            "language": "en",
                            "image": "",
                            "metadata": {
                                "state_id": state_id,
                                "domain": domain,
                                "question_id": q_id,
                                "q_type": "score",
                                "level": lvl_idx,
                                "is_gold": is_gold_lvl,
                                "teacher_prob": p_lvl,
                                "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                            },
                        })
                else:
                    gold_desc = criteria[gold_idx]
                    tpl = rng.choice(SCORE_TEMPLATES)
                    hyp_gold = tpl.format(level=gold_idx, desc=gold_desc)
                    p_gold = float(teacher_probs.get(str(gold_idx), 0.90))
                    soft_gold = [round(max(0.0, 1.0 - p_gold), 4), round(p_gold, 4), 0.0]

                    pairs.append({
                        "id": f"typed_decisions_{state_id}_{q_id}_gold_{pair_counter:02d}",
                        "premise": q_premise,
                        "hypothesis": hyp_gold,
                        "label": ENTAILMENT,
                        "group_id": group_id_val,
                        "is_gold": True,
                        "soft_target": p_gold,
                        "soft_labels": soft_gold,
                        "source": "typed_decisions_synth_score_gold",
                        "language": "en",
                        "image": "",
                        "metadata": {
                            "state_id": state_id,
                            "domain": domain,
                            "question_id": q_id,
                            "q_type": "score",
                            "level": gold_idx,
                            "is_gold": True,
                            "teacher_prob": p_gold,
                            "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                        },
                    })
                    pair_counter += 1

                    alt_indices = [i for i in range(len(criteria)) if i != gold_idx]
                    if alt_indices:
                        adj_candidates = [i for i in alt_indices if abs(i - gold_idx) == 1]
                        alt_idx = rng.choice(adj_candidates) if adj_candidates else rng.choice(alt_indices)
                        alt_desc = criteria[alt_idx]
                        tpl_alt = rng.choice(SCORE_TEMPLATES)
                        hyp_alt = tpl_alt.format(level=alt_idx, desc=alt_desc)
                        p_alt = float(teacher_probs.get(str(alt_idx), 0.05))
                        soft_alt = [round(max(0.0, 1.0 - p_alt), 4), round(p_alt, 4), 0.0]

                        pairs.append({
                            "id": f"typed_decisions_{state_id}_{q_id}_alt_{pair_counter:02d}",
                            "premise": q_premise,
                            "hypothesis": hyp_alt,
                            "label": CONTRADICTION,
                            "group_id": group_id_val,
                            "is_gold": False,
                            "soft_target": p_alt,
                            "soft_labels": soft_alt,
                            "source": "typed_decisions_synth_score_alt",
                            "language": "en",
                            "image": "",
                            "metadata": {
                                "state_id": state_id,
                                "domain": domain,
                                "question_id": q_id,
                                "q_type": "score",
                                "level": alt_idx,
                                "is_gold": False,
                                "teacher_prob": p_alt,
                                "attribution": {"dataset": DATASET_NAME, "license": DATASET_LICENSE},
                            },
                        })
                        pair_counter += 1

    return pairs


def convert_typed_decisions_to_nli(
    split: str = "train",
    max_cases: Optional[int] = None,
    seed: int = 42,
    max_negatives_per_choice: int = 2,
    grouped: bool = False,
) -> List[Dict[str, Any]]:
    """Loads `n4ze3m/typed-decisions-synth` from Hugging Face and converts to NLI pairs.

    Args:
        split: 'train' or 'validation' (matches Hugging Face dataset split).
        max_cases: Optional limit on the number of raw cases to process.
        seed: Random seed for deterministic generation.
        max_negatives_per_choice: Max distractors per choice question.
        grouped: If True, extract all K candidate options per question with group_id.

    Returns:
        List of NLI pair dictionaries ready for training or evaluation.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install datasets (`pip install datasets` or `uv add datasets`)") from e

    logger.info(f"Loading '{DATASET_NAME}' split '{split}'...")
    ds = load_dataset(DATASET_NAME, split=split)

    rng = random.Random(seed)
    all_pairs: List[Dict[str, Any]] = []

    for i, case in enumerate(ds):
        if max_cases is not None and i >= max_cases:
            break
        pairs = convert_case_to_nli_pairs(
            case,
            rng=rng,
            max_negatives_per_choice=max_negatives_per_choice,
            grouped=grouped,
        )
        all_pairs.extend(pairs)

    logger.info(f"Converted {min(len(ds), max_cases or len(ds))} cases into {len(all_pairs)} NLI pairs.")
    return all_pairs
