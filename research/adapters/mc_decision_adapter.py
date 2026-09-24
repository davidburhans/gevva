#!/usr/bin/env python3
"""research/adapters/mc_decision_adapter.py - Multiple-Choice Q&A Adapter.

Converts high-leverage multiple-choice Q&A datasets from Hugging Face into
grouped NLI cross-encoder training pairs formatted for served-distribution
cross-option training (P1/P2) and Phase-2 weak-family remediation.

Supported Datasets:
1. CaseHOLD (coastalcph/lex_glue: case_hold, 45K items, 5 options)
   -> Targets tradeoff / precedence / "which rule applies" (0/6 in baseline).
2. RACE (ehovy/race: all, ~88K items, 4 options)
   -> Targets long multi-paragraph evidence-weighing (Mode C, long_policy).
3. ReClor (tasksource/reclor, 4.6K items, 4 options)
   -> Targets multi-hop logical deduction & reasoning chains.
4. LogiQA 2.0 (jeggers/logiqa2_formatted, 12.6K items, 4 options)
   -> Targets categorical, conditional, and multi-constraint reasoning.
5. PubMedQA (qiaojin/PubMedQA: pqa_artificial, 211K items, yes/no/maybe)
   -> Targets calibrated evidence-weighing and provides authentic NEUTRAL supervision.
6. CommonsenseQA (tau/commonsense_qa, 9.7K items, 5 options)
   -> Targets K=5 group shape matching typed-decisions.

Frozen Serving Parity Format:
- Premise: {Context / Article} \\n\\n Question: {Question}
- Hypothesis: "The correct answer is: {option_text}"
- Group ID: Question-level grouping for atomic batching & cross-option softmax.
- Label: ENTAILMENT (1) for winning choice, CONTRADICTION (0) for distractors.
- Native 3-class mode supported for PubMedQA (yes->1, no->0, maybe->2).

Protocol §8 Decontamination:
- Integrated 8-gram filter against JevBench public items and held-out test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

from nli_labels import CONTRADICTION, ENTAILMENT, NEUTRAL, ID2LABEL

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "staged" / "mc_qa"

# Frozen serving parity template (matches JevBench / SDK / typed_decisions)
SERVING_CHOICE_TEMPLATE = "The correct answer is: {option}"

# Dataset Registry & Licensing Metadata
DATASET_REGISTRY = {
    "casehold": {
        "hf_path": "coastalcph/lex_glue",
        "config": "case_hold",
        "license": "CC BY-NC-SA 4.0 / Research Use",
        "description": "5-choice legal holding selection from US judicial decisions",
        "weakness_target": "tradeoff_precedence",
    },
    "race": {
        "hf_path": "ehovy/race",
        "config": "all",
        "license": "Non-commercial / Educational / Research Use",
        "description": "Long multi-paragraph passage + 4-option reading comprehension",
        "weakness_target": "long_policy_evidence_weighing",
    },
    "reclor": {
        "hf_path": "tasksource/reclor",
        "config": None,
        "license": "Research Use",
        "description": "LSAT-derived logical reasoning multiple-choice questions",
        "weakness_target": "multi_hop_logic",
    },
    "logiqa": {
        "hf_path": "jeggers/logiqa2_formatted",
        "config": None,
        "license": "Research Use",
        "description": "LogiQA 2.0 multi-constraint deductive & conditional reasoning",
        "weakness_target": "multi_hop_constraints",
    },
    "pubmedqa": {
        "hf_path": "qiaojin/PubMedQA",
        "config": "pqa_artificial",
        "license": "MIT License",
        "description": "PubMed biomedical abstract + question with yes/no/maybe decisions",
        "weakness_target": "evidence_weighing_neutral",
    },
    "csqa": {
        "hf_path": "tau/commonsense_qa",
        "config": None,
        "license": "CC BY 4.0",
        "description": "5-choice commonsense knowledge reasoning",
        "weakness_target": "typed_decisions_k5_shape",
    },
    "lsat_ar": {
        "hf_path": "tasksource/lsat-ar",
        "config": None,
        "license": "Research Use",
        "description": "LSAT Analytical Reasoning (logic games / formal constraint satisfaction)",
        "weakness_target": "multi_hop_constraint_satisfaction",
    },
    "strategy_qa": {
        "hf_path": "tasksource/strategy-qa",
        "config": None,
        "license": "MIT License",
        "description": "StrategyQA multi-step implicit reasoning over interconnected facts",
        "weakness_target": "multi_hop_implicit_reasoning",
    },
    "aqua_rat": {
        "hf_path": "deepmind/aqua_rat",
        "config": "raw",
        "license": "Apache 2.0",
        "description": "Algebra & numerical word problems with multi-step calculations",
        "weakness_target": "temporal_numeric_calculations",
    },
}

NGRAM_N = 8


def normalize_text(text: str) -> str:
    """Normalizes whitespace and casing for n-gram checks."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def extract_ngrams(text: str, n: int = NGRAM_N) -> Set[str]:
    """Extracts character or word n-grams from text."""
    words = normalize_text(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_decontamination_reference(
    include_jevbench: bool = True,
    include_test_jsonl: bool = True,
) -> Set[str]:
    """Loads reference 8-grams from JevBench public and internal test split."""
    ref_ngrams: Set[str] = set()

    # 1. JevBench public benchmarks
    if include_jevbench:
        jevbench_dir = REPO_ROOT.parent / "jevbench" / "datasets" / "public"
        if jevbench_dir.exists():
            for fname in ("easy.jsonl", "original.jsonl", "hard.jsonl"):
                p = jevbench_dir / fname
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                                state_str = json.dumps(d.get("state", ""))
                                instr_str = str(d.get("question", {}).get("instructions", ""))
                                ref_ngrams |= extract_ngrams(f"{state_str} {instr_str}")
                            except Exception:
                                pass

    # 2. Internal held-out test split (data/test.jsonl)
    if include_test_jsonl:
        test_path = REPO_ROOT / "data" / "test.jsonl"
        if test_path.exists():
            with open(test_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        ref_ngrams |= extract_ngrams(f"{d.get('premise', '')} {d.get('hypothesis', '')}")
                    except Exception:
                        pass

    return ref_ngrams


# =============================================================================
# Dataset Converters
# =============================================================================


def convert_casehold_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a single CaseHOLD record into grouped NLI pairs.

    Schema:
      context: string (legal text with <HOLDING> marker)
      endings: list of 5 string candidate holdings
      label: int (0 to 4), index of true holding
    """
    context = record.get("context", "").strip()
    endings = record.get("endings", [])
    label = record.get("label")

    if not context or not endings or label is None or not (0 <= label < len(endings)):
        return []

    group_id = f"casehold_{split}_{index}"
    premise = f"Select the legal holding that correctly completes the citation in the context:\n\n{context}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, opt_text in enumerate(endings):
        is_gold = (opt_idx == label)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"casehold_{split}_{index}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "casehold_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "coastalcph/lex_glue/case_hold",
                "license": DATASET_REGISTRY["casehold"]["license"],
                "split": split,
                "original_index": index,
                "option_idx": opt_idx,
                "num_options": len(endings),
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["casehold"]["weakness_target"],
            },
        })
    return pairs


def convert_race_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a single RACE record into grouped NLI pairs.

    Schema:
      example_id: string (e.g. 'high19088.txt')
      article: long passage text
      question: question text
      options: list of 4 choices
      answer: string 'A', 'B', 'C', or 'D'
    """
    article = record.get("article", "").strip()
    question = record.get("question", "").strip()
    options = record.get("options", [])
    ans_str = str(record.get("answer", "")).strip().upper()
    example_id = record.get("example_id", f"ex_{index}")

    if not article or not question or not options:
        return []

    # Map 'A', 'B', 'C', 'D' to index
    if ans_str in ("A", "B", "C", "D"):
        gold_idx = ord(ans_str) - ord("A")
    else:
        try:
            gold_idx = int(ans_str)
        except (ValueError, TypeError):
            return []

    if not (0 <= gold_idx < len(options)):
        return []

    group_id = f"race_{split}_{example_id}"
    premise = f"{article}\n\nQuestion: {question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, opt_text in enumerate(options):
        is_gold = (opt_idx == gold_idx)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"race_{split}_{example_id}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "race_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "ehovy/race",
                "license": DATASET_REGISTRY["race"]["license"],
                "split": split,
                "example_id": example_id,
                "option_idx": opt_idx,
                "num_options": len(options),
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["race"]["weakness_target"],
            },
        })
    return pairs


def convert_reclor_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a single ReClor record into grouped NLI pairs.

    Schema:
      context: logic scenario / argument
      question: question text
      answers: list of 4 choices
      label: int (0 to 3)
      id_string: string id
    """
    context = record.get("context", "").strip()
    question = record.get("question", "").strip()
    answers = record.get("answers", [])
    label = record.get("label")
    id_string = record.get("id_string", f"reclor_{split}_{index}")

    if not context or not question or not answers or label is None or not (0 <= label < len(answers)):
        return []

    group_id = f"reclor_{split}_{id_string}"
    premise = f"{context}\n\nQuestion: {question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, opt_text in enumerate(answers):
        is_gold = (opt_idx == label)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"reclor_{split}_{id_string}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "reclor_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "tasksource/reclor",
                "license": DATASET_REGISTRY["reclor"]["license"],
                "split": split,
                "id_string": id_string,
                "option_idx": opt_idx,
                "num_options": len(answers),
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["reclor"]["weakness_target"],
            },
        })
    return pairs


def convert_logiqa_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a single LogiQA 2.0 record into grouped NLI pairs.

    Schema:
      id: int or str
      text: context passage
      question: question text
      options: list of 4 choices
      answer: int (0 to 3)
    """
    context = record.get("text", "").strip()
    question = record.get("question", "").strip()
    options = record.get("options", [])
    answer = record.get("answer")
    logi_id = record.get("id", index)

    if not context or not question or not options or answer is None or not (0 <= answer < len(options)):
        return []

    group_id = f"logiqa_{split}_{logi_id}"
    premise = f"{context}\n\nQuestion: {question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, opt_text in enumerate(options):
        is_gold = (opt_idx == answer)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"logiqa_{split}_{logi_id}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "logiqa_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "jeggers/logiqa2_formatted",
                "license": DATASET_REGISTRY["logiqa"]["license"],
                "split": split,
                "logi_id": logi_id,
                "option_idx": opt_idx,
                "num_options": len(options),
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["logiqa"]["weakness_target"],
            },
        })
    return pairs


def convert_pubmedqa_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    native_3class: bool = False,
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a PubMedQA record into grouped NLI pairs or native 3-class NLI.

    Schema:
      pubid: int
      question: str
      context: dict with 'contexts': list of str
      final_decision: 'yes', 'no', 'maybe'
      long_answer: str
    """
    pubid = record.get("pubid", index)
    question = record.get("question", "").strip()
    raw_ctx = record.get("context", {})
    if isinstance(raw_ctx, dict):
        ctx_list = raw_ctx.get("contexts", [])
        context_str = "\n".join(str(c).strip() for c in ctx_list if c)
    elif isinstance(raw_ctx, list):
        context_str = "\n".join(str(c).strip() for c in raw_ctx if c)
    else:
        context_str = str(raw_ctx).strip()

    decision = str(record.get("final_decision", "")).strip().lower()
    if not question or not context_str or decision not in ("yes", "no", "maybe"):
        return []

    premise = f"Biomedical Abstract:\n{context_str}\n\nQuestion: {question}"

    # Native 3-class NLI Mode (Direct single triplet mapping)
    if native_3class:
        if decision == "yes":
            nli_label = ENTAILMENT
            soft = [0.0, 1.0, 0.0]
        elif decision == "no":
            nli_label = CONTRADICTION
            soft = [1.0, 0.0, 0.0]
        else:  # maybe -> NEUTRAL
            nli_label = NEUTRAL
            soft = [0.0, 0.0, 1.0]

        return [{
            "id": f"pubmedqa_nli_{split}_{pubid}",
            "premise": premise,
            "hypothesis": f"Based on the biomedical evidence, the answer is: {decision}.",
            "label": nli_label,
            "group_id": -1,  # Ungrouped NLI anchor
            "is_gold": (nli_label == ENTAILMENT),
            "soft_target": 1.0 if nli_label == ENTAILMENT else 0.0,
            "soft_labels": soft,
            "source": "pubmedqa_nli_3class",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "qiaojin/PubMedQA",
                "license": DATASET_REGISTRY["pubmedqa"]["license"],
                "pubid": pubid,
                "decision": decision,
                "weakness_target": "neutral_calibration",
            },
        }]

    # Grouped Mode: 3 candidate options ('yes', 'no', 'maybe')
    options = ["yes", "no", "maybe"]
    group_id = f"pubmedqa_{split}_{pubid}"
    pairs: List[Dict[str, Any]] = []

    for opt_idx, opt_text in enumerate(options):
        is_gold = (opt_text == decision)
        hyp = hypothesis_template.format(option=opt_text)
        pairs.append({
            "id": f"pubmedqa_{split}_{pubid}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "pubmedqa_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "qiaojin/PubMedQA",
                "license": DATASET_REGISTRY["pubmedqa"]["license"],
                "split": split,
                "pubid": pubid,
                "option": opt_text,
                "decision": decision,
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["pubmedqa"]["weakness_target"],
            },
        })
    return pairs


def convert_csqa_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a CommonsenseQA record into grouped NLI pairs.

    Schema:
      id: str
      question: str
      question_concept: str
      choices: dict with 'label': ['A', ...], 'text': ['...', ...]
      answerKey: 'A', 'B', 'C', 'D', or 'E'
    """
    csqa_id = record.get("id", f"csqa_{index}")
    question = record.get("question", "").strip()
    concept = record.get("question_concept", "").strip()
    choices = record.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    answer_key = str(record.get("answerKey", "")).strip().upper()

    if not question or not labels or not texts or not answer_key or len(labels) != len(texts):
        return []

    try:
        gold_idx = labels.index(answer_key)
    except ValueError:
        return []

    group_id = f"csqa_{split}_{csqa_id}"
    concept_prefix = f"Concept: {concept}\n" if concept else ""
    premise = f"{concept_prefix}Question: {question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, (lbl, opt_text) in enumerate(zip(labels, texts)):
        is_gold = (opt_idx == gold_idx)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"csqa_{split}_{csqa_id}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "csqa_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "tau/commonsense_qa",
                "license": DATASET_REGISTRY["csqa"]["license"],
                "split": split,
                "csqa_id": csqa_id,
                "option_idx": opt_idx,
                "option_label": lbl,
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["csqa"]["weakness_target"],
            },
        })
    return pairs


def convert_lsat_ar_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts an LSAT-AR record into grouped NLI pairs.

    Schema:
      context: string (scenario rules & constraints)
      question: question text
      answers: list of 5 choices
      label: int (0 to 4)
      id_string: str id
    """
    context = record.get("context", "").strip()
    question = record.get("question", "").strip()
    answers = record.get("answers", [])
    label = record.get("label")
    id_string = record.get("id_string", f"lsat_ar_{index}")

    if not context or not question or not answers or label is None or not (0 <= label < len(answers)):
        return []

    group_id = f"lsat_ar_{split}_{id_string}"
    premise = f"Analytical Reasoning Puzzle:\n{context}\n\nQuestion: {question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, opt_text in enumerate(answers):
        is_gold = (opt_idx == label)
        hyp = hypothesis_template.format(option=opt_text.strip())
        pairs.append({
            "id": f"lsat_ar_{split}_{id_string}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "lsat_ar_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "tasksource/lsat-ar",
                "license": DATASET_REGISTRY["lsat_ar"]["license"],
                "split": split,
                "id_string": id_string,
                "option_idx": opt_idx,
                "num_options": len(answers),
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["lsat_ar"]["weakness_target"],
            },
        })
    return pairs


def convert_strategy_qa_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts a StrategyQA record into grouped NLI pairs.

    Schema:
      qid: str
      question: str
      facts: list of fact strings
      answer: bool (True / False)
    """
    qid = record.get("qid", f"strat_{index}")
    question = record.get("question", "").strip()
    facts = record.get("facts", [])
    answer = record.get("answer")

    if not question or answer is None:
        return []

    facts_str = "\n".join(f"- {f.strip()}" for f in facts if f.strip())
    premise = f"Contextual Facts:\n{facts_str}\n\nQuestion: {question}" if facts_str else f"Question: {question}"
    group_id = f"strategy_qa_{split}_{qid}"

    # Grouped mode with two competing options: 'Yes' and 'No'
    options = ["Yes", "No"]
    is_true = bool(answer)
    pairs: List[Dict[str, Any]] = []

    for opt_idx, opt_text in enumerate(options):
        is_gold = (opt_text == "Yes" if is_true else opt_text == "No")
        hyp = hypothesis_template.format(option=opt_text)
        pairs.append({
            "id": f"strategy_qa_{split}_{qid}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "strategy_qa_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "tasksource/strategy-qa",
                "license": DATASET_REGISTRY["strategy_qa"]["license"],
                "split": split,
                "qid": qid,
                "option": opt_text,
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["strategy_qa"]["weakness_target"],
            },
        })
    return pairs


def convert_aqua_rat_record(
    record: Dict[str, Any],
    index: int,
    split: str = "train",
    hypothesis_template: str = SERVING_CHOICE_TEMPLATE,
) -> List[Dict[str, Any]]:
    """Converts an AQuA-RAT algebra record into grouped NLI pairs.

    Schema:
      question: str
      options: list of 5 options ('A)...', 'B)...', ...)
      correct: str ('A' to 'E')
    """
    question = record.get("question", "").strip()
    options = record.get("options", [])
    correct = str(record.get("correct", "")).strip().upper()

    if not question or not options or not correct:
        return []

    if correct in ("A", "B", "C", "D", "E"):
        gold_idx = ord(correct) - ord("A")
    else:
        return []

    if not (0 <= gold_idx < len(options)):
        return []

    group_id = f"aqua_rat_{split}_{index}"
    premise = f"Solve the following mathematical/numerical problem:\n\n{question}"

    pairs: List[Dict[str, Any]] = []
    for opt_idx, raw_opt in enumerate(options):
        clean_opt = re.sub(r"^[A-E][\):]\s*", "", str(raw_opt).strip())
        is_gold = (opt_idx == gold_idx)
        hyp = hypothesis_template.format(option=clean_opt)
        pairs.append({
            "id": f"aqua_rat_{split}_{index}_opt_{opt_idx}",
            "premise": premise,
            "hypothesis": hyp,
            "label": ENTAILMENT if is_gold else CONTRADICTION,
            "group_id": group_id,
            "is_gold": is_gold,
            "soft_target": 1.0 if is_gold else 0.0,
            "soft_labels": [0.0, 1.0, 0.0] if is_gold else [1.0, 0.0, 0.0],
            "source": "aqua_rat_mc_grouped",
            "language": "en",
            "image": "",
            "metadata": {
                "dataset": "deepmind/aqua_rat",
                "license": DATASET_REGISTRY["aqua_rat"]["license"],
                "split": split,
                "index": index,
                "option_idx": opt_idx,
                "raw_option": str(raw_opt),
                "clean_option": clean_opt,
                "is_gold": is_gold,
                "weakness_target": DATASET_REGISTRY["aqua_rat"]["weakness_target"],
            },
        })
    return pairs


# =============================================================================
# Ingestion Runner & Pipeline
# =============================================================================


RECORD_CONVERTERS = {
    "casehold": convert_casehold_record,
    "race": convert_race_record,
    "reclor": convert_reclor_record,
    "logiqa": convert_logiqa_record,
    "pubmedqa": convert_pubmedqa_record,
    "csqa": convert_csqa_record,
    "lsat_ar": convert_lsat_ar_record,
    "strategy_qa": convert_strategy_qa_record,
    "aqua_rat": convert_aqua_rat_record,
}


def ingest_dataset(
    dataset_name: str,
    split: str = "train",
    max_cases: Optional[int] = None,
    decontaminate: bool = True,
    ref_ngrams: Optional[Set[str]] = None,
    seed: int = 42,
    native_3class_pubmedqa: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads a dataset from Hugging Face, converts it to grouped NLI pairs,

    and applies 8-gram decontamination filtering.

    Returns:
        (clean_pairs, stats_dict)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")

    import datasets

    meta = DATASET_REGISTRY[dataset_name]
    logger.info(f"Loading '{dataset_name}' from {meta['hf_path']} (config={meta['config']}, split={split})...")

    kwargs = {}
    if meta["config"]:
        ds = datasets.load_dataset(meta["hf_path"], meta["config"], split=split, **kwargs)
    else:
        ds = datasets.load_dataset(meta["hf_path"], split=split, **kwargs)

    total_cases = len(ds)
    logger.info(f"Loaded {total_cases:,} cases from {dataset_name}:{split}")

    rng = random.Random(seed)
    case_indices = list(range(total_cases))
    if max_cases and max_cases < total_cases:
        rng.shuffle(case_indices)
        case_indices = case_indices[:max_cases]
        case_indices.sort()

    converter = RECORD_CONVERTERS[dataset_name]
    if decontaminate and ref_ngrams is None:
        logger.info("Loading reference 8-grams from JevBench public & test split...")
        ref_ngrams = load_decontamination_reference()
        logger.info(f"Decontamination reference contains {len(ref_ngrams):,} 8-grams.")

    converted_pairs: List[Dict[str, Any]] = []
    dropped_cases = 0
    decontaminated_cases = 0

    for idx in case_indices:
        record = ds[idx]
        if dataset_name == "pubmedqa":
            pairs = converter(record, idx, split=split, native_3class=native_3class_pubmedqa)
        else:
            pairs = converter(record, idx, split=split)

        if not pairs:
            dropped_cases += 1
            continue

        # Atomic group decontamination check: if ANY pair in the group leaks an 8-gram,
        # drop the entire group to preserve group atomicity and zero contamination.
        if decontaminate and ref_ngrams:
            leaks = False
            for p in pairs:
                p_text = f"{p['premise']} {p['hypothesis']}"
                if extract_ngrams(p_text) & ref_ngrams:
                    leaks = True
                    break
            if leaks:
                decontaminated_cases += 1
                continue

        converted_pairs.extend(pairs)

    stats = {
        "dataset": dataset_name,
        "hf_path": meta["hf_path"],
        "license": meta["license"],
        "split": split,
        "total_source_cases": total_cases,
        "processed_cases": len(case_indices),
        "dropped_invalid_cases": dropped_cases,
        "dropped_decontaminated_cases": decontaminated_cases,
        "emitted_pairs": len(converted_pairs),
        "unique_groups": len(set(p["group_id"] for p in converted_pairs if p.get("group_id") != -1)),
    }

    return converted_pairs, stats


def main():
    parser = argparse.ArgumentParser(description="Ingest MC Q&A Datasets into Grouped NLI Pairs")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="casehold",
        help="Dataset identifier to process (or 'all' for complete suite)",
    )
    parser.add_argument("--split", default="train", help="Dataset split (train, validation, test)")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to ingest per dataset (None = all)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for staged JSONL files")
    parser.add_argument("--no-decontaminate", action="store_true", help="Disable 8-gram decontamination check")
    parser.add_argument("--native-3class-pubmedqa", action="store_true", help="Emit PubMedQA as native 3-class NLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]

    ref_ngrams = None
    if not args.no_decontaminate:
        logger.info("Pre-loading decontamination 8-grams...")
        ref_ngrams = load_decontamination_reference()
        logger.info(f"Decontamination reference active ({len(ref_ngrams):,} 8-grams).")

    manifest = {}

    for target in targets:
        logger.info(f"\n==================== Ingesting: {target.upper()} ====================")
        pairs, stats = ingest_dataset(
            dataset_name=target,
            split=args.split,
            max_cases=args.max_cases,
            decontaminate=(not args.no_decontaminate),
            ref_ngrams=ref_ngrams,
            seed=args.seed,
            native_3class_pubmedqa=args.native_3class_pubmedqa,
        )

        out_file = out_dir / f"{target}_{args.split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        stats["out_file"] = str(out_file)
        stats["file_size_mb"] = round(file_size_mb, 2)
        manifest[target] = stats

        logger.info(
            f"-> Finished {target}: {stats['emitted_pairs']:,} pairs across {stats['unique_groups']:,} groups "
            f"saved to {out_file.name} ({file_size_mb:.2f} MB)"
        )
        if stats["dropped_decontaminated_cases"] > 0:
            logger.info(f"   (Filtered {stats['dropped_decontaminated_cases']:,} cases due to 8-gram JevBench overlap)")

    manifest_file = out_dir / "ingest_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nAll datasets processed. Manifest saved to {manifest_file}")


if __name__ == "__main__":
    main()
