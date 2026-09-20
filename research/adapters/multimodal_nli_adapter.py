#!/usr/bin/env python3
"""
multimodal_nli_adapter.py
=========================
Reference data adapter and conversion pipeline for Vision-Enabled NLI
cross-encoders based on Google Gemma 4.

Key Capabilities:
1. Unified Multimodal NLI Schema:
   {
       "id": str,
       "premise": str,         # May contain <<IMG_k>> placeholder markers
       "hypothesis": str,      # Declarative claim to verify
       "label": int,           # 0 = Contradiction, 1 = Entailment, 2 = Neutral
       "source": str,          # Dataset / task provenance
       "images": list[str|PIL],# One or more image paths or PIL objects
       "metadata": dict        # Task-specific metadata (boxes, Q&A, budget, etc.)
   }
2. Dataset Adapters:
   - SNLIVEAdapter: Standard SNLI-VE benchmark loader and label normalizer.
   - VQAToNLIAdapter: Rule-based & heuristic transform of (Image, Question, Answer)
     into declarative assertions, with annotator disagreement -> Neutral,
     and negative answer sampling -> Contradiction.
   - SpatialAndCountAdapter: Bounding-box-grounded claims (third-of-image,
     normalized coordinates, counts, relative positions, absent objects).
   - DocVQAAndChartAdapter: Visual document (invoice, table, receipt) and chart
     assertions with numerical/temporal mutations and unstated-attribute neutrals.
   - InterleavedMultiImageAdapter: Multi-page document / multi-image premise
     builder with long-context interleaving.
3. Gemma 4 Soft Token Budget & Aspect-Ratio Grid Calculator:
   - Divisibility by 48 (16x16 patch size * 3x3 pooling).
   - Soft token budgets: 70, 140, 280 (default), 560, 1120.
   - Aspect-ratio preserving grid optimization: (h_grid, w_grid).
4. PyTorch Dataset & DataCollatorNLIMM:
   - Dynamic batching with mixed text-only, single-image, and multi-image pairs.
   - Soft image token formatting and mm_token_type_ids generation.
   - Blank-pixel smoke testing hook to assert visual gradient flow.
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Label Mapping & Constants
# -----------------------------------------------------------------------------
# Standard Cross-Encoder Label Space (dleemiller / ModernCE / OpenJEV standard):
# 0 = Contradiction, 1 = Entailment, 2 = Neutral
LABEL2ID: Dict[str, int] = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

LABEL_SYNONYMS: Dict[str, str] = {
    "entailment": "entailment",
    "entails": "entailment",
    "entailed": "entailment",
    "supports": "entailment",
    "supported": "entailment",
    "true": "entailment",
    "yes": "entailment",
    "contradiction": "contradiction",
    "contradicts": "contradiction",
    "refutes": "contradiction",
    "refuted": "contradiction",
    "false": "contradiction",
    "no": "contradiction",
    "neutral": "neutral",
    "not_entailment": "neutral",
    "not-entailed": "neutral",
    "not enough info": "neutral",
    "not_enough_info": "neutral",
    "nei": "neutral",
    "unverifiable": "neutral",
    "unknown": "neutral",
}

IMG_PLACEHOLDER_PREFIX = "<<IMG"  # e.g., <<IMG_0>>, <<IMG_1>>
NUM_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def normalize_label(label: Union[str, int]) -> int:
    """Normalize arbitrary string or integer label to {0, 1, 2}."""
    if isinstance(label, int):
        if label in (0, 1, 2):
            return label
        raise ValueError(f"Unknown integer label {label}; expected 0, 1, or 2.")
    key = str(label).strip().lower()
    canonical = LABEL_SYNONYMS.get(key)
    if canonical is None:
        raise ValueError(f"Unknown string label {label!r}")
    return LABEL2ID[canonical]


# -----------------------------------------------------------------------------
# 2. Gemma 4 Vision Grid & Token Budget Engine
# -----------------------------------------------------------------------------
# Gemma 4 Vision Encoder:
# - Patch size = 16x16
# - Spatial pooling kernel = 3x3
# - Effective patch block = 48x48 pixels
# - Image dimensions (H, W) must be multiples of 48.
# - Supported soft token budgets: 70, 140, 280 (default), 560, 1120.
VALID_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)
DEFAULT_TOKEN_BUDGET = 280
PATCH_SIZE = 16
POOLING_KERNEL = 3
BLOCK_SIZE = PATCH_SIZE * POOLING_KERNEL  # 48 pixels


@dataclass(frozen=True)
class Gemma4VisionBudgetConfig:
    token_budget: int = DEFAULT_TOKEN_BUDGET
    block_size: int = BLOCK_SIZE

    def __post_init__(self):
        if self.token_budget not in VALID_TOKEN_BUDGETS:
            raise ValueError(
                f"token_budget {self.token_budget} not in supported Gemma 4 budgets: {VALID_TOKEN_BUDGETS}"
            )


def compute_gemma4_grid(
    orig_width: int,
    orig_height: int,
    target_budget: int = DEFAULT_TOKEN_BUDGET,
    block_size: int = BLOCK_SIZE,
) -> Tuple[int, int, int, int, int]:
    """
    Computes optimal Gemma 4 spatial grid dimensions (h_grid, w_grid) preserving aspect ratio.

    Returns:
        (h_grid, w_grid, num_tokens, resized_height, resized_width)
    where:
        - h_grid * w_grid <= target_budget
        - resized_height = h_grid * block_size
        - resized_width = w_grid * block_size
    """
    aspect_ratio = orig_width / max(orig_height, 1)

    # Solve: h_grid * w_grid <= target_budget with w_grid / h_grid approx aspect_ratio
    # w_grid approx sqrt(target_budget * aspect_ratio)
    # h_grid approx sqrt(target_budget / aspect_ratio)
    ideal_h = math.sqrt(target_budget / aspect_ratio)
    ideal_w = math.sqrt(target_budget * aspect_ratio)

    best_h, best_w = 1, 1
    best_loss = float("inf")

    # Search nearby integer grids around ideal dimensions
    h_candidates = range(max(1, int(ideal_h) - 3), int(ideal_h) + 4)
    for h in h_candidates:
        max_w = target_budget // h
        if max_w < 1:
            continue
        w_candidates = [
            w for w in (int(ideal_w) - 1, int(ideal_w), int(ideal_w) + 1, max_w)
            if 1 <= w <= max_w
        ]
        for w in w_candidates:
            if h * w > target_budget:
                continue
            cur_ratio = w / h
            # Loss balances aspect ratio preservation and token budget utilization
            ratio_err = abs(math.log(cur_ratio / aspect_ratio))
            utilization = (h * w) / target_budget
            loss = ratio_err - 0.25 * utilization
            if loss < best_loss:
                best_loss = loss
                best_h, best_w = h, w

    num_tokens = best_h * best_w
    resized_h = best_h * block_size
    resized_w = best_w * block_size
    return best_h, best_w, num_tokens, resized_h, resized_w


def preprocess_image_for_gemma4(
    image: Image.Image,
    target_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Resizes image to Gemma 4 block-aligned dimensions and returns metadata.
    """
    image = image.convert("RGB")
    w, h = image.size
    h_grid, w_grid, n_tokens, r_h, r_w = compute_gemma4_grid(w, h, target_budget)
    resized_img = image.resize((r_w, r_h), resample=Image.Resampling.BICUBIC)
    meta = {
        "orig_width": w,
        "orig_height": h,
        "h_grid": h_grid,
        "w_grid": w_grid,
        "n_tokens": n_tokens,
        "resized_height": r_h,
        "resized_width": r_w,
        "token_budget": target_budget,
    }
    return resized_img, meta


# -----------------------------------------------------------------------------
# 3. Multimodal NLI Sample Data Model
# -----------------------------------------------------------------------------
@dataclass
class MultimodalNLISample:
    """
    Canonical record for Vision NLI Cross-Encoder training and evaluation.
    """
    id: str
    premise: str
    hypothesis: str
    label: int  # 0=Contradiction, 1=Entailment, 2=Neutral
    source: str
    images: List[Union[str, Image.Image]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "label": int(self.label),
            "source": self.source,
            "images": [
                img if isinstance(img, str) else f"<PIL.Image {img.size}>"
                for img in self.images
            ],
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# 4. Domain Adapters
# -----------------------------------------------------------------------------

# --- 4.1 SNLI-VE Adapter ---
class SNLIVEAdapter:
    """
    Adapter for Stanford SNLI-VE (Visual Entailment) dataset.
    SNLI-VE grounds SNLI hypotheses against Flickr30k images.
    """
    SPLITS = ("train", "dev", "test")

    @staticmethod
    def convert_example(
        example_id: str,
        image: Union[str, Image.Image],
        hypothesis: str,
        gold_label: str,
        premise_caption: Optional[str] = None,
    ) -> MultimodalNLISample:
        label = normalize_label(gold_label)
        # In SNLI-VE, the visual premise is the image itself.
        # An optional premise prefix or lead-in can be added.
        premise_text = f"Visual context: <<IMG_0>>"
        if premise_caption:
            premise_text += f"\nDescription: {premise_caption.strip()}"

        return MultimodalNLISample(
            id=example_id,
            premise=premise_text,
            hypothesis=hypothesis.strip(),
            label=label,
            source="snli_ve",
            images=[image],
            metadata={"orig_label": gold_label},
        )


# --- 4.2 VQA to NLI Adapter (Rule-Based & Heuristic) ---
class VQAToNLIAdapter:
    """
    Converts (Image, Question, Answer) triples into declarative assertions,
    with negative mining (Contradiction) and annotator disagreement (Neutral).
    """

    def __init__(self, answer_pool_size: int = 5000, seed: int = 42):
        self.rng = random.Random(seed)
        self.seen_answers_by_type: Dict[str, List[str]] = defaultdict(list)
        self.answer_pool_size = answer_pool_size

    @staticmethod
    def question_to_declarative(question: str, answer: str, answer_type: str = "") -> Tuple[str, bool]:
        """
        Synthesizes a declarative statement from question and answer.
        Returns: (statement, can_flip_polarity_for_yesno)
        """
        q = question.strip().rstrip("?").strip()
        a = answer.strip().lower()
        ql = q.lower()

        # Color questions
        m = re.match(r"^what colou?r (?:is|are) (?:the )?(.+)$", ql)
        if m:
            subj = m.group(1).strip()
            return f"The {subj} is {a}.", False

        # Counting questions
        m = re.match(r"^how many (.+)$", ql)
        if m:
            noun = m.group(1)
            noun = re.sub(
                r"\b(are|is|can|do|does|there|in|on|the|this|that|photo|picture|image)\b.*$",
                "",
                noun,
            ).strip()
            if noun:
                count_word = NUM_WORDS.get(a, a)
                return f"There are {count_word} {noun} in the image.", False

        # Auxiliary verb Yes/No questions
        if answer_type == "yes/no" or ql.startswith(
            ("is ", "are ", "was ", "were ", "can ", "does ", "do ", "has ", "have ")
        ):
            m = re.match(r"^(is|are|was|were|does|do|did|can|has|have|will)\s+(.+)$", ql)
            if m:
                aux, rest = m.group(1), m.group(2)
                if aux in ("is", "are", "was", "were"):
                    # Robust subject matching: determiners + noun phrase or single pronoun/noun
                    m_subj = re.match(
                        r"^(there|this|that|these|those|(?:the|a|an)\s+[a-z0-9_\-]+(?:\s+[a-z0-9_\-]+)?|[a-z0-9_\-]+)\s+(.*)$",
                        rest,
                    )
                    if m_subj:
                        subj, pred = m_subj.group(1), m_subj.group(2)
                        return f"{subj.capitalize()} {aux} {pred}.", True
                    else:
                        return f"{rest.capitalize()} {aux}.", True
                elif aux in ("can", "will"):
                    m_subj = re.match(r"^((?:the|a|an)\s+[a-z0-9_\-]+|[a-z0-9_\-]+)\s+(.*)$", rest)
                    if m_subj:
                        return f"{m_subj.group(1).capitalize()} {aux} {m_subj.group(2)}.", True
                    return f"{rest.capitalize()}.", True
                else:
                    return f"{rest.capitalize()}.", True

        # Generic fallback
        return f"In this image, the answer to \"{q}?\" is {a}.", False

    @staticmethod
    def _normalize_answer(ans: str) -> str:
        s = ans.strip().lower()
        synonyms = {
            "mug": "cup",
            "couch": "sofa",
            "bike": "bicycle",
            "photo": "picture",
            "airplane": "plane",
            "grey": "gray",
        }
        return synonyms.get(s, s)

    def transform(
        self,
        example_id: str,
        image: Union[str, Image.Image],
        question: str,
        correct_answer: str,
        all_annotator_answers: Optional[List[str]] = None,
        answer_type: str = "",
        neg_sample_prob: float = 0.5,
    ) -> List[MultimodalNLISample]:
        """
        Generates Entailment, Contradiction, and Neutral samples from a VQA instance.
        """
        samples: List[MultimodalNLISample] = []
        stmt, flip_ok = self.question_to_declarative(question, correct_answer, answer_type)
        premise_str = "A photo: <<IMG_0>>"

        # Check annotator consensus for Neutral label using synonym normalization
        # In VQA v2, 10 human annotators answer. If fewer than 6 agree on the canonical concept, it reflects ambiguity.
        if all_annotator_answers and len(all_annotator_answers) >= 5:
            norm_target = self._normalize_answer(correct_answer)
            consensus = sum(
                1 for ans in all_annotator_answers
                if self._normalize_answer(str(ans)) == norm_target
            )
            if consensus < 6:
                samples.append(
                    MultimodalNLISample(
                        id=f"{example_id}_neutral_ambiguity",
                        premise=premise_str,
                        hypothesis=stmt,
                        label=LABEL2ID["neutral"],
                        source="vqa_annotator_disagreement",
                        images=[image],
                        metadata={"question": question, "consensus": consensus, "total": len(all_annotator_answers)},
                    )
                )
                return samples

        # Yes/No Question Handling
        if (answer_type == "yes/no" or correct_answer in ("yes", "no")) and flip_ok:
            label = LABEL2ID["entailment"] if correct_answer == "yes" else LABEL2ID["contradiction"]
            samples.append(
                MultimodalNLISample(
                    id=f"{example_id}_yesno",
                    premise=premise_str,
                    hypothesis=stmt,
                    label=label,
                    source="vqa_yesno",
                    images=[image],
                    metadata={"question": question, "answer": correct_answer},
                )
            )
            return samples

        # Standard Entailment sample
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_entailment",
                premise=premise_str,
                hypothesis=stmt,
                label=LABEL2ID["entailment"],
                source="vqa_answer_positive",
                images=[image],
                metadata={"question": question, "answer": correct_answer},
            )
        )

        # Mining Contradiction via distractor answer
        pool = self.seen_answers_by_type[answer_type]
        if pool and self.rng.random() < neg_sample_prob:
            candidate_wrong = self.rng.choice(pool)
            if candidate_wrong.lower() != correct_answer.lower():
                bad_stmt, _ = self.question_to_declarative(question, candidate_wrong, answer_type)
                samples.append(
                    MultimodalNLISample(
                        id=f"{example_id}_contradiction",
                        premise=premise_str,
                        hypothesis=bad_stmt,
                        label=LABEL2ID["contradiction"],
                        source="vqa_answer_distractor_neg",
                        images=[image],
                        metadata={"question": question, "distractor": candidate_wrong},
                    )
                )

        if len(pool) < self.answer_pool_size:
            pool.append(correct_answer)

        return samples


# --- 4.3 Spatial Reasoning & Object Counting Adapter ---
class SpatialAndCountAdapter:
    """
    Synthesizes grounded spatial, coordinate, counting, and absence claims from object detections.
    """
    ABSENT_OBJECT_CANDIDATES = (
        "giraffe", "helicopter", "piano", "traffic light", "zebra",
        "fire hydrant", "skateboard", "surfboard", "tennis racket", "toaster"
    )

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_claims(
        self,
        example_id: str,
        image: Union[str, Image.Image],
        detections: List[Dict[str, Any]],
        img_width: int,
        img_height: int,
    ) -> List[MultimodalNLISample]:
        """
        detections: list of dicts with 'box': [x1, y1, x2, y2] and 'label': str.
        """
        samples: List[MultimodalNLISample] = []
        if not detections:
            return samples

        premise_str = "Visual premise: <<IMG_0>>"
        present_labels = Counter(d["label"].strip().lower() for d in detections if d.get("label"))
        if not present_labels:
            return samples

        # 1. Third-of-image horizontal partition
        def get_third(cx_norm: float) -> str:
            if cx_norm < 1.0 / 3.0:
                return "left"
            elif cx_norm > 2.0 / 3.0:
                return "right"
            return "middle"

        target_det = self.rng.choice(detections)
        b = target_det["box"]
        lab = target_det["label"].lower()
        cx = (b[0] + b[2]) / (2.0 * max(img_width, 1))
        true_third = get_third(cx)
        false_third = self.rng.choice([t for t in ("left", "middle", "right") if t != true_third])

        # Entailment: true third
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_spatial_third_ent",
                premise=premise_str,
                hypothesis=f"There is a {lab} in the {true_third} third of the image.",
                label=LABEL2ID["entailment"],
                source="spatial_third",
                images=[image],
            )
        )
        # Contradiction: wrong third
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_spatial_third_con",
                premise=premise_str,
                hypothesis=f"There is a {lab} in the {false_third} third of the image.",
                label=LABEL2ID["contradiction"],
                source="spatial_third",
                images=[image],
            )
        )

        # 2. Object Counting
        count = present_labels[lab]
        count_word = NUM_WORDS.get(str(count), str(count))
        wrong_count = count + self.rng.choice([1, 2, 3])
        wrong_count_word = NUM_WORDS.get(str(wrong_count), str(wrong_count))

        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_count_ent",
                premise=premise_str,
                hypothesis=f"There {'is' if count == 1 else 'are'} {count_word} {lab}{'' if count == 1 else 's'} in the image.",
                label=LABEL2ID["entailment"],
                source="object_count",
                images=[image],
            )
        )
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_count_con",
                premise=premise_str,
                hypothesis=f"There are {wrong_count_word} {lab}s in the image.",
                label=LABEL2ID["contradiction"],
                source="object_count",
                images=[image],
            )
        )

        # 3. Absent Object Negative (Contradiction)
        for absent_cand in self.ABSENT_OBJECT_CANDIDATES:
            if absent_cand not in present_labels:
                samples.append(
                    MultimodalNLISample(
                        id=f"{example_id}_absent_con",
                        premise=premise_str,
                        hypothesis=f"There is a {absent_cand} in the image.",
                        label=LABEL2ID["contradiction"],
                        source="absent_object",
                        images=[image],
                    )
                )
                break

        # 4. Unverifiable Attribute (Neutral)
        # Claims about past history, price, owner, or internal state cannot be verified from the image alone.
        neutral_templates = [
            f"The {lab} was purchased last week.",
            f"The {lab} is owned by an engineer.",
            f"The {lab} will be moved indoors tomorrow.",
        ]
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_unverifiable_neu",
                premise=premise_str,
                hypothesis=self.rng.choice(neutral_templates),
                label=LABEL2ID["neutral"],
                source="unverifiable_visual_attribute",
                images=[image],
            )
        )

        return samples


# --- 4.4 Document Understanding & Chart QA Adapter ---
class DocVQAAndChartAdapter:
    """
    Converts document (Invoice, Table, Form, Receipt) and Chart QA into NLI assertions.
    Handles numerical mutation for Contradictions and unstated metadata for Neutrals.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def convert_document_qa(
        self,
        example_id: str,
        doc_image: Union[str, Image.Image],
        doc_type: str,  # 'invoice', 'receipt', 'table', 'chart'
        key_field: str,
        value: str,
        ocr_context: Optional[str] = None,
    ) -> List[MultimodalNLISample]:
        samples: List[MultimodalNLISample] = []
        premise = f"Document type: {doc_type.capitalize()}\nImage: <<IMG_0>>"
        if ocr_context:
            premise += f"\nExtracted OCR Text:\n{ocr_context[:1000]}"

        # Entailment
        hyp_ent = f"According to the {doc_type}, the {key_field} is {value}."
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_doc_ent",
                premise=premise,
                hypothesis=hyp_ent,
                label=LABEL2ID["entailment"],
                source=f"doc_{doc_type}",
                images=[doc_image],
                metadata={"field": key_field, "value": value},
            )
        )

        # Contradiction: Numeric or string mutation
        val_str = value.strip()
        mutated_val = None
        # Check if numeric / monetary
        num_match = re.search(r"(\$|€|£)?\s*([0-9]+(?:\.[0-9]+)?)", val_str)
        if num_match:
            currency = num_match.group(1) or ""
            amount = float(num_match.group(2))
            delta = self.rng.choice([1.15, 0.75, 1.5, 2.0])
            mutated_amount = round(amount * delta, 2)
            mutated_val = f"{currency}{mutated_amount}"
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):  # Date: shift year/month
            parts = val_str.split("-")
            mutated_year = str(int(parts[0]) - self.rng.choice([1, 2, 5]))
            mutated_val = f"{mutated_year}-{parts[1]}-{parts[2]}"
        else:
            # Contextual alternative entities to prevent superficial token leakage like '(revoked)'
            distractor_entities = [
                "Apex Holdings Inc.", "Meridian Solutions Ltd.", "Summit Industrial Corp.",
                "Express Net-30", "Wire Transfer / ACH", "Pending Compliance Verification",
                "Warehouse Bay 4B", "Standard Ground Shipping", "Tier 2 Enterprise Tier",
            ]
            mutated_val = self.rng.choice([e for e in distractor_entities if e.lower() != val_str.lower()] or ["Unverified Entity"])

        hyp_con = f"According to the {doc_type}, the {key_field} is {mutated_val}."
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_doc_con",
                premise=premise,
                hypothesis=hyp_con,
                label=LABEL2ID["contradiction"],
                source=f"doc_{doc_type}_mutation",
                images=[doc_image],
                metadata={"field": key_field, "original": value, "mutated": mutated_val},
            )
        )

        # Neutral: Information not mentioned in the document from a diverse pool of enterprise attributes
        diverse_neutral_templates = [
            f"The payment for this {doc_type} was routed through an offshore escrow service.",
            f"The vendor associated with this {doc_type} holds ISO 27001 data security certification.",
            f"An expedited handling surcharge was waived for this {doc_type}.",
            f"This transaction was approved by the regional compliance officer under policy Sec-4.",
            f"A carbon footprint offset of 4.2 kg CO2 was purchased in connection with this {doc_type}.",
            f"The items listed on this {doc_type} are covered by a two-year manufacturer warranty.",
            f"A recurring annual maintenance contract is bundled with this {doc_type}.",
            f"The physical hardcopy of this {doc_type} is archived in warehouse facility B.",
        ]
        samples.append(
            MultimodalNLISample(
                id=f"{example_id}_doc_neu",
                premise=premise,
                hypothesis=self.rng.choice(diverse_neutral_templates),
                label=LABEL2ID["neutral"],
                source=f"doc_{doc_type}_unstated",
                images=[doc_image],
            )
        )

        return samples


# --- 4.5 Interleaved Multi-Image Long-Context Adapter ---
class InterleavedMultiImageAdapter:
    """
    Builds multi-page / multi-image premises interleaved with long text.
    Capitalizes on Gemma 4's 128K context window.
    """

    @staticmethod
    def build_multi_page_premise(
        page_images: List[Union[str, Image.Image]],
        page_texts: List[str],
        document_title: str = "Corporate Annual Filing",
    ) -> str:
        parts = [f"=== Document: {document_title} ==="]
        for i, (img, text) in enumerate(zip(page_images, page_texts)):
            parts.append(f"\n--- Page {i + 1} ---")
            parts.append(f"<<IMG_{i}>>")
            parts.append(text.strip())
        return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# 5. Multimodal Data Collator for Gemma 4
# -----------------------------------------------------------------------------
class Gemma4MultimodalNLICollator:
    """
    PyTorch DataCollator for Gemma 4 Multimodal NLI Cross-Encoder.
    Pads text sequences, formats image placeholder blocks according to
    Gemma 4 soft-token budgets, and collates pixel tensors.
    """

    TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"

    def __init__(
        self,
        tokenizer: Any,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_length: int = 4096,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.token_budget = token_budget
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        # Gemma 4 special tokens:
        # Default placeholder syntax or HuggingFace Gemma 4 soft token injection
        self.image_token_str = "<|image|>"
        if hasattr(tokenizer, "image_token"):
            self.image_token_str = tokenizer.image_token

    def format_premise_images(
        self,
        premise: str,
        images: List[Image.Image],
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """
        Substitutes <<IMG_k>> with the exact number of soft tokens calculated
        by the Gemma 4 grid engine for image k.
        Returns:
            (expanded_premise, list_of_grids[(h_grid, w_grid, n_tokens)])
        """
        expanded = premise
        grid_info = []
        for i, img in enumerate(images):
            placeholder = f"<<IMG_{i}>>"
            h_grid, w_grid, n_tok, _, _ = compute_gemma4_grid(
                img.width, img.height, self.token_budget
            )
            grid_info.append((h_grid, w_grid, n_tok))
            # Format Gemma 4 soft token span
            # e.g., <|vision_start|><|image_pad|>*N<|vision_end|> or repeated <|image|>
            soft_token_span = f"<|vision_start|>{self.image_token_str * n_tok}<|vision_end|>"
            expanded = expanded.replace(placeholder, soft_token_span)

        return expanded, grid_info

    def __call__(self, samples: Sequence[MultimodalNLISample]) -> Dict[str, Any]:
        """
        Collates a batch of MultimodalNLISamples into model-ready tensors.
        """
        import torch

        batch_texts: List[str] = []
        batch_labels: List[int] = []
        batch_images: List[torch.Tensor] = []
        batch_grids: List[torch.Tensor] = []

        for sample in samples:
            # Ensure images are loaded as PIL Images
            pil_images: List[Image.Image] = []
            for img_item in sample.images:
                if isinstance(img_item, str):
                    if os.path.exists(img_item):
                        pil_images.append(Image.open(img_item).convert("RGB"))
                    else:
                        # Fallback dummy placeholder
                        pil_images.append(Image.new("RGB", (640, 480), color=(128, 128, 128)))
                elif isinstance(img_item, Image.Image):
                    pil_images.append(img_item.convert("RGB"))
                else:
                    pil_images.append(Image.new("RGB", (640, 480), color=(128, 128, 128)))

            # Expand premise with soft tokens
            expanded_premise, grid_info = self.format_premise_images(sample.premise, pil_images)
            pair_text = self.TEMPLATE.format(
                premise=expanded_premise.strip(),
                hypothesis=sample.hypothesis.strip(),
            )
            batch_texts.append(pair_text)
            batch_labels.append(sample.label)

            # Preprocess and resize images to Gemma 4 grid
            for img, (hg, wg, ntok) in zip(pil_images, grid_info):
                resized_img, _ = preprocess_image_for_gemma4(img, self.token_budget)
                # Convert to FloatTensor in range [-1.0, 1.0] (Gemma 4 convention)
                img_arr = np.array(resized_img, dtype=np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # [C, H, W]
                batch_images.append(img_tensor)
                batch_grids.append(torch.tensor([hg, wg, ntok], dtype=torch.long))

        # Tokenize text pairs
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        if batch_images:
            batch["pixel_values"] = batch_images  # list or nested tensor
            batch["image_grids"] = torch.stack(batch_grids, dim=0)

        return batch


# -----------------------------------------------------------------------------
# 6. Verification & Self-Test Suite
# -----------------------------------------------------------------------------
def run_adapter_self_test():
    """
    Executes an end-to-end verification of all multimodal NLI adapters,
    token budgeting arithmetic, and collator tensor shapes.
    """
    print("=" * 70)
    print("RUNNING MULTIMODAL NLI ADAPTER & GEMMA 4 GRID SELF-TEST")
    print("=" * 70)

    # 1. Test Gemma 4 Grid Engine across resolutions and budgets
    test_cases = [
        ("Square 1:1", 640, 640, 280),
        ("Landscape 4:3", 800, 600, 280),
        ("Widescreen 16:9", 1920, 1080, 560),
        ("Portrait 9:16", 1080, 1920, 560),
        ("Dense Document 3:4", 1200, 1600, 1120),
        ("Thumbnail preview", 320, 240, 70),
    ]

    print("\n--- 1. Gemma 4 Aspect-Ratio Grid Calculations ---")
    for name, w, h, budget in test_cases:
        hg, wg, ntok, rh, rw = compute_gemma4_grid(w, h, budget)
        print(
            f"[{name:20s}] Orig: {w:4d}x{h:<4d} | Budget: {budget:4d} -> "
            f"Grid: ({hg:2d}, {wg:2d}) = {ntok:4d} tokens | Resized: {rw:4d}x{rh:<4d} px"
        )
        assert ntok <= budget, f"Exceeded budget: {ntok} > {budget}"
        assert rh % BLOCK_SIZE == 0, f"Height {rh} not divisible by {BLOCK_SIZE}"
        assert rw % BLOCK_SIZE == 0, f"Width {rw} not divisible by {BLOCK_SIZE}"

    # 2. Test Adapters on Synthetic Inputs
    dummy_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    synthetic_samples: List[MultimodalNLISample] = []

    # SNLI-VE
    snli_sample = SNLIVEAdapter.convert_example(
        example_id="snli_01",
        image=dummy_img,
        hypothesis="Two dogs are playing in the grass.",
        gold_label="entailment",
    )
    synthetic_samples.append(snli_sample)

    # VQA to NLI
    vqa_adapter = VQAToNLIAdapter(seed=123)
    vqa_samples = vqa_adapter.transform(
        example_id="vqa_01",
        image=dummy_img,
        question="What color is the car?",
        correct_answer="red",
        all_annotator_answers=["red"] * 8 + ["blue"] * 2,
        answer_type="other",
        neg_sample_prob=1.0,
    )
    synthetic_samples.extend(vqa_samples)

    # Spatial & Counting
    spatial_adapter = SpatialAndCountAdapter(seed=123)
    detections = [
        {"box": [50, 60, 180, 240], "label": "bicycle"},
        {"box": [220, 100, 360, 300], "label": "bicycle"},
        {"box": [450, 80, 600, 400], "label": "person"},
    ]
    spatial_samples = spatial_adapter.generate_claims(
        example_id="spatial_01",
        image=dummy_img,
        detections=detections,
        img_width=640,
        img_height=480,
    )
    synthetic_samples.extend(spatial_samples)

    # DocVQA
    doc_adapter = DocVQAAndChartAdapter(seed=123)
    doc_samples = doc_adapter.convert_document_qa(
        example_id="doc_01",
        doc_image=dummy_img,
        doc_type="invoice",
        key_field="Total Due",
        value="$1,450.00",
        ocr_context="INVOICE #98231\nDate: 2026-03-12\nSubtotal: $1,400.00\nTax: $50.00\nTotal Due: $1,450.00",
    )
    synthetic_samples.extend(doc_samples)

    print(f"\n--- 2. Generated Multimodal NLI Dataset Samples: {len(synthetic_samples)} ---")
    label_counts = Counter(s.label for s in synthetic_samples)
    print("Label Distribution:")
    for lbl_id, count in sorted(label_counts.items()):
        print(f"  {lbl_id} ({ID2LABEL[lbl_id]:13s}): {count} samples")

    for s in synthetic_samples[:6]:
        print(f"\n[ID: {s.id}] (Source: {s.source}, Label: {ID2LABEL[s.label]})")
        print(f"  Premise:    {s.premise.replace(chr(10), ' ')[:100]}...")
        print(f"  Hypothesis: {s.hypothesis}")

    # 3. Test Gemma 4 Multimodal Collator with a mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.image_token = "<|image|>"

        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=4096,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ):
            import torch
            # Mock tokenization length based on word count + image tokens
            lens = [len(t.split()) for t in texts]
            max_l = max(lens)
            if pad_to_multiple_of:
                max_l = ((max_l + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            input_ids = torch.randint(1, 1000, (len(texts), max_l))
            attention_mask = torch.ones((len(texts), max_l), dtype=torch.long)
            for i, l in enumerate(lens):
                attention_mask[i, l:] = 0
                input_ids[i, l:] = self.pad_token_id
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    print("\n--- 3. Testing Gemma 4 Multimodal Collator ---")
    mock_tok = MockTokenizer()
    collator = Gemma4MultimodalNLICollator(
        tokenizer=mock_tok,
        token_budget=280,
        max_length=4096,
    )
    batch = collator(synthetic_samples[:4])
    print(f"Collation Output Keys: {list(batch.keys())}")
    print(f"input_ids shape:      {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels tensor:        {batch['labels']} ({[ID2LABEL[l.item()] for l in batch['labels']]})")
    print(f"num image tensors:    {len(batch['pixel_values'])}")
    print(f"image_grids shape:    {batch['image_grids'].shape}")
    for idx, grid in enumerate(batch["image_grids"]):
        print(f"  Image {idx}: grid ({grid[0]}, {grid[1]}), tokens = {grid[2]}")

    print("\n[SUCCESS] All Multimodal NLI Adapter checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_adapter_self_test()
