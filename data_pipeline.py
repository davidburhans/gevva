#!/usr/bin/env python3
"""data_pipeline.py
===================
Dataset compilation and balancing pipeline for Multimodal, Multilingual,
128K NLI Cross-Encoder based on Google Gemma 4.

Label Convention (dleemiller / ModernCE / OpenJEV standard):
    0: contradiction
    1: entailment
    2: neutral

Schema per sample:
    {
        "id": str,
        "premise": str,
        "hypothesis": str,
        "label": int,       # 0=contradiction, 1=entailment, 2=neutral
        "source": str,      # dataset provenance
        "language": str,    # ISO language code (en, es, zh, hi, etc.)
        "image": str,       # path to image or empty string
        "length": int       # approximate token count
    }

Usage:
    python data_pipeline.py --out-dir ./data --quick     # Quick build for fast validation
    python data_pipeline.py --out-dir ./data --full      # Full-scale dataset build
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Canonical Label Space
# -----------------------------------------------------------------------------
CONTRADICTION: int = 0
ENTAILMENT: int = 1
NEUTRAL: int = 2

LABEL2ID: Dict[str, int] = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# Standard SNLI / MNLI native format: 0=entailment, 1=neutral, 2=contradiction
NATIVE_MNLI2OURS: Dict[int, int] = {0: 1, 1: 2, 2: 0}

SYNONYMS: Dict[str, str] = {
    "entailment": "entailment",
    "entails": "entailment",
    "supports": "entailment",
    "contradiction": "contradiction",
    "contradicts": "contradiction",
    "refutes": "contradiction",
    "neutral": "neutral",
    "not_entailment": "neutral",
    "not enough info": "neutral",
    "not_enough_info": "neutral",
    "nei": "neutral",
}


HAYSTACK_VAL_FRACTION = 0.1
TEST_FRACTION_OF_VAL = 0.4
XNLI_VAL_PER_LANG = 100


def _xnli_row(ex: Dict[str, Any], lang: str, split_tag: str, lang_count: int) -> Optional[Dict[str, Any]]:
    """Builds one canonical XNLI row (or None for malformed labels)."""
    p, h, l = ex["premise"], ex["hypothesis"], ex["label"]
    if l not in (0, 1, 2) or not p or not h:
        return None
    return {
        "id": f"xnli_{lang}_{split_tag}_{lang_count:06d}",
        "premise": p.strip(),
        "hypothesis": h.strip(),
        "label": NATIVE_MNLI2OURS[l],
        "source": f"xnli_{lang}",
        "language": lang,
        "image": "",
        "length": len(p.split()) + len(h.split()),
    }


def _pair_key(premise: str, hypothesis: str) -> str:
    """Stable content key for a (premise, hypothesis) pair (leakage detection)."""
    norm = lambda t: " ".join(t.lower().split())
    return hashlib.sha1(f"{norm(premise)}\x1f{norm(hypothesis)}".encode("utf-8")).hexdigest()


def _is_haystack_holdout(premise: str, hypothesis: str) -> bool:
    """Deterministic 10% content-addressed holdout for haystack source pairs."""
    digest = hashlib.md5(_pair_key(premise, hypothesis).encode()).hexdigest()
    return int(digest, 16) % 10 == 0


def _dedupe_split(
    train_rows: List[Dict[str, Any]], val_rows: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """Removes verbatim pair duplicates within train, and val rows duplicating train.

    WHY: verbatim val-in-train duplicates inflate selection/benchmark metrics
    (audit found 218 such rows, incl. all multimodal_synth).

    Example:
        train, val, n_t, n_v = _dedupe_split(train_samples, val_samples)
    """
    seen: set = set()
    clean_train, dropped_train = [], 0
    for row in train_rows:
        # WHY image in key: multimodal rows share one premise template; their content
        # identity is (image, claim), otherwise dedup would collapse the whole set.
        key = f"{_pair_key(row['premise'], row['hypothesis'])}\x1f{row.get('image', '')}"
        if key in seen:
            dropped_train += 1
            continue
        seen.add(key)
        clean_train.append(row)
    clean_val, dropped_val = [], 0
    for row in val_rows:
        key = f"{_pair_key(row['premise'], row['hypothesis'])}\x1f{row.get('image', '')}"
        if key in seen:
            dropped_val += 1
            continue
        seen.add(key)
        clean_val.append(row)
    return clean_train, clean_val, dropped_train, dropped_val
    if isinstance(val, str):
        cleaned = val.strip().lower()
        canonical = SYNONYMS.get(cleaned)
        if canonical:
            return LABEL2ID[canonical]
    return None


# -----------------------------------------------------------------------------
# Long-Context Synthetic Haystack Generator
# -----------------------------------------------------------------------------
def build_haystack_samples(
    pairs: List[Tuple[str, str, int, str]],
    filler_pool: List[str],
    n_samples: int = 2000,
    min_fillers: int = 8,
    max_fillers: int = 25,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Builds synthetic long-document NLI pairs:
    - 40% Entailment (premise needle embedded at random depth)
    - 30% Contradiction (fact corrupted via entity / number mutation)
    - 30% Neutral (needle dropped entirely -> unsupported in document)
    """
    rng = random.Random(seed)
    haystack_rows: List[Dict[str, Any]] = []

    if not filler_pool or not pairs:
        return haystack_rows

    rng.shuffle(pairs)

    for i in range(min(n_samples, len(pairs))):
        premise_needle, hypothesis, orig_label, lang = pairs[i]
        n_fillers = rng.randint(min_fillers, max_fillers)
        fillers = rng.sample(filler_pool, min(n_fillers, len(filler_pool)))

        dice = rng.random()
        if dice < 0.30:
            # Drop needle: evidence is NOT in document -> Neutral ("Not Stated")
            doc = "\n\n".join(fillers)
            label = NEUTRAL
            sub_source = "haystack_drop_neutral"
        elif dice < 0.60:
            # Corrupted needle -> Contradiction
            # Mutate numbers or insert negation
            mutated_premise = re.sub(
                r"\b(\d+)\b",
                lambda m: str(int(m.group(1)) + rng.choice([5, 10, 100])),
                premise_needle,
                count=1,
            )
            if mutated_premise == premise_needle:
                # If no numbers, insert negation
                words = premise_needle.split()
                if len(words) > 3:
                    words.insert(2, "never")
                    mutated_premise = " ".join(words)
                else:
                    mutated_premise = "It is completely false that " + premise_needle

            pos = rng.randrange(len(fillers) + 1)
            doc_paragraphs = fillers[:pos] + [mutated_premise] + fillers[pos:]
            doc = "\n\n".join(doc_paragraphs)
            label = CONTRADICTION
            sub_source = "haystack_corrupted_con"
        else:
            # Genuine needle inserted -> Keep original relation (or entailment)
            pos = rng.randrange(len(fillers) + 1)
            doc_paragraphs = fillers[:pos] + [premise_needle] + fillers[pos:]
            doc = "\n\n".join(doc_paragraphs)
            label = orig_label
            sub_source = "haystack_embedded"

        haystack_rows.append(
            {
                "id": f"haystack_{i:06d}",
                "premise": doc,
                "hypothesis": hypothesis,
                "label": label,
                "source": sub_source,
                "language": lang,
                "image": "",
                "length": len(doc.split()) + len(hypothesis.split()),
            }
        )

    return haystack_rows


# -----------------------------------------------------------------------------
# Local Teacher Model Distillation Client (Port 8080)
# -----------------------------------------------------------------------------
class LocalTeacherClient:
    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str = "unsloth/gemma-4-26B-A4B-it-GGUF:BF16"):
        self.base_url = base_url
        self.model = model

    def query(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> Optional[str]:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


# -----------------------------------------------------------------------------
# Main Dataset Compiler
# -----------------------------------------------------------------------------
def compile_dataset(out_dir: str, mode: str = "quick", seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rng = random.Random(seed)

    print(f"=== Compiling Gemma 4 NLI Dataset (Mode: {mode.upper()}) ===")
    from datasets import concatenate_datasets, load_dataset

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    filler_pool: List[str] = []
    pairs_for_haystack: List[Tuple[str, str, int, str]] = []
    val_haystack_pairs: List[Tuple[str, str, int, str]] = []

    # Target counts
    if mode == "quick":
        N_SNLI = 10000
        N_MNLI = 10000
        N_ANLI = 3000
        N_FEVER = 5000
        N_SCITAIL = 3000
        N_QNLI = 5000
        N_XNLI = 2000
        N_HAYSTACK = 2000
        N_VISION = 2000
    elif mode == "stage2":
        N_SNLI = 80000
        N_MNLI = 120000
        N_ANLI = 20000
        N_FEVER = 60000
        N_SCITAIL = 23000
        N_QNLI = 40000
        N_XNLI = 15000
        N_HAYSTACK = 10000
        N_VISION = 5000
    else:  # full
        N_SNLI = 150000
        N_MNLI = 200000
        N_ANLI = 25000
        N_FEVER = 100000
        N_SCITAIL = 23000
        N_QNLI = 80000
        N_XNLI = 30000
        N_HAYSTACK = 20000
        N_VISION = 30000

    # 1. Stanford SNLI
    print("Loading SNLI...")
    try:
        snli = load_dataset("stanfordnlp/snli")
        for split, max_n in [("train", N_SNLI), ("validation", 1000)]:
            ds = snli[split]
            count = 0
            for ex in ds:
                p, h, l = ex["premise"], ex["hypothesis"], ex["label"]
                if l not in (0, 1, 2) or not p or not h:
                    continue
                label = NATIVE_MNLI2OURS[l]
                row = {
                    "id": f"snli_{split}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"snli_{split}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                if split == "train":
                    if (len(pairs_for_haystack) + len(val_haystack_pairs) < 15000
                            and _is_haystack_holdout(p, h)):
                        # WHY: whole source pairs are held out of training (their plain
                        # rows go to val) so val haystack rows never test memorized
                        # needles (audit finding: pair-shared haystack leakage).
                        val_haystack_pairs.append((p.strip(), h.strip(), label, "en"))
                        val_samples.append(row)
                    else:
                        train_samples.append(row)
                        if len(filler_pool) < 20000:
                            filler_pool.append(p.strip())
                        if len(pairs_for_haystack) < 15000:
                            pairs_for_haystack.append((p.strip(), h.strip(), label, "en"))
                else:
                    val_samples.append(row)
                count += 1
                if count >= max_n:
                    break
        print(f"  SNLI loaded: {count} rows")
    except Exception as e:
        print(f"  SNLI load failed: {e}")

    # 2. NYU MNLI
    print("Loading MNLI...")
    try:
        mnli = load_dataset("nyu-mll/multi_nli")
        for split, max_n in [("train", N_MNLI), ("validation_matched", 1000)]:
            ds = mnli[split]
            count = 0
            for ex in ds:
                p, h, l = ex["premise"], ex["hypothesis"], ex["label"]
                if l not in (0, 1, 2) or not p or not h:
                    continue
                label = NATIVE_MNLI2OURS[l]
                row = {
                    "id": f"mnli_{split}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"mnli_{split}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                if split == "train":
                    if (len(pairs_for_haystack) + len(val_haystack_pairs) < 25000
                            and _is_haystack_holdout(p, h)):
                        val_haystack_pairs.append((p.strip(), h.strip(), label, "en"))
                        val_samples.append(row)
                    else:
                        train_samples.append(row)
                        if len(filler_pool) < 30000:
                            filler_pool.append(p.strip())
                        if len(pairs_for_haystack) < 25000:
                            pairs_for_haystack.append((p.strip(), h.strip(), label, "en"))
                else:
                    val_samples.append(row)
                count += 1
                if count >= max_n:
                    break
        print(f"  MNLI loaded: {count} rows")
    except Exception as e:
        print(f"  MNLI load failed: {e}")

    # 3. Adversarial NLI (ANLI)
    print("Loading ANLI (R1, R2, R3)...")
    try:
        anli = load_dataset("facebook/anli")
        count = 0
        for round_name in ["train_r1", "train_r2", "train_r3"]:
            ds = anli[round_name]
            for ex in ds:
                p, h, l = ex["premise"], ex["hypothesis"], ex["label"]
                if l not in (0, 1, 2) or not p or not h:
                    continue
                label = NATIVE_MNLI2OURS[l]
                row = {
                    "id": f"anli_{round_name}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"anli_{round_name}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                train_samples.append(row)
                count += 1
                if count >= N_ANLI:
                    break
            if count >= N_ANLI:
                break
        print(f"  ANLI loaded: {count} rows")
    except Exception as e:
        print(f"  ANLI load failed: {e}")

    # 4. Multilingual NLI (XNLI)
    # WHY: XNLI's validation set is a translation of MNLI-dev. Training on it while
    # MNLI-dev sits in val would leak the checkpoint-selection set cross-lingually
    # (audit finding). Train rows come from the XNLI *train* split (MNLI-train
    # derived); only genuine validation rows go to val.
    print("Loading XNLI (Multilingual)...")
    xnli_langs = ["es", "fr", "de", "ru", "ar", "zh", "hi", "vi", "sw"]
    xnli_per_lang = int(N_XNLI / len(xnli_langs))
    count = 0
    for lang in xnli_langs:
        try:
            lang_count = 0
            for ex in load_dataset("facebook/xnli", lang, split="train"):
                row = _xnli_row(ex, lang, "train", lang_count)
                if row:
                    train_samples.append(row)
                    lang_count += 1
                if lang_count >= xnli_per_lang:
                    break
            for ex in load_dataset("facebook/xnli", lang, split="validation"):
                row = _xnli_row(ex, lang, "val", lang_count)
                if row:
                    val_samples.append(row)
                    lang_count += 1
                if lang_count >= xnli_per_lang + XNLI_VAL_PER_LANG:
                    break
            count += lang_count
        except Exception as e:
            print(f"  XNLI {lang} load failed: {e}")
    print(f"  XNLI total loaded: {count} rows")

    # 5. NLI-FEVER (Fact Verification & Refutations)
    print("Loading NLI-FEVER...")
    try:
        fever = load_dataset("pietrolesci/nli_fever")
        for split, max_n in [("train", N_FEVER), ("dev", 1000)]:
            ds = fever[split]
            count = 0
            for ex in ds:
                p, h, l = ex["hypothesis"], ex["premise"], ex["label"]
                if l not in (0, 1, 2) or not p or not h:
                    continue
                label = NATIVE_MNLI2OURS[l]
                row = {
                    "id": f"fever_{split}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"fever_{split}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                if split == "train":
                    train_samples.append(row)
                else:
                    val_samples.append(row)
                count += 1
                if count >= max_n:
                    break
        print(f"  FEVER loaded: {count} rows")
    except Exception as e:
        print(f"  FEVER load failed: {e}")

    # 6. SciTail (Scientific Reasoning)
    print("Loading SciTail...")
    try:
        scitail = load_dataset("allenai/scitail", "snli_format")
        for split, max_n in [("train", N_SCITAIL), ("validation", 500)]:
            ds = scitail[split]
            count = 0
            for ex in ds:
                p, h = ex["sentence1"], ex["sentence2"]
                gold = ex.get("gold_label")
                if not p or not h or gold not in LABEL2ID:
                    continue
                label = LABEL2ID[gold]
                row = {
                    "id": f"scitail_{split}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"scitail_{split}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                if split == "train":
                    train_samples.append(row)
                else:
                    val_samples.append(row)
                count += 1
                if count >= max_n:
                    break
        print(f"  SciTail loaded: {count} rows")
    except Exception as e:
        print(f"  SciTail load failed: {e}")

    # 7. QNLI (Passage Question Answering Grounding)
    print("Loading QNLI...")
    try:
        qnli = load_dataset("nyu-mll/glue", "qnli")
        for split, max_n in [("train", N_QNLI), ("validation", 1000)]:
            ds = qnli[split]
            count = 0
            for ex in ds:
                p, h, l = ex["sentence"], ex["question"], ex["label"]
                if l not in (0, 1) or not p or not h:
                    continue
                # Note: 0=entailment (1), 1=not_entailment -> mapped strictly to neutral (2)
                label = ENTAILMENT if l == 0 else NEUTRAL
                row = {
                    "id": f"qnli_{split}_{count:06d}",
                    "premise": p.strip(),
                    "hypothesis": h.strip(),
                    "label": label,
                    "source": f"qnli_{split}",
                    "language": "en",
                    "image": "",
                    "length": len(p.split()) + len(h.split()),
                }
                if split == "train":
                    train_samples.append(row)
                else:
                    val_samples.append(row)
                count += 1
                if count >= max_n:
                    break
        print(f"  QNLI loaded: {count} rows")
    except Exception as e:
        print(f"  QNLI load failed: {e}")

    # 8. Long-Context Synthetic Haystack
    print("Generating Synthetic Haystack Pairs...")
    # WHY pair-disjoint: val haystack rows are built ONLY from source pairs held out
    # of training (their plain NLI rows also went to val), so haystack_drop/embedded
    # val accuracies can never reward memorized needles (audit finding).
    print("Generating Synthetic Haystack Pairs...")
    n_val_haystack = int(N_HAYSTACK * HAYSTACK_VAL_FRACTION)
    train_haystack = build_haystack_samples(
        pairs_for_haystack,
        filler_pool,
        n_samples=N_HAYSTACK - n_val_haystack,
        min_fillers=6,
        max_fillers=18,
        seed=seed,
    )
    val_haystack = build_haystack_samples(
        val_haystack_pairs,
        filler_pool,
        n_samples=n_val_haystack + 50,
        min_fillers=6,
        max_fillers=18,
        seed=seed + 1,
    )[:n_val_haystack]
    train_samples.extend(train_haystack)
    val_samples.extend(val_haystack)
    print(f"  Haystack pairs generated: {len(train_haystack)} train / {len(val_haystack)} val rows")

    # 6. Visual / Multimodal Grounding Pairs
    print("Generating Multimodal Grounding Samples...")
    # Generate spatial & visual grounding claim samples using PIL images
    from PIL import Image, ImageDraw

    vis_count = 0
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    colors = [("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("green", (0, 255, 0)), ("yellow", (255, 255, 0))]
    shapes = ["circle", "rectangle", "square"]

    for i in range(N_VISION):
        img_name = f"synth_{i:04d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        
        # Create a simple synthetic scene
        img = Image.new("RGB", (320, 240), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        color_name, color_val = rng.choice(colors)
        shape_name = rng.choice(shapes)
        
        # Position: left, center, right
        pos_type = rng.choice(["left", "center", "right"])
        if pos_type == "left":
            bbox = (30, 70, 100, 150)
        elif pos_type == "center":
            bbox = (120, 70, 200, 150)
        else:
            bbox = (220, 70, 290, 150)
            
        if shape_name == "circle":
            draw.ellipse(bbox, fill=color_val)
        else:
            draw.rectangle(bbox, fill=color_val)
        img.save(img_path)

        # Generate Entailment, Contradiction, Neutral
        rel_type = rng.choice([ENTAILMENT, CONTRADICTION, NEUTRAL])
        if rel_type == ENTAILMENT:
            claim = f"There is a {color_name} {shape_name} in the {pos_type} of the image."
        elif rel_type == CONTRADICTION:
            wrong_color = [c[0] for c in colors if c[0] != color_name][0]
            claim = f"There is a {wrong_color} {shape_name} in the {pos_type} of the image."
        else:
            claim = f"The {shape_name} was placed there yesterday by a photographer."

        row = {
            "id": f"vision_{i:06d}",
            # WHY plain text (audit CRITICAL): literal <|vision_start|>/<|image_pad|>
            # markers EXPAND to 1,698 image tokens in text-only tokenization - truncated
            # garbage in training and an SDK assertion hazard. The marker string is
            # preserved in metadata for the image-into-collator fix (A7).
            "premise": "An image showing geometric figures.",
            "premise_markers": "<|vision_start|>" + "<|image_pad|>" * 280 + "<|vision_end|>",
            "hypothesis": claim,
            "label": rel_type,
            "source": "multimodal_synth",
            "language": "en",
            "image": os.path.relpath(img_path, out_dir),
            "length": 300,
        }
        if i < int(N_VISION * 0.9):
            train_samples.append(row)
        else:
            val_samples.append(row)
        vis_count += 1
    print(f"  Multimodal samples compiled: {vis_count} rows")

    # 7. SDK-Aligned Synthetic Samples (Tool routing, rubric grading, reranking, cloze, RAG)
    sdk_train_path = os.path.join(out_dir, "sdk_synthetic_train.jsonl")
    sdk_val_path = os.path.join(out_dir, "sdk_synthetic_val.jsonl")
    if os.path.exists(sdk_train_path):
        print(f"Loading SDK-aligned synthetic training pairs from {sdk_train_path}...")
        with open(sdk_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_samples.append(json.loads(line))
    if os.path.exists(sdk_val_path):
        with open(sdk_val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_samples.append(json.loads(line))

    # Contamination guard: drop verbatim pair duplicates within train and across the
    # val boundary (audit finding: 218 val rows duplicated train rows verbatim).
    train_samples, val_samples, dup_train, dup_val = _dedupe_split(train_samples, val_samples)
    print(f"Contamination guard: dropped {dup_train} duplicate train rows, "
          f"{dup_val} val rows duplicating train.")

    # Shuffle datasets
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Carve the report-only TEST split out of the selection pool (audit A1: test is
    # never trained on and never used for checkpoint selection; written once here).
    n_test = int(len(val_samples) * TEST_FRACTION_OF_VAL)
    test_samples = val_samples[:n_test]
    val_samples = val_samples[n_test:]
    print(f"Split: train={len(train_samples):,} val(selection)={len(val_samples):,} test(report)={len(test_samples):,}")

    # Write output jsonl files
    train_file = os.path.join(out_dir, "train.jsonl")
    val_file = os.path.join(out_dir, "val.jsonl")
    test_file = os.path.join(out_dir, "test.jsonl")

    for path, rows in ((train_file, train_samples), (val_file, val_samples), (test_file, test_samples)):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Statistics
    train_dist = Counter(r["label"] for r in train_samples)
    val_dist = Counter(r["label"] for r in val_samples)
    train_sources = Counter(r["source"] for r in train_samples)

    print("\n" + "=" * 60)
    print("DATASET COMPILATION COMPLETE")
    print("=" * 60)
    print(f"Train samples: {len(train_samples):,} | Labels: {dict(train_dist)}")
    print(f"Val samples:   {len(val_samples):,} | Labels: {dict(val_dist)}")
    print("Top Train Sources:", train_sources.most_common(8))
    print(f"Output files:\n  - {train_file}\n  - {val_file}\n  - {test_file}")
    print("=" * 60)

    # Provenance manifest so shipped datasets are reproducible/auditable.
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "seed": seed,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "test_rows": len(test_samples),
        "duplicates_dropped_train": dup_train,
        "val_rows_duplicated_with_train_dropped": dup_val,
        "haystack_val_source_pairs": len(val_haystack_pairs),
        "per_source_train": dict(train_sources),
    }
    manifest_path = os.path.join(out_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./data", help="Output directory")
    parser.add_argument("--mode", choices=["quick", "stage2", "full"], default=None, help="Compilation mode")
    parser.add_argument("--stage2", action="store_true", help="Build Stage 2 production dataset (~370k pairs)")
    parser.add_argument("--full", action="store_true", help="Build full-scale production dataset (~650k pairs)")
    parser.add_argument("--quick", action="store_true", help="Build quick prototype dataset (~40k pairs)")
    args = parser.parse_args()

    mode = "quick"
    if args.mode:
        mode = args.mode
    elif args.stage2:
        mode = "stage2"
    elif args.full:
        mode = "full"
    elif args.quick:
        mode = "quick"

    compile_dataset(args.out_dir, mode=mode)
