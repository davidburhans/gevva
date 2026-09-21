#!/usr/bin/env python3
"""data_engine.py
==================
Canonical Decision Data Engine and Architecture-Tailored Collators.

Key Capabilities:
1. Canonical Representation (`CanonicalDecision`):
   Unified representation of K-way decisions, tool routing, and policy arbitration.
2. Cyclic Permutation Engine:
   Rotates candidate option presentation orders to scale training data by Kx
   and teach models to be position- and token-invariant.
3. Architecture-Tailored Collators:
   - `InContextPermutationCollator`: For Causal LMs (state-first layout, slot-targeted CE).
   - `GroupedDecisionCollator`: For Cross-Encoders (atomic K-option batches, listwise loss).
   - `SetAttentionCollator`: For Set-Transformers (candidate sets with adjacency masks).
4. Deterministic Token-Bucket Samplers:
   Bounds maximum tokens per batch to prevent OOM across variable sequence lengths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

STANDARD_OPTION_KEYS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
]

# Upper bound on options per decision: matches the standard key inventory above.
MAX_STANDARD_OPTIONS = len(STANDARD_OPTION_KEYS)

DEFAULT_SERVING_HYP_FORMAT = "The correct answer is: {}"

CONTRADICTION = 0
ENTAILMENT = 1
NEUTRAL = 2


def _deterministic_decision_id(d: Dict[str, Any]) -> str:
    """Content-hash id for decisions missing one (stable across runs and processes).

    Example:
        >>> _deterministic_decision_id({"question": "q"})
        'decision_0fb9abe146'
    """
    # WHY: replaces random.randint — random ids broke resumable queues and
    # dedup joins because the same record got a different id on every run.
    payload = json.dumps(d, sort_keys=True, default=str)
    return f"decision_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


# -----------------------------------------------------------------------------
# 1. Canonical Decision Item
# -----------------------------------------------------------------------------
@dataclass
class CanonicalDecision:
    """Canonical representation of a single K-way decision scenario."""
    id: str
    domain: str
    context: str
    question: str
    options: List[Dict[str, str]]  # [{"key": "A", "text": "Option text..."}]
    gold_index: int
    gold_key: Optional[str] = None
    soft_distribution: Optional[List[float]] = None
    is_trap: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # WHY: strict validation up front — a bare IndexError on
        # STANDARD_OPTION_KEYS[gold_index] or silent key aliasing for K > 20 used
        # to surface far away from the malformed record that caused it.
        k = len(self.options)
        if not 1 <= k <= MAX_STANDARD_OPTIONS:
            raise ValueError(
                f"CanonicalDecision(id={self.id!r}) requires 1 <= num_options <= "
                f"{MAX_STANDARD_OPTIONS}, got num_options={k} for options={[str(o) for o in self.options][:3]}"
            )
        option_texts = [str(o.get("text", "")) for o in self.options]
        duplicate_texts = sorted({t for t in option_texts if option_texts.count(t) > 1})
        if duplicate_texts:
            raise ValueError(
                f"CanonicalDecision(id={self.id!r}) requires unique option texts, "
                f"got duplicates={duplicate_texts}"
            )
        if not 0 <= self.gold_index < k:
            raise ValueError(
                f"CanonicalDecision(id={self.id!r}) gold_index={self.gold_index} is outside "
                f"the valid range [0, {k - 1}] for num_options={k}"
            )
        if self.gold_key is None or self.gold_key == "":
            # Derive from the gold option itself; gold_index is bounds-checked above
            # so the STANDARD_OPTION_KEYS fallback can never overflow.
            self.gold_key = self.options[self.gold_index].get("key", STANDARD_OPTION_KEYS[self.gold_index])

    @property
    def num_options(self) -> int:
        return len(self.options)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "context": self.context,
            "question": self.question,
            "options": self.options,
            "gold_index": self.gold_index,
            "gold_key": self.gold_key,
            "soft_distribution": self.soft_distribution,
            "is_trap": self.is_trap,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalDecision":
        opts_raw = d.get("options", [])
        formatted_opts = []
        for i, opt in enumerate(opts_raw):
            if isinstance(opt, dict):
                k = opt.get("key", STANDARD_OPTION_KEYS[min(i, len(STANDARD_OPTION_KEYS) - 1)])
                txt = opt.get("text", opt.get("desc", ""))
                formatted_opts.append({"key": str(k), "text": str(txt)})
            else:
                k = STANDARD_OPTION_KEYS[min(i, len(STANDARD_OPTION_KEYS) - 1)]
                formatted_opts.append({"key": k, "text": str(opt)})

        return cls(
            id=str(d.get("id", _deterministic_decision_id(d))),
            domain=str(d.get("domain", "general")),
            context=str(d.get("context", d.get("premise", ""))),
            question=str(d.get("question", "")),
            options=formatted_opts,
            gold_index=int(d.get("gold_index", 0)),
            gold_key=d.get("gold_key"),
            soft_distribution=d.get("soft_distribution"),
            is_trap=bool(d.get("is_trap", False)),
            metadata=d.get("metadata", {}),
        )

    def shifted(self, shift: int) -> "CanonicalDecision":
        """Single cyclic shift of the options, without building the other K-1 permutations.

        Semantically identical to ``to_cyclic_permutations()[shift]``. Hot paths
        that only need one random shift per item (InContextPermutationCollator)
        use this to avoid materializing all K fully-validated permutations —
        an O(K^2) allocation cost per batch item.

        Example:
            >>> dec = CanonicalDecision(id="d", domain="t", context="c", question="q",
            ...     options=[{"key": "A", "text": "a"}, {"key": "B", "text": "b"}], gold_index=0)
            >>> dec.shifted(1).gold_key
            'B'
        """
        k = self.num_options
        if not 0 <= shift < k:
            raise ValueError(
                f"shift={shift} is outside the valid range [0, {k - 1}] for "
                f"num_options={k} (decision id={self.id!r})"
            )
        # Rotated option texts
        rotated_raw = self.options[shift:] + self.options[:shift]
        new_opts = [
            {"key": STANDARD_OPTION_KEYS[min(i, len(STANDARD_OPTION_KEYS) - 1)], "text": opt["text"]}
            for i, opt in enumerate(rotated_raw)
        ]

        new_gold_idx = (self.gold_index - shift) % k
        new_gold_key = STANDARD_OPTION_KEYS[min(new_gold_idx, len(STANDARD_OPTION_KEYS) - 1)]

        new_soft = None
        if self.soft_distribution and len(self.soft_distribution) == k:
            new_soft = self.soft_distribution[shift:] + self.soft_distribution[:shift]

        return CanonicalDecision(
            id=f"{self.id}_perm_{shift}",
            domain=self.domain,
            context=self.context,
            question=self.question,
            options=new_opts,
            gold_index=new_gold_idx,
            gold_key=new_gold_key,
            soft_distribution=new_soft,
            is_trap=self.is_trap,
            metadata={**self.metadata, "permutation_shift": shift, "original_id": self.id},
        )

    def to_cyclic_permutations(self, num_shifts: Optional[int] = None) -> List["CanonicalDecision"]:
        """Generates cyclic permutations of the options list.

        For K options, generates K cyclic shifts:
          Shift 0: (o_0, o_1, ..., o_{K-1})
          Shift 1: (o_1, o_2, ..., o_0)
          ...
        The underlying ground-truth option remains identical, but its assigned
        key ('A', 'B', ...) and spatial slot changes.
        """
        if num_shifts is not None and num_shifts < 1:
            raise ValueError(
                f"num_shifts={num_shifts} is invalid for decision id={self.id!r}; "
                f"expected num_shifts >= 1 or None"
            )
        k = self.num_options
        if k <= 1:
            return [self]

        max_shifts = k if num_shifts is None else min(num_shifts, k)
        return [self.shifted(shift) for shift in range(max_shifts)]

    def format_in_context_prompt(self, include_answer_slot: bool = True) -> Tuple[str, str]:
        """Formats the decision into a State-First Causal LM prompt.

        Returns:
            (prompt_text, gold_key_char)
        """
        lines = []
        if self.context.strip():
            lines.append(f"Context: {self.context.strip()}")
        if self.question.strip():
            lines.append(f"Question: {self.question.strip()}")
        lines.append("Options:")
        for opt in self.options:
            lines.append(f"{opt['key']}: {opt['text'].strip()}")

        if include_answer_slot:
            lines.append("Decision: ")
        prompt_text = "\n".join(lines)
        return prompt_text, self.gold_key or "A"

    def to_grouped_pairs(
        self,
        serving_template: str = DEFAULT_SERVING_HYP_FORMAT,
    ) -> List[Dict[str, Any]]:
        """Flattens the decision into K candidate (premise, hypothesis) pairs for Cross-Encoders.

        All K pairs share the same `group_id`, allowing the GroupedDecisionCollator
        to score them in the same forward pass for listwise competition.
        """
        q_premise = f"{self.context.strip()}\nQuestion: {self.question.strip()}" if self.question else self.context.strip()
        k = self.num_options
        pairs = []

        for idx, opt in enumerate(self.options):
            is_gold = (idx == self.gold_index)
            hyp = serving_template.format(f"{opt['key']}: {opt['text']}")
            p_soft = 1.0 if is_gold else 0.0
            if self.soft_distribution and len(self.soft_distribution) == k:
                p_soft = float(self.soft_distribution[idx])

            pairs.append({
                "id": f"{self.id}_opt_{opt['key']}",
                "premise": q_premise,
                "hypothesis": hyp,
                "label": ENTAILMENT if is_gold else CONTRADICTION,
                "group_id": self.id,
                "is_gold": is_gold,
                "soft_target": p_soft,
                "source": "canonical_decision_grouped",
                "metadata": {
                    "canonical_id": self.id,
                    "option_key": opt["key"],
                    "gold_index": self.gold_index,
                    "is_trap": self.is_trap,
                },
            })
        return pairs


# -----------------------------------------------------------------------------
# 2. In-Context Causal Collator (for system-one-open paradigm)
# -----------------------------------------------------------------------------
class InContextPermutationCollator:
    """Collator for In-Context Causal Decision Models (`system-one-open` paradigm).

    Key Features:
    1. State-First Layout:
       Context -> Question -> Options (A, B, C, D) -> Decision: [SLOT]
    2. Dynamic Permutation Augmentation (Training):
       If `permute_training=True`, randomly rotates option ordering across cyclic
       shifts during training, giving free Kx data scaling and bias destruction.
    3. Strict Slot-Token Masking:
       Labels for all context, question, and option tokens are set to -100.
       Loss is evaluated exclusively on the single decision index token ('A', 'B', ...).

    Permutation randomness comes from an instance-seeded RNG (`seed`), reseeded
    per epoch via `set_epoch(epoch)` as `seed * 1000003 + epoch`. NOTE: with
    DataLoader `num_workers > 0` every worker inherits the same collator state,
    so callers should offset the seed per worker (e.g. `seed + worker_id`) to
    decorrelate worker permutation streams.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 4096,
        pad_to_multiple_of: int = 8,
        permute_training: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.permute_training = permute_training
        self._seed = seed
        self._epoch = 0
        # WHY: seeded instance RNG instead of the global `random` module — every
        # DataLoader worker inherits an identical global python random state,
        # which made all workers emit correlated permutations.
        self._rng = random.Random(self._seed * 1000003 + self._epoch)

        # Ensure right padding for causal training
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        # Pre-cache token IDs for candidate keys "A", "B", "C", "D"...
        self.key_to_token_id: Dict[str, int] = {}
        for k in STANDARD_OPTION_KEYS:
            t_ids = self.tokenizer.encode(f" {k}", add_special_tokens=False)
            if not t_ids:
                t_ids = self.tokenizer.encode(k, add_special_tokens=False)
            self.key_to_token_id[k] = t_ids[-1]

    def set_epoch(self, epoch: int) -> None:
        """Reseeds the permutation RNG for a fresh but reproducible epoch stream."""
        self._epoch = epoch
        self._rng = random.Random(self._seed * 1000003 + self._epoch)

    def __call__(self, batch: List[Union[CanonicalDecision, Dict[str, Any]]]) -> Dict[str, Any]:
        parsed_items: List[CanonicalDecision] = []
        for item in batch:
            if isinstance(item, CanonicalDecision):
                parsed_items.append(item)
            elif isinstance(item, dict):
                parsed_items.append(CanonicalDecision.from_dict(item))

        batch_input_ids = []
        batch_labels = []
        batch_gold_indices = []
        batch_option_token_ids = []

        for dec in parsed_items:
            # If permuting during training, pick a random cyclic shift
            if self.permute_training and dec.num_options > 1:
                shift = self._rng.randint(0, dec.num_options - 1)
                # WHY: build only the randomly chosen shift; to_cyclic_permutations()
                # materializes all K fully-validated permutations per item just to
                # index one, an O(K^2) allocation hot path in the training loop.
                active_dec = dec.shifted(shift)
            else:
                active_dec = dec

            prompt_text, gold_char = active_dec.format_in_context_prompt(include_answer_slot=True)
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)

            # Target decision token
            target_token_id = self.key_to_token_id.get(gold_char, self.key_to_token_id["A"])

            # Full sequence: prompt + target_token_id
            full_ids = prompt_ids + [target_token_id]
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if len(full_ids) > self.max_length:
                # Truncate from the left of prompt, preserving the ending options + slot
                overflow = len(full_ids) - self.max_length
                full_ids = full_ids[overflow:]
                prompt_len = max(0, len(prompt_ids) - overflow)
                # WHY: left-truncation slices off the BOS anchor token that causal
                # models expect at position 0; re-prepend it within the same
                # max_length budget. Skipped when the gold slot already sits at
                # index 0 (prompt_len == 0) so the supervised token is never evicted.
                if bos_token_id is not None and prompt_len > 0 and full_ids[0] != bos_token_id:
                    full_ids = [bos_token_id] + full_ids[1:]
            else:
                prompt_len = len(prompt_ids)

            # Mask all prompt tokens with -100 so loss evaluates only on decision slot
            labels = [-100] * len(full_ids)
            if len(full_ids) > prompt_len:
                labels[prompt_len] = target_token_id

            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            batch_gold_indices.append(active_dec.gold_index)

            # Sliced token IDs for the options present in this question
            cand_token_ids = [
                self.key_to_token_id.get(opt["key"], self.key_to_token_id["A"])
                for opt in active_dec.options
            ]
            batch_option_token_ids.append(cand_token_ids)

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of > 0 and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        padded_ids = []
        padded_labels = []
        attn_masks = []

        for ids, lbls in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "gold_indices": torch.tensor(batch_gold_indices, dtype=torch.long),
            "option_token_ids": batch_option_token_ids,
        }


# -----------------------------------------------------------------------------
# 3. Set-Attention Collator (for latent candidate set transformer)
# -----------------------------------------------------------------------------
class SetAttentionCollator:
    """Collator for Candidate Set-Attention Cross-Encoders (`NanoJev` hybrid).

    Batches premise-hypothesis pairs while maintaining question-level groupings
    and producing candidate adjacency masks for the permutation-equivariant
    Set Transformer head.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer.padding_side = "right"
        # WHY: built once here instead of per __call__ — re-instantiating the base
        # collator (and its Gemma image processor) on every batch repeated setup
        # work on every training step.
        from research.adapters.grouped_decision_collator import GroupedDecisionCollator

        self._base_collator = GroupedDecisionCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = self._base_collator(batch)

        # Build set adjacency mask: mask[i, j] = 1 if group_ids[i] == group_ids[j] >= 0 else 0
        gids = collated["group_ids"]
        if not bool((gids >= 0).any()):
            # WHY: an all-ungrouped batch makes the adjacency mask all-False; an
            # nn.TransformerEncoder consuming a fully-masked src_key_padding_mask
            # emits NaN scores downstream, so the set-attention head requires at
            # least one grouped decision / candidate set per batch.
            raise ValueError(
                "SetAttentionCollator requires at least one grouped decision "
                "(group_id >= 0) per batch for the set-attention head; got "
                f"group_ids={gids.tolist()}"
            )
        valid_groups = gids >= 0
        adj_mask = (gids.unsqueeze(1) == gids.unsqueeze(0)) & (valid_groups.unsqueeze(1) & valid_groups.unsqueeze(0))

        collated["set_adjacency_mask"] = adj_mask.to(dtype=torch.bool)
        return collated


# -----------------------------------------------------------------------------
# 4. Canonical Dataset Loader
# -----------------------------------------------------------------------------
class CanonicalDecisionDataset(Dataset):
    """Loads a JSONL file of CanonicalDecision items or converts heterogeneous formats."""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.items: List[CanonicalDecision] = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset path not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if not isinstance(d, dict) or not isinstance(d.get("options"), list):
                    # WHY: a pairwise NLI file (premise/hypothesis/label, e.g. data/test.jsonl)
                    # used to crash inside CanonicalDecision validation with a bare
                    # num_options=0 error that never named the offending file or line.
                    raise ValueError(
                        f"{jsonl_path}, line {line_idx}: record lacks an 'options' list; "
                        "expected canonical decision schema {'id', 'domain', 'context', 'question', "
                        "'options': [{'key', 'text'}, ...], 'gold_index', 'gold_key', ...}; pairwise "
                        "NLI records (premise/hypothesis/label) are not canonical decisions, got "
                        f"keys={sorted(d) if isinstance(d, dict) else type(d).__name__!r}"
                    )
                self.items.append(CanonicalDecision.from_dict(d))
                if max_samples is not None and len(self.items) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.items)} CanonicalDecision records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> CanonicalDecision:
        return self.items[idx]
