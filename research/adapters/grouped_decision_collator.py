#!/usr/bin/env python3
"""Grouped Decision Collator and Cross-Option Loss for System 1 Decision Models.

Implements P1 (Cross-Option Softmax Objective) and P2 (Serving-Template Parity)
for NLI cross-encoders.

Key Innovations:
1. Grouped Batching:
   Maintains question-level groupings so that all K candidate options for a decision
   question are scored together in the same forward pass.
2. Cross-Option Softmax Loss (P1):
   Softmax across candidate option entailment logits:
     P(option_k | q) = exp(s_k / tau) / sum_j exp(s_j / tau)
   Cross-entropy against hard gold option index or soft teacher distribution.
3. Dual-Head Optimization:
   Unifies cross-option competition with auxiliary 3-class NLI anchor loss (P6)
   and proper multi-class Brier calibration loss (H1).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Sampler

from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    Gemma4ImageProcessorPil,
    tokenize_nli_pair_safe,
)

DEFAULT_SERVING_HYP_FORMAT = "The correct answer is: {}"


def compute_cross_option_loss(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    is_gold: torch.Tensor,
    soft_targets: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes cross-option softmax cross-entropy across grouped decision options (P1).

    Args:
        scores: (N,) tensor of candidate option scores (e.g. each pair's entailment
                logit z_ent). Extraction from (N, 3) NLI heads happens upstream in
                the model, not here.
        group_ids: (N,) integer tensor where items sharing group_id >= 0 belong to the same question.
                   Un-grouped items (e.g. clean NLI pairs) have group_id == -1.
        is_gold: (N,) bool or float tensor indicating the true winning candidate option.
        soft_targets: Optional (N,) float tensor with teacher probabilities for each option.
        temperature: Scalar temperature for softmax competition (default: 1.0).

    Returns:
        total_loss: Scalar cross-option cross-entropy loss (with autograd graph preserved).
        metrics: Dict with 'xopt_loss', 'xopt_accuracy', 'n_groups'.
    """
    valid_mask = group_ids >= 0
    if not valid_mask.any():
        # No grouped items in batch: return zero loss attached to scores graph
        return scores.sum() * 0.0, {"xopt_loss": 0.0, "xopt_accuracy": 0.0, "n_groups": 0}

    unique_groups = torch.unique(group_ids[valid_mask])
    group_losses = []
    correct_groups = 0
    evaluated_groups = 0

    temp = max(1e-4, float(temperature))

    for gid in unique_groups:
        idx = torch.nonzero(group_ids == gid, as_tuple=False).squeeze(-1)
        k = len(idx)
        if k <= 1:
            # Need at least 2 options for cross-option competition
            continue

        grp_scores = scores[idx] / temp
        grp_is_gold = is_gold[idx]

        # Max logit subtraction for numeric stability
        log_probs = F.log_softmax(grp_scores, dim=-1)

        # Check accuracy for logging
        pred_idx = torch.argmax(grp_scores).item()
        if grp_is_gold[pred_idx].item() > 0.5:
            correct_groups += 1
        evaluated_groups += 1

        if soft_targets is not None:
            grp_soft = soft_targets[idx]
            # If soft targets sum to > 0, use soft cross-entropy
            s_sum = grp_soft.sum()
            if s_sum > 0:
                normalized_soft = grp_soft / s_sum
                loss_g = -torch.sum(normalized_soft * log_probs)
            else:
                # Fallback to hard gold
                gold_idx = torch.argmax(grp_is_gold.float())
                loss_g = -log_probs[gold_idx]
        else:
            gold_idx = torch.argmax(grp_is_gold.float())
            loss_g = -log_probs[gold_idx]

        group_losses.append(loss_g)

    if not group_losses:
        return scores.sum() * 0.0, {"xopt_loss": 0.0, "xopt_accuracy": 0.0, "n_groups": 0}

    mean_loss = torch.stack(group_losses).mean()
    acc = correct_groups / max(1, evaluated_groups)

    return mean_loss, {
        "xopt_loss": mean_loss.item(),
        "xopt_accuracy": acc,
        "n_groups": evaluated_groups,
    }


class GroupedDecisionCollator:
    """Collate premise-hypothesis pairs while maintaining question-level grouping metadata.

    Handles mixed batches of:
    1. Grouped decision items (K options sharing a common `group_id`).
    2. Standard pairwise NLI items (MNLI, SNLI, ANLI, FEVER) with no group_id.
    3. Multimodal visual NLI items (with image file paths).
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
        image_processor: Optional[Any] = None,
        image_root: str = "./data",
        is_gemma: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.image_root = image_root
        self.tokenizer.padding_side = "right"
        if image_processor is None and is_gemma:
            try:
                self.image_processor = Gemma4ImageProcessorPil()
            except Exception:
                self.image_processor = None
        else:
            self.image_processor = image_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = []
        pixel_values_list = []
        image_pos_ids_list = []

        raw_group_ids = []
        is_gold_list = []
        soft_targets_list = []
        labels_list = []
        sources_list = []

        # Remap string/int group_ids into compact batch-local integers [0, G-1]
        group_map: Dict[str, int] = {}
        next_gid = 0

        for r in batch:
            gid_raw = r.get("group_id")
            if gid_raw is not None and str(gid_raw).strip() != "":
                gid_key = str(gid_raw)
                if gid_key not in group_map:
                    group_map[gid_key] = next_gid
                    next_gid += 1
                batch_gid = group_map[gid_key]
            else:
                batch_gid = -1
            raw_group_ids.append(batch_gid)

            # Extract gold flag and soft targets
            is_gold_val = bool(r.get("is_gold", r.get("label") == ENTAILMENT))
            is_gold_list.append(1.0 if is_gold_val else 0.0)

            soft_p = r.get("soft_target", r.get("teacher_prob", None))
            if soft_p is not None:
                soft_targets_list.append(float(soft_p))
            else:
                soft_targets_list.append(1.0 if is_gold_val else 0.0)

            labels_list.append(int(r.get("label", ENTAILMENT if is_gold_val else CONTRADICTION)))
            sources_list.append(r.get("source", "custom"))

            # Handle multimodal images if present
            img_field = r.get("image")
            resolved_path = None
            if img_field and self.image_processor is not None:
                p_with_root = os.path.join(self.image_root, img_field)
                if os.path.exists(p_with_root):
                    resolved_path = p_with_root
                elif os.path.exists(img_field):
                    resolved_path = img_field

            if resolved_path is not None:
                img = Image.open(resolved_path).convert("RGB")
                feat = self.image_processor(img, return_tensors="pt")
                n_soft = int(feat["num_soft_tokens_per_image"][0])
                ids = tokenize_nli_pair_safe(
                    tokenizer=self.tokenizer,
                    premise=r["premise"],
                    hypothesis=r["hypothesis"],
                    max_length=self.max_length,
                    image_soft_tokens=n_soft,
                )
                pixel_values_list.append(feat["pixel_values"][0])
                image_pos_ids_list.append(feat["image_position_ids"][0])
            else:
                ids = tokenize_nli_pair_safe(
                    tokenizer=self.tokenizer,
                    premise=r["premise"],
                    hypothesis=r["hypothesis"],
                    max_length=self.max_length,
                    image_soft_tokens=0,
                )
            batch_input_ids.append(ids)

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of > 0 and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_ids = []
        attn_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        out = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "group_ids": torch.tensor(raw_group_ids, dtype=torch.long),
            "is_gold": torch.tensor(is_gold_list, dtype=torch.float32),
            "soft_targets": torch.tensor(soft_targets_list, dtype=torch.float32),
            "sources": sources_list,
        }

        if pixel_values_list:
            out["pixel_values"] = torch.stack(pixel_values_list)
            out["image_position_ids"] = torch.stack(image_pos_ids_list)

        return out


class GroupedTokenBucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that keeps all options for each question group together.

    Ensures that for any question group g, all candidate option indices {i_1, ..., i_K}
    land in the EXACT same batch, while respecting total token limits (max_tokens_per_batch).

    Batch composition is fixed once in __init__ (deterministic greedy packing);
    with shuffle=True only the batch ORDER varies per epoch (set_epoch), so
    len() always equals the number of batches __iter__ yields.
    """

    def __init__(
        self,
        records_or_lengths: Union[Sequence[Dict[str, Any]], Sequence[int]],
        lengths_or_records: Optional[Union[Sequence[int], Sequence[Dict[str, Any]]]] = None,
        group_ids: Optional[Sequence[Any]] = None,
        max_tokens_per_batch: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        # Flexible argument resolution
        if isinstance(records_or_lengths[0], dict):
            self.records = records_or_lengths
            self.lengths = lengths_or_records if lengths_or_records is not None else [512] * len(self.records)
            raw_gids = [r.get("group_id") for r in self.records]
        else:
            self.lengths = records_or_lengths
            self.records = lengths_or_records if (lengths_or_records is not None and isinstance(lengths_or_records[0], dict)) else []
            raw_gids = group_ids if group_ids is not None else [r.get("group_id") for r in self.records] if self.records else []

        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Step 1: Cluster indices into atomic scheduling units (groups or singletons)
        self.units: List[List[int]] = []
        group_to_indices: Dict[str, List[int]] = defaultdict(list)
        singletons: List[int] = []

        for idx, gid in enumerate(raw_gids):
            if gid is not None and str(gid).strip() != "" and str(gid) != "-1":
                group_to_indices[str(gid)].append(idx)
            else:
                singletons.append(idx)

        # Catch any indices if raw_gids was shorter than lengths
        for idx in range(len(raw_gids), len(self.lengths)):
            singletons.append(idx)

        for gid, indices in group_to_indices.items():
            self.units.append(indices)

        for s_idx in singletons:
            self.units.append([s_idx])

        # Precompute deterministic batch composition once; iteration only reorders
        # these batches (shuffle seed), so __len__ always matches what __iter__ yields.
        self._batches: List[List[int]] = self._pack_units(self.units)

    def _unit_max_length(self, unit: List[int]) -> int:
        """Max sequence length inside one atomic scheduling unit (group or singleton)."""
        return max(self.lengths[i] for i in unit)

    def _pack_units(self, units: List[List[int]]) -> List[List[int]]:
        """Greedy token-bucket packing of atomic units into batches.

        Units are sorted by descending max length (tie-break: first index) so the
        composition is independent of dict/insertion order. A unit whose own token
        footprint exceeds max_tokens_per_batch still gets its own batch: group
        atomicity outranks the token budget.
        """
        ordered_units = sorted(units, key=lambda u: (-self._unit_max_length(u), u[0]))
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for unit in ordered_units:
            unit_len = self._unit_max_length(unit)
            cand_max_len = max(current_max_len, unit_len)
            cand_count = len(current_batch) + len(unit)
            cand_total_tokens = cand_max_len * cand_count

            if current_batch and cand_total_tokens > self.max_tokens_per_batch:
                # Flush current batch
                batches.append(current_batch)
                current_batch = list(unit)
                current_max_len = unit_len
            else:
                current_batch.extend(unit)
                current_max_len = cand_max_len

        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
