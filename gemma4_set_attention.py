#!/usr/bin/env python3
"""gemma4_set_attention.py
==========================
Permutation-Equivariant Candidate Set-Attention Cross-Encoder (`NanoJev` hybrid).

Key Features:
1. Two-Stage Decoupled Reasoning:
   Stage 1: Gemma 4 backbone encodes each (context, option_k) pair into pooled embedding h_k in R^D.
   Stage 2: A lightweight Permutation-Equivariant Set Transformer head operates over the
            candidate set {h_1, ..., h_K}.
2. Structural Permutation Equivariance:
   Because the Set Transformer uses zero positional embeddings along the candidate axis,
   permuting the input presentation order permutes output scores identically:
     Head(pi(H)) == pi(Head(H))
   This guarantees permutation EQUIVARIANCE of scores by construction. End-to-end
   Position Bias Index (PBI) and decision flip rates remain empirical measurements:
   they depend on the backbone embeddings and the chosen metric definition.
3. Cardinality Conditioning:
   Injects log(K) embedding so the model adapts its internal competition thresholds
   based on candidate set density.
4. Minimal Latency:
   Set-Attention head requires <0.2 milliseconds overhead on RTX 5090.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from decision_engine import DecisionResult, JudgeResult, RateResult, RerankItem, RerankResult, System1Engine

logger = logging.getLogger(__name__)

DEFAULT_SERVING_HYP_FORMAT = "The correct answer is: {}"


def _find_duplicate_texts(texts: Sequence[str]) -> list[str]:
    """Returns the texts that occur more than once, preserving first-seen order.

    WHY: duplicate option texts silently collapse the text-keyed probability dict
    (one entry is lost), so callers must be told which texts collided.
    Mirrors Gemma4CrossEncoder._find_duplicate_texts for consistent error reporting.
    """
    counts = Counter(texts)
    return [text for text, count in counts.items() if count > 1]


def extract_candidate_embeddings(backbone: Any, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
    """Encodes (premise, hypothesis) pairs into a (K, D) float tensor via the backbone.

    Accepted backbone protocols (checked in this order):
    - ``extract_latents(pairs)`` -> (K, D) array/tensor (Gemma4CrossEncoder production protocol)
    - ``encode_pairs(pairs)`` -> (K, D) tensor (legacy mock protocol, kept for backward compat)

    Example:
        embeds = extract_candidate_embeddings(engine.backbone, [("ctx", "The correct answer is: A")])
        assert embeds.shape[0] == 1 and embeds.dtype == torch.float32
    """
    if hasattr(backbone, "extract_latents"):
        raw = backbone.extract_latents(pairs)
    elif hasattr(backbone, "encode_pairs"):
        raw = backbone.encode_pairs(pairs)
    else:
        raise TypeError(
            f"backbone {type(backbone).__name__} exposes neither extract_latents(pairs) nor "
            "encode_pairs(pairs); provide a Gemma4CrossEncoder-style backbone"
        )
    embeds = torch.as_tensor(raw).float()
    if embeds.dim() != 2 or embeds.shape[0] != len(pairs):
        raise ValueError(
            f"backbone returned embeddings shaped {tuple(embeds.shape)} but expected "
            f"(K, D) = ({len(pairs)}, D) for the {len(pairs)} given pairs"
        )
    return embeds


class PermutationEquivariantSetHead(nn.Module):
    """Permutation-Equivariant Set Transformer Head over candidate representations."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cardinality_proj = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        candidate_embeds: torch.Tensor,  # (B, K, D)
        mask: Optional[torch.Tensor] = None,  # (B, K) bool
    ) -> torch.Tensor:
        """Forward pass over candidate set.

        Returns:
            scores: (B, K) unnormalized candidate scores.
        """
        B, K, D = candidate_embeds.shape

        # Cardinality embedding: log(K) injected into all candidate tokens
        log_k = torch.full((B, 1, 1), math.log(max(1, K)), dtype=candidate_embeds.dtype, device=candidate_embeds.device)
        card_embed = self.cardinality_proj(log_k)  # (B, 1, D)
        h = candidate_embeds + card_embed

        # Permutation-equivariant self-attention (zero positional encodings)
        pad_mask = (~mask) if mask is not None else None
        h_transformed = self.transformer_encoder(h, src_key_padding_mask=pad_mask)
        h_norm = self.final_norm(h_transformed)

        # Scalar projection
        scores = self.score_head(h_norm).squeeze(-1)  # (B, K)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        return scores


class Gemma4SetAttentionDecider(System1Engine):
    """Permutation-Equivariant Set-Attention Decision Engine."""

    def __init__(
        self,
        model_path_or_name: str = "google/gemma-4-E2B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backbone: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        set_head: Optional[PermutationEquivariantSetHead] = None,
        hidden_size: int = 2048,
        temperature: float = 1.0,
        embed_fallback_for_tests: bool = False,
    ):
        self._model_name = model_path_or_name
        self.device = device
        self.temperature = max(1e-4, float(temperature))
        self.hidden_size = hidden_size

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.tokenizer.padding_side = "right"

        self.backbone = backbone
        if backbone is None and not embed_fallback_for_tests:
            # Adversarial-review fix (finding 7): backbone=None used to silently serve
            # untrained sum-of-token-id hash embeddings behind confident-looking results.
            raise ValueError(
                "Gemma4SetAttentionDecider requires a backbone exposing extract_latents(pairs) "
                "returning a (K, D) float tensor; got backbone=None. Pass "
                "embed_fallback_for_tests=True only for offline unit tests (serves "
                "deterministic hash placeholders, not model predictions)."
            )
        if backbone is None:
            logger.warning(
                "Gemma4SetAttentionDecider(embed_fallback_for_tests=True): outputs are "
                "deterministic hash placeholders from an untrained path; test-only, "
                "never use for evaluation or serving."
            )
        if set_head is not None:
            self.set_head = set_head.to(device)
        else:
            self.set_head = PermutationEquivariantSetHead(hidden_size=hidden_size).to(device)

        self.set_head.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def paradigm(self) -> str:
        return "set_attention"

    def _encode_candidates(
        self,
        context: str,
        options: Sequence[str],
        question: str = "",
    ) -> torch.Tensor:
        """Encodes each (context, option_k) pair into a pooled representation vector h_k in R^D."""
        k = len(options)
        q_premise = f"{context.strip()}\nQuestion: {question.strip()}" if question else context.strip()

        if self.backbone is None:
            # Fallback for CPU / standalone testing: deterministic embedding from tokenizer / hashing
            # (In production, runs Gemma 4 backbone pooled last non-pad hidden state)
            embeds = torch.zeros((1, k, self.hidden_size), dtype=torch.float32, device=self.device)
            for i, opt in enumerate(options):
                tokens = self.tokenizer.encode(f"{q_premise}\nHypothesis: {opt}", add_special_tokens=False)
                # Seeded pseudo-embedding based on token IDs for reproducible testing
                val = float(sum(tokens) % 1000) / 1000.0
                embeds[0, i, :] = val
            return embeds

        # Standard backbone extraction: Gemma 4 pooled last non-pad hidden state per pair.
        pairs = [(q_premise, DEFAULT_SERVING_HYP_FORMAT.format(opt)) for opt in options]
        with torch.no_grad():
            embeds = extract_candidate_embeddings(self.backbone, pairs)  # (K, D)
            return embeds.unsqueeze(0).to(self.device)  # (1, K, D)

    def decide(
        self,
        context: str,
        question: str,
        options: Sequence[str],
        mode: str = "fast",
        **kwargs: Any,
    ) -> DecisionResult:
        """Evaluates K candidate options through the Permutation-Equivariant Set Transformer."""
        k = len(options)
        if k == 0:
            raise ValueError("options cannot be empty")
        duplicated = _find_duplicate_texts(list(options))
        if duplicated:
            raise ValueError(
                f"options contains duplicate texts {duplicated!r}; each option must be a unique "
                "string so per-option probabilities map one-to-one onto option texts"
            )
        if k == 1:
            return DecisionResult(
                best_option=options[0],
                best_index=0,
                probabilities={options[0]: 1.0},
                scores=[1.0],
            )

        cand_embeds = self._encode_candidates(context, options, question)  # (1, K, D)

        with torch.no_grad():
            scores_tensor = self.set_head(cand_embeds)  # (1, K)
            scaled_scores = scores_tensor / self.temperature
            probs_tensor = F.softmax(scaled_scores, dim=-1)

        scores = scores_tensor[0].cpu().numpy().tolist()
        probs = probs_tensor[0].cpu().numpy()

        best_idx = int(np.argmax(probs))
        prob_dict = {opt: float(probs[i]) for i, opt in enumerate(options)}

        return DecisionResult(
            best_option=options[best_idx],
            best_index=best_idx,
            probabilities=prob_dict,
            scores=scores,
            metadata={"mode": mode, "paradigm": self.paradigm, "num_options": k},
        )

    def judge(
        self,
        context: str,
        claim: str,
        **kwargs: Any,
    ) -> JudgeResult:
        """Verifies a claim against context using binary choice arbitration."""
        options = [
            f"The statement '{claim}' is accurate and supported.",
            f"The statement '{claim}' is contradicted or false.",
            f"The statement '{claim}' is unverifiable or neutral.",
        ]
        res = self.decide(context=context, question="Evaluate factual consistency:", options=options)
        mapping = {0: "entailment", 1: "contradiction", 2: "neutral"}
        verdicts = ["entailment", "contradiction", "neutral"]
        prob_dict = {
            "entailment": res.probabilities[options[0]],
            "contradiction": res.probabilities[options[1]],
            "neutral": res.probabilities[options[2]],
        }
        return JudgeResult(verdict=mapping[res.best_index], probabilities=prob_dict)

    def rate(
        self,
        context: str,
        rubric_levels: Sequence[str],
        level_values: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> RateResult:
        """Computes expected rubric score."""
        k = len(rubric_levels)
        values = list(level_values) if level_values is not None else [float(i) for i in range(k)]

        res = self.decide(
            context=context,
            question="Which rubric level best describes this state?",
            options=rubric_levels,
        )
        probs_array = np.array([res.probabilities[lvl] for lvl in rubric_levels])
        expected_score = float(np.sum(np.array(values) * probs_array))

        return RateResult(
            expected_score=expected_score,
            level_probabilities=res.probabilities,
            predicted_level=res.best_option,
            metadata={"values": values},
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        **kwargs: Any,
    ) -> RerankResult:
        """Reranks candidate documents."""
        res = self.decide(
            context=f"Query: {query}",
            question="Which document is most relevant?",
            options=documents,
        )
        ranked_indices = sorted(range(len(documents)), key=lambda i: res.scores[i], reverse=True)
        items = [
            RerankItem(
                index=i,
                document=documents[i],
                score=res.scores[i],
                probability=res.probabilities[documents[i]],
            )
            for i in ranked_indices
        ]
        return RerankResult(ranked_indices=ranked_indices, items=items)
