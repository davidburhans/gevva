#!/usr/bin/env python3
"""decision_engine.py
======================
Unified runtime contract and data models for System 1 Decision Engines.

Provides a single standard interface (`System1Engine`) implemented across
all model paradigms:
1. `Gemma4CrossEncoder` (Pointwise & Grouped Listwise Cross-Encoder)
2. `Gemma4CausalDecider` (In-Context Causal Next-Token Decider)
3. `Gemma4SetAttentionDecider` (Permutation-Equivariant Set Transformer Head)

Every engine supports dual operating modes:
- `mode="fast"`: Minimal latency, single-pass forward evaluation (10-25ms).
- `mode="robust"`: Maximum quality, permutation-debiased / cyclic-ensembled evaluation (25-80ms).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DecisionResult:
    """Standard result for a K-way categorical choice."""
    best_option: str
    best_index: int
    probabilities: Dict[str, float]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_confidence(self) -> float:
        return self.probabilities.get(self.best_option, 0.0)


@dataclass
class JudgeResult:
    """Standard result for binary or 3-class fact/policy verification."""
    verdict: str  # "entailment", "contradiction", or "neutral"
    probabilities: Dict[str, float]  # {"entailment": p1, "contradiction": p2, "neutral": p3}
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_entailed(self) -> bool:
        return self.verdict == "entailment"


@dataclass
class RateResult:
    """Standard result for ordinal rubric grading and expected value scoring."""
    expected_score: float  # sum_i(value_i * P(level_i))
    level_probabilities: Dict[str, float]
    predicted_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankItem:
    """Single item in a reranking result."""
    index: int
    document: str
    score: float
    probability: float


@dataclass
class RerankResult:
    """Ranked list of documents or candidates."""
    ranked_indices: List[int]
    items: List[RerankItem]
    metadata: Dict[str, Any] = field(default_factory=dict)


class System1Engine(ABC):
    """Abstract Base Class for all System 1 Decision Engines."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model or checkpoint."""
        ...

    @property
    @abstractmethod
    def paradigm(self) -> str:
        """Architectural paradigm: 'cross_encoder', 'in_context_causal', or 'set_attention'."""
        ...

    @abstractmethod
    def decide(
        self,
        context: str,
        question: str,
        options: Sequence[str],
        mode: str = "fast",
        **kwargs: Any,
    ) -> DecisionResult:
        """Select the best candidate option given a context and question.

        Args:
            context: The premise, document, policy text, or evidence.
            question: The decision query, intent, or instruction.
            options: List of K candidate actions, tools, or multiple-choice answers.
            mode: "fast" (single pass) or "robust" (permutation-averaged / cyclic ensembled).

        Returns:
            DecisionResult with chosen option, index, probabilities, and scores.
        """
        ...

    @abstractmethod
    def judge(
        self,
        context: str,
        claim: str,
        **kwargs: Any,
    ) -> JudgeResult:
        """Verify whether a declarative claim is supported, contradicted, or unverifiable.

        Args:
            context: The premise or source text.
            claim: The hypothesis or assertion to verify.

        Returns:
            JudgeResult with verdict and calibrated probability distribution.
        """
        ...

    @abstractmethod
    def rate(
        self,
        context: str,
        rubric_levels: Sequence[str],
        level_values: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> RateResult:
        """Grade a state against ordered rubric criteria using continuous expected value.

        Args:
            context: The text or work product being graded.
            rubric_levels: Descriptions of ordered performance levels [Level 0, Level 1, ...].
            level_values: Numeric values for each level (default: 0.0, 1.0, 2.0, ...).

        Returns:
            RateResult with continuous expected score and per-level probabilities.
        """
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank candidate passages or documents by relevance / entailment to query."""
        ...
