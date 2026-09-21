#!/usr/bin/env python3
"""gemma4_causal_decider.py
===========================
In-Context Causal Decision Engine (`system-one-open` / `SemIf` paradigm) on Gemma 4.

Key Features:
1. Single-Forward-Pass Joint Competition:
   Prompt layout: Context -> Question -> Options (A, B, C, D) -> Decision: [SLOT]
   Decision distribution extracted via constrained vocabulary softmax over index tokens:
     P(opt_i) = exp(z_i / tau) / sum_j exp(z_j / tau)
2. Dual Operating Modes:
   - `mode="fast"`: Single forward pass (~18-25ms latency).
   - `mode="robust"`: Cyclic permutation ensemble (evaluates K cyclic shifts,
     un-permutes candidate probabilities, and averages them). This yields invariance
     to cyclic ROTATIONS of the option order; arbitrary permutations are NOT
     guaranteed to be invariant.
3. Full `System1Engine` Implementation:
   Implements `decide`, `judge`, `rate`, and `rerank`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_engine import DecisionResult, JudgeResult, RateResult, RerankItem, RerankResult, System1Engine

logger = logging.getLogger(__name__)

STANDARD_OPTION_KEYS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
]


def _find_duplicate_texts(texts: Sequence[str]) -> list[str]:
    """Returns the texts that occur more than once, preserving first-seen order.

    WHY: duplicate option texts silently collapse the text-keyed probability dict
    (one entry is lost), so callers must be told which texts collided.
    Mirrors Gemma4CrossEncoder._find_duplicate_texts for consistent error reporting.
    """
    counts = Counter(texts)
    return [text for text, count in counts.items() if count > 1]


class Gemma4CausalDecider(System1Engine):
    """In-Context Causal Decision Engine on Gemma 4."""

    def __init__(
        self,
        model_path_or_name: str = "google/gemma-4-E2B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        temperature: float = 1.0,
    ):
        self._model_name = model_path_or_name
        self.device = device
        self.temperature = max(1e-4, float(temperature))

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        if model is not None:
            self.model = model.to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_name,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
            )
            if device == "cpu":
                self.model = self.model.to("cpu")

        self.model.eval()

        # Pre-cache candidate option token IDs: " A", " B", " C", " D"...
        self.option_token_ids: Dict[str, int] = {}
        for key in STANDARD_OPTION_KEYS:
            t_ids = self.tokenizer.encode(f" {key}", add_special_tokens=False)
            if not t_ids:
                t_ids = self.tokenizer.encode(key, add_special_tokens=False)
            self.option_token_ids[key] = t_ids[-1]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def paradigm(self) -> str:
        return "in_context_causal"

    def _format_prompt(
        self,
        context: str,
        question: str,
        options: Sequence[str],
        keys: Sequence[str],
    ) -> str:
        lines = []
        if context.strip():
            lines.append(f"Context: {context.strip()}")
        if question.strip():
            lines.append(f"Question: {question.strip()}")
        lines.append("Options:")
        for k, opt in zip(keys, options):
            lines.append(f"{k}: {opt.strip()}")
        lines.append("Decision: ")
        return "\n".join(lines)

    @torch.no_grad()
    def _evaluate_single_prompt(
        self,
        prompt: str,
        active_keys: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates a single prompt and returns normalized probabilities and raw logits across active_keys."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Final token's vocabulary logits
        last_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Extract logits for active option keys
        key_token_ids = [self.option_token_ids[k] for k in active_keys]
        option_logits = last_token_logits[key_token_ids].float()

        # Scaled softmax
        scaled_logits = option_logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
        raw_scores = option_logits.cpu().numpy()

        return probs, raw_scores

    def decide(
        self,
        context: str,
        question: str,
        options: Sequence[str],
        mode: str = "fast",
        **kwargs: Any,
    ) -> DecisionResult:
        """K-way choice evaluation with 'fast' (single pass) or 'robust' (cyclic ensemble) modes.

        'robust' averages over the K cyclic shifts of the option list, which gives
        invariance to cyclic ROTATIONS of the presentation order (not arbitrary
        permutations). Probabilities are keyed one-to-one onto unique option texts.
        """
        k = len(options)
        if k == 0:
            raise ValueError("options cannot be empty")
        duplicated = _find_duplicate_texts(list(options))
        if duplicated:
            raise ValueError(
                f"options contains duplicate texts {duplicated!r}; each option must be a unique "
                "string so per-option probabilities map one-to-one onto option texts"
            )
        if k > len(STANDARD_OPTION_KEYS):
            # Adversarial-review fix (finding 8): k > 20 used to truncate the key list
            # and then crash while building the text-keyed probability dict.
            raise ValueError(
                f"options has length {k} but only {len(STANDARD_OPTION_KEYS)} single-letter "
                f"option keys ({STANDARD_OPTION_KEYS[0]}-{STANDARD_OPTION_KEYS[-1]}) exist; "
                "shorten the option list or extend STANDARD_OPTION_KEYS"
            )
        if k == 1:
            return DecisionResult(
                best_option=options[0],
                best_index=0,
                probabilities={options[0]: 1.0},
                scores=[1.0],
            )

        keys = STANDARD_OPTION_KEYS[:k]

        if mode == "robust":
            # Cyclic Permutation Ensemble: evaluates all K cyclic shifts to eliminate order bias
            accum_probs = np.zeros(k, dtype=np.float64)
            accum_scores = np.zeros(k, dtype=np.float64)

            for shift in range(k):
                # Rotated options
                shifted_options = list(options[shift:]) + list(options[:shift])
                prompt = self._format_prompt(context, question, shifted_options, keys)
                p_shifted, s_shifted = self._evaluate_single_prompt(prompt, keys)

                # Un-permute probabilities and scores back to original indexing
                # Index i in shifted_options corresponds to original index (i + shift) % k
                for shifted_idx in range(k):
                    orig_idx = (shifted_idx + shift) % k
                    accum_probs[orig_idx] += p_shifted[shifted_idx]
                    accum_scores[orig_idx] += s_shifted[shifted_idx]

            final_probs = accum_probs / k
            final_scores = (accum_scores / k).tolist()
        else:
            # Fast mode: single forward pass in presentation order
            prompt = self._format_prompt(context, question, options, keys)
            p_single, s_single = self._evaluate_single_prompt(prompt, keys)
            final_probs = p_single
            final_scores = s_single.tolist()

        best_idx = int(np.argmax(final_probs))
        prob_dict = {opt: float(final_probs[i]) for i, opt in enumerate(options)}

        return DecisionResult(
            best_option=options[best_idx],
            best_index=best_idx,
            probabilities=prob_dict,
            scores=final_scores,
            metadata={"mode": mode, "num_options": k},
        )

    def judge(
        self,
        context: str,
        claim: str,
        **kwargs: Any,
    ) -> JudgeResult:
        """Evaluates whether a claim is Entailed, Contradicted, or Neutral."""
        if not claim:
            raise ValueError(f"claim must be a non-empty string, got {claim!r}")
        prompt = (
            f"Context: {context.strip()}\n"
            f"Claim: {claim.strip()}\n"
            "Options:\n"
            "A: The claim is definitely true and supported by the context.\n"
            "B: The claim is false or contradicts the context.\n"
            "C: The claim is unrelated, neutral, or unverifiable.\n"
            "Decision: "
        )
        keys = ["A", "B", "C"]
        probs, _ = self._evaluate_single_prompt(prompt, keys)
        prob_dict = {
            "entailment": float(probs[0]),
            "contradiction": float(probs[1]),
            "neutral": float(probs[2]),
        }
        verdicts = ["entailment", "contradiction", "neutral"]
        best_verdict = verdicts[int(np.argmax(probs))]
        return JudgeResult(verdict=best_verdict, probabilities=prob_dict)

    def rate(
        self,
        context: str,
        rubric_levels: Sequence[str],
        level_values: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> RateResult:
        """Computes continuous expected score over ordered rubric levels."""
        k = len(rubric_levels)
        if level_values is None:
            values = [float(i) for i in range(k)]
        else:
            values = list(level_values)

        res = self.decide(
            context=context,
            question="Which rubric level best describes this state?",
            options=rubric_levels,
            mode=kwargs.get("mode", "fast"),
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
        """Reranks candidate documents by relevance to query."""
        res = self.decide(
            context=f"Query: {query}",
            question="Which document is most relevant and responsive to the query?",
            options=documents,
            mode=kwargs.get("mode", "fast"),
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
