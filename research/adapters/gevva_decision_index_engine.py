#!/usr/bin/env python3
"""gevva_decision_index_engine.py
==================================
Native Gevva Engine Adapter for the Hugging Face Jev Decision Index
(multimodalart/jev-decision-index & github.com/apolinario/decision-index).

Implements the official Decision Index protocol:
- Supports `choice` multiple-choice questions with full probability distributions
- Supports `noul` (yes/no / binary) questions with calibrated true-probabilities
- Seamlessly evaluates both Gevva E2B and Gevva E4B Flagship models
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Ensure local repository modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    Gemma4CrossEncoder,
)


def _text(x: Any) -> str:
    if x is None or x == "" or x == {} or x == []:
        return ""
    if isinstance(x, str):
        return x.strip()
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    arr = np.array(logits, dtype=np.float64) / max(temperature, 1e-4)
    exp = np.exp(arr - np.max(arr))
    probs = exp / np.sum(exp)
    # Ensure numerical sum is strictly 1.0
    probs = probs / np.sum(probs)
    return probs.tolist()


class GevvaDecisionIndexEngine:
    """Official Gevva Decision Index Engine adapter compatible with decision_index."""

    name = "gevva"
    latency = "Device-synchronized in-process request wall time including NLI formatting and cross-encoder inference."

    def __init__(
        self,
        model_name_or_path: str = "ckpt/gevva-e2b",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_length: int = 4096,
        scoring: str = "margin",
        temperature: float = 1.0,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, dtype, torch.bfloat16)
        self.model_name_or_path = model_name_or_path
        self.scoring = scoring
        self.temperature = temperature

        self.cross_encoder = Gemma4CrossEncoder(
            model_name_or_path=model_name_or_path,
            device=self.device,
            dtype=self.torch_dtype,
            max_length=max_length,
            batch_size=16,
        )

        self.provenance = {
            "kind": "gevva_cross_encoder",
            "model": model_name_or_path,
            "device": self.device,
            "dtype": dtype,
            "scoring": scoring,
            "temperature": temperature,
            "architecture": "Gemma 4 Multimodal System 1 Decision Engine",
        }

    def warmup(self):
        """Standard Decision Index warmup sequence."""
        warm = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        self("The color is red.", warm)

    def runtime(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Processes a state and question dict according to the Decision Index protocol.

        Args:
            state: Text context, string, or structured object
            questions: Dictionary mapping question_id -> question_dict

        Returns:
            {"answers": {q_id: answer_dict}}
        """
        state_str = _text(state)
        answers: Dict[str, Dict[str, Any]] = {}

        for q_id, q_dict in questions.items():
            q_type = q_dict.get("type", "choice")
            instructions = _text(q_dict.get("instructions", ""))
            criteria = q_dict.get("criteria", {})

            if q_type == "choice":
                # Multiple-choice decision
                keys = list(criteria.keys())
                if not keys:
                    raise ValueError(f"Question '{q_id}' has no criteria options.")

                # Format premise
                if state_str and instructions:
                    premise = f"{state_str}\n\nQuestion: {instructions}"
                elif state_str:
                    premise = state_str
                else:
                    premise = instructions

                # Evaluate each option as candidate hypothesis
                pairs = []
                for k in keys:
                    desc = _text(criteria[k])
                    if desc and desc.lower() != str(k).lower():
                        hyp = f"The correct answer is {k}: {desc}."
                    else:
                        hyp = f"The correct answer is: {k}."
                    pairs.append((premise, hyp))

                # Batch scoring
                if self.scoring in ("log_odds", "logit_margin"):
                    logits = self.cross_encoder.predict_logits(pairs)
                    margins = (logits[:, ENTAILMENT] - logits[:, CONTRADICTION]).tolist()
                elif self.scoring == "contrastive":
                    probs = self.cross_encoder.predict(pairs)
                    margins = (probs[:, ENTAILMENT] / (probs[:, ENTAILMENT] + probs[:, CONTRADICTION] + 1e-6)).tolist()
                elif self.scoring == "entailment":
                    probs = self.cross_encoder.predict(pairs)
                    margins = probs[:, ENTAILMENT].tolist()
                else:  # margin
                    probs = self.cross_encoder.predict(pairs)
                    margins = (probs[:, ENTAILMENT] - probs[:, CONTRADICTION]).tolist()

                probs_list = _softmax(margins, temperature=self.temperature)
                prob_map = {k: float(p) for k, p in zip(keys, probs_list)}

                # Argmax choice
                best_key = keys[int(np.argmax(probs_list))]

                answers[q_id] = {
                    "type": "choice",
                    "choice": best_key,
                    "probabilities": prob_map,
                }

            elif q_type == "noul":
                # Binary yes/no question
                if state_str and instructions:
                    premise = state_str
                    hypothesis = instructions
                elif state_str:
                    premise = state_str
                    hypothesis = "The claim is true."
                else:
                    premise = instructions
                    hypothesis = "The statement is true."

                probs = self.cross_encoder.predict([(premise, hypothesis)])
                p_con, p_ent, p_neu = probs[0]

                # Calibrated probability that the statement is true
                if p_ent + p_con > 1e-6:
                    p_true = float(p_ent / (p_ent + p_con))
                else:
                    p_true = float(p_ent)

                p_true = max(0.0, min(1.0, p_true))

                answers[q_id] = {
                    "type": "noul",
                    "noul": p_true,
                }
            else:
                raise ValueError(f"Unsupported question type: {q_type}")

        return {"answers": answers}
