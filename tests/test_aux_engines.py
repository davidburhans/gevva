#!/usr/bin/env python3
"""tests/test_aux_engines.py - Offline CPU tests for the auxiliary decision engines.

Covers adversarial-review findings 3, 7, 8, 14 for:
- gemma4_set_attention.py: backbone=None must raise instead of silently serving
  untrained hash embeddings; extract_candidate_embeddings protocol dispatch;
  full-engine permutation equivariance; duplicate option rejection.
- gemma4_causal_decider.py: robust-mode un-permutation regression (same option
  under every cyclic shift); k > 20 key limit; duplicate option and empty claim
  rejection.

Run: CUDA_VISIBLE_DEVICES= uv run python tests/test_aux_engines.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine import DecisionResult, JudgeResult
from gemma4_causal_decider import STANDARD_OPTION_KEYS, Gemma4CausalDecider
from gemma4_set_attention import Gemma4SetAttentionDecider, extract_candidate_embeddings

VOCAB_SIZE = 256  # char ids are ord(c)+100; 'z' -> 222, so 256 is a safe ceiling


class CharLevelTokenizer:
    """Deterministic mock tokenizer where ' A'/' B'/' C' encode to distinct single ids.

    Pattern: MockTokenizer in tests/test_system1_framework.py. Non-space characters
    map to ord(char)+100, so ' A' -> [165], ' B' -> [166], ' C' -> [167].
    """

    padding_side: str = "right"
    pad_token_id: int = 0
    eos_token_id: int = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [ord(char) + 100 for char in text if not char.isspace()]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(token_id - 100) for token_id in ids)

    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class KeyBiasCausalLM(nn.Module):
    """Fake causal LM: the ' B' decision token always receives the highest logit.

    WHY: a prompt-independent key bias cannot discriminate un-permutation direction
    (any bijective spread of a constant profile averages to a uniform distribution),
    but it DOES regress the missing-un-permutation bug where all key-B mass collapses
    onto a single original index instead of spreading across the option list.
    """

    def __init__(self, winner_key: str = "B", vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.logits_table = torch.zeros(vocab_size)
        self.logits_table[ord(winner_key) + 100] = 8.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        seq_len = input_ids.shape[1]
        logits = self.logits_table.view(1, 1, -1).expand(1, seq_len, self.vocab_size)
        return SimpleNamespace(logits=logits)


class OptionAwareCausalLM(nn.Module):
    """Fake causal LM whose decision logits favor the key slot holding `winner_option`.

    WHY: key preference follows option CONTENT, so robust-mode results must point at
    the same option text under every cyclic shift of the presentation order. A wrongly
    directed un-permutation would spread the winner mass across other options and fail.
    """

    def __init__(
        self,
        winner_option: str,
        keys: Sequence[str] = ("A", "B", "C"),
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.winner_option = winner_option
        self.keys = tuple(keys)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        # CharLevelTokenizer drops whitespace, so prompt lines decode to e.g. 'B:beta'.
        prompt = "".join(chr(token_id - 100) for token_id in input_ids[0].tolist())
        logits = torch.zeros(self.vocab_size)
        for key in self.keys:
            if f"{key}:{self.winner_option}" in prompt:
                logits[ord(key) + 100] = 8.0
                break
        seq_len = input_ids.shape[1]
        return SimpleNamespace(logits=logits.view(1, 1, -1).expand(1, seq_len, self.vocab_size))


class FakeLatentBackbone:
    """Deterministic backbone: row k depends only on pair k's content (torch tensor out).

    WHY content-derived rows: permuting the option list must permute the embedding
    rows, which is what makes the full-engine equivariance assertion meaningful.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.call_count: int = 0

    def extract_latents(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        self.call_count += 1
        rows = []
        for premise, hypothesis in pairs:
            seed = sum(ord(char) for char in premise + hypothesis) % (2**31)
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.rand(self.dim, generator=generator))
        return torch.stack(rows)  # (K, D)


class NumpyLatentBackbone:
    """extract_latents backbone returning a numpy (K, D) array (Gemma4CrossEncoder shape)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def extract_latents(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        rows = [np.full(self.dim, float(index) + 1.0) for index in range(len(pairs))]
        return np.stack(rows)


class LegacyEncodePairsBackbone:
    """Backbone exposing only the legacy encode_pairs protocol (backward compat)."""

    def encode_pairs(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        return torch.ones((len(pairs), 4))


class ProtocolLessBackbone:
    """Backbone exposing neither accepted protocol -> must trigger TypeError."""


class BadShapeLatentBackbone:
    """extract_latents backbone returning a wrong row count -> must trigger ValueError."""

    def extract_latents(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        return torch.zeros((max(0, len(pairs) - 1), 8))


def build_set_attention_engine(**kwargs: object) -> Gemma4SetAttentionDecider:
    """Builds an offline set-attention engine with small dims and injected fakes."""
    params: Dict[str, object] = {
        "model_path_or_name": "offline-fake",
        "tokenizer": CharLevelTokenizer(),
        "hidden_size": 32,
        "device": "cpu",
    }
    params.update(kwargs)
    return Gemma4SetAttentionDecider(**params)  # type: ignore[arg-type]


def build_causal_decider(model: nn.Module) -> Gemma4CausalDecider:
    """Builds an offline causal decider around an injected fake LM."""
    return Gemma4CausalDecider(
        model_path_or_name="offline-fake",
        model=model,
        tokenizer=CharLevelTokenizer(),
        device="cpu",
    )


class TestSetAttentionSafety(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(1234)  # reproducible untrained head init (F.I.R.S.T: repeatable)

    def test_backbone_none_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_set_attention_engine()
        message = str(ctx.exception)
        self.assertIn("extract_latents", message, "error must name the accepted protocol")
        self.assertIn("backbone=None", message)

    def test_fallback_mode_warns_and_returns_offline_result(self) -> None:
        with self.assertLogs("gemma4_set_attention", level="WARNING") as logs:
            engine = build_set_attention_engine(embed_fallback_for_tests=True)
        self.assertTrue(
            any("placeholder" in line for line in logs.output),
            f"warning must say outputs are test-only placeholders, got {logs.output}",
        )
        options = ["alpha", "beta", "gamma"]
        res = engine.decide(context="ctx", question="q?", options=options)
        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(sorted(res.probabilities), sorted(options))
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertIn(res.best_option, options)
        self.assertEqual(res.best_index, options.index(res.best_option))

    def test_extract_latents_backbone_full_engine_equivariance(self) -> None:
        backbone = FakeLatentBackbone(dim=32)
        engine = build_set_attention_engine(backbone=backbone)
        options = ["alpha", "beta", "gamma"]
        res = engine.decide(context="ctx", question="q?", options=options)
        self.assertIsInstance(res, DecisionResult)
        self.assertGreaterEqual(backbone.call_count, 1)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)

        # Full-engine equivariance: permuting the option list permutes the probabilities.
        res_permuted = engine.decide(context="ctx", question="q?", options=["gamma", "alpha", "beta"])
        for option in options:
            self.assertAlmostEqual(
                res.probabilities[option],
                res_permuted.probabilities[option],
                delta=1e-5,
                msg=f"probability for {option!r} must follow the option under permutation",
            )
        self.assertEqual(res.best_option, res_permuted.best_option)
        self.assertEqual(
            res.best_option,
            options[int(np.argmax([res.probabilities[option] for option in options]))],
            "best_option must be the argmax of the text-keyed probabilities",
        )

    def test_set_attention_duplicate_options_raise(self) -> None:
        engine = build_set_attention_engine(embed_fallback_for_tests=True)
        with self.assertRaises(ValueError) as ctx:
            engine.decide(context="ctx", question="q?", options=["alpha", "beta", "alpha"])
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception))

    def test_set_attention_empty_options_raise(self) -> None:
        engine = build_set_attention_engine(embed_fallback_for_tests=True)
        with self.assertRaises(ValueError):
            engine.decide(context="ctx", question="q?", options=[])


class TestExtractCandidateEmbeddings(unittest.TestCase):

    def test_dispatches_to_extract_latents_with_torch_tensor(self) -> None:
        backbone = FakeLatentBackbone(dim=8)
        pairs = [("ctx", "The correct answer is: A"), ("ctx", "The correct answer is: B")]
        embeds = extract_candidate_embeddings(backbone, pairs)
        self.assertEqual(tuple(embeds.shape), (2, 8))
        self.assertEqual(embeds.dtype, torch.float32)
        self.assertEqual(backbone.call_count, 1)

    def test_dispatches_to_extract_latents_with_numpy_array(self) -> None:
        pairs = [("ctx", "a"), ("ctx", "b"), ("ctx", "c")]
        embeds = extract_candidate_embeddings(NumpyLatentBackbone(dim=8), pairs)
        self.assertEqual(tuple(embeds.shape), (3, 8))
        self.assertEqual(embeds.dtype, torch.float32)

    def test_keeps_legacy_encode_pairs_protocol(self) -> None:
        pairs = [("ctx", "a"), ("ctx", "b")]
        embeds = extract_candidate_embeddings(LegacyEncodePairsBackbone(), pairs)
        self.assertEqual(tuple(embeds.shape), (2, 4))

    def test_protocol_less_backbone_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            extract_candidate_embeddings(ProtocolLessBackbone(), [("ctx", "a")])
        message = str(ctx.exception)
        self.assertIn("extract_latents", message)
        self.assertIn("encode_pairs", message)
        self.assertIn("ProtocolLessBackbone", message)

    def test_wrong_row_count_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            extract_candidate_embeddings(BadShapeLatentBackbone(), [("ctx", "a"), ("ctx", "b")])
        self.assertIn("(K, D)", str(ctx.exception))
        self.assertIn("(2, D)", str(ctx.exception))


class TestCausalDecider(unittest.TestCase):

    def test_fast_mode_picks_key_b_option(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM(winner_key="B"))
        options = ["alpha", "beta", "gamma"]
        res = engine.decide(context="ctx", question="q?", options=options, mode="fast")
        self.assertIsInstance(res, DecisionResult)
        self.assertEqual(res.best_index, 1)
        self.assertEqual(res.best_option, "beta")
        self.assertGreater(res.probabilities["beta"], 0.9)

    def test_robust_mode_spreads_constant_key_bias(self) -> None:
        # Regression for the un-permutation math: key-B mass must be redistributed to
        # every original index, not collapsed onto the option sitting at slot B.
        engine = build_causal_decider(KeyBiasCausalLM(winner_key="B"))
        options = ["alpha", "beta", "gamma"]
        res = engine.decide(context="ctx", question="q?", options=options, mode="robust")
        self.assertIsInstance(res, DecisionResult)
        for option in options:
            self.assertAlmostEqual(res.probabilities[option], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertLess(res.probabilities["beta"], 0.5, "un-permutation must not collapse key-B mass")

    def test_robust_mode_selects_same_option_under_every_cyclic_shift(self) -> None:
        engine = build_causal_decider(OptionAwareCausalLM(winner_option="beta"))
        base = ["alpha", "beta", "gamma"]
        fast = engine.decide(context="ctx", question="q?", options=base, mode="fast")
        self.assertEqual(fast.best_option, "beta", "fast mode must pick the key-B option")
        first = engine.decide(context="ctx", question="q?", options=base, mode="robust")
        self.assertEqual(first.best_option, "beta")
        for shift in range(1, len(base)):
            rotated = base[shift:] + base[:shift]
            res = engine.decide(context="ctx", question="q?", options=rotated, mode="robust")
            self.assertEqual(
                res.best_option,
                first.best_option,
                f"robust mode must select the same option under cyclic shift {shift}",
            )
            self.assertEqual(res.best_option, "beta")
            self.assertGreater(res.probabilities["beta"], 0.9)

    def test_k_21_raises_value_error_naming_limit(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM())
        options = [f"option {index:02d}" for index in range(21)]
        with self.assertRaises(ValueError) as ctx:
            engine.decide(context="ctx", question="q?", options=options)
        message = str(ctx.exception)
        self.assertIn("21", message, "error must include the offending k")
        self.assertIn(str(len(STANDARD_OPTION_KEYS)), message, "error must include the key limit")

    def test_causal_duplicate_options_raise(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM())
        with self.assertRaises(ValueError) as ctx:
            engine.decide(context="ctx", question="q?", options=["alpha", "beta", "alpha"])
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception))

    def test_causal_empty_options_raise(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM())
        with self.assertRaises(ValueError):
            engine.decide(context="ctx", question="q?", options=[])

    def test_causal_empty_claim_raises(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM())
        with self.assertRaises(ValueError) as ctx:
            engine.judge(context="ctx", claim="")
        self.assertIn("claim", str(ctx.exception))
        self.assertIn("non-empty", str(ctx.exception))

    def test_causal_judge_returns_valid_distribution(self) -> None:
        engine = build_causal_decider(KeyBiasCausalLM(winner_key="A"))
        res = engine.judge(context="ctx", claim="The sky is blue.")
        self.assertIsInstance(res, JudgeResult)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertIn(res.verdict, ("entailment", "contradiction", "neutral"))


if __name__ == "__main__":
    unittest.main()
