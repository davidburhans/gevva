"""Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine & NLI Cross-Encoder.

#1 on Global JevBench Leaderboard (77.54 Composite Score).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path if not present so internal modules resolve cleanly
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decision_engine import (
    DecisionResult,
    JudgeResult,
    RateResult,
    RerankItem,
    RerankResult,
    System1Engine,
)
from gemma4_cross_encoder import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    ID2LABEL,
    LABEL2ID,
    NATIVE_MNLI2OURS,
    DEFAULT_NLI_TEMPLATE,
    Gemma4CrossEncoder,
    Gemma4ForSequenceClassification,
    GevvaCrossEncoder,
    GevvaForSequenceClassification,
    Gevva,
    OpenJevCrossEncoder,
    LatentMLPHead,
    GradeResult,
    apply_quantization_aware_training,
)

__version__ = "1.0.0"
__all__ = [
    "Gevva",
    "GevvaCrossEncoder",
    "GevvaForSequenceClassification",
    "OpenJevCrossEncoder",
    "Gemma4CrossEncoder",
    "Gemma4ForSequenceClassification",
    "LatentMLPHead",
    "System1Engine",
    "GradeResult",
    "RerankResult",
    "DecisionResult",
    "JudgeResult",
    "RateResult",
    "RerankItem",
    "CONTRADICTION",
    "ENTAILMENT",
    "NEUTRAL",
    "ID2LABEL",
    "LABEL2ID",
    "NATIVE_MNLI2OURS",
    "DEFAULT_NLI_TEMPLATE",
    "apply_quantization_aware_training",
    "load",
    "__version__",
]


def load(model_name_or_path: str = "ckpt/gevva-e2b", device: str = "auto", **kwargs) -> GevvaCrossEncoder:
    """Load a pre-trained or fine-tuned Gevva Cross-Encoder model.

    Args:
        model_name_or_path: Path to checkpoint directory or HuggingFace repo (default: 'ckpt/gevva-e2b')
        device: 'cuda', 'cpu', or 'auto' (default: 'auto')
        **kwargs: Additional arguments passed to GevvaCrossEncoder

    Returns:
        GevvaCrossEncoder instance ready for inference
    """
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return GevvaCrossEncoder(model_name_or_path=model_name_or_path, device=device, **kwargs)
