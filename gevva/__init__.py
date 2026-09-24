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
    "DEFAULT_MODEL_ID",
    "__version__",
]


DEFAULT_MODEL_ID = "davidburhans/gevva-e2b"


def load(model_name_or_path: str = DEFAULT_MODEL_ID, device: str = "auto", **kwargs) -> GevvaCrossEncoder:
    """Load a pre-trained or fine-tuned Gevva Cross-Encoder model.

    Args:
        model_name_or_path: Path to local checkpoint directory or HuggingFace repo ID
            (default: 'davidburhans/gevva-e2b'). If the default repo is requested and a local
            'ckpt/gevva-e2b' directory exists, the local checkpoint is utilized.
        device: 'cuda', 'cpu', or 'auto' (default: 'auto')
        **kwargs: Additional arguments passed to GevvaCrossEncoder

    Returns:
        GevvaCrossEncoder instance ready for inference
    """
    import os
    if model_name_or_path == DEFAULT_MODEL_ID and not os.path.exists(DEFAULT_MODEL_ID):
        if os.path.isdir("ckpt/gevva-e2b"):
            model_name_or_path = "ckpt/gevva-e2b"

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return GevvaCrossEncoder(model_name_or_path=model_name_or_path, device=device, **kwargs)
