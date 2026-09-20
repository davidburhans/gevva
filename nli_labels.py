#!/usr/bin/env python3
"""nli_labels.py - Shared NLI label constants (single source of truth).

Label indexing follows `dleemiller/ModernCE-large-nli` and `AlexWortega/openjev`:
    0 = contradiction, 1 = entailment, 2 = neutral.

Usage example:
    from nli_labels import ID2LABEL, LABEL2ID
    name = ID2LABEL[1]                # "entailment"
    label = LABEL2ID["contradiction"] # 0
"""

CONTRADICTION = 0
ENTAILMENT = 1
NEUTRAL = 2

VALID_LABELS = (CONTRADICTION, ENTAILMENT, NEUTRAL)

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL2ID = {"contradiction": 0, "entailment": 1, "neutral": 2}
