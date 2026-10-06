#!/usr/bin/env python3
"""export_gevva_v2_gguf.py - GGUF Exporter for Gevva v2 Native Decision Engine.

Packages fine-tuned Gemma 4 causal decision models into production GGUF format
configured for stock `llama-server /v1/systemone` using the `lev` protocol.

Metadata Injected:
- `gemma4.decision.type = "lev"`
- `gemma4.decision.temperature.choice.small = 0.85`  (K <= 8)
- `gemma4.decision.temperature.choice.mid = 1.10`    (K <= 26)
- `gemma4.decision.temperature.choice.large = 1.35`  (K <= 255)
- `gemma4.decision.temperature.noul = 1.55`
- `gemma4.decision.temperature.score = 0.90`
- `systemone` chat template formatted for Gemma 4 turn structure.

Usage:
    uv run python scripts/export_gevva_v2_gguf.py --model-dir ckpt/gevva-v2-e2b --out gevva-v2-e2b-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Embedded Gemma 4 systemone Jinja chat template matching the lev decision protocol
GEMMA4_SYSTEMONE_TEMPLATE = (
    "<start_of_turn>user\n"
    "{% for image in images %}{{ image }}{% endfor %}"
    "{% if images %}The image shows the input visual scene.\n\n{% endif %}"
    "State:\n"
    "{{ state if state is string else state | tojson(indent=2) }}\n\n"
    "Question: {{ instructions }}\n"
    "{% if type == 'score' %}Rate along the ordered levels below (lowest first).\n{% endif %}"
    "Options:\n"
    "{% for o in options %}"
    "[{{ o.label }}] {{ o.key }}{% if o.description %}: {{ o.description }}{% endif %}\n"
    "{% endfor %}\n"
    "Answer with the option code only.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def inject_decision_metadata_to_config(model_dir: Path, template: str = GEMMA4_SYSTEMONE_TEMPLATE) -> None:
    """Prepares the checkpoint directory so standard llama.cpp convert tools write the required decision keys."""
    # 1. Update config.json with decision parameters
    config_path = model_dir / "config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg["architectures"] = ["Gemma4ForCausalLM"]
        cfg["decision"] = {
            "type": "lev",
            "max_options": 255,
            "temperatures": {
                "choice.small": 0.85,
                "choice.mid": 1.10,
                "choice.large": 1.35,
                "noul": 1.55,
                "score": 0.90,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated config.json with Gevva v2 `lev` decision configuration.")

    # 2. Write chat_template.jinja with systemone template
    template_path = model_dir / "chat_template.jinja"
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"Written systemone chat template to {template_path}")

    # 3. Update tokenizer_config.json with chat_template
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tok_cfg["chat_template"] = template
        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2)
        print(f"Updated tokenizer_config.json with systemone chat template.")


def main():
    parser = argparse.ArgumentParser(description="Export Gevva v2 model for native llama.cpp serving.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Input directory containing model.safetensors and config.json")
    parser.add_argument("--out", type=Path, default=None, help="Output .gguf path (optional).")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Preparing Gevva v2 Checkpoint for GGUF Export ===")
    print(f"Target Checkpoint: {model_dir}")

    inject_decision_metadata_to_config(model_dir)

    print("\nCheckpoint is fully formatted for Gevva v2 / llama.cpp `lev` decision protocol!")
    print("\nTo generate the final GGUF file using llama.cpp:")
    print(f"    python convert_hf_to_gguf.py {model_dir} --outtype f16 --outfile {args.out or (model_dir.name + '.gguf')}")
    print("\nTo serve with unmodified llama-server:")
    print(f"    llama-server -m {args.out or (model_dir.name + '.gguf')} --port 8080")


if __name__ == "__main__":
    main()
