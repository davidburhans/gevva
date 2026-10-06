#!/usr/bin/env python3
"""scripts/publish_e2b_to_hf.py
==============================
Publishes the Gevva 1.1 E2B distill checkpoint to the Hugging Face Hub: `davidburhans/gevva-e2b`.
Uses HF_WRITE_TOKEN from the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        print("Error: HF_WRITE_TOKEN environment variable not set.")
        sys.exit(1)

    repo_id = "davidburhans/gevva-e2b"
    checkpoint_dir = Path("ckpt/gevva-e2b-distill/best")

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Ensuring repository exists: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # Write model card README.md into checkpoint dir from docs/HUGGINGFACE_MODEL_CARD.md
    model_card_src = Path("docs/HUGGINGFACE_MODEL_CARD.md")
    if not model_card_src.exists():
        print(f"Error: Model card source not found at {model_card_src}")
        sys.exit(1)

    readme_content = model_card_src.read_text(encoding="utf-8")
    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Written model card to {readme_path}")

    print(f"Uploading {checkpoint_dir} to https://huggingface.co/{repo_id}...")
    upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Release Gevva 1.1 e2b Distill Decision Engine",
    )
    print(f"Successfully published Gevva 1.1 e2b: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
