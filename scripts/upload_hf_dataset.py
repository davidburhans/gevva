#!/usr/bin/env python3
"""scripts/upload_hf_dataset.py
================================
Uploads the staged Gevva decision datasets to Hugging Face Hub: `davidburhans/gevva-decisions`.
"""

import os
import subprocess
import sys
from huggingface_hub import HfApi


def main():
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        # Try extracting from ~/.bash_profile
        try:
            token = subprocess.check_output(
                "bash -l -c 'echo $HF_WRITE_TOKEN'", shell=True, text=True
            ).strip()
        except Exception:
            token = None

    if not token:
        print("ERROR: HF_WRITE_TOKEN environment variable not found.")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = "davidburhans/gevva-decisions"
    staging_dir = "data/hf_dataset_staging"

    print(f"Creating / verifying dataset repository '{repo_id}' on Hugging Face...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    print(f"Repository '{repo_id}' ready.")

    print(f"Uploading files from '{staging_dir}' to '{repo_id}'...")
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=staging_dir,
        repo_type="dataset",
        commit_message="Initial release of Gevva Decisions training curriculum and benchmarks",
    )
    print("Upload completed successfully!")
    print(f"Commit URL: {commit_info}")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
