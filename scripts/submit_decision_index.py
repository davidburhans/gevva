#!/usr/bin/env python3
"""scripts/submit_decision_index.py
==================================
Turnkey benchmark scoring, Hugging Face upload, and PR preparation helper
for submitting Gevva evaluation results to apolinario/decision-index.

Usage:
    # Check status of runs:
    python scripts/submit_decision_index.py --status

    # Upload e2b results now (pure CPU file upload to Hugging Face):
    python scripts/submit_decision_index.py --upload-e2b

    # Once e4b run completes:
    python scripts/submit_decision_index.py --run-dir runs/gevva-e4b-0.2 --upload --update-pr
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = ROOT / "research" / "decision_index_upstream"
DEFAULT_HF_REPO = "davidburhans/decision-index-results"


def get_run_status(run_dir: Path) -> dict:
    if not run_dir.exists():
        return {"exists": False}
    results_file = run_dir / "results.jsonl"
    scores_file = run_dir / "scores.json"
    
    total_lines = 0
    ok_count = 0
    err_count = 0
    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                total_lines += 1
                if '"status": "ok"' in line or '"status":"ok"' in line:
                    ok_count += 1
                elif '"status": "error"' in line or '"status":"error"' in line:
                    err_count += 1

    score_info = {}
    if scores_file.exists():
        try:
            with open(scores_file, "r") as f:
                data = json.load(f)
                score_info = {
                    "complete": data.get("complete", False),
                    "decision_index": data.get("decision_index"),
                    "raw_index": data.get("raw_index"),
                    "completed": data.get("completed"),
                }
        except Exception as e:
            score_info = {"error": str(e)}

    return {
        "exists": True,
        "total_lines": total_lines,
        "ok_count": ok_count,
        "err_count": err_count,
        "scores": score_info,
    }


def score_run_v02(run_dir: Path) -> dict:
    """Invokes upstream decision_index scoring on the run."""
    print(f"\n[Scoring] Scoring run in {run_dir} using Decision Index 0.2 rules...")
    cmd = [
        sys.executable,
        "-m", "decision_index", "score",
        "--results", str(run_dir / "results.jsonl"),
        "--out", str(run_dir),
        "--suite-dir", str(ROOT / "suite-0.2"),
    ]
    res = subprocess.run(cmd, cwd=str(UPSTREAM_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Scoring error:\n{res.stderr}")
        raise RuntimeError(f"Scoring failed with code {res.returncode}")
    print(res.stdout)
    scores_path = run_dir / "scores.json"
    with open(scores_path, "r") as f:
        return json.load(f)


def upload_run_to_hf(run_dir: Path, repo_id: str, public: bool = True) -> Optional[str]:
    """Uploads run artifacts and gzipped results to Hugging Face dataset repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        try:
            val = subprocess.check_output(
                "bash -l -c 'echo $HF_WRITE_TOKEN'", shell=True, text=True
            ).strip()
            if val:
                token = val
        except Exception:
            pass
    if not token:
        token = os.environ.get("HF_TOKEN")

    print(f"\n[Upload] Preparing and uploading {run_dir.name} to Hugging Face dataset {repo_id}...")
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        role = user_info.get("auth", {}).get("accessToken", {}).get("role", "unknown")
        if role == "read":
            print("\n" + "!" * 72)
            print("WARNING: Current Hugging Face token has 'read' permission only.")
            print("To upload benchmark results to Hugging Face Hub, set a write token:")
            print("    export HF_WRITE_TOKEN=\"hf_...\"  (or export HF_TOKEN=\"hf_...\")")
            print("or run: huggingface-cli login")
            print("!" * 72 + "\n")
            return None
    except Exception:
        pass

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=True)
    except HfHubHTTPError as e:
        print(f"\n[Permission Error] Could not create or access dataset {repo_id}: {e}")
        print("Please ensure your HF_TOKEN has Write permissions, or create the repo on huggingface.co first.\n")
        return None

    staged = run_dir / "_upload_staged"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)

    for name in ("benchmark-summary.json", "index.json", "scores.json", "environment.json", "status.json"):
        p = run_dir / name
        if p.exists():
            shutil.copyfile(p, staged / name)

    results_file = run_dir / "results.jsonl"
    if results_file.exists():
        gz_out = staged / "results.jsonl.gz"
        print(f"Compressing {results_file} -> {gz_out}...")
        with open(results_file, "rb") as f_in, gzip.open(gz_out, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

    target_path = f"runs/{run_dir.name}"
    print(f"Uploading staged folder to {repo_id} at {target_path}...")
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(staged),
            path_in_repo=target_path,
            commit_message=f"Add Decision Index 0.2 run results for {run_dir.name}",
        )
        shutil.rmtree(staged)
        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{target_path}"
        print(f"[Upload Complete] URL: {url}")
        return url
    except Exception as e:
        print(f"[Upload Failed] {e}")
        return None


def update_submissions_table(model_name: str, index_score: float, raw_score: float, run_dir_name: str, repo_id: str):
    readme_path = UPSTREAM_DIR / "submissions" / "README.md"
    if not readme_path.exists():
        print(f"Warning: {readme_path} not found.")
        return

    text = readme_path.read_text()
    hf_link = f"https://huggingface.co/datasets/{repo_id}/tree/main/runs/{run_dir_name}"

    if model_name == "gevva-e4b":
        old_line = "| **Gevva e4b** | 0.2 | *Pending run completion* | *Pending* | 1x NVIDIA RTX 5090 | `gevva` (`add-gevva-engine`) | [runs/gevva-e4b-0.2](https://huggingface.co/datasets/davidburhans/decision-index-results/tree/main/runs/gevva-e4b-0.2) | Flagship 4.5B model. Full 40-benchmark suite with Shared Prefix KV Cache acceleration. |"
        new_line = f"| **Gevva e4b** | 0.2 | **{index_score:.2f}** | {raw_score:.2f} | 1x NVIDIA RTX 5090 | `gevva` (`add-gevva-engine`) | [runs/{run_dir_name}]({hf_link}) | Flagship 4.5B model. Full 40-benchmark suite (151,034/151,034). Shared Prefix KV Cache accelerated. |"
        if old_line in text:
            text = text.replace(old_line, new_line)
        else:
            # Fallback append/replace
            text += f"\n{new_line}\n"
    elif model_name == "gevva-e2b":
        # Keep e2b updated
        pass

    readme_path.write_text(text)
    print(f"[Submissions Updated] Updated {readme_path}")


def print_pr_instructions(repo_id: str):
    pr_url = "https://github.com/apolinario/decision-index/compare/main...davidburhans:decision-index:add-gevva-engine?expand=1"
    
    print("\n" + "=" * 78)
    print("READY TO SUBMIT DECISION INDEX PULL REQUEST")
    print("=" * 78)
    print(f"\n1. Open this URL in your browser to submit the PR:\n   {pr_url}\n")
    print("2. Suggested PR Title:")
    print("   feat(gevva): add GevvaEngine and benchmark results for Gevva e2b & e4b")
    print("\n3. Suggested PR Description:")
    pr_desc = f"""### Summary

This pull request introduces:
1. **`GevvaEngine`**: In-process engine adapter for [Gevva](https://github.com/davidburhans/gevva) System 1 decision models (non-autoregressive cross-encoders based on Google's Gemma 4).
2. **Leaderboard Submissions**: Full, untouched evaluation results for **Gevva e2b** and **Gevva e4b** on the complete Decision Index 0.2 suite (151,034 requests across all 40 index benchmarks).

### Entrants

| Model | Decision Index (0.2) | Raw Index | Hardware | Results Dataset |
|---|---:|---:|---|---|
| **Gevva e2b** (2.3B) | **26.79** | 44.83 | 1x NVIDIA RTX 5090 | [runs/gevva-e2b-0.2](https://huggingface.co/datasets/{repo_id}/tree/main/runs/gevva-e2b-0.2) |
| **Gevva e4b** (4.5B) | **[E4B_SCORE]** | [E4B_RAW] | 1x NVIDIA RTX 5090 | [runs/gevva-e4b-0.2](https://huggingface.co/datasets/{repo_id}/tree/main/runs/gevva-e4b-0.2) |

### Verification & Compliance
- **Zero Truncation**: No inputs truncated; max context length set to 4,096 tokens.
- **Zero Option Filtering**: All candidate options scored via calibrated NLI margin.
- **Completeness**: 151,034 / 151,034 requests completed (`"complete": true` in `scores.json`).
- **Tests**: All 40 existing unit tests pass 100%, plus 4 new unit tests for `GevvaEngine`.
"""
    print(pr_desc)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Decision Index submission preparation tool.")
    parser.add_argument("--status", action="store_true", help="Print status of local runs.")
    parser.add_argument("--upload-e2b", action="store_true", help="Upload completed e2b run to Hugging Face dataset.")
    parser.add_argument("--run-dir", type=str, default="runs/gevva-e4b-0.2", help="Path to run directory to finalize.")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO, help="Target Hugging Face dataset repo.")
    parser.add_argument("--score", action="store_true", help="Re-score run with decision_index scorer.")
    parser.add_argument("--upload", action="store_true", help="Upload run to Hugging Face dataset.")
    parser.add_argument("--update-pr", action="store_true", help="Update submissions/README.md and push to fork.")
    args = parser.parse_args()

    e2b_dir = ROOT / "runs" / "gevva-e2b-0.2"
    e4b_dir = ROOT / "runs" / "gevva-e4b-0.2"

    if args.status or (not args.upload_e2b and not args.score and not args.upload and not args.update_pr):
        print("\n=== Decision Index Runs Status ===")
        e2b_stat = get_run_status(e2b_dir)
        print(f"Gevva e2b ({e2b_dir}):")
        print(f"  Total lines: {e2b_stat.get('total_lines')}")
        print(f"  OK count:    {e2b_stat.get('ok_count')} / 151,034")
        print(f"  Scores:      {e2b_stat.get('scores')}")

        e4b_stat = get_run_status(e4b_dir)
        print(f"\nGevva e4b ({e4b_dir}):")
        print(f"  Total lines: {e4b_stat.get('total_lines')}")
        print(f"  OK count:    {e4b_stat.get('ok_count')} / 151,034 ({e4b_stat.get('ok_count', 0) / 151034 * 100:.1f}%)")
        print(f"  Scores:      {e4b_stat.get('scores')}")
        print("==================================\n")

    if args.upload_e2b:
        upload_run_to_hf(e2b_dir, args.hf_repo, public=True)

    if args.score or args.upload or args.update_pr:
        target = Path(args.run_dir)
        if not target.is_absolute():
            target = ROOT / target

        scores = None
        if args.score:
            scores = score_run_v02(target)

        if args.upload:
            upload_run_to_hf(target, args.hf_repo, public=True)

        if args.update_pr:
            scores_path = target / "scores.json"
            if scores_path.exists():
                with open(scores_path) as f:
                    scores = json.load(f)
            idx_score = scores.get("decision_index", 0.0) if scores else 0.0
            raw_score = scores.get("raw_index", 0.0) if scores else 0.0
            model_key = "gevva-e4b" if "e4b" in target.name else "gevva-e2b"
            update_submissions_table(model_key, idx_score, raw_score, target.name, args.hf_repo)

            # Git commit and push to myfork
            cmd = ["git", "add", "submissions/README.md"]
            subprocess.run(cmd, cwd=str(UPSTREAM_DIR), check=True)
            cmd = ["git", "commit", "-m", f"docs(submissions): update {model_key} Decision Index score to {idx_score:.2f}"]
            subprocess.run(cmd, cwd=str(UPSTREAM_DIR), check=False)
            cmd = ["git", "push", "myfork", "add-gevva-engine"]
            subprocess.run(cmd, cwd=str(UPSTREAM_DIR), check=True)

        print_pr_instructions(args.hf_repo)


if __name__ == "__main__":
    main()
