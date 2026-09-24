"""gevva.cli: Command-line interface for Gevva System 1 Decision Engine.

Commands:
    gevva version                 Print Gevva version & environment
    gevva predict                 Evaluate a premise-hypothesis pair
    gevva rerank                  Rerank candidate options for a query
    gevva grade                   Grade a candidate response against a reference
    gevva finetune                Launch custom fine-tuning
    gevva eval                    Run JevBench or downstream benchmark evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_version(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from gevva import __version__

    print(f"Gevva version: {__version__}")
    print(f"PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
    print(f"Transformers version: {transformers.__version__}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from gevva import GevvaCrossEncoder

    print(f"Loading Gevva model from '{args.model}'...")
    model = GevvaCrossEncoder(args.model, device=args.device)
    probs = model.predict([(args.premise, args.hypothesis)])[0]
    labels = ["contradiction", "entailment", "neutral"]
    best_idx = int(probs.argmax())
    best_label = labels[best_idx]
    best_conf = float(probs[best_idx])

    result = {
        "verdict": best_label,
        "confidence": round(best_conf, 4),
        "probabilities": {
            "contradiction": round(float(probs[0]), 4),
            "entailment": round(float(probs[1]), 4),
            "neutral": round(float(probs[2]), 4),
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nVerdict:    {best_label.upper()} ({best_conf*100:.2f}%)")
        print(f"Probabilities:")
        print(f"  Contradiction: {probs[0]*100:6.2f}%")
        print(f"  Entailment:    {probs[1]*100:6.2f}%")
        print(f"  Neutral:       {probs[2]*100:6.2f}%")
    return 0


def cmd_rerank(args: argparse.Namespace) -> int:
    from gevva import GevvaCrossEncoder

    print(f"Loading Gevva model from '{args.model}'...")
    model = GevvaCrossEncoder(args.model, device=args.device)
    best_idx, scores = model.rerank(args.query, args.options)

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    if args.json:
        res = [
            {"rank": r + 1, "index": idx, "option": args.options[idx], "score": round(float(s), 4)}
            for r, (idx, s) in enumerate(ranked)
        ]
        print(json.dumps({"query": args.query, "ranking": res}, indent=2))
    else:
        print(f"\nQuery: {args.query}")
        print("Rankings:")
        for r, (idx, s) in enumerate(ranked):
            marker = " *" if idx == best_idx else ""
            print(f"  [{r+1}] Option {idx}: '{args.options[idx]}' (score: {s:.4f}){marker}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    from gevva import GevvaCrossEncoder

    print(f"Loading Gevva model from '{args.model}'...")
    model = GevvaCrossEncoder(args.model, device=args.device)
    grade = model.grade(args.question, reference=args.reference, candidate=args.candidate)

    if args.json:
        print(json.dumps({
            "is_correct": grade.is_correct,
            "label": grade.label,
            "confidence": round(float(grade.score), 4),
            "scores": [round(float(s), 4) for s in grade.scores],
        }, indent=2))
    else:
        verdict = "CORRECT (Entailment)" if grade.is_correct else f"INCORRECT ({grade.label})"
        print(f"\nGrade:      {verdict}")
        print(f"Confidence: {grade.score*100:.2f}%")
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    import subprocess
    cmd = [sys.executable, "finetune.py"] + args.extra_args
    return subprocess.run(cmd).returncode


def cmd_eval(args: argparse.Namespace) -> int:
    import subprocess
    if args.suite == "jevbench":
        script = "scripts/eval_jevbench_public.py"
    elif args.suite == "openjev":
        script = "eval_openjev_benchmarks.py"
    else:
        script = "eval_downstream_decisions.py"
    cmd = [sys.executable, script] + args.extra_args
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gevva",
        description="Gevva: State-of-the-Art Multimodal 128K System 1 Decision Engine (#1 on JevBench)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # version
    p_ver = subparsers.add_parser("version", help="Print Gevva version & environment details")
    p_ver.set_defaults(func=cmd_version)

    # predict
    p_pred = subparsers.add_parser("predict", help="Evaluate a premise-hypothesis pair")
    p_pred.add_argument("--model", default="ckpt/gevva-e2b", help="Model path (default: ckpt/gevva-e2b)")
    p_pred.add_argument("--premise", required=True, help="Premise text context")
    p_pred.add_argument("--hypothesis", required=True, help="Hypothesis / claim statement")
    p_pred.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    p_pred.add_argument("--json", action="store_true", help="Output results in JSON format")
    p_pred.set_defaults(func=cmd_predict)

    # rerank
    p_rerank = subparsers.add_parser("rerank", help="Rerank multiple candidate options for a query")
    p_rerank.add_argument("--model", default="ckpt/gevva-e2b", help="Model path (default: ckpt/gevva-e2b)")
    p_rerank.add_argument("--query", required=True, help="Query / question text")
    p_rerank.add_argument("--options", nargs="+", required=True, help="Candidate options to rank")
    p_rerank.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    p_rerank.add_argument("--json", action="store_true", help="Output results in JSON format")
    p_rerank.set_defaults(func=cmd_rerank)

    # grade
    p_grade = subparsers.add_parser("grade", help="Grade a candidate response against a reference answer")
    p_grade.add_argument("--model", default="ckpt/gevva-e2b", help="Model path (default: ckpt/gevva-e2b)")
    p_grade.add_argument("--question", required=True, help="Task question or prompt")
    p_grade.add_argument("--reference", required=True, help="Reference gold answer or rubric")
    p_grade.add_argument("--candidate", required=True, help="Candidate response to grade")
    p_grade.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    p_grade.add_argument("--json", action="store_true", help="Output results in JSON format")
    p_grade.set_defaults(func=cmd_grade)

    # finetune
    p_ft = subparsers.add_parser("finetune", help="Launch custom fine-tuning engine")
    p_ft.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments passed to finetune.py")
    p_ft.set_defaults(func=cmd_finetune)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run benchmark evaluation suite")
    p_eval.add_argument("--suite", choices=["jevbench", "openjev", "downstream"], default="jevbench", help="Benchmark suite to run")
    p_eval.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments passed to the eval script")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
