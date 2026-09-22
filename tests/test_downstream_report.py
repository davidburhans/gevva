#!/usr/bin/env python3
"""tests/test_downstream_report.py - Structured JSON artifact for the downstream eval.

The chain's downstream-benchmark stage crashed on 2026-09-21 because it passed
--out to a script that had no such argument; results lived only in stdout.
eval_downstream_decisions.py now returns metrics per benchmark and writes a
provenance-complete JSON artifact. These tests pin the report contract.

Run: uv run python tests/test_downstream_report.py
"""

import json
import sys
import tempfile
import traceback
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_downstream_decisions import build_report  # noqa: E402


def _fake_args(val_path=None):
    return Namespace(
        model_path="ckpt/gemma-4-e2b-nli-w4a16-stage2", base_model="google/gemma-4-E2B",
        adapter_path="./ckpt/x/best", val_path=val_path, max_val_samples=1000,
        qat=False, qat_bits=4, qat_group_size=32,
    )


def test_report_carries_provenance_and_results():
    report = build_report(_fake_args(), {"nli_calibration": {"accuracy": 0.9, "n_samples": 10}})
    prov = report["provenance"]
    for key in ("generated_at", "git_sha", "model_path", "adapter_path", "val_sha256_16",
                "qat", "qat_bits", "qat_group_size"):
        assert key in prov, f"provenance missing {key}"
    assert report["results"]["nli_calibration"]["accuracy"] == 0.9


def test_report_hashes_val_file_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        val = Path(tmp, "val.jsonl")
        val.write_text('{"premise":"a","hypothesis":"b","label":1,"source":"t"}\n')
        import hashlib
        expected = hashlib.sha256(val.read_bytes()).hexdigest()[:16]
        report = build_report(_fake_args(val_path=str(val)), {})
        assert report["provenance"]["val_sha256_16"] == expected


def test_report_tolerates_missing_val_file():
    report = build_report(_fake_args(val_path="/nonexistent/val.jsonl"), {})
    assert report["provenance"]["val_sha256_16"] is None


def test_report_is_json_serializable_roundtrip():
    report = build_report(_fake_args(), {"latency": {"e2e_p50_ms": 13.9, "throughput_dps": 71.9}})
    roundtripped = json.loads(json.dumps(report, sort_keys=True))
    assert roundtripped == report


TESTS = [
    test_report_carries_provenance_and_results,
    test_report_hashes_val_file_when_present,
    test_report_tolerates_missing_val_file,
    test_report_is_json_serializable_roundtrip,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
