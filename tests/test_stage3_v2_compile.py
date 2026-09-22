#!/usr/bin/env python3
"""tests/test_stage3_v2_compile.py - Stage 3 v2 mixture filter invariants.

Run: uv run python tests/test_stage3_v2_compile.py
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compile_stage3_v2 import drop_unfalsifiable_embedded  # noqa: E402


def _row(source, label):
    return {"id": f"{source}-{label}", "premise": "p", "hypothesis": "h", "label": label, "source": source}


def test_filter_drops_only_embedded_contradictions():
    rows = [
        _row("haystack_embedded", 0), _row("haystack_embedded", 0),
        _row("haystack_embedded", 1), _row("haystack_embedded", 2),
        _row("haystack_corrupted_con", 0),
        _row("haystack_128k_con", 0),
        _row("mnli_train", 0),
    ]
    kept, dropped = drop_unfalsifiable_embedded(rows)
    assert dropped == 2
    assert len(kept) == 5
    assert all(not (r["source"] == "haystack_embedded" and r["label"] == 0) for r in kept)
    # Contradictions from other sources must survive (they are verifiable).
    assert any(r["source"] == "haystack_corrupted_con" and r["label"] == 0 for r in kept)
    assert any(r["source"] == "mnli_train" and r["label"] == 0 for r in kept)


def test_compiled_artifacts_hold_invariants():
    train = Path(__file__).resolve().parent.parent / "data" / "stage3_v2" / "stage3_train.jsonl"
    val = train.parent / "stage3_val.jsonl"
    manifest = train.parent / "manifest.json"
    if not train.exists():
        print("  (skipped: data/stage3_v2 not compiled yet)")
        return
    for path in (train, val):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        bad = [r["id"] for r in rows if r["source"] == "haystack_embedded" and r["label"] == 0]
        assert not bad, f"{path.name}: {len(bad)} unfalsifiable rows remain (e.g. {bad[:3]})"
    m = json.loads(manifest.read_text())
    assert m["dropped_unfalsifiable"] == {"train": 60, "val": 8}
    assert m["outputs"]["train"]["rows"] == m["inputs"]["haystack_train"]["rows"] + 10000 - 60


TESTS = [test_filter_drops_only_embedded_contradictions, test_compiled_artifacts_hold_invariants]


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
