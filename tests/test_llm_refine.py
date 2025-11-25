from pathlib import Path
import json
import subprocess
import sys

SAMPLE_WEAK = "tests/data_sample_weak.jsonl"
OUTPUT = "tests/data_refined.jsonl"
PROMPT = "prompts/boundary_refine.txt"

def _write_sample():
    lines = [
        json.dumps({
            "text": "I had a mild rash and slight headache after using the cream.",
            "spans": [
                {"text": "rash", "start": 12, "end": 16, "label": "SYMPTOM", "confidence": 0.60},
                {"text": "headache", "start": 28, "end": 36, "label": "SYMPTOM", "confidence": 0.90},
                {"text": "cream", "start": 53, "end": 58, "label": "PRODUCT", "confidence": 0.72}
            ]
        })
    ]
    Path(SAMPLE_WEAK).write_text("\n".join(lines), encoding="utf-8")


def test_llm_refine_stub():
    _write_sample()
    cmd = [sys.executable, "scripts/annotation/refine_llm.py", "--weak", SAMPLE_WEAK, "--out", OUTPUT, "--prompt", PROMPT]
    rc = subprocess.call(cmd)
    assert rc == 0
    data = [json.loads(l) for l in Path(OUTPUT).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(data) == 1
    rec = data[0]
    # Ensure suggestions list exists (stub empty) and metadata present
    assert "llm_suggestions" in rec
    assert isinstance(rec["llm_suggestions"], list)
    assert rec["llm_meta"].get("model")
    # Only mid-confidence spans selected as candidates (rash 0.60, cream 0.72)
    # Stub returns none; verify suggestion_count recorded
    assert rec["llm_meta"].get("suggestion_count") == 0

