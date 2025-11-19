import json
from pathlib import Path
import pytest

ALLOWED_LABELS = {"SYMPTOM", "PRODUCT"}

EXPORT_DIR = Path("data/annotation/exports")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_gold_files():
    if not EXPORT_DIR.exists():
        return []
    return list(EXPORT_DIR.glob("*.jsonl"))

@pytest.mark.parametrize("gold_file", find_gold_files() or [None])
def test_curation_integrity(gold_file):
    if gold_file is None:
        pytest.skip("No gold JSONL files present; skipping curation integrity test.")

    for record in _iter_jsonl(gold_file):
        text = record.get("text", "")
        entities = record.get("entities", [])

        # Provenance fields presence (basic)
        assert "source" in record and record["source"], "Missing provenance 'source'"
        assert "annotator" in record and record["annotator"], "Missing provenance 'annotator'"
        # revision optional; if present must be int >=0
        if "revision" in record:
            assert isinstance(record["revision"], int) and record["revision"] >= 0, "Revision must be non-negative int"

        # Ordering & duplicate detection (allow overlaps; detect conflicting labels)
        entities_sorted = sorted(entities, key=lambda e: e.get("start", -1))
        seen = set()
        for idx, ent in enumerate(entities_sorted):
            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("label")
            assert isinstance(start, int) and isinstance(end, int), "Start/end must be ints"
            assert start < end, "Invalid span boundary (start >= end)"
            assert label in ALLOWED_LABELS, f"Unexpected label: {label}"
            key = (start, end, label)
            assert key not in seen, "Duplicate span encountered"
            seen.add(key)

        # Overlap conflict detection: overlapping spans must share same label
        for i in range(len(entities_sorted)):
            for j in range(i + 1, len(entities_sorted)):
                a = entities_sorted[i]
                b = entities_sorted[j]
                if b["start"] >= a["end"]:
                    break  # since sorted by start, no further overlap possible with 'a'
                overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
                if overlap > 0:
                    assert a["label"] == b["label"], "Conflicting overlapping labels detected"

        # Canonical presence + non-empty
        for ent in entities_sorted:
            assert "canonical" in ent and ent["canonical"], "Missing canonical field on entity"
            if "concept_id" in ent:
                assert isinstance(ent["concept_id"], str) and ent["concept_id"], "Invalid concept_id"

        # Text slice alignment
        for ent in entities_sorted:
            if "text" in ent:
                start = ent["start"]
                end = ent["end"]
                assert text[start:end] == ent["text"], "Entity text mismatch with source substring"
