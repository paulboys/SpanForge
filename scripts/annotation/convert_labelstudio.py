#!/usr/bin/env python
"""Convert Label Studio JSON export to normalized gold JSONL with provenance & canonical mapping.

Usage:
    python scripts/annotation/convert_labelstudio.py \
            --input data/annotation/raw/label_studio_export.json \
            --output data/annotation/exports/gold_full.jsonl \
            --source complaints_batch_2025Q4 \
            --annotator alice \
            --revision 1 \
            --symptom-lexicon data/lexicon/symptoms.csv \
            --product-lexicon data/lexicon/products.csv

Label Studio Export (task list JSON) expected structure (simplified):
[
    {
        "id": 123,
        "data": {"text": "Complaint text..."},
        "annotations": [
            {
                "result": [
                    {"value": {"start": 10, "end": 20, "text": "redness", "labels": ["SYMPTOM"]}}
                ]
            }
        ]
    }
]

Output JSONL lines (extended schema):
{
    "id": <task_id>,
    "text": <text>,
    "source": <batch_source>,
    "annotator": <annotator_id>,
    "revision": <int_optional>,
    "entities": [
            {"start":..., "end":..., "label":..., "text":..., "canonical":..., "concept_id":...}
    ]
}

Enhancements:
- Adds provenance fields (source, annotator, optional revision).
- Derives canonical + concept_id by exact (case-insensitive) match against provided lexicons;
    falls back to raw text as canonical when no match.
- Retains only ALLOWED_LABELS; first annotation used (extend later for adjudication/consensus).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse lexicon loaders for canonical + concept_id enrichment
try:  # Import inside script to avoid circular issues if run standalone
    from src.weak_label import (  # type: ignore
        LexiconEntry,
        load_product_lexicon,
        load_symptom_lexicon,
    )
except ImportError:
    load_symptom_lexicon = None  # type: ignore
    load_product_lexicon = None  # type: ignore
    LexiconEntry = None  # type: ignore

ALLOWED_LABELS = {"SYMPTOM", "PRODUCT"}


def load_export(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Label Studio export not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of tasks")
    return data


def extract_entities(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotations = task.get("annotations", [])
    if not annotations:
        return []
    # Use first annotation result; extend later for consensus logic
    result = annotations[0].get("result", [])
    entities: List[Dict[str, Any]] = []
    for item in result:
        value = item.get("value", {})
        start = value.get("start")
        end = value.get("end")
        text = value.get("text")
        labels = value.get("labels", [])
        if start is None or end is None or text is None or not labels:
            continue
        label = labels[0]
        if label not in ALLOWED_LABELS:
            continue
        if start >= end:
            continue
        entities.append({"start": start, "end": end, "label": label, "text": text})
    # Sort by start position
    entities.sort(key=lambda e: e["start"])
    return entities


def _build_lookup(entries: List["LexiconEntry"]) -> Dict[str, "LexiconEntry"]:
    lookup: Dict[str, "LexiconEntry"] = {}
    for e in entries:
        lookup[e.term.lower()] = e
        # Allow canonical direct lookup if different
        if e.canonical and e.canonical.lower() != e.term.lower():
            lookup[e.canonical.lower()] = e
    return lookup


def _enrich_entity(
    ent: Dict[str, Any],
    symptom_lookup: Dict[str, "LexiconEntry"],
    product_lookup: Dict[str, "LexiconEntry"],
) -> Dict[str, Any]:
    label = ent["label"]
    raw_text = ent["text"].strip()
    key = raw_text.lower()
    lex_entry: Optional["LexiconEntry"] = None
    if label == "SYMPTOM":
        lex_entry = symptom_lookup.get(key)
    elif label == "PRODUCT":
        lex_entry = product_lookup.get(key)
    if lex_entry:
        ent["canonical"] = lex_entry.canonical
        if getattr(lex_entry, "concept_id", None):
            ent["concept_id"] = lex_entry.concept_id
    else:
        # Fallback: canonical = raw text (normalized casing)
        ent["canonical"] = raw_text.lower()
    # Deterministic concept_id if missing and label is SYMPTOM
    if label == "SYMPTOM" and "concept_id" not in ent:
        ent["concept_id"] = f"SYMPTOM:{ent['canonical'].replace(' ', '_')}"
    return ent


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio export to gold JSONL with provenance & canonical enrichment"
    )
    parser.add_argument("--input", required=True, help="Path to Label Studio JSON export file")
    parser.add_argument("--output", required=True, help="Path to output gold JSONL file")
    parser.add_argument(
        "--source",
        required=True,
        help="Source batch identifier for provenance (e.g., complaints_batch_2025Q4)",
    )
    parser.add_argument("--annotator", required=True, help="Primary annotator ID or name")
    parser.add_argument(
        "--revision",
        type=int,
        default=None,
        help="Optional revision number for re-annotated batches",
    )
    parser.add_argument(
        "--symptom-lexicon",
        default=None,
        help="Optional path to symptom lexicon CSV for canonical mapping",
    )
    parser.add_argument(
        "--product-lexicon",
        default=None,
        help="Optional path to product lexicon CSV for canonical mapping",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_export(input_path)

    # Load lexicons if provided and loader available
    symptom_lookup: Dict[str, "LexiconEntry"] = {}
    product_lookup: Dict[str, "LexiconEntry"] = {}
    if load_symptom_lexicon and args.symptom_lexicon:
        sym_path = Path(args.symptom_lexicon)
        if sym_path.exists():
            symptom_lookup = _build_lookup(load_symptom_lexicon(sym_path))
    if load_product_lexicon and args.product_lexicon:
        prod_path = Path(args.product_lexicon)
        if prod_path.exists():
            product_lookup = _build_lookup(load_product_lexicon(prod_path))
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            text = task.get("data", {}).get("text")
            if not text:
                continue
            entities = extract_entities(task)
            enriched = [_enrich_entity(e, symptom_lookup, product_lookup) for e in entities]
            record = {
                "id": task.get("id"),
                "text": text,
                "source": args.source,
                "annotator": args.annotator,
                **({"revision": args.revision} if args.revision is not None else {}),
                "entities": enriched,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(
        f"Converted {count} tasks to {output_path} (source={args.source}, annotator={args.annotator})"
    )


if __name__ == "__main__":
    main()
