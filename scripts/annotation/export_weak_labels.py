#!/usr/bin/env python
"""Export weak labels for a corpus of complaint texts.

Usage (PowerShell):
  python scripts/annotation/export_weak_labels.py --input data/raw/complaints.txt --output data/annotation/exports/weak_labels.jsonl

Input file: one complaint per line (UTF-8)
Output JSONL: lines with {id, text, weak_spans:[{start,end,label,canonical,confidence,negated}]}.

Lexicons expected at data/lexicon/symptoms.csv and data/lexicon/products.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.weak_label import load_product_lexicon, load_symptom_lexicon, weak_label_batch


def read_texts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def serialize_spans(spans_batch: List[List]):
    records = []
    for idx, spans in enumerate(spans_batch):
        out_spans = []
        for s in spans:
            out_spans.append(
                {
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "text": s.text,
                    "canonical": s.canonical,
                    "confidence": s.confidence,
                    "negated": s.negated,
                }
            )
        records.append(
            {"id": idx, "text": texts[idx], "weak_spans": out_spans}
        )  # texts captured from outer scope
    return records


def main():
    parser = argparse.ArgumentParser(description="Export weak labels to JSONL")
    parser.add_argument(
        "--input", required=True, help="Path to input text file (one complaint per line)"
    )
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--scorer", default="wratio", choices=["wratio", "jaccard"], help="Fuzzy scorer to use"
    )
    parser.add_argument("--negation-window", type=int, default=5, help="Negation window token size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global texts  # used inside serialize_spans
    texts = read_texts(input_path)
    if not texts:
        raise SystemExit("No input lines found.")

    symptom_lex = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
    product_lex = load_product_lexicon(Path("data/lexicon/products.csv"))

    spans_batch = weak_label_batch(
        texts, symptom_lex, product_lex, negation_window=args.negation_window, scorer=args.scorer
    )

    records = serialize_spans(spans_batch)
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
