#!/usr/bin/env python
"""Append provenance entry to registry CSV after conversion/adjudication.

Creates file if missing with header.

Usage:
  python scripts/annotation/register_batch.py --batch-id complaints_batch_2025Q4 --gold data/annotation/exports/gold_full.jsonl --annotators alice --revision 1 --notes "initial mock batch"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import json
from datetime import datetime

HEADER = ["timestamp","batch_id","gold_file","n_tasks","annotators","revision","notes"]


def count_tasks(gold_path: Path) -> int:
    n = 0
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Register gold batch provenance")
    parser.add_argument("--batch-id", required=True, help="Batch identifier")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL")
    parser.add_argument("--annotators", required=True, help="Comma-separated annotators")
    parser.add_argument("--revision", type=int, default=0, help="Revision number")
    parser.add_argument("--notes", default="", help="Freeform notes")
    parser.add_argument("--registry", default="data/annotation/registry.csv", help="Registry CSV path")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    n_tasks = count_tasks(gold_path)

    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not registry_path.exists()

    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            args.batch_id,
            str(gold_path),
            n_tasks,
            args.annotators,
            args.revision,
            args.notes,
        ])
    print(f"Registered batch {args.batch_id} ({n_tasks} tasks) -> {registry_path}")


if __name__ == "__main__":
    main()
