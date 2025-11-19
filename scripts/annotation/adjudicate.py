#!/usr/bin/env python
"""Adjudicate multiple annotator exports into a consensus gold JSONL.

Input: Label Studio task JSON (list) or multiple converted gold JSONL files.
Current minimal implementation:
- If multiple JSONL files provided, groups by task id.
- Majority vote on identical (start,end,label) spans.
- Overlapping conflicting labels -> written to conflicts JSON.
- Writes consensus JSONL + conflicts summary.

Usage:
  python scripts/annotation/adjudicate.py --inputs gold_a.jsonl gold_b.jsonl --out data/annotation/exports/gold_consensus.jsonl --conflicts data/annotation/conflicts/conflicts.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def majority_vote(spans_list):
    # spans_list: list of list[span]
    counter = Counter()
    for spans in spans_list:
        for s in spans:
            key = (s["start"], s["end"], s["label"])
            counter[key] += 1
    return counter


def build_consensus(task_id, text, span_groups, min_agree: int):
    # span_groups is Counter of (start,end,label)->count
    consensus = []
    for (start, end, label), count in span_groups.items():
        if count >= min_agree:
            consensus.append({"start": start, "end": end, "label": label, "text": text[start:end]})
    return sorted(consensus, key=lambda e: e["start"])


def detect_conflicts(spans_list):
    # Flatten all spans per task and detect overlapping different labels
    all_spans = []
    for spans in spans_list:
        all_spans.extend(spans)
    conflicts = []
    sorted_spans = sorted(all_spans, key=lambda e: e["start"])
    for i in range(len(sorted_spans)):
        for j in range(i+1, len(sorted_spans)):
            a = sorted_spans[i]; b = sorted_spans[j]
            if b["start"] >= a["end"]:
                break
            if min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0 and a["label"] != b["label"]:
                conflicts.append({"task_id": None, "span_a": a, "span_b": b})
    return conflicts


def main():
    parser = argparse.ArgumentParser(description="Adjudicate multiple gold JSONL files into consensus")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of gold JSONL files")
    parser.add_argument("--out", required=True, help="Consensus output JSONL path")
    parser.add_argument("--conflicts", required=True, help="Conflicts JSON path")
    parser.add_argument("--min-agree", type=int, default=2, help="Minimum annotator agreement for inclusion")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    tasks_map = defaultdict(list)  # task_id -> list[record]
    for p in input_paths:
        for rec in read_jsonl(p):
            tasks_map[rec.get("id")].append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_path = Path(args.conflicts)
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)

    conflict_records = []
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for tid, recs in tasks_map.items():
            texts = {r.get("text") for r in recs}
            text = texts.pop() if texts else ""
            spans_list = [r.get("entities", []) for r in recs]
            vote = majority_vote(spans_list)
            consensus = build_consensus(tid, text, vote, args.min_agree)
            # Conflict detection across all spans
            conflicts = detect_conflicts(spans_list)
            for c in conflicts:
                c["task_id"] = tid
            if conflicts:
                conflict_records.extend(conflicts)
            record = {"id": tid, "text": text, "entities": consensus, "annotators": [r.get("annotator") for r in recs], "min_agree": args.min_agree}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    conflicts_path.write_text(json.dumps(conflict_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Consensus written for {written} tasks -> {out_path}")
    print(f"Conflicts: {len(conflict_records)} -> {conflicts_path}")


if __name__ == "__main__":
    main()
