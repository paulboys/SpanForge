#!/usr/bin/env python
"""Generate annotation quality metrics from converted gold JSONL.

Metrics:
- Per-annotator entity counts & mean spans/task.
- Label distribution.
- Overlap conflicts (different labels intersecting) count.
- Simple pairwise agreement (IOU ≥0.5) for annotator pairs if multiple annotators present.

Usage:
  python scripts/annotation/quality_report.py --gold data/annotation/exports/gold_full.jsonl --out data/annotation/reports/quality.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from itertools import combinations


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute annotation quality metrics")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL")
    parser.add_argument("--out", required=True, help="Output report JSON path")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    annotator_counts = {}
    label_counts = {}
    task_spans = []
    conflict_count = 0

    records = list(iter_records(gold_path))
    for rec in records:
        ann = rec.get("annotator", "unknown")
        ents = rec.get("entities", [])
        task_spans.append(len(ents))
        annotator_counts.setdefault(ann, 0)
        annotator_counts[ann] += len(ents)
        # Conflict detection (overlap diff label)
        sorted_ents = sorted(ents, key=lambda e: e["start"])
        for i in range(len(sorted_ents)):
            for j in range(i + 1, len(sorted_ents)):
                a = sorted_ents[i]; b = sorted_ents[j]
                if b["start"] >= a["end"]:  # non-overlapping forward
                    break
                if min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0 and a["label"] != b["label"]:
                    conflict_count += 1
        for e in ents:
            label_counts.setdefault(e["label"], 0)
            label_counts[e["label"]] += 1

    mean_spans_task = sum(task_spans) / len(task_spans) if task_spans else 0.0

    # Pairwise agreement placeholder (single annotator -> none)
    pairwise = {}
    annotators = list(annotator_counts.keys())
    if len(annotators) > 1:
        # Build mapping: task_id -> annotator -> spans
        task_map = {}
        for rec in records:
            tid = rec.get("id")
            ann = rec.get("annotator", "unknown")
            task_map.setdefault(tid, {})[ann] = rec.get("entities", [])
        for a1, a2 in combinations(annotators, 2):
            agreements = 0; total = 0
            for tid, ann_map in task_map.items():
                if a1 in ann_map and a2 in ann_map:
                    spans1 = [(e["start"], e["end"]) for e in ann_map[a1]]
                    spans2 = [(e["start"], e["end"]) for e in ann_map[a2]]
                    for s1 in spans1:
                        for s2 in spans2:
                            total += 1
                            if iou(s1, s2) >= 0.5:
                                agreements += 1
            pairwise[f"{a1}|{a2}"] = agreements / total if total else None

    report = {
        "annotator_counts": annotator_counts,
        "mean_spans_per_task": mean_spans_task,
        "label_counts": label_counts,
        "conflicts": conflict_count,
        "pairwise_agreement": pairwise,
        "n_tasks": len(records),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Quality report written to {report_path}")


if __name__ == "__main__":
    main()
