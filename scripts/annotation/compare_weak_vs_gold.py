#!/usr/bin/env python
"""Compare weak labels to gold annotations and recommend threshold adjustments.

Usage:
  python scripts/annotation/compare_weak_vs_gold.py \
      --weak data/annotation/exports/weak_labels.jsonl \
      --gold data/annotation/exports/gold_full.jsonl \
      --output data/annotation/exports/threshold_report.json

Matching Rule:
- A weak span is a TP if it overlaps a gold span of the same label with IoG (intersection / gold length) >= 0.5.
- Otherwise weak span counts as FP.
- Gold spans not overlapped by any weak span of same label are FN.

Outputs JSON with per-label precision/recall/F1 and confidence bucket stats, plus heuristic suggestions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

LABELS = {"SYMPTOM", "PRODUCT"}
CONF_BUCKETS = [0.7, 0.8, 0.9, 0.95, 1.01]  # upper bounds


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def index_gold(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    idx: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in records:
        rid = str(r.get("id"))
        entities = r.get("entities", [])
        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # type: ignore
        for e in entities:
            label = e.get("label")
            if label in LABELS:
                by_label[label].append(e)
        idx[rid] = by_label
    return idx


def index_weak(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    idx: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in records:
        rid = str(r.get("id"))
        spans = r.get("weak_spans", [])
        by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # type: ignore
        for s in spans:
            label = s.get("label")
            if label in LABELS:
                by_label[label].append(s)
        idx[rid] = by_label
    return idx


def overlap(g: Dict[str, Any], w: Dict[str, Any]) -> int:
    return max(0, min(g["end"], w["end"]) - max(g["start"], w["start"]))


def compute_metrics(gold_idx, weak_idx) -> Dict[str, Any]:
    label_stats: Dict[str, Dict[str, int]] = {l: {"tp": 0, "fp": 0, "fn": 0} for l in LABELS}
    conf_counts: Dict[str, Counter] = {l: Counter() for l in LABELS}

    for rid, gold_labels in gold_idx.items():
        weak_labels = weak_idx.get(rid, {})
        for label in LABELS:
            gold_spans = gold_labels.get(label, [])
            weak_spans = weak_labels.get(label, [])
            matched_gold = set()
            for w in weak_spans:
                w_conf = float(w.get("confidence", 1.0))
                # bucket confidence
                for b in CONF_BUCKETS:
                    if w_conf < b:
                        conf_counts[label][f"<{b}"] += 1
                        break
                # find best overlapping gold
                best_iog = 0.0
                best_idx = None
                for i, g in enumerate(gold_spans):
                    ov = overlap(g, w)
                    if ov == 0:
                        continue
                    gold_len = g["end"] - g["start"]
                    iog = ov / gold_len if gold_len > 0 else 0
                    if iog >= 0.5 and iog > best_iog:
                        best_iog = iog
                        best_idx = i
                if best_idx is not None:
                    label_stats[label]["tp"] += 1
                    matched_gold.add(best_idx)
                else:
                    label_stats[label]["fp"] += 1
            # FNs
            label_stats[label]["fn"] += len(gold_spans) - len(matched_gold)

    # Compute precision/recall/F1
    report = {"labels": {}, "overall": {}}
    total_tp = total_fp = total_fn = 0
    for label, stats in label_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report["labels"][label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confidence_buckets": dict(conf_counts[label]),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec)
        else 0.0
    )
    report["overall"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(overall_prec, 4),
        "recall": round(overall_rec, 4),
        "f1": round(overall_f1, 4),
    }

    # Suggestions
    suggestions = []
    for label, stats in report["labels"].items():
        high_fp = stats["fp"] > stats["tp"] and stats["tp"] > 0
        low_rec = stats["recall"] < 0.6
        if high_fp:
            suggestions.append(
                f"Label {label}: High FP; consider raising fuzzy threshold or Jaccard gate."
            )
        if low_rec and stats["tp"] > 0:
            suggestions.append(
                f"Label {label}: Low recall; consider lowering fuzzy threshold slightly or expanding lexicon."
            )
    if not suggestions:
        suggestions.append("Metrics acceptable; tune thresholds only after larger validation set.")
    report["suggestions"] = suggestions
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare weak vs gold spans")
    parser.add_argument("--weak", required=True, help="Path to weak label JSONL")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    args = parser.parse_args()

    weak_records = read_jsonl(Path(args.weak))
    gold_records = read_jsonl(Path(args.gold))

    gold_idx = index_gold(gold_records)
    weak_idx = index_weak(weak_records)

    report = compute_metrics(gold_idx, weak_idx)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
