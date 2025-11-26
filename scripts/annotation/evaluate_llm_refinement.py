"""Evaluate LLM refinement quality by comparing weak → LLM → gold annotations.

Comprehensive evaluation script that:
1. Loads weak label predictions, LLM-refined predictions, and gold annotations
2. Computes 3-way comparison metrics (IOU uplift, boundary precision, correction rates)
3. Stratifies results by confidence, label, and span length
4. Generates JSON report and markdown summary

Usage:
    python scripts/annotation/evaluate_llm_refinement.py \\
        --weak data/annotation/exports/weak_labels.jsonl \\
        --refined data/annotation/exports/llm_refined.jsonl \\
        --gold data/annotation/exports/gold_standard.jsonl \\
        --output data/annotation/reports/llm_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    calibration_curve,
    compute_boundary_precision,
    compute_correction_rate,
    compute_iou_delta,
    compute_precision_recall_f1,
    stratify_by_confidence,
    stratify_by_label,
    stratify_by_span_length,
)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of parsed JSON records
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_spans_from_records(
    records: List[Dict[str, Any]], span_key: str = "spans"
) -> List[Dict[str, Any]]:
    """Extract all spans from JSONL records.

    Args:
        records: List of JSONL records
        span_key: Key name for spans list (default: "spans")

    Returns:
        Flattened list of all spans
    """
    all_spans = []
    for record in records:
        spans = record.get(span_key, [])
        all_spans.extend(spans)
    return all_spans


def extract_llm_suggestions_from_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract LLM suggestions from refined JSONL records.

    Args:
        records: List of JSONL records with llm_suggestions

    Returns:
        Flattened list of all LLM suggestions
    """
    all_suggestions = []
    for record in records:
        suggestions = record.get("llm_suggestions", [])
        all_suggestions.extend(suggestions)
    return all_suggestions


def evaluate_overall_metrics(
    weak_spans: List[Dict[str, Any]],
    llm_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute overall evaluation metrics.

    Args:
        weak_spans: Original weak predictions
        llm_spans: LLM-refined predictions
        gold_spans: Gold standard annotations

    Returns:
        Dict with all overall metrics
    """
    # IOU delta (improvement)
    iou_delta = compute_iou_delta(weak_spans, llm_spans, gold_spans)

    # Boundary precision
    weak_boundary = compute_boundary_precision(weak_spans, gold_spans)
    llm_boundary = compute_boundary_precision(llm_spans, gold_spans)

    # Correction rate
    correction = compute_correction_rate(weak_spans, llm_spans, gold_spans)

    # Precision/Recall/F1
    weak_prf = compute_precision_recall_f1(weak_spans, gold_spans)
    llm_prf = compute_precision_recall_f1(llm_spans, gold_spans)

    # Calibration curves
    weak_calibration = calibration_curve(weak_spans, gold_spans, "confidence")
    llm_calibration = calibration_curve(llm_spans, gold_spans, "llm_confidence")

    return {
        "iou_delta": iou_delta,
        "weak_boundary_metrics": weak_boundary,
        "llm_boundary_metrics": llm_boundary,
        "correction_metrics": correction,
        "weak_precision_recall_f1": weak_prf,
        "llm_precision_recall_f1": llm_prf,
        "weak_calibration": weak_calibration,
        "llm_calibration": llm_calibration,
        "span_counts": {
            "weak_total": len(weak_spans),
            "llm_total": len(llm_spans),
            "gold_total": len(gold_spans),
        },
    }


def evaluate_stratified_metrics(
    weak_spans: List[Dict[str, Any]],
    llm_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    stratify_by: str = "label",
) -> Dict[str, Any]:
    """Compute stratified evaluation metrics.

    Args:
        weak_spans: Original weak predictions
        llm_spans: LLM-refined predictions
        gold_spans: Gold standard annotations
        stratify_by: Stratification dimension ("label", "confidence", "span_length")

    Returns:
        Dict with per-stratum metrics
    """
    if stratify_by == "label":
        weak_strata = stratify_by_label(weak_spans)
        llm_strata = stratify_by_label(llm_spans)
        gold_strata = stratify_by_label(gold_spans)
    elif stratify_by == "confidence":
        weak_strata = stratify_by_confidence(weak_spans, "confidence")
        llm_strata = stratify_by_confidence(llm_spans, "llm_confidence")
        gold_strata = {"all": gold_spans}  # Gold doesn't have confidence buckets
    elif stratify_by == "span_length":
        weak_strata = stratify_by_span_length(weak_spans)
        llm_strata = stratify_by_span_length(llm_spans)
        gold_strata = stratify_by_span_length(gold_spans)
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")

    stratified_results = {}

    for stratum_name in weak_strata.keys():
        weak_subset = weak_strata.get(stratum_name, [])
        llm_subset = llm_strata.get(stratum_name, [])

        # For confidence stratification, use all gold spans
        if stratify_by == "confidence":
            gold_subset = gold_spans
        else:
            gold_subset = gold_strata.get(stratum_name, [])

        if not weak_subset and not llm_subset:
            continue

        # Compute metrics for this stratum
        iou_delta = compute_iou_delta(weak_subset, llm_subset, gold_subset)
        weak_prf = compute_precision_recall_f1(weak_subset, gold_subset)
        llm_prf = compute_precision_recall_f1(llm_subset, gold_subset)

        stratified_results[stratum_name] = {
            "iou_delta": iou_delta,
            "weak_metrics": weak_prf,
            "llm_metrics": llm_prf,
            "span_counts": {
                "weak": len(weak_subset),
                "llm": len(llm_subset),
                "gold": len(gold_subset),
            },
        }

    return stratified_results


def generate_markdown_summary(report: Dict[str, Any]) -> str:
    """Generate human-readable markdown summary.

    Args:
        report: Full evaluation report dict

    Returns:
        Markdown-formatted summary string
    """
    overall = report["overall_metrics"]
    iou_delta = overall["iou_delta"]
    weak_prf = overall["weak_precision_recall_f1"]
    llm_prf = overall["llm_precision_recall_f1"]
    correction = overall["correction_metrics"]

    md = f"""# LLM Refinement Evaluation Report

## Overall Performance

### IOU Improvement
- **Weak Labels Mean IOU**: {iou_delta['weak_mean_iou']:.3f}
- **LLM Refined Mean IOU**: {iou_delta['llm_mean_iou']:.3f}
- **Delta**: {iou_delta['delta']:+.3f}
- **Improvement**: {iou_delta['improvement_pct']:+.1f}%

### Precision/Recall/F1

| Metric | Weak Labels | LLM Refined | Delta |
|--------|-------------|-------------|-------|
| Precision | {weak_prf['precision']:.3f} | {llm_prf['precision']:.3f} | {llm_prf['precision'] - weak_prf['precision']:+.3f} |
| Recall | {weak_prf['recall']:.3f} | {llm_prf['recall']:.3f} | {llm_prf['recall'] - weak_prf['recall']:+.3f} |
| F1 | {weak_prf['f1']:.3f} | {llm_prf['f1']:.3f} | {llm_prf['f1'] - weak_prf['f1']:+.3f} |

### Correction Statistics
- **Total Spans**: {correction['total_spans']}
- **Modified by LLM**: {correction['modified_count']} ({correction['modified_count']/correction['total_spans']*100:.1f}%)
- **Improved**: {correction['improved_count']} ({correction['improvement_rate']*100:.1f}% of modified)
- **Worsened**: {correction['worsened_count']} ({correction['false_refinement_rate']*100:.1f}% of modified)
- **Unchanged**: {correction['unchanged_count']}

## Stratified Analysis

"""

    # Add stratified results if available
    for strat_type in ["by_label", "by_confidence", "by_span_length"]:
        if strat_type in report:
            md += f"### Stratified {strat_type.replace('by_', '').replace('_', ' ').title()}\n\n"
            md += "| Stratum | Weak F1 | LLM F1 | IOU Delta | Span Count |\n"
            md += "|---------|---------|--------|-----------|------------|\n"

            for stratum_name, metrics in report[strat_type].items():
                weak_f1 = metrics["weak_metrics"]["f1"]
                llm_f1 = metrics["llm_metrics"]["f1"]
                iou_d = metrics["iou_delta"]["delta"]
                count = metrics["span_counts"]["weak"]

                md += (
                    f"| {stratum_name} | {weak_f1:.3f} | {llm_f1:.3f} | {iou_d:+.3f} | {count} |\n"
                )

            md += "\n"

    md += f"""
## Boundary Precision

### Weak Labels
- **Exact Match Rate**: {overall['weak_boundary_metrics']['exact_match_rate']:.1%}
- **Mean IOU**: {overall['weak_boundary_metrics']['mean_iou']:.3f}

### LLM Refined
- **Exact Match Rate**: {overall['llm_boundary_metrics']['exact_match_rate']:.1%}
- **Mean IOU**: {overall['llm_boundary_metrics']['mean_iou']:.3f}

---

*Generated by SpanForge LLM Evaluation Harness*
"""

    return md


def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM refinement quality against gold standard"
    )
    parser.add_argument("--weak", required=True, help="Path to weak label JSONL file")
    parser.add_argument("--refined", required=True, help="Path to LLM-refined JSONL file")
    parser.add_argument("--gold", required=True, help="Path to gold standard JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument(
        "--stratify",
        nargs="+",
        default=["label"],
        choices=["label", "confidence", "span_length"],
        help="Stratification dimensions (default: label)",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Generate markdown summary alongside JSON report"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading weak labels from {args.weak}...")
    weak_records = load_jsonl(args.weak)
    weak_spans = extract_spans_from_records(weak_records)

    print(f"Loading LLM-refined labels from {args.refined}...")
    refined_records = load_jsonl(args.refined)
    llm_spans = extract_llm_suggestions_from_records(refined_records)

    print(f"Loading gold standard from {args.gold}...")
    gold_records = load_jsonl(args.gold)
    gold_spans = extract_spans_from_records(gold_records)

    print(f"\nSpan counts:")
    print(f"  Weak: {len(weak_spans)}")
    print(f"  LLM:  {len(llm_spans)}")
    print(f"  Gold: {len(gold_spans)}")

    # Compute overall metrics
    print("\nComputing overall metrics...")
    overall_metrics = evaluate_overall_metrics(weak_spans, llm_spans, gold_spans)

    # Build report
    report = {
        "overall_metrics": overall_metrics,
        "input_files": {"weak": args.weak, "refined": args.refined, "gold": args.gold},
    }

    # Compute stratified metrics
    for stratify_dim in args.stratify:
        print(f"Computing stratified metrics by {stratify_dim}...")
        stratified = evaluate_stratified_metrics(
            weak_spans, llm_spans, gold_spans, stratify_by=stratify_dim
        )
        report[f"by_{stratify_dim}"] = stratified

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON report saved to {output_path}")

    # Generate markdown if requested
    if args.markdown:
        md_content = generate_markdown_summary(report)
        md_path = output_path.with_suffix(".md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"✓ Markdown summary saved to {md_path}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    iou_delta = overall_metrics["iou_delta"]
    print(f"IOU Improvement: {iou_delta['improvement_pct']:+.1f}%")
    print(f"  Weak:  {iou_delta['weak_mean_iou']:.3f}")
    print(f"  LLM:   {iou_delta['llm_mean_iou']:.3f}")
    print(f"  Delta: {iou_delta['delta']:+.3f}")

    correction = overall_metrics["correction_metrics"]
    print(f"\nCorrection Rate: {correction['improvement_rate']*100:.1f}%")
    print(f"  Improved:  {correction['improved_count']}/{correction['modified_count']}")
    print(f"  Worsened:  {correction['worsened_count']}/{correction['modified_count']}")

    llm_prf = overall_metrics["llm_precision_recall_f1"]
    print(f"\nLLM F1 Score: {llm_prf['f1']:.3f}")
    print(f"  Precision: {llm_prf['precision']:.3f}")
    print(f"  Recall:    {llm_prf['recall']:.3f}")


if __name__ == "__main__":
    main()
