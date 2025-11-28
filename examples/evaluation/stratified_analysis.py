"""
Evaluation Example: Stratified Analysis
========================================

Demonstrates how to break down evaluation metrics by different stratification
criteria (confidence buckets, entity labels, span lengths) to identify
performance patterns and bottlenecks.

**What You'll Learn:**
- Stratifying metrics by confidence score
- Analyzing performance by entity type (SYMPTOM vs PRODUCT)
- Grouping spans by length (short/medium/long)
- Identifying weak spots in your pipeline
- Prioritizing annotation efforts

**Prerequisites:**
- Completed evaluation/compute_metrics.py
- Understanding of confidence scores
- Familiarity with evaluation metrics

**Runtime:** ~45 seconds

**Use Cases:**
- Finding which confidence ranges need more annotation
- Comparing SYMPTOM vs PRODUCT extraction quality
- Identifying problematic span length patterns
- Targeting LLM refinement improvements
- Quality assurance for production systems
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    compute_boundary_precision,
    compute_precision_recall_f1,
    stratify_by_confidence,
    stratify_by_label,
    stratify_by_span_length,
)


def create_diverse_sample_data():
    """Create sample data with diverse spans for stratification."""

    # Weak labels with varying characteristics
    weak_spans = [
        # High confidence symptoms
        {"text": "burning", "start": 0, "end": 7, "label": "SYMPTOM", "confidence": 1.0},
        {"text": "redness", "start": 10, "end": 17, "label": "SYMPTOM", "confidence": 0.98},
        # Medium confidence symptoms
        {"text": "mild itching", "start": 20, "end": 32, "label": "SYMPTOM", "confidence": 0.75},
        {"text": "discomfort", "start": 35, "end": 45, "label": "SYMPTOM", "confidence": 0.72},
        # Low confidence symptoms (ambiguous)
        {"text": "feeling", "start": 50, "end": 57, "label": "SYMPTOM", "confidence": 0.55},
        # Products (high confidence)
        {"text": "moisturizer", "start": 60, "end": 71, "label": "PRODUCT", "confidence": 0.95},
        {"text": "face cream", "start": 75, "end": 85, "label": "PRODUCT", "confidence": 0.92},
        # Long multi-word spans
        {
            "text": "severe burning sensation",
            "start": 90,
            "end": 114,
            "label": "SYMPTOM",
            "confidence": 0.88,
        },
    ]

    # Gold standard
    gold_spans = [
        {"text": "burning", "start": 0, "end": 7, "label": "SYMPTOM"},
        {"text": "redness", "start": 10, "end": 17, "label": "SYMPTOM"},
        {"text": "itching", "start": 25, "end": 32, "label": "SYMPTOM"},  # "mild" removed
        {"text": "discomfort", "start": 35, "end": 45, "label": "SYMPTOM"},
        # "feeling" is not a real symptom (FP in weak)
        {"text": "moisturizer", "start": 60, "end": 71, "label": "PRODUCT"},
        {"text": "face cream", "start": 75, "end": 85, "label": "PRODUCT"},
        {
            "text": "burning sensation",
            "start": 97,
            "end": 114,
            "label": "SYMPTOM",
        },  # "severe" removed
    ]

    return weak_spans, gold_spans


def demo_stratify_by_confidence():
    """Demonstrate stratification by confidence score."""
    print("\n" + "=" * 70)
    print("📊 STRATIFICATION BY CONFIDENCE")
    print("=" * 70)

    weak_spans, gold_spans = create_diverse_sample_data()

    # Stratify into 3 buckets
    stratified = stratify_by_confidence(weak_spans, n_bins=3)

    print("\n🔍 Confidence Buckets:\n")

    for bucket_name, bucket_spans in stratified.items():
        if not bucket_spans:
            continue

        # Compute metrics for this bucket
        metrics = compute_precision_recall_f1(bucket_spans, gold_spans)
        boundary = compute_boundary_precision(bucket_spans, gold_spans)

        # Get confidence range
        confidences = [s["confidence"] for s in bucket_spans]
        conf_min, conf_max = min(confidences), max(confidences)

        print(f"   {bucket_name.upper().replace('_', ' ')} ({conf_min:.2f}-{conf_max:.2f}):")
        print(f"      Span Count:  {len(bucket_spans)}")
        print(f"      Precision:   {metrics['precision']:.3f}")
        print(f"      Recall:      {metrics['recall']:.3f}")
        print(f"      F1:          {metrics['f1']:.3f}")
        print(f"      Mean IOU:    {boundary['mean_iou']:.3f}")
        print()

    print("💡 Interpretation:")
    print("   • High confidence spans should have F1 > 0.90")
    print("   • Medium confidence spans need LLM refinement or annotation")
    print("   • Low confidence spans should be prioritized for human review")


def demo_stratify_by_label():
    """Demonstrate stratification by entity label."""
    print("\n" + "=" * 70)
    print("🏷️  STRATIFICATION BY LABEL")
    print("=" * 70)

    weak_spans, gold_spans = create_diverse_sample_data()

    # Stratify by label
    stratified = stratify_by_label(weak_spans)

    print("\n🔍 Entity Type Performance:\n")

    for label, label_spans in stratified.items():
        # Get gold spans for this label
        gold_label_spans = [s for s in gold_spans if s["label"] == label]

        # Compute metrics
        metrics = compute_precision_recall_f1(label_spans, gold_label_spans)
        boundary = compute_boundary_precision(label_spans, gold_label_spans)

        # Confidence stats
        confidences = [s["confidence"] for s in label_spans]
        mean_conf = sum(confidences) / len(confidences)

        print(f"   {label}:")
        print(f"      Span Count:      {len(label_spans)} weak / {len(gold_label_spans)} gold")
        print(f"      Mean Confidence: {mean_conf:.3f}")
        print(f"      Precision:       {metrics['precision']:.3f}")
        print(f"      Recall:          {metrics['recall']:.3f}")
        print(f"      F1:              {metrics['f1']:.3f}")
        print(f"      Mean IOU:        {boundary['mean_iou']:.3f}")
        print()

    print("💡 Interpretation:")
    symptom_f1 = compute_precision_recall_f1(
        [s for s in weak_spans if s["label"] == "SYMPTOM"],
        [s for s in gold_spans if s["label"] == "SYMPTOM"],
    )["f1"]
    product_f1 = compute_precision_recall_f1(
        [s for s in weak_spans if s["label"] == "PRODUCT"],
        [s for s in gold_spans if s["label"] == "PRODUCT"],
    )["f1"]

    if symptom_f1 < product_f1:
        print("   → SYMPTOM extraction is weaker. Consider:")
        print("      • Expanding symptom lexicon")
        print("      • Improving fuzzy matching thresholds")
        print("      • Adding more symptom-focused annotation")
    else:
        print("   → PRODUCT extraction is weaker. Consider:")
        print("      • Improving product name normalization")
        print("      • Handling brand name variations")
        print("      • Adding product-specific LLM refinement")


def demo_stratify_by_span_length():
    """Demonstrate stratification by span character length."""
    print("\n" + "=" * 70)
    print("📏 STRATIFICATION BY SPAN LENGTH")
    print("=" * 70)

    weak_spans, gold_spans = create_diverse_sample_data()

    # Stratify by length buckets
    buckets = [(0, 10), (10, 20), (20, 100)]
    stratified = stratify_by_span_length(weak_spans, buckets=buckets)

    print("\n🔍 Length Bucket Performance:\n")

    for bucket_name, bucket_spans in stratified.items():
        if not bucket_spans:
            continue

        # Compute metrics
        metrics = compute_precision_recall_f1(bucket_spans, gold_spans)
        boundary = compute_boundary_precision(bucket_spans, gold_spans)

        # Length stats
        lengths = [len(s["text"]) for s in bucket_spans]
        mean_len = sum(lengths) / len(lengths)

        print(f"   {bucket_name.upper().replace('_', ' ')}:")
        print(f"      Span Count:      {len(bucket_spans)}")
        print(f"      Mean Length:     {mean_len:.1f} chars")
        print(f"      Precision:       {metrics['precision']:.3f}")
        print(f"      Recall:          {metrics['recall']:.3f}")
        print(f"      F1:              {metrics['f1']:.3f}")
        print(f"      Exact Match:     {boundary['exact_match_rate']:.1%}")
        print()

    print("💡 Interpretation:")
    print("   • Short spans (<10 chars) often have boundary issues")
    print("   • Long spans (>20 chars) may include superfluous adjectives")
    print("   • Target LLM refinement on problematic length ranges")


def demo_combined_stratification():
    """Demonstrate combining multiple stratification criteria."""
    print("\n" + "=" * 70)
    print("🔬 COMBINED STRATIFICATION")
    print("=" * 70)

    weak_spans, gold_spans = create_diverse_sample_data()

    print("\n🔍 Cross-Analysis: Label × Confidence:\n")

    # Stratify by label first
    by_label = stratify_by_label(weak_spans)

    for label, label_spans in by_label.items():
        print(f"   {label}:")

        # Stratify each label by confidence
        by_conf = stratify_by_confidence(label_spans, n_bins=2)

        for conf_bucket, conf_spans in by_conf.items():
            if not conf_spans:
                continue

            metrics = compute_precision_recall_f1(
                conf_spans, [s for s in gold_spans if s["label"] == label]
            )

            confidences = [s["confidence"] for s in conf_spans]
            conf_range = f"{min(confidences):.2f}-{max(confidences):.2f}"

            print(f"      {conf_bucket.replace('_', ' ').title()} ({conf_range}):")
            print(f"         Count: {len(conf_spans)}, F1: {metrics['f1']:.3f}")

        print()

    print("💡 Interpretation:")
    print("   • High-confidence SYMPTOMS should be reliable (F1 > 0.90)")
    print("   • Low-confidence PRODUCTS may need brand name expansion")
    print("   • Target annotation on worst-performing cells")


def demo_identify_weak_spots():
    """Identify specific weak spots requiring attention."""
    print("\n" + "=" * 70)
    print("🎯 IDENTIFYING WEAK SPOTS")
    print("=" * 70)

    weak_spans, gold_spans = create_diverse_sample_data()

    # Collect all stratified metrics
    issues = []

    # Check confidence buckets
    by_conf = stratify_by_confidence(weak_spans, n_bins=3)
    for bucket_name, bucket_spans in by_conf.items():
        if not bucket_spans:
            continue
        metrics = compute_precision_recall_f1(bucket_spans, gold_spans)
        if metrics["f1"] < 0.75:
            issues.append(
                {
                    "type": "Confidence",
                    "category": bucket_name.replace("_", " ").title(),
                    "f1": metrics["f1"],
                    "count": len(bucket_spans),
                    "priority": "HIGH" if metrics["f1"] < 0.5 else "MEDIUM",
                }
            )

    # Check label performance
    by_label = stratify_by_label(weak_spans)
    for label, label_spans in by_label.items():
        gold_label = [s for s in gold_spans if s["label"] == label]
        metrics = compute_precision_recall_f1(label_spans, gold_label)
        if metrics["f1"] < 0.80:
            issues.append(
                {
                    "type": "Label",
                    "category": label,
                    "f1": metrics["f1"],
                    "count": len(label_spans),
                    "priority": "HIGH" if metrics["f1"] < 0.60 else "MEDIUM",
                }
            )

    # Check span length
    by_length = stratify_by_span_length(weak_spans, buckets=[(0, 10), (10, 20), (20, 100)])
    for bucket_name, bucket_spans in by_length.items():
        if not bucket_spans:
            continue
        boundary = compute_boundary_precision(bucket_spans, gold_spans)
        if boundary["exact_match_rate"] < 0.70:
            issues.append(
                {
                    "type": "Length",
                    "category": bucket_name.replace("_", " ").title(),
                    "f1": boundary["exact_match_rate"],
                    "count": len(bucket_spans),
                    "priority": "MEDIUM",
                }
            )

    # Display issues
    if issues:
        print("\n⚠️  Identified Issues:\n")

        # Sort by priority and F1
        issues.sort(key=lambda x: (x["priority"] != "HIGH", x["f1"]))

        for i, issue in enumerate(issues, 1):
            print(f"   {i}. [{issue['priority']}] {issue['type']}: {issue['category']}")
            print(f"      F1/Exact Match: {issue['f1']:.3f}")
            print(f"      Affected Spans: {issue['count']}")
            print(f"      Recommendation: ", end="")

            if issue["type"] == "Confidence" and "low" in issue["category"].lower():
                print("Prioritize for human annotation")
            elif issue["type"] == "Label" and issue["category"] == "SYMPTOM":
                print("Expand symptom lexicon, improve fuzzy matching")
            elif issue["type"] == "Label" and issue["category"] == "PRODUCT":
                print("Add product variants, improve brand recognition")
            elif issue["type"] == "Length" and "long" in issue["category"].lower():
                print("Target LLM refinement for adjective removal")
            else:
                print("Review heuristics and add targeted examples")
            print()
    else:
        print("\n✓ No critical issues found! Performance is acceptable across all strata.")

    print("💡 Next Steps:")
    print("   • Focus annotation efforts on HIGH priority issues")
    print("   • Use LLM refinement for boundary problems")
    print("   • Expand lexicons for low-recall categories")


def main():
    """Run all stratified analysis examples."""
    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates breaking down evaluation metrics by confidence,")
    print("entity type, and span length to identify performance patterns")
    print("and prioritize improvements.")

    # Run demos
    demo_stratify_by_confidence()
    demo_stratify_by_label()
    demo_stratify_by_span_length()
    demo_combined_stratification()
    demo_identify_weak_spots()

    print("\n" + "=" * 70)
    print("✓ All stratified analysis examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Try visualization.py for calibration curves and plots")
    print("   • Use compare_baselines.py to benchmark different approaches")
    print("   • Apply insights to annotation/prepare_batch.py for targeted sampling")
    print()


if __name__ == "__main__":
    main()
