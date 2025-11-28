"""
Evaluation Example: Computing NER Metrics
==========================================

Demonstrates how to evaluate weak labels, LLM-refined labels, and gold standard
annotations using SpanForge's evaluation harness.

**What You'll Learn:**
- Computing precision, recall, and F1 scores
- Calculating boundary precision (exact match rate, mean IOU)
- Measuring IOU delta (weak → LLM improvement)
- Understanding correction rates (improved/worsened/unchanged)
- Generating evaluation reports

**Prerequisites:**
- Completed basic/simple_ner.py
- Understanding of NER evaluation metrics
- Familiarity with weak labeling concepts

**Runtime:** ~30 seconds

**Use Cases:**
- Validating weak labeling quality before annotation
- Measuring LLM refinement effectiveness
- Comparing annotation approaches
- Quality assurance for production pipelines
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    calibration_curve,
    compute_boundary_precision,
    compute_correction_rate,
    compute_iou_delta,
    compute_precision_recall_f1,
)


def create_sample_data():
    """Create sample weak, refined, and gold standard annotations."""

    # Sample text
    text = "After using the facial moisturizer, I experienced severe burning sensation and redness on my cheeks."

    # Weak labels (from lexicon-based extraction)
    weak_spans = [
        {
            "text": "severe burning sensation",  # Too broad
            "start": 55,
            "end": 79,
            "label": "SYMPTOM",
            "confidence": 0.85,
        },
        {"text": "redness", "start": 84, "end": 91, "label": "SYMPTOM", "confidence": 1.0},
        {
            "text": "facial moisturizer",
            "start": 16,
            "end": 34,
            "label": "PRODUCT",
            "confidence": 0.95,
        },
    ]

    # LLM-refined labels (boundary corrections)
    llm_spans = [
        {
            "text": "burning sensation",  # Removed "severe"
            "start": 62,
            "end": 79,
            "label": "SYMPTOM",
            "confidence": 0.95,
        },
        {"text": "redness", "start": 84, "end": 91, "label": "SYMPTOM", "confidence": 1.0},
        {
            "text": "facial moisturizer",
            "start": 16,
            "end": 34,
            "label": "PRODUCT",
            "confidence": 0.95,
        },
    ]

    # Gold standard (human-annotated)
    gold_spans = [
        {"text": "burning sensation", "start": 62, "end": 79, "label": "SYMPTOM"},
        {"text": "redness", "start": 84, "end": 91, "label": "SYMPTOM"},
        {"text": "facial moisturizer", "start": 16, "end": 34, "label": "PRODUCT"},
    ]

    return text, weak_spans, llm_spans, gold_spans


def demo_precision_recall_f1():
    """Demonstrate P/R/F1 computation."""
    print("\n" + "=" * 70)
    print("📊 PRECISION, RECALL, F1")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute metrics for weak labels
    weak_metrics = compute_precision_recall_f1(weak_spans, gold_spans)

    print("\n🔍 Weak Labels vs Gold:")
    print(f"   Precision: {weak_metrics['precision']:.3f}")
    print(f"   Recall:    {weak_metrics['recall']:.3f}")
    print(f"   F1 Score:  {weak_metrics['f1']:.3f}")
    print(f"   Predicted: {len(weak_spans)}, Gold: {len(gold_spans)}")

    # Compute metrics for LLM-refined labels
    llm_metrics = compute_precision_recall_f1(llm_spans, gold_spans)

    print("\n✨ LLM-Refined vs Gold:")
    print(f"   Precision: {llm_metrics['precision']:.3f}")
    print(f"   Recall:    {llm_metrics['recall']:.3f}")
    print(f"   F1 Score:  {llm_metrics['f1']:.3f}")
    print(f"   Predicted: {len(llm_spans)}, Gold: {len(gold_spans)}")

    # Show improvement
    f1_improvement = llm_metrics["f1"] - weak_metrics["f1"]
    print(f"\n📈 F1 Improvement: {f1_improvement:+.3f} ({f1_improvement*100:+.1f}%)")


def demo_boundary_precision():
    """Demonstrate boundary precision calculation."""
    print("\n" + "=" * 70)
    print("🎯 BOUNDARY PRECISION")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute boundary precision for weak labels
    weak_boundary = compute_boundary_precision(weak_spans, gold_spans)

    print("\n🔍 Weak Labels:")
    print(f"   Exact Match Rate: {weak_boundary['exact_match_rate']:.1%}")
    print(f"   Mean IOU:         {weak_boundary['mean_iou']:.3f}")
    print(f"   Median IOU:       {weak_boundary['median_iou']:.3f}")

    # Compute boundary precision for LLM-refined labels
    llm_boundary = compute_boundary_precision(llm_spans, gold_spans)

    print("\n✨ LLM-Refined:")
    print(f"   Exact Match Rate: {llm_boundary['exact_match_rate']:.1%}")
    print(f"   Mean IOU:         {llm_boundary['mean_iou']:.3f}")
    print(f"   Median IOU:       {llm_boundary['median_iou']:.3f}")

    # Show improvement
    exact_match_improvement = llm_boundary["exact_match_rate"] - weak_boundary["exact_match_rate"]
    iou_improvement = llm_boundary["mean_iou"] - weak_boundary["mean_iou"]

    print(f"\n📈 Improvements:")
    print(f"   Exact Match: {exact_match_improvement:+.1%}")
    print(f"   Mean IOU:    {iou_improvement:+.3f}")


def demo_iou_delta():
    """Demonstrate IOU delta tracking."""
    print("\n" + "=" * 70)
    print("📊 IOU DELTA (Weak → LLM Improvement)")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute IOU delta
    iou_delta = compute_iou_delta(weak_spans, llm_spans, gold_spans)

    print(f"\n🔍 Weak Labels:")
    print(f"   Mean IOU: {iou_delta['weak_mean_iou']:.3f}")

    print(f"\n✨ LLM-Refined:")
    print(f"   Mean IOU: {iou_delta['llm_mean_iou']:.3f}")

    print(f"\n📈 Delta:")
    print(f"   Improvement: {iou_delta['delta']:+.3f}")
    print(f"   Percentage:  {iou_delta['improvement_pct']:+.1f}%")

    print("\n💡 Interpretation:")
    if iou_delta["delta"] > 0.05:
        print("   ✓ Significant improvement! LLM refinement is effective.")
    elif iou_delta["delta"] > 0:
        print("   → Modest improvement. Consider tuning LLM prompts.")
    else:
        print("   ⚠ No improvement. Review weak labeling heuristics.")


def demo_correction_rate():
    """Demonstrate correction rate analysis."""
    print("\n" + "=" * 70)
    print("🔧 CORRECTION RATE ANALYSIS")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute correction rates
    correction = compute_correction_rate(weak_spans, llm_spans, gold_spans)

    print(f"\n📊 Span-Level Corrections:")
    print(f"   Total Spans:  {correction['total_spans']}")
    print(f"   Modified:     {correction['modified_count']}")
    print(f"   Improved:     {correction['improved_count']}")
    print(f"   Worsened:     {correction['worsened_count']}")
    print(f"   Unchanged:    {correction['unchanged_count']}")
    print(f"   Improvement Rate:   {correction['improvement_rate']:.1%}")
    print(f"   False Refinement:   {correction['false_refinement_rate']:.1%}")

    print("\n💡 Interpretation:")
    if correction["improvement_rate"] > 0.50:
        print(f"   ✓ High success rate! LLM is making good corrections.")
    elif correction["false_refinement_rate"] > 0.20:
        print(f"   ⚠ High false refinement rate! Review LLM prompt and examples.")
    else:
        print(f"   → Mixed results. Consider targeted LLM refinement.")


def demo_calibration():
    """Demonstrate confidence calibration analysis."""
    print("\n" + "=" * 70)
    print("📈 CONFIDENCE CALIBRATION")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute calibration curve
    calibration = calibration_curve(weak_spans, gold_spans, num_bins=3)

    print("\n🔍 Calibration Analysis:")
    print("   (Shows if confidence scores match actual accuracy)\n")

    for i, (center, acc, count) in enumerate(
        zip(calibration["bin_centers"], calibration["accuracy"], calibration["counts"])
    ):
        if count == 0:
            continue
        conf_min = max(0.0, center - 1.0 / 6)
        conf_max = min(1.0, center + 1.0 / 6)
        print(f"   Confidence {conf_min:.2f}-{conf_max:.2f}:")
        print(f"      Expected IOU: {center:.3f}")
        print(f"      Actual Accuracy:   {acc:.3f}")
        print(f"      Span Count:   {count}")
        print()

    print("💡 Interpretation:")
    print("   Well-calibrated: Expected ≈ Actual")
    print("   Overconfident:   Expected > Actual")
    print("   Underconfident:  Expected < Actual")


def demo_complete_report():
    """Generate a complete evaluation report."""
    print("\n" + "=" * 70)
    print("📋 COMPLETE EVALUATION REPORT")
    print("=" * 70)

    text, weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute all metrics
    weak_prf = compute_precision_recall_f1(weak_spans, gold_spans)
    llm_prf = compute_precision_recall_f1(llm_spans, gold_spans)
    weak_boundary = compute_boundary_precision(weak_spans, gold_spans)
    llm_boundary = compute_boundary_precision(llm_spans, gold_spans)
    iou_delta = compute_iou_delta(weak_spans, llm_spans, gold_spans)
    correction = compute_correction_rate(weak_spans, llm_spans, gold_spans)

    print("\n" + "─" * 70)
    print("OVERALL PERFORMANCE")
    print("─" * 70)

    print("\n                    Weak Labels    LLM-Refined    Delta")
    print("                    ───────────    ───────────    ─────")
    print(
        f"Precision           {weak_prf['precision']:>7.3f}        {llm_prf['precision']:>7.3f}        {llm_prf['precision']-weak_prf['precision']:+.3f}"
    )
    print(
        f"Recall              {weak_prf['recall']:>7.3f}        {llm_prf['recall']:>7.3f}        {llm_prf['recall']-weak_prf['recall']:+.3f}"
    )
    print(
        f"F1 Score            {weak_prf['f1']:>7.3f}        {llm_prf['f1']:>7.3f}        {llm_prf['f1']-weak_prf['f1']:+.3f}"
    )
    print(
        f"Exact Match Rate    {weak_boundary['exact_match_rate']:>6.1%}         {llm_boundary['exact_match_rate']:>6.1%}         {llm_boundary['exact_match_rate']-weak_boundary['exact_match_rate']:+.1%}"
    )
    print(
        f"Mean IOU            {weak_boundary['mean_iou']:>7.3f}        {llm_boundary['mean_iou']:>7.3f}        {llm_boundary['mean_iou']-weak_boundary['mean_iou']:+.3f}"
    )

    print("\n" + "─" * 70)
    print("CORRECTION ANALYSIS")
    print("─" * 70)
    print(
        f"\nImproved:   {correction['improved_count']:>2} spans ({correction['improvement_rate']*100:>5.1f}%)"
    )
    print(
        f"Worsened:   {correction['worsened_count']:>2} spans ({correction['false_refinement_rate']*100:>5.1f}%)"
    )
    print(f"Unchanged:  {correction['unchanged_count']:>2} spans")

    print("\n" + "─" * 70)
    print("RECOMMENDATION")
    print("─" * 70)

    if llm_prf["f1"] > 0.95:
        print("\n✓ Excellent performance! Ready for production use.")
    elif llm_prf["f1"] > 0.85:
        print("\n→ Good performance. Consider additional annotation for edge cases.")
    else:
        print("\n⚠ Needs improvement. Review weak labeling heuristics and LLM prompts.")


def main():
    """Run all evaluation examples."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates SpanForge's evaluation harness for measuring")
    print("weak label quality, LLM refinement effectiveness, and overall")
    print("NER performance against gold standard annotations.")

    # Run demos
    demo_precision_recall_f1()
    demo_boundary_precision()
    demo_iou_delta()
    demo_correction_rate()
    demo_calibration()
    demo_complete_report()

    print("\n" + "=" * 70)
    print("✓ All evaluation examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Try stratified_analysis.py for confidence/label/length breakdowns")
    print("   • See visualization.py for calibration curves and plots")
    print("   • Use compare_baselines.py to benchmark against RoBERTa")
    print()


if __name__ == "__main__":
    main()
