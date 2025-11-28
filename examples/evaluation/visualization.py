"""
Evaluation Example: Visualization
==================================

Demonstrates how to create publication-quality visualizations for NER evaluation,
including calibration curves, performance comparison plots, and stratified analysis
charts. Note: Requires optional visualization dependencies.

**What You'll Learn:**
- Creating calibration curves (confidence reliability)
- Plotting IOU uplift (weak vs LLM comparison)
- Generating correction rate breakdowns
- Visualizing stratified metrics
- Producing publication-ready figures

**Prerequisites:**
- Completed evaluation/compute_metrics.py
- Completed evaluation/stratified_analysis.py
- Optional: matplotlib, seaborn (install via requirements-viz.txt)

**Runtime:** ~1 minute (or instant if matplotlib not installed)

**Use Cases:**
- Creating evaluation reports for stakeholders
- Publishing research results
- Identifying visual performance patterns
- Quality assurance dashboards
- Academic presentations
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import (
    calibration_curve,
    compute_boundary_precision,
    compute_precision_recall_f1,
    stratify_by_confidence,
    stratify_by_label,
)

# Try to import visualization libraries
try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def create_sample_data():
    """Create sample evaluation data for visualization."""

    # Weak labels
    weak_spans = [
        {"text": "severe burning", "start": 0, "end": 14, "label": "SYMPTOM", "confidence": 0.85},
        {"text": "redness", "start": 20, "end": 27, "label": "SYMPTOM", "confidence": 1.0},
        {"text": "mild itching", "start": 30, "end": 42, "label": "SYMPTOM", "confidence": 0.70},
        {"text": "face cream", "start": 50, "end": 60, "label": "PRODUCT", "confidence": 0.95},
        {"text": "discomfort", "start": 65, "end": 75, "label": "SYMPTOM", "confidence": 0.60},
    ]

    # LLM-refined
    llm_spans = [
        {"text": "burning", "start": 7, "end": 14, "label": "SYMPTOM", "confidence": 0.95},
        {"text": "redness", "start": 20, "end": 27, "label": "SYMPTOM", "confidence": 1.0},
        {"text": "itching", "start": 35, "end": 42, "label": "SYMPTOM", "confidence": 0.88},
        {"text": "face cream", "start": 50, "end": 60, "label": "PRODUCT", "confidence": 0.95},
        {"text": "discomfort", "start": 65, "end": 75, "label": "SYMPTOM", "confidence": 0.75},
    ]

    # Gold standard
    gold_spans = [
        {"text": "burning", "start": 7, "end": 14, "label": "SYMPTOM"},
        {"text": "redness", "start": 20, "end": 27, "label": "SYMPTOM"},
        {"text": "itching", "start": 35, "end": 42, "label": "SYMPTOM"},
        {"text": "face cream", "start": 50, "end": 60, "label": "PRODUCT"},
        {"text": "discomfort", "start": 65, "end": 75, "label": "SYMPTOM"},
    ]

    return weak_spans, llm_spans, gold_spans


def demo_text_based_calibration():
    """Show text-based calibration curve (no plotting libraries needed)."""
    print("\n" + "=" * 70)
    print("📈 CALIBRATION CURVE (Text-Based)")
    print("=" * 70)

    weak_spans, _, gold_spans = create_sample_data()

    # Compute calibration
    calibration = calibration_curve(weak_spans, gold_spans, n_bins=3)

    print("\n🔍 Confidence Calibration:\n")
    print("   Expected vs Actual IOU by Confidence Bucket\n")

    # Create text-based bar chart
    for bin_data in calibration["bins"]:
        conf_label = f"{bin_data['confidence_min']:.2f}-{bin_data['confidence_max']:.2f}"
        expected = bin_data["mean_confidence"]
        actual = bin_data["mean_iou"]
        count = bin_data["count"]

        # Scale bars to 40 chars max
        expected_bar = "█" * int(expected * 40)
        actual_bar = "█" * int(actual * 40)

        print(f"   Conf {conf_label}:")
        print(f"      Expected: {expected_bar} {expected:.3f}")
        print(f"      Actual:   {actual_bar} {actual:.3f}")
        print(f"      Count:    {count} spans")

        # Show calibration status
        diff = actual - expected
        if abs(diff) < 0.05:
            print(f"      Status:   ✓ Well-calibrated")
        elif diff > 0:
            print(f"      Status:   → Underconfident ({diff:+.3f})")
        else:
            print(f"      Status:   ⚠ Overconfident ({diff:+.3f})")
        print()

    print("💡 Interpretation:")
    print("   • Well-calibrated: Confidence scores match actual performance")
    print("   • Overconfident: Predictions are less accurate than confidence suggests")
    print("   • Underconfident: Predictions are better than confidence suggests")


def demo_text_based_comparison():
    """Show text-based performance comparison."""
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON (Text-Based)")
    print("=" * 70)

    weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute metrics
    weak_metrics = compute_precision_recall_f1(weak_spans, gold_spans)
    llm_metrics = compute_precision_recall_f1(llm_spans, gold_spans)
    weak_boundary = compute_boundary_precision(weak_spans, gold_spans)
    llm_boundary = compute_boundary_precision(llm_spans, gold_spans)

    print("\n🔍 Weak vs LLM-Refined Comparison:\n")

    metrics = [
        ("Precision", weak_metrics["precision"], llm_metrics["precision"]),
        ("Recall", weak_metrics["recall"], llm_metrics["recall"]),
        ("F1 Score", weak_metrics["f1"], llm_metrics["f1"]),
        ("Mean IOU", weak_boundary["mean_iou"], llm_boundary["mean_iou"]),
    ]

    for name, weak_val, llm_val in metrics:
        # Scale bars to 40 chars max
        weak_bar = "█" * int(weak_val * 40)
        llm_bar = "█" * int(llm_val * 40)
        delta = llm_val - weak_val

        print(f"   {name:12}:")
        print(f"      Weak: {weak_bar:<40} {weak_val:.3f}")
        print(f"      LLM:  {llm_bar:<40} {llm_val:.3f} ({delta:+.3f})")
        print()


def demo_text_based_stratified():
    """Show text-based stratified analysis."""
    print("\n" + "=" * 70)
    print("🏷️  STRATIFIED BY LABEL (Text-Based)")
    print("=" * 70)

    weak_spans, llm_spans, gold_spans = create_sample_data()

    # Stratify by label
    weak_by_label = stratify_by_label(weak_spans)
    llm_by_label = stratify_by_label(llm_spans)

    print("\n🔍 F1 Score by Entity Type:\n")

    for label in ["SYMPTOM", "PRODUCT"]:
        weak_label_spans = weak_by_label.get(label, [])
        llm_label_spans = llm_by_label.get(label, [])
        gold_label_spans = [s for s in gold_spans if s["label"] == label]

        if not weak_label_spans:
            continue

        weak_metrics = compute_precision_recall_f1(weak_label_spans, gold_label_spans)
        llm_metrics = compute_precision_recall_f1(llm_label_spans, gold_label_spans)

        weak_f1 = weak_metrics["f1"]
        llm_f1 = llm_metrics["f1"]
        delta = llm_f1 - weak_f1

        # Create bars
        weak_bar = "█" * int(weak_f1 * 40)
        llm_bar = "█" * int(llm_f1 * 40)

        print(f"   {label}:")
        print(f"      Weak: {weak_bar:<40} {weak_f1:.3f}")
        print(f"      LLM:  {llm_bar:<40} {llm_f1:.3f} ({delta:+.3f})")
        print()


def demo_matplotlib_plots():
    """Show how to create matplotlib plots (if available)."""
    if not HAS_MATPLOTLIB:
        print("\n" + "=" * 70)
        print("📊 MATPLOTLIB PLOTS (Not Available)")
        print("=" * 70)
        print("\n⚠️  matplotlib not installed. To enable plotting:")
        print("   pip install -r requirements-viz.txt")
        print("\n   This will install:")
        print("   • matplotlib (plotting)")
        print("   • seaborn (styling)")
        print("   • numpy (numerical operations)")
        return

    print("\n" + "=" * 70)
    print("📊 MATPLOTLIB PLOTS")
    print("=" * 70)

    weak_spans, llm_spans, gold_spans = create_sample_data()

    # Compute metrics
    weak_metrics = compute_precision_recall_f1(weak_spans, gold_spans)
    llm_metrics = compute_precision_recall_f1(llm_spans, gold_spans)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Precision/Recall/F1 comparison
    metrics_names = ["Precision", "Recall", "F1"]
    weak_values = [weak_metrics["precision"], weak_metrics["recall"], weak_metrics["f1"]]
    llm_values = [llm_metrics["precision"], llm_metrics["recall"], llm_metrics["f1"]]

    x = range(len(metrics_names))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], weak_values, width, label="Weak Labels", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], llm_values, width, label="LLM-Refined", alpha=0.8)
    ax1.set_ylabel("Score")
    ax1.set_title("Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Calibration curve
    calibration = calibration_curve(weak_spans, gold_spans, n_bins=3)

    expected = [b["mean_confidence"] for b in calibration["bins"]]
    actual = [b["mean_iou"] for b in calibration["bins"]]
    labels = [f"{b['confidence_min']:.2f}-{b['confidence_max']:.2f}" for b in calibration["bins"]]

    x2 = range(len(expected))
    ax2.plot(x2, expected, "o-", label="Expected IOU", linewidth=2, markersize=8)
    ax2.plot(x2, actual, "s-", label="Actual IOU", linewidth=2, markersize=8)
    ax2.plot(
        x2, [expected[i] for i in x2], "--", color="gray", alpha=0.5, label="Perfect Calibration"
    )
    ax2.set_xlabel("Confidence Bucket")
    ax2.set_ylabel("IOU")
    ax2.set_title("Calibration Curve")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.set_ylim([0, 1.1])
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent.parent / "data" / "annotation" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_example.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")
    print("\n💡 To view the plot:")
    print(f"   1. Navigate to: {output_dir}")
    print(f"   2. Open: evaluation_example.png")


def demo_production_script():
    """Show how to use the production visualization script."""
    print("\n" + "=" * 70)
    print("🎨 PRODUCTION VISUALIZATION SCRIPT")
    print("=" * 70)

    print("\n📋 For real evaluation reports, use the production script:")
    print("\n   python scripts/annotation/plot_llm_metrics.py \\")
    print("       --report data/annotation/reports/evaluation.json \\")
    print("       --output-dir data/annotation/plots/ \\")
    print("       --formats png pdf \\")
    print("       --dpi 300 \\")
    print("       --plots all")

    print("\n📊 Generated Plots:")
    print("   1. iou_uplift.png - Weak vs LLM IOU distribution")
    print("   2. calibration_curve.png - Confidence reliability")
    print("   3. correction_rate.png - Improved/worsened/unchanged breakdown")
    print("   4. prf_comparison.png - P/R/F1 side-by-side")
    print("   5. stratified_label.png - F1 by entity type")
    print("   6. stratified_confidence.png - IOU delta by confidence")

    print("\n⚙️  Configuration:")
    print("   • Publication-quality (300 DPI)")
    print("   • Colorblind-safe palette")
    print("   • Annotated with counts and deltas")
    print("   • Multiple formats (PNG, PDF, SVG)")

    print("\n💡 Prerequisites:")
    if HAS_MATPLOTLIB and HAS_SEABORN:
        print("   ✓ matplotlib installed")
        print("   ✓ seaborn installed")
        print("   ✓ Ready to generate plots!")
    else:
        print("   ⚠️  Install visualization dependencies:")
        print("      pip install -r requirements-viz.txt")


def main():
    """Run all visualization examples."""
    print("\n" + "=" * 70)
    print("VISUALIZATION EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates creating publication-quality visualizations for")
    print("NER evaluation, including calibration curves, performance")
    print("comparisons, and stratified analysis charts.")

    # Always show text-based visualizations
    demo_text_based_calibration()
    demo_text_based_comparison()
    demo_text_based_stratified()

    # Show matplotlib examples if available
    demo_matplotlib_plots()

    # Show production script info
    demo_production_script()

    print("\n" + "=" * 70)
    print("✓ All visualization examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Install requirements-viz.txt for full plotting capabilities")
    print("   • Use scripts/annotation/evaluate_llm_refinement.py for reports")
    print("   • Try compare_baselines.py to benchmark different models")
    print()


if __name__ == "__main__":
    main()
