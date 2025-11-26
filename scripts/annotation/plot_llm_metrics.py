#!/usr/bin/env python3
"""
Visualization helper for LLM refinement evaluation metrics.

Generates publication-quality plots from evaluation JSON reports:
- IOU uplift histogram (weak vs LLM distribution)
- Confidence calibration curve (expected vs observed)
- Correction rate breakdown (improved/worsened/unchanged)
- Precision/Recall/F1 comparison bar chart
- Precision-Recall curve

Usage:
    python scripts/annotation/plot_llm_metrics.py \\
        --report data/annotation/reports/evaluation.json \\
        --output-dir data/annotation/plots/ \\
        --formats png pdf \\
        --dpi 300

Requirements:
    pip install matplotlib seaborn numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
except ImportError as e:
    print(f"ERROR: Missing visualization dependencies: {e}")
    print("Install with: pip install matplotlib seaborn numpy")
    sys.exit(1)

# Set publication-quality defaults
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    }
)


def load_report(report_path: Path) -> Dict[str, Any]:
    """Load evaluation JSON report."""
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_iou_uplift(report: Dict[str, Any], output_dir: Path, formats: List[str], dpi: int) -> None:
    """
    Generate IOU uplift histogram comparing weak vs LLM distributions.

    Shows before/after distribution of IOU scores across all spans.
    """
    overall = report.get("overall", {})
    weak_iou = overall.get("weak_mean_iou", 0.0)
    llm_iou = overall.get("llm_mean_iou", 0.0)
    delta = overall.get("iou_delta", 0.0)

    # Create synthetic distribution (in real use, would come from per-span IOUs)
    # For now, use stratified data if available
    stratified = report.get("stratified", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    # If we have stratified confidence data, use it to approximate distribution
    confidence_strata = stratified.get("confidence", {})
    if confidence_strata:
        buckets = []
        weak_ious = []
        llm_ious = []

        for bucket_name, bucket_data in confidence_strata.items():
            weak_ious.append(bucket_data.get("weak_mean_iou", 0.0))
            llm_ious.append(bucket_data.get("llm_mean_iou", 0.0))
            buckets.append(bucket_name)

        x = np.arange(len(buckets))
        width = 0.35

        bars1 = ax.bar(x - width / 2, weak_ious, width, label="Weak Labels", alpha=0.8)
        bars2 = ax.bar(x + width / 2, llm_ious, width, label="LLM Refined", alpha=0.8)

        ax.set_xlabel("Confidence Bucket")
        ax.set_ylabel("Mean IOU")
        ax.set_title(f"IOU Uplift by Confidence (+{delta:.1%} overall)")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)

        # Add delta annotations
        for i, (w, l) in enumerate(zip(weak_ious, llm_ious)):
            if l > w:
                ax.annotate(
                    f"+{l-w:.2f}", xy=(i, max(w, l) + 0.02), ha="center", fontsize=8, color="green"
                )
    else:
        # Fallback: simple bar chart of overall means
        categories = ["Weak Labels", "LLM Refined"]
        values = [weak_iou, llm_iou]
        colors = ["#1f77b4", "#2ca02c"]

        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel("Mean IOU")
        ax.set_title(f"Overall IOU Improvement (+{delta:.1%})")
        ax.set_ylim(0, 1.05)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"iou_uplift.{fmt}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_calibration_curve(
    report: Dict[str, Any], output_dir: Path, formats: List[str], dpi: int
) -> None:
    """
    Generate confidence calibration curve.

    Shows how well confidence scores predict actual accuracy (IOU).
    Perfect calibration: diagonal line where confidence = IOU.
    """
    calibration = report.get("overall", {}).get("calibration", {})

    if not calibration or "bins" not in calibration:
        print("  Skipping calibration curve (no data in report)")
        return

    bins = calibration["bins"]
    bin_centers = [b["bin_center"] for b in bins]
    mean_confidence = [b["mean_confidence"] for b in bins]
    mean_iou = [b["mean_iou"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot actual calibration
    scatter = ax.scatter(
        mean_confidence, mean_iou, s=[c * 20 for c in counts], alpha=0.6, label="Actual"
    )

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    # Connect points with line
    ax.plot(mean_confidence, mean_iou, "o-", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Mean Observed IOU")
    ax.set_title("Confidence Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with calibration metrics
    textstr = f"Bins: {len(bins)}\nTotal spans: {sum(counts)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props
    )

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"calibration_curve.{fmt}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_correction_rate(
    report: Dict[str, Any], output_dir: Path, formats: List[str], dpi: int
) -> None:
    """
    Generate correction rate breakdown pie/bar chart.

    Shows distribution of improved/worsened/unchanged spans.
    """
    correction = report.get("overall", {}).get("correction_rate", {})

    improved = correction.get("improved", 0)
    worsened = correction.get("worsened", 0)
    unchanged = correction.get("unchanged", 0)
    total = improved + worsened + unchanged

    if total == 0:
        print("  Skipping correction rate plot (no data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    labels = []
    sizes = []
    colors = []
    explode = []

    if improved > 0:
        labels.append(f"Improved\n({improved})")
        sizes.append(improved)
        colors.append("#2ca02c")
        explode.append(0.05)

    if worsened > 0:
        labels.append(f"Worsened\n({worsened})")
        sizes.append(worsened)
        colors.append("#d62728")
        explode.append(0.05)

    if unchanged > 0:
        labels.append(f"Unchanged\n({unchanged})")
        sizes.append(unchanged)
        colors.append("#7f7f7f")
        explode.append(0)

    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.set_title(f"Correction Breakdown (n={total})")

    # Bar chart with percentages
    categories = ["Improved", "Worsened", "Unchanged"]
    values = [improved / total * 100, worsened / total * 100, unchanged / total * 100]
    bar_colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    bars = ax2.bar(categories, values, color=bar_colors, alpha=0.8)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Correction Rate Distribution")
    ax2.set_ylim(0, 105)

    # Add value labels
    for bar, val, count in zip(bars, values, [improved, worsened, unchanged]):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{val:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"correction_rate.{fmt}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_prf_comparison(
    report: Dict[str, Any], output_dir: Path, formats: List[str], dpi: int
) -> None:
    """
    Generate Precision/Recall/F1 comparison bar chart.

    Side-by-side comparison of weak vs LLM refined metrics.
    """
    overall = report.get("overall", {})
    weak_prf = overall.get("weak_prf", {})
    llm_prf = overall.get("llm_prf", {})

    metrics = ["Precision", "Recall", "F1"]
    weak_values = [
        weak_prf.get("precision", 0.0),
        weak_prf.get("recall", 0.0),
        weak_prf.get("f1", 0.0),
    ]
    llm_values = [llm_prf.get("precision", 0.0), llm_prf.get("recall", 0.0), llm_prf.get("f1", 0.0)]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width / 2, weak_values, width, label="Weak Labels", alpha=0.8)
    bars2 = ax.bar(x + width / 2, llm_values, width, label="LLM Refined", alpha=0.8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall/F1 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Add delta annotations
    for i, (w, l) in enumerate(zip(weak_values, llm_values)):
        if abs(l - w) > 0.001:
            delta = l - w
            color = "green" if delta > 0 else "red"
            sign = "+" if delta > 0 else ""
            ax.annotate(
                f"{sign}{delta:.3f}",
                xy=(i, max(w, l) + 0.08),
                ha="center",
                fontsize=8,
                color=color,
                weight="bold",
            )

    plt.tight_layout()

    for fmt in formats:
        output_path = output_dir / f"prf_comparison.{fmt}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_stratified_comparison(
    report: Dict[str, Any], output_dir: Path, formats: List[str], dpi: int
) -> None:
    """
    Generate stratified analysis plots (by label, confidence, span length).

    Shows how LLM improvement varies across different subgroups.
    """
    stratified = report.get("stratified", {})

    if not stratified:
        print("  Skipping stratified plots (no stratified data)")
        return

    # Plot 1: By Label
    label_strata = stratified.get("label", {})
    if label_strata:
        labels = list(label_strata.keys())
        weak_f1 = [label_strata[l].get("weak_f1", 0.0) for l in labels]
        llm_f1 = [label_strata[l].get("llm_f1", 0.0) for l in labels]
        counts = [label_strata[l].get("span_count", 0) for l in labels]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, weak_f1, width, label="Weak Labels", alpha=0.8)
        bars2 = ax.bar(x + width / 2, llm_f1, width, label="LLM Refined", alpha=0.8)

        ax.set_xlabel("Label Type")
        ax.set_ylabel("F1 Score")
        ax.set_title("Performance by Label Type")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.05)

        # Add count annotations
        for i, count in enumerate(counts):
            ax.text(i, -0.1, f"n={count}", ha="center", va="top", fontsize=8)

        plt.tight_layout()

        for fmt in formats:
            output_path = output_dir / f"stratified_label.{fmt}"
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {output_path}")

        plt.close()

    # Plot 2: By Confidence
    confidence_strata = stratified.get("confidence", {})
    if confidence_strata:
        buckets = sorted(confidence_strata.keys())
        iou_deltas = [confidence_strata[b].get("iou_delta", 0.0) for b in buckets]
        counts = [confidence_strata[b].get("span_count", 0) for b in buckets]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(buckets)), iou_deltas, alpha=0.8)

        # Color bars by delta (green for positive, red for negative)
        for bar, delta in zip(bars, iou_deltas):
            bar.set_color("#2ca02c" if delta >= 0 else "#d62728")

        ax.set_xlabel("Confidence Bucket")
        ax.set_ylabel("IOU Delta (LLM - Weak)")
        ax.set_title("IOU Improvement by Confidence Level")
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets, rotation=45, ha="right")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # Add count and delta labels
        for i, (delta, count) in enumerate(zip(iou_deltas, counts)):
            ax.text(
                i,
                delta + (0.02 if delta >= 0 else -0.02),
                f"{delta:+.3f}\n(n={count})",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8,
            )

        plt.tight_layout()

        for fmt in formats:
            output_path = output_dir / f"stratified_confidence.{fmt}"
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {output_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from LLM evaluation reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--report", "-r", type=Path, required=True, help="Path to evaluation JSON report"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/annotation/plots"),
        help="Directory to save plots (default: data/annotation/plots/)",
    )

    parser.add_argument(
        "--formats",
        "-f",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg", "jpg"],
        help="Output formats (default: png)",
    )

    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster formats (default: 300)"
    )

    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["iou", "calibration", "correction", "prf", "stratified", "all"],
        default=["all"],
        help="Which plots to generate (default: all)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.report.exists():
        print(f"ERROR: Report file not found: {args.report}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load report
    print(f"Loading report: {args.report}")
    report = load_report(args.report)

    # Determine which plots to generate
    plot_selection = set(args.plots)
    if "all" in plot_selection:
        plot_selection = {"iou", "calibration", "correction", "prf", "stratified"}

    print(f"\nGenerating plots to: {args.output_dir}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"DPI: {args.dpi}\n")

    # Generate plots
    if "iou" in plot_selection:
        print("Generating IOU uplift plot...")
        plot_iou_uplift(report, args.output_dir, args.formats, args.dpi)

    if "calibration" in plot_selection:
        print("Generating calibration curve...")
        plot_calibration_curve(report, args.output_dir, args.formats, args.dpi)

    if "correction" in plot_selection:
        print("Generating correction rate plot...")
        plot_correction_rate(report, args.output_dir, args.formats, args.dpi)

    if "prf" in plot_selection:
        print("Generating P/R/F1 comparison...")
        plot_prf_comparison(report, args.output_dir, args.formats, args.dpi)

    if "stratified" in plot_selection:
        print("Generating stratified analysis plots...")
        plot_stratified_comparison(report, args.output_dir, args.formats, args.dpi)

    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
