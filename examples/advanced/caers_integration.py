"""
FDA CAERS Data Integration

Demonstrates: Download, filter, and weak label FDA adverse event reports
Prerequisites: pandas (included), ~125MB download
Runtime: 30 seconds for 1K records, 5 minutes for full dataset
"""

import sys
from pathlib import Path

# Add scripts/caers to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "caers"))


def demo_quick_download():
    """Download small CAERS subset for testing."""
    print("=" * 70)
    print("1. QUICK DOWNLOAD (100 cosmetics complaints)")
    print("=" * 70)
    print()

    # Import from scripts/caers
    try:
        import sys

        from download_caers import main as download_main

        # Override sys.argv to pass arguments
        sys.argv = [
            "download_caers.py",
            "--output",
            "data/caers/example_100.jsonl",
            "--categories",
            "cosmetics",
            "--limit",
            "100",
        ]

        download_main()

    except ImportError:
        print("⚠ Error: scripts/caers/download_caers.py not found")
        print("   Run from project root: python scripts/caers/download_caers.py --help")
    print()


def demo_analysis():
    """Analyze downloaded CAERS data."""
    print("=" * 70)
    print("2. ANALYZE CAERS DATA")
    print("=" * 70)
    print()

    import json
    from collections import Counter
    from pathlib import Path

    caers_file = Path("data/caers/example_100.jsonl")

    if not caers_file.exists():
        print(f"⚠ File not found: {caers_file}")
        print("   Run demo_quick_download() first")
        return

    print(f"Analyzing: {caers_file}")
    print()

    records = []
    with caers_file.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    print()

    # Span statistics
    all_spans = [span for record in records for span in record.get("spans", [])]
    label_counts = Counter(span["label"] for span in all_spans)

    print("Entity Distribution:")
    for label, count in label_counts.most_common():
        pct = count / len(all_spans) * 100 if all_spans else 0
        print(f"   • {label:12} {count:4} ({pct:.1f}%)")
    print()

    # Canonical terms
    canonical_counts = Counter(
        span.get("canonical", "Unknown") for span in all_spans if span["label"] == "SYMPTOM"
    )

    print("Top 10 Symptoms (by canonical):")
    for canonical, count in canonical_counts.most_common(10):
        print(f"   • {canonical:30} {count:3}")
    print()


def main():
    """Run CAERS integration examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "FDA CAERS INTEGRATION" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("ℹ This example uses scripts/caers/download_caers.py")
    print()
    print("Available categories:")
    print("   • cosmetics (facial care, hair care, makeup)")
    print("   • personal_care (soaps, deodorants)")
    print("   • supplements (vitamins, minerals)")
    print("   • foods (beverages, snacks)")
    print("   • baby (infant formula, baby food)")
    print()

    demo_quick_download()
    demo_analysis()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Download full dataset:")
    print("   python scripts/caers/download_caers.py \\")
    print("       --output data/caers/all_cosmetics.jsonl \\")
    print("       --categories cosmetics")
    print()
    print("2. Filter by minimum spans:")
    print("   python scripts/caers/download_caers.py \\")
    print("       --categories cosmetics \\")
    print("       --min-spans 3 \\")
    print("       --limit 5000")
    print()
    print("3. See full documentation:")
    print("   cat scripts/caers/README.md")
    print()


if __name__ == "__main__":
    main()
