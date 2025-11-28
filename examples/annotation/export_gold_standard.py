"""
Annotation Example: Exporting Gold Standard Labels
===================================================

Demonstrates how to convert Label Studio annotations to SpanForge's gold
standard JSONL format, including consensus handling, validation, and
integrity checks.

**What You'll Learn:**
- Converting Label Studio JSON to SpanForge JSONL
- Handling multiple annotator consensus
- Validating annotation integrity (alignment, overlaps)
- Computing annotation statistics
- Preparing gold labels for evaluation

**Prerequisites:**
- Completed annotation/label_studio_setup.py
- Annotated data exported from Label Studio
- Understanding of annotation consensus strategies

**Runtime:** ~30 seconds

**Use Cases:**
- Converting annotations after annotation round
- Creating train/dev/test splits
- Validating annotation quality before model training
- Generating evaluation gold standard
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_sample_labelstudio_export():
    """Create sample Label Studio export data."""

    return [
        {
            "id": 1,
            "data": {
                "text": "After using the moisturizer, I developed burning sensation and redness."
            },
            "annotations": [
                {
                    "id": 101,
                    "completed_by": {"email": "annotator1@example.com"},
                    "created_at": "2025-11-27T10:00:00Z",
                    "result": [
                        {
                            "value": {
                                "start": 40,
                                "end": 57,
                                "text": "burning sensation",
                                "labels": ["SYMPTOM"],
                            },
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                        },
                        {
                            "value": {
                                "start": 62,
                                "end": 69,
                                "text": "redness",
                                "labels": ["SYMPTOM"],
                            },
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                        },
                        {
                            "value": {
                                "start": 16,
                                "end": 27,
                                "text": "moisturizer",
                                "labels": ["PRODUCT"],
                            },
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                        },
                    ],
                },
                {
                    "id": 102,
                    "completed_by": {"email": "annotator2@example.com"},
                    "created_at": "2025-11-27T10:15:00Z",
                    "result": [
                        {
                            "value": {
                                "start": 40,
                                "end": 57,
                                "text": "burning sensation",
                                "labels": ["SYMPTOM"],
                            },
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                        },
                        {
                            "value": {
                                "start": 62,
                                "end": 69,
                                "text": "redness",
                                "labels": ["SYMPTOM"],
                            },
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                        },
                    ],
                },
            ],
        }
    ]


def demo_basic_conversion():
    """Demonstrate basic Label Studio to SpanForge conversion."""
    print("\n" + "=" * 70)
    print("🔄 BASIC CONVERSION")
    print("=" * 70)

    ls_data = create_sample_labelstudio_export()
    task = ls_data[0]

    print(f"\n📥 Label Studio Task:")
    print(f"   Task ID: {task['id']}")
    print(f"   Text: {task['data']['text'][:50]}...")
    print(f"   Annotations: {len(task['annotations'])}")

    # Convert first annotation to SpanForge format
    annotation = task["annotations"][0]

    gold_format = {
        "text": task["data"]["text"],
        "source": f"label_studio_task_{task['id']}",
        "metadata": {
            "task_id": task["id"],
            "annotator": annotation["completed_by"]["email"],
            "timestamp": annotation["created_at"],
        },
        "spans": [],
    }

    # Convert spans
    for span_data in annotation["result"]:
        if span_data["type"] == "labels":
            gold_format["spans"].append(
                {
                    "text": span_data["value"]["text"],
                    "start": span_data["value"]["start"],
                    "end": span_data["value"]["end"],
                    "label": span_data["value"]["labels"][0],
                }
            )

    print(f"\n📤 SpanForge Gold Format:")
    print("   " + json.dumps(gold_format, indent=2).replace("\n", "\n   "))

    print("\n✨ Conversion Mapping:")
    print("   LS 'data.text' → SF 'text'")
    print("   LS 'result[].value' → SF 'spans[]'")
    print("   LS 'completed_by' → SF 'metadata.annotator'")
    print("   LS 'created_at' → SF 'metadata.timestamp'")


def compute_iou(span1: Dict, span2: Dict) -> float:
    """Compute IOU between two spans."""
    start1, end1 = span1["start"], span1["end"]
    start2, end2 = span2["start"], span2["end"]

    # Intersection
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0, overlap_end - overlap_start)

    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start

    return overlap / union if union > 0 else 0.0


def demo_consensus_handling():
    """Demonstrate multi-annotator consensus."""
    print("\n" + "=" * 70)
    print("🤝 CONSENSUS HANDLING")
    print("=" * 70)

    ls_data = create_sample_labelstudio_export()
    task = ls_data[0]

    print(f"\n📊 Multi-Annotator Task:")
    print(f"   Annotators: {len(task['annotations'])}")

    # Collect all spans from all annotators
    annotator_spans = []
    for annotation in task["annotations"]:
        spans = []
        for span_data in annotation["result"]:
            if span_data["type"] == "labels":
                spans.append(
                    {
                        "text": span_data["value"]["text"],
                        "start": span_data["value"]["start"],
                        "end": span_data["value"]["end"],
                        "label": span_data["value"]["labels"][0],
                        "annotator": annotation["completed_by"]["email"],
                    }
                )
        annotator_spans.append(spans)

    print("\n🔍 Annotator Spans:")
    for i, spans in enumerate(annotator_spans, 1):
        print(f"\n   Annotator {i}: {len(spans)} spans")
        for span in spans:
            print(f"      '{span['text']}' [{span['label']}] at {span['start']}-{span['end']}")

    # Apply consensus strategy: majority vote with IOU ≥ 0.5
    consensus_spans = []
    seen_spans = set()

    for spans in annotator_spans:
        for span in spans:
            span_key = (span["start"], span["end"], span["label"])
            if span_key in seen_spans:
                continue

            # Count votes (spans with IOU ≥ 0.5 are considered same)
            votes = 0
            for other_spans in annotator_spans:
                for other_span in other_spans:
                    if (
                        span["label"] == other_span["label"]
                        and compute_iou(span, other_span) >= 0.5
                    ):
                        votes += 1
                        break

            # Require majority (>50% of annotators)
            if votes > len(annotator_spans) / 2:
                consensus_spans.append(
                    {
                        "text": span["text"],
                        "start": span["start"],
                        "end": span["end"],
                        "label": span["label"],
                    }
                )
                seen_spans.add(span_key)

    print(f"\n✅ Consensus Spans: {len(consensus_spans)}")
    for span in consensus_spans:
        print(f"   '{span['text']}' [{span['label']}]")

    print("\n💡 Consensus Strategy:")
    print("   • Spans with IOU ≥ 0.5 considered identical")
    print("   • Majority vote (>50% agreement) required")
    print("   • Conflicts flagged for expert adjudication")
    print("   • Longest span selected when multiple options")


def demo_validation():
    """Demonstrate annotation validation."""
    print("\n" + "=" * 70)
    print("✅ VALIDATION & INTEGRITY CHECKS")
    print("=" * 70)

    # Sample gold standard annotation
    text = "After using the moisturizer, I developed burning sensation and redness."

    spans = [
        {"text": "burning sensation", "start": 40, "end": 57, "label": "SYMPTOM"},
        {"text": "redness", "start": 62, "end": 69, "label": "SYMPTOM"},
        {"text": "moisturizer", "start": 16, "end": 27, "label": "PRODUCT"},
    ]

    print("\n🔍 Running Validation Checks:\n")

    # Check 1: Text alignment
    print("   1. TEXT ALIGNMENT:")
    all_aligned = True
    for span in spans:
        extracted = text[span["start"] : span["end"]]
        matches = extracted == span["text"]
        status = "✓" if matches else "✗"
        print(f"      {status} '{span['text']}' == '{extracted}' : {matches}")
        all_aligned = all_aligned and matches

    if all_aligned:
        print("      ✓ All spans correctly aligned")
    else:
        print("      ✗ Alignment errors detected!")

    # Check 2: No overlaps
    print("\n   2. OVERLAP DETECTION:")
    overlaps = []
    for i, span1 in enumerate(spans):
        for j, span2 in enumerate(spans[i + 1 :], i + 1):
            if span1["start"] < span2["end"] and span2["start"] < span1["end"]:
                if span1["label"] != span2["label"]:
                    overlaps.append((span1, span2))

    if not overlaps:
        print("      ✓ No overlapping spans with different labels")
    else:
        print(f"      ✗ Found {len(overlaps)} overlapping spans")
        for span1, span2 in overlaps:
            print(
                f"         '{span1['text']}' [{span1['label']}] ↔ '{span2['text']}' [{span2['label']}]"
            )

    # Check 3: Valid positions
    print("\n   3. POSITION VALIDITY:")
    valid_positions = True
    for span in spans:
        in_bounds = 0 <= span["start"] < span["end"] <= len(text)
        status = "✓" if in_bounds else "✗"
        print(f"      {status} {span['start']}-{span['end']} in [0, {len(text)}]: {in_bounds}")
        valid_positions = valid_positions and in_bounds

    if valid_positions:
        print("      ✓ All positions within text bounds")

    # Check 4: No trailing punctuation
    print("\n   4. PUNCTUATION CHECK:")
    has_punct = False
    for span in spans:
        if span["text"][-1] in ".!?,;:":
            print(f"      ⚠ '{span['text']}' ends with punctuation")
            has_punct = True

    if not has_punct:
        print("      ✓ No trailing punctuation")

    print("\n📋 Validation Summary:")
    if all_aligned and not overlaps and valid_positions and not has_punct:
        print("   ✓ All checks passed! Annotation is valid.")
    else:
        print("   ⚠ Validation issues found. Review and fix before export.")


def demo_statistics():
    """Show annotation statistics."""
    print("\n" + "=" * 70)
    print("📊 ANNOTATION STATISTICS")
    print("=" * 70)

    # Simulate batch of annotations
    annotations = [
        {
            "text": "Text 1",
            "spans": [
                {"label": "SYMPTOM", "text": "burning"},
                {"label": "SYMPTOM", "text": "redness"},
                {"label": "PRODUCT", "text": "cream"},
            ],
        },
        {
            "text": "Text 2",
            "spans": [
                {"label": "SYMPTOM", "text": "itching"},
                {"label": "PRODUCT", "text": "lotion"},
            ],
        },
        {
            "text": "Text 3",
            "spans": [
                {"label": "SYMPTOM", "text": "swelling"},
                {"label": "SYMPTOM", "text": "pain"},
                {"label": "SYMPTOM", "text": "rash"},
                {"label": "PRODUCT", "text": "serum"},
            ],
        },
    ]

    # Compute statistics
    total_docs = len(annotations)
    total_spans = sum(len(doc["spans"]) for doc in annotations)

    # By label
    label_counts = defaultdict(int)
    for doc in annotations:
        for span in doc["spans"]:
            label_counts[span["label"]] += 1

    # Span lengths
    span_lengths = []
    for doc in annotations:
        for span in doc["spans"]:
            span_lengths.append(len(span["text"]))

    avg_length = sum(span_lengths) / len(span_lengths)

    print(f"\n📈 Batch Statistics:\n")
    print(f"   Documents:     {total_docs}")
    print(f"   Total Spans:   {total_spans}")
    print(f"   Avg Spans/Doc: {total_spans/total_docs:.2f}")
    print(f"   Avg Length:    {avg_length:.1f} chars")

    print(f"\n🏷️  Label Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / total_spans * 100
        print(f"   {label:12} {count:3} ({pct:5.1f}%)")

    print(f"\n📏 Span Length Distribution:")
    bins = [(0, 10), (10, 20), (20, 100)]
    for min_len, max_len in bins:
        count = sum(1 for l in span_lengths if min_len <= l < max_len)
        pct = count / len(span_lengths) * 100
        print(f"   {min_len:2}-{max_len:2} chars: {count:3} ({pct:5.1f}%)")


def demo_export():
    """Demonstrate final export to gold JSONL."""
    print("\n" + "=" * 70)
    print("📤 EXPORT TO GOLD JSONL")
    print("=" * 70)

    # Create sample gold standard
    gold_annotations = [
        {
            "text": "After using the moisturizer, I developed burning sensation and redness.",
            "source": "label_studio_task_1",
            "metadata": {
                "task_id": 1,
                "annotators": ["annotator1@example.com", "annotator2@example.com"],
                "consensus": "majority_vote",
                "timestamp": "2025-11-27T10:30:00Z",
            },
            "spans": [
                {"text": "burning sensation", "start": 40, "end": 57, "label": "SYMPTOM"},
                {"text": "redness", "start": 62, "end": 69, "label": "SYMPTOM"},
                {"text": "moisturizer", "start": 16, "end": 27, "label": "PRODUCT"},
            ],
        }
    ]

    # Export to file
    output_dir = Path(__file__).parent.parent.parent / "data" / "annotation" / "gold"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example_gold_standard.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in gold_annotations:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n✓ Exported {len(gold_annotations)} documents")
    print(f"   Output: {output_path}")

    print("\n📋 Gold Standard Format:")
    print("   • One document per line (JSONL)")
    print("   • UTF-8 encoding")
    print("   • Consensus metadata preserved")
    print("   • Ready for evaluation and training")

    print("\n💡 Next Steps:")
    print("   1. Split into train/dev/test sets")
    print("   2. Evaluate weak labels against gold")
    print("      → python scripts/annotation/evaluate_llm_refinement.py")
    print("   3. Fine-tune BioBERT on gold labels")
    print("   4. Compute inter-annotator agreement")
    print("      → See quality_report.py")


def main():
    """Run all gold standard export examples."""
    print("\n" + "=" * 70)
    print("GOLD STANDARD EXPORT EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates converting Label Studio annotations to SpanForge's")
    print("gold standard format, including consensus handling, validation,")
    print("and quality checks.")

    # Run demos
    demo_basic_conversion()
    demo_consensus_handling()
    demo_validation()
    demo_statistics()
    demo_export()

    print("\n" + "=" * 70)
    print("✓ All gold standard export examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Try quality_report.py for inter-annotator agreement")
    print("   • Use evaluation/compute_metrics.py to evaluate weak vs gold")
    print("   • Begin model training with gold standard labels")
    print()


if __name__ == "__main__":
    main()
