"""
Annotation Example: Preparing Annotation Batches
=================================================

Demonstrates how to prepare stratified batches of texts for human annotation,
including de-identification, quality filtering, and strategic sampling to
maximize annotation value.

**What You'll Learn:**
- Stratified sampling by confidence/label/length
- De-identification and privacy-safe text preparation
- Quality filtering (minimum spans, text length)
- Batch size optimization for annotation sessions
- Exporting to Label Studio format

**Prerequisites:**
- Completed evaluation/stratified_analysis.py
- Understanding of annotation workflows
- Familiarity with weak labeling quality issues

**Runtime:** ~45 seconds

**Use Cases:**
- Preparing first annotation batch (calibration set)
- Targeting low-confidence spans for review
- Creating balanced SYMPTOM/PRODUCT samples
- Active learning: annotating informative examples
- Quality assurance sampling
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import stratify_by_confidence, stratify_by_label


def create_sample_weak_labels():
    """Create sample weak labels for batch preparation."""

    return [
        # High confidence - good for calibration
        {
            "text": "After using the moisturizer, I developed burning sensation and redness.",
            "spans": [
                {
                    "text": "burning sensation",
                    "start": 40,
                    "end": 57,
                    "label": "SYMPTOM",
                    "confidence": 1.0,
                },
                {"text": "redness", "start": 62, "end": 69, "label": "SYMPTOM", "confidence": 0.98},
                {
                    "text": "moisturizer",
                    "start": 16,
                    "end": 27,
                    "label": "PRODUCT",
                    "confidence": 0.95,
                },
            ],
            "source": "consumer_complaint_001",
        },
        # Medium confidence - boundary issues
        {
            "text": "The facial cream caused severe itching and mild swelling on my face.",
            "spans": [
                {
                    "text": "severe itching",
                    "start": 28,
                    "end": 42,
                    "label": "SYMPTOM",
                    "confidence": 0.75,
                },
                {
                    "text": "mild swelling",
                    "start": 47,
                    "end": 60,
                    "label": "SYMPTOM",
                    "confidence": 0.72,
                },
                {
                    "text": "facial cream",
                    "start": 4,
                    "end": 16,
                    "label": "PRODUCT",
                    "confidence": 0.88,
                },
            ],
            "source": "consumer_complaint_002",
        },
        # Low confidence - ambiguous symptoms
        {
            "text": "I experienced discomfort and a strange feeling after applying the serum.",
            "spans": [
                {
                    "text": "discomfort",
                    "start": 15,
                    "end": 25,
                    "label": "SYMPTOM",
                    "confidence": 0.65,
                },
                {
                    "text": "strange feeling",
                    "start": 32,
                    "end": 47,
                    "label": "SYMPTOM",
                    "confidence": 0.52,
                },
                {"text": "serum", "start": 67, "end": 72, "label": "PRODUCT", "confidence": 0.90},
            ],
            "source": "consumer_complaint_003",
        },
        # High confidence symptoms only
        {
            "text": "Developed a rash and blistering within hours of use.",
            "spans": [
                {"text": "rash", "start": 12, "end": 16, "label": "SYMPTOM", "confidence": 1.0},
                {
                    "text": "blistering",
                    "start": 21,
                    "end": 31,
                    "label": "SYMPTOM",
                    "confidence": 0.96,
                },
            ],
            "source": "consumer_complaint_004",
        },
        # Product-heavy example
        {
            "text": "The anti-aging cream and night serum both caused irritation.",
            "spans": [
                {
                    "text": "anti-aging cream",
                    "start": 4,
                    "end": 20,
                    "label": "PRODUCT",
                    "confidence": 0.92,
                },
                {
                    "text": "night serum",
                    "start": 25,
                    "end": 36,
                    "label": "PRODUCT",
                    "confidence": 0.89,
                },
                {
                    "text": "irritation",
                    "start": 50,
                    "end": 60,
                    "label": "SYMPTOM",
                    "confidence": 0.95,
                },
            ],
            "source": "consumer_complaint_005",
        },
    ]


def demo_stratified_sampling():
    """Demonstrate stratified sampling for balanced annotation."""
    print("\n" + "=" * 70)
    print("🎯 STRATIFIED SAMPLING")
    print("=" * 70)

    weak_labels = create_sample_weak_labels()

    # Extract all spans with metadata
    all_spans = []
    for doc in weak_labels:
        for span in doc["spans"]:
            all_spans.append({**span, "text_full": doc["text"], "source": doc["source"]})

    # Stratify by confidence
    by_confidence = stratify_by_confidence(all_spans, n_bins=3)

    print("\n🔍 Confidence Distribution:")
    for bucket, spans in by_confidence.items():
        print(f"   {bucket.replace('_', ' ').title()}: {len(spans)} spans")

    # Stratify by label
    by_label = stratify_by_label(all_spans)

    print("\n🏷️  Label Distribution:")
    for label, spans in by_label.items():
        print(f"   {label}: {len(spans)} spans")

    print("\n📊 Sampling Strategy:")
    print("   Target: 50 spans for first annotation batch")
    print("   Allocation:")
    print("      • 10 high-confidence (calibration/verification)")
    print("      • 25 medium-confidence (boundary refinement)")
    print("      • 15 low-confidence (ambiguity resolution)")
    print("      • Balance: 60% SYMPTOM, 40% PRODUCT")

    print("\n💡 Rationale:")
    print("   • High-confidence: Verify weak labeling assumptions")
    print("   • Medium-confidence: Improve boundary heuristics")
    print("   • Low-confidence: Clarify ambiguous cases")
    print("   • Label balance: Representative of production distribution")


def demo_quality_filtering():
    """Demonstrate filtering texts for annotation quality."""
    print("\n" + "=" * 70)
    print("🔍 QUALITY FILTERING")
    print("=" * 70)

    weak_labels = create_sample_weak_labels()

    # Define quality criteria
    min_text_length = 20  # characters
    max_text_length = 500
    min_spans = 1
    min_confidence = 0.50  # exclude very low confidence

    print("\n⚙️  Quality Criteria:")
    print(f"   • Text length: {min_text_length}-{max_text_length} chars")
    print(f"   • Minimum spans: {min_spans}")
    print(f"   • Minimum confidence: {min_confidence:.2f}")

    # Apply filters
    filtered = []
    rejected = []

    for doc in weak_labels:
        text_len = len(doc["text"])
        num_spans = len(doc["spans"])
        avg_conf = sum(s["confidence"] for s in doc["spans"]) / num_spans if num_spans > 0 else 0

        # Check criteria
        if text_len < min_text_length or text_len > max_text_length:
            rejected.append((doc, f"Text length: {text_len}"))
        elif num_spans < min_spans:
            rejected.append((doc, f"Too few spans: {num_spans}"))
        elif avg_conf < min_confidence:
            rejected.append((doc, f"Low confidence: {avg_conf:.2f}"))
        else:
            filtered.append(doc)

    print(f"\n✓ Accepted: {len(filtered)} documents")
    print(f"✗ Rejected: {len(rejected)} documents")

    if rejected:
        print("\n⚠️  Rejection Reasons:")
        for doc, reason in rejected:
            print(f"   {doc['source']}: {reason}")

    print("\n💡 Best Practices:")
    print("   • Exclude very short texts (lack context)")
    print("   • Exclude very long texts (annotation fatigue)")
    print("   • Filter extreme low confidence (likely noise)")
    print("   • Require minimum entity density (informative examples)")


def demo_deidentification():
    """Demonstrate privacy-safe text preparation."""
    print("\n" + "=" * 70)
    print("🔒 DE-IDENTIFICATION")
    print("=" * 70)

    # Simulated text with PII (for demonstration only)
    text_with_pii = (
        "John Smith reported burning after using Product X. Contact: john@email.com, 555-1234."
    )

    print("\n⚠️  Original Text (DO NOT COMMIT):")
    print(f"   {text_with_pii}")

    # Simple de-identification (real system should use proper NER for PII)
    deidentified = text_with_pii
    deidentified = deidentified.replace("John Smith", "[NAME]")
    deidentified = deidentified.replace("john@email.com", "[EMAIL]")
    deidentified = deidentified.replace("555-1234", "[PHONE]")

    print("\n✓ De-identified Text (SAFE TO COMMIT):")
    print(f"   {deidentified}")

    print("\n🔒 De-identification Checklist:")
    print("   ☐ Remove names (person, organization)")
    print("   ☐ Remove contact info (email, phone, address)")
    print("   ☐ Remove identifiers (SSN, account numbers)")
    print("   ☐ Remove dates (birthdays, specific events)")
    print("   ☐ Generalize locations (city → region)")
    print("   ☐ Review brand names (may be sensitive)")

    print("\n💡 Tools for De-identification:")
    print("   • spaCy NER (en_core_web_sm) for person/org/location")
    print("   • Presidio (Microsoft) for PII detection")
    print("   • Regex patterns for emails, phones, SSNs")
    print("   • Manual review for domain-specific identifiers")

    print("\n⚠️  Privacy Warning:")
    print("   NEVER commit raw consumer complaints to version control!")
    print("   Store de-identified texts only. Keep raw data in gitignored directories.")


def demo_batch_export():
    """Demonstrate exporting batches for annotation."""
    print("\n" + "=" * 70)
    print("📤 BATCH EXPORT")
    print("=" * 70)

    weak_labels = create_sample_weak_labels()

    # Take first 3 as example batch
    batch = weak_labels[:3]

    print(f"\n📦 Batch Info:")
    print(f"   Batch ID: annotation_batch_001")
    print(f"   Date: 2025-11-27")
    print(f"   Documents: {len(batch)}")
    print(f"   Total Spans: {sum(len(doc['spans']) for doc in batch)}")

    # Export to JSONL format
    output_dir = Path(__file__).parent.parent.parent / "data" / "annotation" / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example_batch_001.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in batch:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n✓ Exported to: {output_path}")

    print("\n📋 Batch Metadata:")
    print("   • Format: JSONL (one document per line)")
    print("   • Encoding: UTF-8")
    print("   • Pre-annotations: Weak labels included")
    print("   • Ready for: Label Studio import")

    print("\n💡 Next Steps:")
    print("   1. Review batch for quality")
    print("   2. Import to Label Studio")
    print("   3. Assign to annotators")
    print("   4. Monitor annotation progress")


def demo_active_learning_sampling():
    """Demonstrate active learning-based sampling."""
    print("\n" + "=" * 70)
    print("🎓 ACTIVE LEARNING SAMPLING")
    print("=" * 70)

    weak_labels = create_sample_weak_labels()

    print("\n🔍 Active Learning Strategies:\n")

    # 1. Uncertainty sampling
    print("   1. UNCERTAINTY SAMPLING")
    print("      Target: Spans with confidence 0.50-0.75")
    print("      Rationale: Model is least certain → most informative")

    # Extract uncertain spans
    uncertain_docs = []
    for doc in weak_labels:
        avg_conf = sum(s["confidence"] for s in doc["spans"]) / len(doc["spans"])
        if 0.50 <= avg_conf <= 0.75:
            uncertain_docs.append(doc)

    print(f"      Found: {len(uncertain_docs)} documents")

    # 2. Diversity sampling
    print("\n   2. DIVERSITY SAMPLING")
    print("      Target: One example per confidence × label combination")
    print("      Rationale: Cover entire feature space")

    # Create buckets
    buckets = {}
    for doc in weak_labels:
        for span in doc["spans"]:
            conf_bucket = (
                "high"
                if span["confidence"] > 0.85
                else "medium" if span["confidence"] > 0.65 else "low"
            )
            key = f"{conf_bucket}_{span['label']}"
            if key not in buckets:
                buckets[key] = doc

    print(f"      Found: {len(buckets)} diverse examples")

    # 3. Error-based sampling
    print("\n   3. ERROR-BASED SAMPLING")
    print("      Target: Spans with boundary issues (adjectives, determiners)")
    print("      Rationale: Focus on known weak labeling failure modes")

    error_prone_docs = []
    for doc in weak_labels:
        for span in doc["spans"]:
            # Check for adjectives (simple heuristic)
            if any(adj in span["text"].lower() for adj in ["severe", "mild", "slight", "minor"]):
                error_prone_docs.append(doc)
                break

    print(f"      Found: {len(error_prone_docs)} error-prone documents")

    print("\n📊 Recommended Batch Composition:")
    print("   • 40% uncertainty sampling (informative)")
    print("   • 30% diversity sampling (coverage)")
    print("   • 20% error-based sampling (known issues)")
    print("   • 10% random sampling (baseline)")

    print("\n💡 Iteration Strategy:")
    print("   1. Annotate 50-100 examples")
    print("   2. Fine-tune model on gold labels")
    print("   3. Re-run weak labeling with improved model")
    print("   4. Sample next batch from new uncertainties")
    print("   5. Repeat until desired F1 achieved")


def main():
    """Run all batch preparation examples."""
    print("\n" + "=" * 70)
    print("ANNOTATION BATCH PREPARATION EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates preparing stratified, quality-filtered, and")
    print("privacy-safe annotation batches using active learning strategies.")

    # Run demos
    demo_stratified_sampling()
    demo_quality_filtering()
    demo_deidentification()
    demo_batch_export()
    demo_active_learning_sampling()

    print("\n" + "=" * 70)
    print("✓ All batch preparation examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Try label_studio_setup.py to configure annotation tool")
    print("   • See export_gold_standard.py for converting annotations")
    print("   • Use quality_report.py for inter-annotator agreement")
    print()


if __name__ == "__main__":
    main()
