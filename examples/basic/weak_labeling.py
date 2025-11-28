"""
Weak Labeling Deep Dive

Demonstrates: Lexicon-based span detection with fuzzy matching and confidence scores
Prerequisites: None
Runtime: <1 second
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weak_label import LexiconEntry, match_products, match_symptoms, weak_label


def create_comprehensive_symptom_lexicon():
    """Create a rich symptom lexicon with variations."""
    return [
        # Exact match examples
        LexiconEntry(term="headache", canonical="Headache", source="MedDRA"),
        LexiconEntry(term="nausea", canonical="Nausea", source="MedDRA"),
        # Multi-word terms
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
        LexiconEntry(term="chest pain", canonical="Chest Pain", source="MedDRA"),
        # Terms requiring fuzzy matching
        LexiconEntry(term="dizziness", canonical="Dizziness", source="MedDRA"),
        LexiconEntry(term="itching", canonical="Pruritus", source="MedDRA"),
        # Spelling variations
        LexiconEntry(term="redness", canonical="Erythema", source="MedDRA"),
        LexiconEntry(term="swelling", canonical="Edema", source="MedDRA"),
    ]


def demo_exact_matching():
    """Demonstrate exact phrase matching with word boundaries."""
    print("=" * 70)
    print("1. EXACT PHRASE MATCHING")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="headache", canonical="Headache", source="MedDRA"),
        LexiconEntry(term="nausea", canonical="Nausea", source="MedDRA"),
    ]

    test_cases = [
        ("I have a headache today", "✓ Exact match"),
        ("Experiencing severe nausea", "✓ Exact match with adjective"),
        ("No headache reported", "✓ Match (check negation separately)"),
        ("The medication is a headache", "✓ Match (different context)"),
        ("I'm ahead ache free", "✗ No match (word boundary)"),
    ]

    for text, expected in test_cases:
        spans = match_symptoms(text, lexicon)
        result = "✓ FOUND" if spans else "✗ NOT FOUND"
        print(f"   {result:12} | {text}")
        if spans:
            print(f'                  → "{spans[0].text}" [{spans[0].start}:{spans[0].end}]')
        print()


def demo_fuzzy_matching():
    """Demonstrate fuzzy matching with threshold."""
    print("=" * 70)
    print("2. FUZZY MATCHING (WRatio ≥88%)")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
        LexiconEntry(term="dizziness", canonical="Dizziness", source="MedDRA"),
    ]

    test_cases = [
        # Typos and variations
        ("I feel burnnig sensation", "Typo: 'burnnig'"),
        ("Experiencing diziness", "Missing letter: 'diziness'"),
        ("Severe buring sensations", "Plural form"),
        # Adjective handling
        ("Severe burning sensation", "With adjective"),
        ("Mild dizziness reported", "With modifier"),
        # Should NOT match (too different)
        ("I am burning paper", "Different context (low overlap)"),
    ]

    for text, description in test_cases:
        spans = match_symptoms(text, lexicon)

        if spans:
            print(f"   ✓ MATCHED: {description}")
            print(f'      Text: "{text}"')
            print(f'      Found: "{spans[0].text}" → {spans[0].canonical}')
            print(f"      Confidence: {spans[0].confidence:.2f}")
        else:
            print(f"   ✗ NO MATCH: {description}")
            print(f'      Text: "{text}"')
        print()


def demo_confidence_scoring():
    """Explain confidence score calculation."""
    print("=" * 70)
    print("3. CONFIDENCE SCORING")
    print("=" * 70)
    print()

    print("Confidence = 0.8 × (fuzzy_score/100) + 0.2 × (jaccard_score/100)")
    print()
    print("Examples:")
    print()

    lexicon = [
        LexiconEntry(term="redness", canonical="Erythema", source="MedDRA"),
        LexiconEntry(term="itching", canonical="Pruritus", source="MedDRA"),
    ]

    test_cases = [
        "Experiencing redness",  # Exact match → 1.00
        "Severe reddness",  # Typo → ~0.92
        "Mild itching sensation",  # Extra words → ~0.85
    ]

    for text in test_cases:
        spans = match_symptoms(text, lexicon)
        if spans:
            span = spans[0]
            print(f'   Text: "{text}"')
            print(f'   Detected: "{span.text}" → {span.canonical}')
            print(f"   Confidence: {span.confidence:.2f}")

            # Explain score
            if span.confidence == 1.0:
                print("   Explanation: Exact phrase match")
            elif span.confidence >= 0.9:
                print("   Explanation: Minor typo or variation")
            else:
                print("   Explanation: Fuzzy match with extra words")
            print()


def demo_negation_detection():
    """Show how negation affects entity detection."""
    print("=" * 70)
    print("4. NEGATION DETECTION (±5 token window)")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="pain", canonical="Pain", source="MedDRA"),
        LexiconEntry(term="swelling", canonical="Edema", source="MedDRA"),
    ]

    test_cases = [
        ("I have severe pain", False, "Positive mention"),
        ("No pain reported today", True, "Negation before"),
        ("Pain is absent", True, "Negation after"),
        ("Patient denies any swelling", True, "Negation with extra words"),
        ("Without pain or discomfort", True, "Negation with 'without'"),
        ("Pain free for 3 days", True, "Negation compound"),
        ("The pain was intense yesterday but no pain today", "Mixed", "Both positive and negated"),
    ]

    for text, expected_negated, description in test_cases:
        spans = match_symptoms(text, lexicon, negation_window=5)

        print(f'   Text: "{text}"')
        if spans:
            for span in spans:
                negated_marker = "🚫 NEGATED" if span.negated else "✓ POSITIVE"
                print(f'      {negated_marker}: "{span.text}" [{span.start}:{span.end}]')
        else:
            print("      ℹ No spans detected")
        print(f"   Explanation: {description}")
        print()


def demo_multi_word_phrases():
    """Demonstrate multi-word phrase detection."""
    print("=" * 70)
    print("5. MULTI-WORD PHRASE HANDLING")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="chest pain", canonical="Chest Pain", source="MedDRA"),
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
        LexiconEntry(term="shortness of breath", canonical="Dyspnea", source="MedDRA"),
    ]

    print("Challenge: Ensure entire phrase is captured, not individual words")
    print()

    test_cases = [
        "Experiencing chest pain",
        "Severe burning sensation on skin",
        "Patient reports shortness of breath",
        "Chest discomfort and pain",  # Should match "pain", not "chest pain"
    ]

    for text in test_cases:
        spans = match_symptoms(text, lexicon)
        print(f'   Text: "{text}"')
        if spans:
            for span in spans:
                words = len(span.text.split())
                print(f"      → \"{span.text}\" ({words} word{'s' if words > 1 else ''})")
        else:
            print("      ℹ No complete phrases matched")
        print()


def demo_products_vs_symptoms():
    """Show difference between product and symptom matching."""
    print("=" * 70)
    print("6. PRODUCTS vs SYMPTOMS (Negation Difference)")
    print("=" * 70)
    print()

    symptom_lexicon = [
        LexiconEntry(term="rash", canonical="Rash", source="MedDRA"),
    ]

    product_lexicon = [
        LexiconEntry(term="face cream", canonical="Facial Moisturizer", source="Product DB"),
    ]

    text = "No rash after using face cream"

    print(f'   Text: "{text}"')
    print()

    # Symptoms WITH negation detection
    symptom_spans = match_symptoms(text, symptom_lexicon, negation_window=5)
    print("   SYMPTOMS (negation enabled):")
    for span in symptom_spans:
        negated = " [NEGATED]" if span.negated else ""
        print(f'      • "{span.text}"{negated}')
    print()

    # Products WITHOUT negation detection
    product_spans = match_products(text, product_lexicon)
    print("   PRODUCTS (negation disabled):")
    for span in product_spans:
        print(f'      • "{span.text}"')
    print()

    print("   Explanation:")
    print("      • Symptoms track negation for clinical accuracy")
    print("      • Products always detected (negation irrelevant for product mentions)")


def main():
    """Run all weak labeling demonstrations."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WEAK LABELING DEEP DIVE" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_exact_matching()
    demo_fuzzy_matching()
    demo_confidence_scoring()
    demo_negation_detection()
    demo_multi_word_phrases()
    demo_products_vs_symptoms()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ Exact matches → confidence = 1.00")
    print("✓ Fuzzy matches → confidence = weighted fuzzy + jaccard scores")
    print("✓ Negation detection → ±5 token window by default")
    print("✓ Multi-word phrases → last-token alignment required")
    print("✓ Products vs Symptoms → different negation handling")
    print()
    print("💡 Tuning Tips:")
    print("   • Lower fuzzy_threshold (88 → 85) for more lenient matching")
    print("   • Increase negation_window (5 → 7) for longer sentences")
    print("   • Use scorer='jaccard' for strict token overlap")
    print()


if __name__ == "__main__":
    main()
