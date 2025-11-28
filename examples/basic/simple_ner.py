"""
Simple Named Entity Recognition Example

Demonstrates: Basic entity extraction from text using pre-built lexicons
Prerequisites: None (uses default lexicons)
Runtime: <1 second
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import AppConfig, get_config
from src.weak_label import LexiconEntry, weak_label


def load_sample_lexicons():
    """Load symptom and product lexicons from CSV files.

    Returns:
        tuple: (symptom_lexicon, product_lexicon) as lists of LexiconEntry objects
    """
    # In production, load from CSV files using pandas
    # For this example, we'll create sample entries programmatically

    symptom_lexicon = [
        LexiconEntry(
            term="burning sensation",
            canonical="Burning Sensation",
            source="MedDRA",
            concept_id="10006784",
        ),
        LexiconEntry(
            term="redness",
            canonical="Erythema",
            source="MedDRA",
            concept_id="10015150",
        ),
        LexiconEntry(
            term="itching",
            canonical="Pruritus",
            source="MedDRA",
            concept_id="10037087",
        ),
        LexiconEntry(
            term="swelling",
            canonical="Edema",
            source="MedDRA",
            concept_id="10014210",
        ),
        LexiconEntry(
            term="rash",
            canonical="Rash",
            source="MedDRA",
            concept_id="10037844",
        ),
    ]

    product_lexicon = [
        LexiconEntry(
            term="face cream",
            canonical="Facial Moisturizer",
            source="Product DB",
            category="Cosmetic",
        ),
        LexiconEntry(
            term="shampoo",
            canonical="Hair Cleanser",
            source="Product DB",
            category="Personal Care",
        ),
        LexiconEntry(
            term="sunscreen",
            canonical="Sun Protection Cream",
            source="Product DB",
            category="Cosmetic",
        ),
    ]

    return symptom_lexicon, product_lexicon


def main():
    """Run simple NER example with sample text."""
    print("=" * 70)
    print("SpanForge - Simple Named Entity Recognition Example")
    print("=" * 70)
    print()

    # Sample consumer complaint text
    sample_texts = [
        "After using this face cream, I developed burning sensation and redness.",
        "The shampoo caused severe itching and swelling on my scalp.",
        "Applied sunscreen yesterday, now I have a rash on my arms.",
        "No side effects reported - product works great!",
    ]

    # Load lexicons
    print("📚 Loading lexicons...")
    symptom_lexicon, product_lexicon = load_sample_lexicons()
    print(f"   • Loaded {len(symptom_lexicon)} symptom terms")
    print(f"   • Loaded {len(product_lexicon)} product terms")
    print()

    # Process each text
    for i, text in enumerate(sample_texts, 1):
        print(f"📝 Example {i}:")
        print(f'   Text: "{text}"')
        print()

        # Extract entities
        spans = weak_label(
            text=text,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
        )

        # Display results
        if spans:
            print(f"   ✓ Detected {len(spans)} entities:")
            for span in spans:
                # Build annotation string
                annotation = f'   • "{span.text}" [{span.label}]'
                annotation += f" at position {span.start}-{span.end}"
                annotation += f" (confidence: {span.confidence:.2f})"

                # Add canonical form if different
                if span.canonical and span.canonical.lower() != span.text.lower():
                    annotation += f" → {span.canonical}"

                # Add negation flag
                if span.negated:
                    annotation += " [NEGATED]"

                print(annotation)
        else:
            print("   ℹ No entities detected")

        print()

    # Configuration info
    print("⚙️  Configuration:")
    config = get_config()
    print(f"   • Fuzzy threshold: {config.fuzzy_scorer} @ ≥88%")
    print(f"   • Negation window: ±{config.negation_window} tokens")
    print(f"   • Device: {config.device}")
    print()

    print("💡 Next Steps:")
    print("   1. Try modifying the sample texts above")
    print("   2. Add custom symptom/product terms to the lexicons")
    print("   3. Adjust fuzzy_threshold in weak_label() call")
    print("   4. Explore weak_labeling.py for detailed explanations")
    print()


if __name__ == "__main__":
    main()
