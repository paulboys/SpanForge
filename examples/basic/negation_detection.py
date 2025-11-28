"""
Negation Detection Examples

Demonstrates: Bidirectional negation window and edge cases
Prerequisites: None
Runtime: <1 second
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weak_label import LexiconEntry, detect_negated_regions, match_symptoms


def demo_basic_negation():
    """Show common negation patterns."""
    print("=" * 70)
    print("1. BASIC NEGATION PATTERNS")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="pain", canonical="Pain", source="MedDRA"),
        LexiconEntry(term="fever", canonical="Pyrexia", source="MedDRA"),
        LexiconEntry(term="rash", canonical="Rash", source="MedDRA"),
    ]

    test_cases = [
        # Negation BEFORE symptom
        "No pain reported",
        "Patient denies fever",
        "Without any rash",
        "Absence of pain",
        # Negation AFTER symptom
        "Pain is absent",
        "Fever not present",
        "Rash denied",
        # Positive controls
        "Severe pain present",
        "Patient has fever",
    ]

    print("Negation window: ±5 tokens (bidirectional)")
    print()

    for text in test_cases:
        spans = match_symptoms(text, lexicon, negation_window=5)

        if spans:
            span = spans[0]
            status = "🚫 NEGATED" if span.negated else "✓ POSITIVE"
            print(f"   {status:12} | {text}")
            print(f'                  → "{span.text}" [{span.start}:{span.end}]')
        else:
            print(f"   ℹ NO MATCH    | {text}")
        print()


def demo_negation_cues():
    """List all negation trigger words."""
    print("=" * 70)
    print("2. NEGATION CUE WORDS")
    print("=" * 70)
    print()

    print("The following words trigger negation detection:")
    print()

    negation_terms = [
        ("Primary", ["no", "not", "none", "neither", "never", "nobody", "nowhere"]),
        ("Denial", ["denies", "denied", "deny", "denying"]),
        ("Absence", ["absent", "absence", "without", "lack", "lacking"]),
        ("Compound", ["pain free", "symptom free", "side effect free"]),
    ]

    for category, terms in negation_terms:
        print(f"   {category:12} → {', '.join(terms)}")
    print()

    print("Note: Detection is case-insensitive")
    print()


def demo_window_size():
    """Show how window size affects detection."""
    print("=" * 70)
    print("3. NEGATION WINDOW SIZE IMPACT")
    print("=" * 70)
    print()

    lexicon = [LexiconEntry(term="headache", canonical="Headache", source="MedDRA")]

    text = "Patient reports no severe persistent throbbing headache today"
    #                      ^                                ^
    #                     neg                            symptom
    # Token distance: "no" is 4 tokens before "headache"

    print(f'Text: "{text}"')
    print()
    print("Token distance between 'no' and 'headache': 4 tokens")
    print()

    window_sizes = [2, 3, 4, 5, 7]

    for window_size in window_sizes:
        spans = match_symptoms(text, lexicon, negation_window=window_size)
        if spans:
            span = spans[0]
            status = "NEGATED" if span.negated else "POSITIVE"
            print(
                f"   Window ±{window_size}: {status:8} ({'✓ caught' if span.negated else '✗ missed'})"
            )
        else:
            print(f"   Window ±{window_size}: NO MATCH")

    print()
    print("Recommendation: Use window ≥5 for typical consumer complaints")
    print()


def demo_bidirectional():
    """Demonstrate bidirectional window (before AND after)."""
    print("=" * 70)
    print("4. BIDIRECTIONAL NEGATION")
    print("=" * 70)
    print()

    lexicon = [LexiconEntry(term="swelling", canonical="Edema", source="MedDRA")]

    test_cases = [
        ("No swelling observed", "Negation BEFORE"),
        ("Swelling is absent", "Negation AFTER"),
        ("Patient denies swelling or redness", "Negation BEFORE with extra words"),
        ("Swelling not present today", "Negation AFTER with extra words"),
    ]

    print("Window searches BOTH before and after the symptom:")
    print()

    for text, description in test_cases:
        spans = match_symptoms(text, lexicon, negation_window=5)

        if spans:
            span = spans[0]
            print(f"   {description}")
            print(f'      Text: "{text}"')
            print(f"      Result: {'NEGATED ✓' if span.negated else 'POSITIVE ✗'}")
        print()


def demo_edge_cases():
    """Show tricky negation scenarios."""
    print("=" * 70)
    print("5. EDGE CASES & GOTCHAS")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="pain", canonical="Pain", source="MedDRA"),
        LexiconEntry(term="nausea", canonical="Nausea", source="MedDRA"),
    ]

    print("CASE 1: Multiple symptoms, mixed negation")
    text1 = "Patient has pain but no nausea"
    spans1 = match_symptoms(text1, lexicon, negation_window=5)
    print(f'   Text: "{text1}"')
    for span in spans1:
        status = "NEGATED" if span.negated else "POSITIVE"
        print(f'      • "{span.text}" → {status}')
    print()

    print("CASE 2: Double negatives (not without)")
    text2 = "Not without some pain"
    spans2 = match_symptoms(text2, lexicon, negation_window=5)
    print(f'   Text: "{text2}"')
    if spans2:
        # Will detect negation (doesn't handle double negatives)
        print(f"      • \"{spans2[0].text}\" → {'NEGATED' if spans2[0].negated else 'POSITIVE'}")
        print("      ⚠ Double negatives not resolved (advanced NLP needed)")
    print()

    print("CASE 3: Negation in different context")
    text3 = "No one complains about pain medication"
    spans3 = match_symptoms(text3, lexicon, negation_window=5)
    print(f'   Text: "{text3}"')
    if spans3:
        print(f"      • \"{spans3[0].text}\" → {'NEGATED' if spans3[0].negated else 'POSITIVE'}")
        print("      ℹ 'No one' applies to 'complains', not 'pain' (requires syntax)")
    print()

    print("CASE 4: Past vs present")
    text4 = "Previous pain is now absent"
    spans4 = match_symptoms(text4, lexicon, negation_window=5)
    print(f'   Text: "{text4}"')
    if spans4:
        for span in spans4:
            print(f"      • \"{span.text}\" → {'NEGATED' if span.negated else 'POSITIVE'}")
        print("      ℹ Temporal context not analyzed (current implementation)")
    print()


def demo_overlap_threshold():
    """Explain 50% overlap requirement."""
    print("=" * 70)
    print("6. NEGATION OVERLAP THRESHOLD")
    print("=" * 70)
    print()

    print("A span is negated if ≥50% of its tokens overlap with negation region")
    print()

    lexicon = [
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA")
    ]

    text = "No burning but mild sensation"
    #       ^--------^         ^--------^
    #       neg region         symptom span

    print(f'Text: "{text}"')
    print()
    print("Negation region: covers 'burning'")
    print("Symptom span: 'burning sensation' (2 tokens)")
    print("Overlap: 1 token / 2 tokens = 50% ✓")
    print()

    spans = match_symptoms(text, lexicon, negation_window=5)
    if spans:
        span = spans[0]
        print(f"Result: \"{span.text}\" → {'NEGATED' if span.negated else 'POSITIVE'}")
    print()


def visualize_negation_regions():
    """Show negation region boundaries."""
    print("=" * 70)
    print("7. VISUALIZING NEGATION REGIONS")
    print("=" * 70)
    print()

    text = "No pain or discomfort but severe rash present"

    print(f'Text: "{text}"')
    print()

    # Detect negation regions (window=5)
    regions = detect_negated_regions(text, window=5)

    print("Negation regions (character positions):")
    for i, (start, end) in enumerate(regions, 1):
        negated_text = text[start:end]
        print(f'   Region {i}: [{start:3}:{end:3}] → "{negated_text}"')
    print()

    # Visual representation
    print("Visual representation:")
    print(f"   {text}")
    print("   ", end="")
    for i, char in enumerate(text):
        in_neg_region = any(start <= i < end for start, end in regions)
        print("^" if in_neg_region else " ", end="")
    print()
    print("   (^ = inside negation region)")
    print()


def main():
    """Run all negation detection examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "NEGATION DETECTION GUIDE" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_basic_negation()
    demo_negation_cues()
    demo_window_size()
    demo_bidirectional()
    demo_edge_cases()
    demo_overlap_threshold()
    visualize_negation_regions()

    print("=" * 70)
    print("BEST PRACTICES")
    print("=" * 70)
    print()
    print("✓ Use default window=5 for most consumer complaints")
    print("✓ Increase to window=7 for complex clinical notes")
    print("✓ Always validate negated spans in gold standard annotation")
    print("✓ Consider LLM refinement for double negatives and temporal context")
    print()
    print("⚠ Limitations:")
    print("   • Does not resolve double negatives")
    print("   • Does not analyze syntactic scope")
    print("   • Does not distinguish past vs present")
    print("   → Use LLM agent for advanced negation handling")
    print()


if __name__ == "__main__":
    main()
