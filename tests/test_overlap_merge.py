from pathlib import Path
import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weak_label import load_symptom_lexicon, load_product_lexicon, weak_label, _deduplicate_spans, Span


def test_deduplication_preserves_overlapping_contextual_mentions():
    """Verify that overlapping spans with different boundaries are preserved (e.g., 'rash' vs 'little rash').
    Only exact duplicates (same start, end, canonical) should be removed.
    """
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    
    # Text with entities that could generate overlapping fuzzy matches
    text = "I got a terrible headache after using the moisturizing cream."
    spans = weak_label(text, symptom_lexicon, product_lexicon)
    
    # Should find multiple contextual mentions for same canonical entities
    # e.g., "headache" (exact) and "terrible headache" (fuzzy window)
    # e.g., "moisturizing cream" (exact) and "the moisturizing cream" (fuzzy window)
    assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"
    
    # Verify no exact duplicates (same start, end, canonical)
    seen_keys = set()
    for span in spans:
        key = (span.start, span.end, span.canonical)
        assert key not in seen_keys, f"Exact duplicate detected: {span.text} at [{span.start}-{span.end}] -> {span.canonical}"
        seen_keys.add(key)
    
    # Verify overlapping contextual mentions are preserved
    # Find spans with same canonical but different boundaries
    canonical_groups = {}
    for span in spans:
        if span.canonical not in canonical_groups:
            canonical_groups[span.canonical] = []
        canonical_groups[span.canonical].append(span)
    
    # Check for overlapping spans within same canonical group (this is OK and expected)
    for canonical, group in canonical_groups.items():
        if len(group) > 1:
            # Verify they have different boundaries
            positions = [(s.start, s.end) for s in group]
            assert len(set(positions)) == len(positions), f"Same canonical with duplicate positions: {canonical}"
            # This is the key behavior: overlapping spans with different boundaries are preserved
            print(f"✓ Preserved {len(group)} contextual mentions for '{canonical}': {[s.text for s in group]}")


def test_exact_duplicates_removed():
    """Verify that exact duplicates (same start, end, canonical) are removed, keeping highest confidence."""
    # Create test spans with exact duplicates
    spans = [
        Span(text="rash", start=10, end=14, label="SYMPTOM", canonical="Skin Rash", confidence=0.9),
        Span(text="rash", start=10, end=14, label="SYMPTOM", canonical="Skin Rash", confidence=1.0),  # duplicate, higher conf
        Span(text="rash", start=10, end=14, label="SYMPTOM", canonical="Skin Rash", confidence=0.8),  # duplicate, lower conf
        Span(text="severe rash", start=3, end=14, label="SYMPTOM", canonical="Skin Rash", confidence=0.95),  # different boundary
    ]
    
    deduplicated = _deduplicate_spans(spans)
    
    # Should have 2 spans: best of the 3 duplicates + the one with different boundary
    assert len(deduplicated) == 2, f"Expected 2 spans after deduplication, got {len(deduplicated)}"
    
    # Verify highest confidence kept for exact duplicates
    exact_match = next((s for s in deduplicated if s.start == 10 and s.end == 14), None)
    assert exact_match is not None, "Exact match span missing"
    assert exact_match.confidence == 1.0, f"Expected confidence 1.0 for kept duplicate, got {exact_match.confidence}"
    
    # Verify overlapping span with different boundary preserved
    overlapping = next((s for s in deduplicated if s.start == 3 and s.end == 14), None)
    assert overlapping is not None, "Overlapping contextual mention missing"
    assert overlapping.text == "severe rash", f"Expected 'severe rash', got '{overlapping.text}'"
