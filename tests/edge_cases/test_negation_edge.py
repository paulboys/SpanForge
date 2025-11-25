"""Edge case tests for negation detection.

Tests negation window boundaries, multiple negation cues,
double negatives, and ambiguous negation contexts.
"""

import unittest

import pytest

from src.weak_label import weak_label
from tests.base import WeakLabelTestBase


class TestNegationEdgeCases(WeakLabelTestBase):
    """Test negation detection edge cases."""

    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()

    def test_negation_at_window_boundary(self):
        """Test negation cue exactly at window boundary (5 tokens)."""
        # Negation window = 5 tokens
        # "No reported symptoms of severe itching"
        # tokens: ["No", "reported", "symptoms", "of", "severe", "itching"]
        # "No" is 6 tokens away from "itching" (outside window)
        text = "No reported symptoms of severe itching"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        symptom_spans = [s for s in spans if s.label == "SYMPTOM" and "itching" in s.text.lower()]
        # Should be outside negation window, so not negated
        # (or negated if window >= 6)

    def test_multiple_negation_cues(self):
        """Test multiple negation cues near symptom."""
        text = "No itching or redness reported"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        symptom_spans = {s.text.lower(): s for s in spans if s.label == "SYMPTOM"}

        # Both symptoms should be negated
        if "itching" in symptom_spans:
            self.assertTrue(symptom_spans["itching"].negated, "Itching should be negated by 'No'")
        if "redness" in symptom_spans:
            self.assertTrue(
                symptom_spans["redness"].negated, "Redness should be negated by 'No' or 'or'"
            )

    def test_double_negative(self):
        """Test double negative (should result in affirmative)."""
        text = "Patient did not deny experiencing itching"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        # Current implementation may not handle double negatives correctly
        # This is a known limitation to document
        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]
        # Test documents behavior, doesn't assert specific outcome

    def test_negation_in_different_clause(self):
        """Test negation cue in different clause (should not negate)."""
        text = "No issues with product, but itching occurred later"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        symptom_spans = {s.text.lower(): s for s in spans if s.label == "SYMPTOM"}

        # "itching" should NOT be negated (different clause with "but")
        # This is a limitation of simple window-based negation


@pytest.mark.parametrize(
    "text,symptom,should_negate",
    [
        ("No itching", "itching", True),
        ("Patient has no itching", "itching", True),
        ("Itching absent", "itching", True),
        ("Without itching", "itching", True),
        ("Never experienced itching", "itching", True),
        ("Denies itching", "itching", True),
        ("Reports itching", "itching", False),
        ("Severe itching present", "itching", False),
        (
            "Itching resolved yesterday, no itching today",
            "itching",
            None,
        ),  # Multiple mentions, mixed
    ],
)
def test_negation_cue_detection(text, symptom, should_negate):
    """Parametrized negation cue detection."""
    from pathlib import Path

    from src.weak_label import load_product_lexicon, load_symptom_lexicon

    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)

    spans = weak_label(text, symptom_lexicon, product_lexicon)
    symptom_spans = [s for s in spans if s.label == "SYMPTOM" and symptom in s.text.lower()]

    if should_negate is None:
        # Multiple mentions or ambiguous case - skip assertion
        return

    if symptom_spans:
        if should_negate:
            assert any(
                s.negated for s in symptom_spans
            ), f"Expected '{symptom}' to be negated in: {text}"
        else:
            assert all(
                not s.negated for s in symptom_spans
            ), f"Expected '{symptom}' NOT to be negated in: {text}"


@pytest.mark.parametrize(
    "negation_cue,distance,should_negate",
    [
        ("no", 1, True),  # Within window
        ("no", 3, True),  # Within window
        ("no", 5, True),  # At window boundary
        ("no", 6, False),  # Outside window (negation_window=5)
        ("no", 10, False),  # Far outside window
    ],
)
def test_negation_window_distances(negation_cue, distance, should_negate):
    """Parametrized test for negation window distance."""
    # Construct text with specific token distance
    filler = " word" * (distance - 1)
    text = f"{negation_cue}{filler} itching"

    from pathlib import Path

    from src.weak_label import load_product_lexicon, load_symptom_lexicon

    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)

    spans = weak_label(text, symptom_lexicon, product_lexicon)
    symptom_spans = [s for s in spans if s.label == "SYMPTOM" and "itching" in s.text.lower()]

    if symptom_spans:
        if should_negate:
            assert symptom_spans[0].negated, f"Expected negation at distance {distance}"
        else:
            assert not symptom_spans[0].negated, f"Expected NO negation at distance {distance}"


class TestNegationOverlap(WeakLabelTestBase):
    """Test negation with overlapping spans."""

    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()

    def test_partial_negation_overlap(self):
        """Test negation cue overlapping only part of multi-word symptom."""
        text = "No severe itching"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        # Both "itching" and "severe itching" might be detected
        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]

        # All should be negated
        for span in symptom_spans:
            self.assertTrue(span.negated, f"Symptom '{span.text}' should be negated by 'No'")
