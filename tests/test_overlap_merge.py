"""Overlap and deduplication tests with OverlapChecker composition.

Tests span deduplication logic, exact duplicate removal, contextual mention
preservation, and parametrized overlap scenarios.
"""

import pathlib
import sys
import unittest
from pathlib import Path

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weak_label import (
    Span,
    _deduplicate_spans,
    load_product_lexicon,
    load_symptom_lexicon,
    weak_label,
)
from tests.assertions import OverlapChecker, SpanAsserter
from tests.base import WeakLabelTestBase


class TestSpanDeduplication(WeakLabelTestBase):
    """Test span deduplication and overlap handling."""

    def setUp(self):
        super().setUp()
        self.overlap_checker = OverlapChecker(self)
        self.span_asserter = SpanAsserter(self)

    def test_exact_duplicates_removed(self):
        """Verify exact duplicates (same start, end, canonical) are removed, keeping highest confidence."""
        spans = [
            Span(
                text="rash",
                start=10,
                end=14,
                label="SYMPTOM",
                canonical="Skin Rash",
                confidence=0.9,
            ),
            Span(
                text="rash",
                start=10,
                end=14,
                label="SYMPTOM",
                canonical="Skin Rash",
                confidence=1.0,
            ),  # duplicate, higher conf
            Span(
                text="rash",
                start=10,
                end=14,
                label="SYMPTOM",
                canonical="Skin Rash",
                confidence=0.8,
            ),  # duplicate, lower conf
            Span(
                text="severe rash",
                start=3,
                end=14,
                label="SYMPTOM",
                canonical="Skin Rash",
                confidence=0.95,
            ),  # different boundary
        ]

        deduplicated = _deduplicate_spans(spans)

        # Should have 2 spans: best of 3 duplicates + different boundary
        self.assertEqual(
            len(deduplicated), 2, f"Expected 2 spans after deduplication, got {len(deduplicated)}"
        )

        # Verify highest confidence kept for exact duplicates
        exact_match = next((s for s in deduplicated if s.start == 10 and s.end == 14), None)
        self.assertIsNotNone(exact_match, "Exact match span missing")
        self.assertEqual(
            exact_match.confidence,
            1.0,
            f"Expected confidence 1.0 for kept duplicate, got {exact_match.confidence}",
        )

        # Verify overlapping span with different boundary preserved
        overlapping = next((s for s in deduplicated if s.start == 3 and s.end == 14), None)
        self.assertIsNotNone(overlapping, "Overlapping contextual mention missing")
        self.assertEqual(
            overlapping.text, "severe rash", f"Expected 'severe rash', got '{overlapping.text}'"
        )

    def test_contextual_mentions_preserved(self):
        """Verify overlapping spans with different boundaries are preserved."""
        symptom_lexicon = self.create_symptom_lexicon()
        product_lexicon = self.create_product_lexicon()

        text = "I got a terrible headache after using the moisturizing cream."
        spans = weak_label(text, symptom_lexicon, product_lexicon)

        self.assertGreaterEqual(len(spans), 2, f"Expected at least 2 spans, got {len(spans)}")

        # Verify no exact duplicates (same start, end, canonical)
        self.span_asserter.assert_no_duplicate_spans(
            [
                {"start": s.start, "end": s.end, "label": s.label, "canonical": s.canonical}
                for s in spans
            ]
        )

        # Group by canonical to find contextual variations
        canonical_groups = {}
        for span in spans:
            if span.canonical not in canonical_groups:
                canonical_groups[span.canonical] = []
            canonical_groups[span.canonical].append(span)

        # Check for overlapping spans within same canonical group
        for canonical, group in canonical_groups.items():
            if len(group) > 1:
                # Verify different boundaries
                positions = [(s.start, s.end) for s in group]
                self.assertEqual(
                    len(set(positions)),
                    len(positions),
                    f"Same canonical with duplicate positions: {canonical}",
                )


@pytest.mark.parametrize(
    "span_a,span_b,expected_overlap,expected_iou",
    [
        # Adjacent (no overlap)
        ({"start": 0, "end": 5}, {"start": 6, "end": 10}, 0, 0.0),
        # Exact overlap
        ({"start": 10, "end": 20}, {"start": 10, "end": 20}, 10, 1.0),
        # Nested (A contains B)
        ({"start": 10, "end": 30}, {"start": 15, "end": 25}, 10, 0.5),
        # Partial overlap (IOU = 5 / (15+15-5) = 5/25 = 0.2)
        ({"start": 10, "end": 25}, {"start": 20, "end": 35}, 5, 0.2),
        # Disjoint
        ({"start": 0, "end": 10}, {"start": 20, "end": 30}, 0, 0.0),
    ],
)
def test_overlap_computation(span_a, span_b, expected_overlap, expected_iou):
    """Parametrized test for overlap and IOU computation."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    checker = OverlapChecker(test_case)

    actual_overlap = checker.compute_overlap(span_a, span_b)
    actual_iou = checker.compute_iou(span_a, span_b)

    assert (
        actual_overlap == expected_overlap
    ), f"Expected overlap {expected_overlap}, got {actual_overlap}"
    assert abs(actual_iou - expected_iou) < 0.01, f"Expected IOU {expected_iou}, got {actual_iou}"


class TestConflictingLabels(unittest.TestCase):
    """Test overlap conflict detection."""

    def setUp(self):
        self.overlap_checker = OverlapChecker(self)

    def test_same_label_overlap_allowed(self):
        """Verify overlapping spans with same label don't raise conflict."""
        spans = [
            {"start": 10, "end": 30, "label": "SYMPTOM"},
            {"start": 20, "end": 35, "label": "SYMPTOM"},
        ]
        # Should not raise
        self.overlap_checker.assert_no_conflicting_labels(spans)

    def test_different_label_overlap_raises(self):
        """Verify overlapping spans with different labels raise conflict."""
        spans = [
            {"start": 10, "end": 30, "label": "SYMPTOM"},
            {"start": 20, "end": 35, "label": "PRODUCT"},
        ]
        with self.assertRaises(AssertionError):
            self.overlap_checker.assert_no_conflicting_labels(spans)
