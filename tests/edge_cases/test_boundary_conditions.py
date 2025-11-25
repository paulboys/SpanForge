"""Edge case tests for boundary conditions.

Tests span boundaries at text extremes, zero-width, single-char,
and multi-word boundary edge cases.
"""

import unittest

import pytest

from tests.assertions import SpanAsserter
from tests.base import WeakLabelTestBase


class TestBoundaryConditions(WeakLabelTestBase):
    """Test span extraction at text boundaries."""

    def setUp(self):
        super().setUp()
        self.span_asserter = SpanAsserter(self)

    def test_span_at_text_start(self):
        """Test span starting at position 0."""
        spans = [{"start": 0, "end": 7, "label": "SYMPTOM", "text": "itching"}]
        text = "itching after cream use"

        # Should not raise
        self.span_asserter.assert_boundaries_valid(text, spans)
        self.span_asserter.assert_text_slices_match(text, spans)

    def test_span_at_text_end(self):
        """Test span ending at len(text)."""
        text = "Patient reports itching"
        spans = [{"start": 16, "end": 23, "label": "SYMPTOM", "text": "itching"}]

        self.span_asserter.assert_boundaries_valid(text, spans)
        self.span_asserter.assert_text_slices_match(text, spans)

    def test_single_char_span(self):
        """Test valid single-character span."""
        text = "Patient has a rash on face"
        spans = [{"start": 14, "end": 15, "label": "SYMPTOM", "text": "a"}]  # Unusual but valid

        self.span_asserter.assert_boundaries_valid(text, spans)

    def test_full_text_span(self):
        """Test span covering entire text."""
        text = "severe allergic reaction"
        spans = [{"start": 0, "end": 24, "label": "SYMPTOM", "text": text}]

        self.span_asserter.assert_boundaries_valid(text, spans)
        self.span_asserter.assert_text_slices_match(text, spans)


@pytest.mark.parametrize(
    "start,end,text_len,should_fail",
    [
        (0, 5, 10, False),  # Valid
        (5, 10, 10, False),  # Valid at end
        (-1, 5, 10, True),  # Negative start
        (5, 4, 10, True),  # End before start
        (5, 5, 10, True),  # Zero-width (start == end)
        (0, 11, 10, True),  # End beyond text
        (15, 20, 10, True),  # Both beyond text
    ],
)
def test_boundary_validation(start, end, text_len, should_fail):
    """Parametrized boundary validation tests."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    span_asserter = SpanAsserter(test_case)

    text = "x" * text_len
    spans = [{"start": start, "end": end, "label": "SYMPTOM"}]

    if should_fail:
        with pytest.raises(AssertionError):
            span_asserter.assert_boundaries_valid(text, spans)
    else:
        span_asserter.assert_boundaries_valid(text, spans)


@pytest.mark.parametrize(
    "text,span_text,start,end,should_match",
    [
        ("Patient has itching", "itching", 12, 19, True),
        ("Patient has itching", "ITCHING", 12, 19, False),  # Case mismatch
        ("Patient has itching", "itch", 12, 16, True),  # Substring
        ("Patient has itching", "itching!", 12, 20, False),  # Boundary exceeded
        ("Pruritus observed", "Pruritus", 0, 8, True),
        ("Pruritus observed", "Pruritus ", 0, 9, True),  # With trailing space
    ],
)
def test_text_slice_alignment(text, span_text, start, end, should_match):
    """Parametrized text slice alignment validation."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    span_asserter = SpanAsserter(test_case)

    spans = [{"start": start, "end": end, "label": "SYMPTOM", "text": span_text}]

    if should_match:
        span_asserter.assert_text_slices_match(text, spans)
    else:
        with pytest.raises(AssertionError):
            span_asserter.assert_text_slices_match(text, spans)
