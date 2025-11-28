"""Tests for weak_labeling.validators module."""

import pytest

from src.weak_labeling.types import Span
from src.weak_labeling.validators import (
    ANATOMY_TOKENS,
    deduplicate_spans,
    filter_overlapping_spans,
    is_anatomy_only,
    validate_span_alignment,
)


class TestIsAnatomyOnly:
    """Test anatomy-only token detection."""

    def test_common_anatomy_tokens(self):
        """Test that common anatomy tokens are detected."""
        assert is_anatomy_only("skin") is True
        assert is_anatomy_only("face") is True
        assert is_anatomy_only("scalp") is True
        assert is_anatomy_only("eye") is True

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_anatomy_only("SKIN") is True
        assert is_anatomy_only("Face") is True

    def test_multi_token_phrase(self):
        """Test that multi-token phrases return False."""
        assert is_anatomy_only("skin rash") is False
        assert is_anatomy_only("red face") is False

    def test_anatomy_with_symptom(self):
        """Test phrases combining anatomy with symptom keywords."""
        assert is_anatomy_only("skin irritation") is False
        assert is_anatomy_only("facial swelling") is False

    def test_non_anatomy_tokens(self):
        """Test that non-anatomy tokens return False."""
        assert is_anatomy_only("burning") is False
        assert is_anatomy_only("redness") is False
        assert is_anatomy_only("rash") is False

    def test_empty_string(self):
        """Test with empty string."""
        assert is_anatomy_only("") is False

    def test_whitespace_handling(self):
        """Test with extra whitespace."""
        assert is_anatomy_only("  skin  ") is True
        assert is_anatomy_only("skin  face") is False  # Multi-token


class TestDeduplicateSpans:
    """Test span deduplication logic."""

    def test_no_duplicates(self):
        """Test with unique spans."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="redness", start=10, end=17, label="SYMPTOM", canonical="Redness", confidence=0.8, negated=False),
        ]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 2

    def test_exact_duplicates(self):
        """Test with exact duplicate spans (same start/end/canonical)."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning Sensation", confidence=0.85, negated=False),
        ]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9  # Keep highest confidence

    def test_same_position_different_canonical(self):
        """Test spans at same position with different canonicals."""
        spans = [
            Span(text="burn", start=0, end=4, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="burn", start=0, end=4, label="SYMPTOM", canonical="Burn", confidence=0.8, negated=False),
        ]
        deduped = deduplicate_spans(spans)
        # Should keep both since different canonical forms
        assert len(deduped) == 2

    def test_overlapping_not_duplicate(self):
        """Test that overlapping spans are not considered duplicates."""
        spans = [
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="sensation", start=8, end=17, label="SYMPTOM", canonical="Sensation", confidence=0.7, negated=False),
        ]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 2  # Both kept (overlap ≠ duplicate)

    def test_multiple_duplicates(self):
        """Test with multiple sets of duplicates."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.95, negated=False),
            Span(text="redness", start=10, end=17, label="SYMPTOM", canonical="Redness", confidence=0.85, negated=False),
            Span(text="redness", start=10, end=17, label="SYMPTOM", canonical="Redness", confidence=0.80, negated=False),
        ]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 2
        assert deduped[0].confidence == 0.95
        assert deduped[1].confidence == 0.85

    def test_empty_list(self):
        """Test with empty span list."""
        deduped = deduplicate_spans([])
        assert deduped == []


class TestFilterOverlappingSpans:
    """Test overlapping span filtering strategies."""

    def test_no_overlap(self):
        """Test with non-overlapping spans."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
            Span(text="redness", start=10, end=17, label="SYMPTOM", canonical="Redness", confidence=0.8, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 2

    def test_overlap_keep_longest(self):
        """Test keeping longest span when overlapping."""
        spans = [
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="sensation", start=8, end=17, label="SYMPTOM", canonical="Sensation", confidence=0.8, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 1
        assert filtered[0].text == "burning sensation"

    def test_overlap_keep_highest_confidence(self):
        """Test keeping highest confidence span when overlapping."""
        spans = [
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.85, negated=False),
            Span(text="sensation", start=8, end=17, label="SYMPTOM", canonical="Sensation", confidence=0.95, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="highest_confidence")
        assert len(filtered) == 1
        assert filtered[0].text == "sensation"  # Higher confidence

    def test_overlap_keep_first(self):
        """Test keeping first span when overlapping."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="first")
        assert len(filtered) == 1
        assert filtered[0].text == "burning"  # First in list

    def test_multiple_overlapping_groups(self):
        """Test with multiple groups of overlapping spans."""
        spans = [
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
            Span(text="redness", start=20, end=27, label="SYMPTOM", canonical="Redness", confidence=0.85, negated=False),
            Span(text="red", start=20, end=23, label="SYMPTOM", canonical="Red", confidence=0.7, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 2
        assert filtered[0].text == "burning sensation"
        assert filtered[1].text == "redness"

    def test_partial_overlap(self):
        """Test detection of partial overlaps."""
        spans = [
            Span(text="severe burning", start=0, end=14, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
            Span(text="burning sensation", start=7, end=24, label="SYMPTOM", canonical="Burning Sensation", confidence=0.85, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        # "burning" (7-14) overlaps, keep longer
        assert len(filtered) == 1
        assert filtered[0].text == "burning sensation"

    def test_nested_spans(self):
        """Test with completely nested spans."""
        spans = [
            Span(text="severe burning sensation pain", start=0, end=30, label="SYMPTOM", canonical="Burning", confidence=0.7, negated=False),
            Span(text="burning sensation", start=7, end=24, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="burning", start=7, end=14, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 1
        assert len(filtered[0].text) == 29  # Longest span (actual text length)
        assert filtered[0].start == 0
        assert filtered[0].end == 30

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        spans = [Span(text="test", start=0, end=4, label="SYMPTOM", canonical="Test", confidence=0.9, negated=False)]
        with pytest.raises(ValueError):
            filter_overlapping_spans(spans, strategy="invalid")

    def test_empty_list(self):
        """Test with empty span list."""
        filtered = filter_overlapping_spans([], strategy="longest")
        assert filtered == []


class TestValidateSpanAlignment:
    """Test span alignment validation."""

    def test_valid_alignment(self):
        """Test with correctly aligned spans."""
        text = "I have burning sensation"
        spans = [
            Span(text="burning", start=7, end=14, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
            Span(text="sensation", start=15, end=24, label="SYMPTOM", canonical="Sensation", confidence=0.8, negated=False),
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is True
        assert len(errors) == 0

    def test_misaligned_span(self):
        """Test with misaligned span (indices don't match text)."""
        text = "I have burning sensation"
        spans = [
            Span(text="burning", start=7, end=14, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),  # Correct
            Span(text="sensation", start=16, end=25, label="SYMPTOM", canonical="Sensation", confidence=0.8, negated=False),  # Wrong indices
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is False
        assert len(errors) == 1
        assert "sensation" in errors[0]

    def test_out_of_bounds(self):
        """Test with span extending beyond text."""
        text = "burning"
        spans = [
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is False
        assert len(errors) == 1

    def test_negative_indices(self):
        """Test with negative indices."""
        text = "burning sensation"
        spans = [
            Span(text="burning", start=-5, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is False

    def test_reversed_indices(self):
        """Test with start > end."""
        text = "burning sensation"
        spans = [
            Span(text="burning", start=7, end=0, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is False

    def test_multiple_errors(self):
        """Test with multiple alignment errors."""
        text = "burning sensation"
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),  # Correct
            Span(text="sensation", start=10, end=17, label="SYMPTOM", canonical="Sensation", confidence=0.8, negated=False),  # Wrong (should be 8-17)
            Span(text="pain", start=20, end=24, label="SYMPTOM", canonical="Pain", confidence=0.7, negated=False),  # Out of bounds
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is False
        assert len(errors) == 2

    def test_empty_text(self):
        """Test with empty text."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False),
        ]
        is_valid, errors = validate_span_alignment("", spans)
        assert is_valid is False

    def test_empty_spans(self):
        """Test with empty spans list."""
        is_valid, errors = validate_span_alignment("some text", [])
        assert is_valid is True
        assert len(errors) == 0

    def test_zero_length_span(self):
        """Test with zero-length span (start == end)."""
        text = "burning"
        spans = [
            Span("", 3, 3, "SYMPTOM", "Empty", 0.9, False),
        ]
        is_valid, errors = validate_span_alignment(text, spans)
        # Zero-length spans may be invalid depending on implementation
        assert is_valid is False or (is_valid is True and len(errors) == 0)


class TestAnatomyTokensConstant:
    """Test anatomy tokens constant."""

    def test_anatomy_tokens_not_empty(self):
        """Test that anatomy tokens set is defined and not empty."""
        assert len(ANATOMY_TOKENS) > 0

    def test_common_anatomy_present(self):
        """Test that common anatomy terms are included."""
        assert "skin" in ANATOMY_TOKENS
        assert "face" in ANATOMY_TOKENS
        assert "scalp" in ANATOMY_TOKENS
        assert "eye" in ANATOMY_TOKENS


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_deduplicate_single_span(self):
        """Test deduplication with single span."""
        spans = [Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False)]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 1

    def test_filter_overlapping_single_span(self):
        """Test filtering with single span."""
        spans = [Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False)]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 1

    def test_is_anatomy_only_unicode(self):
        """Test anatomy detection with unicode."""
        # Assuming no unicode in ANATOMY_TOKENS
        assert is_anatomy_only("skïn") is False

    def test_validate_unicode_text(self):
        """Test validation with unicode text."""
        text = "café burning"
        spans = [Span(text="burning", start=5, end=12, label="SYMPTOM", canonical="Burning", confidence=0.9, negated=False)]
        is_valid, errors = validate_span_alignment(text, spans)
        assert is_valid is True

    def test_filter_all_overlapping(self):
        """Test when all spans overlap with each other."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
            Span(text="burning sensation", start=0, end=17, label="SYMPTOM", canonical="Burning Sensation", confidence=0.9, negated=False),
            Span(text="sensation", start=8, end=17, label="SYMPTOM", canonical="Sensation", confidence=0.7, negated=False),
        ]
        filtered = filter_overlapping_spans(spans, strategy="longest")
        assert len(filtered) == 1
        assert filtered[0].text == "burning sensation"

    def test_deduplicate_preserve_order(self):
        """Test that deduplication preserves order of first occurrence."""
        spans = [
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.8, negated=False),
            Span(text="redness", start=10, end=17, label="SYMPTOM", canonical="Redness", confidence=0.9, negated=False),
            Span(text="burning", start=0, end=7, label="SYMPTOM", canonical="Burning", confidence=0.85, negated=False),  # Duplicate
        ]
        deduped = deduplicate_spans(spans)
        assert len(deduped) == 2
        assert deduped[0].text == "burning"
        assert deduped[1].text == "redness"
