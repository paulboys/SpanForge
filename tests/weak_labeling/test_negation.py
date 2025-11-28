"""Tests for weak_labeling.negation module."""

import pytest

from src.weak_labeling.negation import (
    NEGATION_TOKENS,
    detect_negated_regions,
    get_negation_cues,
    is_negated,
)


class TestDetectNegatedRegions:
    """Test negation region detection with bidirectional windows."""

    def test_simple_negation(self):
        """Test basic negation with 'no' cue."""
        text = "no burning sensation"
        regions = detect_negated_regions(text, window_size=5)
        # "no" at position 0-2, window extends forward 5 tokens
        assert len(regions) > 0
        # Should cover "burning sensation"
        assert any(0 <= r[0] and r[1] >= 20 for r in regions)

    def test_no_negation(self):
        """Test text without negation cues."""
        text = "burning sensation present"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) == 0

    def test_backward_window(self):
        """Test negation with backward window."""
        text = "burning sensation absent"
        regions = detect_negated_regions(text, window_size=5)
        # "absent" at end, window extends backward
        assert len(regions) > 0
        # Should cover "burning sensation"
        assert any(r[0] <= 0 and r[1] >= 17 for r in regions)

    def test_multiple_negations(self):
        """Test with multiple negation cues."""
        text = "no burning and no redness"
        regions = detect_negated_regions(text, window_size=5)
        # Should have regions for both "no" instances
        assert len(regions) >= 2

    def test_negation_at_start(self):
        """Test negation cue at text start."""
        text = "no symptoms"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) == 1
        # Forward window covers "symptoms" (not "no" itself)
        assert regions[0][0] == 3  # Start of "symptoms"
        assert regions[0][1] == 11  # End of "symptoms"

    def test_negation_at_end(self):
        """Test negation cue at text end."""
        text = "symptoms are absent"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) == 1
        # Should extend backward from "absent"

    def test_window_size_zero(self):
        """Test with zero window size (only negation token itself)."""
        text = "no burning sensation"
        regions = detect_negated_regions(text, window_size=0)
        # Should only cover "no" itself (0-2)
        assert len(regions) == 1
        assert regions[0][1] - regions[0][0] <= 3

    def test_large_window(self):
        """Test with large window size."""
        text = "no burning sensation or redness"
        regions = detect_negated_regions(text, window_size=20)
        # Large window should cover entire sentence
        assert len(regions) >= 1
        assert any(r[1] >= 30 for r in regions)

    def test_case_insensitive_negation(self):
        """Test that negation detection is case-insensitive."""
        text = "NO burning sensation"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) > 0

    def test_negation_in_middle(self):
        """Test negation cue in middle of text."""
        text = "patient reports no burning sensation today"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) > 0
        # Should cover "burning sensation"
        assert any(19 <= r[0] and r[1] >= 36 for r in regions)

    def test_empty_text(self):
        """Test with empty text."""
        regions = detect_negated_regions("", window_size=5)
        assert len(regions) == 0

    def test_overlapping_windows(self):
        """Test that overlapping negation windows are handled."""
        text = "no burning without redness"
        regions = detect_negated_regions(text, window_size=10)
        # Both "no" and "without" should create regions
        assert len(regions) >= 2


class TestIsNegated:
    """Test negation overlap detection."""

    def test_span_in_negated_region(self):
        """Test span fully within negated region."""
        negated_regions = [(0, 20)]
        # Span from 5-15 is within 0-20
        assert is_negated(5, 15, negated_regions) is True

    def test_span_outside_negated_region(self):
        """Test span outside all negated regions."""
        negated_regions = [(0, 10)]
        # Span from 20-30 is outside
        assert is_negated(20, 30, negated_regions) is False

    def test_span_partial_overlap_below_threshold(self):
        """Test partial overlap below 50% threshold."""
        negated_regions = [(0, 10)]
        # Span 8-18: overlap 8-10 (2 chars) / 10 chars total = 20% < 50%
        assert is_negated(8, 18, negated_regions) is False

    def test_span_partial_overlap_above_threshold(self):
        """Test partial overlap above 50% threshold."""
        negated_regions = [(0, 15)]
        # Span 5-15: overlap 5-15 (10 chars) / 10 chars total = 100% >= 50%
        assert is_negated(5, 15, negated_regions) is True

    def test_span_exactly_50_percent(self):
        """Test overlap at exactly 50% threshold."""
        negated_regions = [(0, 10)]
        # Span 5-15: overlap 5-10 (5 chars) / 10 chars total = 50% >= 50%
        assert is_negated(5, 15, negated_regions) is True

    def test_multiple_regions(self):
        """Test with multiple negated regions."""
        negated_regions = [(0, 10), (20, 30)]
        # Span in second region
        assert is_negated(22, 28, negated_regions) is True
        # Span between regions
        assert is_negated(12, 18, negated_regions) is False

    def test_no_negated_regions(self):
        """Test with empty negated regions list."""
        assert is_negated(0, 10, []) is False

    def test_span_overlaps_multiple_regions(self):
        """Test span overlapping multiple negated regions."""
        negated_regions = [(0, 10), (15, 25)]
        # Span 5-20: overlaps (5-10) = 5 chars, and (15-20) = 5 chars
        # Total overlap = 10 chars out of span length 15 = 66.67%
        # But is_negated checks if ANY region overlaps >=50%, not cumulative
        # (5-10) overlap with (5-20) = 5/15 = 33.3% < 50%
        # (15-25) overlap with (5-20) = 5/15 = 33.3% < 50%
        assert is_negated(5, 20, negated_regions) is False

    def test_zero_length_span(self):
        """Test with zero-length span."""
        negated_regions = [(0, 10)]
        # Zero-length span has no overlap
        assert is_negated(5, 5, negated_regions) is False

    def test_span_starts_before_region(self):
        """Test span starting before negated region."""
        negated_regions = [(10, 20)]
        # Span 0-15: overlap 10-15 (5 chars) / 15 chars = 33% < 50%
        assert is_negated(0, 15, negated_regions) is False

    def test_span_ends_after_region(self):
        """Test span ending after negated region."""
        negated_regions = [(0, 10)]
        # Span 5-20: overlap 5-10 (5 chars) / 15 chars = 33% < 50%
        assert is_negated(5, 20, negated_regions) is False


class TestGetNegationCues:
    """Test negation cue accessor."""

    def test_returns_negation_tokens(self):
        """Test that function returns the negation tokens set."""
        cues = get_negation_cues()
        assert isinstance(cues, set)
        assert len(cues) > 0

    def test_contains_common_negations(self):
        """Test that common negation words are included."""
        cues = get_negation_cues()
        assert "no" in cues
        assert "not" in cues
        assert "without" in cues
        assert "absent" in cues

    def test_returns_same_as_constant(self):
        """Test that function returns same as NEGATION_TOKENS constant."""
        cues = get_negation_cues()
        assert cues == NEGATION_TOKENS


class TestNegationTokensConstant:
    """Test NEGATION_TOKENS constant."""

    def test_negation_tokens_not_empty(self):
        """Test that negation tokens set is defined and not empty."""
        assert len(NEGATION_TOKENS) > 0

    def test_negation_tokens_is_set(self):
        """Test that NEGATION_TOKENS is a set."""
        assert isinstance(NEGATION_TOKENS, set)

    def test_common_negations_present(self):
        """Test that common negation terms are included."""
        expected = {"no", "not", "without", "absent", "denies", "negative"}
        assert expected.issubset(NEGATION_TOKENS)

    def test_negation_tokens_lowercase(self):
        """Test that all negation tokens are lowercase."""
        assert all(token.islower() or not token.isalpha() for token in NEGATION_TOKENS)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_detect_negation_only_whitespace(self):
        """Test with text containing only whitespace."""
        regions = detect_negated_regions("   ", window_size=5)
        assert len(regions) == 0

    def test_detect_negation_single_word(self):
        """Test with single negation word as entire text."""
        text = "no"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) == 1
        assert regions[0][0] == 0
        assert regions[0][1] == 2

    def test_is_negated_reversed_indices(self):
        """Test with start > end (invalid span)."""
        negated_regions = [(0, 10)]
        # Invalid span should return False
        assert is_negated(10, 5, negated_regions) is False

    def test_detect_negation_unicode(self):
        """Test negation detection with unicode text."""
        text = "no café or burning"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) > 0

    def test_detect_negation_punctuation_around_cue(self):
        """Test negation cue surrounded by punctuation."""
        text = "(no burning sensation)"
        regions = detect_negated_regions(text, window_size=5)
        assert len(regions) > 0
        # Should still detect "no" despite parentheses

    def test_is_negated_span_equals_region(self):
        """Test span exactly matching negated region."""
        negated_regions = [(10, 20)]
        # Span 10-20 has 100% overlap
        assert is_negated(10, 20, negated_regions) is True

    def test_detect_negation_repeated_cues(self):
        """Test with repeated negation cues."""
        text = "no no no burning"
        regions = detect_negated_regions(text, window_size=5)
        # Multiple "no" instances
        assert len(regions) >= 3

    def test_is_negated_adjacent_regions(self):
        """Test with adjacent but non-overlapping negated regions."""
        negated_regions = [(0, 10), (10, 20)]
        # Span 5-15 overlaps both
        assert is_negated(5, 15, negated_regions) is True

    def test_detect_negation_very_long_text(self):
        """Test with very long text."""
        text = "no " + "word " * 100 + "burning"
        regions = detect_negated_regions(text, window_size=50)
        assert len(regions) > 0

    def test_is_negated_very_small_overlap(self):
        """Test with very small overlap (1 character)."""
        negated_regions = [(0, 10)]
        # Span 9-20: overlap 9-10 (1 char) / 11 chars = 9% < 50%
        assert is_negated(9, 20, negated_regions) is False

    def test_detect_negation_window_larger_than_text(self):
        """Test with window size larger than text length."""
        text = "no burn"
        regions = detect_negated_regions(text, window_size=1000)
        assert len(regions) == 1
        # Should cover entire text despite large window

    def test_is_negated_custom_threshold(self):
        """Test with custom overlap threshold."""
        negated_regions = [(0, 10)]
        # Span 5-15: 50% overlap
        # Default threshold is 0.5, so this should be True
        result = is_negated(5, 15, negated_regions, overlap_threshold=0.5)
        assert result is True
        # With higher threshold (0.6), should be False
        result = is_negated(5, 15, negated_regions, overlap_threshold=0.6)
        assert result is False
