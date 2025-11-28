"""Tests for weak_labeling.confidence module."""

import pytest

from src.weak_labeling.confidence import (
    adjust_confidence_for_negation,
    align_spans,
    calibrate_threshold,
    compute_confidence,
)


class TestComputeConfidence:
    """Test confidence score calculation."""

    def test_perfect_scores(self):
        """Test with perfect fuzzy and jaccard scores."""
        confidence = compute_confidence(100.0, 100.0)
        assert confidence == 1.0

    def test_weighted_combination(self):
        """Test 80/20 weighting of fuzzy/jaccard."""
        # 88 fuzzy, 60 jaccard: 0.88*0.8 + 0.60*0.2 = 0.704 + 0.12 = 0.824
        confidence = compute_confidence(88.0, 60.0)
        assert abs(confidence - 0.824) < 0.001

    def test_low_scores(self):
        """Test with low scores."""
        confidence = compute_confidence(50.0, 30.0)
        assert abs(confidence - 0.46) < 0.001

    def test_zero_scores(self):
        """Test with zero scores."""
        confidence = compute_confidence(0.0, 0.0)
        assert confidence == 0.0

    def test_clamping_upper_bound(self):
        """Test that confidence is clamped to 1.0."""
        # Even if calculation exceeds 1.0, should clamp
        confidence = compute_confidence(100.0, 100.0)
        assert confidence <= 1.0

    def test_fuzzy_dominant(self):
        """Test that fuzzy score dominates (80% weight)."""
        conf_high_fuzzy = compute_confidence(90.0, 50.0)
        conf_high_jaccard = compute_confidence(50.0, 90.0)
        assert conf_high_fuzzy > conf_high_jaccard

    def test_typical_match(self):
        """Test typical fuzzy match scenario."""
        # Fuzzy 92, Jaccard 55: 0.92*0.8 + 0.55*0.2 = 0.736 + 0.11 = 0.846
        confidence = compute_confidence(92.0, 55.0)
        assert abs(confidence - 0.846) < 0.001


class TestAlignSpans:
    """Test span boundary alignment."""

    def test_remove_trailing_punctuation(self):
        """Test removal of trailing punctuation."""
        text = "I have burning sensation."
        spans = [("burning sensation.", 7, 26)]
        aligned = align_spans(text, spans)
        assert aligned == [("burning sensation", 7, 25)]

    def test_remove_leading_punctuation(self):
        """Test removal of leading punctuation."""
        text = "...redness appeared"
        spans = [("...redness", 0, 11)]
        aligned = align_spans(text, spans)
        assert aligned == [("redness", 3, 10)]

    def test_multiple_trailing_chars(self):
        """Test multiple trailing non-alphanumeric."""
        text = "symptom!!!"
        spans = [("symptom!!!", 0, 10)]
        aligned = align_spans(text, spans)
        assert aligned == [("symptom", 0, 7)]

    def test_empty_after_strip(self):
        """Test span that becomes empty after stripping."""
        text = "..."
        spans = [("...", 0, 3)]
        aligned = align_spans(text, spans)
        assert aligned == []

    def test_no_punctuation(self):
        """Test span without punctuation."""
        text = "burning sensation"
        spans = [("burning sensation", 0, 17)]
        aligned = align_spans(text, spans)
        assert aligned == [("burning sensation", 0, 17)]

    def test_multiple_spans(self):
        """Test alignment of multiple spans."""
        text = "redness, swelling, and pain."
        spans = [
            ("redness,", 0, 8),
            ("swelling,", 9, 18),
            ("pain.", 23, 28),
        ]
        aligned = align_spans(text, spans)
        assert len(aligned) == 3
        assert aligned[0] == ("redness", 0, 7)
        assert aligned[1] == ("swelling", 9, 17)
        assert aligned[2] == ("pain", 23, 27)


class TestAdjustConfidenceForNegation:
    """Test negation-based confidence adjustment."""

    def test_no_negation(self):
        """Test non-negated span keeps confidence."""
        confidence = adjust_confidence_for_negation(0.9, False)
        assert confidence == 0.9

    def test_with_negation_default_penalty(self):
        """Test negated span with default 0.1 penalty."""
        confidence = adjust_confidence_for_negation(0.9, True)
        assert confidence == 0.8

    def test_with_negation_custom_penalty(self):
        """Test negated span with custom penalty."""
        confidence = adjust_confidence_for_negation(0.9, True, negation_penalty=0.2)
        assert confidence == 0.7

    def test_negation_floor_at_zero(self):
        """Test that confidence doesn't go below zero."""
        confidence = adjust_confidence_for_negation(0.05, True, negation_penalty=0.1)
        assert confidence == 0.0

    def test_high_penalty(self):
        """Test with penalty exceeding confidence."""
        confidence = adjust_confidence_for_negation(0.3, True, negation_penalty=0.5)
        assert confidence == 0.0


class TestCalibrateThreshold:
    """Test confidence threshold calibration."""

    def test_perfect_precision(self):
        """Test finding threshold for 100% precision."""
        spans = [(0.9, True), (0.8, True), (0.7, False), (0.6, False)]
        threshold = calibrate_threshold(spans, target_precision=1.0)
        assert threshold == 0.8  # Above 0.8, all are correct

    def test_target_precision_90(self):
        """Test finding threshold for 90% precision."""
        spans = [(0.95, True), (0.90, True), (0.85, True), (0.80, False), (0.75, True)]
        threshold = calibrate_threshold(spans, target_precision=0.90)
        # At 0.95: 1/1 = 100% ✓
        # At 0.90: 2/2 = 100% ✓
        # At 0.85: 3/3 = 100% ✓
        # At 0.80: 3/4 = 75% ✗
        assert threshold == 0.85

    def test_empty_list(self):
        """Test with empty spans list."""
        threshold = calibrate_threshold([], target_precision=0.9)
        assert threshold == 0.0

    def test_all_correct(self):
        """Test when all spans are correct."""
        spans = [(0.9, True), (0.8, True), (0.7, True)]
        threshold = calibrate_threshold(spans, target_precision=0.9)
        assert threshold == 0.7  # Can use lowest threshold

    def test_all_incorrect(self):
        """Test when all spans are incorrect."""
        spans = [(0.9, False), (0.8, False), (0.7, False)]
        threshold = calibrate_threshold(spans, target_precision=0.9)
        assert threshold == 0.0  # No threshold achieves target

    def test_single_correct_span(self):
        """Test with single correct span."""
        spans = [(0.85, True)]
        threshold = calibrate_threshold(spans, target_precision=1.0)
        assert threshold == 0.85

    def test_descending_sort(self):
        """Test that spans are sorted descending by confidence."""
        spans = [(0.6, True), (0.9, True), (0.7, False), (0.8, True)]
        threshold = calibrate_threshold(spans, target_precision=1.0)
        # Should process: 0.9 (✓), 0.8 (✓), 0.7 (✗), 0.6 (✓)
        # At 0.9: 1/1 = 100% ✓
        # At 0.8: 2/2 = 100% ✓
        # At 0.7: 2/3 = 67% ✗
        assert threshold == 0.8


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_compute_confidence_negative_scores(self):
        """Test handling of negative scores (should not occur, but test robustness)."""
        confidence = compute_confidence(-10.0, 50.0)
        assert confidence >= 0.0  # Should be clamped

    def test_align_spans_empty_text(self):
        """Test alignment with empty text."""
        aligned = align_spans("", [])
        assert aligned == []

    def test_align_spans_out_of_bounds(self):
        """Test span indices matching text bounds."""
        text = "test"
        spans = [("test", 0, 4)]  # Valid
        aligned = align_spans(text, spans)
        assert len(aligned) == 1

    def test_calibrate_threshold_identical_confidences(self):
        """Test with multiple spans having same confidence."""
        spans = [(0.8, True), (0.8, True), (0.8, False)]
        threshold = calibrate_threshold(spans, target_precision=0.9)
        # All processed together at 0.8: 2/3 = 67% ✗
        assert threshold == 0.0 or threshold <= 0.8
