"""Edge case tests for overlap scenarios.

Tests nested spans, partial overlaps, adjacent spans,
and overlap resolution strategies.
"""
import pytest
import unittest
from tests.base import TestBase
from tests.assertions import OverlapChecker, SpanAsserter


class TestOverlapScenarios(unittest.TestCase):
    """Test various span overlap patterns."""
    
    def setUp(self):
        self.overlap_checker = OverlapChecker(self)
        self.span_asserter = SpanAsserter(self)
    
    def test_nested_same_label(self):
        """Test nested spans with same label (allowed)."""
        spans = [
            {"start": 10, "end": 30, "label": "SYMPTOM", 
             "text": "severe itching", "canonical": "pruritus"},
            {"start": 17, "end": 30, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus"},
        ]
        
        # Should not raise conflict
        self.overlap_checker.assert_no_conflicting_labels(spans)
    
    def test_nested_different_labels(self):
        """Test nested spans with different labels (conflict)."""
        spans = [
            {"start": 10, "end": 30, "label": "SYMPTOM", 
             "text": "dry skin cream", "canonical": "dryness"},
            {"start": 18, "end": 30, "label": "PRODUCT", 
             "text": "skin cream", "canonical": "moisturizer"},
        ]
        
        # Should raise conflict
        with self.assertRaises(AssertionError):
            self.overlap_checker.assert_no_conflicting_labels(spans)
    
    def test_partial_overlap_same_label(self):
        """Test partial overlap with same label."""
        spans = [
            {"start": 10, "end": 25, "label": "SYMPTOM", 
             "text": "itching and burn", "canonical": "pruritus"},
            {"start": 20, "end": 30, "label": "SYMPTOM", 
             "text": "burning", "canonical": "burning"},
        ]
        
        # Overlap exists
        overlap = self.overlap_checker.compute_overlap(spans[0], spans[1])
        self.assertGreater(overlap, 0, "Should have positive overlap")
        
        # No conflict (same label)
        self.overlap_checker.assert_no_conflicting_labels(spans)
    
    def test_adjacent_spans_no_overlap(self):
        """Test adjacent spans with no overlap."""
        spans = [
            {"start": 10, "end": 18, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus"},
            {"start": 19, "end": 26, "label": "SYMPTOM", 
             "text": "redness", "canonical": "erythema"},
        ]
        
        # No overlap
        overlap = self.overlap_checker.compute_overlap(spans[0], spans[1])
        self.assertEqual(overlap, 0, "Adjacent spans should have zero overlap")
    
    def test_exact_duplicate_spans(self):
        """Test exact duplicate detection."""
        spans = [
            {"start": 10, "end": 18, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus"},
            {"start": 10, "end": 18, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus"},
        ]
        
        # Should detect duplicate
        with self.assertRaises(AssertionError):
            self.span_asserter.assert_no_duplicate_spans(spans)


@pytest.mark.parametrize("span_a,span_b,expected_overlap,expected_conflict", [
    # (span_a, span_b, expected_overlap_chars, expected_conflict)
    # Adjacent - no overlap
    ({"start": 0, "end": 5, "label": "SYMPTOM"}, 
     {"start": 6, "end": 10, "label": "SYMPTOM"}, 
     0, False),
    
    # Touching boundaries - no overlap
    ({"start": 0, "end": 5, "label": "SYMPTOM"}, 
     {"start": 5, "end": 10, "label": "SYMPTOM"}, 
     0, False),
    
    # Nested - same label (no conflict)
    ({"start": 0, "end": 20, "label": "SYMPTOM"}, 
     {"start": 5, "end": 15, "label": "SYMPTOM"}, 
     10, False),
    
    # Nested - different labels (conflict)
    ({"start": 0, "end": 20, "label": "SYMPTOM"}, 
     {"start": 5, "end": 15, "label": "PRODUCT"}, 
     10, True),
    
    # Partial overlap - same label (no conflict)
    ({"start": 0, "end": 15, "label": "SYMPTOM"}, 
     {"start": 10, "end": 25, "label": "SYMPTOM"}, 
     5, False),
    
    # Partial overlap - different labels (conflict)
    ({"start": 0, "end": 15, "label": "SYMPTOM"}, 
     {"start": 10, "end": 25, "label": "PRODUCT"}, 
     5, True),
    
    # Exact duplicate - same label (no conflict per se, but duplicate)
    ({"start": 10, "end": 20, "label": "SYMPTOM"}, 
     {"start": 10, "end": 20, "label": "SYMPTOM"}, 
     10, False),
    
    # Single char overlap - different labels (conflict)
    ({"start": 10, "end": 20, "label": "SYMPTOM"}, 
     {"start": 19, "end": 25, "label": "PRODUCT"}, 
     1, True),
])
def test_overlap_patterns(span_a, span_b, expected_overlap, expected_conflict):
    """Parametrized overlap pattern tests."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    overlap_checker = OverlapChecker(test_case)
    
    # Check overlap computation
    actual_overlap = overlap_checker.compute_overlap(span_a, span_b)
    assert actual_overlap == expected_overlap, \
        f"Expected overlap {expected_overlap}, got {actual_overlap}"
    
    # Check conflict detection
    spans = [span_a, span_b]
    if expected_conflict:
        with pytest.raises(AssertionError):
            overlap_checker.assert_no_conflicting_labels(spans)
    else:
        overlap_checker.assert_no_conflicting_labels(spans)


@pytest.mark.parametrize("iou_threshold,span_a,span_b,should_match", [
    # IOU threshold, span_a, span_b, should_match
    (0.5, {"start": 10, "end": 20}, {"start": 10, "end": 20}, True),  # Exact match, IOU=1.0
    (0.5, {"start": 10, "end": 20}, {"start": 12, "end": 22}, True),  # Partial overlap, IOU=0.6
    (0.5, {"start": 10, "end": 20}, {"start": 15, "end": 25}, False), # Partial overlap, IOU=0.33
    (0.5, {"start": 10, "end": 20}, {"start": 25, "end": 35}, False), # No overlap, IOU=0.0
    (0.8, {"start": 10, "end": 20}, {"start": 11, "end": 19}, True),  # Nested, IOU=0.8 (exact boundary)
    (0.8, {"start": 10, "end": 20}, {"start": 10, "end": 19}, True),  # Subset, IOU=0.9
])
def test_iou_thresholds(iou_threshold, span_a, span_b, should_match):
    """Parametrized IOU threshold tests."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    overlap_checker = OverlapChecker(test_case)
    
    iou = overlap_checker.compute_iou(span_a, span_b)
    
    if should_match:
        assert iou >= iou_threshold, \
            f"Expected IOU >= {iou_threshold}, got {iou}"
    else:
        assert iou < iou_threshold, \
            f"Expected IOU < {iou_threshold}, got {iou}"


class TestOverlapResolution(TestBase):
    """Test overlap resolution strategies."""
    
    def setUp(self):
        super().setUp()
        self.overlap_checker = OverlapChecker(self)
    
    def test_longest_span_preference(self):
        """Test that longest span is preferred in deduplication."""
        spans = [
            {"start": 10, "end": 20, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus", "confidence": 0.9},
            {"start": 10, "end": 30, "label": "SYMPTOM", 
             "text": "severe itching", "canonical": "pruritus", "confidence": 0.85},
        ]
        
        # Longer span (10-30) should be preferred if confidence is similar
        # This is a policy decision to document
    
    def test_highest_confidence_preference(self):
        """Test that highest confidence span is preferred."""
        spans = [
            {"start": 10, "end": 20, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus", "confidence": 0.95},
            {"start": 10, "end": 20, "label": "SYMPTOM", 
             "text": "itching", "canonical": "pruritus", "confidence": 0.75},
        ]
        
        # Higher confidence (0.95) should be kept in deduplication
        # Exact duplicates → keep highest confidence
