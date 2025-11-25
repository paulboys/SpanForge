"""Composition-based assertion helpers for SpanForge tests.

Provides reusable assertion utilities that can be composed into test classes
to validate spans, overlaps, integrity constraints, and provenance.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Set, Tuple


class SpanAsserter:
    """Assertion helper for span validation."""

    def __init__(self, test_case: unittest.TestCase):
        self.tc = test_case

    def assert_span_equals(
        self, expected: Dict[str, Any], actual: Dict[str, Any], check_confidence: bool = True
    ):
        """Assert two spans are equal (start, end, label, canonical).

        Args:
            expected: Expected span dict
            actual: Actual span dict
            check_confidence: Whether to check confidence field
        """
        self.tc.assertEqual(
            expected["start"], actual["start"], f"Span start mismatch: {expected} vs {actual}"
        )
        self.tc.assertEqual(
            expected["end"], actual["end"], f"Span end mismatch: {expected} vs {actual}"
        )
        self.tc.assertEqual(
            expected["label"], actual["label"], f"Span label mismatch: {expected} vs {actual}"
        )

        if "canonical" in expected:
            self.tc.assertEqual(
                expected["canonical"],
                actual.get("canonical"),
                f"Canonical mismatch: {expected} vs {actual}",
            )

        if check_confidence and "confidence" in expected:
            self.tc.assertAlmostEqual(
                expected["confidence"],
                actual.get("confidence", 1.0),
                places=2,
                msg=f"Confidence mismatch: {expected} vs {actual}",
            )

    def assert_span_list_contains(
        self, expected_span: Dict[str, Any], span_list: List[Dict[str, Any]]
    ):
        """Assert span list contains a span matching expected.

        Args:
            expected_span: Span to find
            span_list: List of spans to search
        """
        for span in span_list:
            try:
                self.assert_span_equals(expected_span, span, check_confidence=False)
                return  # Found match
            except AssertionError:
                continue
        self.tc.fail(f"Expected span not found in list: {expected_span}\nList: {span_list}")

    def assert_boundaries_valid(self, text: str, spans: List[Dict[str, Any]]):
        """Assert all span boundaries are valid within text.

        Args:
            text: Source text
            spans: List of spans to validate
        """
        for span in spans:
            start = span.get("start")
            end = span.get("end")
            self.tc.assertIsInstance(start, int, f"Span start not int: {span}")
            self.tc.assertIsInstance(end, int, f"Span end not int: {span}")
            self.tc.assertGreaterEqual(start, 0, f"Negative start: {span}")
            self.tc.assertLess(end, len(text) + 1, f"End beyond text: {span}")
            self.tc.assertLess(start, end, f"Invalid boundaries (start >= end): {span}")

    def assert_text_slices_match(self, text: str, spans: List[Dict[str, Any]]):
        """Assert span text field matches text slice at span boundaries.

        Args:
            text: Source text
            spans: List of spans with 'text' field
        """
        for span in spans:
            if "text" not in span:
                continue
            start, end = span["start"], span["end"]
            expected_text = text[start:end]
            self.tc.assertEqual(
                expected_text, span["text"], f"Text slice mismatch at [{start}:{end}]"
            )

    def assert_no_duplicate_spans(self, spans: List[Dict[str, Any]]):
        """Assert no exact duplicate spans (same start, end, label, canonical).

        Args:
            spans: List of spans
        """
        seen: Set[Tuple[int, int, str, Optional[str]]] = set()
        for span in spans:
            key = (span["start"], span["end"], span["label"], span.get("canonical"))
            self.tc.assertNotIn(key, seen, f"Duplicate span detected: {span}")
            seen.add(key)


class OverlapChecker:
    """Assertion helper for overlap validation."""

    def __init__(self, test_case: unittest.TestCase):
        self.tc = test_case

    def compute_overlap(self, span_a: Dict[str, Any], span_b: Dict[str, Any]) -> int:
        """Compute character overlap between two spans.

        Returns:
            Number of overlapping characters
        """
        return max(0, min(span_a["end"], span_b["end"]) - max(span_a["start"], span_b["start"]))

    def compute_iou(self, span_a: Dict[str, Any], span_b: Dict[str, Any]) -> float:
        """Compute intersection-over-union for two spans.

        Returns:
            IOU score (0.0 to 1.0)
        """
        overlap = self.compute_overlap(span_a, span_b)
        if overlap == 0:
            return 0.0
        union = (span_a["end"] - span_a["start"]) + (span_b["end"] - span_b["start"]) - overlap
        return overlap / union if union > 0 else 0.0

    def assert_no_conflicting_labels(self, spans: List[Dict[str, Any]]):
        """Assert overlapping spans have same label.

        Args:
            spans: List of spans to check
        """
        for i, span_a in enumerate(spans):
            for span_b in spans[i + 1 :]:
                overlap = self.compute_overlap(span_a, span_b)
                if overlap > 0:
                    self.tc.assertEqual(
                        span_a["label"],
                        span_b["label"],
                        f"Conflicting labels on overlapping spans: {span_a} vs {span_b}",
                    )

    def assert_overlap_matrix_valid(
        self, spans: List[Dict[str, Any]], allow_same_label_overlap: bool = True
    ):
        """Validate all pairwise overlaps follow rules.

        Args:
            spans: List of spans
            allow_same_label_overlap: Whether overlaps with same label are permitted
        """
        for i, span_a in enumerate(spans):
            for span_b in spans[i + 1 :]:
                overlap = self.compute_overlap(span_a, span_b)
                if overlap > 0:
                    if not allow_same_label_overlap or span_a["label"] != span_b["label"]:
                        self.tc.fail(
                            f"Unexpected overlap: {span_a} overlaps {span_b} by {overlap} chars"
                        )


class IntegrityValidator:
    """Assertion helper for gold dataset integrity."""

    ALLOWED_LABELS = {"SYMPTOM", "PRODUCT"}

    def __init__(self, test_case: unittest.TestCase):
        self.tc = test_case

    def assert_provenance_present(self, record: Dict[str, Any]):
        """Assert provenance fields are present and valid.

        Args:
            record: Gold record dict
        """
        self.tc.assertIn("source", record, "Missing 'source' field")
        self.tc.assertTrue(record.get("source"), "'source' field is empty")

        self.tc.assertIn("annotator", record, "Missing 'annotator' field")
        self.tc.assertTrue(record.get("annotator"), "'annotator' field is empty")

        if "revision" in record:
            self.tc.assertIsInstance(record["revision"], int, "'revision' not int")
            self.tc.assertGreaterEqual(record["revision"], 0, "'revision' negative")

    def assert_canonical_present(self, entities: List[Dict[str, Any]]):
        """Assert all entities have non-empty canonical field.

        Args:
            entities: List of entity dicts
        """
        for ent in entities:
            self.tc.assertIn("canonical", ent, f"Missing 'canonical' in entity: {ent}")
            self.tc.assertTrue(ent.get("canonical"), f"Empty 'canonical' in entity: {ent}")

    def assert_labels_valid(self, entities: List[Dict[str, Any]]):
        """Assert all entity labels are in allowed set and text is non-empty.

        Args:
            entities: List of entity dicts
        """
        for ent in entities:
            label = ent.get("label")
            text = ent.get("text", "")

            self.tc.assertIn(
                label, self.ALLOWED_LABELS, f"Invalid label '{label}' in entity: {ent}"
            )
            self.tc.assertTrue(len(text) > 0, f"Empty text in entity: {ent}")

    def assert_no_duplicates(self, entities: List[Dict[str, Any]]):
        """Assert no duplicate entities (same start, end, label).

        Args:
            entities: List of entity dicts
        """
        seen: Set[Tuple[int, int, str]] = set()
        for ent in entities:
            key = (ent["start"], ent["end"], ent["label"])
            self.tc.assertNotIn(key, seen, f"Duplicate entity: {ent}")
            seen.add(key)

    def assert_sorted_by_start(self, entities: List[Dict[str, Any]]):
        """Assert entities are sorted by start position.

        Args:
            entities: List of entity dicts
        """
        for i in range(len(entities) - 1):
            self.tc.assertLessEqual(
                entities[i]["start"],
                entities[i + 1]["start"],
                f"Entities not sorted: {entities[i]} before {entities[i+1]}",
            )

    def validate_full_record(self, record: Dict[str, Any]):
        """Run all integrity checks on a gold record.

        Args:
            record: Gold record dict

        Raises:
            KeyError: If required fields (text, entities) are missing
            AssertionError: If validation checks fail
        """
        # Check required fields first (raises KeyError if missing)
        if "text" not in record:
            raise KeyError("Missing required field 'text'")
        if "entities" not in record:
            raise KeyError("Missing required field 'entities'")

        text = record["text"]
        entities = record["entities"]

        self.assert_provenance_present(record)
        self.assert_canonical_present(entities)
        self.assert_labels_valid(entities)
        self.assert_no_duplicates(entities)
        self.assert_sorted_by_start(entities)

        # Boundary validation
        span_asserter = SpanAsserter(self.tc)
        span_asserter.assert_boundaries_valid(text, entities)
        span_asserter.assert_text_slices_match(text, entities)

        # Overlap validation
        overlap_checker = OverlapChecker(self.tc)
        overlap_checker.assert_no_conflicting_labels(entities)
