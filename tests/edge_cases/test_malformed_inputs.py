"""Edge case tests for malformed inputs.

Tests handling of empty texts, missing fields, invalid span data,
and corrupted JSONL records.
"""

import json
import unittest

import pytest

from src.weak_label import weak_label
from tests.assertions import IntegrityValidator, SpanAsserter
from tests.base import IntegrationTestBase, WeakLabelTestBase


class TestMalformedTextInputs(WeakLabelTestBase):
    """Test handling of malformed text inputs."""

    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()

    def test_empty_text(self):
        """Test weak labeling with empty text."""
        text = ""
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        self.assertEqual(len(spans), 0, "Empty text should produce no spans")

    def test_whitespace_only_text(self):
        """Test text containing only whitespace."""
        text = "   \n\t  "
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        self.assertEqual(len(spans), 0, "Whitespace-only text should produce no spans")

    def test_very_long_text(self):
        """Test handling of very long text (>10k chars)."""
        text = "Patient reports itching. " * 1000  # ~25k chars
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        # Should handle without error
        self.assertGreater(len(spans), 0, "Should detect symptoms in long text")

    def test_text_with_null_bytes(self):
        """Test text containing null bytes."""
        text = "Patient has\x00itching"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        # Should handle gracefully (may or may not detect symptom)
        # Main concern: no crash


class TestMalformedSpanData(unittest.TestCase):
    """Test handling of malformed span data."""

    def setUp(self):
        self.span_asserter = SpanAsserter(self)

    def test_missing_start_field(self):
        """Test span missing 'start' field."""
        text = "Patient has itching"
        spans = [{"end": 19, "label": "SYMPTOM"}]  # Missing 'start'

        with self.assertRaises((KeyError, AssertionError)):
            self.span_asserter.assert_boundaries_valid(text, spans)

    def test_missing_end_field(self):
        """Test span missing 'end' field."""
        text = "Patient has itching"
        spans = [{"start": 12, "label": "SYMPTOM"}]  # Missing 'end'

        with self.assertRaises((KeyError, AssertionError)):
            self.span_asserter.assert_boundaries_valid(text, spans)

    def test_non_integer_boundaries(self):
        """Test span with non-integer start/end."""
        text = "Patient has itching"
        spans = [{"start": "12", "end": "19", "label": "SYMPTOM"}]  # String instead of int

        with self.assertRaises(AssertionError):
            self.span_asserter.assert_boundaries_valid(text, spans)

    def test_float_boundaries(self):
        """Test span with float start/end."""
        text = "Patient has itching"
        spans = [{"start": 12.5, "end": 19.0, "label": "SYMPTOM"}]

        with self.assertRaises(AssertionError):
            self.span_asserter.assert_boundaries_valid(text, spans)


@pytest.mark.parametrize(
    "record,expected_error",
    [
        # Missing 'text' field
        ({"entities": [], "source": "test"}, KeyError),
        # Missing 'entities' field
        ({"text": "Sample text", "source": "test"}, KeyError),
        # entities is not a list
        (
            {"text": "Sample", "entities": "not_a_list", "source": "test"},
            (TypeError, AssertionError),
        ),
        # Missing 'source' field
        ({"text": "Sample", "entities": []}, AssertionError),
        # Empty 'source' field
        ({"text": "Sample", "entities": [], "source": "", "annotator": "test"}, AssertionError),
        # Missing 'annotator' field
        ({"text": "Sample", "entities": [], "source": "test"}, AssertionError),
    ],
)
def test_malformed_gold_records(record, expected_error):
    """Parametrized test for malformed gold records."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    integrity_validator = IntegrityValidator(test_case)

    with pytest.raises(expected_error):
        integrity_validator.validate_full_record(record)


class TestMalformedJSONL(IntegrationTestBase):
    """Test handling of malformed JSONL files."""

    def test_invalid_json_line(self):
        """Test JSONL with invalid JSON syntax."""
        jsonl_file = self.create_temp_file(
            "invalid.jsonl",
            '{"text": "Valid line", "entities": []}\n'
            "{invalid json syntax here}\n"
            '{"text": "Another valid line", "entities": []}\n',
        )

        valid_lines = []
        invalid_lines = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        valid_lines.append(record)
                    except json.JSONDecodeError:
                        invalid_lines.append(i)

        self.assertEqual(len(valid_lines), 2, "Should parse 2 valid lines")
        self.assertEqual(len(invalid_lines), 1, "Should detect 1 invalid line")

    def test_empty_jsonl_file(self):
        """Test empty JSONL file."""
        jsonl_file = self.create_temp_file("empty.jsonl", "")

        records = self.load_jsonl(jsonl_file)
        self.assertEqual(len(records), 0, "Empty file should yield no records")

    def test_jsonl_with_blank_lines(self):
        """Test JSONL with interspersed blank lines."""
        jsonl_file = self.create_temp_file(
            "blank_lines.jsonl",
            '{"text": "Line 1", "entities": []}\n'
            "\n"
            '{"text": "Line 2", "entities": []}\n'
            "\n\n"
            '{"text": "Line 3", "entities": []}\n',
        )

        records = self.load_jsonl(jsonl_file)
        self.assertEqual(len(records), 3, "Should parse 3 records, ignoring blank lines")


@pytest.mark.parametrize(
    "text,label,should_fail",
    [
        ("", "SYMPTOM", True),  # Empty label text
        ("itching", "", True),  # Empty label
        ("itching", "INVALID", True),  # Invalid label (not in ALLOWED_LABELS)
        ("itching", "SYMPTOM", False),  # Valid
        ("product", "PRODUCT", False),  # Valid
        ("test", "symptom", True),  # Lowercase label (invalid)
    ],
)
def test_span_label_validation(text, label, should_fail):
    """Parametrized span label validation."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    integrity_validator = IntegrityValidator(test_case)

    entities = [{"start": 0, "end": len(text), "label": label, "text": text, "canonical": "test"}]

    if should_fail:
        with pytest.raises(AssertionError):
            integrity_validator.assert_labels_valid(entities)
    else:
        integrity_validator.assert_labels_valid(entities)
