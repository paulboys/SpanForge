"""Tests for src.knowledge_retrieval module (lexicon knowledge retrieval)."""

import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.knowledge_retrieval import _load_csv, build_index, context_for_spans
from tests.base import TestBase


class KnowledgeRetrievalTestBase(TestBase):
    """Base class for knowledge retrieval tests with common fixtures."""

    def setUp(self):
        super().setUp()
        # Create test lexicon CSVs
        self.symptom_csv = self.temp_dir / "symptoms.csv"
        self.product_csv = self.temp_dir / "products.csv"

    def create_symptom_csv(self, entries: List[Dict[str, str]]) -> Path:
        """Create symptom CSV with given entries."""
        lines = ["term,canonical,source,concept_id"]
        for entry in entries:
            term = entry.get("term", "")
            canonical = entry.get("canonical", term)
            source = entry.get("source", "")
            concept_id = entry.get("concept_id", "")
            lines.append(f"{term},{canonical},{source},{concept_id}")
        self.symptom_csv.write_text("\n".join(lines), encoding="utf-8")
        return self.symptom_csv

    def create_product_csv(self, entries: List[Dict[str, str]]) -> Path:
        """Create product CSV with given entries."""
        lines = ["term,sku,category"]
        for entry in entries:
            term = entry.get("term", "")
            sku = entry.get("sku", "")
            category = entry.get("category", "")
            lines.append(f"{term},{sku},{category}")
        self.product_csv.write_text("\n".join(lines), encoding="utf-8")
        return self.product_csv


class TestLoadCSV(KnowledgeRetrievalTestBase):
    """Tests for _load_csv function."""

    def test_load_csv_with_valid_file(self):
        """Test loading valid CSV file."""
        csv_path = self.temp_dir / "test.csv"
        csv_path.write_text(
            "term,canonical\n" "headache,Headache\n" "rash,Rash\n", encoding="utf-8"
        )

        rows = _load_csv(csv_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["term"], "headache")
        self.assertEqual(rows[1]["term"], "rash")

    def test_load_csv_nonexistent_file(self):
        """Test loading nonexistent file returns empty list."""
        csv_path = self.temp_dir / "nonexistent.csv"

        rows = _load_csv(csv_path)

        self.assertEqual(rows, [])

    def test_load_csv_empty_file(self):
        """Test loading empty CSV file."""
        csv_path = self.temp_dir / "empty.csv"
        csv_path.write_text("term,canonical\n", encoding="utf-8")

        rows = _load_csv(csv_path)

        self.assertEqual(rows, [])

    def test_load_csv_filters_empty_terms(self):
        """Test that rows with empty terms are filtered out."""
        csv_path = self.temp_dir / "test.csv"
        csv_path.write_text(
            "term,canonical\n" "headache,Headache\n" ",Empty\n" "rash,Rash\n", encoding="utf-8"
        )

        rows = _load_csv(csv_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["term"], "headache")
        self.assertEqual(rows[1]["term"], "rash")

    def test_load_csv_with_unicode(self):
        """Test loading CSV with unicode characters."""
        csv_path = self.temp_dir / "unicode.csv"
        csv_path.write_text("term,canonical\nröte,Redness\nübelkeit,Nausea\n", encoding="utf-8")

        rows = _load_csv(csv_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["term"], "röte")
        self.assertEqual(rows[1]["term"], "übelkeit")


class TestBuildIndex(KnowledgeRetrievalTestBase):
    """Tests for build_index function."""

    def test_build_index_from_symptoms(self):
        """Test building index from symptom lexicon."""
        self.create_symptom_csv(
            [
                {"term": "Headache", "canonical": "Headache"},
                {"term": "headache", "canonical": "Headache"},
                {"term": "Rash", "canonical": "Rash"},
            ]
        )
        self.create_product_csv([])

        index = build_index(self.symptom_csv, self.product_csv)

        # Check lowercase keys
        self.assertIn("headache", index)
        self.assertIn("rash", index)
        self.assertEqual(index["headache"]["canonical"], "Headache")
        # Check examples include both variations
        self.assertIn("Headache", index["headache"]["examples"])
        self.assertIn("headache", index["headache"]["examples"])

    def test_build_index_from_products(self):
        """Test building index from product lexicon."""
        self.create_symptom_csv([])
        self.create_product_csv(
            [
                {"term": "Aspirin", "sku": "ASP-001"},
                {"term": "Cream", "sku": "CRM-001"},
            ]
        )

        index = build_index(self.symptom_csv, self.product_csv)

        self.assertIn("aspirin", index)
        self.assertIn("cream", index)
        # Canonical defaults to lowercase term when not explicitly provided
        self.assertEqual(index["aspirin"]["canonical"], "aspirin")

    def test_build_index_combined(self):
        """Test building index from both symptom and product lexicons."""
        self.create_symptom_csv([{"term": "Headache", "canonical": "Headache"}])
        self.create_product_csv([{"term": "Aspirin", "sku": "ASP-001"}])

        index = build_index(self.symptom_csv, self.product_csv)

        self.assertIn("headache", index)
        self.assertIn("aspirin", index)

    def test_build_index_with_duplicates(self):
        """Test that duplicate terms are merged with all examples."""
        self.create_symptom_csv(
            [
                {"term": "Headache", "canonical": "Headache"},
                {"term": "headache", "canonical": "Headache"},
                {"term": "Head ache", "canonical": "Headache"},
            ]
        )
        self.create_product_csv([])

        index = build_index(self.symptom_csv, self.product_csv)

        # All variations should be in examples
        self.assertIn("headache", index)
        examples = index["headache"]["examples"]
        self.assertGreaterEqual(len(examples), 2)  # At least 2 variations

    def test_build_index_missing_canonical(self):
        """Test that missing canonical field defaults to term."""
        csv_path = self.temp_dir / "no_canonical.csv"
        csv_path.write_text("term\nheadache\nrash\n", encoding="utf-8")

        index = build_index(csv_path, self.product_csv)

        self.assertIn("headache", index)
        self.assertEqual(index["headache"]["canonical"], "headache")

    def test_build_index_empty_lexicons(self):
        """Test building index from empty lexicons."""
        self.create_symptom_csv([])
        self.create_product_csv([])

        index = build_index(self.symptom_csv, self.product_csv)

        self.assertEqual(index, {})

    def test_build_index_examples_sorted(self):
        """Test that examples are sorted alphabetically."""
        self.create_symptom_csv(
            [
                {"term": "Zheadache", "canonical": "Headache"},
                {"term": "Aheadache", "canonical": "Headache"},
                {"term": "Mheadache", "canonical": "Headache"},
            ]
        )
        self.create_product_csv([])

        index = build_index(self.symptom_csv, self.product_csv)

        # Check first entry (all variations map to same key)
        key = list(index.keys())[0]
        examples = index[key]["examples"]
        self.assertEqual(examples, sorted(examples))


class TestContextForSpans(KnowledgeRetrievalTestBase):
    """Tests for context_for_spans function."""

    def test_context_for_spans_with_matches(self):
        """Test retrieving context for matched spans."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache", "Headache"]},
            "rash": {"canonical": "Rash", "examples": ["rash", "Rash"]},
        }

        spans = [
            {"text": "headache", "label": "SYMPTOM"},
            {"text": "rash", "label": "SYMPTOM"},
        ]

        context = context_for_spans(spans, index)

        self.assertIn("entries", context)
        self.assertIn("headache", context["entries"])
        self.assertIn("rash", context["entries"])
        self.assertEqual(context["entries"]["headache"]["canonical"], "Headache")

    def test_context_for_spans_case_insensitive(self):
        """Test that context lookup is case-insensitive."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache", "Headache"]},
        }

        spans = [
            {"text": "HEADACHE", "label": "SYMPTOM"},
            {"text": "HeadAche", "label": "SYMPTOM"},
        ]

        context = context_for_spans(spans, index)

        # Both should match lowercase key
        self.assertIn("headache", context["entries"])
        self.assertEqual(len(context["entries"]), 1)

    def test_context_for_spans_no_matches(self):
        """Test context for spans with no matches in index."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache"]},
        }

        spans = [
            {"text": "unknown", "label": "SYMPTOM"},
            {"text": "mystery", "label": "SYMPTOM"},
        ]

        context = context_for_spans(spans, index)

        self.assertIn("entries", context)
        self.assertEqual(context["entries"], {})

    def test_context_for_spans_empty_spans(self):
        """Test context for empty spans list."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache"]},
        }

        spans: List[Dict[str, Any]] = []

        context = context_for_spans(spans, index)

        self.assertIn("entries", context)
        self.assertEqual(context["entries"], {})

    def test_context_for_spans_missing_text_field(self):
        """Test handling of spans with missing text field."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache"]},
        }

        spans = [
            {"label": "SYMPTOM"},  # No text field
            {"text": "headache", "label": "SYMPTOM"},
        ]

        context = context_for_spans(spans, index)

        # Should only match the span with text field
        self.assertIn("headache", context["entries"])
        self.assertEqual(len(context["entries"]), 1)

    def test_context_for_spans_partial_matches(self):
        """Test context with some matching and some non-matching spans."""
        index = {
            "headache": {"canonical": "Headache", "examples": ["headache"]},
            "rash": {"canonical": "Rash", "examples": ["rash"]},
        }

        spans = [
            {"text": "headache", "label": "SYMPTOM"},
            {"text": "unknown", "label": "SYMPTOM"},
            {"text": "rash", "label": "SYMPTOM"},
        ]

        context = context_for_spans(spans, index)

        self.assertIn("headache", context["entries"])
        self.assertIn("rash", context["entries"])
        self.assertNotIn("unknown", context["entries"])
        self.assertEqual(len(context["entries"]), 2)


class TestKnowledgeRetrievalIntegration(KnowledgeRetrievalTestBase):
    """Integration tests for knowledge retrieval workflow."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: load CSVs, build index, retrieve context."""
        # Create lexicons
        self.create_symptom_csv(
            [
                {"term": "Headache", "canonical": "Headache", "source": "MedDRA"},
                {"term": "headache", "canonical": "Headache", "source": "MedDRA"},
                {"term": "Rash", "canonical": "Rash", "source": "MedDRA"},
            ]
        )
        self.create_product_csv(
            [
                {"term": "Aspirin", "sku": "ASP-001", "category": "medication"},
            ]
        )

        # Build index
        index = build_index(self.symptom_csv, self.product_csv)

        # Simulate spans from weak labeling
        spans = [
            {"text": "Headache", "label": "SYMPTOM", "start": 0, "end": 8},
            {"text": "aspirin", "label": "PRODUCT", "start": 20, "end": 27},
        ]

        # Get context
        context = context_for_spans(spans, index)

        # Verify complete workflow
        self.assertIn("entries", context)
        self.assertIn("headache", context["entries"])
        self.assertIn("aspirin", context["entries"])
        self.assertEqual(context["entries"]["headache"]["canonical"], "Headache")
        self.assertIn("Headache", context["entries"]["headache"]["examples"])

    def test_workflow_with_large_lexicon(self):
        """Test workflow with larger lexicon."""
        # Create larger symptom lexicon
        entries = []
        for i in range(100):
            entries.append(
                {
                    "term": f"symptom{i}",
                    "canonical": f"Symptom{i}",
                    "source": "test",
                }
            )
        self.create_symptom_csv(entries)
        self.create_product_csv([])

        # Build index
        index = build_index(self.symptom_csv, self.product_csv)

        # Should have 100 entries
        self.assertEqual(len(index), 100)

        # Test retrieval
        spans = [{"text": "symptom50", "label": "SYMPTOM"}]
        context = context_for_spans(spans, index)

        self.assertIn("symptom50", context["entries"])


if __name__ == "__main__":
    unittest.main()
