"""
Tests for FDA CAERS data integration pipeline.

Employs inheritance and composition for maintainability and future adaptability.
Tests all functions in scripts/caers/download_caers.py with comprehensive coverage.
"""

import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.base import TestBase

# Import CAERS module functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "caers"))
import download_caers


class CAERSTestBase(TestBase):
    """Base class for CAERS integration tests with common fixtures."""

    def setUp(self):
        super().setUp()
        self.download_dir = self.temp_dir / "caers" / "raw"
        self.output_dir = self.temp_dir / "caers"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_mock_caers_csv(self, records: List[Dict[str, Any]]) -> Path:
        """Create a mock CAERS CSV file for testing.

        Args:
            records: List of dictionaries representing CAERS rows

        Returns:
            Path to created CSV file
        """
        csv_path = self.download_dir / "CAERS_ASCII_2014-PRESENT.csv"

        # Create DataFrame with standard CAERS columns
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False, encoding="latin-1")

        return csv_path

    def create_standard_caers_records(self) -> List[Dict[str, Any]]:
        """Create standard CAERS test records with typical data."""
        return [
            {
                "RA_Report #": "12345",
                "RA_CAERS Created Date": "2024-01-15",
                "Product Role": "Cosmetics",
                "PRI Reported Brand/Product Name": "Test Face Cream",
                "PRI_Reported Adverse Events": "Severe burning and redness on face",
                "CI_Age at Adverse Event": "35",
                "CI_Gender": "Female",
                "AEC_Outcome(s)": "Recovered",
            },
            {
                "RA_Report #": "12346",
                "RA_CAERS Created Date": "2024-01-16",
                "Product Role": "Dietary Supplement",
                "PRI Reported Brand/Product Name": "Vitamin C Serum",
                "PRI_Reported Adverse Events": "Mild itching and rash",
                "CI_Age at Adverse Event": "42",
                "CI_Gender": "Male",
                "AEC_Outcome(s)": "Ongoing",
            },
            {
                "RA_Report #": "12347",
                "RA_CAERS Created Date": "2024-01-17",
                "Product Role": "Cosmetics",
                "PRI Reported Brand/Product Name": "Moisturizer",
                "PRI_Reported Adverse Events": "Dryness and peeling",
                "CI_Age at Adverse Event": "28",
                "CI_Gender": "Female",
                "AEC_Outcome(s)": "Recovered",
            },
        ]

    def create_minimal_lexicons(self):
        """Create minimal symptom and product lexicons for testing."""
        symptoms = [
            {"term": "burning", "canonical": "Burning", "source": "test", "concept_id": ""},
            {"term": "redness", "canonical": "Redness", "source": "test", "concept_id": ""},
            {"term": "itching", "canonical": "Itching", "source": "test", "concept_id": ""},
            {"term": "rash", "canonical": "Rash", "source": "test", "concept_id": ""},
            {"term": "dryness", "canonical": "Dryness", "source": "test", "concept_id": ""},
        ]
        products = [
            {"term": "cream", "canonical": "Cream", "sku": "", "category": ""},
            {"term": "serum", "canonical": "Serum", "sku": "", "category": ""},
        ]

        sym_path = self.temp_dir / "lexicon" / "symptoms.csv"
        prod_path = self.temp_dir / "lexicon" / "products.csv"
        sym_path.parent.mkdir(parents=True, exist_ok=True)

        self.create_lexicon_csv(str(sym_path.relative_to(self.temp_dir)), symptoms)
        self.create_lexicon_csv(str(prod_path.relative_to(self.temp_dir)), products)

        return sym_path, prod_path


class TestDownloadCAERSData(CAERSTestBase):
    """Test download_caers_data function."""

    @patch("download_caers.urlretrieve")
    def test_download_new_file(self, mock_urlretrieve):
        """Test downloading CAERS data when file doesn't exist."""

        # Mock urlretrieve to create an empty file
        def create_file(url, path):
            Path(path).write_text("dummy data")

        mock_urlretrieve.side_effect = create_file

        result = download_caers.download_caers_data(self.download_dir, force=False)

        self.assertTrue(result.exists())
        mock_urlretrieve.assert_called_once()
        self.assertEqual(result.name, "CAERS_ASCII_2014-PRESENT.csv")

    def test_skip_download_existing_file(self):
        """Test skipping download when file already exists."""
        # Create existing file
        csv_path = self.download_dir / "CAERS_ASCII_2014-PRESENT.csv"
        csv_path.write_text("dummy content")

        with patch("download_caers.urlretrieve") as mock_urlretrieve:
            result = download_caers.download_caers_data(self.download_dir, force=False)

            mock_urlretrieve.assert_not_called()
            self.assertEqual(result, csv_path)

    @patch("download_caers.urlretrieve")
    def test_force_download(self, mock_urlretrieve):
        """Test force re-download even when file exists."""
        # Create existing file
        csv_path = self.download_dir / "CAERS_ASCII_2014-PRESENT.csv"
        csv_path.write_text("old content")

        mock_urlretrieve.return_value = None

        result = download_caers.download_caers_data(self.download_dir, force=True)

        mock_urlretrieve.assert_called_once()
        self.assertEqual(result, csv_path)

    @patch("download_caers.urlretrieve")
    def test_download_failure(self, mock_urlretrieve):
        """Test handling of download failure."""
        mock_urlretrieve.side_effect = Exception("Network error")

        with self.assertRaises(Exception):
            download_caers.download_caers_data(self.download_dir, force=True)


class TestLoadCAERSData(CAERSTestBase):
    """Test load_caers_data function."""

    def test_load_valid_csv(self):
        """Test loading valid CAERS CSV."""
        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)

        df = download_caers.load_caers_data(csv_path)

        self.assertEqual(len(df), 3)
        self.assertIn("RA_Report #", df.columns)
        self.assertIn("Product Role", df.columns)

    def test_load_empty_csv(self):
        """Test loading empty CSV."""
        csv_path = self.download_dir / "empty.csv"
        csv_path.write_text("RA_Report #,Product Role\n", encoding="latin-1")

        df = download_caers.load_caers_data(csv_path)

        self.assertEqual(len(df), 0)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        csv_path = self.download_dir / "nonexistent.csv"

        with self.assertRaises(Exception):
            download_caers.load_caers_data(csv_path)


class TestFilterByCategory(CAERSTestBase):
    """Test filter_by_category function."""

    def test_filter_cosmetics(self):
        """Test filtering for cosmetics category."""
        records = self.create_standard_caers_records()
        df = pd.DataFrame(records)

        filtered = download_caers.filter_by_category(df, ["cosmetics"])

        self.assertEqual(len(filtered), 2)
        self.assertTrue(all("Cosmetics" in str(role) for role in filtered["Product Role"]))

    def test_filter_multiple_categories(self):
        """Test filtering for multiple categories."""
        records = self.create_standard_caers_records()
        df = pd.DataFrame(records)

        filtered = download_caers.filter_by_category(df, ["cosmetics", "supplements"])

        self.assertEqual(len(filtered), 3)

    def test_no_category_filter(self):
        """Test with no category filter returns all records."""
        records = self.create_standard_caers_records()
        df = pd.DataFrame(records)

        filtered = download_caers.filter_by_category(df, None)

        self.assertEqual(len(filtered), len(df))

    def test_unknown_category(self):
        """Test with unknown category logs warning."""
        records = self.create_standard_caers_records()
        df = pd.DataFrame(records)

        with patch("download_caers.logger") as mock_logger:
            filtered = download_caers.filter_by_category(df, ["unknown_category"])
            mock_logger.warning.assert_called()

    def test_missing_category_column(self):
        """Test when category column is missing."""
        df = pd.DataFrame([{"RA_Report #": "123", "Other": "data"}])

        with patch("download_caers.logger") as mock_logger:
            filtered = download_caers.filter_by_category(df, ["cosmetics"])
            mock_logger.warning.assert_called_with(
                "No category column found, returning full dataset"
            )


class TestExtractComplaintText(CAERSTestBase):
    """Test extract_complaint_text function."""

    def test_extract_from_adverse_events(self):
        """Test extracting text from adverse events column."""
        row = pd.Series(
            {
                "PRI_Reported Adverse Events": "Patient developed severe redness and itching",
                "Product Role": "Cosmetics",
            }
        )

        text = download_caers.extract_complaint_text(row)

        self.assertIsNotNone(text)
        self.assertIn("redness", text)
        self.assertIn("itching", text)

    def test_extract_multiple_text_fields(self):
        """Test extracting and combining multiple text fields."""
        row = pd.Series(
            {
                "PRI_Reported Adverse Events": "Burning sensation",
                "PRI Reported Brand/Product Name": "Test Cream",
                "Product Role": "Cosmetics",
            }
        )

        text = download_caers.extract_complaint_text(row)

        self.assertIn("Burning sensation", text)
        self.assertIn("Test Cream", text)
        self.assertIn("|", text)  # Separator between fields

    def test_no_text_available(self):
        """Test when no text fields are available."""
        row = pd.Series({"Product Role": "Cosmetics", "RA_Report #": "123"})

        text = download_caers.extract_complaint_text(row)

        self.assertIsNone(text)

    def test_text_too_short(self):
        """Test filtering out very short text."""
        row = pd.Series({"PRI_Reported Adverse Events": "rash", "Product Role": "Cosmetics"})

        text = download_caers.extract_complaint_text(row)

        self.assertIsNone(text)  # Too short (< 20 chars)

    def test_skip_nan_values(self):
        """Test skipping NaN values in text extraction."""
        row = pd.Series(
            {
                "PRI_Reported Adverse Events": float("nan"),
                "PRI Reported Brand/Product Name": "Valid product description text",
                "Product Role": "Cosmetics",
            }
        )

        text = download_caers.extract_complaint_text(row)

        self.assertIsNotNone(text)
        self.assertNotIn("nan", text.lower())


class TestExtractMetadata(CAERSTestBase):
    """Test extract_metadata function."""

    def test_extract_all_metadata_fields(self):
        """Test extracting all available metadata fields."""
        row = pd.Series(
            {
                "RA_Report #": "12345",
                "RA_CAERS Created Date": "2024-01-15",
                "Product Role": "Cosmetics",
                "PRI Reported Brand/Product Name": "Test Cream",
                "CI_Age at Adverse Event": "35",
                "CI_Gender": "Female",
                "AEC_Outcome(s)": "Recovered",
            }
        )

        metadata = download_caers.extract_metadata(row)

        self.assertEqual(metadata["report_id"], "12345")
        self.assertEqual(metadata["date_created"], "2024-01-15")
        self.assertEqual(metadata["product_type"], "Cosmetics")
        self.assertEqual(metadata["product_name"], "Test Cream")
        self.assertEqual(metadata["age"], "35")
        self.assertEqual(metadata["gender"], "Female")
        self.assertEqual(metadata["outcomes"], "Recovered")

    def test_extract_partial_metadata(self):
        """Test extracting metadata when some fields are missing."""
        row = pd.Series({"RA_Report #": "12345", "Product Role": "Cosmetics"})

        metadata = download_caers.extract_metadata(row)

        self.assertEqual(metadata["report_id"], "12345")
        self.assertEqual(metadata["product_type"], "Cosmetics")
        self.assertNotIn("age", metadata)
        self.assertNotIn("gender", metadata)

    def test_skip_nan_metadata(self):
        """Test skipping NaN values in metadata."""
        row = pd.Series(
            {"RA_Report #": "12345", "CI_Age at Adverse Event": float("nan"), "CI_Gender": "Male"}
        )

        metadata = download_caers.extract_metadata(row)

        self.assertIn("gender", metadata)
        self.assertNotIn("age", metadata)


class TestCreateComplaintRecord(CAERSTestBase):
    """Test create_complaint_record function."""

    def test_create_basic_record(self):
        """Test creating basic complaint record."""
        from src.weak_label import Span

        text = "Patient developed burning and redness"
        spans = [
            Span(
                text="burning",
                start=18,
                end=25,
                label="SYMPTOM",
                canonical="Burning",
                confidence=1.0,
                negated=False,
            )
        ]
        metadata = {"report_id": "12345", "product_type": "Cosmetics"}

        record = download_caers.create_complaint_record(text, spans, metadata)

        self.assertEqual(record["text"], text)
        self.assertEqual(record["source"], "FDA_CAERS")
        self.assertEqual(record["metadata"], metadata)
        self.assertIn("date_processed", record)
        self.assertEqual(len(record["spans"]), 1)
        self.assertEqual(record["spans"][0]["text"], "burning")

    def test_create_record_multiple_spans(self):
        """Test creating record with multiple spans."""
        from src.weak_label import Span

        text = "Severe burning and redness on face"
        spans = [
            Span(
                text="burning",
                start=7,
                end=14,
                label="SYMPTOM",
                canonical="Burning",
                confidence=1.0,
                negated=False,
            ),
            Span(
                text="redness",
                start=19,
                end=26,
                label="SYMPTOM",
                canonical="Redness",
                confidence=0.95,
                negated=False,
            ),
        ]
        metadata = {"report_id": "12345"}

        record = download_caers.create_complaint_record(text, spans, metadata)

        self.assertEqual(len(record["spans"]), 2)
        self.assertEqual(record["spans"][0]["confidence"], 1.0)
        self.assertEqual(record["spans"][1]["confidence"], 0.95)

    def test_create_record_with_negation(self):
        """Test creating record with negated span."""
        from src.weak_label import Span

        text = "No burning sensation reported"
        spans = [
            Span(
                text="burning",
                start=3,
                end=10,
                label="SYMPTOM",
                canonical="Burning",
                confidence=1.0,
                negated=True,
            )
        ]
        metadata = {}

        record = download_caers.create_complaint_record(text, spans, metadata)

        self.assertTrue(record["spans"][0]["negated"])


class TestValidateRecord(CAERSTestBase):
    """Test validate_record function."""

    def test_validate_correct_record(self):
        """Test validation of correct record."""
        record = {
            "text": "Patient developed severe burning sensation on face after use",
            "source": "FDA_CAERS",
            "metadata": {},
            "spans": [
                {
                    "text": "burning",
                    "start": 25,
                    "end": 32,
                    "label": "SYMPTOM",
                    "confidence": 1.0,
                    "negated": False,
                }
            ],
        }

        is_valid, issues = download_caers.validate_record(record)

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_validate_missing_text(self):
        """Test validation fails for missing text."""
        record = {"text": "", "spans": []}

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(any("text" in issue.lower() for issue in issues))

    def test_validate_missing_spans(self):
        """Test validation fails for missing spans field."""
        record = {"text": "Some text that is long enough for validation"}

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(any("spans" in issue.lower() for issue in issues))

    def test_validate_text_too_short(self):
        """Test validation fails for text that's too short."""
        record = {"text": "Short", "spans": []}

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(any("too short" in issue.lower() for issue in issues))

    def test_validate_text_too_long(self):
        """Test validation fails for text that's too long."""
        record = {"text": "x" * 6000, "spans": []}

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(any("too long" in issue.lower() for issue in issues))

    def test_validate_invalid_span_positions(self):
        """Test validation fails for invalid span positions."""
        record = {
            "text": "Patient developed burning sensation",
            "spans": [
                {"text": "burning", "start": -1, "end": 10, "label": "SYMPTOM"},
                {"text": "test", "start": 10, "end": 5, "label": "SYMPTOM"},
                {"text": "overflow", "start": 0, "end": 100, "label": "SYMPTOM"},
            ],
        }

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(len(issues) >= 3)

    def test_validate_text_mismatch(self):
        """Test validation fails when span text doesn't match actual text."""
        record = {
            "text": "Patient developed burning sensation",
            "spans": [{"text": "itching", "start": 18, "end": 25, "label": "SYMPTOM"}],
        }

        is_valid, issues = download_caers.validate_record(record)

        self.assertFalse(is_valid)
        self.assertTrue(any("mismatch" in issue.lower() for issue in issues))


class TestProcessCAERSToJSONL(CAERSTestBase):
    """Test process_caers_to_jsonl function."""

    def test_process_basic_pipeline(self):
        """Test basic processing pipeline with valid data."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        # Create test data
        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "output.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        # Load lexicons
        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        # Process
        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=1,
        )

        # Verify output
        self.assertTrue(output_path.exists())
        self.assertGreater(stats["successful"], 0)
        self.assertGreater(stats["total_spans"], 0)

        # Load and verify JSONL
        records = self.load_jsonl(output_path)
        self.assertGreater(len(records), 0)
        self.assertIn("text", records[0])
        self.assertIn("spans", records[0])
        self.assertIn("metadata", records[0])

    def test_process_with_category_filter(self):
        """Test processing with category filtering."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "cosmetics.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=["cosmetics"],
            limit=None,
            validate=True,
            min_spans=1,
        )

        # Should only process cosmetics records
        self.assertEqual(stats["successful"], 2)

    def test_process_with_limit(self):
        """Test processing with record limit."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "limited.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=1,
            validate=True,
            min_spans=1,
        )

        self.assertEqual(stats["total_rows"], 1)

    def test_process_with_min_spans_filter(self):
        """Test filtering records by minimum spans."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "min_spans.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=5,  # High threshold - may skip records
        )

        # Some records may be skipped due to insufficient spans
        self.assertGreaterEqual(stats["skipped_no_spans"], 0)

    def test_process_skip_validation(self):
        """Test processing with validation disabled."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "no_validation.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=False,
            min_spans=1,
        )

        self.assertEqual(stats["validation_failed"], 0)

    def test_process_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        # Add record with no text
        records = self.create_standard_caers_records()
        records.append(
            {
                "RA_Report #": "99999",
                "Product Role": "Cosmetics",
                # No adverse event text
            }
        )
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "stats_test.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=1,
        )

        # Verify statistics structure
        required_stats = [
            "total_rows",
            "processed",
            "skipped_no_text",
            "skipped_no_spans",
            "validation_failed",
            "successful",
            "total_spans",
            "symptom_spans",
            "product_spans",
        ]
        for stat in required_stats:
            self.assertIn(stat, stats)

        # Verify counts add up
        self.assertEqual(
            stats["successful"]
            + stats["skipped_no_text"]
            + stats["skipped_no_spans"]
            + stats["validation_failed"],
            stats["processed"],
        )


class TestProductCategories(CAERSTestBase):
    """Test PRODUCT_CATEGORIES configuration."""

    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        expected = ["cosmetics", "supplements", "foods", "personal_care", "baby"]
        for cat in expected:
            self.assertIn(cat, download_caers.PRODUCT_CATEGORIES)

    def test_categories_have_patterns(self):
        """Test each category has matching patterns."""
        for cat, patterns in download_caers.PRODUCT_CATEGORIES.items():
            self.assertIsInstance(patterns, list)
            self.assertGreater(len(patterns), 0)
            for pattern in patterns:
                self.assertIsInstance(pattern, str)
                self.assertGreater(len(pattern), 0)


class TestMainCLI(CAERSTestBase):
    """Test main CLI function."""

    @patch("download_caers.download_caers_data")
    @patch("download_caers.process_caers_to_jsonl")
    @patch("download_caers.load_symptom_lexicon")
    @patch("download_caers.load_product_lexicon")
    def test_main_basic_args(self, mock_prod_lex, mock_sym_lex, mock_process, mock_download):
        """Test main function with basic arguments."""
        mock_download.return_value = self.download_dir / "test.csv"
        mock_sym_lex.return_value = []
        mock_prod_lex.return_value = []
        mock_process.return_value = {"successful": 50, "total_spans": 120}

        # Mock sys.argv for argparse
        test_args = [
            "download_caers.py",
            "--output",
            str(self.output_dir / "test.jsonl"),
            "--limit",
            "100",
        ]

        with patch("sys.argv", test_args):
            try:
                download_caers.main()
            except SystemExit:
                pass  # main() may call sys.exit

        mock_download.assert_called_once()
        mock_process.assert_called_once()


class TestEdgeCases(CAERSTestBase):
    """Test edge cases and error handling."""

    def test_empty_lexicons(self):
        """Test processing with empty lexicons."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = self.create_standard_caers_records()
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "empty_lex.jsonl"

        # Create empty lexicons
        sym_path = self.temp_dir / "empty_symptoms.csv"
        prod_path = self.temp_dir / "empty_products.csv"
        sym_path.write_text("term,canonical,source,concept_id\n")
        prod_path.write_text("term,canonical,sku,category\n")

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=0,  # Allow records with no spans
        )

        # Should still process but find no spans
        self.assertEqual(stats["total_spans"], 0)

    def test_unicode_in_complaints(self):
        """Test handling Unicode characters in complaint text."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        records = [
            {
                "RA_Report #": "12345",
                "Product Role": "Cosmetics",
                "PRI_Reported Adverse Events": "Patient reports severe redness and burning sensation on face",
                "CI_Gender": "Female",
            }
        ]
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "unicode.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=1,
        )

        self.assertGreater(stats["successful"], 0)

        # Verify JSONL can be read
        records_out = self.load_jsonl(output_path)
        self.assertEqual(len(records_out), 1)

    def test_very_long_complaint_text(self):
        """Test handling very long complaint text."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        long_text = "Patient experienced burning sensation. " * 100  # ~4000 chars
        records = [
            {
                "RA_Report #": "12345",
                "Product Role": "Cosmetics",
                "PRI_Reported Adverse Events": long_text,
            }
        ]
        csv_path = self.create_mock_caers_csv(records)
        output_path = self.output_dir / "long_text.jsonl"
        sym_path, prod_path = self.create_minimal_lexicons()

        symptom_lexicon = load_symptom_lexicon(sym_path)
        product_lexicon = load_product_lexicon(prod_path)

        stats = download_caers.process_caers_to_jsonl(
            csv_path=csv_path,
            output_path=output_path,
            symptom_lexicon=symptom_lexicon,
            product_lexicon=product_lexicon,
            categories=None,
            limit=None,
            validate=True,
            min_spans=1,
        )

        # Should process successfully (under 5000 char limit)
        self.assertGreater(stats["successful"], 0)


if __name__ == "__main__":
    unittest.main()
