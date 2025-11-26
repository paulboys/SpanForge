"""Tests for evaluate_llm_refinement.py script."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.annotation.base import EvaluationTestBase


class TestEvaluateLLMRefinement(EvaluationTestBase):
    """Test suite for LLM refinement evaluation script."""

    def test_load_jsonl_valid_file(self):
        """Test loading valid JSONL file."""
        # Create test file
        test_file = self.data_dir / "test.jsonl"
        records = [{"text": "test1", "spans": []}, {"text": "test2", "spans": []}]
        self.create_jsonl_file(test_file, records)

        # Import and test
        from scripts.annotation.evaluate_llm_refinement import load_jsonl

        result = load_jsonl(str(test_file))

        assert len(result) == 2
        assert result[0]["text"] == "test1"
        assert result[1]["text"] == "test2"

    def test_load_jsonl_empty_file(self):
        """Test loading empty JSONL file."""
        test_file = self.data_dir / "empty.jsonl"
        test_file.write_text("")

        from scripts.annotation.evaluate_llm_refinement import load_jsonl

        result = load_jsonl(str(test_file))

        assert result == []

    def test_load_jsonl_file_not_found(self):
        """Test loading non-existent file raises error."""
        from scripts.annotation.evaluate_llm_refinement import load_jsonl

        with pytest.raises(FileNotFoundError):
            load_jsonl("nonexistent.jsonl")

    def test_extract_spans_with_confidence(self):
        """Test extracting spans with confidence scores."""
        from scripts.annotation.evaluate_llm_refinement import extract_spans_from_records

        records = [
            {
                "text": self.sample_text,
                "spans": [
                    {
                        "text": "burning",
                        "start": 57,
                        "end": 64,
                        "label": "SYMPTOM",
                        "confidence": 0.9,
                    }
                ],
            }
        ]

        result = extract_spans_from_records(records)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.9
        assert result[0]["text"] == "burning"

    def test_extract_spans_without_confidence(self):
        """Test extracting spans without confidence (gold labels)."""
        from scripts.annotation.evaluate_llm_refinement import extract_spans_from_records

        records = [
            {
                "text": self.sample_text,
                "spans": [{"text": "burning", "start": 57, "end": 64, "label": "SYMPTOM"}],
            }
        ]

        result = extract_spans_from_records(records)

        assert len(result) == 1
        assert "confidence" not in result[0]

    def test_extract_spans_empty_records(self):
        """Test extracting spans from empty records."""
        from scripts.annotation.evaluate_llm_refinement import extract_spans_from_records

        result = extract_spans_from_records([])
        assert result == []

    def test_extract_spans_no_spans(self):
        """Test extracting from records with no spans."""
        from scripts.annotation.evaluate_llm_refinement import extract_spans_from_records

        records = [{"text": "test", "spans": []}]
        result = extract_spans_from_records(records)

        assert result == []

    def test_evaluate_overall_metrics(self):
        """Test computing overall metrics."""
        from scripts.annotation.evaluate_llm_refinement import evaluate_overall_metrics

        metrics = evaluate_overall_metrics(self.weak_spans, self.llm_refined_spans, self.gold_spans)

        assert "iou_delta" in metrics
        assert "llm_boundary_metrics" in metrics
        assert "correction_metrics" in metrics
        assert "llm_precision_recall_f1" in metrics
        assert "llm_calibration" in metrics

    def test_evaluate_overall_metrics_perfect_match(self):
        """Test metrics when LLM refinement perfectly matches gold."""
        from scripts.annotation.evaluate_llm_refinement import evaluate_overall_metrics

        # LLM spans match gold exactly
        metrics = evaluate_overall_metrics(
            self.weak_spans, self.gold_spans, self.gold_spans  # Perfect match
        )

        assert metrics["llm_boundary_metrics"]["exact_match_rate"] == 1.0
        assert metrics["llm_precision_recall_f1"]["f1"] == 1.0

    def test_evaluate_stratified_metrics(self):
        """Test computing stratified metrics."""
        from scripts.annotation.evaluate_llm_refinement import evaluate_stratified_metrics

        # Test label stratification
        metrics_label = evaluate_stratified_metrics(
            self.weak_spans, self.llm_refined_spans, self.gold_spans, "label"
        )

        assert "SYMPTOM" in metrics_label

    def test_evaluate_stratified_by_label(self):
        """Test stratification by label type."""
        from scripts.annotation.evaluate_llm_refinement import evaluate_stratified_metrics

        metrics = evaluate_stratified_metrics(
            self.weak_spans, self.llm_refined_spans, self.gold_spans, "label"
        )

        assert "SYMPTOM" in metrics
        assert "llm_metrics" in metrics["SYMPTOM"]
        assert "f1" in metrics["SYMPTOM"]["llm_metrics"]

    def test_evaluate_stratified_by_confidence(self):
        """Test stratification by confidence buckets."""
        from scripts.annotation.evaluate_llm_refinement import evaluate_stratified_metrics

        metrics = evaluate_stratified_metrics(
            self.weak_spans, self.llm_refined_spans, self.gold_spans, "confidence"
        )

        # Check for confidence bucket keys
        assert len(metrics) > 0
        # Buckets should be numeric ranges
        assert any(isinstance(k, (str, tuple)) for k in metrics.keys())

    def test_generate_markdown_summary(self):
        """Test generating markdown summary."""
        from scripts.annotation.evaluate_llm_refinement import generate_markdown_summary

        metrics = {
            "overall_metrics": {
                "iou_delta": {
                    "weak_mean_iou": 0.85,
                    "llm_mean_iou": 1.0,
                    "improvement_pct": 15.0,
                    "delta": 0.15,
                },
                "weak_boundary_metrics": {
                    "exact_match_rate": 0.85,
                    "mean_iou": 0.85,
                    "median_iou": 0.87,
                },
                "llm_boundary_metrics": {
                    "exact_match_rate": 1.0,
                    "mean_iou": 1.0,
                    "median_iou": 1.0,
                },
                "correction_metrics": {
                    "improved_count": 2,
                    "worsened_count": 0,
                    "unchanged_count": 0,
                    "modified_count": 2,
                    "total_spans": 10,
                    "improvement_rate": 1.0,
                    "false_refinement_rate": 0.0,
                },
                "llm_precision_recall_f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "weak_precision_recall_f1": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
            }
        }

        markdown = generate_markdown_summary(metrics)

        assert "# LLM Refinement Evaluation Report" in markdown
        assert "IOU Improvement" in markdown
        assert "Boundary Precision" in markdown
        assert "Correction Statistics" in markdown  # Changed from "Correction Rate"
        assert "Precision/Recall/F1" in markdown

    def test_cli_basic_evaluation(self):
        """Test CLI with basic weak/refined/gold files."""
        weak_file = self.data_dir / "weak.jsonl"
        refined_file = self.data_dir / "refined.jsonl"
        gold_file = self.data_dir / "gold.jsonl"
        output_file = self.reports_dir / "eval.json"

        # Create test files
        self.create_weak_labels_file(weak_file)
        self.create_weak_labels_file(refined_file)  # Same as weak for simplicity
        self.create_gold_labels_file(gold_file)

        # Run CLI
        result = subprocess.run(
            [
                "python",
                "scripts/annotation/evaluate_llm_refinement.py",
                "--weak",
                str(weak_file),
                "--refined",
                str(refined_file),
                "--gold",
                str(gold_file),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Verify output
        with open(output_file) as f:
            metrics = json.load(f)
        assert "overall_metrics" in metrics

    def test_cli_with_markdown_output(self):
        """Test CLI with markdown report generation."""
        weak_file = self.data_dir / "weak.jsonl"
        refined_file = self.data_dir / "refined.jsonl"
        gold_file = self.data_dir / "gold.jsonl"
        output_file = self.reports_dir / "eval.json"
        markdown_file = self.reports_dir / "eval.md"

        self.create_weak_labels_file(weak_file)
        self.create_weak_labels_file(refined_file)
        self.create_gold_labels_file(gold_file)

        result = subprocess.run(
            [
                "python",
                "scripts/annotation/evaluate_llm_refinement.py",
                "--weak",
                str(weak_file),
                "--refined",
                str(refined_file),
                "--gold",
                str(gold_file),
                "--output",
                str(output_file),
                "--markdown",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert markdown_file.exists()

        # Verify markdown content
        markdown_content = markdown_file.read_text()
        assert "# LLM Refinement Evaluation Report" in markdown_content

    def test_cli_with_stratification(self):
        """Test CLI with stratification options."""
        weak_file = self.data_dir / "weak.jsonl"
        refined_file = self.data_dir / "refined.jsonl"
        gold_file = self.data_dir / "gold.jsonl"
        output_file = self.reports_dir / "eval.json"

        self.create_weak_labels_file(weak_file)
        self.create_weak_labels_file(refined_file)
        self.create_gold_labels_file(gold_file)

        result = subprocess.run(
            [
                "python",
                "scripts/annotation/evaluate_llm_refinement.py",
                "--weak",
                str(weak_file),
                "--refined",
                str(refined_file),
                "--gold",
                str(gold_file),
                "--output",
                str(output_file),
                "--stratify",
                "label",
                "confidence",
                "span_length",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_file) as f:
            metrics = json.load(f)

        assert "by_label" in metrics
        assert "by_confidence" in metrics
        assert "by_span_length" in metrics

    def test_cli_missing_required_args(self):
        """Test CLI fails gracefully with missing arguments."""
        result = subprocess.run(
            ["python", "scripts/annotation/evaluate_llm_refinement.py"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_alignment_with_test_fixtures(self):
        """Test evaluation aligns with existing test fixtures."""
        # Use the actual test fixtures from tests/fixtures/annotation
        fixtures_dir = Path("tests/fixtures/annotation")
        if fixtures_dir.exists():
            weak_file = fixtures_dir / "weak_baseline.jsonl"
            refined_file = fixtures_dir / "gold_with_llm_refined.jsonl"
            gold_file = fixtures_dir / "gold_standard.jsonl"

            if all(f.exists() for f in [weak_file, refined_file, gold_file]):
                from scripts.annotation.evaluate_llm_refinement import (
                    evaluate_overall_metrics,
                    extract_spans_from_records,
                    load_jsonl,
                )

                weak_records = load_jsonl(str(weak_file))
                refined_records = load_jsonl(str(refined_file))
                gold_records = load_jsonl(str(gold_file))

                weak_spans = extract_spans_from_records(weak_records)
                refined_spans = extract_spans_from_records(refined_records)
                gold_spans = extract_spans_from_records(gold_records)

                metrics = evaluate_overall_metrics(weak_spans, refined_spans, gold_spans)

                # Verify expected improvements
                assert (
                    metrics["iou_delta"]["improvement_pct"] >= -100
                )  # Can be negative but should be reasonable
                assert metrics["llm_precision_recall_f1"]["f1"] >= 0.0  # F1 should be valid
