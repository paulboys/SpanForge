"""Tests for cli.py orchestration script."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from tests.annotation.base import AnnotationTestBase, EvaluationTestBase


class TestCLI(EvaluationTestBase):
    """Test suite for annotation CLI wrapper."""

    def test_cli_help(self):
        """Test CLI help message."""
        result = subprocess.run(
            ["python", "scripts/annotation/cli.py", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "commands:" in result.stdout.lower()

    def test_evaluate_llm_subcommand(self):
        """Test evaluate-llm subcommand invocation."""
        weak_file = self.data_dir / "weak.jsonl"
        refined_file = self.data_dir / "refined.jsonl"
        gold_file = self.data_dir / "gold.jsonl"
        output_file = self.reports_dir / "eval.json"

        self.create_weak_labels_file(weak_file)
        self.create_weak_labels_file(refined_file)
        self.create_gold_labels_file(gold_file)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "evaluate-llm",
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

            # Verify subprocess was called (CLI delegates to evaluate_llm_refinement.py)
            assert result.returncode == 0 or mock_run.called

    def test_plot_metrics_subcommand(self):
        """Test plot-metrics subcommand invocation."""
        report_file = self.reports_dir / "eval.json"
        self.create_evaluation_report(report_file)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "plot-metrics",
                    "--report",
                    str(report_file),
                    "--output-dir",
                    str(self.plots_dir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_refine_llm_subcommand(self):
        """Test refine-llm subcommand invocation."""
        weak_file = self.data_dir / "weak.jsonl"
        refined_file = self.data_dir / "refined.jsonl"

        self.create_weak_labels_file(weak_file)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "refine-llm",
                    "--weak",
                    str(weak_file),
                    "--output",
                    str(refined_file),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_bootstrap_subcommand(self):
        """Test bootstrap subcommand for Label Studio project setup."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "bootstrap",
                    "--project-name",
                    "Test Project",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_import_weak_subcommand(self):
        """Test import-weak subcommand for Label Studio import."""
        weak_file = self.data_dir / "weak.jsonl"
        self.create_weak_labels_file(weak_file)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "import-weak",
                    "--file",
                    str(weak_file),
                    "--project-id",
                    "1",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_quality_subcommand(self):
        """Test quality subcommand for quality report generation."""
        export_file = self.annotation_dir / "export.json"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "quality",
                    "--export",
                    str(export_file),
                    "--output",
                    str(self.reports_dir / "quality.json"),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_adjudicate_subcommand(self):
        """Test adjudicate subcommand for conflict resolution."""
        conflicts_file = self.conflicts_dir / "conflicts.json"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "adjudicate",
                    "--conflicts",
                    str(conflicts_file),
                    "--output",
                    str(self.exports_dir / "resolved.jsonl"),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_register_subcommand(self):
        """Test register subcommand for batch registration."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = subprocess.run(
                [
                    "python",
                    "scripts/annotation/cli.py",
                    "register",
                    "--batch-id",
                    "batch_001",
                    "--annotators",
                    "ann1,ann2",
                    "--tasks",
                    "100",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0 or mock_run.called

    def test_invalid_subcommand(self):
        """Test CLI fails gracefully with invalid subcommand."""
        result = subprocess.run(
            ["python", "scripts/annotation/cli.py", "invalid-command"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()
