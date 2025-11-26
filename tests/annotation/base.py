"""Base classes for annotation testing with composition/inheritance patterns."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.base import TestBase


class AnnotationTestBase(TestBase):
    """Base class for annotation testing with fixture/directory management."""

    @pytest.fixture(autouse=True)
    def setup_annotation_fixtures(self, tmp_path):
        """Set up common annotation fixtures and directories."""
        self.tmp_path = tmp_path

        # Create directory structure
        self.data_dir = tmp_path / "data"
        self.annotation_dir = self.data_dir / "annotation"
        self.reports_dir = self.annotation_dir / "reports"
        self.plots_dir = self.annotation_dir / "plots"
        self.exports_dir = self.annotation_dir / "exports"
        self.conflicts_dir = self.annotation_dir / "conflicts"

        for dir_path in [
            self.data_dir,
            self.annotation_dir,
            self.reports_dir,
            self.plots_dir,
            self.exports_dir,
            self.conflicts_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Common test data
        self.sample_text = (
            "After using this facial moisturizer, I developed severe burning sensation and redness."
        )
        self.sample_spans = [
            {
                "text": "burning sensation",
                "start": 57,
                "end": 74,
                "label": "SYMPTOM",
                "canonical": "Burning Sensation",
                "confidence": 0.95,
                "negated": False,
            },
            {
                "text": "redness",
                "start": 79,
                "end": 86,
                "label": "SYMPTOM",
                "canonical": "Erythema",
                "confidence": 0.88,
                "negated": False,
            },
        ]

    def create_jsonl_file(self, path: Path, records: List[Dict]) -> Path:
        """Helper to create JSONL test files."""
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def create_weak_labels_file(self, path: Path, n_records: int = 3) -> Path:
        """Create sample weak labels JSONL file."""
        records = []
        for i in range(n_records):
            records.append(
                {
                    "text": self.sample_text,
                    "source": "test",
                    "metadata": {"id": f"test_{i}"},
                    "spans": self.sample_spans.copy(),
                }
            )
        return self.create_jsonl_file(path, records)

    def create_gold_labels_file(self, path: Path, n_records: int = 3) -> Path:
        """Create sample gold labels JSONL file with corrected spans."""
        records = []
        for i in range(n_records):
            # Gold labels have corrected boundaries
            gold_spans = [
                {
                    "text": "burning sensation",
                    "start": 57,
                    "end": 74,
                    "label": "SYMPTOM",
                    "canonical": "Burning Sensation",
                    "confidence": 1.0,
                    "negated": False,
                },
                {
                    "text": "redness",
                    "start": 79,
                    "end": 86,
                    "label": "SYMPTOM",
                    "canonical": "Erythema",
                    "confidence": 1.0,
                    "negated": False,
                },
            ]
            records.append(
                {
                    "text": self.sample_text,
                    "source": "test",
                    "metadata": {"id": f"test_{i}"},
                    "spans": gold_spans,
                }
            )
        return self.create_jsonl_file(path, records)


class LabelStudioTestBase(AnnotationTestBase):
    """Base class for Label Studio integration testing with HTTP mocking."""

    @pytest.fixture(autouse=True)
    def setup_label_studio_mocks(self):
        """Set up Label Studio API mocks."""
        self.api_key = "test_api_key_12345"
        self.base_url = "http://localhost:8080"
        self.project_id = 1

        # Mock responses
        self.mock_project_response = {
            "id": self.project_id,
            "title": "NER Annotation Project",
            "description": "Test project",
            "created_at": "2025-01-01T00:00:00Z",
        }

        self.mock_task_response = {
            "id": 1,
            "data": {"text": self.sample_text},
            "predictions": [],
            "annotations": [],
        }

        # Patcher for requests
        self.requests_patcher = patch("requests.Session")
        self.mock_session = self.requests_patcher.start()

        # Configure mock session
        mock_instance = MagicMock()
        self.mock_session.return_value = mock_instance

        # Default successful responses
        mock_instance.get.return_value.status_code = 200
        mock_instance.get.return_value.json.return_value = self.mock_project_response
        mock_instance.post.return_value.status_code = 201
        mock_instance.post.return_value.json.return_value = self.mock_task_response

        self.mock_session_instance = mock_instance

        yield

        self.requests_patcher.stop()

    def create_label_studio_export(self, path: Path, n_tasks: int = 3) -> Path:
        """Create sample Label Studio export JSON."""
        tasks = []
        for i in range(n_tasks):
            task = {
                "id": i + 1,
                "data": {"text": self.sample_text},
                "annotations": [
                    {
                        "id": i * 10 + 1,
                        "completed_by": 1,
                        "result": [
                            {
                                "value": {
                                    "start": 57,
                                    "end": 74,
                                    "text": "burning sensation",
                                    "labels": ["SYMPTOM"],
                                },
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                            },
                            {
                                "value": {
                                    "start": 79,
                                    "end": 86,
                                    "text": "redness",
                                    "labels": ["SYMPTOM"],
                                },
                                "from_name": "label",
                                "to_name": "text",
                                "type": "labels",
                            },
                        ],
                    }
                ],
            }
            tasks.append(task)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return path


class EvaluationTestBase(AnnotationTestBase):
    """Base class for evaluation testing with metrics fixtures."""

    @pytest.fixture(autouse=True)
    def setup_evaluation_fixtures(self):
        """Set up evaluation-specific fixtures."""
        # Sample spans for evaluation
        self.weak_spans = [
            {
                "text": "severe burning sensation",
                "start": 50,
                "end": 74,
                "label": "SYMPTOM",
                "confidence": 0.85,
            },
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM", "confidence": 0.88},
        ]

        self.llm_refined_spans = [
            {
                "text": "burning sensation",
                "start": 57,
                "end": 74,
                "label": "SYMPTOM",
                "confidence": 0.95,
            },
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM", "confidence": 0.88},
        ]

        self.gold_spans = [
            {"text": "burning sensation", "start": 57, "end": 74, "label": "SYMPTOM"},
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM"},
        ]

        # Expected metrics
        self.expected_iou_improvement = 0.15  # 15% improvement
        self.expected_exact_match_rate = 1.0  # 100% after LLM refinement
        self.expected_f1_score = 1.0

    def create_evaluation_report(self, path: Path, metrics: Optional[Dict] = None) -> Path:
        """Create sample evaluation report JSON."""
        if metrics is None:
            metrics = {
                "overall": {
                    "iou_delta": {"mean_weak": 0.85, "mean_llm": 1.0, "improvement": 0.15},
                    "boundary_precision": {"exact_match_rate": 1.0, "mean_iou": 1.0},
                    "correction_rate": {"improved": 2, "worsened": 0, "unchanged": 0},
                    "precision_recall_f1": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                },
                "stratified": {"by_label": {"SYMPTOM": {"f1": 1.0, "count": 2}}},
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return path


class VisualizationTestBase(AnnotationTestBase):
    """Base class for visualization testing with matplotlib mocking."""

    @pytest.fixture(autouse=True)
    def setup_visualization_mocks(self):
        """Set up matplotlib/seaborn mocks."""
        # Mock matplotlib - must import first for mocking to work
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            pytest.skip("matplotlib/seaborn not installed")

        # Mock matplotlib
        self.plt_patcher = patch("matplotlib.pyplot")
        self.mock_plt = self.plt_patcher.start()

        # Mock seaborn
        self.sns_patcher = patch("seaborn")
        self.mock_sns = self.sns_patcher.start()

        # Configure mocks
        self.mock_fig = MagicMock()
        self.mock_ax = MagicMock()
        self.mock_plt.subplots.return_value = (self.mock_fig, self.mock_ax)
        self.mock_plt.figure.return_value = self.mock_fig

        yield

        self.plt_patcher.stop()
        self.sns_patcher.stop()

    def verify_plot_created(self, plot_path: Path, format: str = "png"):
        """Verify that a plot was saved to the expected path."""
        self.mock_fig.savefig.assert_called()
        call_args = self.mock_fig.savefig.call_args
        assert str(plot_path) in str(
            call_args
        ), f"Expected plot path {plot_path} not found in savefig calls"


class WorkflowTestBase(AnnotationTestBase):
    """Base class for end-to-end workflow testing."""

    @pytest.fixture(autouse=True)
    def setup_workflow_fixtures(self):
        """Set up workflow-specific fixtures."""
        # Create lexicon files
        self.symptom_lexicon = self.data_dir / "lexicon" / "symptoms.csv"
        self.product_lexicon = self.data_dir / "lexicon" / "products.csv"

        self.symptom_lexicon.parent.mkdir(parents=True, exist_ok=True)

        with open(self.symptom_lexicon, "w", encoding="utf-8") as f:
            f.write("term,canonical\n")
            f.write("burning sensation,Burning Sensation\n")
            f.write("redness,Erythema\n")
            f.write("swelling,Edema\n")

        with open(self.product_lexicon, "w", encoding="utf-8") as f:
            f.write("term,canonical\n")
            f.write("moisturizer,Facial Moisturizer\n")
            f.write("face cream,Facial Cream\n")

        # Create raw input file
        self.raw_input_file = self.data_dir / "raw_complaints.txt"
        with open(self.raw_input_file, "w", encoding="utf-8") as f:
            f.write(self.sample_text + "\n")
            f.write("I experienced swelling and itching after using the face cream.\n")
            f.write("No adverse reactions to the moisturizer.\n")

    def run_full_workflow(self) -> Dict[str, Path]:
        """Execute full annotation workflow and return output paths."""
        output_paths = {
            "weak": self.data_dir / "weak_labels.jsonl",
            "refined": self.data_dir / "llm_refined.jsonl",
            "gold": self.data_dir / "gold_standard.jsonl",
            "report": self.reports_dir / "evaluation.json",
            "plots": self.plots_dir,
        }
        return output_paths


class QualityControlTestBase(AnnotationTestBase):
    """Base class for quality control and multi-annotator testing."""

    @pytest.fixture(autouse=True)
    def setup_quality_fixtures(self):
        """Set up quality control fixtures."""
        # Multi-annotator data
        self.annotator1_spans = [
            {"text": "burning sensation", "start": 57, "end": 74, "label": "SYMPTOM"},
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM"},
        ]

        self.annotator2_spans = [
            {"text": "burning sensation", "start": 57, "end": 74, "label": "SYMPTOM"},
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM"},
        ]

        # Annotator with disagreement
        self.annotator3_spans = [
            {"text": "severe burning sensation", "start": 50, "end": 74, "label": "SYMPTOM"},
            {"text": "redness", "start": 79, "end": 86, "label": "SYMPTOM"},
        ]

        self.registry_file = self.annotation_dir / "registry.csv"
        self.conflicts_file = self.conflicts_dir / "conflicts_batch1.json"

    def create_multi_annotator_export(self, path: Path) -> Path:
        """Create Label Studio export with multiple annotators."""
        task = {
            "id": 1,
            "data": {"text": self.sample_text},
            "annotations": [
                {
                    "id": 1,
                    "completed_by": 1,
                    "result": [
                        {
                            "value": {
                                "start": span["start"],
                                "end": span["end"],
                                "text": span["text"],
                                "labels": [span["label"]],
                            },
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for span in self.annotator1_spans
                    ],
                },
                {
                    "id": 2,
                    "completed_by": 2,
                    "result": [
                        {
                            "value": {
                                "start": span["start"],
                                "end": span["end"],
                                "text": span["text"],
                                "labels": [span["label"]],
                            },
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for span in self.annotator2_spans
                    ],
                },
                {
                    "id": 3,
                    "completed_by": 3,
                    "result": [
                        {
                            "value": {
                                "start": span["start"],
                                "end": span["end"],
                                "text": span["text"],
                                "labels": [span["label"]],
                            },
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for span in self.annotator3_spans
                    ],
                },
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump([task], f, indent=2)
        return path
