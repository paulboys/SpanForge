"""Tests for src.model_token_cls module (token classification model scaffold)."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.model_token_cls import get_token_cls_model, load_labels
from tests.base import TestBase


class ModelTokenClsTestBase(TestBase):
    """Base class for token classification model tests."""

    def setUp(self):
        super().setUp()
        self.labels_path = self.temp_dir / "labels.json"

    def create_labels_file(self, labels: list) -> Path:
        """Create labels.json file with given labels."""
        self.labels_path.write_text(json.dumps(labels), encoding="utf-8")
        return self.labels_path


class TestLoadLabels(ModelTokenClsTestBase):
    """Tests for load_labels function."""

    def test_load_labels_valid_file(self):
        """Test loading valid labels file."""
        labels = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-PRODUCT", "I-PRODUCT"]
        self.create_labels_file(labels)

        loaded = load_labels(self.labels_path)

        self.assertEqual(loaded, labels)

    def test_load_labels_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        nonexistent = self.temp_dir / "nonexistent.json"

        with self.assertRaises(FileNotFoundError) as ctx:
            load_labels(nonexistent)

        self.assertIn("Labels file not found", str(ctx.exception))
        self.assertIn(str(nonexistent), str(ctx.exception))

    def test_load_labels_empty_list(self):
        """Test loading empty labels list."""
        self.create_labels_file([])

        loaded = load_labels(self.labels_path)

        self.assertEqual(loaded, [])

    def test_load_labels_invalid_type(self):
        """Test loading file with invalid type raises ValueError."""
        # Write dict instead of list
        self.labels_path.write_text(json.dumps({"labels": ["O"]}), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_labels(self.labels_path)

        self.assertIn("must be a JSON list of strings", str(ctx.exception))

    def test_load_labels_non_string_elements(self):
        """Test loading file with non-string elements raises ValueError."""
        self.create_labels_file(["O", 1, "B-SYMPTOM"])  # Contains integer

        with self.assertRaises(ValueError) as ctx:
            load_labels(self.labels_path)

        self.assertIn("must be a JSON list of strings", str(ctx.exception))

    def test_load_labels_malformed_json(self):
        """Test loading file with malformed JSON raises error."""
        self.labels_path.write_text("not valid json", encoding="utf-8")

        with self.assertRaises(json.JSONDecodeError):
            load_labels(self.labels_path)

    def test_load_labels_with_unicode(self):
        """Test loading labels with unicode characters."""
        labels = ["O", "B-SYMPTÔM", "I-SYMPTÔM"]
        self.create_labels_file(labels)

        loaded = load_labels(self.labels_path)

        self.assertEqual(loaded, labels)
        self.assertIn("B-SYMPTÔM", loaded)

    def test_load_labels_large_list(self):
        """Test loading large labels list."""
        labels = [f"LABEL-{i}" for i in range(1000)]
        self.create_labels_file(labels)

        loaded = load_labels(self.labels_path)

        self.assertEqual(len(loaded), 1000)
        self.assertEqual(loaded[0], "LABEL-0")
        self.assertEqual(loaded[-1], "LABEL-999")


class TestGetTokenClsModel(ModelTokenClsTestBase):
    """Tests for get_token_cls_model function."""

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_get_token_cls_model_basic(self, mock_get_config, mock_tokenizer_cls, mock_model_cls):
        """Test basic model loading."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create labels file
        labels = ["O", "B-SYMPTOM", "I-SYMPTOM"]
        self.create_labels_file(labels)

        # Load model
        tokenizer, model = get_token_cls_model(self.labels_path)

        # Verify tokenizer loading
        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "dmis-lab/biobert-base-cased-v1.1"
        )

        # Verify model loading with correct parameters
        mock_model_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["num_labels"], 3)
        self.assertIn("id2label", call_kwargs)
        self.assertIn("label2id", call_kwargs)

        # Verify model moved to device
        mock_model.to.assert_called_once_with("cpu")

        # Verify returns
        self.assertEqual(tokenizer, mock_tokenizer)
        self.assertEqual(model, mock_model)

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_get_token_cls_model_label_mappings(
        self, mock_get_config, mock_tokenizer_cls, mock_model_cls
    ):
        """Test that label mappings are created correctly."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create labels file
        labels = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-PRODUCT"]
        self.create_labels_file(labels)

        # Load model
        get_token_cls_model(self.labels_path)

        # Extract label mappings from call
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        id2label = call_kwargs["id2label"]
        label2id = call_kwargs["label2id"]

        # Verify id2label
        self.assertEqual(id2label[0], "O")
        self.assertEqual(id2label[1], "B-SYMPTOM")
        self.assertEqual(id2label[2], "I-SYMPTOM")
        self.assertEqual(id2label[3], "B-PRODUCT")

        # Verify label2id (inverse mapping)
        self.assertEqual(label2id["O"], 0)
        self.assertEqual(label2id["B-SYMPTOM"], 1)
        self.assertEqual(label2id["I-SYMPTOM"], 2)
        self.assertEqual(label2id["B-PRODUCT"], 3)

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_get_token_cls_model_with_cuda(
        self, mock_get_config, mock_tokenizer_cls, mock_model_cls
    ):
        """Test model loading with CUDA device."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cuda:0"
        mock_get_config.return_value = mock_config

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create labels file
        labels = ["O", "B-SYMPTOM"]
        self.create_labels_file(labels)

        # Load model
        get_token_cls_model(self.labels_path)

        # Verify model moved to CUDA device
        mock_model.to.assert_called_once_with("cuda:0")

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_get_token_cls_model_default_labels_path(
        self, mock_get_config, mock_tokenizer_cls, mock_model_cls
    ):
        """Test model loading with default labels path."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        # Create labels file at default location
        default_labels = Path("labels.json")
        original_content = None
        if default_labels.exists():
            original_content = default_labels.read_text()

        try:
            default_labels.write_text(json.dumps(["O", "B-SYMPTOM"]), encoding="utf-8")

            # Load model without specifying path
            get_token_cls_model()

            # Should have loaded from default path
            mock_model_cls.from_pretrained.assert_called_once()

        finally:
            # Cleanup: restore or remove
            if original_content is not None:
                default_labels.write_text(original_content)
            elif default_labels.exists():
                default_labels.unlink()

    def test_get_token_cls_model_invalid_labels_file(self):
        """Test that invalid labels file raises appropriate error."""
        # Create invalid labels file
        self.labels_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

        with self.assertRaises(ValueError):
            get_token_cls_model(self.labels_path)

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_get_token_cls_model_single_label(
        self, mock_get_config, mock_tokenizer_cls, mock_model_cls
    ):
        """Test model loading with single label."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create labels file with single label
        labels = ["O"]
        self.create_labels_file(labels)

        # Load model
        get_token_cls_model(self.labels_path)

        # Verify num_labels
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs["num_labels"], 1)


class TestModelTokenClsIntegration(ModelTokenClsTestBase):
    """Integration tests for token classification model loading."""

    @patch("src.model_token_cls.AutoModelForTokenClassification")
    @patch("src.model_token_cls.AutoTokenizer")
    @patch("src.model_token_cls.get_config")
    def test_full_workflow(self, mock_get_config, mock_tokenizer_cls, mock_model_cls):
        """Test complete workflow from labels file to model."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "dmis-lab/biobert-base-cased-v1.1"
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Create realistic BIO labels
        labels = [
            "O",
            "B-SYMPTOM",
            "I-SYMPTOM",
            "B-PRODUCT",
            "I-PRODUCT",
        ]
        self.create_labels_file(labels)

        # Load model
        tokenizer, model = get_token_cls_model(self.labels_path)

        # Verify complete setup
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)

        # Verify label mappings are correct
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        id2label = call_kwargs["id2label"]
        label2id = call_kwargs["label2id"]

        # Check BIO tag structure preserved
        self.assertEqual(len(id2label), 5)
        # Verify correct mapping between label2id and id2label
        self.assertEqual(id2label[label2id["B-SYMPTOM"]], "B-SYMPTOM")
        self.assertEqual(id2label[label2id["I-SYMPTOM"]], "I-SYMPTOM")

        # Verify bidirectional mapping
        for label_id, label_text in id2label.items():
            self.assertEqual(label2id[label_text], label_id)


if __name__ == "__main__":
    unittest.main()
