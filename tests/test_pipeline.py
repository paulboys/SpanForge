"""Tests for src.pipeline module (end-to-end inference pipeline)."""

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import torch

from src.pipeline import postprocess_predictions, predict_tokens, simple_inference, tokenize_batch
from tests.base import TestBase


class PipelineTestBase(TestBase):
    """Base class for pipeline tests with common fixtures."""

    def setUp(self):
        super().setUp()
        # Create minimal lexicons
        self.symptom_lexicon_path = self.temp_dir / "lexicon" / "symptoms.csv"
        self.product_lexicon_path = self.temp_dir / "lexicon" / "products.csv"
        self.symptom_lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        self.product_lexicon_path.parent.mkdir(parents=True, exist_ok=True)

        # Write minimal symptom lexicon
        self.symptom_lexicon_path.write_text(
            "term,canonical,source,concept_id\n"
            "headache,Headache,test,\n"
            "rash,Rash,test,\n"
            "nausea,Nausea,test,\n",
            encoding="utf-8",
        )

        # Write minimal product lexicon
        self.product_lexicon_path.write_text(
            "term,sku,category\n" "aspirin,ASP-001,medication\n" "cream,CRM-001,topical\n",
            encoding="utf-8",
        )

        # Patch data paths
        self.data_patch = patch("src.pipeline.Path")
        self.mock_path = self.data_patch.start()
        self.mock_path.return_value = self.temp_dir / "lexicon"

        def path_side_effect(path_str: str):
            if "symptoms.csv" in path_str:
                return self.symptom_lexicon_path
            elif "products.csv" in path_str:
                return self.product_lexicon_path
            return Path(path_str)

        self.mock_path.side_effect = path_side_effect

    def tearDown(self):
        self.data_patch.stop()
        super().tearDown()

    def create_mock_tokenizer(self) -> MagicMock:
        """Create mock tokenizer with typical behavior."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        tokenizer.tokenize.return_value = ["patient", "has", "headache"]
        return tokenizer

    def create_mock_model(self) -> MagicMock:
        """Create mock model with typical output."""
        model = MagicMock()
        model.return_value = Mock(
            last_hidden_state=torch.randn(1, 4, 768)  # batch=1, seq_len=4, hidden=768
        )
        return model


class TestTokenizeBatch(PipelineTestBase):
    """Tests for tokenize_batch function."""

    def test_tokenize_single_text(self):
        """Test tokenizing single text."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        result = tokenize_batch(["Patient has headache"], tokenizer, max_len=128)

        tokenizer.assert_called_once_with(
            ["Patient has headache"],
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(result["input_ids"], [[1, 2, 3]])

    def test_tokenize_multiple_texts(self):
        """Test tokenizing batch of texts."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }

        texts = ["Patient has headache", "Another complaint"]
        result = tokenize_batch(texts, tokenizer, max_len=64)

        tokenizer.assert_called_once()
        call_args = tokenizer.call_args[0][0]
        self.assertEqual(call_args, texts)

    def test_tokenize_with_truncation(self):
        """Test max_length parameter is passed correctly."""
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [[1]], "attention_mask": [[1]]}

        tokenize_batch(["Long text"], tokenizer, max_len=32)

        call_kwargs = tokenizer.call_args[1]
        self.assertEqual(call_kwargs["max_length"], 32)
        self.assertTrue(call_kwargs["truncation"])
        self.assertTrue(call_kwargs["padding"])


class TestPredictTokens(PipelineTestBase):
    """Tests for predict_tokens function."""

    def test_predict_returns_hidden_state(self):
        """Test model inference returns last_hidden_state."""
        model = MagicMock()
        hidden_state = torch.randn(2, 10, 768)
        model.return_value = Mock(last_hidden_state=hidden_state)

        encodings = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        result = predict_tokens(model, encodings, device="cpu")

        self.assertIn("last_hidden_state", result)
        self.assertTrue(torch.equal(result["last_hidden_state"], hidden_state))

    def test_predict_moves_to_device(self):
        """Test tensors are moved to correct device."""
        model = MagicMock()
        model.return_value = Mock(last_hidden_state=torch.randn(1, 4, 768))

        encodings = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        predict_tokens(model, encodings, device="cpu")

        # Check model was called with tensors moved to device
        model.assert_called_once()
        call_args = model.call_args[1]
        self.assertIn("input_ids", call_args)
        self.assertIn("attention_mask", call_args)

    def test_predict_no_grad_context(self):
        """Test inference runs in no_grad context."""
        model = MagicMock()
        model.return_value = Mock(last_hidden_state=torch.randn(1, 4, 768))

        encodings = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        with patch("torch.no_grad") as mock_no_grad:
            mock_context = MagicMock()
            mock_no_grad.return_value.__enter__ = Mock(return_value=mock_context)
            mock_no_grad.return_value.__exit__ = Mock(return_value=False)

            predict_tokens(model, encodings, device="cpu")

            mock_no_grad.assert_called_once()


class TestPostprocessPredictions(PipelineTestBase):
    """Tests for postprocess_predictions function."""

    def test_postprocess_returns_token_counts(self):
        """Test postprocessing returns token counts."""
        batch_tokens = [["patient", "has", "headache"], ["rash", "appeared"]]
        model_outputs = {"last_hidden_state": torch.randn(2, 5, 768)}

        result = postprocess_predictions(batch_tokens, model_outputs)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["token_count"], 3)
        self.assertEqual(result[1]["token_count"], 2)

    def test_postprocess_empty_batch(self):
        """Test postprocessing empty batch."""
        batch_tokens: List[List[str]] = []
        model_outputs = {"last_hidden_state": torch.randn(0, 0, 768)}

        result = postprocess_predictions(batch_tokens, model_outputs)

        self.assertEqual(result, [])

    def test_postprocess_single_token(self):
        """Test postprocessing single token."""
        batch_tokens = [["headache"]]
        model_outputs = {"last_hidden_state": torch.randn(1, 2, 768)}

        result = postprocess_predictions(batch_tokens, model_outputs)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["token_count"], 1)


class TestSimpleInference(PipelineTestBase):
    """Tests for simple_inference end-to-end pipeline."""

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    def test_simple_inference_basic(self, mock_config_cls, mock_get_tok, mock_get_model):
        """Test basic inference pipeline without persistence."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = self.create_mock_tokenizer()
        mock_get_tok.return_value = tokenizer

        model = self.create_mock_model()
        mock_get_model.return_value = model

        # Run inference
        texts = ["Patient has headache"]
        results = simple_inference(texts, persist_path=None)

        # Verify
        self.assertEqual(len(results), 1)
        self.assertIn("weak_spans", results[0])
        self.assertIn("token_count", results[0])

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    def test_simple_inference_with_lexicon_match(
        self, mock_config_cls, mock_get_tok, mock_get_model
    ):
        """Test inference detects symptom spans."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = self.create_mock_tokenizer()
        mock_get_tok.return_value = tokenizer

        model = self.create_mock_model()
        mock_get_model.return_value = model

        # Run inference with text containing known symptom
        texts = ["Patient has headache and rash"]
        results = simple_inference(texts, persist_path=None)

        # Verify spans were detected
        self.assertEqual(len(results), 1)
        self.assertIn("weak_spans", results[0])
        # Should detect at least one symptom (headache or rash)
        self.assertGreaterEqual(len(results[0]["weak_spans"]), 1)

        # Check span structure
        if results[0]["weak_spans"]:
            span = results[0]["weak_spans"][0]
            self.assertIn("text", span)
            self.assertIn("label", span)
            self.assertIn("confidence", span)
            self.assertIn("negated", span)

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    @patch("src.pipeline.persist_weak_labels_jsonl")
    def test_simple_inference_with_persistence(
        self, mock_persist, mock_config_cls, mock_get_tok, mock_get_model
    ):
        """Test inference with JSONL persistence."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = self.create_mock_tokenizer()
        mock_get_tok.return_value = tokenizer

        model = self.create_mock_model()
        mock_get_model.return_value = model

        # Run inference with persistence
        persist_path = str(self.temp_dir / "output.jsonl")
        texts = ["Patient has headache"]
        results = simple_inference(texts, persist_path=persist_path)

        # Verify persistence was called
        mock_persist.assert_called_once()
        call_args = mock_persist.call_args[0]
        self.assertEqual(call_args[0], texts)  # texts
        # call_args[1] is weak labels
        self.assertEqual(str(call_args[2]), persist_path)  # path

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    def test_simple_inference_empty_lexicons(self, mock_config_cls, mock_get_tok, mock_get_model):
        """Test inference with empty lexicons."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = self.create_mock_tokenizer()
        mock_get_tok.return_value = tokenizer

        model = self.create_mock_model()
        mock_get_model.return_value = model

        # Empty lexicons
        self.symptom_lexicon_path.write_text("term,canonical,source,concept_id\n")
        self.product_lexicon_path.write_text("term,sku,category\n")

        # Run inference
        texts = ["Patient has headache"]
        results = simple_inference(texts, persist_path=None)

        # Should still work but with no spans
        self.assertEqual(len(results), 1)
        self.assertIn("weak_spans", results[0])
        self.assertEqual(results[0]["weak_spans"], [])

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    def test_simple_inference_batch(self, mock_config_cls, mock_get_tok, mock_get_model):
        """Test inference with multiple texts."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        tokenizer.tokenize.side_effect = [
            ["patient", "has", "headache"],
            ["another", "rash"],
        ]
        mock_get_tok.return_value = tokenizer

        model = MagicMock()
        model.return_value = Mock(last_hidden_state=torch.randn(2, 3, 768))
        mock_get_model.return_value = model

        # Run inference with batch
        texts = ["Patient has headache", "Another rash case"]
        results = simple_inference(texts, persist_path=None)

        # Verify batch processing
        self.assertEqual(len(results), 2)
        self.assertIn("weak_spans", results[0])
        self.assertIn("weak_spans", results[1])


class TestPipelineIntegration(PipelineTestBase):
    """Integration tests for pipeline components."""

    @patch("src.pipeline.get_model")
    @patch("src.pipeline.get_tokenizer")
    @patch("src.pipeline.AppConfig")
    def test_end_to_end_with_real_structures(self, mock_config_cls, mock_get_tok, mock_get_model):
        """Test end-to-end pipeline with realistic data structures."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.max_seq_len = 128
        mock_config.device = "cpu"
        mock_config.negation_window = 5
        mock_config.fuzzy_scorer = "WRatio"
        mock_config_cls.return_value = mock_config

        tokenizer = self.create_mock_tokenizer()
        mock_get_tok.return_value = tokenizer

        model = self.create_mock_model()
        mock_get_model.return_value = model

        # Run with typical medical complaint
        texts = [
            "After using the cream, I developed severe rash and headache",
            "No side effects, product works well",
        ]
        results = simple_inference(texts, persist_path=None)

        # Verify structure
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn("token_count", result)
            self.assertIn("weak_spans", result)
            self.assertIsInstance(result["weak_spans"], list)

            # Check span structure if any detected
            for span in result["weak_spans"]:
                self.assertIn("text", span)
                self.assertIn("start", span)
                self.assertIn("end", span)
                self.assertIn("label", span)
                self.assertIn("canonical", span)
                self.assertIn("confidence", span)
                self.assertIn("negated", span)


if __name__ == "__main__":
    unittest.main()
