"""Tests for src.config module (configuration management)."""

import os
import unittest
from unittest.mock import Mock, patch

from src.config import AppConfig, get_config, set_seed
from tests.base import TestBase


class TestAppConfig(TestBase):
    """Tests for AppConfig class."""

    def test_default_config_values(self):
        """Test that default configuration values are set correctly."""
        config = AppConfig()

        self.assertEqual(config.model_name, "dmis-lab/biobert-base-cased-v1.1")
        self.assertEqual(config.max_seq_len, 256)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.negation_window, 5)
        self.assertEqual(config.fuzzy_scorer, "wratio")
        self.assertFalse(config.llm_enabled)
        self.assertEqual(config.llm_provider, "stub")
        self.assertEqual(config.llm_model, "gpt-4")
        self.assertEqual(config.llm_min_confidence, 0.65)

    @patch.dict(os.environ, {"MODEL_NAME": "custom-model"})
    def test_env_override_model_name(self):
        """Test that environment variables override defaults."""
        config = AppConfig()
        self.assertEqual(config.model_name, "custom-model")

    @patch.dict(os.environ, {"MAX_SEQ_LEN": "512"})
    def test_env_override_max_seq_len(self):
        """Test that environment variables can override integer values."""
        config = AppConfig()
        self.assertEqual(config.max_seq_len, 512)

    @patch.dict(os.environ, {"SEED": "123"})
    def test_env_override_seed(self):
        """Test environment override for seed value."""
        config = AppConfig()
        self.assertEqual(config.seed, 123)

    @patch.dict(os.environ, {"NEGATION_WINDOW": "10"})
    def test_env_override_negation_window(self):
        """Test environment override for negation window."""
        config = AppConfig()
        self.assertEqual(config.negation_window, 10)

    @patch.dict(os.environ, {"FUZZY_SCORER": "jaccard"})
    def test_env_override_fuzzy_scorer(self):
        """Test environment override for fuzzy scorer."""
        config = AppConfig()
        self.assertEqual(config.fuzzy_scorer, "jaccard")

    @patch.dict(os.environ, {"LLM_ENABLED": "true"})
    def test_env_override_llm_enabled(self):
        """Test environment override for boolean LLM enabled flag."""
        config = AppConfig()
        self.assertTrue(config.llm_enabled)

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai"})
    def test_env_override_llm_provider(self):
        """Test environment override for LLM provider."""
        config = AppConfig()
        self.assertEqual(config.llm_provider, "openai")

    @patch.dict(os.environ, {"LLM_MODEL": "gpt-4o"})
    def test_env_override_llm_model(self):
        """Test environment override for LLM model."""
        config = AppConfig()
        self.assertEqual(config.llm_model, "gpt-4o")

    @patch.dict(os.environ, {"LLM_MIN_CONFIDENCE": "0.8"})
    def test_env_override_llm_min_confidence(self):
        """Test environment override for LLM minimum confidence."""
        config = AppConfig()
        self.assertEqual(config.llm_min_confidence, 0.8)

    @patch.dict(os.environ, {"DEVICE": "cuda"})
    def test_device_cuda_override(self):
        """Test device can be overridden via environment variable."""
        config = AppConfig()
        self.assertEqual(config.device, "cuda")

    @patch("src.config.torch")
    def test_device_cuda_not_available(self, mock_torch):
        """Test device defaults to cpu when cuda not available."""
        mock_torch.cuda.is_available.return_value = False
        config = AppConfig()
        self.assertEqual(config.device, "cpu")

    @patch("src.config.torch", None)
    def test_device_torch_not_installed(self):
        """Test device defaults to cpu when torch not installed."""
        config = AppConfig()
        self.assertEqual(config.device, "cpu")


class TestGetConfig(TestBase):
    """Tests for get_config function."""

    def test_get_config_returns_app_config(self):
        """Test that get_config returns an AppConfig instance."""
        config = get_config()
        self.assertIsInstance(config, AppConfig)

    def test_get_config_multiple_calls(self):
        """Test that multiple calls return fresh instances."""
        config1 = get_config()
        config2 = get_config()
        # They are different instances
        self.assertIsNot(config1, config2)
        # But have same default values
        self.assertEqual(config1.model_name, config2.model_name)


class TestSetSeed(TestBase):
    """Tests for set_seed function."""

    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("numpy.random.seed")
    @patch("random.seed")
    def test_set_seed_all_libraries(
        self, mock_random, mock_numpy, mock_cuda_available, mock_cuda_seed, mock_torch_seed
    ):
        """Test that set_seed sets seeds for all libraries."""
        set_seed(42)

        mock_random.assert_called_once_with(42)
        mock_numpy.assert_called_once_with(42)
        mock_torch_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_called_once_with(42)

    @patch("torch.manual_seed")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("numpy.random.seed")
    @patch("random.seed")
    def test_set_seed_no_cuda(self, mock_random, mock_numpy, mock_cuda_available, mock_torch_seed):
        """Test that set_seed works when CUDA is not available."""
        set_seed(123)

        mock_random.assert_called_once_with(123)
        mock_numpy.assert_called_once_with(123)
        mock_torch_seed.assert_called_once_with(123)
        # cuda.manual_seed_all should not be called

    @patch("src.config.torch", None)
    @patch("numpy.random.seed")
    @patch("random.seed")
    def test_set_seed_no_torch(self, mock_random, mock_numpy):
        """Test that set_seed works when torch is not installed."""
        set_seed(456)

        mock_random.assert_called_once_with(456)
        mock_numpy.assert_called_once_with(456)
        # torch seeds should not be called (no error should occur)

    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("numpy.random.seed")
    @patch("random.seed")
    def test_set_seed_different_values(
        self, mock_random, mock_numpy, mock_cuda_available, mock_cuda_seed, mock_torch_seed
    ):
        """Test that set_seed can be called with different values."""
        set_seed(1)
        mock_random.assert_called_with(1)

        set_seed(999)
        mock_random.assert_called_with(999)

        # Verify called twice
        self.assertEqual(mock_random.call_count, 2)


class TestConfigIntegration(TestBase):
    """Integration tests for configuration workflow."""

    def test_config_used_in_workflow(self):
        """Test that config can be used in typical workflow."""
        config = get_config()

        # Verify typical workflow attributes are accessible
        self.assertIsNotNone(config.model_name)
        self.assertIsNotNone(config.device)
        self.assertIsNotNone(config.max_seq_len)
        self.assertIsNotNone(config.negation_window)

    @patch.dict(
        os.environ,
        {
            "MODEL_NAME": "test-model",
            "MAX_SEQ_LEN": "128",
            "SEED": "999",
            "NEGATION_WINDOW": "3",
            "FUZZY_SCORER": "jaccard",
            "LLM_ENABLED": "true",
            "LLM_PROVIDER": "anthropic",
        },
    )
    def test_config_multiple_env_overrides(self):
        """Test configuration with multiple environment overrides."""
        config = get_config()

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.max_seq_len, 128)
        self.assertEqual(config.seed, 999)
        self.assertEqual(config.negation_window, 3)
        self.assertEqual(config.fuzzy_scorer, "jaccard")
        self.assertTrue(config.llm_enabled)
        self.assertEqual(config.llm_provider, "anthropic")


if __name__ == "__main__":
    unittest.main()
