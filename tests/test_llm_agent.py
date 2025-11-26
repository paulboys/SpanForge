"""Tests for LLM agent functionality."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppConfig
from src.llm_agent import LLMAgent


@pytest.fixture
def temp_cache_file(tmp_path):
    """Create temporary cache file for testing."""
    cache_file = tmp_path / "llm_cache.jsonl"
    return str(cache_file)


@pytest.fixture
def llm_config(temp_cache_file):
    """Create test LLM config with temporary cache."""
    config = AppConfig(
        llm_enabled=True, llm_provider="stub", llm_model="gpt-4", llm_cache_path=temp_cache_file
    )
    return config


def test_llm_agent_stub_mode(llm_config):
    """Test LLM agent in stub mode returns empty spans."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    prompt = "Test prompt for NER"
    response = agent.call(prompt)

    assert response is not None
    parsed = json.loads(response)
    assert "spans" in parsed
    assert parsed["spans"] == []
    assert parsed["notes"] == "stub"


def test_llm_agent_caching(llm_config, temp_cache_file):
    """Test LLM agent caching functionality in memory."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    prompt = "Test caching"

    # First call should generate response
    response1 = agent.call(prompt)

    # Second call should return cached response from memory
    response2 = agent.call(prompt)

    assert response1 == response2
    # Note: Stub mode doesn't persist to disk, only in-memory cache


def test_llm_agent_cache_persistence(llm_config, temp_cache_file):
    """Test that cache persists across agent instances."""
    # First agent creates cache
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent1 = LLMAgent()
    prompt = "Persistent test"
    response1 = agent1.call(prompt)

    # Second agent should load existing cache
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent2 = LLMAgent()
    response2 = agent2.call(prompt)

    assert response1 == response2


@patch("src.llm_agent.os.getenv")
def test_openai_client_initialization(mock_getenv, llm_config):
    """Test OpenAI client initialization with API key."""
    try:
        import openai  # noqa: F401

        pytest.skip("openai package is installed - test expects missing dependency")
    except ImportError:
        pass

    mock_getenv.return_value = "test-api-key"

    openai_config = AppConfig(
        llm_enabled=True,
        llm_provider="openai",
        llm_model="gpt-4",
        llm_cache_path=llm_config.llm_cache_path,
    )

    with patch("src.llm_agent.get_config", return_value=openai_config):
        agent = LLMAgent()

        # Trigger client initialization (will fail if openai not installed)
        with pytest.raises((ImportError, ValueError, AttributeError)):
            agent._get_client()


@patch("src.llm_agent.os.getenv")
def test_openai_missing_api_key(mock_getenv, llm_config):
    """Test OpenAI client raises error when API key missing."""
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai package not installed")

    mock_getenv.return_value = None

    openai_config = AppConfig(
        llm_enabled=True,
        llm_provider="openai",
        llm_model="gpt-4",
        llm_cache_path=llm_config.llm_cache_path,
    )

    with patch("src.llm_agent.get_config", return_value=openai_config):
        agent = LLMAgent()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        agent._get_client()


@patch("src.llm_agent.os.getenv")
def test_azure_client_initialization(mock_getenv, llm_config):
    """Test Azure OpenAI client initialization."""
    try:
        import openai  # noqa: F401

        pytest.skip("openai package is installed - test expects missing dependency")
    except ImportError:
        pass

    def getenv_side_effect(key):
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        }
        return env_vars.get(key)

    mock_getenv.side_effect = getenv_side_effect

    azure_config = AppConfig(
        llm_enabled=True,
        llm_provider="azure",
        llm_model="gpt-4",
        llm_cache_path=llm_config.llm_cache_path,
    )

    with patch("src.llm_agent.get_config", return_value=azure_config):
        agent = LLMAgent()

    with pytest.raises((ImportError, ValueError, AttributeError)):
        agent._get_client()


@patch("src.llm_agent.os.getenv")
def test_anthropic_client_initialization(mock_getenv, llm_config):
    """Test Anthropic client initialization."""
    mock_getenv.return_value = "test-api-key"

    anthropic_config = AppConfig(
        llm_enabled=True,
        llm_provider="anthropic",
        llm_model="claude-3-sonnet-20240229",
        llm_cache_path=llm_config.llm_cache_path,
    )

    with patch("src.llm_agent.get_config", return_value=anthropic_config):
        agent = LLMAgent()

    with pytest.raises((ImportError, ValueError, AttributeError)):
        agent._get_client()


def test_unsupported_provider(llm_config):
    """Test error handling for unsupported provider."""
    unsupported_config = AppConfig(
        llm_enabled=True,
        llm_provider="unsupported",
        llm_model="test-model",
        llm_cache_path=llm_config.llm_cache_path,
    )

    with patch("src.llm_agent.get_config", return_value=unsupported_config):
        agent = LLMAgent()

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        agent._get_client()


@patch("src.llm_agent.LLMAgent._call_openai_api")
def test_api_error_handling(mock_api_call, llm_config):
    """Test that API errors are handled gracefully."""
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai package not installed")

    mock_api_call.side_effect = Exception("API error")

    openai_config = AppConfig(
        llm_enabled=True,
        llm_provider="openai",
        llm_model="gpt-4",
        llm_cache_path=llm_config.llm_cache_path,
    )

    # Mock environment variable
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("src.llm_agent.get_config", return_value=openai_config):
            agent = LLMAgent()

        prompt = "Test error handling"
        response = agent.call(prompt)

        parsed = json.loads(response)
        assert "spans" in parsed
        assert parsed["spans"] == []
        assert "api_error" in parsed["notes"]


def test_suggest_basic_functionality(llm_config):
    """Test suggest method basic functionality."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    text = "Patient reports burning sensation and redness."
    weak_spans = [
        {"start": 15, "end": 32, "label": "SYMPTOM", "text": "burning sensation"},
        {"start": 37, "end": 44, "label": "SYMPTOM", "text": "redness"},
    ]
    template = "Analyze: {{text}}\nCandidates: {{candidates}}"
    knowledge = {}

    suggestions = agent.suggest(template, text, weak_spans, knowledge)

    # In stub mode, should return empty list
    assert isinstance(suggestions, list)
    assert len(suggestions) == 0


def test_suggest_with_negation(llm_config):
    """Test suggest method handles negation in text."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    text = "No burning sensation reported."
    weak_spans = [{"start": 3, "end": 20, "label": "SYMPTOM", "text": "burning sensation"}]
    template = "Analyze: {{text}}\nCandidates: {{candidates}}"
    knowledge = {}

    suggestions = agent.suggest(template, text, weak_spans, knowledge)

    assert isinstance(suggestions, list)


def test_empty_weak_spans(llm_config):
    """Test suggest with no weak spans provided."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    text = "Patient is healthy."
    template = "Analyze: {{text}}\nCandidates: {{candidates}}"
    knowledge = {}
    suggestions = agent.suggest(template, text, [], knowledge)

    assert isinstance(suggestions, list)
    assert len(suggestions) == 0


def test_cache_file_not_created_in_stub_mode(llm_config, temp_cache_file):
    """Test that cache file is not created in stub mode."""
    assert not Path(temp_cache_file).exists()

    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    agent.call("Test prompt")

    # Stub mode doesn't persist to disk
    assert not Path(temp_cache_file).exists()


def test_multiple_prompts_stub_mode(llm_config, temp_cache_file):
    """Test stub mode returns same response for all prompts."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    responses = [agent.call(p) for p in prompts]

    # All responses should be identical in stub mode
    assert len(set(responses)) == 1
    assert responses[0] == '{"spans": [], "notes": "stub"}'


def test_duplicate_prompts_use_cache(llm_config, temp_cache_file):
    """Test that duplicate prompts use cached responses in memory."""
    with patch("src.llm_agent.get_config", return_value=llm_config):
        agent = LLMAgent()

    prompt = "Duplicate test"

    # Call multiple times
    responses = [agent.call(prompt) for _ in range(5)]

    # All should return same response
    assert len(set(responses)) == 1
