"""Tests for LLM agent functionality."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppConfig
from src.llm_agent import LLMAgent, LLMSuggestion


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


def test_llm_suggestion_with_defaults():
    """Test LLMSuggestion dataclass can be instantiated with default parameters.

    This test ensures Python 3.9 compatibility by verifying that Optional type hints
    work correctly in dataclass field defaults (not using PEP 604 union syntax).
    """
    # Test with minimal required parameters
    suggestion = LLMSuggestion(start=0, end=10, label="SYMPTOM")
    assert suggestion.start == 0
    assert suggestion.end == 10
    assert suggestion.label == "SYMPTOM"
    assert suggestion.negated is None
    assert suggestion.canonical is None
    assert suggestion.confidence_reason is None
    assert suggestion.llm_confidence is None

    # Test with all parameters specified
    full_suggestion = LLMSuggestion(
        start=5,
        end=20,
        label="PRODUCT",
        negated=True,
        canonical="standardized_name",
        confidence_reason="High confidence based on context",
        llm_confidence=0.95,
    )
    assert full_suggestion.start == 5
    assert full_suggestion.end == 20
    assert full_suggestion.label == "PRODUCT"
    assert full_suggestion.negated is True
    assert full_suggestion.canonical == "standardized_name"
    assert full_suggestion.confidence_reason == "High confidence based on context"
    assert full_suggestion.llm_confidence == 0.95


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
    try:
        import anthropic  # noqa: F401

        pytest.skip("anthropic package is installed - test expects missing dependency")
    except ImportError:
        pass

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


# ===== NEW COMPREHENSIVE TESTS FOR COVERAGE =====


class TestLLMProviderInitialization:
    """Test suite for LLM provider initialization logic."""

    @patch("src.llm_agent.os.getenv")
    def test_openai_client_with_valid_credentials(self, mock_getenv, llm_config):
        """Test OpenAI client initialization succeeds with valid API key."""
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        mock_getenv.return_value = "sk-test-valid-key-123"

        config = AppConfig(
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()
            # Client initialization should not raise
            client = agent._get_client()
            assert client is not None
            assert agent._client is not None  # Cached

    @patch("src.llm_agent.os.getenv")
    def test_azure_missing_endpoint(self, mock_getenv, llm_config):
        """Test Azure client raises error when endpoint is missing."""
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        # API key provided but endpoint missing
        def getenv_side_effect(key):
            if key == "AZURE_OPENAI_API_KEY":
                return "test-key"
            return None

        mock_getenv.side_effect = getenv_side_effect

        config = AppConfig(
            llm_enabled=True,
            llm_provider="azure",
            llm_model="gpt-4",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            agent._get_client()

    @patch("src.llm_agent.os.getenv")
    def test_azure_missing_api_key(self, mock_getenv, llm_config):
        """Test Azure client raises error when API key is missing."""
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        # Endpoint provided but API key missing
        def getenv_side_effect(key):
            if key == "AZURE_OPENAI_ENDPOINT":
                return "https://test.openai.azure.com/"
            return None

        mock_getenv.side_effect = getenv_side_effect

        config = AppConfig(
            llm_enabled=True,
            llm_provider="azure",
            llm_model="gpt-4",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
            agent._get_client()

    @patch("src.llm_agent.os.getenv")
    def test_anthropic_missing_api_key(self, mock_getenv, llm_config):
        """Test Anthropic client raises error when API key is missing."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pytest.skip("anthropic package not installed")

        mock_getenv.return_value = None

        config = AppConfig(
            llm_enabled=True,
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            agent._get_client()


class TestLLMCachingBehavior:
    """Test suite for LLM caching mechanisms."""

    def test_cache_loading_with_corrupted_file(self, llm_config, temp_cache_file):
        """Test cache loading handles corrupted JSONL file gracefully."""
        # Create corrupted cache file
        Path(temp_cache_file).write_text("not valid json\n{incomplete", encoding="utf-8")

        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        # Should not crash, just skip corrupted entries
        assert isinstance(agent._cache, dict)

    def test_cache_hit_returns_immediately(self, llm_config, temp_cache_file):
        """Test that cached responses are returned without API call."""
        # Pre-populate cache
        prompt = "Cached prompt"
        cached_response = '{"spans": [{"start": 0, "end": 5}], "notes": "from_cache"}'

        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        # Manually add to cache
        prompt_hash = str(hash(prompt))
        agent._cache[prompt_hash] = cached_response

        # Should return cached response
        response = agent.call(prompt)
        assert response == cached_response


class TestLLMAPIInteractions:
    """Test suite for LLM API call logic with mocking."""

    @patch("src.llm_agent.os.getenv")
    def test_openai_api_response_structure(self, mock_getenv, llm_config):
        """Test OpenAI API response parsing."""
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        mock_getenv.return_value = "sk-test-key"

        config = AppConfig(
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        # Mock client with successful response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spans": [], "notes": "success"}'
        mock_client.chat.completions.create.return_value = mock_response

        # Call with both client and prompt as required by signature
        result = agent._call_openai_api(mock_client, "Test prompt")

        assert result == '{"spans": [], "notes": "success"}'
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.llm_agent.os.getenv")
    def test_anthropic_api_call_structure(self, mock_getenv, llm_config):
        """Test Anthropic API call structure and parameters."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pytest.skip("anthropic package not installed")

        mock_getenv.return_value = "sk-ant-test-key"

        config = AppConfig(
            llm_enabled=True,
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            llm_cache_path=llm_config.llm_cache_path,
            llm_temperature=0.2,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '{"spans": [], "notes": "anthropic_response"}'
        mock_client.messages.create.return_value = mock_response

        agent._client = mock_client

        result = agent._call_anthropic_api(mock_client, "Test prompt for Anthropic")

        # Verify API was called with correct parameters
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-sonnet-20240229"
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 2048
        assert "Test prompt for Anthropic" in str(call_kwargs["messages"])

    @patch("src.llm_agent.os.getenv")
    def test_api_call_error_handling(self, mock_getenv, llm_config):
        """Test that API call errors are caught and return empty spans."""
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        mock_getenv.return_value = "sk-test-key"

        config = AppConfig(
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        # Mock _call_openai_api to raise exception
        with patch.object(agent, "_call_openai_api", side_effect=RuntimeError("API failed")):
            response = agent.call("Test error")

        parsed = json.loads(response)
        assert parsed["spans"] == []
        assert "api_error" in parsed["notes"]


class TestLLMSuggestMethod:
    """Test suite for the suggest() method with various scenarios."""

    def test_suggest_with_knowledge_context(self, llm_config):
        """Test suggest method incorporates knowledge context in prompt."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        text = "Patient has pruritus and erythema."
        weak_spans = [
            {"start": 12, "end": 20, "label": "SYMPTOM", "text": "pruritus"},
            {"start": 25, "end": 33, "label": "SYMPTOM", "text": "erythema"},
        ]
        template = "Text: {{text}}\nKnowledge: {{knowledge}}\nCandidates: {{candidates}}"
        knowledge = {
            "synonyms": {"pruritus": "itching", "erythema": "redness"},
            "negation_terms": ["no", "without", "absent"],
        }

        suggestions = agent.suggest(template, text, weak_spans, knowledge)

        # Stub mode returns empty, but method should execute without error
        assert isinstance(suggestions, list)

    def test_suggest_json_parsing_robust(self, llm_config):
        """Test suggest handles malformed JSON from LLM gracefully."""
        config = AppConfig(
            llm_enabled=True,
            llm_provider="stub",
            llm_model="test",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        # Mock call to return malformed JSON
        with patch.object(agent, "call", return_value="not valid json at all"):
            text = "Test text"
            suggestions = agent.suggest("template", text, [], {})

            # Should return empty list on parse failure
            assert suggestions == []

    def test_suggest_filters_low_confidence_spans(self, llm_config):
        """Test suggest filters out spans below confidence threshold."""
        config = AppConfig(
            llm_enabled=True,
            llm_provider="stub",
            llm_model="test",
            llm_cache_path=llm_config.llm_cache_path,
            llm_min_confidence=0.8,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        # Mock response with mixed confidence spans
        mock_response = json.dumps(
            {
                "spans": [
                    {"start": 0, "end": 5, "label": "SYMPTOM", "llm_confidence": 0.9},
                    {"start": 10, "end": 15, "label": "SYMPTOM", "llm_confidence": 0.6},
                    {"start": 20, "end": 25, "label": "PRODUCT", "llm_confidence": 0.85},
                ],
                "notes": "mixed_confidence",
            }
        )

        with patch.object(agent, "call", return_value=mock_response):
            suggestions = agent.suggest("template", "test text", [], {})

            # Only spans >= 0.8 should be returned
            assert len(suggestions) == 2
            assert suggestions[0].llm_confidence == 0.9
            assert suggestions[1].llm_confidence == 0.85

    def test_suggest_preserves_all_fields(self, llm_config):
        """Test suggest preserves all LLMSuggestion fields from response."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        mock_response = json.dumps(
            {
                "spans": [
                    {
                        "start": 5,
                        "end": 20,
                        "label": "SYMPTOM",
                        "negated": True,
                        "canonical": "burning_sensation",
                        "confidence_reason": "Clear symptom mention with negation",
                        "llm_confidence": 0.95,
                    }
                ],
                "notes": "complete_fields",
            }
        )

        with patch.object(agent, "call", return_value=mock_response):
            suggestions = agent.suggest("template", "no burning sensation", [], {})

            assert len(suggestions) == 1
            s = suggestions[0]
            assert s.start == 5
            assert s.end == 20
            assert s.label == "SYMPTOM"
            assert s.negated is True
            assert s.canonical == "burning_sensation"
            assert s.confidence_reason == "Clear symptom mention with negation"
            assert s.llm_confidence == 0.95


class TestLLMEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_empty_prompt_handling(self, llm_config):
        """Test agent handles empty prompt string."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        response = agent.call("")

        parsed = json.loads(response)
        assert "spans" in parsed

    def test_very_long_prompt_handling(self, llm_config):
        """Test agent handles very long prompts (token limit testing)."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        # Generate long prompt
        long_prompt = "Analyze this text: " + ("symptom " * 10000)

        response = agent.call(long_prompt)

        # Should not crash, returns valid JSON
        parsed = json.loads(response)
        assert "spans" in parsed

    def test_special_characters_in_text(self, llm_config):
        """Test suggest handles special characters and unicode."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        text = "Patient reports 💊 medication side effects: rash™ and ödëma® (severe)."
        weak_spans = [
            {"start": 44, "end": 48, "label": "SYMPTOM", "text": "rash"},
            {"start": 55, "end": 60, "label": "SYMPTOM", "text": "ödëma"},
        ]

        suggestions = agent.suggest("template", text, weak_spans, {})

        # Should handle unicode without crashing
        assert isinstance(suggestions, list)

    def test_concurrent_cache_access(self, llm_config, temp_cache_file):
        """Test multiple agents can access same cache file."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent1 = LLMAgent()
            agent2 = LLMAgent()

        prompt = "Concurrent test"

        # Both agents call with same prompt
        response1 = agent1.call(prompt)
        response2 = agent2.call(prompt)

        # Should return same cached response
        assert response1 == response2

    def test_stub_provider_returns_valid_json(self, llm_config):
        """Test stub provider always returns valid JSON response."""
        config = AppConfig(
            llm_enabled=True,
            llm_provider="stub",
            llm_model="test",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        response = agent.call("Test stub response")
        assert response is not None

        # Should be valid JSON
        parsed = json.loads(response)
        assert "spans" in parsed
        assert "notes" in parsed

    def test_suggest_with_overlapping_spans(self, llm_config):
        """Test suggest handles overlapping candidate spans."""
        with patch("src.llm_agent.get_config", return_value=llm_config):
            agent = LLMAgent()

        text = "severe burning sensation"
        weak_spans = [
            {"start": 0, "end": 24, "label": "SYMPTOM", "text": "severe burning sensation"},
            {"start": 7, "end": 24, "label": "SYMPTOM", "text": "burning sensation"},
            {"start": 0, "end": 6, "label": "SYMPTOM", "text": "severe"},
        ]

        suggestions = agent.suggest("template", text, weak_spans, {})

        # Should handle overlaps without crashing
        assert isinstance(suggestions, list)

    def test_client_reuse_after_initialization(self, llm_config):
        """Test that _get_client caches client instance."""
        config = AppConfig(
            llm_enabled=True,
            llm_provider="stub",
            llm_model="test",
            llm_cache_path=llm_config.llm_cache_path,
        )

        with patch("src.llm_agent.get_config", return_value=config):
            agent = LLMAgent()

        # First call initializes
        client1 = agent._get_client()
        # Second call returns cached
        client2 = agent._get_client()

        assert client1 is client2
