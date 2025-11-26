"""LLM agent for span refinement and enhancement.

Provides LLM-based entity span refinement with support for multiple providers.
Supports OpenAI, Azure OpenAI, and Anthropic APIs with caching and rate limiting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

    # Fallback decorator that does nothing
    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    stop_after_attempt = wait_exponential = retry_if_exception_type = None

from .config import get_config


@dataclass
class LLMSuggestion:
    """LLM-generated span suggestion with confidence and reasoning.

    Attributes:
        start: Character start position of suggested span
        end: Character end position of suggested span
        label: Entity label (SYMPTOM or PRODUCT)
        negated: Whether the entity is negated (optional)
        canonical: Canonical form of the entity (optional)
        confidence_reason: Textual explanation for confidence score (optional)
        llm_confidence: Confidence score from LLM (optional)
    """

    start: int
    end: int
    label: str
    negated: bool | None = None
    canonical: str | None = None
    confidence_reason: str | None = None
    llm_confidence: float | None = None


class LLMAgent:
    """LLM agent for entity span refinement with multi-provider support.

    Provides interface for LLM-based span suggestion and refinement.
    Supports OpenAI, Azure OpenAI, Anthropic, and stub mode.

    Attributes:
        provider: LLM provider name ('stub', 'openai', 'azure', 'anthropic')
        model: Model identifier for the LLM
        min_conf: Minimum confidence threshold for accepting suggestions
        cache_path: Path to JSONL cache file for LLM responses
        _client: Lazily initialized API client

    Environment Variables:
        OPENAI_API_KEY: API key for OpenAI provider
        AZURE_OPENAI_API_KEY: API key for Azure provider
        AZURE_OPENAI_ENDPOINT: Endpoint URL for Azure provider
        ANTHROPIC_API_KEY: API key for Anthropic provider

    Example:
        >>> # Stub mode (no API calls)
        >>> agent = LLMAgent()
        >>> suggestions = agent.suggest(template, text, spans, knowledge)

        >>> # OpenAI mode (requires OPENAI_API_KEY)
        >>> import os
        >>> os.environ['OPENAI_API_KEY'] = 'sk-...'
        >>> config = AppConfig(llm_provider='openai', llm_model='gpt-4-turbo')
        >>> agent = LLMAgent()
        >>> suggestions = agent.suggest(template, text, spans, knowledge)
    """

    def __init__(self) -> None:
        """Initialize LLM agent with configuration."""
        cfg = get_config()
        self.provider: str = cfg.llm_provider
        self.model: str = cfg.llm_model
        self.min_conf: float = cfg.llm_min_confidence
        self.cache_path: Path = Path(cfg.llm_cache_path)
        self._client: Any = None
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached LLM responses from disk."""
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    prompt_hash = str(hash(entry.get("prompt", "")))
                    self._cache[prompt_hash] = entry.get("response", "{}")
        except Exception:
            pass

    def _save_to_cache(self, prompt: str, response: str) -> None:
        """Save LLM response to cache file.

        Args:
            prompt: The prompt sent to LLM
            response: The response received from LLM
        """
        prompt_hash = str(hash(prompt))
        self._cache[prompt_hash] = response

        # Append to cache file
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                entry = {
                    "timestamp": time.time(),
                    "provider": self.provider,
                    "model": self.model,
                    "prompt": prompt,
                    "response": response,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _get_client(self) -> Any:
        """Lazily initialize and return API client.

        Returns:
            Initialized API client for the configured provider

        Raises:
            ImportError: If required SDK not installed
            ValueError: If API credentials not found or invalid provider
        """
        if self._client is not None:
            return self._client

        if self.provider == "stub":
            self._client = None
            return None

        elif self.provider == "openai":
            try:
                import openai

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self._client = openai.OpenAI(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.provider == "azure":
            try:
                import openai

                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if not api_key or not endpoint:
                    raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT required")
                self._client = openai.AzureOpenAI(
                    api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview"
                )
                return self._client
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.provider == "anthropic":
            try:
                import anthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                self._client = anthropic.Anthropic(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def format_prompt(
        self, template: str, text: str, spans: List[Dict[str, Any]], knowledge: Dict[str, Any]
    ) -> str:
        """Format prompt template with text, spans, and knowledge.

        Args:
            template: Prompt template with {{text}}, {{candidates}}, {{knowledge}} placeholders
            text: Source text being analyzed
            spans: List of candidate spans to refine
            knowledge: Additional domain knowledge dictionary

        Returns:
            Formatted prompt string ready for LLM
        """
        candidates = [
            f"{s['text']} [{s['start']},{s['end']}] {s['label']} conf={s.get('confidence',1):.2f}"
            for s in spans
        ]
        return (
            template.replace("{{text}}", text)
            .replace("{{candidates}}", "\n".join(candidates))
            .replace("{{knowledge}}", json.dumps(knowledge, ensure_ascii=False))
        )

    def _call_openai_api(self, client: Any, prompt: str) -> str:
        """Call OpenAI/Azure API with retry logic.

        Args:
            client: OpenAI client instance
            prompt: Prompt to send

        Returns:
            Response content as string

        Raises:
            Exception: If API call fails after retries
        """
        if TENACITY_AVAILABLE:
            # Define retry decorator dynamically
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((Exception,)),
            )
            def _api_call():
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical NER expert. Analyze text and suggest entity spans in JSON format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                return completion.choices[0].message.content or "{}"

            return _api_call()
        else:
            # No retry if tenacity not available
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical NER expert. Analyze text and suggest entity spans in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or "{}"

    def _call_anthropic_api(self, client: Any, prompt: str) -> str:
        """Call Anthropic API with retry logic.

        Args:
            client: Anthropic client instance
            prompt: Prompt to send

        Returns:
            Response content as string

        Raises:
            Exception: If API call fails after retries
        """
        if TENACITY_AVAILABLE:

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((Exception,)),
            )
            def _api_call():
                message = client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.3,
                    system="You are a medical NER expert. Analyze text and suggest entity spans in JSON format.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            return _api_call()
        else:
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.3,
                system="You are a medical NER expert. Analyze text and suggest entity spans in JSON format.",
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

    def call(self, prompt: str) -> str:
        """Call LLM with formatted prompt.

        Supports caching to avoid redundant API calls. Checks cache first,
        then calls appropriate provider API if not cached.

        Args:
            prompt: Formatted prompt string

        Returns:
            JSON string with LLM response

        Raises:
            ImportError: If required SDK not installed
            ValueError: If API credentials missing
            Exception: If API call fails

        Note:
            - Stub mode returns empty suggestions for deterministic testing
            - All providers return JSON with {"spans": [...], "notes": "..."}
            - Responses are cached to disk for reproducibility
        """
        # Check cache first
        prompt_hash = str(hash(prompt))
        if prompt_hash in self._cache:
            return self._cache[prompt_hash]

        # Stub mode: return empty JSON (no API call)
        if self.provider == "stub":
            response = json.dumps({"spans": [], "notes": "stub"})
            return response

        # Get API client
        client = self._get_client()

        try:
            # OpenAI / Azure OpenAI
            if self.provider in ["openai", "azure"]:
                response = self._call_openai_api(client, prompt)

            # Anthropic
            elif self.provider == "anthropic":
                response = self._call_anthropic_api(client, prompt) if message.content else "{}"

            else:
                response = json.dumps({"spans": [], "notes": "unknown_provider"})

            # Save to cache
            self._save_to_cache(prompt, response)
            return response

        except Exception as e:
            # Return error response instead of raising
            error_response = json.dumps({"spans": [], "notes": f"api_error: {str(e)[:100]}"})
            return error_response

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse LLM response JSON string.

        Args:
            response: JSON string from LLM

        Returns:
            Parsed dictionary with 'spans' list and optional 'notes'
        """
        try:
            data = json.loads(response)
            if "spans" not in data:
                data["spans"] = []
            return data
        except Exception:
            return {"spans": [], "notes": "parse_error"}

    def suggest(
        self, template: str, text: str, spans: List[Dict[str, Any]], knowledge: Dict[str, Any]
    ) -> List[LLMSuggestion]:
        """Generate span suggestions using LLM.

        Complete workflow: format prompt -> call LLM -> parse response -> filter by confidence.

        Args:
            template: Prompt template string
            text: Source text being analyzed
            spans: List of candidate span dictionaries
            knowledge: Domain knowledge dictionary

        Returns:
            List of LLMSuggestion objects meeting confidence threshold

        Example:
            >>> agent = LLMAgent()
            >>> template = "Analyze: {{text}}\nCandidates: {{candidates}}"
            >>> suggestions = agent.suggest(template, "Test", [], {})
            >>> print(len(suggestions))  # 0 in stub mode
            0
        """
        prompt = self.format_prompt(template, text, spans, knowledge)
        raw = self.call(prompt)
        parsed = self.parse(raw)
        suggestions: List[LLMSuggestion] = []
        for s in parsed.get("spans", []):
            try:
                ls = LLMSuggestion(
                    start=int(s["start"]),
                    end=int(s["end"]),
                    label=str(s["label"]),
                    negated=bool(s.get("negated", False)),
                    canonical=s.get("canonical"),
                    confidence_reason=s.get("confidence_reason"),
                    llm_confidence=float(s.get("llm_confidence", self.min_conf)),
                )
                if ls.llm_confidence >= self.min_conf:
                    suggestions.append(ls)
            except Exception:
                continue
        return suggestions
