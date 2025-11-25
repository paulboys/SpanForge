"""LLM agent for span refinement and enhancement.

Provides stub implementation for LLM-based entity span refinement.
Designed for future integration with OpenAI, Azure, or Anthropic providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    """Stub LLM agent for entity span refinement.

    Provides interface for LLM-based span suggestion and refinement.
    Current implementation is a stub returning empty suggestions for
    deterministic testing. Extend with real provider integration later.

    Attributes:
        provider: LLM provider name ('stub', 'openai', 'azure', 'anthropic')
        model: Model identifier for the LLM
        min_conf: Minimum confidence threshold for accepting suggestions

    Example:
        >>> agent = LLMAgent()
        >>> suggestions = agent.suggest(template, text, spans, knowledge)
        >>> # Returns empty list in stub mode
    """

    def __init__(self) -> None:
        """Initialize LLM agent with configuration."""
        cfg = get_config()
        self.provider: str = cfg.llm_provider
        self.model: str = cfg.llm_model
        self.min_conf: float = cfg.llm_min_confidence

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

    def call(self, prompt: str) -> str:
        """Call LLM with formatted prompt (stub implementation).

        Args:
            prompt: Formatted prompt string

        Returns:
            JSON string with LLM response (empty in stub mode)

        Note:
            Stub returns empty suggestions for deterministic testing.
            Extend this method to integrate real LLM providers.
        """
        # Stub: return empty JSON structure (no suggestions) for deterministic tests
        return json.dumps({"spans": [], "notes": "stub"})

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
