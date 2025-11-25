from __future__ import annotations
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from .config import get_config

@dataclass
class LLMSuggestion:
    start: int
    end: int
    label: str
    negated: bool | None = None
    canonical: str | None = None
    confidence_reason: str | None = None
    llm_confidence: float | None = None

class LLMAgent:
    """Stub LLM agent; extend with real provider integration later."""
    def __init__(self):
        cfg = get_config()
        self.provider = cfg.llm_provider
        self.model = cfg.llm_model
        self.min_conf = cfg.llm_min_confidence

    def format_prompt(self, template: str, text: str, spans: List[Dict[str, Any]], knowledge: Dict[str, Any]) -> str:
        candidates = [f"{s['text']} [{s['start']},{s['end']}] {s['label']} conf={s.get('confidence',1):.2f}" for s in spans]
        return (template
                .replace("{{text}}", text)
                .replace("{{candidates}}", "\n".join(candidates))
                .replace("{{knowledge}}", json.dumps(knowledge, ensure_ascii=False)))

    def call(self, prompt: str) -> str:
        # Stub: return empty JSON structure (no suggestions) for deterministic tests
        return json.dumps({"spans": [], "notes": "stub"})

    def parse(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            if "spans" not in data:
                data["spans"] = []
            return data
        except Exception:
            return {"spans": [], "notes": "parse_error"}

    def suggest(self, template: str, text: str, spans: List[Dict[str, Any]], knowledge: Dict[str, Any]) -> List[LLMSuggestion]:
        prompt = self.format_prompt(template, text, spans, knowledge)
        raw = self.call(prompt)
        parsed = self.parse(raw)
        suggestions: List[LLMSuggestion] = []
        for s in parsed.get('spans', []):
            try:
                ls = LLMSuggestion(
                    start=int(s['start']),
                    end=int(s['end']),
                    label=str(s['label']),
                    negated=bool(s.get('negated', False)),
                    canonical=s.get('canonical'),
                    confidence_reason=s.get('confidence_reason'),
                    llm_confidence=float(s.get('llm_confidence', self.min_conf))
                )
                if ls.llm_confidence >= self.min_conf:
                    suggestions.append(ls)
            except Exception:
                continue
        return suggestions
