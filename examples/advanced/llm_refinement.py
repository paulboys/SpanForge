"""
LLM-Based Span Refinement

Demonstrates: Using GPT-4/Claude to improve weak label boundaries and negation
Prerequisites: pip install -e ".[llm]" + API key environment variable
Runtime: 5-10 seconds per text (with API calls)
"""

import json
import os
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm_agent import LLMAgent, LLMSuggestion
from src.weak_label import LexiconEntry, weak_label


def setup_llm_agent(provider: str = "openai", model: str = "gpt-4") -> LLMAgent:
    """Initialize LLM agent with specified provider.

    Args:
        provider: 'openai', 'azure', or 'anthropic'
        model: Model identifier (e.g., 'gpt-4', 'claude-3-5-sonnet-20241022')

    Returns:
        Configured LLMAgent instance
    """
    from unittest.mock import patch

    from src.config import AppConfig

    # Check for API key
    key_vars = {
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    key_var = key_vars.get(provider)
    if key_var and not os.getenv(key_var):
        print(f"⚠ Warning: {key_var} not set. Using stub mode (no actual API calls).")
        provider = "stub"

    config = AppConfig(
        llm_provider=provider,
        llm_model=model,
        llm_temperature=0.1,  # Low temperature for consistent extraction
        llm_cache_path="data/cache/llm_refinement.jsonl",
    )

    with patch("src.llm_agent.get_config", return_value=config):
        return LLMAgent()


def create_sample_lexicons():
    """Create symptom lexicons for demonstration."""
    return [
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
        LexiconEntry(term="redness", canonical="Erythema", source="MedDRA"),
        LexiconEntry(term="itching", canonical="Pruritus", source="MedDRA"),
        LexiconEntry(term="swelling", canonical="Edema", source="MedDRA"),
        LexiconEntry(term="rash", canonical="Rash", source="MedDRA"),
        LexiconEntry(term="dryness", canonical="Xerosis", source="MedDRA"),
    ]


def demo_boundary_correction():
    """Show LLM removing superfluous adjectives."""
    print("=" * 70)
    print("1. BOUNDARY CORRECTION")
    print("=" * 70)
    print()

    lexicon = create_sample_lexicons()

    # Weak labels often include extra words
    test_cases = [
        "I experienced severe burning sensation on my face",
        "Developed mild redness and slight itching",
        "Extreme swelling occurred after application",
    ]

    agent = setup_llm_agent(provider="openai", model="gpt-4")

    for text in test_cases:
        print(f'Text: "{text}"')
        print()

        # Get weak labels
        weak_spans = weak_label(text, lexicon, [])

        print("   WEAK LABELS (may include adjectives):")
        for span in weak_spans:
            print(f'      • "{span.text}" [{span.start}:{span.end}]')

        # Convert to dict format for LLM agent
        weak_dicts = [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "label": s.label,
                "confidence": s.confidence,
            }
            for s in weak_spans
        ]

        # LLM refinement template
        template = """Analyze medical text and refine entity spans.

Text: {{text}}

Weak labels detected: {{candidates}}

Task: Remove superfluous adjectives (severe, mild, slight, extreme) from spans.
Keep only the core medical term.

Return JSON: {"spans": [{"start": int, "end": int, "label": str, "canonical": str}]}"""

        # Get LLM suggestions
        suggestions = agent.suggest(template, text, weak_dicts, {})

        print()
        print("   LLM REFINED:")
        if suggestions:
            for sugg in suggestions:
                refined_text = text[sugg.start : sugg.end]
                print(f'      • "{refined_text}" [{sugg.start}:{sugg.end}]')
                if sugg.canonical:
                    print(f"         → {sugg.canonical}")
        else:
            print("      (stub mode - no refinement)")

        print()


def demo_negation_validation():
    """Show LLM confirming negation accuracy."""
    print("=" * 70)
    print("2. NEGATION VALIDATION")
    print("=" * 70)
    print()

    lexicon = create_sample_lexicons()

    test_cases = [
        "No burning sensation reported",
        "Patient denies any redness or swelling",
        "Without itching or discomfort",
    ]

    agent = setup_llm_agent()

    for text in test_cases:
        print(f'Text: "{text}"')
        print()

        weak_spans = weak_label(text, lexicon, [], negation_window=5)

        print("   WEAK LABELS (rule-based negation):")
        for span in weak_spans:
            negated = " [NEGATED]" if span.negated else ""
            print(f'      • "{span.text}"{negated}')

        # LLM validation
        weak_dicts = [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "label": s.label,
                "negated": s.negated,
            }
            for s in weak_spans
        ]

        template = """Validate negation detection in medical text.

Text: {{text}}
Detected entities: {{candidates}}

Task: Confirm if negation flags are correct.
Return JSON: {"spans": [{"start": int, "end": int, "label": str, "negated": bool}]}"""

        suggestions = agent.suggest(template, text, weak_dicts, {})

        print()
        print("   LLM VALIDATED:")
        if suggestions:
            for sugg in suggestions:
                negated = " [NEGATED]" if sugg.negated else ""
                print(f'      • "{text[sugg.start:sugg.end]}"{negated}')
        else:
            print("      (stub mode)")

        print()


def demo_canonical_normalization():
    """Show LLM mapping colloquial terms to medical terminology."""
    print("=" * 70)
    print("3. CANONICAL NORMALIZATION")
    print("=" * 70)
    print()

    # Colloquial language in consumer complaints
    test_cases = [
        "My skin got super dry and flaky",
        "Face turned bright red like a tomato",
        "Started peeling really bad",
    ]

    lexicon = [
        LexiconEntry(term="dry", canonical="Xerosis", source="MedDRA"),
        LexiconEntry(term="red", canonical="Erythema", source="MedDRA"),
        LexiconEntry(term="peeling", canonical="Desquamation", source="MedDRA"),
    ]

    agent = setup_llm_agent()

    for text in test_cases:
        print(f'Text: "{text}"')
        print()

        weak_spans = weak_label(text, lexicon, [])

        print("   WEAK LABELS:")
        for span in weak_spans:
            print(f'      • "{span.text}" → {span.canonical}')

        weak_dicts = [
            {"text": s.text, "start": s.start, "end": s.end, "label": s.label} for s in weak_spans
        ]

        template = """Map colloquial symptoms to medical terminology.

Text: {{text}}
Detected: {{candidates}}

Medical lexicon: {{knowledge}}

Task: Provide canonical medical term for each symptom.
Return JSON: {"spans": [{"start": int, "end": int, "label": str, "canonical": str}]}"""

        knowledge = {
            "canonical_map": {
                "dry": "Xerosis",
                "flaky": "Desquamation",
                "red": "Erythema",
                "peeling": "Desquamation",
            }
        }

        suggestions = agent.suggest(template, text, weak_dicts, knowledge)

        print()
        print("   LLM NORMALIZED:")
        if suggestions:
            for sugg in suggestions:
                print(f'      • "{text[sugg.start:sugg.end]}" → {sugg.canonical}')
        else:
            print("      (stub mode)")

        print()


def demo_confidence_reasoning():
    """Show LLM providing explanation for confidence scores."""
    print("=" * 70)
    print("4. CONFIDENCE WITH REASONING")
    print("=" * 70)
    print()

    lexicon = create_sample_lexicons()

    text = "Mild burning that quickly became severe redness"

    print(f'Text: "{text}"')
    print()

    weak_spans = weak_label(text, lexicon, [])

    print("WEAK LABELS (numeric confidence only):")
    for span in weak_spans:
        print(f'   • "{span.text}" (confidence: {span.confidence:.2f})')
    print()

    agent = setup_llm_agent()

    weak_dicts = [
        {
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "label": s.label,
            "confidence": s.confidence,
        }
        for s in weak_spans
    ]

    template = """Analyze entity confidence and provide reasoning.

Text: {{text}}
Detected: {{candidates}}

Task: Assign confidence (0-1) with textual explanation.
Return JSON: {"spans": [{"start": int, "end": int, "label": str, "llm_confidence": float, "confidence_reason": str}]}"""

    suggestions = agent.suggest(template, text, weak_dicts, {})

    print("LLM CONFIDENCE (with reasoning):")
    if suggestions:
        for sugg in suggestions:
            print(f'   • "{text[sugg.start:sugg.end]}"')
            print(f"      Confidence: {sugg.llm_confidence or 'N/A'}")
            print(f"      Reason: {sugg.confidence_reason or 'N/A'}")
    else:
        print("   (stub mode)")
    print()


def demo_multi_provider_comparison():
    """Compare results from different LLM providers."""
    print("=" * 70)
    print("5. MULTI-PROVIDER COMPARISON")
    print("=" * 70)
    print()

    text = "Severe burning sensation and mild redness"
    lexicon = create_sample_lexicons()
    weak_spans = weak_label(text, lexicon, [])

    print(f'Text: "{text}"')
    print()
    print("WEAK LABELS:")
    for span in weak_spans:
        print(f'   • "{span.text}" [{span.start}:{span.end}]')
    print()

    providers = [
        ("stub", "test"),
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
    ]

    weak_dicts = [
        {"text": s.text, "start": s.start, "end": s.end, "label": s.label} for s in weak_spans
    ]

    template = "Remove adjectives. Text: {{text}}. Candidates: {{candidates}}. Return JSON with refined spans."

    for provider, model in providers:
        print(f"{provider.upper()} ({model}):")

        agent = setup_llm_agent(provider=provider, model=model)
        suggestions = agent.suggest(template, text, weak_dicts, {})

        if suggestions:
            for sugg in suggestions:
                print(f'   • "{text[sugg.start:sugg.end]}" [{sugg.start}:{sugg.end}]')
        else:
            print("   (no suggestions or stub mode)")
        print()


def main():
    """Run all LLM refinement examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "LLM SPAN REFINEMENT" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("⚙️  Setup Requirements:")
    print("   1. Install LLM dependencies: pip install -e '.[llm]'")
    print("   2. Set API key: export OPENAI_API_KEY='sk-...'")
    print("   3. Or use Azure/Anthropic with respective env vars")
    print()
    print("Note: Examples run in stub mode if API keys not configured")
    print()

    demo_boundary_correction()
    demo_negation_validation()
    demo_canonical_normalization()
    demo_confidence_reasoning()
    demo_multi_provider_comparison()

    print("=" * 70)
    print("KEY BENEFITS OF LLM REFINEMENT")
    print("=" * 70)
    print()
    print("✓ Boundary Correction: Remove 'severe', 'mild', 'slight'")
    print("✓ Negation Validation: Confirm rule-based negation accuracy")
    print("✓ Canonical Mapping: Normalize colloquial → medical terms")
    print("✓ Confidence Reasoning: Explainable confidence scores")
    print("✓ Context Understanding: Syntax and semantic analysis")
    print()
    print("💰 Cost Considerations:")
    print("   • GPT-4: ~$0.03/1K input, ~$0.06/1K output")
    print("   • Claude 3.5: ~$0.003/1K input, ~$0.015/1K output")
    print("   • Typical span: 50-150 tokens → ~$0.001-0.01 per text")
    print()
    print("💡 Best Practices:")
    print("   • Use caching for repeated texts (automatic)")
    print("   • Start with subset for cost estimation")
    print("   • Compare weak vs LLM with evaluation metrics")
    print("   • Validate improvements justify API costs")
    print()


if __name__ == "__main__":
    main()
