"""
Configuration Examples

Demonstrates: Customizing SpanForge behavior via AppConfig
Prerequisites: None
Runtime: <1 second
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import AppConfig
from src.weak_label import LexiconEntry, weak_label


def demo_default_config():
    """Show default configuration values."""
    print("=" * 70)
    print("1. DEFAULT CONFIGURATION")
    print("=" * 70)
    print()

    # Load default config
    config = AppConfig()

    print("Model Settings:")
    print(f"   • model_name: {config.model_name}")
    print(f"   • max_seq_len: {config.max_seq_len}")
    print(f"   • device: {config.device}")
    print()

    print("Weak Labeling Settings:")
    print(f"   • fuzzy_scorer: {config.fuzzy_scorer} (wratio or jaccard)")
    print(f"   • negation_window: ±{config.negation_window} tokens")
    print()

    print("LLM Settings:")
    print(f"   • llm_enabled: {config.llm_enabled}")
    print(f"   • llm_provider: {config.llm_provider}")
    print(f"   • llm_model: {config.llm_model}")
    print()


def demo_fuzzy_threshold():
    """Show impact of fuzzy matching threshold."""
    print("=" * 70)
    print("2. FUZZY MATCHING THRESHOLD")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="dizziness", canonical="Dizziness", source="MedDRA"),
    ]

    # Text with typo
    text = "Experiencing diziness today"

    print(f"Text: \"{text}\" (typo: 'diziness' vs 'dizziness')")
    print()

    thresholds = [85.0, 88.0, 92.0, 95.0]

    for threshold in thresholds:
        spans = weak_label(
            text=text,
            symptom_lexicon=lexicon,
            product_lexicon=[],
        )

        # Note: weak_label uses config default, but match_symptoms accepts threshold
        from src.weak_label import match_symptoms

        spans = match_symptoms(text, lexicon, fuzzy_threshold=threshold)

        if spans:
            print(f"   Threshold ≥{threshold}%: MATCHED (confidence: {spans[0].confidence:.2f})")
        else:
            print(f"   Threshold ≥{threshold}%: NO MATCH (score below threshold)")

    print()
    print("Recommendation:")
    print("   • Use 88% (default) for balanced precision/recall")
    print("   • Lower to 85% for more lenient matching (more typos)")
    print("   • Raise to 92% for stricter matching (fewer false positives)")
    print()


def demo_negation_window_tuning():
    """Show negation window size effects."""
    print("=" * 70)
    print("3. NEGATION WINDOW TUNING")
    print("=" * 70)
    print()

    lexicon = [LexiconEntry(term="headache", canonical="Headache", source="MedDRA")]

    # Long distance between negation and symptom
    text = "Patient reports absolutely no severe persistent headache"

    print(f'Text: "{text}"')
    print("Distance from 'no' to 'headache': 4 tokens")
    print()

    window_sizes = [3, 5, 7]

    for window in window_sizes:
        from src.weak_label import match_symptoms

        spans = match_symptoms(text, lexicon, negation_window=window)

        if spans and spans[0].negated:
            print(f"   Window ±{window}: NEGATED ✓")
        elif spans:
            print(f"   Window ±{window}: POSITIVE (missed negation)")
        else:
            print(f"   Window ±{window}: NO MATCH")

    print()
    print("Recommendation:")
    print("   • Use 5 (default) for short consumer complaints")
    print("   • Use 7 for longer clinical narratives")
    print("   • Use 3 for immediate negation only ('no pain')")
    print()


def demo_scorer_comparison():
    """Compare WRatio vs Jaccard scoring."""
    print("=" * 70)
    print("4. SCORER COMPARISON (wratio vs jaccard)")
    print("=" * 70)
    print()

    lexicon = [
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
    ]

    test_cases = [
        "Severe burning sensation",  # Extra adjective
        "Burning sensations",  # Plural
        "Burning skin sensation",  # Extra word in middle
    ]

    for text in test_cases:
        print(f'   Text: "{text}"')

        from src.weak_label import match_symptoms

        # WRatio (default)
        spans_wratio = match_symptoms(text, lexicon, scorer="wratio")
        if spans_wratio:
            print(f"      wratio:  MATCHED (confidence: {spans_wratio[0].confidence:.2f})")
        else:
            print(f"      wratio:  NO MATCH")

        # Jaccard
        spans_jaccard = match_symptoms(text, lexicon, scorer="jaccard")
        if spans_jaccard:
            print(f"      jaccard: MATCHED (confidence: {spans_jaccard[0].confidence:.2f})")
        else:
            print(f"      jaccard: NO MATCH")

        print()

    print("Differences:")
    print("   • wratio:  More lenient, character-based similarity")
    print("   • jaccard: Stricter, token set overlap")
    print()
    print("Recommendation:")
    print("   • Use wratio (default) for typos and spelling variations")
    print("   • Use jaccard for strict semantic overlap requirements")
    print()


def demo_device_selection():
    """Show device configuration for model loading."""
    print("=" * 70)
    print("5. DEVICE SELECTION (CPU vs GPU)")
    print("=" * 70)
    print()

    import torch

    print("Available devices:")
    print(f"   • CPU: always available")
    print(f"   • CUDA: {'✓ available' if torch.cuda.is_available() else '✗ not available'}")
    if torch.cuda.is_available():
        print(f"      GPU name: {torch.cuda.get_device_name(0)}")
        print(f"      GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load config with explicit device
    config_cpu = AppConfig(device="cpu")
    print(f"Configured device: {config_cpu.device}")
    print()

    print("Note: Device selection affects model loading, not weak labeling")
    print("Weak labeling (lexicon-based) runs on CPU regardless of device setting")
    print()
    print("When to use GPU:")
    print("   • Token classification fine-tuning (future)")
    print("   • Inference with classification head (future)")
    print("   • Embedding computation at scale")
    print()


def demo_llm_config():
    """Show LLM provider configuration."""
    print("=" * 70)
    print("6. LLM PROVIDER CONFIGURATION")
    print("=" * 70)
    print()

    print("Supported providers:")
    print("   • stub (default): No API calls, returns empty spans")
    print("   • openai: GPT-3.5/4 via OpenAI API")
    print("   • azure: Azure OpenAI service")
    print("   • anthropic: Claude models via Anthropic API")
    print()

    # Example configs for each provider
    configs = [
        ("stub", AppConfig(llm_provider="stub", llm_model="test")),
        ("openai", AppConfig(llm_provider="openai", llm_model="gpt-4")),
        ("azure", AppConfig(llm_provider="azure", llm_model="gpt-4")),
        ("anthropic", AppConfig(llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022")),
    ]

    for provider, config in configs:
        print(f"   {provider:12} → model: {config.llm_model}")

    print()
    print("Environment variables required:")
    print("   • OpenAI:    OPENAI_API_KEY")
    print("   • Azure:     AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
    print("   • Anthropic: ANTHROPIC_API_KEY")
    print()

    print("Example usage:")
    print("   import os")
    print("   os.environ['OPENAI_API_KEY'] = 'sk-...'")
    print("   config = AppConfig(llm_provider='openai', llm_model='gpt-4')")
    print()


def demo_config_from_env():
    """Show loading config from environment variables."""
    print("=" * 70)
    print("7. ENVIRONMENT VARIABLE CONFIGURATION")
    print("=" * 70)
    print()

    print("You can override config via environment variables:")
    print()

    print("Model Settings:")
    print("   export SPANFORGE_MODEL_NAME='dmis-lab/biobert-v1.1'")
    print("   export SPANFORGE_DEVICE='cuda'")
    print()

    print("Weak Labeling Settings:")
    print("   export SPANFORGE_FUZZY_SCORER='jaccard'")
    print("   export SPANFORGE_NEGATION_WINDOW='7'")
    print()

    print("LLM Settings:")
    print("   export SPANFORGE_LLM_PROVIDER='openai'")
    print("   export SPANFORGE_LLM_MODEL='gpt-4'")
    print()

    print("Note: Pydantic-settings automatically reads from environment")
    print("Prefix: All settings use 'SPANFORGE_' prefix")
    print()


def demo_custom_config_object():
    """Create custom configuration programmatically."""
    print("=" * 70)
    print("8. CUSTOM CONFIGURATION OBJECT")
    print("=" * 70)
    print()

    # Create config with custom settings
    custom_config = AppConfig(
        model_name="dmis-lab/biobert-base-cased-v1.1",
        device="cpu",
        negation_window=7,
        fuzzy_scorer="wratio",
        llm_provider="openai",
        llm_model="gpt-4-turbo",
        llm_temperature=0.1,
        llm_cache_path="data/cache/llm_responses.jsonl",
    )

    print("Custom configuration created:")
    print(f"   • model_name: {custom_config.model_name}")
    print(f"   • device: {custom_config.device}")
    print(f"   • negation_window: {custom_config.negation_window}")
    print(f"   • fuzzy_scorer: {custom_config.fuzzy_scorer}")
    print(f"   • llm_provider: {custom_config.llm_provider}")
    print(f"   • llm_model: {custom_config.llm_model}")
    print(f"   • llm_temperature: {custom_config.llm_temperature}")
    print()

    print("Note: Pass config to functions that accept it:")
    print("   # Most weak labeling functions use module-level config")
    print("   # For custom config, use functional API with explicit params")
    print()


def main():
    """Run all configuration examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "CONFIGURATION GUIDE" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_default_config()
    demo_fuzzy_threshold()
    demo_negation_window_tuning()
    demo_scorer_comparison()
    demo_device_selection()
    demo_llm_config()
    demo_config_from_env()
    demo_custom_config_object()

    print("=" * 70)
    print("QUICK REFERENCE")
    print("=" * 70)
    print()
    print("Common Tuning Scenarios:")
    print()
    print("1. More lenient matching (catch typos):")
    print("      → Lower fuzzy_threshold to 85%")
    print()
    print("2. Longer sentences with distant negation:")
    print("      → Increase negation_window to 7")
    print()
    print("3. Strict semantic overlap:")
    print("      → Use scorer='jaccard'")
    print()
    print("4. Enable LLM refinement:")
    print("      → Set llm_provider='openai' + OPENAI_API_KEY env var")
    print()
    print("5. GPU acceleration (future classification):")
    print("      → Set device='cuda' if torch.cuda.is_available()")
    print()


if __name__ == "__main__":
    main()
