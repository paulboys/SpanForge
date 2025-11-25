"""Configuration management for SpanForge.

This module provides centralized configuration using Pydantic BaseSettings,
allowing both default values and environment variable overrides.
"""

from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:  # Fallback if not yet installed; will raise when used.
    from pydantic import BaseSettings  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # Allow config load before torch installation


class AppConfig(BaseSettings):
    """Application configuration with defaults and environment override support.

    Attributes:
        model_name: HuggingFace model identifier for BioBERT base model
        max_seq_len: Maximum sequence length for tokenization
        device: Computation device ('cuda' or 'cpu'), auto-detected if available
        seed: Random seed for reproducibility across runs
        negation_window: Number of tokens after negation cue to mark as negated
        fuzzy_scorer: Fuzzy matching algorithm ('wratio' or 'jaccard')
        llm_enabled: Enable experimental LLM refinement pipeline
        llm_provider: LLM provider name ('stub', 'openai', 'azure', 'anthropic')
        llm_model: LLM model identifier
        llm_min_confidence: Minimum confidence threshold for LLM suggestions
        llm_cache_path: Path to LLM response cache file
        llm_prompt_version: Version identifier for prompt templates
    """

    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    max_seq_len: int = 256
    device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    seed: int = 42
    negation_window: int = 5
    fuzzy_scorer: str = "wratio"
    llm_enabled: bool = False
    llm_provider: str = "stub"
    llm_model: str = "gpt-4"
    llm_min_confidence: float = 0.65
    llm_cache_path: str = "data/annotation/exports/llm_cache.jsonl"
    llm_prompt_version: str = "v1"


def get_config() -> AppConfig:
    """Get application configuration instance.

    Returns:
        AppConfig instance with default or environment-overridden values

    Example:
        >>> config = get_config()
        >>> print(config.model_name)
        'dmis-lab/biobert-base-cased-v1.1'
    """
    return AppConfig()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Integer seed value for all random number generators

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
