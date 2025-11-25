try:
    from pydantic_settings import BaseSettings
except ImportError:  # Fallback if not yet installed; will raise when used.
    from pydantic import BaseSettings  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # Allow config load before torch installation

class AppConfig(BaseSettings):
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    max_seq_len: int = 256
    device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    seed: int = 42
    negation_window: int = 5  # tokens after negation word to mark as negated
    fuzzy_scorer: str = "wratio"  # options: wratio, jaccard
    # LLM refinement (experimental)
    llm_enabled: bool = False
    llm_provider: str = "stub"  # e.g. openai, azure, anthropic
    llm_model: str = "gpt-4"    # placeholder model name
    llm_min_confidence: float = 0.65  # discard suggestions below this
    llm_cache_path: str = "data/annotation/exports/llm_cache.jsonl"
    llm_prompt_version: str = "v1"

def get_config() -> AppConfig:
    return AppConfig()

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
