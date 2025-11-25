"""BioBERT model loading and text encoding utilities.

Provides cached model and tokenizer loading with GPU support.
Uses singleton pattern to avoid reloading models on repeated calls.
"""

from typing import Any, Dict, Optional, Tuple

from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .config import AppConfig

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModel] = None


def get_tokenizer(config: AppConfig) -> AutoTokenizer:
    """Get or load BioBERT tokenizer with caching.

    Uses singleton pattern to cache tokenizer after first load.

    Args:
        config: Application configuration containing model_name

    Returns:
        PreTrainedTokenizer instance for BioBERT

    Example:
        >>> config = AppConfig()
        >>> tokenizer = get_tokenizer(config)
        >>> tokens = tokenizer.tokenize("Patient has itching")
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return _tokenizer


def get_model(config: AppConfig) -> AutoModel:
    """Get or load BioBERT model with caching and device placement.

    Uses singleton pattern to cache model after first load.
    Automatically moves model to configured device (CPU/GPU).
    Sets model to evaluation mode.

    Args:
        config: Application configuration containing model_name and device

    Returns:
        PreTrainedModel instance for BioBERT in eval mode

    Example:
        >>> config = AppConfig()
        >>> model = get_model(config)
        >>> # Model is on correct device and in eval mode
    """
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(config.model_name)
        _model.to(config.device)
        _model.eval()
    return _model


def encode_text(text: str, tokenizer: AutoTokenizer, max_len: int) -> BatchEncoding:
    """Encode text string to model input format.

    Tokenizes and encodes text with truncation and tensor conversion.

    Args:
        text: Input text string to encode
        tokenizer: PreTrainedTokenizer instance
        max_len: Maximum sequence length for truncation

    Returns:
        BatchEncoding with input_ids, attention_mask, and token_type_ids

    Example:
        >>> config = AppConfig()
        >>> tokenizer = get_tokenizer(config)
        >>> encoding = encode_text("Test text", tokenizer, 256)
        >>> print(encoding.keys())
        dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    """
    return tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
