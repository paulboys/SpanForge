from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from .config import AppConfig

_tokenizer = None
_model = None

def get_tokenizer(config: AppConfig):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return _tokenizer


def get_model(config: AppConfig):
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(config.model_name)
        _model.to(config.device)
        _model.eval()
    return _model


def encode_text(text: str, tokenizer, max_len: int) -> Tuple:
    return tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
