"""End-to-end inference pipeline combining BioBERT and weak labeling.

Provides batch processing with model inference, weak labeling, and
optional persistence to JSONL format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, BatchEncoding

from .config import AppConfig
from .model import encode_text, get_model, get_tokenizer
from .weak_labeling import (
    load_product_lexicon,
    load_symptom_lexicon,
    persist_weak_labels_jsonl,
    weak_label_batch,
)


def tokenize_batch(texts: List[str], tokenizer: AutoTokenizer, max_len: int) -> BatchEncoding:
    """Tokenize batch of texts for model input.

    Args:
        texts: List of input text strings
        tokenizer: PreTrainedTokenizer instance
        max_len: Maximum sequence length for truncation

    Returns:
        BatchEncoding with padded and truncated sequences
    """
    return tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def predict_tokens(model: Any, encodings: BatchEncoding, device: str) -> Dict[str, Any]:
    """Run model inference on encoded batch.

    Args:
        model: PreTrainedModel instance
        encodings: BatchEncoding from tokenizer
        device: Device string ('cuda' or 'cpu')

    Returns:
        Dictionary containing last_hidden_state from model output
    """
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in encodings.items()})
    return {"last_hidden_state": outputs.last_hidden_state}


def postprocess_predictions(
    batch_tokens: List[List[str]], model_outputs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Postprocess model outputs (placeholder for future expansion).

    Args:
        batch_tokens: List of tokenized texts
        model_outputs: Dictionary containing model predictions

    Returns:
        List of dictionaries with basic token count

    Note:
        This is a placeholder for future lexicon-based span extraction.
    """
    # Placeholder: extend with lexicon match, span extraction, normalization.
    return [{"token_count": len(tokens)} for tokens in batch_tokens]


def simple_inference(texts: List[str], persist_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run end-to-end inference pipeline with weak labeling.

    Combines BioBERT inference with lexicon-based weak labeling.
    Optionally persists results to JSONL format.

    Args:
        texts: List of input text strings to analyze
        persist_path: Optional path to save weak labels in JSONL format

    Returns:
        List of dictionaries containing:
            - token_count: Number of tokens in text
            - weak_spans: Detected symptom/product spans with metadata

    Example:
        Running on a single text returns a record where the spans field
        contains detected entities and their positions, labels, canonical
        forms, confidence, and negation flags.
    """
    config = AppConfig()
    tokenizer = get_tokenizer(config)
    model = get_model(config)
    encodings = tokenize_batch(texts, tokenizer, config.max_seq_len)
    outputs = predict_tokens(model, encodings, config.device)
    batch_tokens = [tokenizer.tokenize(t) for t in texts]
    base = postprocess_predictions(batch_tokens, outputs)

    # Load lexicons
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)

    # Weak labeling with config params
    wl = (
        weak_label_batch(
            texts,
            symptom_lexicon,
            product_lexicon,
            negation_window=config.negation_window,
            scorer=config.fuzzy_scorer,
        )
        if (symptom_lexicon or product_lexicon)
        else [[] for _ in texts]
    )

    # Persist if requested
    if persist_path:
        persist_weak_labels_jsonl(texts, wl, Path(persist_path))

    # Merge
    for rec, spans in zip(base, wl):
        rec["weak_spans"] = [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "label": s.label,
                "canonical": s.canonical,
                "sku": s.sku,
                "category": s.category,
                "confidence": s.confidence,
                "negated": s.negated,
            }
            for s in spans
        ]
    return base
