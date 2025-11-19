from typing import List, Dict, Any
import torch
from .config import AppConfig
from .model import get_model, get_tokenizer, encode_text
from pathlib import Path
from .weak_label import load_symptom_lexicon, load_product_lexicon, weak_label_batch, persist_weak_labels_jsonl


def tokenize_batch(texts: List[str], tokenizer, max_len: int):
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt"
    )


def predict_tokens(model, encodings, device: str) -> Dict[str, Any]:
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in encodings.items()})
    return {"last_hidden_state": outputs.last_hidden_state}


def postprocess_predictions(batch_tokens, model_outputs) -> List[Dict[str, Any]]:
    # Placeholder: extend with lexicon match, span extraction, normalization.
    return [{"token_count": len(tokens)} for tokens in batch_tokens]


def simple_inference(texts: List[str], persist_path: str = None) -> List[Dict[str, Any]]:
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
    wl = weak_label_batch(
        texts, symptom_lexicon, product_lexicon,
        negation_window=config.negation_window,
        scorer=config.fuzzy_scorer
    ) if (symptom_lexicon or product_lexicon) else [[] for _ in texts]
    
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
            } for s in spans
        ]
    return base
