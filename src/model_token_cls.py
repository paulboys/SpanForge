"""Token classification model scaffold for BioBERT fine-tuning.

Provides factory to load `AutoModelForTokenClassification` using labels from `labels.json`.
Actual training loop lives in `scripts/train_token_cls.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.config import get_config


def load_labels(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    import json

    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValueError("labels.json must be a JSON list of strings")
    return labels


def get_token_cls_model(labels_path: Path = Path("labels.json")):
    config = get_config()
    labels = load_labels(labels_path)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(config.device)
    return tokenizer, model


__all__ = ["get_token_cls_model", "load_labels"]
