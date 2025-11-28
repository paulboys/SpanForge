"""
Weak Labeling Package for Biomedical NER

This package provides modular components for lexicon-based entity extraction
with fuzzy matching, negation detection, and confidence scoring.

Modules:
    - matchers: Fuzzy and exact string matching algorithms
    - negation: Bidirectional negation detection
    - confidence: Confidence score computation
    - validators: Span validation and filtering rules
    - labeler: Main orchestrator (WeakLabeler class)
    - types: Shared data types (LexiconEntry, Span)

Quick Start:
    >>> from src.weak_labeling import WeakLabeler
    >>> labeler = WeakLabeler()
    >>> spans = labeler.label_text("Patient has burning sensation", symptom_lexicon)

See Also:
    - Original module: src/weak_label.py (deprecated, maintained for compatibility)
    - User Guide: docs/user-guide/weak-labeling.md
    - API Reference: docs/api/weak_labeling.md
"""

from __future__ import annotations

from src.weak_labeling.confidence import (
    adjust_confidence_for_negation,
    align_spans,
    calibrate_threshold,
    compute_confidence,
)

# Import utility functions for backward compatibility
# Import main labeler
from src.weak_labeling.labeler import (
    WeakLabeler,
    assemble_spans,
    match_products,
    match_symptoms,
    persist_weak_labels_jsonl,
    weak_label,
    weak_label_batch,
)

# Import core functions for advanced usage
from src.weak_labeling.matchers import (
    exact_match,
    fuzzy_match,
    jaccard_token_score,
    tokenize,
    tokenize_clean,
)
from src.weak_labeling.negation import detect_negated_regions, get_negation_cues, is_negated

# Import lexicon loaders
# Import data types
from src.weak_labeling.types import LexiconEntry, Span, load_product_lexicon, load_symptom_lexicon
from src.weak_labeling.validators import (
    deduplicate_spans,
    filter_overlapping_spans,
    get_anatomy_tokens,
    is_anatomy_only,
    validate_span_alignment,
)

__all__ = [
    # Types
    "LexiconEntry",
    "Span",
    # Main API
    "WeakLabeler",
    # Backward compatibility functions
    "weak_label",
    "weak_label_batch",
    "match_symptoms",
    "match_products",
    "assemble_spans",
    "persist_weak_labels_jsonl",
    # Lexicon loaders
    "load_symptom_lexicon",
    "load_product_lexicon",
    # Matchers
    "fuzzy_match",
    "exact_match",
    "jaccard_token_score",
    "tokenize",
    "tokenize_clean",
    # Negation
    "detect_negated_regions",
    "is_negated",
    "get_negation_cues",
    # Confidence
    "compute_confidence",
    "align_spans",
    "adjust_confidence_for_negation",
    "calibrate_threshold",
    # Validators
    "is_anatomy_only",
    "deduplicate_spans",
    "validate_span_alignment",
    "filter_overlapping_spans",
    "get_anatomy_tokens",
]
