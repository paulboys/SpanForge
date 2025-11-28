"""
Main weak labeling orchestrator with WeakLabeler class and legacy API.

Provides both class-based and function-based interfaces for backward compatibility.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import List, Optional

from src.weak_labeling.confidence import compute_confidence
from src.weak_labeling.matchers import (
    EMOJI_PATTERN,
    HAVE_RAPIDFUZZ,
    STOPWORDS,
    jaccard_token_score,
    tokenize,
)
from src.weak_labeling.negation import detect_negated_regions
from src.weak_labeling.types import LexiconEntry, Span
from src.weak_labeling.validators import ANATOMY_TOKENS, deduplicate_spans, is_anatomy_only

if HAVE_RAPIDFUZZ:
    from rapidfuzz import fuzz, process


class WeakLabeler:
    """Main orchestrator for weak labeling with lexicons.

    Provides class-based API for entity extraction with configurable
    fuzzy matching, negation detection, and confidence scoring.

    Example:
        >>> from pathlib import Path
        >>> labeler = WeakLabeler(
        ...     symptom_lexicon_path=Path("data/lexicon/symptoms.csv"),
        ...     product_lexicon_path=Path("data/lexicon/products.csv")
        ... )
        >>> text = "After using the cream, I developed burning sensation"
        >>> spans = labeler.label_text(text)
        >>> len(spans)
        2
    """

    def __init__(
        self,
        symptom_lexicon: Optional[List[LexiconEntry]] = None,
        product_lexicon: Optional[List[LexiconEntry]] = None,
        symptom_lexicon_path: Optional[Path] = None,
        product_lexicon_path: Optional[Path] = None,
        fuzzy_threshold: float = 88.0,
        max_term_words: int = 6,
        negation_window: int = 5,
        scorer: str = "wratio",
    ):
        """Initialize WeakLabeler with lexicons.

        Args:
            symptom_lexicon: Pre-loaded symptom lexicon
            product_lexicon: Pre-loaded product lexicon
            symptom_lexicon_path: Path to symptom CSV (if lexicon not provided)
            product_lexicon_path: Path to product CSV (if lexicon not provided)
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            max_term_words: Maximum words per candidate phrase
            negation_window: Token window for bidirectional negation
            scorer: Fuzzy scoring method ('wratio' or 'jaccard')
        """
        from src.weak_labeling.types import load_product_lexicon, load_symptom_lexicon

        self.symptom_lexicon = symptom_lexicon or (
            load_symptom_lexicon(symptom_lexicon_path) if symptom_lexicon_path else []
        )
        self.product_lexicon = product_lexicon or (
            load_product_lexicon(product_lexicon_path) if product_lexicon_path else []
        )
        self.fuzzy_threshold = fuzzy_threshold
        self.max_term_words = max_term_words
        self.negation_window = negation_window
        self.scorer = scorer

    def label_text(self, text: str) -> List[Span]:
        """Label a single text with symptoms and products.

        Args:
            text: Input text to label

        Returns:
            List of Span objects with entity mentions
        """
        symptom_spans = match_symptoms(
            text,
            self.symptom_lexicon,
            fuzzy_threshold=self.fuzzy_threshold,
            max_term_words=self.max_term_words,
            negation_window=self.negation_window,
            scorer=self.scorer,
        )
        product_spans = match_products(
            text,
            self.product_lexicon,
            fuzzy_threshold=self.fuzzy_threshold,
            max_term_words=self.max_term_words,
            scorer=self.scorer,
        )
        return assemble_spans(symptom_spans, product_spans)

    def label_batch(self, texts: List[str]) -> List[List[Span]]:
        """Label multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of span lists, one per text
        """
        return [self.label_text(text) for text in texts]


def _match_entities(
    text: str,
    lexicon: List[LexiconEntry],
    label: str,
    fuzzy_threshold: float = 88.0,
    max_term_words: int = 6,
    apply_negation: bool = False,
    negation_window: int = 5,
    scorer: str = "wratio",
) -> List[Span]:
    """Core entity matching logic with fuzzy + exact matching.

    Implements two-pass strategy:
    1. Exact phrase matching with word boundaries
    2. Fuzzy sliding window with candidate filtering

    Args:
        text: Input text to extract entities from
        lexicon: List of lexicon entries to match against
        label: Entity label to assign ("SYMPTOM" or "PRODUCT")
        fuzzy_threshold: Minimum fuzzy match score (0-100), default 88.0
        max_term_words: Maximum words in multi-word terms, default 6
        apply_negation: Whether to detect and flag negated entities
        negation_window: Bidirectional window for negation cues
        scorer: Fuzzy scorer ("wratio" or "jaccard")

    Returns:
        List of Span objects with entity mentions

    Example:
        >>> from src.weak_labeling.types import load_symptom_lexicon
        >>> from pathlib import Path
        >>> symptom_lex = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
        >>> spans = _match_entities(
        ...     "I have burning sensation",
        ...     symptom_lex,
        ...     "SYMPTOM",
        ...     apply_negation=True
        ... )
        >>> len(spans) > 0
        True
    """
    if not lexicon:
        return []

    # Clean emojis for better matching if applying negation (symptoms only)
    text_for_matching = EMOJI_PATTERN.sub(" ", text) if apply_negation else text

    tokens = tokenize(text_for_matching)
    neg_regions = detect_negated_regions(text, window=negation_window) if apply_negation else []
    spans: List[Span] = []

    # Build list/map of lexicon terms + metadata for rapid fuzzy searching
    term_map = {e.term.lower(): e for e in lexicon}
    term_meta = []
    for e in lexicon:
        t = e.term.lower()
        toks = t.split()
        term_meta.append(
            {
                "term": t,
                "tok_len": len(toks),
                "first": toks[0] if toks else "",
            }
        )
    term_list = [m["term"] for m in term_meta]
    max_tok_len = max((m["tok_len"] for m in term_meta), default=0)

    # Exact phrase matching (greedy up to max_term_words)
    lower_text = text_for_matching.lower()
    for entry in lexicon:
        term = entry.term.lower()
        idx = 0
        while True:
            idx = lower_text.find(term, idx)
            if idx == -1:
                break
            end_idx = idx + len(term)
            before_ok = idx == 0 or not lower_text[idx - 1].isalnum()
            after_ok = end_idx == len(lower_text) or not lower_text[end_idx].isalnum()
            if before_ok and after_ok:
                # Check negation status if applicable
                is_negated_flag = False
                if apply_negation:
                    for neg_start, neg_end in neg_regions:
                        if not (end_idx <= neg_start or idx >= neg_end):
                            is_negated_flag = True
                            break

                spans.append(
                    Span(
                        text=text[idx:end_idx],
                        start=idx,
                        end=end_idx,
                        label=label,
                        canonical=entry.canonical,
                        source=entry.source,
                        concept_id=entry.concept_id if label == "SYMPTOM" else None,
                        sku=entry.sku if label == "PRODUCT" else None,
                        category=entry.category if label == "PRODUCT" else None,
                        confidence=1.0,
                        negated=is_negated_flag if apply_negation else False,
                    )
                )
            idx = end_idx

    # Fuzzy matching (sliding window)
    for window_size in range(1, max_term_words + 1):
        for i in range(0, len(tokens) - window_size + 1):
            window_tokens = tokens[i : i + window_size]
            phrase_tokens = [t[0].lower() for t in window_tokens]
            phrase_clean = " ".join(phrase_tokens)

            # Skip if already exact match
            if phrase_clean in term_map:
                continue

            # Basic filters
            if len(phrase_clean) < 3:
                continue
            if window_size == 1 and is_anatomy_only(phrase_clean):
                continue
            if not any(tok not in STOPWORDS for tok in phrase_tokens):
                continue
            if not any(len(tok) > 3 for tok in phrase_tokens if tok not in STOPWORDS):
                continue

            span_text = text[window_tokens[0][1] : window_tokens[-1][2]]
            if any(p in span_text for p in [",", ";", "."]):
                continue

            phrase_tok_len = len(phrase_tokens)

            # Candidate filtering: first token match + similar length
            candidates = [
                m["term"]
                for m in term_meta
                if m["first"] == phrase_tokens[0] and abs(m["tok_len"] - phrase_tok_len) <= 1
            ]
            if not candidates:
                continue

            # Jaccard token-set filter
            content_phrase = {tok for tok in phrase_tokens if tok not in STOPWORDS}
            filtered_candidates = []
            for c in candidates:
                c_tokens = c.split()
                content_c = {ct for ct in c_tokens if ct not in STOPWORDS}
                overlap = content_phrase & content_c
                if not overlap:
                    continue
                min_size = min(len(content_phrase), len(content_c)) or 1
                if len(overlap) / min_size < 0.5:
                    continue
                # Last-token alignment filter
                if phrase_tok_len > 1 and c_tokens[-1] != phrase_tokens[-1]:
                    continue
                filtered_candidates.append(c)

            if not filtered_candidates:
                continue

            # Fuzzy matching with RapidFuzz or difflib fallback
            cutoff = fuzzy_threshold
            if HAVE_RAPIDFUZZ:
                if scorer == "jaccard":
                    best = process.extractOne(
                        phrase_clean,
                        filtered_candidates,
                        scorer=lambda a, b: jaccard_token_score(a, b),
                        score_cutoff=cutoff,
                    )
                else:
                    best = process.extractOne(
                        phrase_clean, filtered_candidates, scorer=fuzz.WRatio, score_cutoff=cutoff
                    )
                if not best:
                    continue
                matched_term, score, _ = best
            else:
                matched_term = None
                score = 0.0
                for candidate in filtered_candidates:
                    if scorer == "jaccard":
                        ratio = jaccard_token_score(phrase_clean, candidate)
                    else:
                        ratio = difflib.SequenceMatcher(None, phrase_clean, candidate).ratio() * 100
                    if ratio >= cutoff and ratio > score:
                        matched_term = candidate
                        score = ratio
                if matched_term is None:
                    continue

            # Jaccard gate
            jaccard = jaccard_token_score(phrase_clean, matched_term)
            if jaccard < 40.0:
                continue

            entry = term_map[matched_term]
            start = window_tokens[0][1]
            end = window_tokens[-1][2]

            # Check negation status if applicable
            is_negated_flag = False
            if apply_negation:
                for neg_start, neg_end in neg_regions:
                    if not (end <= neg_start or start >= neg_end):
                        is_negated_flag = True
                        break

            spans.append(
                Span(
                    text=text[start:end],
                    start=start,
                    end=end,
                    label=label,
                    canonical=entry.canonical,
                    source=entry.source,
                    concept_id=entry.concept_id if label == "SYMPTOM" else None,
                    sku=entry.sku if label == "PRODUCT" else None,
                    category=entry.category if label == "PRODUCT" else None,
                    confidence=compute_confidence(score, jaccard),
                    negated=is_negated_flag if apply_negation else False,
                )
            )

    # Deduplicate exact duplicate spans, preserve overlapping contextual mentions
    return deduplicate_spans(spans)


def match_symptoms(
    text: str,
    lexicon: List[LexiconEntry],
    fuzzy_threshold: float = 88.0,
    max_term_words: int = 6,
    negation_window: int = 5,
    scorer: str = "wratio",
) -> List[Span]:
    """Match symptom entities with negation detection.

    Wrapper around _match_entities() with negation enabled.

    Args:
        text: Input text to extract symptoms from.
        lexicon: List of symptom lexicon entries.
        fuzzy_threshold: Minimum fuzzy match score (0-100).
        max_term_words: Maximum tokens per candidate phrase.
        negation_window: Token window for bidirectional negation.
        scorer: Fuzzy scoring method ('wratio' or 'jaccard').

    Returns:
        List of symptom spans with negation flags.
    """
    return _match_entities(
        text=text,
        lexicon=lexicon,
        label="SYMPTOM",
        fuzzy_threshold=fuzzy_threshold,
        max_term_words=max_term_words,
        apply_negation=True,
        negation_window=negation_window,
        scorer=scorer,
    )


def match_products(
    text: str,
    lexicon: List[LexiconEntry],
    fuzzy_threshold: float = 88.0,
    max_term_words: int = 6,
    scorer: str = "wratio",
) -> List[Span]:
    """Match product entities without negation detection.

    Wrapper around _match_entities() with negation disabled.

    Args:
        text: Input text to extract products from.
        lexicon: List of product lexicon entries.
        fuzzy_threshold: Minimum fuzzy match score (0-100).
        max_term_words: Maximum tokens per candidate phrase.
        scorer: Fuzzy scoring method ('wratio' or 'jaccard').

    Returns:
        List of product spans.
    """
    return _match_entities(
        text=text,
        lexicon=lexicon,
        label="PRODUCT",
        fuzzy_threshold=fuzzy_threshold,
        max_term_words=max_term_words,
        apply_negation=False,
        scorer=scorer,
    )


def assemble_spans(symptom_spans: List[Span], product_spans: List[Span]) -> List[Span]:
    """Combine symptom and product spans into single list.

    Args:
        symptom_spans: List of symptom spans
        product_spans: List of product spans

    Returns:
        Combined list of spans
    """
    return symptom_spans + product_spans


def weak_label(
    text: str,
    symptom_lexicon: List[LexiconEntry],
    product_lexicon: List[LexiconEntry],
    negation_window: int = 5,
    scorer: str = "wratio",
) -> List[Span]:
    """Label text with both symptoms and products (legacy API).

    Args:
        text: Input text to label
        symptom_lexicon: Symptom lexicon entries
        product_lexicon: Product lexicon entries
        negation_window: Token window for negation
        scorer: Fuzzy scoring method

    Returns:
        List of all entity spans
    """
    symptom_spans = match_symptoms(
        text, symptom_lexicon, negation_window=negation_window, scorer=scorer
    )
    product_spans = match_products(text, product_lexicon, scorer=scorer)
    return assemble_spans(symptom_spans, product_spans)


def weak_label_batch(
    texts: List[str],
    symptom_lexicon: List[LexiconEntry],
    product_lexicon: List[LexiconEntry],
    negation_window: int = 5,
    scorer: str = "wratio",
) -> List[List[Span]]:
    """Label multiple texts (legacy API).

    Args:
        texts: List of input texts
        symptom_lexicon: Symptom lexicon entries
        product_lexicon: Product lexicon entries
        negation_window: Token window for negation
        scorer: Fuzzy scoring method

    Returns:
        List of span lists, one per text
    """
    return [weak_label(t, symptom_lexicon, product_lexicon, negation_window, scorer) for t in texts]


def persist_weak_labels_jsonl(
    texts: List[str], spans_batch: List[List[Span]], output_path: Path
) -> None:
    """Persist weak labels to JSONL for annotation triage.

    Args:
        texts: List of input texts
        spans_batch: List of span lists (one per text)
        output_path: Path to output JSONL file
    """
    with output_path.open("w", encoding="utf-8") as f:
        for text, spans in zip(texts, spans_batch):
            record = {
                "text": text,
                "spans": [
                    {
                        "text": s.text,
                        "start": s.start,
                        "end": s.end,
                        "label": s.label,
                        "canonical": s.canonical,
                        "source": s.source,
                        "concept_id": s.concept_id,
                        "sku": s.sku,
                        "category": s.category,
                        "confidence": s.confidence,
                        "negated": s.negated,
                    }
                    for s in spans
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
