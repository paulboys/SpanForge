"""
Weak labeling for biomedical named entity recognition.

**DEPRECATION NOTICE**: This module has been refactored into `src.weak_labeling` package.
For new code, please use: `from src.weak_labeling import weak_label`
This compatibility module will be maintained through version 0.2.x but may be
removed in version 0.3.0.

This module implements lexicon-based fuzzy matching with rule-based filters
for automated span extraction. Suitable for bootstrapping annotation, active
learning, and evaluation baselines.

Key Features:
    - Fuzzy matching with RapidFuzz (WRatio ≥88, Jaccard ≥40)
    - Bidirectional negation detection (forward + backward windows)
    - Last-token alignment filter (prevents partial-word matches)
    - Anatomy singleton filter (rejects generic anatomy mentions)
    - Emoji and unicode handling (robust multi-byte support)
    - Confidence scoring (0.8×fuzzy + 0.2×jaccard)

Typical Usage:
    Load a symptom lexicon, match entities in text, and inspect the
    first span's text, label, and confidence.

See Also:
    - User Guide: docs/user-guide/weak-labeling.md
    - Negation Guide: docs/user-guide/negation.md
    - API Reference: docs/api/weak_label.md
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process  # type: ignore

    HAVE_RAPIDFUZZ = True
except ImportError:  # Fallback to difflib if rapidfuzz unavailable at runtime
    import difflib

    HAVE_RAPIDFUZZ = False

# Expanded negation cue list (Phase 3)
NEGATION_TOKENS = {
    "no",
    "not",
    "without",
    "never",
    "none",
    "n't",
    "absent",
    "denies",
    "denied",
    "deny",
    "negative",
    "unremarkable",
    "resolv",
    "cleared",
    "improved",
}
WORD_PATTERN = re.compile(r"\b\w[\w\-']*\b")
# Emoji pattern for detection (Phase 3 - improved unicode handling)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
    "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa00-\U0001fa6f"  # Chess Symbols
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)
STOPWORDS = {
    "i",
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "but",
    "so",
    "with",
    "after",
    "before",
    "from",
    "my",
    "me",
    "we",
    "our",
    "this",
    "that",
    "it",
    "is",
    "was",
    "are",
    "were",
}
ANATOMY_TOKENS = {
    "face",
    "skin",
    "arm",
    "arms",
    "leg",
    "legs",
    "hand",
    "hands",
    "eye",
    "eyes",
    "ear",
    "ears",
    "nose",
    "mouth",
    "lip",
    "lips",
    "cheek",
    "cheeks",
    "forehead",
    "scalp",
    "neck",
    "back",
}


@dataclass
class LexiconEntry:
    term: str
    canonical: str
    source: str
    concept_id: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None


@dataclass
class Span:
    text: str
    start: int  # character start
    end: int  # character end (exclusive)
    label: str
    canonical: Optional[str] = None
    source: Optional[str] = None
    concept_id: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 1.0
    negated: bool = False


def load_symptom_lexicon(path: Path) -> List[LexiconEntry]:
    if not path.exists():
        return []
    entries: List[LexiconEntry] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term_raw = row.get("term", "").strip()
            if not term_raw:
                continue  # skip empty term rows
            canonical_raw = row.get("canonical", "").strip() or term_raw
            entries.append(
                LexiconEntry(
                    term=term_raw,
                    canonical=canonical_raw,
                    source=row.get("source", "").strip() or "unknown",
                    concept_id=row.get("concept_id", "").strip() or None,
                )
            )
    return entries


def load_product_lexicon(path: Path) -> List[LexiconEntry]:
    if not path.exists():
        return []
    entries: List[LexiconEntry] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term_raw = row.get("term", "").strip()
            if not term_raw:
                continue
            canonical_raw = row.get("canonical", "").strip() or term_raw
            entries.append(
                LexiconEntry(
                    term=term_raw,
                    canonical=canonical_raw,
                    source="product_lexicon",
                    sku=row.get("sku", "").strip() or None,
                    category=row.get("category", "").strip() or None,
                )
            )
    return entries


def _tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text and return (token, start, end) tuples using original positions."""
    tokens = []
    for m in WORD_PATTERN.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def _tokenize_clean(text: str) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Phase 3: Return cleaned text AND tokens with adjusted positions.

    Returns:
        (cleaned_text, tokens) where tokens have positions relative to cleaned text
    """
    # Build mapping from original to cleaned positions
    cleaned_text = EMOJI_PATTERN.sub(" ", text)
    tokens = []
    for m in WORD_PATTERN.finditer(cleaned_text):
        tokens.append((m.group(0), m.start(), m.end()))
    return cleaned_text, tokens


def _jaccard_token_score(a: str, b: str) -> float:
    """Jaccard token-set similarity (0-100 scale)."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return (len(intersection) / len(union)) * 100.0


def detect_negated_regions(text: str, window: int = 5) -> List[Tuple[int, int]]:
    """Phase 3: Enhanced negation detection with forward/backward windows and prefix matching.

    Forward window: negation precedes symptom (e.g., "no itching")
    Backward window: negation follows symptom (e.g., "itching absent")
    """
    tokens = _tokenize(text)
    neg_spans: List[Tuple[int, int]] = []
    for i, (tok, s, e) in enumerate(tokens):
        tok_lower = tok.lower()
        # Check exact match or prefix match for flexible negation detection
        is_negation = tok_lower in NEGATION_TOKENS or any(
            tok_lower.startswith(neg) for neg in NEGATION_TOKENS if len(neg) > 3
        )
        if is_negation:
            # Forward window: next N tokens
            window_tokens = tokens[i + 1 : i + 1 + window]
            if window_tokens:
                neg_start = window_tokens[0][1]
                neg_end = window_tokens[-1][2]
                neg_spans.append((neg_start, neg_end))

            # Backward window: previous N tokens (Phase 3: handle "symptom absent" pattern)
            back_tokens = tokens[max(0, i - window) : i]
            if back_tokens:
                back_start = back_tokens[0][1]
                back_end = back_tokens[-1][2]
                neg_spans.append((back_start, back_end))
    return neg_spans


def _is_negated(span_start: int, span_end: int, neg_regions: List[Tuple[int, int]]) -> bool:
    for ns, ne in neg_regions:
        # If span overlaps a negated region significantly (>=50% of span) mark negated
        overlap = max(0, min(span_end, ne) - max(span_start, ns))
        if overlap >= 0.5 * (span_end - span_start):
            return True
    return False


def _deduplicate_spans(spans: List[Span]) -> List[Span]:
    """Remove exact duplicate spans (same start, end, canonical), keep highest confidence.

    Preserves overlapping spans with different boundaries (e.g., 'rash' vs 'little rash')
    as they represent distinct contextual mentions, not redundancy.
    """
    if not spans:
        return []

    # Group spans by (start, end, canonical) tuple
    from collections import defaultdict

    groups: Dict[Tuple[int, int, Optional[str]], List[Span]] = defaultdict(list)
    for span in spans:
        key = (span.start, span.end, span.canonical)
        groups[key].append(span)

    # Keep highest confidence span from each group
    deduplicated = []
    for group_spans in groups.values():
        best = max(group_spans, key=lambda s: s.confidence)
        deduplicated.append(best)

    # Sort by start position for consistent output
    deduplicated.sort(key=lambda s: s.start)
    return deduplicated


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
    """Core entity matching logic shared by match_symptoms() and match_products().

    Performs lexicon-based entity extraction with fuzzy matching and optional negation detection.

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
        >>> symptom_lex = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
        >>> spans = _match_entities(text, symptom_lex, "SYMPTOM", apply_negation=True)
    """
    if not lexicon:
        return []

    # Phase 3: Clean emojis for better matching if applying negation (symptoms only)
    text_for_matching = EMOJI_PATTERN.sub(" ", text) if apply_negation else text

    tokens = _tokenize(text_for_matching)
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
    # Use cleaned text for matching
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
                is_negated = False
                if apply_negation:
                    for neg_start, neg_end in neg_regions:
                        if not (end_idx <= neg_start or idx >= neg_end):
                            is_negated = True
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
                        negated=is_negated if apply_negation else False,
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
            if window_size == 1 and phrase_tokens[0] in ANATOMY_TOKENS:
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
                        scorer=lambda a, b: _jaccard_token_score(a, b),
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
                        ratio = _jaccard_token_score(phrase_clean, candidate)
                    else:
                        ratio = difflib.SequenceMatcher(None, phrase_clean, candidate).ratio() * 100
                    if ratio >= cutoff and ratio > score:
                        matched_term = candidate
                        score = ratio
                if matched_term is None:
                    continue

            # Jaccard gate
            jaccard = _jaccard_token_score(phrase_clean, matched_term)
            if jaccard < 40.0:
                continue

            entry = term_map[matched_term]
            start = window_tokens[0][1]
            end = window_tokens[-1][2]

            # Check negation status if applicable
            is_negated = False
            if apply_negation:
                for neg_start, neg_end in neg_regions:
                    if not (end <= neg_start or start >= neg_end):
                        is_negated = True
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
                    confidence=min(1.0, (score / 100.0) * 0.8 + (jaccard / 100.0) * 0.2),
                    negated=is_negated if apply_negation else False,
                )
            )

    # Deduplicate exact duplicate spans, preserve overlapping contextual mentions
    return _deduplicate_spans(spans)


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
    return symptom_spans + product_spans


def weak_label(
    text: str,
    symptom_lexicon: List[LexiconEntry],
    product_lexicon: List[LexiconEntry],
    negation_window: int = 5,
    scorer: str = "wratio",
) -> List[Span]:
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
    return [weak_label(t, symptom_lexicon, product_lexicon, negation_window, scorer) for t in texts]


def persist_weak_labels_jsonl(
    texts: List[str], spans_batch: List[List[Span]], output_path: Path
) -> None:
    """Persist weak labels to JSONL for annotation triage."""
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
