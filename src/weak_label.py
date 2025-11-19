from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv
import re
import json
from pathlib import Path
try:
    from rapidfuzz import fuzz, process  # type: ignore
    HAVE_RAPIDFUZZ = True
except ImportError:  # Fallback to difflib if rapidfuzz unavailable at runtime
    import difflib
    HAVE_RAPIDFUZZ = False

NEGATION_TOKENS = {"no", "not", "without", "never", "none", "n't"}
WORD_PATTERN = re.compile(r"\b\w[\w\-']*\b")
STOPWORDS = {
    "i","a","an","the","and","or","to","for","of","in","on","at","but","so","with","after","before","from","my","me","we","our","this","that","it","is","was","are","were"
}
ANATOMY_TOKENS = {
    "face","skin","arm","arms","leg","legs","hand","hands","eye","eyes","ear","ears","nose","mouth","lip","lips","cheek","cheeks","forehead","scalp","neck","back"
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
    end: int    # character end (exclusive)
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
    tokens = []
    for m in WORD_PATTERN.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


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
    tokens = _tokenize(text)
    neg_spans: List[Tuple[int, int]] = []
    for i, (tok, s, e) in enumerate(tokens):
        if tok.lower() in NEGATION_TOKENS:
            # Negation window: next N tokens
            window_tokens = tokens[i + 1 : i + 1 + window]
            if window_tokens:
                neg_start = window_tokens[0][1]
                neg_end = window_tokens[-1][2]
                neg_spans.append((neg_start, neg_end))
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


def match_symptoms(text: str, lexicon: List[LexiconEntry],
                   fuzzy_threshold: float = 88.0,
                   max_term_words: int = 6,
                   negation_window: int = 5,
                   scorer: str = "wratio") -> List[Span]:
    if not lexicon:
        return []
    tokens = _tokenize(text)
    neg_regions = detect_negated_regions(text, window=negation_window)
    spans: List[Span] = []

    # Build list/map of lexicon terms + metadata for rapid fuzzy searching
    term_map = {e.term.lower(): e for e in lexicon}
    term_meta = []
    for e in lexicon:
        t = e.term.lower()
        toks = t.split()
        term_meta.append({
            "term": t,
            "tok_len": len(toks),
            "first": toks[0] if toks else "",
        })
    term_list = [m["term"] for m in term_meta]
    max_tok_len = max((m["tok_len"] for m in term_meta), default=0)

    # Exact phrase matching (greedy up to max_term_words)
    lower_text = text.lower()
    for entry in lexicon:
        term = entry.term.lower()
        idx = 0
        while True:
            idx = lower_text.find(term, idx)
            if idx == -1:
                break
            end_idx = idx + len(term)
            # Boundaries: ensure not partial word
            before_ok = idx == 0 or not lower_text[idx - 1].isalnum()
            after_ok = end_idx == len(lower_text) or not lower_text[end_idx].isalnum()
            if before_ok and after_ok:
                negated = _is_negated(idx, end_idx, neg_regions)
                spans.append(Span(
                    text=text[idx:end_idx], start=idx, end=end_idx, label="SYMPTOM",
                    canonical=entry.canonical, source=entry.source, concept_id=entry.concept_id,
                    confidence=1.0, negated=negated
                ))
            idx = end_idx

    # Fuzzy single / multi-word matching by sliding window
    # Construct candidate windows up to max_term_words
    for window_size in range(1, max_term_words + 1):
        for i in range(0, len(tokens) - window_size + 1):
            window_tokens = tokens[i:i+window_size]
            phrase_tokens = [t[0].lower() for t in window_tokens]
            phrase_clean = " ".join(phrase_tokens)
            if phrase_clean in term_map:
                continue  # already captured exact
            if len(phrase_clean) < 3:
                continue  # too short for fuzzy
            # Skip pure anatomy single tokens to avoid generic matches
            if window_size == 1 and phrase_tokens[0] in ANATOMY_TOKENS:
                continue
            # Require at least one non-stopword token
            if not any(tok not in STOPWORDS for tok in phrase_tokens):
                continue
            # Require at least one token length > 3 for semantic signal
            if not any(len(tok) > 3 for tok in phrase_tokens if tok not in STOPWORDS):
                continue
            # Skip phrases spanning punctuation boundaries (comma/period/semicolon) to reduce drift
            span_text = text[window_tokens[0][1]:window_tokens[-1][2]]
            if any(p in span_text for p in [',',';','.']):
                continue
            phrase_tok_len = len(phrase_tokens)
            # Candidates must share first token and have similar length
            candidates = [m["term"] for m in term_meta
                          if m["first"] == phrase_tokens[0]
                          and abs(m["tok_len"] - phrase_tok_len) <= 1]
            if not candidates:
                continue
            # Ensure token overlap (non-stopword) with candidate before scoring
            content_phrase = {tok for tok in phrase_tokens if tok not in STOPWORDS}
            filtered_candidates = []
            for c in candidates:
                c_tokens = c.split()
                content_c = {ct for ct in c_tokens if ct not in STOPWORDS}
                overlap = content_phrase & content_c
                if not overlap:
                    continue
                # Require at least 50% overlap relative to smaller set
                min_size = min(len(content_phrase), len(content_c)) or 1
                if len(overlap) / min_size < 0.5:
                    continue
                # Require last token alignment for multi-token phrases
                if phrase_tok_len > 1 and c_tokens[-1] != phrase_tokens[-1]:
                    continue
                filtered_candidates.append(c)
            if not filtered_candidates:
                continue
            # Use selected scorer without dynamic threshold relaxation
            cutoff = fuzzy_threshold
            if HAVE_RAPIDFUZZ:
                if scorer == "jaccard":
                    best = process.extractOne(
                        phrase_clean, filtered_candidates,
                        scorer=lambda a, b: _jaccard_token_score(a, b),
                        score_cutoff=cutoff
                    )
                else:
                    best = process.extractOne(
                        phrase_clean, filtered_candidates,
                        scorer=fuzz.WRatio,
                        score_cutoff=cutoff
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
            # Jaccard gate (token-set) to enforce semantic overlap
            jaccard = _jaccard_token_score(phrase_clean, matched_term)
            if jaccard < 40.0:
                continue
            entry = term_map[matched_term]
            start = window_tokens[0][1]
            end = window_tokens[-1][2]
            negated = _is_negated(start, end, neg_regions)
            spans.append(Span(
                text=text[start:end], start=start, end=end, label="SYMPTOM",
                canonical=entry.canonical, source=entry.source, concept_id=entry.concept_id,
                confidence=min(1.0, (score/100.0)*0.8 + (jaccard/100.0)*0.2), negated=negated
            ))

    # Deduplicate exact duplicate spans, preserve overlapping contextual mentions
    return _deduplicate_spans(spans)


def match_products(text: str, lexicon: List[LexiconEntry],
                   fuzzy_threshold: float = 88.0,
                   max_term_words: int = 6,
                   scorer: str = "wratio") -> List[Span]:
    if not lexicon:
        return []
    tokens = _tokenize(text)
    spans: List[Span] = []
    
    # Build term map + metadata
    term_map = {e.term.lower(): e for e in lexicon}
    term_meta = []
    for e in lexicon:
        t = e.term.lower()
        toks = t.split()
        term_meta.append({
            "term": t,
            "tok_len": len(toks),
            "first": toks[0] if toks else "",
        })
    term_list = [m["term"] for m in term_meta]
    max_tok_len = max((m["tok_len"] for m in term_meta), default=0)
    
    # Exact phrase matching
    lower_text = text.lower()
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
                spans.append(Span(
                    text=text[idx:end_idx], start=idx, end=end_idx, label="PRODUCT",
                    canonical=entry.canonical, source=entry.source, sku=entry.sku,
                    category=entry.category, confidence=1.0
                ))
            idx = end_idx
    
    # Fuzzy matching
    for window_size in range(1, max_term_words + 1):
        for i in range(0, len(tokens) - window_size + 1):
            window_tokens = tokens[i:i+window_size]
            phrase_tokens = [t[0].lower() for t in window_tokens]
            phrase_clean = " ".join(phrase_tokens)
            if phrase_clean in term_map:
                continue
            if len(phrase_clean) < 3:
                continue
            if window_size == 1 and phrase_tokens[0] in ANATOMY_TOKENS:
                continue
            if not any(tok not in STOPWORDS for tok in phrase_tokens):
                continue
            if not any(len(tok) > 3 for tok in phrase_tokens if tok not in STOPWORDS):
                continue
            span_text = text[window_tokens[0][1]:window_tokens[-1][2]]
            if any(p in span_text for p in [',',';','.']):
                continue
            phrase_tok_len = len(phrase_tokens)
            candidates = [m["term"] for m in term_meta
                          if m["first"] == phrase_tokens[0]
                          and abs(m["tok_len"] - phrase_tok_len) <= 1]
            if not candidates:
                continue
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
                if phrase_tok_len > 1 and c_tokens[-1] != phrase_tokens[-1]:
                    continue
                filtered_candidates.append(c)
            if not filtered_candidates:
                continue
            cutoff = fuzzy_threshold
            if HAVE_RAPIDFUZZ:
                if scorer == "jaccard":
                    best = process.extractOne(
                        phrase_clean, filtered_candidates,
                        scorer=lambda a, b: _jaccard_token_score(a, b),
                        score_cutoff=cutoff
                    )
                else:
                    best = process.extractOne(
                        phrase_clean, filtered_candidates,
                        scorer=fuzz.WRatio,
                        score_cutoff=cutoff
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
            jaccard = _jaccard_token_score(phrase_clean, matched_term)
            if jaccard < 40.0:
                continue
            entry = term_map[matched_term]
            start = window_tokens[0][1]
            end = window_tokens[-1][2]
            spans.append(Span(
                text=text[start:end], start=start, end=end, label="PRODUCT",
                canonical=entry.canonical, source=entry.source, sku=entry.sku,
                category=entry.category, confidence=min(1.0, (score/100.0)*0.8 + (jaccard/100.0)*0.2)
            ))
    
    # Deduplicate exact duplicate spans, preserve overlapping contextual mentions
    return _deduplicate_spans(spans)


def assemble_spans(symptom_spans: List[Span], product_spans: List[Span]) -> List[Span]:
    return symptom_spans + product_spans


def weak_label(text: str, symptom_lexicon: List[LexiconEntry],
               product_lexicon: List[LexiconEntry],
               negation_window: int = 5,
               scorer: str = "wratio") -> List[Span]:
    symptom_spans = match_symptoms(text, symptom_lexicon, negation_window=negation_window, scorer=scorer)
    product_spans = match_products(text, product_lexicon, scorer=scorer)
    return assemble_spans(symptom_spans, product_spans)


def weak_label_batch(texts: List[str], symptom_lexicon: List[LexiconEntry],
                     product_lexicon: List[LexiconEntry],
                     negation_window: int = 5,
                     scorer: str = "wratio") -> List[List[Span]]:
    return [weak_label(t, symptom_lexicon, product_lexicon, negation_window, scorer) for t in texts]


def persist_weak_labels_jsonl(texts: List[str], spans_batch: List[List[Span]], output_path: Path) -> None:
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
                    } for s in spans
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
