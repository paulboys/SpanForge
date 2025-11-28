"""
String matching algorithms for fuzzy entity extraction.

Provides fuzzy matching (RapidFuzz WRatio), exact matching, and Jaccard
token similarity for comparing lexicon terms against text spans.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    from rapidfuzz import fuzz

    HAVE_RAPIDFUZZ = True
except ImportError:
    import difflib

    HAVE_RAPIDFUZZ = False

# Regular expressions and constants
WORD_PATTERN = re.compile(r"\b\w[\w\-']*\b")

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


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text and return (token, start, end) tuples using original positions.

    Args:
        text: Input text to tokenize

    Returns:
        List of (token_string, start_pos, end_pos) tuples

    Example:
        >>> tokenize("Hello world")
        [('Hello', 0, 5), ('world', 6, 11)]
    """
    tokens = []
    for m in WORD_PATTERN.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def tokenize_clean(text: str) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Return cleaned text AND tokens with adjusted positions.

    Removes emojis and returns tokens relative to cleaned text.

    Args:
        text: Input text potentially containing emojis

    Returns:
        Tuple of (cleaned_text, tokens) where tokens have positions relative to cleaned text

    Example:
        >>> clean_text, tokens = tokenize_clean("Hello 😊 world")
        >>> clean_text
        'Hello   world'
    """
    cleaned_text = EMOJI_PATTERN.sub(" ", text)
    tokens = []
    for m in WORD_PATTERN.finditer(cleaned_text):
        tokens.append((m.group(0), m.start(), m.end()))
    return cleaned_text, tokens


def jaccard_token_score(a: str, b: str) -> float:
    """Compute Jaccard token-set similarity (0-100 scale).

    Args:
        a: First string
        b: Second string

    Returns:
        Jaccard similarity as percentage (0.0-100.0)

    Example:
        >>> jaccard_token_score("severe itching", "itching")
        50.0
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return (len(intersection) / len(union)) * 100.0


def fuzzy_match(query: str, candidate: str, scorer: str = "WRatio") -> float:
    """Compute fuzzy string similarity score.

    Uses RapidFuzz if available, otherwise falls back to difflib.

    Args:
        query: Query string
        candidate: Candidate string to match against
        scorer: Scoring algorithm ("WRatio", "ratio", "partial_ratio", "token_set_ratio")

    Returns:
        Similarity score (0.0-100.0)

    Example:
        >>> fuzzy_match("burning", "burning sensation")
        85.71
    """
    if HAVE_RAPIDFUZZ:
        if scorer == "WRatio":
            return fuzz.WRatio(query, candidate)
        elif scorer == "ratio":
            return fuzz.ratio(query, candidate)
        elif scorer == "partial_ratio":
            return fuzz.partial_ratio(query, candidate)
        elif scorer == "token_set_ratio":
            return fuzz.token_set_ratio(query, candidate)
        else:
            return fuzz.WRatio(query, candidate)
    else:
        # Fallback to difflib
        return difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio() * 100


def exact_match(query: str, candidate: str, case_sensitive: bool = False) -> bool:
    """Check for exact string match.

    Args:
        query: Query string
        candidate: Candidate string
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        True if strings match exactly

    Example:
        >>> exact_match("burning", "Burning")
        True
        >>> exact_match("burning", "Burning", case_sensitive=True)
        False
    """
    if case_sensitive:
        return query == candidate
    return query.lower() == candidate.lower()
