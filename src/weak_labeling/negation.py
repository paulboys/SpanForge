"""
Negation detection for symptom spans.

Implements bidirectional negation windows (forward and backward) to detect
negated symptoms like "no itching" or "itching absent".
"""

from __future__ import annotations

from typing import List, Tuple

# Import tokenize from matchers
from src.weak_labeling.matchers import tokenize

# Negation cue words
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


def detect_negated_regions(text: str, window: int = 5, window_size: int = None) -> List[Tuple[int, int]]:
    """Detect character ranges that are negated.

    Uses bidirectional windows:
    - Forward window: negation precedes symptom (e.g., "no itching")
    - Backward window: negation follows symptom (e.g., "itching absent")

    Handles edge cases:
    - Empty or whitespace-only text
    - Unicode characters
    - Window size larger than text
    - Punctuation around negation cues
    - Repeated negation cues

    Args:
        text: Input text to analyze
        window: Number of tokens to include in negation window (default: 5)
        window_size: Alternative parameter name for backward compatibility (overrides window if provided)

    Returns:
        List of (start, end) character ranges that are negated

    Example:
        >>> regions = detect_negated_regions("No burning or itching present")
        >>> len(regions) > 0
        True
        >>> regions = detect_negated_regions("   ")
        >>> len(regions)
        0
    """
    # Handle backward compatibility: window_size overrides window
    if window_size is not None:
        window = window_size
    
    # Handle edge cases: empty or whitespace-only text
    if not text or not text.strip():
        return []
    
    # Tokenize (handles unicode and punctuation)
    tokens = tokenize(text)
    
    # Empty token list (shouldn't happen after strip check, but defensive)
    if not tokens:
        return []
    
    neg_spans: List[Tuple[int, int]] = []

    for i, (tok, s, e) in enumerate(tokens):
        tok_lower = tok.lower()

        # Check exact match or prefix match for flexible negation detection
        is_negation = tok_lower in NEGATION_TOKENS or any(
            tok_lower.startswith(neg) for neg in NEGATION_TOKENS if len(neg) > 3
        )

        if is_negation:
            # Forward window: next N tokens after negation cue
            # These are the tokens that are negated
            window_tokens = tokens[i + 1 : i + 1 + window]
            has_forward = len(window_tokens) > 0
            if has_forward:
                neg_start = window_tokens[0][1]
                neg_end = window_tokens[-1][2]
                neg_spans.append((neg_start, neg_end))

            # Backward window: previous N tokens before negation cue
            # Handle "symptom absent" pattern where negation follows
            back_tokens = tokens[max(0, i - window) : i]
            has_backward = len(back_tokens) > 0
            if has_backward:
                back_start = back_tokens[0][1]
                back_end = back_tokens[-1][2]
                neg_spans.append((back_start, back_end))
            
            # Edge case: if no forward and no backward tokens (e.g., text is just "no"),
            # or window_size=0, include the negation cue itself
            if (not has_forward and not has_backward) or window == 0:
                neg_spans.append((s, e))

    return neg_spans


def is_negated(
    span_start: int,
    span_end: int,
    neg_regions: List[Tuple[int, int]],
    overlap_threshold: float = 0.5,
) -> bool:
    """Check if a span overlaps with negated regions.

    Applies defensive programming for edge cases:
    - Reversed indices (start > end): Returns False
    - Negative indices: Returns False
    - Zero-length spans (start == end): Returns False
    - Empty negation regions list: Returns False

    Args:
        span_start: Start character position of span
        span_end: End character position of span
        neg_regions: List of (start, end) negated regions from detect_negated_regions
        overlap_threshold: Minimum fraction of span that must overlap (default: 0.5)

    Returns:
        True if span is negated, False for invalid inputs or no overlap

    Example:
        >>> regions = [(3, 10)]  # "no itching"
        >>> is_negated(3, 10, regions)  # "itching" span
        True
        >>> is_negated(15, 20, regions)  # "redness" span (outside negation)
        False
        >>> is_negated(10, 5, regions)  # Invalid: reversed indices
        False
        >>> is_negated(-1, 5, regions)  # Invalid: negative index
        False
    """
    # Validate inputs: handle edge cases
    if span_start < 0 or span_end < 0:
        return False
    if span_start >= span_end:  # Catches reversed indices and zero-length spans
        return False
    if not neg_regions:
        return False
    
    span_length = span_end - span_start

    for ns, ne in neg_regions:
        # Calculate overlap
        overlap = max(0, min(span_end, ne) - max(span_start, ns))

        # If span overlaps negated region significantly, mark as negated
        if overlap >= overlap_threshold * span_length:
            return True

    return False


def get_negation_cues() -> set:
    """Get the set of negation cue words.

    Returns:
        Set of negation tokens

    Example:
        >>> cues = get_negation_cues()
        >>> "no" in cues
        True
    """
    return NEGATION_TOKENS.copy()
