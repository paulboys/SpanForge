"""
Span validation and filtering rules.

Provides validation logic for filtering out spurious matches like
standalone anatomy tokens and deduplicating exact matches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from src.weak_labeling.matchers import tokenize
from src.weak_labeling.types import Span

# Anatomy tokens that should not be labeled alone
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
    "head",
    "neck",
    "chest",
    "back",
    "stomach",
    "abdomen",
    "finger",
    "fingers",
    "toe",
    "toes",
    "foot",
    "feet",
    "scalp",
    "forehead",
    "cheek",
    "cheeks",
    "chin",
    "jaw",
}


def is_anatomy_only(text: str) -> bool:
    """Check if span is just a single anatomy token without symptom context.

    Single anatomy words like "face" should not be labeled as symptoms.
    However, "facial rash" or "face cream" are valid because they have context.

    Uses composition: delegates tokenization to matchers.tokenize() for clean separation
    of concerns.

    Args:
        text: The span text to check

    Returns:
        True if span is only anatomy without symptom context

    Example:
        >>> is_anatomy_only("face")
        True
        >>> is_anatomy_only("facial rash")
        False
        >>> is_anatomy_only("")
        False
    """
    # Handle edge cases
    if not text or not text.strip():
        return False

    # Use composition: delegate tokenization to matchers module
    tokens = tokenize(text.strip())

    # Extract just the token strings (tokenize returns (token, start, end) tuples)
    token_strings = [tok[0] for tok in tokens]

    if len(token_strings) == 1 and token_strings[0].lower() in ANATOMY_TOKENS:
        return True
    return False


def _validate_single_span(text: str, span_text: str, start: int, end: int) -> bool:
    """Internal function to validate a single span alignment.

    Args:
        text: Full input text
        span_text: The extracted span text
        start: Start position in full text
        end: End position in full text

    Returns:
        True if span is correctly aligned, False otherwise
    """
    # Handle empty inputs
    if not text:
        return False

    # Validate indices: must be non-negative, in bounds, and properly ordered
    if start < 0 or end < 0:
        return False
    if start >= end:  # Catches reversed indices and zero-length spans
        return False
    if end > len(text):
        return False

    extracted = text[start:end]
    return extracted == span_text


def validate_span_alignment(
    text: str,
    span_text_or_spans: Union[str, List[Span]],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Union[bool, Tuple[bool, List[str]]]:
    """Validate span alignment - polymorphic function using composition.

    This function demonstrates the Strategy pattern by dispatching to different
    validation strategies based on arguments:
    - Single span validation: validate_span_alignment(text, span_text, start, end) -> bool
    - Batch validation: validate_span_alignment(text, spans) -> Tuple[bool, List[str]]

    Args:
        text: Full input text
        span_text_or_spans: Either a span text string OR a list of Span objects
        start: Start position (required if span_text_or_spans is a string)
        end: End position (required if span_text_or_spans is a string)

    Returns:
        - If validating single span: bool indicating if aligned
        - If validating batch: Tuple of (is_valid, errors)

    Example:
        >>> # Single span validation
        >>> validate_span_alignment("I have burning", "burning", 7, 14)
        True
        >>> # Batch validation
        >>> from src.weak_labeling.types import Span
        >>> spans = [Span("burning", 7, 14, "SYMPTOM", confidence=0.9)]
        >>> is_valid, errors = validate_span_alignment("I have burning", spans)
        >>> is_valid
        True
    """
    # Strategy 1: Batch validation (list of Span objects)
    if isinstance(span_text_or_spans, list):
        spans = span_text_or_spans

        # Handle edge case: empty text with spans
        if not text and spans:
            return False, ["Cannot validate spans against empty text"]

        # Handle edge case: empty spans list (valid by definition)
        if not spans:
            return True, []

        errors = []
        for span in spans:
            # Use composition: delegate to single-span validator
            if not _validate_single_span(text, span.text, span.start, span.end):
                errors.append(
                    f"Span '{span.text}' at ({span.start}, {span.end}) does not match text[{span.start}:{span.end}]"
                )

        return len(errors) == 0, errors

    # Strategy 2: Single span validation (requires start and end)
    else:
        span_text = span_text_or_spans
        if start is None or end is None:
            raise TypeError("start and end are required when validating a single span")
        return _validate_single_span(text, span_text, start, end)


def deduplicate_spans(spans: List[Span]) -> List[Span]:
    """Remove exact duplicate spans, keeping the highest confidence one.

    Preserves overlapping spans with different boundaries (e.g., 'rash' vs 'little rash')
    as they represent distinct contextual mentions.

    Args:
        spans: List of spans to deduplicate

    Returns:
        Deduplicated list of spans, sorted by start position

    Example:
        >>> from src.weak_labeling.types import Span
        >>> spans = [
        ...     Span("burning", 0, 7, "SYMPTOM", confidence=0.9),
        ...     Span("burning", 0, 7, "SYMPTOM", confidence=0.8),
        ... ]
        >>> result = deduplicate_spans(spans)
        >>> len(result)
        1
        >>> result[0].confidence
        0.9
    """
    if not spans:
        return []

    # Group spans by (start, end, canonical) tuple
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


def filter_overlapping_spans(spans: List[Span], strategy: str = "longest") -> List[Span]:
    """Filter overlapping spans using specified strategy.

    Args:
        spans: List of spans to filter
        strategy: Strategy to use. Supported: "longest", "highest_confidence", "first".
                  Raises ValueError for invalid strategies.

    Returns:
        Filtered list without overlaps

    Raises:
        ValueError: If strategy is not one of the supported values

    Example:
        >>> from src.weak_labeling.types import Span
        >>> spans = [
        ...     Span("burning", 0, 7, "SYMPTOM", confidence=0.9),
        ...     Span("burning sensation", 0, 18, "SYMPTOM", confidence=0.85),
        ... ]
        >>> result = filter_overlapping_spans(spans, strategy="longest")
        >>> result[0].text
        'burning sensation'
    """
    # Validate strategy parameter
    valid_strategies = {"longest", "highest_confidence", "first"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of: {', '.join(sorted(valid_strategies))}"
        )

    if not spans:
        return []

    # Sort by start position
    sorted_spans = sorted(spans, key=lambda s: s.start)

    filtered = []
    for span in sorted_spans:
        # Check if overlaps with any already-kept span
        overlaps = False
        for kept in filtered:
            if span.start < kept.end and kept.start < span.end:
                overlaps = True
                # Decide which to keep based on strategy
                if strategy == "longest":
                    if (span.end - span.start) > (kept.end - kept.start):
                        filtered.remove(kept)
                        overlaps = False
                elif strategy == "highest_confidence":
                    if span.confidence > kept.confidence:
                        filtered.remove(kept)
                        overlaps = False
                elif strategy == "first":
                    pass  # Keep the first one (already in filtered)
                break

        if not overlaps:
            filtered.append(span)

    return filtered


def get_anatomy_tokens() -> set:
    """Get the set of anatomy tokens.

    Returns:
        Set of anatomy token strings

    Example:
        >>> tokens = get_anatomy_tokens()
        >>> "face" in tokens
        True
    """
    return ANATOMY_TOKENS.copy()
