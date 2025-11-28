"""
Confidence scoring for weak label matches.

Implements the confidence formula: 0.8 * fuzzy_score + 0.2 * jaccard_score
"""

from __future__ import annotations

from typing import List, Tuple


def compute_confidence(fuzzy_score: float, jaccard_score: float) -> float:
    """Compute confidence score for a matched span.

    Uses weighted combination: 80% fuzzy match score, 20% Jaccard token similarity.
    This prioritizes semantic similarity (fuzzy) while ensuring token overlap (Jaccard).

    Args:
        fuzzy_score: Fuzzy match score (0-100)
        jaccard_score: Jaccard token similarity score (0-100)

    Returns:
        Confidence score clamped to [0.0, 1.0]

    Example:
        >>> compute_confidence(88.0, 60.0)
        0.816
        >>> compute_confidence(100.0, 100.0)
        1.0
        >>> compute_confidence(50.0, 30.0)
        0.46
    """
    # Normalize to 0-1 range
    fuzzy_norm = fuzzy_score / 100.0
    jaccard_norm = jaccard_score / 100.0

    # Weighted combination: 80% fuzzy, 20% Jaccard
    confidence = fuzzy_norm * 0.8 + jaccard_norm * 0.2

    # Clamp to valid range
    return min(1.0, max(0.0, confidence))


def align_spans(
    text: str, spans_with_positions: List[Tuple[str, int, int]]
) -> List[Tuple[str, int, int]]:
    """Align span boundaries to exact text positions.

    Adjusts span boundaries to match word boundaries and remove trailing punctuation.

    Args:
        text: Full input text
        spans_with_positions: List of (span_text, start, end) tuples

    Returns:
        List of aligned (span_text, start, end) tuples

    Example:
        >>> text = "I have burning sensation."
        >>> spans = [("burning sensation.", 7, 26)]
        >>> align_spans(text, spans)
        [('burning sensation', 7, 25)]
    """
    aligned = []

    for span_text, start, end in spans_with_positions:
        # Extract actual text at this position
        actual_text = text[start:end]

        # Strip trailing punctuation
        while actual_text and not actual_text[-1].isalnum():
            actual_text = actual_text[:-1]
            end -= 1

        # Strip leading punctuation
        while actual_text and not actual_text[0].isalnum():
            actual_text = actual_text[1:]
            start += 1

        if actual_text:  # Only keep non-empty spans
            aligned.append((actual_text, start, end))

    return aligned


def adjust_confidence_for_negation(
    confidence: float, is_negated: bool, negation_penalty: float = 0.1
) -> float:
    """Optionally adjust confidence based on negation status.

    Args:
        confidence: Base confidence score
        is_negated: Whether span is negated
        negation_penalty: Amount to reduce confidence for negated spans

    Returns:
        Adjusted confidence score

    Example:
        >>> adjust_confidence_for_negation(0.9, False)
        0.9
        >>> adjust_confidence_for_negation(0.9, True, penalty=0.1)
        0.8
    """
    if is_negated:
        return max(0.0, confidence - negation_penalty)
    return confidence


def calibrate_threshold(
    spans_with_gold: List[Tuple[float, bool]], target_precision: float = 0.9
) -> float:
    """Find confidence threshold that achieves target precision.

    Args:
        spans_with_gold: List of (confidence, is_correct) tuples
        target_precision: Desired precision (0-1)

    Returns:
        Confidence threshold that achieves target precision

    Example:
        >>> spans = [(0.9, True), (0.8, True), (0.7, False), (0.6, False)]
        >>> calibrate_threshold(spans, target_precision=1.0)
        0.8
    """
    if not spans_with_gold:
        return 0.0

    # Sort by confidence descending
    sorted_spans = sorted(spans_with_gold, key=lambda x: x[0], reverse=True)

    best_threshold = 0.0
    for i, (conf, is_correct) in enumerate(sorted_spans):
        # Calculate precision at this threshold
        above_threshold = sorted_spans[: i + 1]
        correct = sum(1 for _, c in above_threshold if c)
        precision = correct / len(above_threshold)

        if precision >= target_precision:
            best_threshold = conf
        else:
            break

    return best_threshold
