"""Evaluation metrics for span quality assessment.

Provides comprehensive metrics for evaluating NER span quality including:
- Boundary precision and IOU calculations
- Correction rate tracking (weak → LLM improvements)
- Stratification helpers for confidence/label/length analysis
- Calibration curves for confidence scoring
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_overlap(span_a: Dict[str, Any], span_b: Dict[str, Any]) -> int:
    """Compute character overlap between two spans.

    Args:
        span_a: First span with 'start' and 'end' keys
        span_b: Second span with 'start' and 'end' keys

    Returns:
        Number of overlapping characters (0 if no overlap)

    Example:
        >>> span1 = {"start": 10, "end": 20}
        >>> span2 = {"start": 15, "end": 25}
        >>> compute_overlap(span1, span2)
        5
    """
    return max(0, min(span_a["end"], span_b["end"]) - max(span_a["start"], span_b["start"]))


def compute_iou(span_a: Dict[str, Any], span_b: Dict[str, Any]) -> float:
    """Compute intersection-over-union for two spans.

    Args:
        span_a: First span with 'start' and 'end' keys
        span_b: Second span with 'start' and 'end' keys

    Returns:
        IOU score between 0.0 and 1.0

    Example:
        >>> span1 = {"start": 10, "end": 20}
        >>> span2 = {"start": 10, "end": 20}
        >>> compute_iou(span1, span2)
        1.0
    """
    overlap = compute_overlap(span_a, span_b)
    if overlap == 0:
        return 0.0
    union = (span_a["end"] - span_a["start"]) + (span_b["end"] - span_b["start"]) - overlap
    return overlap / union if union > 0 else 0.0


def compute_boundary_precision(
    predicted_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    exact_match_threshold: float = 1.0,
) -> Dict[str, float]:
    """Compute boundary precision metrics.

    Measures how accurately predicted spans match gold span boundaries.

    Args:
        predicted_spans: List of predicted spans
        gold_spans: List of gold standard spans
        exact_match_threshold: IOU threshold for exact match (default: 1.0)

    Returns:
        Dict with 'exact_match_rate', 'mean_iou', 'median_iou'

    Example:
        >>> pred = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> result = compute_boundary_precision(pred, gold)
        >>> result["exact_match_rate"]
        1.0
    """
    if not predicted_spans or not gold_spans:
        return {"exact_match_rate": 0.0, "mean_iou": 0.0, "median_iou": 0.0}

    iou_scores = []
    exact_matches = 0

    for pred_span in predicted_spans:
        best_iou = 0.0
        for gold_span in gold_spans:
            if pred_span.get("label") == gold_span.get("label"):
                iou = compute_iou(pred_span, gold_span)
                best_iou = max(best_iou, iou)

        iou_scores.append(best_iou)
        if best_iou >= exact_match_threshold:
            exact_matches += 1

    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    median_iou = sorted(iou_scores)[len(iou_scores) // 2] if iou_scores else 0.0

    return {
        "exact_match_rate": exact_matches / len(predicted_spans),
        "mean_iou": mean_iou,
        "median_iou": median_iou,
    }


def compute_iou_delta(
    weak_spans: List[Dict[str, Any]],
    llm_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute IOU improvement from weak labels to LLM refinement.

    Args:
        weak_spans: Original weak label predictions
        llm_spans: LLM-refined predictions
        gold_spans: Gold standard annotations

    Returns:
        Dict with 'weak_mean_iou', 'llm_mean_iou', 'delta', 'improvement_pct'

    Example:
        >>> weak = [{"start": 10, "end": 22, "label": "SYMPTOM"}]
        >>> llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> result = compute_iou_delta(weak, llm, gold)
        >>> result["improvement_pct"] > 0
        True
    """
    weak_metrics = compute_boundary_precision(weak_spans, gold_spans)
    llm_metrics = compute_boundary_precision(llm_spans, gold_spans)

    weak_iou = weak_metrics["mean_iou"]
    llm_iou = llm_metrics["mean_iou"]
    delta = llm_iou - weak_iou
    improvement_pct = (delta / weak_iou * 100) if weak_iou > 0 else 0.0

    return {
        "weak_mean_iou": weak_iou,
        "llm_mean_iou": llm_iou,
        "delta": delta,
        "improvement_pct": improvement_pct,
    }


def compute_correction_rate(
    weak_spans: List[Dict[str, Any]],
    llm_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute LLM correction success rate.

    Tracks which weak spans were improved, worsened, or left unchanged by LLM.

    Args:
        weak_spans: Original weak predictions
        llm_spans: LLM-refined predictions
        gold_spans: Gold standard
        iou_threshold: Minimum IOU to consider a match

    Returns:
        Dict with correction statistics

    Example:
        >>> weak = [{"start": 10, "end": 25, "label": "SYMPTOM"}]
        >>> llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> result = compute_correction_rate(weak, llm, gold)
        >>> result["improved_count"] >= 0
        True
    """
    improved = 0
    worsened = 0
    unchanged = 0
    total_modified = 0

    # Match weak and LLM spans (by best IOU overlap)
    for weak_span in weak_spans:
        # Find best matching LLM span
        best_llm_match = None
        best_llm_iou = 0.0
        for llm_span in llm_spans:
            if weak_span.get("label") == llm_span.get("label"):
                iou = compute_iou(weak_span, llm_span)
                if iou > best_llm_iou:
                    best_llm_iou = iou
                    best_llm_match = llm_span

        if best_llm_match is None:
            continue  # No LLM refinement for this weak span

        # Check if span was modified
        if (
            weak_span["start"] != best_llm_match["start"]
            or weak_span["end"] != best_llm_match["end"]
        ):
            total_modified += 1

            # Compare to gold standard
            weak_gold_iou = max(
                [
                    compute_iou(weak_span, g)
                    for g in gold_spans
                    if g.get("label") == weak_span.get("label")
                ]
                or [0.0]
            )
            llm_gold_iou = max(
                [
                    compute_iou(best_llm_match, g)
                    for g in gold_spans
                    if g.get("label") == best_llm_match.get("label")
                ]
                or [0.0]
            )

            if llm_gold_iou > weak_gold_iou:
                improved += 1
            elif llm_gold_iou < weak_gold_iou:
                worsened += 1
            else:
                unchanged += 1
        else:
            unchanged += 1

    return {
        "total_spans": len(weak_spans),
        "modified_count": total_modified,
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "improvement_rate": improved / total_modified if total_modified > 0 else 0.0,
        "false_refinement_rate": worsened / total_modified if total_modified > 0 else 0.0,
    }


def calibration_curve(
    spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    confidence_key: str = "confidence",
    num_bins: int = 10,
) -> Dict[str, List[float]]:
    """Compute calibration curve for confidence scores.

    Measures how well predicted confidence correlates with actual accuracy.

    Args:
        spans: Predicted spans with confidence scores
        gold_spans: Gold standard spans
        confidence_key: Key name for confidence score in span dict
        num_bins: Number of confidence bins (default: 10)

    Returns:
        Dict with 'bin_centers', 'accuracy', 'counts' lists

    Example:
        >>> spans = [{"start": 10, "end": 20, "label": "SYMPTOM", "confidence": 0.9}]
        >>> gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> curve = calibration_curve(spans, gold, num_bins=5)
        >>> len(curve["bin_centers"]) == 5
        True
    """
    # Initialize bins
    bin_width = 1.0 / num_bins
    bins = [[] for _ in range(num_bins)]

    # Assign spans to bins and compute accuracy
    for span in spans:
        conf = span.get(confidence_key, 0.0)
        bin_idx = min(int(conf / bin_width), num_bins - 1)

        # Check if span matches gold
        matches_gold = any(
            compute_iou(span, g) >= 0.5 and span.get("label") == g.get("label") for g in gold_spans
        )
        bins[bin_idx].append(1.0 if matches_gold else 0.0)

    # Compute per-bin statistics
    bin_centers = [(i + 0.5) * bin_width for i in range(num_bins)]
    accuracy = [sum(b) / len(b) if b else 0.0 for b in bins]
    counts = [len(b) for b in bins]

    return {"bin_centers": bin_centers, "accuracy": accuracy, "counts": counts}


def stratify_by_confidence(
    spans: List[Dict[str, Any]], confidence_key: str = "confidence", buckets: List[float] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Stratify spans into confidence buckets.

    Args:
        spans: List of spans with confidence scores
        confidence_key: Key name for confidence in span dict
        buckets: Bucket thresholds (default: [0.6, 0.7, 0.8, 0.9, 1.0])

    Returns:
        Dict mapping bucket labels to span lists

    Example:
        >>> spans = [
        ...     {"start": 0, "end": 5, "confidence": 0.65},
        ...     {"start": 10, "end": 15, "confidence": 0.95}
        ... ]
        >>> result = stratify_by_confidence(spans)
        >>> "0.60-0.70" in result
        True
    """
    if buckets is None:
        buckets = [0.6, 0.7, 0.8, 0.9, 1.0]

    stratified = {}
    for i, threshold in enumerate(buckets):
        lower = buckets[i - 1] if i > 0 else 0.0
        label = f"{lower:.2f}-{threshold:.2f}"
        stratified[label] = []

    for span in spans:
        conf = span.get(confidence_key, 0.0)
        for i, threshold in enumerate(buckets):
            lower = buckets[i - 1] if i > 0 else 0.0
            if lower <= conf <= threshold:
                label = f"{lower:.2f}-{threshold:.2f}"
                stratified[label].append(span)
                break

    return stratified


def stratify_by_label(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Stratify spans by entity label.

    Args:
        spans: List of spans with 'label' key

    Returns:
        Dict mapping labels to span lists

    Example:
        >>> spans = [
        ...     {"start": 0, "end": 5, "label": "SYMPTOM"},
        ...     {"start": 10, "end": 15, "label": "PRODUCT"}
        ... ]
        >>> result = stratify_by_label(spans)
        >>> len(result["SYMPTOM"])
        1
    """
    stratified: Dict[str, List[Dict[str, Any]]] = {}
    for span in spans:
        label = span.get("label", "UNKNOWN")
        if label not in stratified:
            stratified[label] = []
        stratified[label].append(span)
    return stratified


def stratify_by_span_length(
    spans: List[Dict[str, Any]], text_key: str = "text"
) -> Dict[str, List[Dict[str, Any]]]:
    """Stratify spans by token/word length.

    Args:
        spans: List of spans
        text_key: Key for span text (default: "text")

    Returns:
        Dict with 'single_word', 'multi_word' lists

    Example:
        >>> spans = [
        ...     {"start": 0, "end": 4, "text": "rash"},
        ...     {"start": 10, "end": 25, "text": "burning sensation"}
        ... ]
        >>> result = stratify_by_span_length(spans)
        >>> len(result["single_word"])
        1
    """
    single_word = []
    multi_word = []

    for span in spans:
        text = span.get(text_key, "")
        # Simple word count based on whitespace
        word_count = len(text.split()) if text else 0

        if word_count <= 1:
            single_word.append(span)
        else:
            multi_word.append(span)

    return {"single_word": single_word, "multi_word": multi_word}


def compute_precision_recall_f1(
    predicted_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score.

    Args:
        predicted_spans: List of predicted spans
        gold_spans: List of gold standard spans
        iou_threshold: Minimum IOU for match (default: 0.5)

    Returns:
        Dict with 'precision', 'recall', 'f1' keys

    Example:
        >>> pred = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        >>> metrics = compute_precision_recall_f1(pred, gold)
        >>> metrics["f1"]
        1.0
    """
    if not predicted_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Match predicted to gold
    matched_gold = set()
    true_positives = 0

    for pred_span in predicted_spans:
        for i, gold_span in enumerate(gold_spans):
            if i in matched_gold:
                continue
            if pred_span.get("label") == gold_span.get("label"):
                if compute_iou(pred_span, gold_span) >= iou_threshold:
                    matched_gold.add(i)
                    true_positives += 1
                    break

    precision = true_positives / len(predicted_spans)
    recall = true_positives / len(gold_spans)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
