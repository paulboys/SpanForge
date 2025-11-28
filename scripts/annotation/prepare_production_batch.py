"""Prepare production annotation batches with stratified sampling and de-identification.

This script prepares annotation-ready batches from CAERS or custom data sources:
- Stratified sampling by span density, text length, categories
- Optional PII de-identification (placeholder replacement)
- Quality filtering (minimum span requirements, text length constraints)
- Metadata preservation for provenance tracking
- Output to annotation-ready JSONL format

Usage:
    # Prepare 100-task batch from CAERS cosmetics data
    python scripts/annotation/prepare_production_batch.py \\
        --input data/caers/cosmetics_1000.jsonl \\
        --output data/annotation/exports/batch_001.jsonl \\
        --n-tasks 100 \\
        --strategy stratified \\
        --min-spans 1 \\
        --max-text-len 500 \\
        --deidentify

    # Prepare from custom weak labels
    python scripts/annotation/prepare_production_batch.py \\
        --input data/output/custom_weak.jsonl \\
        --output data/annotation/exports/batch_custom.jsonl \\
        --n-tasks 50 \\
        --strategy balanced \\
        --category-balance

Example:
    >>> from prepare_production_batch import prepare_batch
    >>> prepare_batch(
    ...     input_path="data/caers/cosmetics.jsonl",
    ...     output_path="data/annotation/exports/batch.jsonl",
    ...     n_tasks=100,
    ...     strategy="stratified"
    ... )
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PII patterns for de-identification
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "date": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "name_pattern": re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),  # Simple name pattern
}


def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of records.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of record dictionaries

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If JSONL is malformed
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {line_num}: {e}")

    return records


def deidentify_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Remove PII from text using pattern matching.

    Args:
        text: Original text with potential PII

    Returns:
        Tuple of (cleaned text, list of replacements made)

    Note:
        This is a basic implementation. For production, consider:
        - NER-based PII detection
        - Context-aware name detection
        - Medical number patterns
    """
    cleaned = text
    replacements = []

    # Email replacement
    for match in PII_PATTERNS["email"].finditer(text):
        replacement = "[EMAIL]"
        cleaned = cleaned.replace(match.group(), replacement)
        replacements.append(
            {"type": "email", "original": match.group(), "start": match.start(), "end": match.end()}
        )

    # Phone replacement
    for match in PII_PATTERNS["phone"].finditer(text):
        replacement = "[PHONE]"
        cleaned = cleaned.replace(match.group(), replacement)
        replacements.append(
            {"type": "phone", "original": match.group(), "start": match.start(), "end": match.end()}
        )

    # SSN replacement
    for match in PII_PATTERNS["ssn"].finditer(text):
        replacement = "[SSN]"
        cleaned = cleaned.replace(match.group(), replacement)
        replacements.append(
            {"type": "ssn", "original": match.group(), "start": match.start(), "end": match.end()}
        )

    # Date replacement (optional - may want to keep dates)
    # Uncomment if dates should be removed
    # for match in PII_PATTERNS["date"].finditer(text):
    #     replacement = "[DATE]"
    #     cleaned = cleaned.replace(match.group(), replacement)

    return cleaned, replacements


def adjust_spans_for_deidentification(
    spans: List[Dict[str, Any]],
    replacements: List[Dict[str, Any]],
    original_text: str,
    cleaned_text: str,
) -> List[Dict[str, Any]]:
    """Adjust span positions after text de-identification.

    Args:
        spans: Original span list with start/end positions
        replacements: List of PII replacements made
        original_text: Original text before cleaning
        cleaned_text: Text after PII removal

    Returns:
        Adjusted spans with corrected positions

    Note:
        Spans that overlap with PII are discarded for safety.
    """
    if not replacements:
        return spans

    adjusted_spans = []
    offset = 0

    # Sort replacements by position
    replacements_sorted = sorted(replacements, key=lambda x: x["start"])

    for span in spans:
        span_start = span["start"]
        span_end = span["end"]

        # Check if span overlaps with any PII
        overlaps_pii = False
        for repl in replacements_sorted:
            if not (span_end <= repl["start"] or span_start >= repl["end"]):
                overlaps_pii = True
                break

        if overlaps_pii:
            # Skip spans that overlap with PII
            continue

        # Calculate offset for this span
        span_offset = 0
        for repl in replacements_sorted:
            if repl["end"] <= span_start:
                original_len = len(repl["original"])
                replacement_len = len(f"[{repl['type'].upper()}]")
                span_offset += replacement_len - original_len

        # Create adjusted span
        adjusted_span = span.copy()
        adjusted_span["start"] = span_start + span_offset
        adjusted_span["end"] = span_end + span_offset

        # Verify adjustment
        if 0 <= adjusted_span["start"] < adjusted_span["end"] <= len(cleaned_text):
            adjusted_spans.append(adjusted_span)

    return adjusted_spans


def stratify_by_span_density(records: List[Dict], n_bins: int = 3) -> Dict[str, List]:
    """Stratify records by number of spans.

    Args:
        records: List of complaint records with 'spans' field
        n_bins: Number of density bins (default 3: low/medium/high)

    Returns:
        Dictionary mapping bin names to record lists
    """
    span_counts = [len(r.get("spans", [])) for r in records]

    if not span_counts:
        return {"all": records}

    min_count = min(span_counts)
    max_count = max(span_counts)

    if min_count == max_count:
        return {"all": records}

    bin_size = (max_count - min_count) / n_bins
    bins = defaultdict(list)

    for record in records:
        count = len(record.get("spans", []))
        if count == 0:
            bin_name = "empty"
        else:
            bin_idx = min(int((count - min_count) / bin_size), n_bins - 1)
            bin_name = ["low", "medium", "high"][bin_idx] if n_bins == 3 else f"bin_{bin_idx}"
        bins[bin_name].append(record)

    return dict(bins)


def stratify_by_text_length(records: List[Dict], n_bins: int = 3) -> Dict[str, List]:
    """Stratify records by text character length.

    Args:
        records: List of complaint records with 'text' field
        n_bins: Number of length bins (default 3: short/medium/long)

    Returns:
        Dictionary mapping bin names to record lists
    """
    text_lengths = [len(r.get("text", "")) for r in records]

    if not text_lengths:
        return {"all": records}

    min_len = min(text_lengths)
    max_len = max(text_lengths)

    if min_len == max_len:
        return {"all": records}

    bin_size = (max_len - min_len) / n_bins
    bins = defaultdict(list)

    for record in records:
        length = len(record.get("text", ""))
        if length == 0:
            bin_name = "empty"
        else:
            bin_idx = min(int((length - min_len) / bin_size), n_bins - 1)
            bin_name = ["short", "medium", "long"][bin_idx] if n_bins == 3 else f"bin_{bin_idx}"
        bins[bin_name].append(record)

    return dict(bins)


def stratify_by_category(records: List[Dict]) -> Dict[str, List]:
    """Stratify records by product category (CAERS data).

    Args:
        records: List of complaint records with metadata containing 'product_type'

    Returns:
        Dictionary mapping category names to record lists
    """
    bins = defaultdict(list)

    for record in records:
        category = record.get("metadata", {}).get("product_type", "unknown")
        bins[category].append(record)

    return dict(bins)


def sample_stratified(
    records: List[Dict], n_tasks: int, strategy: str = "span_density", seed: Optional[int] = 42
) -> List[Dict]:
    """Sample tasks using stratified sampling.

    Args:
        records: Full list of candidate records
        n_tasks: Number of tasks to sample
        strategy: Stratification strategy ('span_density', 'text_length', 'category')
        seed: Random seed for reproducibility

    Returns:
        Sampled list of n_tasks records
    """
    if seed is not None:
        random.seed(seed)

    if n_tasks >= len(records):
        return random.sample(records, len(records))

    # Stratify by chosen strategy
    if strategy == "span_density":
        bins = stratify_by_span_density(records)
    elif strategy == "text_length":
        bins = stratify_by_text_length(records)
    elif strategy == "category":
        bins = stratify_by_category(records)
    else:
        bins = {"all": records}

    # Sample proportionally from each bin
    sampled = []
    total_records = len(records)

    for bin_name, bin_records in bins.items():
        bin_proportion = len(bin_records) / total_records
        bin_sample_size = max(1, int(n_tasks * bin_proportion))

        # Ensure we don't oversample
        bin_sample_size = min(bin_sample_size, len(bin_records))

        bin_sample = random.sample(bin_records, bin_sample_size)
        sampled.extend(bin_sample)

    # Adjust if over/under sampled
    if len(sampled) > n_tasks:
        sampled = random.sample(sampled, n_tasks)
    elif len(sampled) < n_tasks:
        # Add more from largest bin
        remaining_needed = n_tasks - len(sampled)
        largest_bin = max(bins.values(), key=len)
        remaining_candidates = [r for r in largest_bin if r not in sampled]
        if remaining_candidates:
            additional = random.sample(
                remaining_candidates, min(remaining_needed, len(remaining_candidates))
            )
            sampled.extend(additional)

    return sampled


def filter_records(
    records: List[Dict],
    min_spans: int = 0,
    max_text_len: Optional[int] = None,
    min_text_len: int = 20,
) -> List[Dict]:
    """Filter records by quality criteria.

    Args:
        records: List of records to filter
        min_spans: Minimum number of spans required
        max_text_len: Maximum text length (None for no limit)
        min_text_len: Minimum text length

    Returns:
        Filtered list of records
    """
    filtered = []

    for record in records:
        text = record.get("text", "")
        spans = record.get("spans", [])

        # Check minimum text length
        if len(text) < min_text_len:
            continue

        # Check maximum text length
        if max_text_len and len(text) > max_text_len:
            continue

        # Check minimum spans
        if len(spans) < min_spans:
            continue

        filtered.append(record)

    return filtered


def prepare_batch(
    input_path: Path,
    output_path: Path,
    n_tasks: int,
    strategy: str = "stratified",
    min_spans: int = 1,
    max_text_len: Optional[int] = None,
    min_text_len: int = 20,
    deidentify: bool = False,
    category_balance: bool = False,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Prepare a production annotation batch.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output batch JSONL file
        n_tasks: Number of tasks to include in batch
        strategy: Sampling strategy ('stratified', 'random', 'balanced')
        min_spans: Minimum spans per task
        max_text_len: Maximum text length (None for no limit)
        min_text_len: Minimum text length
        deidentify: Whether to remove PII
        category_balance: Balance by product category
        seed: Random seed for reproducibility

    Returns:
        Dictionary with batch statistics
    """
    # Load input records
    print(f"Loading records from {input_path}...")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records")

    # Filter by quality criteria
    print(f"Filtering records (min_spans={min_spans}, text_len={min_text_len}-{max_text_len})...")
    filtered_records = filter_records(records, min_spans, max_text_len, min_text_len)
    print(f"Filtered to {len(filtered_records)} records")

    if len(filtered_records) == 0:
        raise ValueError("No records passed filtering criteria")

    # Sample using chosen strategy
    print(f"Sampling {n_tasks} tasks using '{strategy}' strategy...")
    if strategy == "random":
        sampled = random.sample(filtered_records, min(n_tasks, len(filtered_records)))
    elif strategy == "balanced" or category_balance:
        sampled = sample_stratified(filtered_records, n_tasks, "category", seed)
    elif strategy == "stratified":
        sampled = sample_stratified(filtered_records, n_tasks, "span_density", seed)
    else:
        sampled = random.sample(filtered_records, min(n_tasks, len(filtered_records)))

    print(f"Sampled {len(sampled)} tasks")

    # De-identify if requested
    if deidentify:
        print("De-identifying PII...")
        deidentified = []
        total_replacements = 0

        for record in sampled:
            cleaned_text, replacements = deidentify_text(record["text"])
            total_replacements += len(replacements)

            # Adjust spans if text changed
            if replacements:
                adjusted_spans = adjust_spans_for_deidentification(
                    record.get("spans", []), replacements, record["text"], cleaned_text
                )
                record = record.copy()
                record["text"] = cleaned_text
                record["spans"] = adjusted_spans
                if "metadata" not in record:
                    record["metadata"] = {}
                record["metadata"]["deidentified"] = True
                record["metadata"]["pii_replacements"] = len(replacements)

            deidentified.append(record)

        sampled = deidentified
        print(f"Made {total_replacements} PII replacements across {len(sampled)} tasks")

    # Add batch metadata
    batch_id = output_path.stem
    timestamp = datetime.now().isoformat()

    for i, record in enumerate(sampled):
        if "metadata" not in record:
            record["metadata"] = {}
        record["metadata"]["batch_id"] = batch_id
        record["metadata"]["batch_timestamp"] = timestamp
        record["metadata"]["task_number"] = i + 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in sampled:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} tasks to {output_path}")

    # Compute statistics
    span_counts = [len(r.get("spans", [])) for r in sampled]
    text_lengths = [len(r.get("text", "")) for r in sampled]
    label_counts = Counter()
    for record in sampled:
        for span in record.get("spans", []):
            label_counts[span.get("label", "unknown")] += 1

    stats = {
        "batch_id": batch_id,
        "timestamp": timestamp,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "n_tasks": len(sampled),
        "strategy": strategy,
        "deidentified": deidentify,
        "filters": {
            "min_spans": min_spans,
            "max_text_len": max_text_len,
            "min_text_len": min_text_len,
        },
        "statistics": {
            "total_spans": sum(span_counts),
            "avg_spans_per_task": sum(span_counts) / len(sampled) if sampled else 0,
            "avg_text_length": sum(text_lengths) / len(sampled) if sampled else 0,
            "label_distribution": dict(label_counts),
        },
    }

    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare production annotation batches with stratified sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", type=Path, required=True, help="Input JSONL file (weak labels or CAERS data)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output batch JSONL file")
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=100,
        help="Number of tasks to include in batch (default: 100)",
    )
    parser.add_argument(
        "--strategy",
        choices=["stratified", "random", "balanced"],
        default="stratified",
        help="Sampling strategy (default: stratified by span density)",
    )
    parser.add_argument(
        "--min-spans", type=int, default=1, help="Minimum spans per task (default: 1)"
    )
    parser.add_argument(
        "--max-text-len", type=int, default=None, help="Maximum text length (default: no limit)"
    )
    parser.add_argument(
        "--min-text-len", type=int, default=20, help="Minimum text length (default: 20)"
    )
    parser.add_argument("--deidentify", action="store_true", help="Remove PII from text")
    parser.add_argument(
        "--category-balance", action="store_true", help="Balance by product category"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--stats-output", type=Path, default=None, help="Output path for statistics JSON"
    )

    args = parser.parse_args()

    # Prepare batch
    stats = prepare_batch(
        input_path=args.input,
        output_path=args.output,
        n_tasks=args.n_tasks,
        strategy=args.strategy,
        min_spans=args.min_spans,
        max_text_len=args.max_text_len,
        min_text_len=args.min_text_len,
        deidentify=args.deidentify,
        category_balance=args.category_balance,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Batch ID: {stats['batch_id']}")
    print(f"Tasks: {stats['n_tasks']}")
    print(f"Total spans: {stats['statistics']['total_spans']}")
    print(f"Avg spans/task: {stats['statistics']['avg_spans_per_task']:.2f}")
    print(f"Avg text length: {stats['statistics']['avg_text_length']:.1f} chars")
    print(f"\nLabel distribution:")
    for label, count in stats["statistics"]["label_distribution"].items():
        print(f"  {label}: {count}")

    # Save statistics if requested
    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to {args.stats_output}")


if __name__ == "__main__":
    main()
