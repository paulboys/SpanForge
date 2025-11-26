"""
FDA CAERS Data Download and Preprocessing Script

Downloads FDA CAERS (Consumer Adverse Event Reporting System) data and
converts it to SpanForge JSONL format with weak labeling.

Features:
- Downloads latest CAERS quarterly data from FDA
- Filters for cosmetics, personal care, and related product categories
- Extracts symptom narratives and product information
- Applies weak labeling using SpanForge lexicons
- Exports to JSONL format for annotation workflow

Usage:
    python scripts/caers/download_caers.py --output data/caers/caers_labeled.jsonl --limit 1000
    python scripts/caers/download_caers.py --categories cosmetics supplements --validate
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import AppConfig, get_config
from src.weak_label import (
    LexiconEntry,
    Span,
    load_product_lexicon,
    load_symptom_lexicon,
    weak_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FDA CAERS data URL (updated quarterly)
CAERS_DOWNLOAD_URL = "https://www.fda.gov/media/180475/download"
CAERS_FILENAME = "CAERS_ASCII_2014-PRESENT.csv"

# Category filters for household/personal care products
PRODUCT_CATEGORIES = {
    "cosmetics": ["cosmetics", "cosmetic", "makeup", "beauty"],
    "supplements": ["dietary supplement", "vitamin", "supplement"],
    "foods": ["food", "beverage", "drink"],
    "personal_care": ["personal care", "toiletries", "hygiene"],
    "baby": ["infant", "baby", "pediatric"],
}


def download_caers_data(output_dir: Path, force: bool = False) -> Path:
    """Download latest FDA CAERS data.

    Args:
        output_dir: Directory to save downloaded file
        force: Force re-download even if file exists

    Returns:
        Path to downloaded CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CAERS_FILENAME

    if output_path.exists() and not force:
        logger.info(f"CAERS data already exists at {output_path}")
        return output_path

    logger.info(f"Downloading CAERS data from {CAERS_DOWNLOAD_URL}")
    try:
        urlretrieve(CAERS_DOWNLOAD_URL, output_path)
        logger.info(f"Downloaded {output_path.stat().st_size / 1e6:.1f} MB")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download CAERS data: {e}")
        raise


def load_caers_data(csv_path: Path) -> pd.DataFrame:
    """Load CAERS CSV data with proper encoding and column handling.

    Args:
        csv_path: Path to CAERS CSV file

    Returns:
        DataFrame with CAERS records
    """
    logger.info(f"Loading CAERS data from {csv_path}")
    try:
        # CAERS files use Latin-1 encoding and have variable columns
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        logger.info(f"Loaded {len(df):,} CAERS records")
        logger.info(f"Columns: {', '.join(df.columns[:10])}...")
        return df
    except Exception as e:
        logger.error(f"Failed to load CAERS data: {e}")
        raise


def filter_by_category(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """Filter CAERS data by product category.

    Args:
        df: CAERS DataFrame
        categories: List of category keys from PRODUCT_CATEGORIES

    Returns:
        Filtered DataFrame
    """
    if not categories:
        return df

    # Build regex pattern from selected categories
    patterns = []
    for cat in categories:
        if cat in PRODUCT_CATEGORIES:
            patterns.extend(PRODUCT_CATEGORIES[cat])
        else:
            logger.warning(f"Unknown category: {cat}")

    if not patterns:
        return df

    pattern = "|".join(patterns)
    logger.info(f"Filtering by category pattern: {pattern}")

    # Try common column names for product category/role
    category_columns = ["Product Role", "Product_Role", "PRODUCT ROLE", "Role"]
    for col in category_columns:
        if col in df.columns:
            filtered = df[df[col].fillna("").str.contains(pattern, case=False, regex=True)]
            logger.info(f"Filtered to {len(filtered):,} records using column '{col}'")
            return filtered

    logger.warning("No category column found, returning full dataset")
    return df


def extract_complaint_text(row: pd.Series) -> Optional[str]:
    """Extract complaint narrative text from CAERS row.

    Args:
        row: Single CAERS record

    Returns:
        Complaint text or None if no text available
    """
    # Try common narrative column names
    text_columns = [
        "PRI_Reported Adverse Events",
        "PRI Reported Brand/Product Name",
        "Symptom(s)",
        "Symptom",
        "Adverse Event",
        "Event Description",
    ]

    texts = []
    for col in text_columns:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text and text.lower() not in ["nan", "none", ""]:
                texts.append(text)

    if not texts:
        return None

    # Combine multiple text fields with separator
    combined = " | ".join(texts)
    return combined if len(combined) > 20 else None  # Minimum text length


def extract_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract metadata from CAERS row.

    Args:
        row: Single CAERS record

    Returns:
        Dictionary of metadata fields
    """
    metadata = {}

    # Common metadata columns
    meta_columns = {
        "report_id": ["RA_Report #", "Report_ID", "Report Number"],
        "date_created": ["RA_CAERS Created Date", "Date_Created", "Created Date"],
        "product_type": ["Product Role", "Product_Role", "Role"],
        "product_name": [
            "PRI Reported Brand/Product Name",
            "Brand_Product_Name",
            "Product Name",
        ],
        "age": ["CI_Age at Adverse Event", "Age", "Patient Age"],
        "gender": ["CI_Gender", "Gender", "Sex"],
        "outcomes": ["AEC_Outcome(s)", "Outcomes", "Outcome"],
    }

    for key, possible_cols in meta_columns.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                metadata[key] = str(row[col]).strip()
                break

    return metadata


def create_complaint_record(
    complaint_text: str,
    spans: List[Span],
    metadata: Dict[str, Any],
    source: str = "FDA_CAERS",
) -> Dict[str, Any]:
    """Create SpanForge JSONL record from complaint data.

    Args:
        complaint_text: Full complaint text
        spans: Weak labeled spans
        metadata: Complaint metadata
        source: Data source identifier

    Returns:
        JSONL record dictionary
    """
    return {
        "text": complaint_text,
        "source": source,
        "metadata": metadata,
        "date_processed": datetime.now().isoformat(),
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


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate SpanForge JSONL record.

    Args:
        record: JSONL record dictionary

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check required fields
    if not record.get("text"):
        issues.append("Missing or empty text field")

    if "spans" not in record:
        issues.append("Missing spans field")
    elif not isinstance(record["spans"], list):
        issues.append("Spans field must be a list")

    # Check text length
    text = record.get("text", "")
    if len(text) < 20:
        issues.append(f"Text too short: {len(text)} chars (minimum 20)")
    if len(text) > 5000:
        issues.append(f"Text too long: {len(text)} chars (maximum 5000)")

    # Check spans validity
    spans = record.get("spans", [])
    for i, span in enumerate(spans):
        if "start" not in span or "end" not in span:
            issues.append(f"Span {i}: Missing start or end position")
            continue

        start, end = span["start"], span["end"]
        if start < 0 or end > len(text) or start >= end:
            issues.append(f"Span {i}: Invalid position (start={start}, end={end})")

        # Verify text slice matches span text
        span_text = span.get("text", "")
        actual_text = text[start:end]
        if span_text != actual_text:
            issues.append(f"Span {i}: Text mismatch ('{span_text}' vs '{actual_text}')")

    return len(issues) == 0, issues


def process_caers_to_jsonl(
    csv_path: Path,
    output_path: Path,
    symptom_lexicon: List[LexiconEntry],
    product_lexicon: List[LexiconEntry],
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
    validate: bool = True,
    min_spans: int = 1,
) -> Dict[str, Any]:
    """Process CAERS CSV to SpanForge JSONL format with weak labeling.

    Args:
        csv_path: Path to CAERS CSV file
        output_path: Path to output JSONL file
        symptom_lexicon: Symptom lexicon entries
        product_lexicon: Product lexicon entries
        categories: Product categories to filter (None = all)
        limit: Maximum records to process (None = all)
        validate: Enable validation checks
        min_spans: Minimum spans required per complaint

    Returns:
        Statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("FDA CAERS to SpanForge JSONL Processing Pipeline")
    logger.info("=" * 80)

    # Load and filter data
    df = load_caers_data(csv_path)

    if categories:
        df = filter_by_category(df, categories)

    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit:,} records")

    # Initialize statistics
    stats = {
        "total_rows": len(df),
        "processed": 0,
        "skipped_no_text": 0,
        "skipped_no_spans": 0,
        "validation_failed": 0,
        "successful": 0,
        "total_spans": 0,
        "symptom_spans": 0,
        "product_spans": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = get_config()

    logger.info(f"Processing {stats['total_rows']:,} CAERS records...")
    logger.info(f"Symptom lexicon: {len(symptom_lexicon)} terms")
    logger.info(f"Product lexicon: {len(product_lexicon)} terms")

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            stats["processed"] += 1

            # Extract complaint text
            text = extract_complaint_text(row)
            if not text:
                stats["skipped_no_text"] += 1
                continue

            # Apply weak labeling
            spans = weak_label(
                text,
                symptom_lexicon,
                product_lexicon,
                negation_window=config.negation_window,
                scorer=config.fuzzy_scorer,
            )

            if len(spans) < min_spans:
                stats["skipped_no_spans"] += 1
                continue

            # Extract metadata
            metadata = extract_metadata(row)

            # Create JSONL record
            record = create_complaint_record(text, spans, metadata)

            # Validate if requested
            if validate:
                is_valid, issues = validate_record(record)
                if not is_valid:
                    stats["validation_failed"] += 1
                    logger.warning(f"Record {idx} validation failed: {'; '.join(issues)}")
                    continue

            # Write record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update statistics
            stats["successful"] += 1
            stats["total_spans"] += len(spans)
            stats["symptom_spans"] += sum(1 for s in spans if s.label == "SYMPTOM")
            stats["product_spans"] += sum(1 for s in spans if s.label == "PRODUCT")

            # Progress logging
            if stats["successful"] % 100 == 0:
                logger.info(
                    f"Processed {stats['successful']:,} complaints "
                    f"({stats['total_spans']:,} spans)"
                )

    logger.info("=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f"Total rows processed: {stats['processed']:,}")
    logger.info(f"Successful records: {stats['successful']:,}")
    logger.info(f"Skipped (no text): {stats['skipped_no_text']:,}")
    logger.info(f"Skipped (no spans): {stats['skipped_no_spans']:,}")
    logger.info(f"Validation failed: {stats['validation_failed']:,}")
    logger.info(f"Total spans: {stats['total_spans']:,}")
    logger.info(f"  - Symptom spans: {stats['symptom_spans']:,}")
    logger.info(f"  - Product spans: {stats['product_spans']:,}")
    logger.info(
        f"Average spans per complaint: {stats['total_spans'] / max(stats['successful'], 1):.2f}"
    )
    logger.info(f"Output: {output_path}")

    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download and process FDA CAERS data for SpanForge NER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/caers/caers_labeled.jsonl"),
        help="Output JSONL file path (default: data/caers/caers_labeled.jsonl)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/caers/raw"),
        help="Directory for downloaded CSV (default: data/caers/raw)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(PRODUCT_CATEGORIES.keys()),
        help="Filter by product categories (cosmetics, supplements, foods, personal_care, baby)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of complaints to process (default: all)",
    )
    parser.add_argument(
        "--min-spans",
        type=int,
        default=1,
        help="Minimum spans required per complaint (default: 1)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if CSV exists",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable validation checks (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable validation checks",
    )
    parser.add_argument(
        "--symptom-lexicon",
        type=Path,
        default=Path("data/lexicon/symptoms.csv"),
        help="Path to symptom lexicon CSV (default: data/lexicon/symptoms.csv)",
    )
    parser.add_argument(
        "--product-lexicon",
        type=Path,
        default=Path("data/lexicon/products.csv"),
        help="Path to product lexicon CSV (default: data/lexicon/products.csv)",
    )

    args = parser.parse_args()

    # Step 1: Download CAERS data
    csv_path = download_caers_data(args.download_dir, force=args.force_download)

    # Step 2: Load lexicons
    logger.info("Loading lexicons...")
    symptom_lexicon = load_symptom_lexicon(args.symptom_lexicon)
    product_lexicon = load_product_lexicon(args.product_lexicon)

    if not symptom_lexicon:
        logger.warning(f"No symptom lexicon loaded from {args.symptom_lexicon}")
    if not product_lexicon:
        logger.warning(f"No product lexicon loaded from {args.product_lexicon}")

    # Step 3: Process to JSONL with weak labeling
    stats = process_caers_to_jsonl(
        csv_path=csv_path,
        output_path=args.output,
        symptom_lexicon=symptom_lexicon,
        product_lexicon=product_lexicon,
        categories=args.categories,
        limit=args.limit,
        validate=args.validate,
        min_spans=args.min_spans,
    )

    # Step 4: Write statistics report
    stats_path = args.output.parent / f"{args.output.stem}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
