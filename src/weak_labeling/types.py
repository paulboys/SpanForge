"""
Shared data types for weak labeling.

Defines core data structures used throughout the weak labeling package.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LexiconEntry:
    """Entry in a symptom or product lexicon.

    Attributes:
        term: The text term to match (e.g., "burning sensation")
        canonical: Canonical/normalized form (e.g., "Burning Sensation")
        source: Source of this entry (e.g., "MedDRA", "manual")
        concept_id: Optional concept identifier
        sku: Optional product SKU (for products)
        category: Optional category label
    """

    term: str
    canonical: str
    source: str
    concept_id: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None


@dataclass
class Span:
    """A labeled span of text with metadata.

    Attributes:
        text: The extracted text (e.g., "burning sensation")
        start: Character start position (inclusive)
        end: Character end position (exclusive)
        label: Entity label (e.g., "SYMPTOM", "PRODUCT")
        canonical: Canonical form from lexicon
        source: Source lexicon identifier
        concept_id: Optional concept identifier
        sku: Optional product SKU
        category: Optional category
        confidence: Confidence score (0.0-1.0)
        negated: Whether span is negated
    """

    text: str
    start: int
    end: int
    label: str
    canonical: Optional[str] = None
    source: Optional[str] = None
    concept_id: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 1.0
    negated: bool = False


def load_symptom_lexicon(path: Path) -> List[LexiconEntry]:
    """Load symptom lexicon from CSV file.

    Expected CSV columns: term, canonical, source, concept_id (optional)

    Args:
        path: Path to symptoms.csv file

    Returns:
        List of LexiconEntry objects

    Example:
        >>> from pathlib import Path
        >>> lexicon = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
        >>> len(lexicon) > 0
        True
    """
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
    """Load product lexicon from CSV file.

    Expected CSV columns: term, canonical, sku (optional), category (optional)

    Args:
        path: Path to products.csv file

    Returns:
        List of LexiconEntry objects

    Example:
        >>> from pathlib import Path
        >>> lexicon = load_product_lexicon(Path("data/lexicon/products.csv"))
        >>> len(lexicon) > 0
        True
    """
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
