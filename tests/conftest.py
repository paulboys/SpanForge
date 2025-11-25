"""Pytest fixtures for SpanForge test suite.

Provides reusable fixtures for lexicons, configurations, sample texts,
and gold files to eliminate setup duplication across test modules.
"""
from __future__ import annotations
import sys
import pathlib
import pytest
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig


@pytest.fixture
def temp_lexicon(tmp_path: Path) -> Path:
    """Create temporary lexicon directory with sample symptom/product CSVs.
    
    Returns:
        Path to temp lexicon directory
    """
    lexicon_dir = tmp_path / "lexicon"
    lexicon_dir.mkdir()
    
    # Sample symptom lexicon
    symptoms = lexicon_dir / "symptoms.csv"
    symptoms.write_text(
        "canonical,terms,concept_id\n"
        "pruritus,itching|itchy|pruritus,C0033774\n"
        "erythema,redness|erythema|red,C0041834\n"
        "dryness,dryness|dry skin,C0151908\n"
        "burning,burning|burn sensation,C0085624\n",
        encoding="utf-8"
    )
    
    # Sample product lexicon
    products = lexicon_dir / "products.csv"
    products.write_text(
        "canonical,terms,concept_id\n"
        "moisturizer,moisturizer|lotion|cream,P12345\n"
        "cleanser,cleanser|face wash|soap,P67890\n",
        encoding="utf-8"
    )
    
    return lexicon_dir


@pytest.fixture
def mock_config(temp_lexicon: Path) -> AppConfig:
    """Create mock AppConfig with temp lexicon paths.
    
    Args:
        temp_lexicon: Fixture providing temp lexicon directory
    
    Returns:
        AppConfig instance
    """
    return AppConfig(
        model_name="dmis-lab/biobert-base-cased-v1.1",
        max_seq_len=128,
        device="cpu",
        seed=42,
        negation_window=5,
        fuzzy_scorer="WRatio",
        fuzzy_threshold=88,
        jaccard_threshold=40,
        symptom_lexicon=str(temp_lexicon / "symptoms.csv"),
        product_lexicon=str(temp_lexicon / "products.csv"),
        llm_enabled=False,
        llm_provider="stub",
        llm_model="stub",
        llm_min_confidence=0.65,
        llm_cache_path=None,
        llm_prompt_version="v1"
    )


@pytest.fixture
def sample_texts() -> List[str]:
    """Provide sample texts for testing.
    
    Returns:
        List of test strings
    """
    return [
        "Patient reports severe itching and redness after using the moisturizer.",
        "No adverse effects noted.",
        "Mild dryness observed on face, resolved with cleanser change.",
        "Burning sensation and pruritus around eyes, suspect allergic reaction.",
        "Used lotion as directed, experienced no issues.",
    ]


@pytest.fixture
def sample_texts_unicode() -> List[str]:
    """Provide sample texts with unicode/emoji for edge case testing.
    
    Returns:
        List of test strings with unicode
    """
    return [
        "Patiënt reports itching 🔥 after using cream.",
        "患者说用后发痒",  # Chinese: "Patient says itching after use"
        "Rötung und Brennen 😢",  # German: "Redness and burning"
        "Démangeaisons sévères 🤒",  # French: "Severe itching"
    ]


@pytest.fixture
def gold_files(tmp_path: Path) -> List[Path]:
    """Create temporary gold JSONL files for integrity testing.
    
    Args:
        tmp_path: Pytest-provided temp directory
    
    Returns:
        List of paths to gold JSONL files
    """
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    
    # Valid gold file
    valid_gold = gold_dir / "valid.jsonl"
    records = [
        {
            "id": "test_001",
            "text": "Patient reports itching and redness.",
            "entities": [
                {"start": 16, "end": 23, "label": "SYMPTOM", "text": "itching", "canonical": "pruritus"},
                {"start": 28, "end": 35, "label": "SYMPTOM", "text": "redness", "canonical": "erythema"}
            ],
            "source": "weak_label",
            "annotator": "system",
            "revision": 1
        },
        {
            "id": "test_002",
            "text": "Used moisturizer without issues.",
            "entities": [
                {"start": 5, "end": 16, "label": "PRODUCT", "text": "moisturizer", "canonical": "moisturizer"}
            ],
            "source": "weak_label",
            "annotator": "system",
            "revision": 1
        }
    ]
    with valid_gold.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Gold file with edge cases (negation, overlaps)
    edge_gold = gold_dir / "edge_cases.jsonl"
    edge_records = [
        {
            "id": "edge_001",
            "text": "No itching or redness observed.",
            "entities": [
                {"start": 3, "end": 10, "label": "SYMPTOM", "text": "itching", "canonical": "pruritus", "negated": True},
                {"start": 14, "end": 21, "label": "SYMPTOM", "text": "redness", "canonical": "erythema", "negated": True}
            ],
            "source": "manual",
            "annotator": "user_001",
            "revision": 2
        }
    ]
    with edge_gold.open("w", encoding="utf-8") as f:
        for rec in edge_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    return [valid_gold, edge_gold]


@pytest.fixture
def sample_weak_labels() -> List[Dict[str, Any]]:
    """Provide sample weak label spans for testing.
    
    Returns:
        List of span dicts with confidence scores
    """
    return [
        {
            "start": 16, "end": 23, "label": "SYMPTOM",
            "text": "itching", "canonical": "pruritus",
            "confidence": 0.92, "fuzzy_score": 0.95, "jaccard_score": 0.85
        },
        {
            "start": 28, "end": 35, "label": "SYMPTOM",
            "text": "redness", "canonical": "erythema",
            "confidence": 0.88, "fuzzy_score": 0.90, "jaccard_score": 0.80
        },
        {
            "start": 50, "end": 61, "label": "PRODUCT",
            "text": "moisturizer", "canonical": "moisturizer",
            "confidence": 0.95, "fuzzy_score": 1.0, "jaccard_score": 0.75
        },
        # Low confidence span (should be filtered in some tests)
        {
            "start": 70, "end": 75, "label": "SYMPTOM",
            "text": "marks", "canonical": "unknown",
            "confidence": 0.45, "fuzzy_score": 0.50, "jaccard_score": 0.35
        }
    ]


@pytest.fixture
def sample_overlap_spans() -> List[Dict[str, Any]]:
    """Provide sample spans with various overlap patterns.
    
    Returns:
        List of span dicts demonstrating overlap scenarios
    """
    return [
        # Adjacent (no overlap)
        {"start": 0, "end": 5, "label": "SYMPTOM", "text": "itch", "canonical": "pruritus"},
        {"start": 6, "end": 10, "label": "SYMPTOM", "text": "burn", "canonical": "burning"},
        
        # Nested overlap (same label)
        {"start": 15, "end": 30, "label": "SYMPTOM", "text": "severe itching", "canonical": "pruritus"},
        {"start": 22, "end": 30, "label": "SYMPTOM", "text": "itching", "canonical": "pruritus"},
        
        # Partial overlap (different labels - conflict)
        {"start": 35, "end": 45, "label": "SYMPTOM", "text": "dry skin", "canonical": "dryness"},
        {"start": 40, "end": 50, "label": "PRODUCT", "text": "skin cream", "canonical": "moisturizer"},
        
        # Exact duplicate
        {"start": 55, "end": 65, "label": "SYMPTOM", "text": "redness", "canonical": "erythema"},
        {"start": 55, "end": 65, "label": "SYMPTOM", "text": "redness", "canonical": "erythema"},
    ]
