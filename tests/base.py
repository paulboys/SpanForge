"""Base test infrastructure for SpanForge test suite.

Provides abstract base classes with common setup/teardown, temp file management,
and fixture utilities to reduce code duplication across test modules.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch


class TestBase(unittest.TestCase):
    """Abstract base class for all SpanForge tests.

    Provides:
    - Temporary directory management per test
    - Config mocking utilities
    - Lexicon file builders
    - JSONL fixture loaders
    """

    def setUp(self):
        """Create temporary directory for test isolation."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="spanforge_test_"))
        self.temp_files: List[Path] = []

    def tearDown(self):
        """Clean up temporary files and directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_temp_file(self, name: str, content: str = "") -> Path:
        """Create a temporary file with optional content.

        Args:
            name: Filename (can include subdirs like "data/test.csv")
            content: File content as string

        Returns:
            Path to created file
        """
        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        self.temp_files.append(file_path)
        return file_path

    def create_lexicon_csv(self, name: str, entries: List[Dict[str, str]]) -> Path:
        """Create a lexicon CSV file with headers.

        Args:
            name: Filename
            entries: List of dicts with keys matching CSV columns

        Returns:
            Path to created CSV
        """
        if not entries:
            return self.create_temp_file(name, "term,canonical,source\n")

        file_path = self.temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(entries[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)

        self.temp_files.append(file_path)
        return file_path

    def create_jsonl_file(self, name: str, records: List[Dict[str, Any]]) -> Path:
        """Create a JSONL file.

        Args:
            name: Filename
            records: List of dictionaries to serialize

        Returns:
            Path to created JSONL file
        """
        content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        return self.create_temp_file(name, content)

    def load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load records from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of parsed records
        """
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def mock_config(self, **overrides) -> patch:
        """Create a mock config context manager.

        Args:
            **overrides: Config attributes to override

        Returns:
            Context manager for patching AppConfig

        Example:
            with self.mock_config(llm_enabled=True, device="cpu"):
                # test code
        """
        from src.config import AppConfig

        mock_cfg = AppConfig(**overrides)
        return patch("src.config.get_config", return_value=mock_cfg)


class WeakLabelTestBase(TestBase):
    """Base class for weak labeling tests with lexicon fixtures."""

    def setUp(self):
        super().setUp()
        self.symptom_lexicon_path: Optional[Path] = None
        self.product_lexicon_path: Optional[Path] = None

    def create_symptom_lexicon(self, entries: Optional[List[Dict[str, str]]] = None):
        """Create symptom lexicon with standard schema and load it.

        Args:
            entries: Lexicon entries. If None, uses default entries.

        Returns:
            Loaded symptom lexicon data
        """
        if entries is None:
            # Default symptom lexicon entries
            entries = [
                {"term": "headache", "canonical": "Headache", "source": "test"},
                {"term": "itching", "canonical": "Pruritus", "source": "test"},
                {"term": "redness", "canonical": "Erythema", "source": "test"},
                {"term": "rash", "canonical": "Skin Rash", "source": "test"},
                {"term": "nausea", "canonical": "Nausea", "source": "test"},
                {"term": "dryness", "canonical": "Dryness", "source": "test"},
                {"term": "burning", "canonical": "Burning", "source": "test"},
                {"term": "skin rash", "canonical": "Skin Rash", "source": "test"},
            ]
        self.symptom_lexicon_path = self.create_lexicon_csv("symptoms.csv", entries)
        return self.load_lexicon(self.symptom_lexicon_path)

    def create_product_lexicon(self, entries: Optional[List[Dict[str, str]]] = None):
        """Create product lexicon with standard schema and load it.

        Args:
            entries: Lexicon entries. If None, uses default entries.

        Returns:
            Loaded product lexicon data
        """
        if entries is None:
            # Default product lexicon entries
            entries = [
                {"term": "aspirin", "canonical": "Aspirin", "source": "test"},
                {"term": "cream", "canonical": "Topical Cream", "source": "test"},
                {"term": "moisturizing cream", "canonical": "Moisturizing Cream", "source": "test"},
                {"term": "serum", "canonical": "Serum", "source": "test"},
            ]
        self.product_lexicon_path = self.create_lexicon_csv("products.csv", entries)
        return self.load_lexicon(self.product_lexicon_path)

    def load_lexicon(self, path: Path):
        """Load lexicon entries using weak_label module."""
        from src.weak_label import load_product_lexicon, load_symptom_lexicon

        if "symptom" in str(path):
            return load_symptom_lexicon(path)
        else:
            return load_product_lexicon(path)


class IntegrationTestBase(TestBase):
    """Base class for integration tests requiring full pipeline components."""

    def setUp(self):
        super().setUp()
        # Create standard data structure
        (self.temp_dir / "data" / "lexicon").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "data" / "output").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "data" / "annotation" / "exports").mkdir(parents=True, exist_ok=True)

    def create_standard_lexicons(self):
        """Create minimal symptom and product lexicons for integration tests."""
        symptoms = [
            {"term": "headache", "canonical": "Headache", "source": "test"},
            {"term": "rash", "canonical": "Rash", "source": "test"},
            {"term": "nausea", "canonical": "Nausea", "source": "test"},
        ]
        products = [
            {"term": "aspirin", "canonical": "Aspirin", "source": "test"},
            {"term": "cream", "canonical": "Topical Cream", "source": "test"},
        ]
        sym_path = self.temp_dir / "data" / "lexicon" / "symptoms.csv"
        prod_path = self.temp_dir / "data" / "lexicon" / "products.csv"

        self.create_lexicon_csv(str(sym_path.relative_to(self.temp_dir)), symptoms)
        self.create_lexicon_csv(str(prod_path.relative_to(self.temp_dir)), products)

        return sym_path, prod_path
