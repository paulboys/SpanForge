"""Curation integrity tests with IntegrityValidator composition.

Validates gold JSONL files for provenance, canonical presence, overlap rules,
and boundary correctness using composition-based assertions.
"""

import json
import unittest
from pathlib import Path

import pytest

from tests.assertions import IntegrityValidator, OverlapChecker, SpanAsserter
from tests.base import IntegrationTestBase

ALLOWED_LABELS = {"SYMPTOM", "PRODUCT"}
EXPORT_DIR = Path("data/annotation/exports")


def _iter_jsonl(path: Path):
    """Iterate over JSONL records in a file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_gold_files():
    """Discover gold JSONL files for parametrization (only files with 'gold' in name)."""
    if not EXPORT_DIR.exists():
        return []
    return list(EXPORT_DIR.glob("*gold*.jsonl"))


class TestCurationIntegrity(unittest.TestCase):
    """Integrity tests for gold annotation files."""

    def setUp(self):
        self.integrity_validator = IntegrityValidator(self)
        self.overlap_checker = OverlapChecker(self)
        self.span_asserter = SpanAsserter(self)


@pytest.mark.parametrize("gold_file", find_gold_files() or [None])
def test_curation_integrity(gold_file):
    """Parametrized test for all gold files."""
    if gold_file is None:
        pytest.skip("No gold JSONL files present; skipping curation integrity test.")

    # Create test instance for validators
    test_case = TestCurationIntegrity()
    test_case.setUp()

    for record in _iter_jsonl(gold_file):
        # Run full integrity validation
        test_case.integrity_validator.validate_full_record(record)

        # Additional record-level checks
        if "revision" in record:
            assert (
                isinstance(record["revision"], int) and record["revision"] >= 0
            ), "Revision must be non-negative int"

        entities = record.get("entities", [])

        # Verify concept_id format if present
        for ent in entities:
            if "concept_id" in ent:
                assert (
                    isinstance(ent["concept_id"], str) and ent["concept_id"]
                ), f"Invalid concept_id in entity: {ent}"
