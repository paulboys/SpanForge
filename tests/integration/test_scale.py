"""Scale and stress tests for SpanForge pipeline.

Tests performance, memory usage, and correctness with large document volumes.
"""

import json
import time
import unittest
from pathlib import Path
from typing import List

from src.weak_label import (
    load_product_lexicon,
    load_symptom_lexicon,
    persist_weak_labels_jsonl,
    weak_label,
)
from tests.assertions import OverlapChecker, SpanAsserter
from tests.base import IntegrationTestBase


class TestScalePerformance(IntegrationTestBase):
    """Test pipeline performance with increasing document counts."""

    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)
        self.span_asserter = SpanAsserter(self)

    def test_100_documents(self):
        """Test processing 100 documents (fast baseline)."""
        texts = self._generate_test_texts(100)

        start_time = time.time()
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]
        elapsed = time.time() - start_time

        self.assertEqual(len(spans_batch), 100)
        self.assertLess(elapsed, 10.0, f"100 docs should process in <10s, took {elapsed:.2f}s")

        # Verify all completed
        total_spans = sum(len(s) for s in spans_batch)
        self.assertGreater(total_spans, 0, "Should detect spans across 100 docs")

    def test_1000_documents(self):
        """Test processing 1000 documents (stress test)."""
        texts = self._generate_test_texts(1000)

        start_time = time.time()
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]
        elapsed = time.time() - start_time

        self.assertEqual(len(spans_batch), 1000)
        self.assertLess(elapsed, 120.0, f"1000 docs should process in <2min, took {elapsed:.2f}s")

        # Sample validation (don't validate all 1000)
        sample_indices = [0, 100, 500, 999]
        for idx in sample_indices:
            if spans_batch[idx]:
                for span in spans_batch[idx]:
                    span_dict = {"start": span.start, "end": span.end, "label": span.label}
                    self.span_asserter.assert_boundaries_valid(texts[idx], [span_dict])

    def test_jsonl_persistence_large_batch(self):
        """Test JSONL persistence with 500 documents."""
        texts = self._generate_test_texts(500)
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]

        output_file = self.temp_dir / "data" / "output" / "large_batch.jsonl"

        start_time = time.time()
        persist_weak_labels_jsonl(texts, spans_batch, output_file)
        elapsed = time.time() - start_time

        self.assertTrue(output_file.exists())
        self.assertLess(elapsed, 5.0, f"JSONL persistence should be fast, took {elapsed:.2f}s")

        # Verify file integrity
        records = self.load_jsonl(output_file)
        self.assertEqual(len(records), 500)

    def test_memory_efficiency_repeated_processing(self):
        """Test that repeated processing doesn't accumulate memory issues."""
        # Process same 50 docs multiple times
        texts = self._generate_test_texts(50)

        for iteration in range(10):
            spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]
            self.assertEqual(len(spans_batch), 50)

        # If memory leaks exist, this test would slow down or crash
        # Completion indicates memory is released properly

    def _generate_test_texts(self, count: int) -> List[str]:
        """Generate test texts with variety."""
        templates = [
            "Patient {idx} reports headache after using aspirin.",
            "Subject {idx} experienced rash from cream application.",
            "Case {idx}: nausea and headache noted with aspirin.",
            "Report {idx}: mild rash observed after cream use.",
            "Patient {idx} has no adverse effects from aspirin.",
        ]

        texts = []
        for i in range(count):
            template = templates[i % len(templates)]
            texts.append(template.format(idx=i + 1))

        return texts


class TestLongTextHandling(IntegrationTestBase):
    """Test handling of very long texts."""

    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)

    def test_very_long_text_5k_chars(self):
        """Test text with ~5000 characters."""
        base = "Patient reports headache. "
        long_text = base * 200  # ~5000 chars

        start_time = time.time()
        spans = weak_label(long_text, self.symptom_lexicon, self.product_lexicon)
        elapsed = time.time() - start_time

        self.assertLess(elapsed, 2.0, f"Long text should process quickly, took {elapsed:.2f}s")
        self.assertGreater(len(spans), 0, "Should detect spans in long text")

    def test_very_long_text_10k_chars(self):
        """Test text with ~10000 characters."""
        base = "Subject experienced rash after cream. "
        very_long_text = base * 250  # ~10000 chars

        start_time = time.time()
        spans = weak_label(very_long_text, self.symptom_lexicon, self.product_lexicon)
        elapsed = time.time() - start_time

        self.assertLess(elapsed, 5.0, f"Very long text processing, took {elapsed:.2f}s")
        self.assertIsInstance(spans, list)

    def test_single_sentence_repeated_many_times(self):
        """Test repeated pattern detection in long text."""
        sentence = "Patient has headache. "
        repeated_text = sentence * 500  # 500 repetitions

        spans = weak_label(repeated_text, self.symptom_lexicon, self.product_lexicon)

        # Should detect many headache mentions
        headache_spans = [s for s in spans if "headache" in s.text.lower()]
        self.assertGreater(len(headache_spans), 100, "Should detect repeated symptom mentions")


class TestConcurrentDetection(IntegrationTestBase):
    """Test detection patterns with many symptoms/products in one text."""

    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)
        self.overlap_checker = OverlapChecker(self)

    def test_text_with_many_symptoms(self):
        """Test text mentioning multiple different symptoms."""
        text = (
            "Patient reports headache, rash, nausea, and describes "
            "multiple adverse reactions including additional headache "
            "episodes and persistent rash."
        )

        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]

        self.assertGreater(len(symptom_spans), 3, "Should detect multiple symptom mentions")

        # Verify no conflicting overlaps
        span_dicts = [{"start": s.start, "end": s.end, "label": s.label} for s in symptom_spans]
        self.overlap_checker.assert_no_conflicting_labels(span_dicts)

    def test_interleaved_symptoms_and_products(self):
        """Test text with symptoms and products interleaved."""
        text = (
            "Used aspirin, got headache. Tried cream, developed rash. "
            "Switched to aspirin again, nausea occurred. "
            "Stopped cream, rash improved."
        )

        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]
        product_spans = [s for s in spans if s.label == "PRODUCT"]

        self.assertGreater(len(symptom_spans), 2, "Should detect multiple symptoms")
        self.assertGreater(len(product_spans), 1, "Should detect multiple products")

    def test_dense_span_coverage(self):
        """Test text where spans cover most of the text."""
        text = "headache rash nausea headache aspirin cream"

        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)

        # Calculate coverage
        covered_chars = set()
        for span in spans:
            for pos in range(span.start, span.end):
                covered_chars.add(pos)

        coverage_ratio = len(covered_chars) / len(text)
        self.assertGreater(coverage_ratio, 0.5, "Dense text should have >50% span coverage")


class TestRobustness(IntegrationTestBase):
    """Test pipeline robustness to various edge cases at scale."""

    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)

    def test_mixed_valid_and_empty_texts(self):
        """Test batch with mix of valid and empty texts."""
        texts = ["Patient has headache.", "", "Rash from cream.", "   ", "Nausea reported."]

        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]

        self.assertEqual(len(spans_batch), len(texts))

        # Valid texts should have spans, empty should not
        self.assertGreater(len(spans_batch[0]), 0)
        self.assertEqual(len(spans_batch[1]), 0)  # Empty text
        self.assertGreater(len(spans_batch[2]), 0)
        self.assertEqual(len(spans_batch[3]), 0)  # Whitespace
        self.assertGreater(len(spans_batch[4]), 0)

    def test_special_characters_at_scale(self):
        """Test 100 texts with various special characters."""
        special_chars = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]

        texts = [
            f"Patient{char} reports headache{char} after aspirin."
            for char in special_chars * 10  # 100 texts
        ]

        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]

        # All should complete without error
        self.assertEqual(len(spans_batch), 100)

        # Each should still detect symptoms despite special chars
        for spans in spans_batch:
            self.assertGreater(len(spans), 0, "Should detect spans despite special characters")

    def test_unicode_texts_at_scale(self):
        """Test batch processing with unicode content."""
        texts = []
        for i in range(34):  # 34*3 = 102 texts
            texts.extend(
                [
                    f"Patiënt {i} reports hëadachë.",
                    f"Subject {i}: démangeaisons observées.",
                    f"Case {i}: 患者报告头痛 with headache.",
                ]
            )

        spans_batch = [
            weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts[:100]
        ]

        self.assertEqual(len(spans_batch), 100)
        # Should handle unicode gracefully


class TestPerformanceBenchmarks(IntegrationTestBase):
    """Benchmark tests for performance regression detection."""

    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)

    def test_average_time_per_document(self):
        """Benchmark average processing time per document."""
        texts = [f"Patient {i} has headache after aspirin use." for i in range(100)]

        start_time = time.time()
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) for t in texts]
        elapsed = time.time() - start_time

        avg_time = elapsed / len(texts)

        # Should average less than 100ms per doc
        self.assertLess(
            avg_time, 0.1, f"Avg time per doc: {avg_time*1000:.2f}ms (should be <100ms)"
        )

    def test_lexicon_lookup_efficiency(self):
        """Test that lexicon lookups remain efficient with repeated use."""
        text = "Patient has headache, rash, and nausea."

        # Process same text 1000 times to test lookup caching/efficiency
        start_time = time.time()
        for _ in range(1000):
            spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        elapsed = time.time() - start_time

        # 1000 iterations should complete quickly
        self.assertLess(elapsed, 10.0, f"1000 iterations took {elapsed:.2f}s (should be <10s)")


if __name__ == "__main__":
    unittest.main()
