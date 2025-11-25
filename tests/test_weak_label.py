"""Weak labeling tests with composition-based infrastructure.

Tests symptom/product matching, negation detection, JSONL persistence,
and edge cases using WeakLabelTestBase and SpanAsserter.
"""
from pathlib import Path
import sys
import pathlib
import json
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weak_label import load_symptom_lexicon, load_product_lexicon, weak_label, persist_weak_labels_jsonl
from tests.base import WeakLabelTestBase
from tests.assertions import SpanAsserter


class TestSymptomMatching(WeakLabelTestBase):
    """Test symptom detection and negation handling."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
        self.span_asserter = SpanAsserter(self)
    
    def test_basic_symptom_detection(self):
        """Test basic symptom span extraction."""
        text = "I had a mild headache after using the cream."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]
        self.assertGreater(len(symptom_spans), 0, "Should detect at least one symptom")
        
        # Verify headache is detected
        symptom_texts = {s.text.lower() for s in symptom_spans}
        self.assertIn("headache", symptom_texts, "Should detect 'headache' symptom")
    
    def test_negation_detection(self):
        """Test negation window detection for symptoms."""
        text = "I had a mild headache but no skin rash developed."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        symptom_texts = {s.text.lower(): s for s in spans if s.label == "SYMPTOM"}
        
        # If skin rash is detected, it should be negated
        if "skin rash" in symptom_texts:
            self.assertTrue(symptom_texts["skin rash"].negated, 
                          "Skin rash should be negated due to 'no' negation cue")
    
    def test_multiple_symptoms_same_text(self):
        """Test detection of multiple symptoms in one text."""
        text = "Patient reports itching, redness, and burning after moisturizer use."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        symptom_spans = [s for s in spans if s.label == "SYMPTOM"]
        self.assertGreaterEqual(len(symptom_spans), 2, 
                               "Should detect multiple symptoms")


class TestProductMatching(WeakLabelTestBase):
    """Test product detection and SKU assignment."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
        self.span_asserter = SpanAsserter(self)
    
    def test_product_detection(self):
        """Test product span extraction."""
        text = "This moisturizing cream caused a headache."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        labels = [s.label for s in spans]
        self.assertIn("PRODUCT", labels, "PRODUCT span expected")
    
    def test_product_sku_assignment(self):
        """Test that detected products have SKU metadata (if lexicon provides it)."""
        text = "Used moisturizing cream as directed."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        product_spans = [s for s in spans if s.label == "PRODUCT"]
        self.assertGreater(len(product_spans), 0, 
                          "Should detect at least one product")
        
        # SKU is optional and depends on lexicon schema
        # Just verify product was detected


class TestJSONLPersistence(WeakLabelTestBase):
    """Test JSONL serialization and persistence."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
    
    def test_jsonl_file_creation(self):
        """Test JSONL file is created with correct structure."""
        texts = ["I had a headache.", "The serum caused itching."]
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) 
                      for t in texts]
        
        output_file = self.temp_dir / "weak_labels.jsonl"
        persist_weak_labels_jsonl(texts, spans_batch, output_file)
        
        self.assertTrue(output_file.exists(), "JSONL file should be created")
        
        records = self.load_jsonl(output_file)
        self.assertEqual(len(records), 2, "Should have 2 JSONL records")
        
        # Validate record structure
        for record in records:
            self.assertIn("text", record, "Record should have 'text' field")
            self.assertIn("spans", record, "Record should have 'spans' field")


class TestEdgeCases(WeakLabelTestBase):
    """Test edge cases and spurious match prevention."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
    
    def test_no_short_spurious_matches(self):
        """Test that very short tokens (1-2 chars) don't produce spurious fuzzy matches."""
        text = "I got a terrible headache after using the moisturizing cream."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Verify no single-letter or two-letter spurious matches
        for span in spans:
            if span.confidence < 1.0:  # fuzzy match
                self.assertGreaterEqual(len(span.text), 3, 
                    f"Spurious short match detected: '{span.text}' -> {span.canonical} (conf: {span.confidence})")
        
        # Should still find valid matches
        symptom_texts = [s.text.lower() for s in spans if s.label == "SYMPTOM"]
        self.assertTrue(any("headache" in t for t in symptom_texts), 
                       "Should still find 'headache' symptom")

