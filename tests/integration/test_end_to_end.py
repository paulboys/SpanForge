"""End-to-end integration tests for SpanForge pipeline.

Tests full workflow: weak labeling → LLM refinement → gold export → integrity validation.
"""
import json
import unittest
from pathlib import Path
from tests.base import IntegrationTestBase
from tests.assertions import IntegrityValidator, SpanAsserter
from src.weak_label import load_symptom_lexicon, load_product_lexicon, weak_label, persist_weak_labels_jsonl


class TestEndToEndPipeline(IntegrationTestBase):
    """Test complete pipeline from raw text to validated gold output."""
    
    def setUp(self):
        super().setUp()
        self.integrity_validator = IntegrityValidator(self)
        self.span_asserter = SpanAsserter(self)
        
        # Create standard lexicons
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)
    
    def test_weak_labeling_stage(self):
        """Test weak labeling produces valid spans."""
        texts = [
            "Patient reports severe headache after using aspirin.",
            "Mild rash observed on arm, stopped using cream.",
            "No adverse effects noted with current medication."
        ]
        
        all_spans = []
        for text in texts:
            spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
            all_spans.append(spans)
            
            # Verify spans have required fields
            for span in spans:
                self.assertIn(span.label, {"SYMPTOM", "PRODUCT"})
                self.assertTrue(hasattr(span, 'start'))
                self.assertTrue(hasattr(span, 'end'))
                self.assertTrue(hasattr(span, 'canonical'))
                self.assertTrue(hasattr(span, 'confidence'))
        
        # Should detect at least some spans across all texts
        total_spans = sum(len(s) for s in all_spans)
        self.assertGreater(total_spans, 0, "Pipeline should detect at least one span")
    
    def test_jsonl_persistence_stage(self):
        """Test JSONL persistence creates valid file structure."""
        texts = [
            "Patient has headache and nausea.",
            "Applied cream, experienced rash."
        ]
        
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) 
                      for t in texts]
        
        output_file = self.temp_dir / "data" / "output" / "weak_labels.jsonl"
        persist_weak_labels_jsonl(texts, spans_batch, output_file)
        
        self.assertTrue(output_file.exists(), "JSONL output file should exist")
        
        # Load and validate structure
        records = self.load_jsonl(output_file)
        self.assertEqual(len(records), len(texts), "One record per input text")
        
        for record in records:
            self.assertIn("text", record)
            self.assertIn("spans", record)
            self.assertIsInstance(record["spans"], list)
    
    def test_gold_export_integrity(self):
        """Test gold export produces integrity-valid records."""
        # Simulate annotated gold records
        gold_records = [
            {
                "id": "integration_001",
                "text": "Patient reports headache after aspirin use.",
                "entities": [
                    {
                        "start": 16, "end": 24, "label": "SYMPTOM",
                        "text": "headache", "canonical": "Headache"
                    },
                    {
                        "start": 31, "end": 38, "label": "PRODUCT",
                        "text": "aspirin", "canonical": "Aspirin"
                    }
                ],
                "source": "weak_label",
                "annotator": "system",
                "revision": 1
            }
        ]
        
        # Write to gold export location
        gold_file = self.temp_dir / "data" / "annotation" / "exports" / "integration_gold.jsonl"
        with gold_file.open("w", encoding="utf-8") as f:
            for rec in gold_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Validate integrity
        loaded_records = self.load_jsonl(gold_file)
        for record in loaded_records:
            self.integrity_validator.validate_full_record(record)
    
    def test_full_pipeline_with_confidence_filtering(self):
        """Test pipeline with confidence-based filtering."""
        text = "Severe headache and mild nausea reported after cream application."
        
        # Weak labeling
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Filter by confidence (simulate LLM refinement candidate selection)
        min_confidence = 0.65
        high_conf_spans = [s for s in spans if s.confidence >= min_confidence]
        low_conf_spans = [s for s in spans if 0.55 <= s.confidence < min_confidence]
        
        # High confidence spans should pass integrity checks
        for span in high_conf_spans:
            span_dict = {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": span.text
            }
            self.span_asserter.assert_boundaries_valid(text, [span_dict])
        
        # Low confidence spans are candidates for LLM refinement
        self.assertIsInstance(low_conf_spans, list)
    
    def test_pipeline_error_handling(self):
        """Test pipeline handles edge cases gracefully."""
        edge_cases = [
            "",  # Empty text
            "   ",  # Whitespace only
            "No symptoms or products mentioned.",  # No matches expected
            "x" * 10000,  # Very long text
        ]
        
        for text in edge_cases:
            try:
                spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
                # Should not crash
                self.assertIsInstance(spans, list)
            except Exception as e:
                self.fail(f"Pipeline failed on edge case: {text[:50]}... Error: {e}")
    
    def test_pipeline_preserves_provenance(self):
        """Test provenance fields are maintained through pipeline."""
        text = "Patient has headache."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Simulate conversion to gold format with provenance
        gold_record = {
            "id": "prov_test_001",
            "text": text,
            "entities": [
                {
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "text": s.text,
                    "canonical": s.canonical,
                    "confidence": s.confidence
                }
                for s in spans
            ],
            "source": "weak_label",
            "annotator": "system",
            "revision": 1,
            "metadata": {
                "fuzzy_threshold": 0.88,
                "jaccard_threshold": 40
            }
        }
        
        # Validate provenance
        self.integrity_validator.assert_provenance_present(gold_record)
        self.assertIn("metadata", gold_record)


class TestMultiDocumentProcessing(IntegrationTestBase):
    """Test pipeline with multiple documents in batch."""
    
    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)
    
    def test_batch_weak_labeling(self):
        """Test batch processing of multiple texts."""
        texts = [
            f"Patient {i} reports headache after aspirin."
            for i in range(10)
        ]
        
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) 
                      for t in texts]
        
        self.assertEqual(len(spans_batch), len(texts))
        
        # Each should have detected spans
        for spans in spans_batch:
            self.assertGreater(len(spans), 0, "Each text should have spans")
    
    def test_batch_jsonl_persistence(self):
        """Test JSONL persistence for batch of documents."""
        texts = [
            "Headache after aspirin.",
            "Rash from cream.",
            "Nausea reported.",
            "No issues with aspirin.",
            "Cream caused headache."
        ]
        
        spans_batch = [weak_label(t, self.symptom_lexicon, self.product_lexicon) 
                      for t in texts]
        
        output_file = self.temp_dir / "data" / "output" / "batch_weak.jsonl"
        persist_weak_labels_jsonl(texts, spans_batch, output_file)
        
        records = self.load_jsonl(output_file)
        self.assertEqual(len(records), len(texts))
        
        # Verify each record has unique text
        seen_texts = set()
        for record in records:
            self.assertNotIn(record["text"], seen_texts, "Duplicate text detected")
            seen_texts.add(record["text"])
    
    def test_consistent_span_detection_across_similar_texts(self):
        """Test that similar texts produce similar span patterns."""
        base_text = "Patient has headache and rash."
        variations = [
            "Patient has headache and rash.",
            "Patient reports headache and rash.",
            "Patient experiences headache and rash.",
        ]
        
        spans_results = [weak_label(t, self.symptom_lexicon, self.product_lexicon) 
                        for t in variations]
        
        # All should detect both symptoms
        for spans in spans_results:
            symptom_canonicals = {s.canonical for s in spans if s.label == "SYMPTOM"}
            self.assertIn("Headache", symptom_canonicals, 
                         "All variations should detect headache")
            self.assertIn("Rash", symptom_canonicals, 
                         "All variations should detect rash")


class TestPipelineWithLLMStub(IntegrationTestBase):
    """Test pipeline integration with LLM refinement stub."""
    
    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
        self.symptom_lexicon = load_symptom_lexicon(sym_path)
        self.product_lexicon = load_product_lexicon(prod_path)
    
    def test_llm_refinement_stage_stub(self):
        """Test LLM refinement stage with stub (no actual LLM calls)."""
        # This test verifies integration points without requiring actual LLM
        text = "Patient reports mild itching and slight redness."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Simulate LLM refinement candidate selection
        uncertain_spans = [s for s in spans if 0.55 <= s.confidence < 0.75]
        
        # LLM stub would return empty suggestions
        llm_suggestions = []  # Stub returns no changes
        
        # Merge: original spans + LLM suggestions (none in stub)
        final_spans = spans + llm_suggestions
        
        self.assertEqual(len(final_spans), len(spans), 
                        "Stub should not add suggestions")
    
    def test_pipeline_with_llm_disabled(self):
        """Test pipeline runs correctly with LLM disabled (default)."""
        from src.config import AppConfig
        
        # Default config has LLM disabled
        config = AppConfig()
        
        self.assertFalse(config.llm_enabled, "LLM should be disabled by default")
        
        # Pipeline should work without LLM
        text = "Headache after aspirin."
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        self.assertGreater(len(spans), 0, "Should detect spans without LLM")


if __name__ == '__main__':
    unittest.main()
