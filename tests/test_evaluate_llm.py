"""Tests for LLM evaluation metrics and harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    calibration_curve,
    compute_boundary_precision,
    compute_correction_rate,
    compute_iou,
    compute_iou_delta,
    compute_overlap,
    compute_precision_recall_f1,
    stratify_by_confidence,
    stratify_by_label,
    stratify_by_span_length,
)


class TestBasicMetrics:
    """Test basic span comparison metrics."""

    def test_compute_overlap_exact(self):
        """Test overlap computation for exact match."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 10, "end": 20}
        assert compute_overlap(span1, span2) == 10

    def test_compute_overlap_partial(self):
        """Test overlap computation for partial overlap."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 15, "end": 25}
        assert compute_overlap(span1, span2) == 5

    def test_compute_overlap_none(self):
        """Test overlap computation for no overlap."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 25, "end": 35}
        assert compute_overlap(span1, span2) == 0

    def test_compute_iou_exact(self):
        """Test IOU for exact match."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 10, "end": 20}
        assert compute_iou(span1, span2) == 1.0

    def test_compute_iou_partial(self):
        """Test IOU for partial overlap."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 15, "end": 25}
        # Overlap=5, Union=15, IOU=5/15=0.333...
        assert abs(compute_iou(span1, span2) - 0.333) < 0.01

    def test_compute_iou_none(self):
        """Test IOU for no overlap."""
        span1 = {"start": 10, "end": 20}
        span2 = {"start": 25, "end": 35}
        assert compute_iou(span1, span2) == 0.0


class TestBoundaryPrecision:
    """Test boundary precision calculations."""

    def test_boundary_precision_perfect(self):
        """Test perfect boundary match."""
        pred = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_boundary_precision(pred, gold)
        
        assert result["exact_match_rate"] == 1.0
        assert result["mean_iou"] == 1.0

    def test_boundary_precision_partial(self):
        """Test partial boundary match."""
        pred = [{"start": 10, "end": 22, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_boundary_precision(pred, gold)
        
        assert result["exact_match_rate"] == 0.0  # Not exact
        assert 0.8 < result["mean_iou"] < 0.9  # Partial match

    def test_boundary_precision_empty(self):
        """Test with empty predictions."""
        result = compute_boundary_precision([], [{"start": 10, "end": 20, "label": "SYMPTOM"}])
        
        assert result["exact_match_rate"] == 0.0
        assert result["mean_iou"] == 0.0


class TestIOUDelta:
    """Test IOU delta (improvement) calculations."""

    def test_iou_delta_improvement(self):
        """Test LLM improves weak labels."""
        weak = [{"start": 10, "end": 25, "label": "SYMPTOM"}]
        llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_iou_delta(weak, llm, gold)
        
        assert result["llm_mean_iou"] > result["weak_mean_iou"]
        assert result["delta"] > 0
        assert result["improvement_pct"] > 0

    def test_iou_delta_no_change(self):
        """Test LLM doesn't change spans."""
        weak = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_iou_delta(weak, llm, gold)
        
        assert result["delta"] == 0.0
        assert result["improvement_pct"] == 0.0


class TestCorrectionRate:
    """Test correction rate tracking."""

    def test_correction_rate_improved(self):
        """Test LLM improves weak span."""
        weak = [{"start": 10, "end": 25, "label": "SYMPTOM"}]
        llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_correction_rate(weak, llm, gold)
        
        assert result["total_spans"] == 1
        assert result["modified_count"] == 1
        assert result["improved_count"] == 1
        assert result["improvement_rate"] == 1.0

    def test_correction_rate_worsened(self):
        """Test LLM worsens weak span."""
        weak = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        llm = [{"start": 10, "end": 25, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_correction_rate(weak, llm, gold)
        
        assert result["worsened_count"] == 1
        assert result["false_refinement_rate"] > 0

    def test_correction_rate_unchanged(self):
        """Test LLM doesn't modify spans."""
        weak = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        llm = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_correction_rate(weak, llm, gold)
        
        assert result["modified_count"] == 0
        assert result["unchanged_count"] == 1


class TestCalibration:
    """Test confidence calibration curve."""

    def test_calibration_curve_basic(self):
        """Test basic calibration curve computation."""
        spans = [
            {"start": 10, "end": 20, "label": "SYMPTOM", "confidence": 0.9},
            {"start": 30, "end": 40, "label": "SYMPTOM", "confidence": 0.5}
        ]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        curve = calibration_curve(spans, gold, num_bins=5)
        
        assert len(curve["bin_centers"]) == 5
        assert len(curve["accuracy"]) == 5
        assert len(curve["counts"]) == 5

    def test_calibration_curve_perfect(self):
        """Test calibration with perfect predictions."""
        spans = [{"start": 10, "end": 20, "label": "SYMPTOM", "confidence": 0.9}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        curve = calibration_curve(spans, gold, num_bins=10)
        
        # High confidence bin should have 100% accuracy
        high_conf_bin = curve["accuracy"][-1]
        assert high_conf_bin == 1.0 or curve["counts"][-1] == 0


class TestStratification:
    """Test stratification helpers."""

    def test_stratify_by_confidence(self):
        """Test confidence-based stratification."""
        spans = [
            {"start": 0, "end": 5, "confidence": 0.65},
            {"start": 10, "end": 15, "confidence": 0.95},
            {"start": 20, "end": 25, "confidence": 0.75}
        ]
        
        result = stratify_by_confidence(spans)
        
        assert "0.60-0.70" in result
        assert "0.90-1.00" in result
        assert len(result["0.60-0.70"]) == 1
        assert len(result["0.90-1.00"]) == 1

    def test_stratify_by_label(self):
        """Test label-based stratification."""
        spans = [
            {"start": 0, "end": 5, "label": "SYMPTOM"},
            {"start": 10, "end": 15, "label": "PRODUCT"},
            {"start": 20, "end": 25, "label": "SYMPTOM"}
        ]
        
        result = stratify_by_label(spans)
        
        assert "SYMPTOM" in result
        assert "PRODUCT" in result
        assert len(result["SYMPTOM"]) == 2
        assert len(result["PRODUCT"]) == 1

    def test_stratify_by_span_length(self):
        """Test span length stratification."""
        spans = [
            {"start": 0, "end": 4, "text": "rash"},
            {"start": 10, "end": 27, "text": "burning sensation"},
            {"start": 30, "end": 35, "text": "itch"}
        ]
        
        result = stratify_by_span_length(spans)
        
        assert "single_word" in result
        assert "multi_word" in result
        assert len(result["single_word"]) == 2
        assert len(result["multi_word"]) == 1


class TestPrecisionRecallF1:
    """Test precision/recall/F1 calculations."""

    def test_prf_perfect(self):
        """Test perfect P/R/F1."""
        pred = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_prf_false_positive(self):
        """Test with false positive."""
        pred = [
            {"start": 10, "end": 20, "label": "SYMPTOM"},
            {"start": 30, "end": 40, "label": "SYMPTOM"}
        ]
        gold = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        
        result = compute_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 0.5  # 1 TP, 1 FP
        assert result["recall"] == 1.0  # 1 TP, 0 FN
        assert 0.6 < result["f1"] < 0.7

    def test_prf_false_negative(self):
        """Test with false negative."""
        pred = [{"start": 10, "end": 20, "label": "SYMPTOM"}]
        gold = [
            {"start": 10, "end": 20, "label": "SYMPTOM"},
            {"start": 30, "end": 40, "label": "SYMPTOM"}
        ]
        
        result = compute_precision_recall_f1(pred, gold)
        
        assert result["precision"] == 1.0  # 1 TP, 0 FP
        assert result["recall"] == 0.5  # 1 TP, 1 FN
        assert 0.6 < result["f1"] < 0.7

    def test_prf_empty_predictions(self):
        """Test with no predictions."""
        result = compute_precision_recall_f1([], [{"start": 10, "end": 20, "label": "SYMPTOM"}])
        
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


class TestEndToEnd:
    """End-to-end integration tests using fixtures."""

    @pytest.fixture
    def fixture_dir(self):
        """Get fixtures directory path."""
        return Path(__file__).parent / "fixtures" / "annotation"

    def test_load_weak_fixture(self, fixture_dir):
        """Test loading weak label fixture."""
        weak_path = fixture_dir / "weak_baseline.jsonl"
        
        records = []
        with open(weak_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        assert len(records) == 3
        assert "spans" in records[0]
        assert len(records[0]["spans"]) > 0

    def test_load_refined_fixture(self, fixture_dir):
        """Test loading LLM-refined fixture."""
        refined_path = fixture_dir / "gold_with_llm_refined.jsonl"
        
        records = []
        with open(refined_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        assert len(records) == 3
        assert "llm_suggestions" in records[0]
        assert "llm_meta" in records[0]

    def test_load_gold_fixture(self, fixture_dir):
        """Test loading gold standard fixture."""
        gold_path = fixture_dir / "gold_standard.jsonl"
        
        records = []
        with open(gold_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        assert len(records) == 3
        # Check that spans have gold provenance
        assert all(len(r.get("spans", [])) > 0 for r in records)
        assert all(s.get("source") == "gold" for r in records for s in r.get("spans", []))

    def test_fixture_alignment(self, fixture_dir):
        """Test that all fixtures have matching records."""
        weak_path = fixture_dir / "weak_baseline.jsonl"
        refined_path = fixture_dir / "gold_with_llm_refined.jsonl"
        gold_path = fixture_dir / "gold_standard.jsonl"
        
        with open(weak_path, 'r') as f:
            weak_records = [json.loads(line) for line in f if line.strip()]
        
        with open(refined_path, 'r') as f:
            refined_records = [json.loads(line) for line in f if line.strip()]
        
        with open(gold_path, 'r') as f:
            gold_records = [json.loads(line) for line in f if line.strip()]
        
        # All should have same number of records
        assert len(weak_records) == len(refined_records) == len(gold_records)
        
        # Texts should match
        for w, r, g in zip(weak_records, refined_records, gold_records):
            assert w["text"] == r["text"] == g["text"]
