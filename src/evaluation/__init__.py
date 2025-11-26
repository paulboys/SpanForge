"""Evaluation package for span quality metrics."""

from .metrics import (
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

__all__ = [
    "compute_overlap",
    "compute_iou",
    "compute_boundary_precision",
    "compute_iou_delta",
    "compute_correction_rate",
    "calibration_curve",
    "stratify_by_confidence",
    "stratify_by_label",
    "stratify_by_span_length",
    "compute_precision_recall_f1",
]
