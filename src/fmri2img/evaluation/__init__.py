"""Canonical evaluation utilities for the shared/private decoder workflows."""

from .decoder import (
    collect_predictions,
    compute_decoder_metrics,
    compute_pair_metrics,
    compute_roi_summary,
    normalize_condition_semantics_payload,
    write_evaluation_bundle,
)

__all__ = [
    "collect_predictions",
    "compute_decoder_metrics",
    "compute_pair_metrics",
    "compute_roi_summary",
    "normalize_condition_semantics_payload",
    "write_evaluation_bundle",
]
