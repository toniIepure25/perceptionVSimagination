"""Canonical evaluation utilities for the shared/private decoder workflows."""

from .decoder import (
    collect_predictions,
    compute_decoder_metrics,
    compute_pair_metrics,
    compute_roi_summary,
    write_evaluation_bundle,
)

__all__ = [
    "collect_predictions",
    "compute_decoder_metrics",
    "compute_pair_metrics",
    "compute_roi_summary",
    "write_evaluation_bundle",
]
