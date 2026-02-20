"""
Evaluation Package
==================

Metrics for fMRI-to-image reconstruction quality: embedding retrieval,
perceptual image similarity, brain alignment, and uncertainty calibration.
"""

from .retrieval import cosine_sim, retrieval_at_k, compute_ranking_metrics
from .retrieval import clip_score as clip_score_embeddings
from .image_metrics import (
    clip_score,
    batch_clip_score,
    ssim_score,
    lpips_score,
    compute_all_metrics,
    pixel_mse,
)
from .uncertainty import (
    enable_dropout,
    predict_with_mc_dropout,
    compute_uncertainty_error_correlation,
    evaluate_uncertainty_calibration,
    compute_confidence_intervals,
)

__all__ = [
    # Retrieval metrics
    "cosine_sim",
    "retrieval_at_k",
    "compute_ranking_metrics",
    "clip_score_embeddings",
    # Image quality metrics
    "clip_score",
    "batch_clip_score",
    "ssim_score",
    "lpips_score",
    "compute_all_metrics",
    "pixel_mse",
    # Uncertainty estimation
    "enable_dropout",
    "predict_with_mc_dropout",
    "compute_uncertainty_error_correlation",
    "evaluate_uncertainty_calibration",
    "compute_confidence_intervals",
]
