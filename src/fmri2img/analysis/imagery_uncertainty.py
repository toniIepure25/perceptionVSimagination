"""
Direction 2: Uncertainty as Vividness
======================================

Uses MC Dropout variance as an objective, trial-level proxy for imagery
vividness. Compares uncertainty distributions between perception and imagery,
and correlates uncertainty with decoding accuracy and stimulus type.

Extends the existing MC Dropout infrastructure (src/fmri2img/eval/uncertainty.py)
for perception-vs-imagery analysis.

References:
    Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning"
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import stats as scipy_stats

from .core import EmbeddingBundle

logger = logging.getLogger(__name__)


def mc_dropout_on_embeddings(
    model: torch.nn.Module,
    voxels: np.ndarray,
    n_mc_samples: int = 30,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Run MC Dropout inference on a batch of voxels and return per-trial statistics.

    Returns:
        mean: (N, D) mean prediction across MC samples
        std: (N, D) std across MC samples
        uncertainty: (N,) scalar uncertainty per trial (mean of std across dims)
        all_samples: (n_mc, N, D) all MC forward pass outputs
    """
    from fmri2img.eval.uncertainty import enable_dropout

    model.eval()
    enable_dropout(model)

    n = voxels.shape[0]
    all_samples = []

    with torch.no_grad():
        for mc_idx in range(n_mc_samples):
            preds = []
            for start in range(0, n, batch_size):
                batch = torch.from_numpy(
                    voxels[start : start + batch_size]
                ).float().to(device)
                out = model(batch).cpu().numpy()
                preds.append(out)
            all_samples.append(np.concatenate(preds, axis=0))

    samples = np.stack(all_samples, axis=0)  # (n_mc, N, D)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    uncertainty = std.mean(axis=-1)  # (N,)

    model.eval()

    return {
        "mean": mean,
        "std": std,
        "uncertainty": uncertainty,
        "all_samples": samples,
    }


def compare_uncertainty_distributions(
    perc_uncertainty: np.ndarray,
    imag_uncertainty: np.ndarray,
) -> Dict:
    """
    Statistical comparison of uncertainty distributions between conditions.
    """
    ks_stat, ks_p = scipy_stats.ks_2samp(perc_uncertainty, imag_uncertainty)
    mw_stat, mw_p = scipy_stats.mannwhitneyu(
        perc_uncertainty, imag_uncertainty, alternative="two-sided"
    )

    # Effect size: rank-biserial correlation from Mann-Whitney U
    n1, n2 = len(perc_uncertainty), len(imag_uncertainty)
    rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)

    return {
        "perception_mean": float(np.mean(perc_uncertainty)),
        "perception_std": float(np.std(perc_uncertainty)),
        "perception_median": float(np.median(perc_uncertainty)),
        "imagery_mean": float(np.mean(imag_uncertainty)),
        "imagery_std": float(np.std(imag_uncertainty)),
        "imagery_median": float(np.median(imag_uncertainty)),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "mannwhitney_U": float(mw_stat),
        "mannwhitney_p": float(mw_p),
        "rank_biserial_r": float(rank_biserial),
    }


def uncertainty_accuracy_correlation(
    uncertainty: np.ndarray,
    cosine_scores: np.ndarray,
) -> Dict:
    """
    Correlate per-trial uncertainty with decoding accuracy (cosine similarity).

    Negative correlation expected: higher uncertainty → lower accuracy.
    """
    valid = np.isfinite(uncertainty) & np.isfinite(cosine_scores)
    u, c = uncertainty[valid], cosine_scores[valid]

    if len(u) < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "n": 0}

    pearson_r, pearson_p = scipy_stats.pearsonr(u, c)
    spearman_r, spearman_p = scipy_stats.spearmanr(u, c)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(len(u)),
    }


def uncertainty_by_stimulus_type(
    uncertainty: np.ndarray,
    meta: List[Dict],
) -> Dict[str, Dict]:
    """Break down uncertainty statistics by stimulus type."""
    from collections import defaultdict

    by_type = defaultdict(list)
    for i, m in enumerate(meta):
        stype = m.get("stimulus_type", "unknown")
        by_type[stype].append(uncertainty[i])

    results = {}
    for stype, vals in by_type.items():
        vals = np.array(vals)
        results[stype] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "count": len(vals),
        }
    return results


def analyze_imagery_uncertainty(
    bundle: EmbeddingBundle,
    model: Optional[torch.nn.Module] = None,
    perc_voxels: Optional[np.ndarray] = None,
    imag_voxels: Optional[np.ndarray] = None,
    n_mc_samples: int = 30,
    device: str = "cpu",
) -> Dict:
    """
    Full uncertainty-as-vividness analysis.

    If model and voxels are provided, runs MC Dropout inference.
    Otherwise, uses synthetic uncertainty derived from embedding variance.
    """
    logger.info("Running uncertainty-as-vividness analysis...")

    if model is not None and perc_voxels is not None and imag_voxels is not None:
        logger.info(f"  Running MC Dropout with {n_mc_samples} samples...")
        perc_mc = mc_dropout_on_embeddings(model, perc_voxels, n_mc_samples, device)
        imag_mc = mc_dropout_on_embeddings(model, imag_voxels, n_mc_samples, device)
        perc_uncertainty = perc_mc["uncertainty"]
        imag_uncertainty = imag_mc["uncertainty"]
    else:
        logger.info("  No model provided; deriving uncertainty from embedding variance")
        perc_uncertainty = np.std(bundle.perception, axis=1)
        imag_uncertainty = np.std(bundle.imagery, axis=1)

    # Compare distributions
    dist_comparison = compare_uncertainty_distributions(perc_uncertainty, imag_uncertainty)

    # Correlate with accuracy
    perc_corr = uncertainty_accuracy_correlation(perc_uncertainty, bundle.perception_cosines)
    imag_corr = uncertainty_accuracy_correlation(imag_uncertainty, bundle.imagery_cosines)

    # Break down by stimulus type
    imag_by_type = uncertainty_by_stimulus_type(imag_uncertainty, bundle.imagery_meta)

    results = {
        "distribution_comparison": dist_comparison,
        "perception_accuracy_correlation": perc_corr,
        "imagery_accuracy_correlation": imag_corr,
        "imagery_by_stimulus_type": imag_by_type,
        "n_mc_samples": n_mc_samples,
        "has_mc_dropout": model is not None,
    }

    logger.info(f"  Perception uncertainty: {dist_comparison['perception_mean']:.4f} "
                f"± {dist_comparison['perception_std']:.4f}")
    logger.info(f"  Imagery uncertainty:    {dist_comparison['imagery_mean']:.4f} "
                f"± {dist_comparison['imagery_std']:.4f}")
    logger.info(f"  KS test p-value: {dist_comparison['ks_p_value']:.4f}")
    logger.info(f"  Imagery uncertainty-accuracy correlation: "
                f"r={imag_corr['spearman_r']:.3f} (p={imag_corr['spearman_p']:.4f})")

    return results
