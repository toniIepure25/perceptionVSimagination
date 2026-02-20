"""
Direction 8: Reality Confusion Mapping
=======================================

For shared stimuli (same image seen and imagined), computes a per-trial
"confusion index" and maps the boundary where perception and imagery
become indistinguishable.

Tests the PRM prediction that confusion depends on signal strength
rather than semantic category, and estimates a quantitative "reality
threshold" by fitting a sigmoid to the signal-strength--confusion curve.

References:
    Dijkstra et al. (2025). "A neural basis for distinguishing
    imagination from reality." Neuron.
    Dijkstra & Fleming (2023). "Subjective signal strength distinguishes
    reality from imagination." Nature Communications.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


def compute_confusion_index(bundle: EmbeddingBundle) -> Dict:
    """
    For each shared stimulus, compute how similar the perception and
    imagery decoded embeddings are (the "confusion index").

    A high confusion index means the decoder produced nearly identical
    outputs for both conditions — the representations are indistinguishable.

    Returns per-trial metrics and aggregate statistics.
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        logger.warning("No shared stimuli found; generating fallback paired analysis")
        n_paired = min(bundle.perception.shape[0], bundle.imagery.shape[0])
        perc_idx = np.arange(n_paired)
        imag_idx = np.arange(n_paired)
        shared_ids = perc_idx.copy()
    else:
        shared_ids, perc_idx, imag_idx = pairs

    perc_emb = _l2(bundle.perception[perc_idx])
    imag_emb = _l2(bundle.imagery[imag_idx])

    # Per-trial confusion: cosine similarity between conditions for same stimulus
    cosine_sim = np.sum(perc_emb * imag_emb, axis=1)

    # Euclidean distance in normalized space
    euclidean_dist = np.linalg.norm(perc_emb - imag_emb, axis=2 if perc_emb.ndim > 2 else 1)

    # Confusion index: higher = more confusable (normalized to [0, 1])
    confusion_index = (cosine_sim + 1.0) / 2.0

    # Signal strength for the imagery trials (the "weaker" condition)
    imag_norms = np.linalg.norm(bundle.imagery[imag_idx], axis=1)
    perc_norms = np.linalg.norm(bundle.perception[perc_idx], axis=1)

    # Decoding confidence for paired trials
    perc_targets_paired = _l2(bundle.perception_targets[perc_idx])
    imag_targets_paired = _l2(bundle.imagery_targets[imag_idx])
    perc_confidence = np.sum(perc_emb * perc_targets_paired, axis=1)
    imag_confidence = np.sum(imag_emb * imag_targets_paired, axis=1)

    return {
        "shared_nsd_ids": shared_ids.tolist(),
        "n_shared": len(shared_ids),
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_dist,
        "confusion_index": confusion_index,
        "imagery_norms": imag_norms,
        "perception_norms": perc_norms,
        "imagery_confidence": imag_confidence,
        "perception_confidence": perc_confidence,
        "perc_idx": perc_idx,
        "imag_idx": imag_idx,
        "mean_confusion": float(np.mean(confusion_index)),
        "std_confusion": float(np.std(confusion_index)),
        "mean_cosine": float(np.mean(cosine_sim)),
        "confusion_norm_correlation": float(
            scipy_stats.spearmanr(imag_norms, confusion_index).correlation
            if len(confusion_index) > 2 else 0.0
        ),
    }


def map_category_confusability(
    bundle: EmbeddingBundle,
    confusion_data: Dict,
) -> Dict:
    """
    Compute per-category confusion scores, testing whether abstract/
    conceptual categories are more confusable than visual/structural ones.

    PRM prediction: confusion depends on signal strength, not category —
    so after controlling for signal strength, category effects should
    be minimal.
    """
    imag_idx = confusion_data["imag_idx"]
    confusion_index = confusion_data["confusion_index"]
    imagery_norms = confusion_data["imagery_norms"]

    # Group by stimulus type
    from collections import defaultdict
    by_category = defaultdict(lambda: {"confusions": [], "norms": []})

    for i, idx in enumerate(imag_idx):
        if idx < len(bundle.imagery_meta):
            stype = bundle.imagery_meta[idx].get("stimulus_type", "unknown")
        else:
            stype = "unknown"
        by_category[stype]["confusions"].append(confusion_index[i])
        by_category[stype]["norms"].append(imagery_norms[i])

    category_results = {}
    for cat, data in by_category.items():
        confusions = np.array(data["confusions"])
        norms = np.array(data["norms"])
        category_results[cat] = {
            "mean_confusion": float(np.mean(confusions)),
            "std_confusion": float(np.std(confusions)),
            "mean_norm": float(np.mean(norms)),
            "count": len(confusions),
        }

    # Test category independence (Kruskal-Wallis across categories)
    groups = [np.array(v["confusions"]) for v in by_category.values() if len(v["confusions"]) > 1]
    if len(groups) >= 2:
        kw_stat, kw_p = scipy_stats.kruskal(*groups)
    else:
        kw_stat, kw_p = 0.0, 1.0

    # Partial correlation: confusion ~ category, controlling for norm
    # If PRM is correct, this should be non-significant
    all_confusions = confusion_index
    all_norms = imagery_norms
    if len(all_confusions) > 3:
        norm_residual_corr = _partial_correlation_with_norm(
            all_confusions, all_norms, imag_idx, bundle.imagery_meta
        )
    else:
        norm_residual_corr = {"residual_category_effect": float("nan")}

    return {
        "per_category": category_results,
        "kruskal_wallis_stat": float(kw_stat),
        "kruskal_wallis_p": float(kw_p),
        "category_independent": bool(kw_p > 0.05),
        "norm_controlled_analysis": norm_residual_corr,
    }


def _partial_correlation_with_norm(
    confusions: np.ndarray,
    norms: np.ndarray,
    imag_idx: np.ndarray,
    imagery_meta: List[Dict],
) -> Dict:
    """After regressing out norm, test whether category still predicts confusion."""
    from sklearn.linear_model import LinearRegression

    # Regress out norm from confusion
    norm_model = LinearRegression().fit(norms.reshape(-1, 1), confusions)
    residuals = confusions - norm_model.predict(norms.reshape(-1, 1))

    # Encode categories numerically
    categories = []
    for i, idx in enumerate(imag_idx):
        if idx < len(imagery_meta):
            categories.append(imagery_meta[idx].get("stimulus_type", "unknown"))
        else:
            categories.append("unknown")

    unique_cats = list(set(categories))
    if len(unique_cats) < 2:
        return {"residual_category_effect": float("nan")}

    cat_to_num = {c: i for i, c in enumerate(unique_cats)}
    cat_numeric = np.array([cat_to_num[c] for c in categories])

    # Kruskal-Wallis on residuals
    groups = [residuals[cat_numeric == i] for i in range(len(unique_cats)) if np.sum(cat_numeric == i) > 1]
    if len(groups) >= 2:
        kw_stat, kw_p = scipy_stats.kruskal(*groups)
    else:
        kw_stat, kw_p = 0.0, 1.0

    return {
        "residual_kruskal_wallis_stat": float(kw_stat),
        "residual_kruskal_wallis_p": float(kw_p),
        "residual_category_effect": bool(kw_p < 0.05),
    }


def _sigmoid(x, L, k, x0, b):
    """Sigmoid function for threshold fitting."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def estimate_reality_boundary(
    confusion_scores: np.ndarray,
    signal_strength: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """
    Fit a sigmoid to the signal-strength vs confusion relationship to
    estimate the computational "reality threshold."

    At this threshold, confusion transitions from low (easily distinguished)
    to high (indistinguishable) — the computational analog of Dijkstra's
    fusiform reality threshold.
    """
    valid = np.isfinite(confusion_scores) & np.isfinite(signal_strength)
    conf = confusion_scores[valid]
    strength = signal_strength[valid]

    if len(conf) < 10:
        return {"threshold_estimated": False, "reason": "insufficient_data"}

    # Bin signal strength and compute mean confusion per bin
    sorted_idx = np.argsort(strength)
    bin_size = max(1, len(sorted_idx) // n_bins)
    bin_centers = []
    bin_means = []
    for i in range(0, len(sorted_idx), bin_size):
        chunk = sorted_idx[i:i + bin_size]
        if len(chunk) > 0:
            bin_centers.append(float(np.mean(strength[chunk])))
            bin_means.append(float(np.mean(conf[chunk])))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)

    # Fit sigmoid
    try:
        popt, pcov = curve_fit(
            _sigmoid, bin_centers, bin_means,
            p0=[0.5, 5.0, np.median(strength), 0.3],
            maxfev=5000,
            bounds=([0, 0.01, strength.min(), -1], [1.5, 100, strength.max(), 1.5]),
        )
        L, k, x0, b = popt
        perr = np.sqrt(np.diag(pcov))

        # x0 is the inflection point = reality threshold
        threshold = float(x0)
        fitted_curve = _sigmoid(bin_centers, *popt)
        r_squared = 1 - np.sum((bin_means - fitted_curve) ** 2) / np.sum((bin_means - np.mean(bin_means)) ** 2)

        return {
            "threshold_estimated": True,
            "reality_threshold": threshold,
            "sigmoid_params": {"L": float(L), "k": float(k), "x0": float(x0), "b": float(b)},
            "sigmoid_stderr": {"L": float(perr[0]), "k": float(perr[1]),
                               "x0": float(perr[2]), "b": float(perr[3])},
            "r_squared": float(r_squared),
            "bin_centers": bin_centers.tolist(),
            "bin_means": bin_means.tolist(),
            "steepness": float(k),
        }
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Sigmoid fit failed: {e}")
        # Fall back to linear correlation
        r, p = scipy_stats.spearmanr(strength, conf)
        return {
            "threshold_estimated": False,
            "reason": "sigmoid_fit_failed",
            "spearman_r": float(r),
            "spearman_p": float(p),
            "bin_centers": bin_centers.tolist(),
            "bin_means": bin_means.tolist(),
        }


def analyze_reality_confusion(bundle: EmbeddingBundle) -> Dict:
    """
    Full reality confusion mapping analysis.

    Computes per-trial confusion indices for shared stimuli, maps
    category-level confusability, and estimates the reality boundary.
    """
    logger.info("Running Reality Confusion Mapping analysis...")

    confusion_data = compute_confusion_index(bundle)
    logger.info(f"  Shared stimuli: {confusion_data['n_shared']}")
    logger.info(f"  Mean confusion index: {confusion_data['mean_confusion']:.4f}")
    logger.info(f"  Confusion-norm correlation: {confusion_data['confusion_norm_correlation']:.4f}")

    category_data = map_category_confusability(bundle, confusion_data)
    logger.info(f"  Category-independent (PRM prediction): {category_data['category_independent']}")

    boundary = estimate_reality_boundary(
        confusion_data["confusion_index"],
        confusion_data["imagery_norms"],
    )
    if boundary["threshold_estimated"]:
        logger.info(f"  Reality threshold: {boundary['reality_threshold']:.4f}")
        logger.info(f"  Sigmoid R²: {boundary['r_squared']:.4f}")
    else:
        logger.info(f"  Boundary estimation: {boundary.get('reason', 'unknown')}")

    # Serializable subset
    results = {
        "confusion_summary": {
            "n_shared": confusion_data["n_shared"],
            "mean_confusion": confusion_data["mean_confusion"],
            "std_confusion": confusion_data["std_confusion"],
            "mean_cosine": confusion_data["mean_cosine"],
            "confusion_norm_correlation": confusion_data["confusion_norm_correlation"],
        },
        "category_confusability": category_data,
        "reality_boundary": boundary,
    }

    return results
