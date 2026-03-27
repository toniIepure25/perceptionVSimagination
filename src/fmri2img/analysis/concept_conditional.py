"""
Concept-Conditional Transfer Profiles
======================================

Analyzes the perception-imagery transfer gap broken down by semantic
category. Uses CLIP zero-shot classification to assign categories, then
computes per-category transfer gaps with bootstrap confidence intervals.

Key questions:
    - Which semantic categories transfer best from perception to imagery?
    - Do faces, scenes, objects, or abstract stimuli show different gaps?
    - Clinical implication: categories with large gaps may relate to
      aphantasia subtypes or PTSD imagery phenomena.

Uses both stimulus-set labels (A=simple, B=complex, C=conceptual) and
CLIP-inferred semantic categories for fine-grained analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Default semantic categories for CLIP zero-shot classification
DEFAULT_CATEGORIES = [
    "a photo of a person or face",
    "a photo of an animal",
    "a photo of a vehicle or transportation",
    "a photo of food or drink",
    "a photo of an indoor scene",
    "a photo of an outdoor natural scene",
    "a photo of a building or architecture",
    "a photo of an everyday object",
    "a simple geometric pattern",
    "abstract art or texture",
]

CATEGORY_SHORT_NAMES = [
    "faces", "animals", "vehicles", "food",
    "indoor", "outdoor", "architecture", "objects",
    "geometric", "abstract",
]


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / norms


def _cosine_similarities(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-sample cosine similarity. (N,) array."""
    return np.sum(_l2(preds) * _l2(targets), axis=-1)


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower, upper).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = np.array([
        values[rng.randint(0, n, n)].mean() for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def classify_stimuli_by_category(
    target_embeddings: np.ndarray,
    category_texts: Optional[List[str]] = None,
    category_names: Optional[List[str]] = None,
    clip_model: Optional[object] = None,
    clip_preprocess: Optional[object] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Assign each stimulus a category via CLIP zero-shot classification.

    Parameters
    ----------
    target_embeddings : (N, D) CLIP embeddings of target stimuli
    category_texts : text prompts for each category
    category_names : short names for reporting
    clip_model : CLIP model (if None, uses text embedding approximation)

    Returns
    -------
    (N,) int array of category indices
    """
    if category_texts is None:
        category_texts = DEFAULT_CATEGORIES
    if category_names is None:
        category_names = CATEGORY_SHORT_NAMES

    # If we have a CLIP model, compute text embeddings properly
    if clip_model is not None:
        import torch
        import clip as clip_lib
        tokens = clip_lib.tokenize(category_texts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
    else:
        # Fallback: use pre-computed text embeddings if available
        # This won't work perfectly but allows the analysis to run
        logger.warning(
            "No CLIP model provided — using target embedding clustering "
            "as category proxy. Results will be approximate."
        )
        return _cluster_categories(target_embeddings, n_categories=len(category_texts))

    # Zero-shot classification: argmax cosine(target, text_category)
    target_norm = _l2(target_embeddings)
    similarities = target_norm @ text_features.T  # (N, n_categories)
    categories = similarities.argmax(axis=1)  # (N,)

    return categories


def _cluster_categories(
    embeddings: np.ndarray,
    n_categories: int = 10,
) -> np.ndarray:
    """Fallback: cluster embeddings into categories via k-means."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(n_categories, len(embeddings)), random_state=42, n_init=10)
    return km.fit_predict(_l2(embeddings))


def analyze_concept_conditional_transfer(
    perception_preds: np.ndarray,
    imagery_preds: np.ndarray,
    perception_targets: np.ndarray,
    imagery_targets: np.ndarray,
    perception_categories: Optional[np.ndarray] = None,
    imagery_categories: Optional[np.ndarray] = None,
    category_names: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """Per-category transfer gap analysis.

    Parameters
    ----------
    perception_preds : (N_p, D) predicted embeddings for perception
    imagery_preds : (N_i, D) predicted embeddings for imagery
    perception_targets : (N_p, D) CLIP ground truth for perception
    imagery_targets : (N_i, D) CLIP ground truth for imagery
    perception_categories : (N_p,) category indices (if None, auto-classify)
    imagery_categories : (N_i,) category indices
    category_names : list of category labels

    Returns
    -------
    dict with per-category metrics and summary
    """
    if category_names is None:
        category_names = CATEGORY_SHORT_NAMES

    # Auto-classify if categories not provided
    if perception_categories is None:
        perception_categories = classify_stimuli_by_category(perception_targets)
    if imagery_categories is None:
        imagery_categories = classify_stimuli_by_category(imagery_targets)

    perc_cosines = _cosine_similarities(perception_preds, perception_targets)
    imag_cosines = _cosine_similarities(imagery_preds, imagery_targets)

    results = {"categories": {}, "summary": {}}

    all_unique_cats = sorted(set(
        list(np.unique(perception_categories)) +
        list(np.unique(imagery_categories))
    ))

    for cat_idx in all_unique_cats:
        name = category_names[cat_idx] if cat_idx < len(category_names) else f"cat_{cat_idx}"

        perc_mask = perception_categories == cat_idx
        imag_mask = imagery_categories == cat_idx

        n_perc = perc_mask.sum()
        n_imag = imag_mask.sum()

        if n_perc < 2 or n_imag < 2:
            continue

        perc_cos = perc_cosines[perc_mask]
        imag_cos = imag_cosines[imag_mask]

        perc_mean, perc_lo, perc_hi = _bootstrap_ci(perc_cos, n_bootstrap)
        imag_mean, imag_lo, imag_hi = _bootstrap_ci(imag_cos, n_bootstrap)
        gap = perc_mean - imag_mean

        # Bootstrap the gap directly
        gap_mean, gap_lo, gap_hi = _bootstrap_gap(perc_cos, imag_cos, n_bootstrap)

        # Two-sample t-test
        t_stat, p_val = scipy_stats.ttest_ind(perc_cos, imag_cos, equal_var=False)

        results["categories"][name] = {
            "n_perception": int(n_perc),
            "n_imagery": int(n_imag),
            "perception_cosine": perc_mean,
            "perception_ci": [perc_lo, perc_hi],
            "imagery_cosine": imag_mean,
            "imagery_ci": [imag_lo, imag_hi],
            "gap": gap,
            "gap_ci": [gap_lo, gap_hi],
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_005": p_val < 0.05,
        }

    # Summary: which categories have largest/smallest gaps
    if results["categories"]:
        cats = results["categories"]
        sorted_by_gap = sorted(cats.items(), key=lambda x: x[1]["gap"], reverse=True)
        results["summary"]["largest_gap_category"] = sorted_by_gap[0][0]
        results["summary"]["largest_gap_value"] = sorted_by_gap[0][1]["gap"]
        results["summary"]["smallest_gap_category"] = sorted_by_gap[-1][0]
        results["summary"]["smallest_gap_value"] = sorted_by_gap[-1][1]["gap"]
        results["summary"]["n_significant"] = sum(
            1 for c in cats.values() if c["significant_005"]
        )
        results["summary"]["n_categories_tested"] = len(cats)

    return results


def _bootstrap_gap(
    perc_cos: np.ndarray,
    imag_cos: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap the difference in means (gap)."""
    rng = np.random.RandomState(seed)
    n_p, n_i = len(perc_cos), len(imag_cos)

    gaps = np.array([
        perc_cos[rng.randint(0, n_p, n_p)].mean() -
        imag_cos[rng.randint(0, n_i, n_i)].mean()
        for _ in range(n_bootstrap)
    ])

    return float(gaps.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def analyze_by_stimulus_set(
    perception_preds: np.ndarray,
    imagery_preds: np.ndarray,
    perception_targets: np.ndarray,
    imagery_targets: np.ndarray,
    perception_stim_types: List[str],
    imagery_stim_types: List[str],
    n_bootstrap: int = 1000,
) -> Dict:
    """Analyze transfer gap by stimulus set (A=simple, B=complex, C=conceptual).

    This uses the actual experimental stimulus-set labels rather than
    CLIP-inferred categories.
    """
    perc_cosines = _cosine_similarities(perception_preds, perception_targets)
    imag_cosines = _cosine_similarities(imagery_preds, imagery_targets)

    perc_types = np.array(perception_stim_types)
    imag_types = np.array(imagery_stim_types)

    results = {"stimulus_sets": {}}

    for stim_type in sorted(set(list(perc_types) + list(imag_types))):
        perc_mask = perc_types == stim_type
        imag_mask = imag_types == stim_type

        n_perc = perc_mask.sum()
        n_imag = imag_mask.sum()

        if n_perc < 2 or n_imag < 2:
            continue

        perc_cos = perc_cosines[perc_mask]
        imag_cos = imag_cosines[imag_mask]

        perc_mean, perc_lo, perc_hi = _bootstrap_ci(perc_cos, n_bootstrap)
        imag_mean, imag_lo, imag_hi = _bootstrap_ci(imag_cos, n_bootstrap)
        gap_mean, gap_lo, gap_hi = _bootstrap_gap(perc_cos, imag_cos, n_bootstrap)

        t_stat, p_val = scipy_stats.ttest_ind(perc_cos, imag_cos, equal_var=False)

        results["stimulus_sets"][stim_type] = {
            "n_perception": int(n_perc),
            "n_imagery": int(n_imag),
            "perception_cosine": perc_mean,
            "perception_ci": [perc_lo, perc_hi],
            "imagery_cosine": imag_mean,
            "imagery_ci": [imag_lo, imag_hi],
            "gap": gap_mean,
            "gap_ci": [gap_lo, gap_hi],
            "transfer_ratio": imag_mean / perc_mean if perc_mean > 0.01 else 1.0,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
        }

    return results


def plot_concept_profiles(
    results: Dict,
    save_path: str,
    title: str = "Concept-Conditional Transfer Gap",
) -> None:
    """Publication-quality bar chart of per-category transfer gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = results.get("categories", {})
    if not cats:
        logger.warning("No categories to plot")
        return

    # Sort by gap magnitude
    sorted_cats = sorted(cats.items(), key=lambda x: x[1]["gap"], reverse=True)
    names = [c[0] for c in sorted_cats]
    gaps = [c[1]["gap"] for c in sorted_cats]
    gap_cis = [c[1]["gap_ci"] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(names))
    colors = ["#F44336" if g > 0 else "#4CAF50" for g in gaps]
    yerr_lo = [g - ci[0] for g, ci in zip(gaps, gap_cis)]
    yerr_hi = [ci[1] - g for g, ci in zip(gaps, gap_cis)]

    ax.bar(x, gaps, 0.6, color=colors, alpha=0.8,
           yerr=[yerr_lo, yerr_hi], capsize=3, ecolor="gray")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Transfer Gap (Perception − Imagery)")
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved concept profile plot to {save_path}")
