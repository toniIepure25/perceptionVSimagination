"""
Direction 1: The Dimensionality Gap
====================================

Measures the effective dimensionality of perception vs imagery representations
to test the hypothesis that mental imagery collapses representations into a
lower-dimensional linear subspace.

Metrics:
- PCA participation ratio (PR): PR = (sum λ_i)^2 / sum λ_i^2
  PR ≈ 1 means all variance in one dimension; PR ≈ D means uniform.
- Explained variance at k components
- Intrinsic dimensionality via nearest-neighbor (MLE estimator)
- Linear probe accuracy at varying dimensionality

References:
    Gao et al. (2017). "A theory of multineuronal dimensionality, dynamics
    and measurement" (participation ratio)
    Levina & Bickel (2005). "Maximum Likelihood Estimation of Intrinsic
    Dimension" (MLE intrinsic dim)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .core import EmbeddingBundle

logger = logging.getLogger(__name__)


def participation_ratio(embeddings: np.ndarray) -> float:
    """
    Compute the participation ratio (PR) of the embedding covariance spectrum.

    PR = (sum λ_i)^2 / sum λ_i^2

    Higher PR → more distributed variance → higher effective dimensionality.
    """
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)

    sum_sq = np.sum(eigenvalues) ** 2
    sq_sum = np.sum(eigenvalues ** 2)
    if sq_sum < 1e-12:
        return 0.0
    return float(sum_sq / sq_sum)


def explained_variance_curve(
    embeddings: np.ndarray, max_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative explained variance ratio as a function of PCA components.

    Returns (n_components_array, cumulative_variance_array).
    """
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total < 1e-12:
        total = 1.0

    if max_components is None:
        max_components = len(eigenvalues)
    else:
        max_components = min(max_components, len(eigenvalues))

    cumvar = np.cumsum(eigenvalues[:max_components]) / total
    components = np.arange(1, max_components + 1)
    return components, cumvar


def intrinsic_dimensionality_mle(
    embeddings: np.ndarray, k: int = 10
) -> float:
    """
    MLE estimator of intrinsic dimensionality (Levina & Bickel 2005).

    Uses k nearest neighbors to estimate the local dimensionality, then
    averages across all points.
    """
    n = embeddings.shape[0]
    if n < k + 1:
        logger.warning(f"Too few samples ({n}) for k={k}; returning NaN")
        return float("nan")

    dists = squareform(pdist(embeddings, metric="euclidean"))
    np.fill_diagonal(dists, np.inf)

    dims = []
    for i in range(n):
        sorted_d = np.sort(dists[i])[:k]
        sorted_d = sorted_d[sorted_d > 1e-10]
        if len(sorted_d) < 2:
            continue
        T_k = sorted_d[-1]
        if T_k < 1e-10:
            continue
        log_ratios = np.log(T_k / sorted_d[:-1])
        if log_ratios.sum() < 1e-10:
            continue
        dims.append((len(log_ratios)) / log_ratios.sum())

    if not dims:
        return float("nan")
    return float(np.mean(dims))


def dimensionality_at_threshold(
    embeddings: np.ndarray, threshold: float = 0.95
) -> int:
    """Number of PCA components needed to explain `threshold` fraction of variance."""
    _, cumvar = explained_variance_curve(embeddings)
    idx = np.searchsorted(cumvar, threshold)
    return int(min(idx + 1, len(cumvar)))


def analyze_dimensionality_gap(
    bundle: EmbeddingBundle,
    k_nn: int = 10,
    variance_threshold: float = 0.95,
    max_pca_components: int = 200,
) -> Dict:
    """
    Full dimensionality gap analysis comparing perception and imagery.

    Returns a dictionary with all metrics for both conditions plus the gap.
    """
    logger.info("Running dimensionality gap analysis...")

    pr_perc = participation_ratio(bundle.perception)
    pr_imag = participation_ratio(bundle.imagery)

    id_perc = intrinsic_dimensionality_mle(bundle.perception, k=k_nn)
    id_imag = intrinsic_dimensionality_mle(bundle.imagery, k=k_nn)

    dim95_perc = dimensionality_at_threshold(bundle.perception, variance_threshold)
    dim95_imag = dimensionality_at_threshold(bundle.imagery, variance_threshold)

    comp_p, cumvar_p = explained_variance_curve(bundle.perception, max_pca_components)
    comp_i, cumvar_i = explained_variance_curve(bundle.imagery, max_pca_components)

    results = {
        "perception": {
            "participation_ratio": pr_perc,
            "intrinsic_dim_mle": id_perc,
            f"dims_at_{int(variance_threshold*100)}pct": dim95_perc,
            "n_samples": bundle.perception.shape[0],
        },
        "imagery": {
            "participation_ratio": pr_imag,
            "intrinsic_dim_mle": id_imag,
            f"dims_at_{int(variance_threshold*100)}pct": dim95_imag,
            "n_samples": bundle.imagery.shape[0],
        },
        "gap": {
            "pr_ratio": pr_imag / max(pr_perc, 1e-8),
            "id_ratio": id_imag / max(id_perc, 1e-8),
            "dim95_ratio": dim95_imag / max(dim95_perc, 1),
        },
        "explained_variance_curves": {
            "perception": {"components": comp_p.tolist(), "cumvar": cumvar_p.tolist()},
            "imagery": {"components": comp_i.tolist(), "cumvar": cumvar_i.tolist()},
        },
    }

    logger.info(f"  Participation ratio: perception={pr_perc:.1f}, imagery={pr_imag:.1f} "
                f"(ratio={results['gap']['pr_ratio']:.3f})")
    logger.info(f"  Intrinsic dim (MLE): perception={id_perc:.1f}, imagery={id_imag:.1f}")
    logger.info(f"  Dims at {int(variance_threshold*100)}%: "
                f"perception={dim95_perc}, imagery={dim95_imag}")

    return results
