"""
Direction 4: Topological Signatures of Imagination
====================================================

Applies topological representational similarity analysis (tRSA) to compare
the topology and geometry of perception vs imagery representations.

Tests whether imagery preserves the neighborhood structure (topology) of
perception while distorting metric distances (geometry).

Metrics:
- Representational Dissimilarity Matrices (RDMs)
- RDM correlation (Spearman) between conditions
- RDM contraction ratio (imagery distances / perception distances)
- Persistent homology (Betti numbers) when ripser is available
- Neighborhood preservation (k-NN overlap between conditions)

References:
    Kriegeskorte et al. (2008). "Representational similarity analysis"
    Carlsson (2009). "Topology and data" (persistent homology)
    Rosenblatt et al. (2024). "Topology and geometry of neural
    representations" (PNAS)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats as scipy_stats

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


def compute_rdm(embeddings: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """
    Compute a representational dissimilarity matrix.
    Returns a square symmetric matrix (N, N) of pairwise distances.
    """
    dists = pdist(embeddings, metric=metric)
    return squareform(dists)


def rdm_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> Dict:
    """
    Compare two RDMs using upper-triangle correlation (standard RSA).
    """
    triu_idx = np.triu_indices_from(rdm1, k=1)
    v1 = rdm1[triu_idx]
    v2 = rdm2[triu_idx]

    spearman_r, spearman_p = scipy_stats.spearmanr(v1, v2)
    pearson_r, pearson_p = scipy_stats.pearsonr(v1, v2)

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_pairs": int(len(v1)),
    }


def contraction_ratio(perc_rdm: np.ndarray, imag_rdm: np.ndarray) -> Dict:
    """
    Measure whether imagery representations are a contracted version of perception.

    Computes the ratio of imagery to perception distances for matched pairs.
    Ratio < 1 means imagery contracts the representational space.
    """
    triu_idx = np.triu_indices_from(perc_rdm, k=1)
    perc_dists = perc_rdm[triu_idx]
    imag_dists = imag_rdm[triu_idx]

    # Avoid division by zero
    valid = perc_dists > 1e-8
    ratios = np.full_like(perc_dists, np.nan)
    ratios[valid] = imag_dists[valid] / perc_dists[valid]

    finite_ratios = ratios[np.isfinite(ratios)]

    return {
        "mean_ratio": float(np.mean(finite_ratios)) if len(finite_ratios) > 0 else float("nan"),
        "median_ratio": float(np.median(finite_ratios)) if len(finite_ratios) > 0 else float("nan"),
        "std_ratio": float(np.std(finite_ratios)) if len(finite_ratios) > 0 else float("nan"),
        "fraction_contracted": float(np.mean(finite_ratios < 1.0)) if len(finite_ratios) > 0 else float("nan"),
    }


def neighborhood_preservation(
    perc_embeddings: np.ndarray,
    imag_embeddings: np.ndarray,
    k_values: Tuple[int, ...] = (5, 10, 20, 50),
) -> Dict[str, float]:
    """
    Measure how well k-nearest neighbor structure is preserved between conditions.

    For each sample, computes overlap between its k-NN set in perception space
    and its k-NN set in imagery space. High overlap means topology is preserved.

    Requires equal numbers of matched samples (same stimuli in both conditions).
    """
    n = min(perc_embeddings.shape[0], imag_embeddings.shape[0])
    perc = perc_embeddings[:n]
    imag = imag_embeddings[:n]

    perc_dists = squareform(pdist(perc, metric="cosine"))
    imag_dists = squareform(pdist(imag, metric="cosine"))

    results = {}
    for k in k_values:
        if k >= n:
            continue
        overlaps = []
        for i in range(n):
            perc_nn = set(np.argsort(perc_dists[i])[1 : k + 1])
            imag_nn = set(np.argsort(imag_dists[i])[1 : k + 1])
            overlap = len(perc_nn & imag_nn) / k
            overlaps.append(overlap)
        results[f"knn_overlap@{k}"] = float(np.mean(overlaps))

    return results


def persistent_homology_comparison(
    perc_embeddings: np.ndarray,
    imag_embeddings: np.ndarray,
    max_dim: int = 1,
    n_samples: Optional[int] = 200,
) -> Dict:
    """
    Compare persistent homology (Betti numbers / persistence diagrams).

    Requires `ripser` or `giotto-tda`. Falls back to a stub if unavailable.
    """
    if n_samples and perc_embeddings.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx_p = rng.choice(perc_embeddings.shape[0], n_samples, replace=False)
        idx_i = rng.choice(imag_embeddings.shape[0], n_samples, replace=False)
        perc_sub = perc_embeddings[idx_p]
        imag_sub = imag_embeddings[idx_i]
    else:
        perc_sub = perc_embeddings
        imag_sub = imag_embeddings

    try:
        from ripser import ripser as ripser_fn

        perc_ph = ripser_fn(perc_sub, maxdim=max_dim)["dgms"]
        imag_ph = ripser_fn(imag_sub, maxdim=max_dim)["dgms"]

        betti_perc = [len(dgm[dgm[:, 1] > dgm[:, 0] + 1e-6]) for dgm in perc_ph]
        betti_imag = [len(dgm[dgm[:, 1] > dgm[:, 0] + 1e-6]) for dgm in imag_ph]

        # Total persistence (sum of lifetimes) per dimension
        persistence_perc = [float(np.sum(dgm[:, 1] - dgm[:, 0])) for dgm in perc_ph]
        persistence_imag = [float(np.sum(dgm[:, 1] - dgm[:, 0])) for dgm in imag_ph]

        return {
            "available": True,
            "betti_perception": betti_perc,
            "betti_imagery": betti_imag,
            "total_persistence_perception": persistence_perc,
            "total_persistence_imagery": persistence_imag,
            "max_dim": max_dim,
            "n_samples": perc_sub.shape[0],
        }

    except ImportError:
        logger.warning("ripser not installed; skipping persistent homology. "
                       "Install with: pip install ripser")
        return {"available": False, "reason": "ripser not installed"}


def analyze_topological_signatures(
    bundle: EmbeddingBundle,
    k_values: Tuple[int, ...] = (5, 10, 20, 50),
    max_ph_samples: int = 200,
) -> Dict:
    """
    Full topological RSA analysis comparing perception and imagery.
    """
    logger.info("Running topological RSA analysis...")

    # Use matched samples (minimum of both sets)
    n = min(bundle.perception.shape[0], bundle.imagery.shape[0])
    perc = bundle.perception[:n]
    imag = bundle.imagery[:n]

    # RDMs on predictions
    logger.info("  Computing RDMs...")
    perc_rdm = compute_rdm(perc, metric="correlation")
    imag_rdm = compute_rdm(imag, metric="correlation")

    # RDMs on ground-truth targets (for reference)
    perc_gt_rdm = compute_rdm(bundle.perception_targets[:n], metric="correlation")
    imag_gt_rdm = compute_rdm(bundle.imagery_targets[:n], metric="correlation")

    # RSA comparisons
    logger.info("  Computing RSA correlations...")
    pred_rdm_corr = rdm_correlation(perc_rdm, imag_rdm)
    gt_rdm_corr = rdm_correlation(perc_gt_rdm, imag_gt_rdm)
    perc_pred_vs_gt = rdm_correlation(perc_rdm, perc_gt_rdm)
    imag_pred_vs_gt = rdm_correlation(imag_rdm, imag_gt_rdm)

    # Contraction
    logger.info("  Computing contraction ratio...")
    contract = contraction_ratio(perc_rdm, imag_rdm)

    # Neighborhood preservation
    logger.info("  Computing neighborhood preservation...")
    knn_overlap = neighborhood_preservation(perc, imag, k_values)

    # Persistent homology
    logger.info("  Computing persistent homology...")
    ph_results = persistent_homology_comparison(perc, imag, n_samples=max_ph_samples)

    results = {
        "rdm_correlation_pred_perc_vs_imag": pred_rdm_corr,
        "rdm_correlation_gt_perc_vs_imag": gt_rdm_corr,
        "rdm_correlation_perc_pred_vs_gt": perc_pred_vs_gt,
        "rdm_correlation_imag_pred_vs_gt": imag_pred_vs_gt,
        "contraction": contract,
        "neighborhood_preservation": knn_overlap,
        "persistent_homology": ph_results,
        "n_matched_samples": n,
    }

    logger.info(f"  RDM correlation (pred P vs I): "
                f"r={pred_rdm_corr['spearman_r']:.3f}")
    logger.info(f"  Contraction ratio: {contract['mean_ratio']:.3f} "
                f"({contract['fraction_contracted']:.0%} contracted)")
    for key, val in knn_overlap.items():
        logger.info(f"  {key}: {val:.3f}")

    return results
