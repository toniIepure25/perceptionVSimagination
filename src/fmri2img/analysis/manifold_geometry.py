"""
Direction 13: Imagination Manifold Geometry
============================================

Maps the full geometric relationship between perception and imagery
manifolds in CLIP space: centroid distance, isotropy, convex hull
containment, centrality bias (schema attraction), and whether
relative inter-stimulus distances are preserved.

Core hypothesis: Imagery embeddings are not random degradations --
they are systematic transformations biased toward a central prototype,
analogous to schema-biased memory recall.

References:
    "Mind-to-Image: Projecting Visual Mental Imagination of the Brain
    from fMRI" (ICML 2024)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


def compute_manifold_metrics(embeddings: np.ndarray) -> Dict:
    """
    Compute geometric properties of an embedding manifold:
    centroid, mean pairwise distance, convex hull volume estimate,
    and isotropy score.
    """
    centroid = embeddings.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))

    dists_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    mean_centroid_dist = float(np.mean(dists_to_centroid))

    n = embeddings.shape[0]
    if n > 1:
        pw_dists = pdist(embeddings, metric="euclidean")
        mean_pairwise = float(np.mean(pw_dists))
        max_pairwise = float(np.max(pw_dists))
    else:
        mean_pairwise = 0.0
        max_pairwise = 0.0

    # Isotropy: how uniformly distributed are embedding directions?
    # Computed as ratio of min to max singular value of centered embeddings.
    centered = embeddings - centroid
    if n > 2 and centered.shape[1] > 1:
        n_components = min(50, min(centered.shape))
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        sv = sv[:n_components]
        isotropy = float(sv[-1] / (sv[0] + 1e-8))
    else:
        isotropy = 0.0

    # Convex hull volume estimate via random 2D projections
    volume_estimate = _estimate_hull_volume(embeddings, n_projections=20)

    return {
        "centroid_norm": centroid_norm,
        "mean_centroid_distance": mean_centroid_dist,
        "mean_pairwise_distance": mean_pairwise,
        "max_pairwise_distance": max_pairwise,
        "isotropy": isotropy,
        "hull_volume_estimate": volume_estimate,
        "n_samples": n,
    }


def _estimate_hull_volume(
    embeddings: np.ndarray,
    n_projections: int = 20,
    seed: int = 42,
) -> float:
    """Estimate relative hull volume via random 2D projection areas."""
    rng = np.random.RandomState(seed)
    areas = []
    d = embeddings.shape[1]
    for _ in range(n_projections):
        proj = rng.randn(d, 2).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        pts_2d = embeddings @ proj
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts_2d)
            areas.append(hull.volume)
        except Exception:
            x_range = pts_2d[:, 0].max() - pts_2d[:, 0].min()
            y_range = pts_2d[:, 1].max() - pts_2d[:, 1].min()
            areas.append(x_range * y_range)
    return float(np.mean(areas))


def analyze_centrality_bias(bundle: EmbeddingBundle) -> Dict:
    """
    For shared stimuli: test whether imagery embeddings are systematically
    closer to the perception centroid than the corresponding perception
    embeddings ("schema bias" / prototype attraction).
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        logger.warning("No shared stimuli for centrality bias analysis")
        return {"error": "no_shared_stimuli"}

    shared_ids, perc_idx, imag_idx = pairs
    perc_shared = bundle.perception[perc_idx]
    imag_shared = bundle.imagery[imag_idx]

    perc_centroid = bundle.perception.mean(axis=0)

    perc_dist_to_centroid = np.linalg.norm(perc_shared - perc_centroid, axis=1)
    imag_dist_to_centroid = np.linalg.norm(imag_shared - perc_centroid, axis=1)

    bias = perc_dist_to_centroid - imag_dist_to_centroid
    mean_bias = float(np.mean(bias))

    # Positive mean_bias => imagery is closer to centroid (schema bias confirmed)
    stat, p_val = scipy_stats.wilcoxon(bias, alternative="greater")

    fraction_closer = float(np.mean(bias > 0))

    return {
        "mean_centrality_bias": mean_bias,
        "fraction_imagery_closer_to_centroid": fraction_closer,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_val),
        "schema_bias_confirmed": bool(p_val < 0.05 and mean_bias > 0),
        "n_shared": len(shared_ids),
        "perc_mean_dist": float(np.mean(perc_dist_to_centroid)),
        "imag_mean_dist": float(np.mean(imag_dist_to_centroid)),
    }


def compute_relative_position_preservation(bundle: EmbeddingBundle) -> Dict:
    """
    Test whether relative positions (pairwise distance rank order) between
    stimuli are preserved from perception to imagery using Procrustes analysis.
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        return {"error": "no_shared_stimuli"}

    shared_ids, perc_idx, imag_idx = pairs
    perc_shared = bundle.perception[perc_idx]
    imag_shared = bundle.imagery[imag_idx]

    n = len(shared_ids)
    if n < 3:
        return {"error": "too_few_shared_stimuli"}

    # PCA to shared low-dimensional space for stable Procrustes
    n_components = min(50, min(n, perc_shared.shape[1]))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    perc_pca = pca.fit_transform(perc_shared)
    imag_pca = pca.transform(imag_shared)

    # Center both
    perc_c = perc_pca - perc_pca.mean(axis=0)
    imag_c = imag_pca - imag_pca.mean(axis=0)

    # Scale to unit Frobenius norm
    perc_s = perc_c / (np.linalg.norm(perc_c, "fro") + 1e-8)
    imag_s = imag_c / (np.linalg.norm(imag_c, "fro") + 1e-8)

    # Procrustes: find optimal rotation
    U, _, Vt = np.linalg.svd(imag_s.T @ perc_s)
    R = U @ Vt
    imag_aligned = imag_s @ R

    procrustes_distance = float(np.linalg.norm(perc_s - imag_aligned, "fro"))

    # Rank-order preservation (Spearman on pairwise distances)
    perc_pw = pdist(perc_shared, "cosine")
    imag_pw = pdist(imag_shared, "cosine")
    rho, rho_p = scipy_stats.spearmanr(perc_pw, imag_pw)

    return {
        "procrustes_distance": procrustes_distance,
        "rank_order_spearman": float(rho) if np.isfinite(rho) else 0.0,
        "rank_order_p_value": float(rho_p) if np.isfinite(rho_p) else 1.0,
        "structure_preserved": bool(np.isfinite(rho) and rho > 0.3 and rho_p < 0.05),
        "n_shared": n,
        "n_pca_components": n_components,
    }


def analyze_interpolation_structure(bundle: EmbeddingBundle) -> Dict:
    """
    Test whether imagery embeddings can be expressed as interpolations
    between perception embeddings (lying on geodesics in perception space).
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        return {"error": "no_shared_stimuli"}

    shared_ids, perc_idx, imag_idx = pairs
    perc_shared = bundle.perception[perc_idx]
    imag_shared = bundle.imagery[imag_idx]
    all_perc = _l2(bundle.perception)

    interpolation_coefficients = []
    reconstruction_errors = []

    for i in range(len(shared_ids)):
        imag_point = _l2(imag_shared[i].reshape(1, -1)).squeeze()

        # Find 2 closest perception embeddings (excluding the matched one)
        cosines = all_perc @ imag_point
        # Exclude exact match
        if perc_idx[i] < len(cosines):
            cosines[perc_idx[i]] = -2.0
        top2 = np.argsort(cosines)[-2:]
        p1, p2 = all_perc[top2[0]], all_perc[top2[1]]

        # Optimal interpolation: imag ≈ α*p1 + (1-α)*p2
        # Solve: α = <imag - p2, p1 - p2> / <p1 - p2, p1 - p2>
        diff = p1 - p2
        denom = np.dot(diff, diff)
        if denom > 1e-8:
            alpha = np.dot(imag_point - p2, diff) / denom
            alpha = float(np.clip(alpha, 0, 1))
        else:
            alpha = 0.5

        reconstructed = alpha * p1 + (1 - alpha) * p2
        reconstructed = reconstructed / (np.linalg.norm(reconstructed) + 1e-8)
        error = float(np.linalg.norm(imag_point - reconstructed))

        interpolation_coefficients.append(alpha)
        reconstruction_errors.append(error)

    coeffs = np.array(interpolation_coefficients)
    errors = np.array(reconstruction_errors)

    # Baseline: random point reconstruction error
    rng = np.random.RandomState(42)
    random_errors = []
    for _ in range(len(shared_ids)):
        rand_point = _l2(rng.randn(1, bundle.embed_dim).astype(np.float32)).squeeze()
        top2 = np.argsort(all_perc @ rand_point)[-2:]
        p1, p2 = all_perc[top2[0]], all_perc[top2[1]]
        diff = p1 - p2
        denom = np.dot(diff, diff)
        alpha = np.dot(rand_point - p2, diff) / (denom + 1e-8)
        alpha = float(np.clip(alpha, 0, 1))
        recon = alpha * p1 + (1 - alpha) * p2
        recon = recon / (np.linalg.norm(recon) + 1e-8)
        random_errors.append(float(np.linalg.norm(rand_point - recon)))

    random_errors = np.array(random_errors)

    stat, p_val = scipy_stats.mannwhitneyu(errors, random_errors, alternative="less")

    return {
        "mean_interpolation_error": float(np.mean(errors)),
        "std_interpolation_error": float(np.std(errors)),
        "mean_random_error": float(np.mean(random_errors)),
        "improvement_over_random": float(np.mean(random_errors) - np.mean(errors)),
        "mann_whitney_stat": float(stat),
        "p_value": float(p_val),
        "imagery_on_geodesics": bool(p_val < 0.05),
        "mean_alpha": float(np.mean(coeffs)),
        "std_alpha": float(np.std(coeffs)),
        "n_shared": len(shared_ids),
    }


def analyze_manifold_geometry(bundle: EmbeddingBundle) -> Dict:
    """
    Full imagination manifold geometry analysis.
    """
    logger.info("Running Imagination Manifold Geometry analysis...")

    logger.info("  Computing perception manifold metrics...")
    perc_metrics = compute_manifold_metrics(bundle.perception)
    logger.info(f"    Isotropy={perc_metrics['isotropy']:.4f}, "
                f"Hull={perc_metrics['hull_volume_estimate']:.4f}")

    logger.info("  Computing imagery manifold metrics...")
    imag_metrics = compute_manifold_metrics(bundle.imagery)
    logger.info(f"    Isotropy={imag_metrics['isotropy']:.4f}, "
                f"Hull={imag_metrics['hull_volume_estimate']:.4f}")

    logger.info("  Analyzing centrality bias (schema attraction)...")
    centrality = analyze_centrality_bias(bundle)
    if "error" not in centrality:
        logger.info(f"    Bias={centrality['mean_centrality_bias']:.4f}, "
                     f"p={centrality['p_value']:.4f}")

    logger.info("  Computing relative position preservation...")
    positions = compute_relative_position_preservation(bundle)
    if "error" not in positions:
        logger.info(f"    Procrustes={positions['procrustes_distance']:.4f}, "
                     f"Spearman={positions['rank_order_spearman']:.4f}")

    logger.info("  Analyzing interpolation structure...")
    interp = analyze_interpolation_structure(bundle)
    if "error" not in interp:
        logger.info(f"    Interp error={interp['mean_interpolation_error']:.4f} "
                     f"(random={interp['mean_random_error']:.4f})")

    # Ratios
    hull_ratio = (imag_metrics["hull_volume_estimate"] /
                  (perc_metrics["hull_volume_estimate"] + 1e-8))
    isotropy_gap = perc_metrics["isotropy"] - imag_metrics["isotropy"]
    centroid_distance_ratio = (imag_metrics["mean_centroid_distance"] /
                               (perc_metrics["mean_centroid_distance"] + 1e-8))

    results = {
        "perception_manifold": perc_metrics,
        "imagery_manifold": imag_metrics,
        "hull_volume_ratio": float(hull_ratio),
        "isotropy_gap": float(isotropy_gap),
        "centroid_distance_ratio": float(centroid_distance_ratio),
        "centrality_bias": centrality,
        "position_preservation": positions,
        "interpolation_structure": interp,
    }

    logger.info(f"  Hull ratio (imag/perc): {hull_ratio:.4f}")
    logger.info(f"  Isotropy gap: {isotropy_gap:.4f}")

    return results
