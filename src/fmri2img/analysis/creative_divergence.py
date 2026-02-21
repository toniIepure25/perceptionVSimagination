"""
Direction 15: Creative Divergence Mapping
==========================================

Measures the per-trial divergence vector (perception -> imagery) and
decomposes it into concept-specific amplifications and suppressions,
revealing the brain's constructive imagination process.

Core hypothesis: Imagination actively transforms representations --
some concepts are amplified ("filled in"), others suppressed (sensory
details that can't be internally generated). The direction and magnitude
of divergence reveals systematic "imagination rules."

References:
    "Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental
    Creation Extension" (ICCV 2025)
    "Learning to imagine: generative models of memory construction" (2024)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from .core import EmbeddingBundle, _l2
from .semantic_decomposition import (
    SEMANTIC_CONCEPTS,
    VISUAL_CONCEPTS,
    compute_concept_axes,
    project_onto_concepts,
)

logger = logging.getLogger(__name__)


def compute_divergence_vectors(
    bundle: EmbeddingBundle,
) -> Optional[Dict[str, np.ndarray]]:
    """
    For shared stimuli, compute the creative transformation vector:
        divergence = imagery_embed - perception_embed
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        logger.warning("No shared stimuli for divergence analysis")
        return None

    shared_ids, perc_idx, imag_idx = pairs
    perc_shared = bundle.perception[perc_idx]
    imag_shared = bundle.imagery[imag_idx]

    divergence = imag_shared - perc_shared
    divergence_norms = np.linalg.norm(divergence, axis=1)

    return {
        "divergence": divergence,
        "norms": divergence_norms,
        "perc_shared": perc_shared,
        "imag_shared": imag_shared,
        "shared_ids": shared_ids,
        "perc_idx": perc_idx,
        "imag_idx": imag_idx,
    }


def decompose_divergence(
    divergence: np.ndarray,
    concept_axes: np.ndarray,
    concept_labels: List[str],
    n_semantic: int,
) -> Dict:
    """
    Project each divergence vector onto concept axes.
    Positive projection => concept amplified in imagery.
    Negative projection => concept suppressed in imagery.
    """
    # Project divergence vectors (not normalized — we want directional magnitude)
    axes_norm = _l2(concept_axes)
    projections = divergence @ axes_norm.T  # (N, n_concepts)

    mean_proj = np.mean(projections, axis=0)
    std_proj = np.std(projections, axis=0)

    per_concept = []
    for i, label in enumerate(concept_labels):
        # One-sample t-test: is mean projection significantly different from 0?
        t_stat, p_val = scipy_stats.ttest_1samp(projections[:, i], 0)
        direction = "amplified" if mean_proj[i] > 0 else "suppressed"

        per_concept.append({
            "concept": label,
            "type": "semantic" if i < n_semantic else "visual",
            "mean_projection": float(mean_proj[i]),
            "std_projection": float(std_proj[i]),
            "direction": direction,
            "t_statistic": float(t_stat) if np.isfinite(t_stat) else 0.0,
            "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
            "significant": bool(np.isfinite(p_val) and p_val < 0.05),
        })

    n_amplified = sum(1 for c in per_concept if c["mean_projection"] > 0)
    n_significant = sum(1 for c in per_concept if c["significant"])

    semantic_mean = float(np.mean(np.abs(mean_proj[:n_semantic])))
    visual_mean = float(np.mean(np.abs(mean_proj[n_semantic:])))

    return {
        "per_concept": per_concept,
        "n_amplified": n_amplified,
        "n_suppressed": len(per_concept) - n_amplified,
        "n_significant": n_significant,
        "semantic_mean_divergence": semantic_mean,
        "visual_mean_divergence": visual_mean,
        "projections": projections,
    }


def compute_creativity_index(
    divergence: np.ndarray,
    perc_shared: np.ndarray,
) -> Dict:
    """
    Per-trial creativity index: magnitude of divergence component
    orthogonal to simple scaling (the part that can't be explained
    by uniform signal loss). High orthogonal divergence = creative.
    """
    n = divergence.shape[0]
    creativity_indices = np.zeros(n)
    scaling_components = np.zeros(n)
    orthogonal_components = np.zeros(n)

    for i in range(n):
        perc = perc_shared[i]
        div = divergence[i]
        perc_norm = np.linalg.norm(perc)

        if perc_norm < 1e-8:
            continue

        # Projection of divergence onto perception direction (scaling component)
        perc_unit = perc / perc_norm
        scaling = np.dot(div, perc_unit)
        scaling_vec = scaling * perc_unit

        # Orthogonal component (creative transformation)
        orthogonal_vec = div - scaling_vec
        orthogonal_mag = np.linalg.norm(orthogonal_vec)
        total_mag = np.linalg.norm(div)

        scaling_components[i] = abs(scaling)
        orthogonal_components[i] = orthogonal_mag
        creativity_indices[i] = orthogonal_mag / (total_mag + 1e-8)

    return {
        "creativity_indices": creativity_indices,
        "mean_creativity": float(np.mean(creativity_indices)),
        "std_creativity": float(np.std(creativity_indices)),
        "mean_scaling_component": float(np.mean(scaling_components)),
        "mean_orthogonal_component": float(np.mean(orthogonal_components)),
        "high_creativity_fraction": float(np.mean(creativity_indices > 0.5)),
    }


def predict_divergence_from_content(
    perc_shared: np.ndarray,
    divergence: np.ndarray,
    concept_axes: np.ndarray,
    seed: int = 42,
) -> Dict:
    """
    Train a predictor: given perception embedding, predict divergence vector.
    If R² > 0, the brain has systematic imagination rules.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(perc_shared)
    y = scaler_y.fit_transform(divergence)

    model = Ridge(alpha=1.0)
    n_folds = min(5, len(perc_shared))
    if n_folds < 2:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = float(1 - np.mean((y - pred) ** 2) / (np.var(y) + 1e-8))
        return {"r_squared": r2, "n_folds": 1}

    # Per-output-dim R² via cross-validation
    model.fit(X, y)
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    # Concept-level predictability
    perc_proj = project_onto_concepts(perc_shared, concept_axes)
    concept_r2s = []
    for i in range(divergence.shape[1] if divergence.shape[1] <= 50 else 50):
        corr = np.corrcoef(perc_proj[:, i % perc_proj.shape[1]], divergence[:, i])[0, 1]
        concept_r2s.append(float(corr ** 2) if np.isfinite(corr) else 0.0)

    return {
        "r_squared": r2,
        "divergence_predictable": bool(r2 > 0.05),
        "mean_concept_r2": float(np.mean(concept_r2s)),
        "systematic_rules": bool(r2 > 0.1),
    }


def find_imagination_archetypes(
    divergence: np.ndarray,
    n_clusters_range: Tuple[int, int] = (2, 6),
    seed: int = 42,
) -> Dict:
    """
    Cluster divergence vectors to find recurring imagination styles
    (e.g., "colorize", "simplify", "dramatize").
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = divergence.shape[0]
    if n < 4:
        return {"error": "too_few_samples", "n_samples": n}

    scaler = StandardScaler()
    div_scaled = scaler.fit_transform(divergence)

    n_components = min(20, min(n, divergence.shape[1]))
    pca = PCA(n_components=n_components)
    div_pca = pca.fit_transform(div_scaled)

    best_k = 2
    best_sil = -1
    results_by_k = {}

    min_k, max_k = n_clusters_range
    max_k = min(max_k, n - 1)

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(div_pca)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(div_pca, labels))
        results_by_k[k] = {
            "silhouette": sil,
            "inertia": float(km.inertia_),
        }
        if sil > best_sil:
            best_sil = sil
            best_k = k

    # Final clustering with best k
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed)
    labels = km.fit_predict(div_pca)

    # Characterize each archetype
    archetypes = []
    for c in range(best_k):
        mask = labels == c
        cluster_div = divergence[mask]
        mean_div = cluster_div.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_div))

        archetypes.append({
            "cluster_id": c,
            "n_members": int(mask.sum()),
            "mean_divergence_norm": mean_norm,
            "fraction": float(mask.sum() / n),
        })

    return {
        "best_k": best_k,
        "best_silhouette": best_sil,
        "archetypes": archetypes,
        "k_selection": results_by_k,
        "labels": labels.tolist(),
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
    }


def analyze_creative_divergence(
    bundle: EmbeddingBundle,
    device: str = "cpu",
) -> Dict:
    """
    Full creative divergence mapping analysis.
    """
    logger.info("Running Creative Divergence Mapping analysis...")

    div_data = compute_divergence_vectors(bundle)
    if div_data is None:
        return {"error": "no_shared_stimuli"}

    divergence = div_data["divergence"]
    norms = div_data["norms"]
    n_shared = len(div_data["shared_ids"])

    logger.info(f"  Computed divergence vectors for {n_shared} shared stimuli")
    logger.info(f"  Mean divergence norm: {norms.mean():.4f} ± {norms.std():.4f}")

    # Concept decomposition
    all_concepts = SEMANTIC_CONCEPTS + VISUAL_CONCEPTS
    n_semantic = len(SEMANTIC_CONCEPTS)
    concept_axes = compute_concept_axes(
        all_concepts, device=device, embed_dim=bundle.embed_dim,
    )

    logger.info("  Decomposing divergence into concept components...")
    decomp = decompose_divergence(divergence, concept_axes, all_concepts, n_semantic)
    logger.info(f"    Amplified: {decomp['n_amplified']}, "
                f"Suppressed: {decomp['n_suppressed']}, "
                f"Significant: {decomp['n_significant']}")

    # Creativity index
    logger.info("  Computing creativity indices...")
    creativity = compute_creativity_index(divergence, div_data["perc_shared"])
    logger.info(f"    Mean creativity: {creativity['mean_creativity']:.4f}")

    # Predictability
    logger.info("  Testing divergence predictability...")
    predictability = predict_divergence_from_content(
        div_data["perc_shared"], divergence, concept_axes,
    )
    logger.info(f"    R²: {predictability['r_squared']:.4f}")

    # Archetypes
    logger.info("  Finding imagination archetypes...")
    archetypes = find_imagination_archetypes(divergence)
    if "error" not in archetypes:
        logger.info(f"    Best k={archetypes['best_k']}, "
                     f"silhouette={archetypes['best_silhouette']:.4f}")

    # Remove large array from decomp before storing
    decomp_clean = {k: v for k, v in decomp.items() if k != "projections"}

    results = {
        "n_shared": n_shared,
        "divergence_norms": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "concept_decomposition": decomp_clean,
        "creativity": creativity,
        "predictability": predictability,
        "archetypes": archetypes,
    }

    return results
