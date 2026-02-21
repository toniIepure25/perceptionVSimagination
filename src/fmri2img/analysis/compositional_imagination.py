"""
Direction 11: Compositional Imagination (Brain Algebra)
========================================================

Tests whether vector arithmetic on decoded CLIP embeddings produces
semantically meaningful new concepts, implementing "Brain Algebra"
(Communications Biology, 2025) in decoded representation space.

If the brain uses algebraic composition to build novel concepts,
then concept perturbation vectors extracted from decoded embeddings
should transfer across stimuli, enabling computational creativity.

References:
    "Evidence for compositionality in fMRI visual representations
    via Brain Algebra" (Communications Biology, 2025)
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


def compute_concept_perturbations(
    embeddings: np.ndarray,
    concept_axes: np.ndarray,
    top_fraction: float = 0.25,
) -> np.ndarray:
    """
    For each concept axis, compute a perturbation vector as the difference
    between the mean embedding of high-scoring and low-scoring trials.

    This yields a direction in decoded CLIP space that "adds" or "removes"
    that concept from an embedding.

    Returns (n_concepts, D) array of perturbation vectors.
    """
    projections = project_onto_concepts(embeddings, concept_axes)
    n_concepts = concept_axes.shape[0]
    n = embeddings.shape[0]
    k = max(1, int(n * top_fraction))

    perturbations = np.zeros_like(concept_axes)
    for c in range(n_concepts):
        scores = projections[:, c]
        high_idx = np.argsort(scores)[-k:]
        low_idx = np.argsort(scores)[:k]
        perturbations[c] = embeddings[high_idx].mean(axis=0) - embeddings[low_idx].mean(axis=0)

    return perturbations


def apply_brain_algebra(
    embedding: np.ndarray,
    add_vectors: Optional[List[np.ndarray]] = None,
    subtract_vectors: Optional[List[np.ndarray]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply vector arithmetic to an embedding:
        result = embedding + scale * sum(add) - scale * sum(subtract)
    Then L2-normalize.
    """
    result = embedding.copy()
    if add_vectors:
        for v in add_vectors:
            result = result + scale * v
    if subtract_vectors:
        for v in subtract_vectors:
            result = result - scale * v
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result


def evaluate_composition_quality(
    embeddings: np.ndarray,
    targets: np.ndarray,
    concept_axes: np.ndarray,
    perturbations: np.ndarray,
    n_trials: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Systematically test concept perturbations: for each trial, add a
    concept perturbation and check if the result moves closer to
    embeddings that naturally have that concept (composition success).
    """
    rng = np.random.RandomState(seed)
    projections = project_onto_concepts(embeddings, concept_axes)
    n = embeddings.shape[0]
    n_concepts = concept_axes.shape[0]

    successes = 0
    total = 0
    cosine_shifts = []
    per_concept_success = np.zeros(n_concepts)
    per_concept_total = np.zeros(n_concepts)

    for _ in range(n_trials):
        trial_idx = rng.randint(0, n)
        concept_idx = rng.randint(0, n_concepts)

        original = embeddings[trial_idx]
        perturbation = perturbations[concept_idx]

        # Apply perturbation (add concept)
        result = apply_brain_algebra(original, add_vectors=[perturbation], scale=0.5)

        # Find trials that are high on this concept as "expected targets"
        scores = projections[:, concept_idx]
        high_k = max(1, n // 4)
        high_idx = np.argsort(scores)[-high_k:]
        target_mean = _l2(embeddings[high_idx].mean(axis=0, keepdims=True)).squeeze()

        original_norm = _l2(original.reshape(1, -1)).squeeze()
        result_norm = _l2(result.reshape(1, -1)).squeeze()

        cos_before = float(np.dot(original_norm, target_mean))
        cos_after = float(np.dot(result_norm, target_mean))
        shift = cos_after - cos_before
        cosine_shifts.append(shift)

        if shift > 0:
            successes += 1
            per_concept_success[concept_idx] += 1
        per_concept_total[concept_idx] += 1
        total += 1

    per_concept_rate = np.where(
        per_concept_total > 0, per_concept_success / per_concept_total, 0.0
    )

    return {
        "success_rate": float(successes / max(total, 1)),
        "mean_cosine_shift": float(np.mean(cosine_shifts)),
        "std_cosine_shift": float(np.std(cosine_shifts)),
        "n_trials": total,
        "per_concept_success_rate": per_concept_rate.tolist(),
        "shift_positive_fraction": float(np.mean(np.array(cosine_shifts) > 0)),
    }


def compare_perception_vs_imagery_compositionality(
    bundle: EmbeddingBundle,
    concept_axes: np.ndarray,
    n_trials: int = 200,
) -> Dict:
    """
    Test whether perception embeddings are more compositional than
    imagery embeddings by running Brain Algebra on both conditions.
    """
    perc_perturbations = compute_concept_perturbations(
        bundle.perception, concept_axes
    )
    imag_perturbations = compute_concept_perturbations(
        bundle.imagery, concept_axes
    )

    perc_quality = evaluate_composition_quality(
        bundle.perception, bundle.perception_targets,
        concept_axes, perc_perturbations, n_trials=n_trials,
    )
    imag_quality = evaluate_composition_quality(
        bundle.imagery, bundle.imagery_targets,
        concept_axes, imag_perturbations, n_trials=n_trials,
    )

    gap = perc_quality["success_rate"] - imag_quality["success_rate"]

    return {
        "perception_compositionality": perc_quality,
        "imagery_compositionality": imag_quality,
        "compositionality_gap": float(gap),
        "perception_more_compositional": bool(gap > 0),
    }


def analyze_compositional_imagination(
    bundle: EmbeddingBundle,
    device: str = "cpu",
    n_trials: int = 200,
) -> Dict:
    """
    Full compositional imagination (Brain Algebra) analysis.
    """
    logger.info("Running Compositional Imagination (Brain Algebra) analysis...")

    all_concepts = SEMANTIC_CONCEPTS + VISUAL_CONCEPTS
    n_semantic = len(SEMANTIC_CONCEPTS)
    concept_axes = compute_concept_axes(
        all_concepts, device=device, embed_dim=bundle.embed_dim,
    )
    logger.info(f"  Computed {len(all_concepts)} concept axes")

    # Concept perturbations from perception embeddings
    perturbations = compute_concept_perturbations(bundle.perception, concept_axes)
    perturbation_norms = np.linalg.norm(perturbations, axis=1)
    logger.info(f"  Perturbation norms: {perturbation_norms.mean():.4f} "
                f"(semantic={perturbation_norms[:n_semantic].mean():.4f}, "
                f"visual={perturbation_norms[n_semantic:].mean():.4f})")

    # Overall composition quality
    logger.info("  Evaluating composition quality on perception embeddings...")
    overall_quality = evaluate_composition_quality(
        bundle.perception, bundle.perception_targets,
        concept_axes, perturbations, n_trials=n_trials,
    )
    logger.info(f"    Success rate: {overall_quality['success_rate']:.4f}")
    logger.info(f"    Mean cosine shift: {overall_quality['mean_cosine_shift']:.4f}")

    # Perception vs imagery comparison
    logger.info("  Comparing perception vs imagery compositionality...")
    comparison = compare_perception_vs_imagery_compositionality(
        bundle, concept_axes, n_trials=n_trials,
    )
    logger.info(f"    Perception success: "
                f"{comparison['perception_compositionality']['success_rate']:.4f}")
    logger.info(f"    Imagery success: "
                f"{comparison['imagery_compositionality']['success_rate']:.4f}")
    logger.info(f"    Gap: {comparison['compositionality_gap']:.4f}")

    # Semantic vs visual concept compositionality
    semantic_rates = overall_quality["per_concept_success_rate"][:n_semantic]
    visual_rates = overall_quality["per_concept_success_rate"][n_semantic:]

    results = {
        "overall_quality": overall_quality,
        "comparison": comparison,
        "semantic_mean_success": float(np.mean(semantic_rates)),
        "visual_mean_success": float(np.mean(visual_rates)),
        "semantic_vs_visual_gap": float(np.mean(semantic_rates) - np.mean(visual_rates)),
        "perturbation_norms": {
            "mean": float(perturbation_norms.mean()),
            "semantic_mean": float(perturbation_norms[:n_semantic].mean()),
            "visual_mean": float(perturbation_norms[n_semantic:].mean()),
        },
        "concept_labels": all_concepts,
        "n_semantic": n_semantic,
    }

    return results
