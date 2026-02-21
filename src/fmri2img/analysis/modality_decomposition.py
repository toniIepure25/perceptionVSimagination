"""
Direction 14: Modality-Invariant Decomposition
================================================

Decomposes decoded embeddings into a modality-invariant core (content
that is the same whether seen or imagined) and a modality-specific
residual (what changes between perception and imagery).

Core hypothesis: The invariant core carries semantic content; the
residual carries sensory/vividness information. If so, the core alone
should suffice for semantic retrieval, and the residual should be
classifiable as perception vs. imagery.

References:
    "Modality-Agnostic Decoding of Vision and Language from fMRI"
    (eLife, 2025)
    "Finding Shared Decodable Concepts" (2024)
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


def decompose_shared_embeddings(
    bundle: EmbeddingBundle,
) -> Optional[Dict[str, np.ndarray]]:
    """
    For shared stimuli, decompose each pair into:
      core = (perception + imagery) / 2  (modality-invariant)
      perc_residual = perception - core
      imag_residual = imagery - core
    """
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is None:
        logger.warning("No shared stimuli for modality decomposition")
        return None

    shared_ids, perc_idx, imag_idx = pairs
    perc_shared = bundle.perception[perc_idx]
    imag_shared = bundle.imagery[imag_idx]

    core = (perc_shared + imag_shared) / 2.0
    perc_residual = perc_shared - core
    imag_residual = imag_shared - core

    return {
        "core": core,
        "perc_residual": perc_residual,
        "imag_residual": imag_residual,
        "perc_shared": perc_shared,
        "imag_shared": imag_shared,
        "shared_ids": shared_ids,
        "perc_idx": perc_idx,
        "imag_idx": imag_idx,
    }


def analyze_core_content(
    core: np.ndarray,
    concept_axes: np.ndarray,
    concept_labels: List[str],
    n_semantic: int,
) -> Dict:
    """
    Project modality-invariant cores onto concept axes to characterize
    what information is preserved regardless of modality.
    """
    projections = project_onto_concepts(core, concept_axes)
    mean_proj = np.mean(np.abs(projections), axis=0)

    semantic_strength = float(np.mean(mean_proj[:n_semantic]))
    visual_strength = float(np.mean(mean_proj[n_semantic:]))

    per_concept = [
        {"concept": concept_labels[i], "mean_abs_projection": float(mean_proj[i])}
        for i in range(len(concept_labels))
    ]

    return {
        "semantic_strength": semantic_strength,
        "visual_strength": visual_strength,
        "semantic_dominance": float(semantic_strength / (visual_strength + 1e-8)),
        "per_concept": per_concept,
    }


def analyze_residual_content(
    perc_residual: np.ndarray,
    imag_residual: np.ndarray,
    concept_axes: np.ndarray,
    concept_labels: List[str],
    n_semantic: int,
) -> Dict:
    """
    Project residuals onto concept axes. Expect: residuals carry
    modality-specific (mainly visual/sensory) information.
    """
    perc_proj = project_onto_concepts(perc_residual, concept_axes)
    imag_proj = project_onto_concepts(imag_residual, concept_axes)

    perc_semantic = float(np.mean(np.abs(perc_proj[:, :n_semantic])))
    perc_visual = float(np.mean(np.abs(perc_proj[:, n_semantic:])))
    imag_semantic = float(np.mean(np.abs(imag_proj[:, :n_semantic])))
    imag_visual = float(np.mean(np.abs(imag_proj[:, n_semantic:])))

    # Residual asymmetry: perception residuals should have stronger visual content
    perc_visual_dominance = perc_visual / (perc_semantic + 1e-8)
    imag_visual_dominance = imag_visual / (imag_semantic + 1e-8)

    return {
        "perception_residual": {
            "semantic_strength": perc_semantic,
            "visual_strength": perc_visual,
            "visual_dominance": float(perc_visual_dominance),
        },
        "imagery_residual": {
            "semantic_strength": imag_semantic,
            "visual_strength": imag_visual,
            "visual_dominance": float(imag_visual_dominance),
        },
        "perc_stronger_visual_residual": bool(perc_visual > imag_visual),
    }


def predict_condition_from_residual(
    perc_residual: np.ndarray,
    imag_residual: np.ndarray,
) -> Dict:
    """
    Train a logistic regression classifier on residuals to verify
    they carry modality-specific information.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([perc_residual, imag_residual])
    y = np.array([0] * len(perc_residual) + [1] * len(imag_residual))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, C=1.0)
    n_folds = min(5, min(len(perc_residual), len(imag_residual)))
    if n_folds < 2:
        clf.fit(X_scaled, y)
        train_acc = float(clf.score(X_scaled, y))
        return {"cv_accuracy": train_acc, "cv_std": 0.0, "n_folds": 1}

    scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")
    clf.fit(X_scaled, y)

    try:
        from sklearn.metrics import roc_auc_score
        proba = clf.predict_proba(X_scaled)[:, 1]
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = float(np.mean(scores))

    return {
        "cv_accuracy": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "auc": auc,
        "n_folds": n_folds,
        "residuals_carry_modality_info": bool(np.mean(scores) > 0.6),
    }


def core_sufficiency_test(
    bundle: EmbeddingBundle,
    core: np.ndarray,
    shared_perc_idx: np.ndarray,
    shared_imag_idx: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict:
    """
    Test whether the modality-invariant core is sufficient for
    semantic retrieval. Compare retrieval@K from core embeddings
    vs full embeddings.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    perc_targets = bundle.perception_targets[shared_perc_idx]
    targets_norm = _l2(perc_targets)

    # Full-embedding retrieval: imagery -> target
    imag_shared = _l2(bundle.imagery[shared_imag_idx])
    # Core retrieval
    core_norm = _l2(core)

    n = core.shape[0]
    results = {}

    for k in k_values:
        if k > n:
            continue

        full_hits = 0
        core_hits = 0
        for i in range(n):
            # Cosine similarities to all targets
            full_sims = imag_shared[i] @ targets_norm.T
            core_sims = core_norm[i] @ targets_norm.T

            # True match is at index i
            full_topk = np.argsort(full_sims)[-k:]
            core_topk = np.argsort(core_sims)[-k:]

            if i in full_topk:
                full_hits += 1
            if i in core_topk:
                core_hits += 1

        full_recall = float(full_hits / n)
        core_recall = float(core_hits / n)

        results[f"retrieval@{k}"] = {
            "full_embedding": full_recall,
            "core_only": core_recall,
            "core_sufficient": bool(core_recall >= full_recall * 0.8),
            "retention_ratio": float(core_recall / (full_recall + 1e-8)),
        }

    return results


def analyze_modality_decomposition(
    bundle: EmbeddingBundle,
    device: str = "cpu",
) -> Dict:
    """
    Full modality-invariant decomposition analysis.
    """
    logger.info("Running Modality-Invariant Decomposition analysis...")

    decomp = decompose_shared_embeddings(bundle)
    if decomp is None:
        return {"error": "no_shared_stimuli"}

    core = decomp["core"]
    perc_res = decomp["perc_residual"]
    imag_res = decomp["imag_residual"]
    n_shared = len(decomp["shared_ids"])

    logger.info(f"  Decomposed {n_shared} shared stimuli")

    # Norm analysis
    core_norms = np.linalg.norm(core, axis=1)
    perc_res_norms = np.linalg.norm(perc_res, axis=1)
    imag_res_norms = np.linalg.norm(imag_res, axis=1)

    logger.info(f"  Core norm: {core_norms.mean():.4f}, "
                f"perc_residual norm: {perc_res_norms.mean():.4f}, "
                f"imag_residual norm: {imag_res_norms.mean():.4f}")

    # Concept-axis analysis
    all_concepts = SEMANTIC_CONCEPTS + VISUAL_CONCEPTS
    n_semantic = len(SEMANTIC_CONCEPTS)
    concept_axes = compute_concept_axes(
        all_concepts, device=device, embed_dim=bundle.embed_dim,
    )

    logger.info("  Analyzing core content (modality-invariant)...")
    core_content = analyze_core_content(core, concept_axes, all_concepts, n_semantic)
    logger.info(f"    Semantic={core_content['semantic_strength']:.4f}, "
                f"Visual={core_content['visual_strength']:.4f}")

    logger.info("  Analyzing residual content (modality-specific)...")
    residual_content = analyze_residual_content(
        perc_res, imag_res, concept_axes, all_concepts, n_semantic,
    )

    logger.info("  Classifying condition from residuals...")
    residual_classification = predict_condition_from_residual(perc_res, imag_res)
    logger.info(f"    CV accuracy: {residual_classification['cv_accuracy']:.4f}")

    logger.info("  Running core sufficiency test...")
    sufficiency = core_sufficiency_test(
        bundle, core, decomp["perc_idx"], decomp["imag_idx"],
    )

    # Per-category invariance: how similar are core embeddings across conditions?
    core_similarity = float(np.mean([
        np.dot(_l2(decomp["perc_shared"][i:i+1]).squeeze(),
               _l2(decomp["imag_shared"][i:i+1]).squeeze())
        for i in range(n_shared)
    ]))

    results = {
        "n_shared": n_shared,
        "norm_analysis": {
            "core_mean_norm": float(core_norms.mean()),
            "perc_residual_mean_norm": float(perc_res_norms.mean()),
            "imag_residual_mean_norm": float(imag_res_norms.mean()),
            "residual_ratio": float(
                perc_res_norms.mean() / (imag_res_norms.mean() + 1e-8)
            ),
        },
        "core_content": core_content,
        "residual_content": residual_content,
        "residual_classification": residual_classification,
        "core_sufficiency": sufficiency,
        "mean_perc_imag_similarity": core_similarity,
    }

    return results
