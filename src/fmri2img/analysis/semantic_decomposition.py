"""
Direction 3: Semantic Survival
===============================

Decomposes CLIP embeddings into interpretable semantic and visual axes to
determine which information survives the perception-to-imagery transition.

Uses CLIP text embeddings of concept descriptions as basis vectors, then
measures how well perception and imagery decoder outputs align with each axis.

References:
    Radford et al. (2021). "Learning Transferable Visual Models From Natural
    Language Supervision" (CLIP structure)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)

# Concept axes for decomposition — semantic (high-level) vs visual (low-level)
SEMANTIC_CONCEPTS = [
    # Object categories
    "a photo of an animal",
    "a photo of a person",
    "a photo of a vehicle",
    "a photo of food",
    "a photo of a building",
    "a photo of furniture",
    "a photo of nature",
    "a photo of text or writing",
    # Scene types
    "an indoor scene",
    "an outdoor scene",
    "a close-up photo",
    "a landscape photo",
    # Abstract
    "something happy or positive",
    "something scary or negative",
    "an action or activity",
]

VISUAL_CONCEPTS = [
    # Color
    "a mostly red image",
    "a mostly blue image",
    "a mostly green image",
    "a bright colorful image",
    "a dark image",
    # Texture and pattern
    "a smooth texture",
    "a rough texture",
    "a striped pattern",
    "a detailed complex image",
    "a simple minimal image",
    # Spatial
    "objects on the left side",
    "objects on the right side",
    "a symmetric composition",
    "a cluttered scene",
    "a single centered object",
]


def compute_concept_axes(
    concepts: List[str],
    clip_model=None,
    preprocess=None,
    device: str = "cpu",
    embed_dim: int = 512,
) -> np.ndarray:
    """
    Compute CLIP text embeddings for concept descriptions.
    Returns (n_concepts, embed_dim) array of L2-normalized embeddings.
    """
    try:
        import clip as clip_module
    except ImportError:
        logger.warning("CLIP not available; generating random concept axes")
        return _l2(np.random.randn(len(concepts), embed_dim).astype(np.float32))

    import torch

    if clip_model is None:
        clip_model, preprocess = clip_module.load("ViT-B/32", device=device)
        clip_model.eval()

    embeddings = []
    with torch.no_grad():
        for text in concepts:
            tokens = clip_module.tokenize([text]).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])

    return np.array(embeddings, dtype=np.float32)


def project_onto_concepts(
    embeddings: np.ndarray, concept_axes: np.ndarray
) -> np.ndarray:
    """
    Project embeddings onto concept axes via cosine similarity.
    Returns (N, n_concepts) matrix of projection strengths.
    """
    emb_norm = _l2(embeddings)
    axes_norm = _l2(concept_axes)
    return emb_norm @ axes_norm.T


def compute_preservation_profile(
    perc_proj: np.ndarray,
    imag_proj: np.ndarray,
    perc_gt_proj: np.ndarray,
    imag_gt_proj: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compare how well each concept axis is preserved in imagery vs perception.

    For each concept axis c, preservation(c) measures the correlation between
    the decoder's projection and the ground-truth projection.
    """
    n_concepts = perc_proj.shape[1]

    perc_accuracy = np.zeros(n_concepts)
    imag_accuracy = np.zeros(n_concepts)

    for c in range(n_concepts):
        perc_corr = np.corrcoef(perc_proj[:, c], perc_gt_proj[:, c])[0, 1]
        perc_accuracy[c] = perc_corr if np.isfinite(perc_corr) else 0.0

        imag_corr = np.corrcoef(imag_proj[:, c], imag_gt_proj[:, c])[0, 1]
        imag_accuracy[c] = imag_corr if np.isfinite(imag_corr) else 0.0

    preservation_ratio = np.where(
        np.abs(perc_accuracy) > 1e-6,
        imag_accuracy / perc_accuracy,
        0.0,
    )

    return {
        "perception_accuracy": perc_accuracy,
        "imagery_accuracy": imag_accuracy,
        "preservation_ratio": preservation_ratio,
    }


def analyze_semantic_survival(
    bundle: EmbeddingBundle,
    device: str = "cpu",
    semantic_concepts: Optional[List[str]] = None,
    visual_concepts: Optional[List[str]] = None,
) -> Dict:
    """
    Full semantic survival analysis.

    Decomposes both perception and imagery embeddings along semantic and visual
    axes, then compares preservation ratios across the two groups.
    """
    logger.info("Running semantic survival analysis...")

    if semantic_concepts is None:
        semantic_concepts = SEMANTIC_CONCEPTS
    if visual_concepts is None:
        visual_concepts = VISUAL_CONCEPTS

    all_concepts = semantic_concepts + visual_concepts
    n_semantic = len(semantic_concepts)

    concept_axes = compute_concept_axes(
        all_concepts, device=device, embed_dim=bundle.embed_dim
    )
    logger.info(f"  Computed {len(all_concepts)} concept axes "
                f"({n_semantic} semantic, {len(visual_concepts)} visual)")

    # Project all embedding sets
    perc_proj = project_onto_concepts(bundle.perception, concept_axes)
    imag_proj = project_onto_concepts(bundle.imagery, concept_axes)
    perc_gt_proj = project_onto_concepts(bundle.perception_targets, concept_axes)
    imag_gt_proj = project_onto_concepts(bundle.imagery_targets, concept_axes)

    profile = compute_preservation_profile(perc_proj, imag_proj, perc_gt_proj, imag_gt_proj)

    semantic_preservation = profile["preservation_ratio"][:n_semantic]
    visual_preservation = profile["preservation_ratio"][n_semantic:]

    # Build per-concept detail
    concept_details = []
    for i, concept in enumerate(all_concepts):
        concept_details.append({
            "concept": concept,
            "type": "semantic" if i < n_semantic else "visual",
            "perception_accuracy": float(profile["perception_accuracy"][i]),
            "imagery_accuracy": float(profile["imagery_accuracy"][i]),
            "preservation_ratio": float(profile["preservation_ratio"][i]),
        })

    results = {
        "semantic_mean_preservation": float(np.mean(semantic_preservation)),
        "visual_mean_preservation": float(np.mean(visual_preservation)),
        "semantic_std_preservation": float(np.std(semantic_preservation)),
        "visual_std_preservation": float(np.std(visual_preservation)),
        "semantic_vs_visual_gap": float(
            np.mean(semantic_preservation) - np.mean(visual_preservation)
        ),
        "n_semantic_concepts": n_semantic,
        "n_visual_concepts": len(visual_concepts),
        "concept_details": concept_details,
    }

    logger.info(f"  Semantic preservation: {results['semantic_mean_preservation']:.3f} "
                f"± {results['semantic_std_preservation']:.3f}")
    logger.info(f"  Visual preservation:   {results['visual_mean_preservation']:.3f} "
                f"± {results['visual_std_preservation']:.3f}")
    logger.info(f"  Gap (semantic - visual): {results['semantic_vs_visual_gap']:.3f}")

    return results
