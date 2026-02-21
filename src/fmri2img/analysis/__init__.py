"""
Novel Analysis Modules for Perception vs. Imagination Research
==============================================================

Fifteen research directions for neuroscience discovery:
 1. Dimensionality Gap — imagery as a compressed manifold
 2. Imagery Uncertainty — MC Dropout as vividness proxy
 3. Semantic Survival — what information survives imagery
 4. Topological RSA — topology of perception vs imagery
 5. Cross-Subject Imagery — individual imagery fingerprints
 6. Semantic-Structural Dissociation — differential preservation
 7. Computational Reality Monitor — Dijkstra's PRM theory
 8. Reality Confusion Mapping — confusion index & boundary
 9. Adversarial Reality Probing — GAN-based reality gap
10. Hierarchical Reality Gradient — per-layer emergence
11. Compositional Imagination — Brain Algebra in decoded space
12. Predictive Coding Residuals — information flow direction
13. Imagination Manifold Geometry — centrality bias & structure
14. Modality-Invariant Decomposition — core vs residual
15. Creative Divergence Mapping — imagination transformation rules
"""

from .core import (
    EmbeddingBundle,
    collect_embeddings,
    generate_synthetic_embeddings,
    load_model_for_analysis,
)
from . import (
    dimensionality,
    imagery_uncertainty,
    semantic_decomposition,
    topological_rsa,
    cross_subject,
    semantic_structural_dissociation,
    reality_monitor,
    reality_confusion,
    adversarial_reality,
    hierarchical_reality,
    compositional_imagination,
    predictive_coding,
    manifold_geometry,
    modality_decomposition,
    creative_divergence,
)

__all__ = [
    # Core utilities
    "EmbeddingBundle",
    "collect_embeddings",
    "generate_synthetic_embeddings",
    "load_model_for_analysis",
    # Directions 1-6
    "dimensionality",
    "imagery_uncertainty",
    "semantic_decomposition",
    "topological_rsa",
    "cross_subject",
    "semantic_structural_dissociation",
    # Directions 7-10
    "reality_monitor",
    "reality_confusion",
    "adversarial_reality",
    "hierarchical_reality",
    # Directions 11-15
    "compositional_imagination",
    "predictive_coding",
    "manifold_geometry",
    "modality_decomposition",
    "creative_divergence",
]
