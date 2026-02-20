"""
Novel Analysis Modules for Perception vs. Imagination Research
==============================================================

Six research directions for neuroscience discovery:
1. Dimensionality Gap — imagery as a compressed manifold
2. Imagery Uncertainty — MC Dropout as vividness proxy
3. Semantic Survival — what information survives imagery
4. Topological RSA — topology of perception vs imagery
5. Cross-Subject Imagery — individual imagery fingerprints
6. Semantic-Structural Dissociation — differential preservation
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
)

__all__ = [
    # Core utilities
    "EmbeddingBundle",
    "collect_embeddings",
    "generate_synthetic_embeddings",
    "load_model_for_analysis",
    # Analysis submodules
    "dimensionality",
    "imagery_uncertainty",
    "semantic_decomposition",
    "topological_rsa",
    "cross_subject",
    "semantic_structural_dissociation",
]
