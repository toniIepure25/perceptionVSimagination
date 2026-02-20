"""
Novel Analysis Modules for Perception vs Imagery Research
==========================================================

Five research directions for neuroscience discovery:
1. Dimensionality Gap — imagery as a compressed manifold
2. Imagery Uncertainty — MC Dropout as vividness proxy
3. Semantic Survival — what information survives imagery
4. Topological RSA — topology of perception vs imagery
5. Cross-Subject Imagery — individual imagery fingerprints
"""

from .core import (
    collect_embeddings,
    generate_synthetic_embeddings,
    load_model_for_analysis,
)

__all__ = [
    "collect_embeddings",
    "generate_synthetic_embeddings",
    "load_model_for_analysis",
]
