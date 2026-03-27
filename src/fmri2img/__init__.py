"""
fmri2img -- Canonical perception/imagery decoding platform
==========================================================

The package retains the historical ``fmri2img`` name for compatibility while
expanding the repository into a broader research platform for disentangling
shared content, private neural variance, and subjective experience from fMRI.

Canonical platform modules:
- ``data`` / ``preprocessing`` / ``roi`` / ``targets``
- ``models`` / ``training`` / ``evaluation`` / ``export`` / ``workflows``

Legacy perception-only and reconstruction-first code paths remain available for
comparison, but the shared/private decoder workflows are now the official path.
"""

__version__ = "0.4.0"

__all__ = [
    "analysis",
    "data",
    "evaluation",
    "eval",
    "export",
    "generation",
    "io",
    "legacy",
    "models",
    "preprocessing",
    "reliability",
    "roi",
    "stats",
    "targets",
    "training",
    "utils",
    "workflows",
]
