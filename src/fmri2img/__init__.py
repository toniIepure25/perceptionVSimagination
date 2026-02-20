"""
fmri2img -- Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI
===============================================================================

A framework for decoding visual content from brain activity (Natural Scenes
Dataset) and investigating cross-domain transfer between visual perception
and mental imagery.

Subpackages
-----------
analysis    Six novel analysis directions (dimensionality, uncertainty, etc.)
data        Data loading, preprocessing, indexing, and imagery datasets
eval        Evaluation metrics (retrieval, image quality, brain alignment)
generation  Diffusion-based image reconstruction from decoded embeddings
io          I/O layer for NSD data on S3 and local filesystems
models      Neural and linear decoders, adapters, and loss functions
reliability Noise-ceiling estimation and voxel reliability
stats       Statistical inference (bootstrap CI, permutation tests)
training    Training loops for ridge, MLP, two-stage, and adapter models
utils       Configuration, logging, CLIP utilities, and manifests
"""

__version__ = "0.2.0"

__all__ = [
    "analysis",
    "data",
    "eval",
    "generation",
    "io",
    "models",
    "reliability",
    "stats",
    "training",
    "utils",
]
