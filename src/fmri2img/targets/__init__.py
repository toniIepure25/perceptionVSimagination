"""Canonical target-space definitions and loaders."""

from .prepare import build_target_cache_from_index, canonicalize_target_cache
from .specs import LatentTargetSpec, LatentTargetStore

__all__ = [
    "LatentTargetSpec",
    "LatentTargetStore",
    "canonicalize_target_cache",
    "build_target_cache_from_index",
]
