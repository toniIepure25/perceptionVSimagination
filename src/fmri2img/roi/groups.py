from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_ROI_GROUPS: dict[str, list[str]] = {
    "early_visual": ["V1", "V2", "V3"],
    "ventral_visual": ["V4", "LO", "FFA", "PPA"],
    "metacognitive": ["mPFC", "Precuneus", "IPS"],
}


def normalize_roi_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _alias_matches_name(alias: str, roi_name: str) -> bool:
    alias_norm = normalize_roi_name(alias)
    roi_norm = normalize_roi_name(roi_name)
    if alias_norm == roi_norm:
        return True

    tokens = [token for token in re.split(r"[^a-z0-9]+", roi_name.lower()) if token]
    if not tokens:
        tokens = [roi_name.lower()]

    for token in tokens:
        token_norm = normalize_roi_name(token)
        if token_norm == alias_norm:
            return True
        if token_norm.startswith(alias_norm) and len(token_norm) > len(alias_norm):
            next_char = token_norm[len(alias_norm)]
            if not next_char.isdigit():
                return True
    return False


@dataclass(frozen=True)
class ROIGroupSpec:
    groups: Mapping[str, Sequence[str]] = field(default_factory=lambda: DEFAULT_ROI_GROUPS)
    missing_policy: str = "error"
    fallback_policy: str = "error"


@dataclass(frozen=True)
class ResolvedROIGroup:
    name: str
    aliases: tuple[str, ...]
    roi_indices: tuple[int, ...]
    roi_names: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return len(self.roi_indices)


class ROIGroupResolver:
    """Resolve human-readable ROI groups onto loaded ROI-mask names."""

    def __init__(self, spec: ROIGroupSpec | None = None):
        self.spec = spec or ROIGroupSpec()

    def resolve(self, available_roi_names: Iterable[str]) -> dict[str, ResolvedROIGroup]:
        names = list(available_roi_names)
        resolved: dict[str, ResolvedROIGroup] = {}

        for group_name, aliases in self.spec.groups.items():
            indices = []
            matched_names = []
            for alias in aliases:
                for idx, candidate_name in enumerate(names):
                    if _alias_matches_name(alias, candidate_name):
                        if idx not in indices:
                            indices.append(idx)
                            matched_names.append(candidate_name)
            if not indices:
                message = (
                    f"ROI group '{group_name}' could not be resolved from available ROI names: "
                    f"{names}"
                )
                if self.spec.missing_policy == "error":
                    raise ValueError(message)
                logger.warning(message)
                continue
            resolved[group_name] = ResolvedROIGroup(
                name=group_name,
                aliases=tuple(aliases),
                roi_indices=tuple(indices),
                roi_names=tuple(matched_names),
            )

        return resolved


def project_group_features(
    roi_values: np.ndarray,
    roi_names: Sequence[str],
    spec: ROIGroupSpec | None = None,
    fallback_vector: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Slice a full ROI-pooled vector into canonical ROI groups.

    The canonical workflows treat each group as a shallow branch input whose
    dimensionality equals the number of resolved ROI masks in that group.
    """
    resolver = ROIGroupResolver(spec)
    if roi_values.size == 0 or not roi_names:
        if spec is not None and spec.fallback_policy == "full_feature_vector" and fallback_vector is not None:
            return {
                group_name: np.asarray(fallback_vector, dtype=np.float32)
                for group_name in spec.groups
            }
        raise ValueError("Cannot project ROI groups without pooled ROI values and ROI names.")

    resolved = resolver.resolve(roi_names)
    features: dict[str, np.ndarray] = {}
    for group_name, group in resolved.items():
        features[group_name] = np.asarray(roi_values[list(group.roi_indices)], dtype=np.float32)
    return features


def summarize_roi_groups(resolved: Mapping[str, ResolvedROIGroup]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "aliases": list(group.aliases),
            "roi_names": list(group.roi_names),
            "input_dim": group.input_dim,
        }
        for name, group in resolved.items()
    }
