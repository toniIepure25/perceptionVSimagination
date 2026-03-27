"""Canonical ROI grouping utilities for shared/private decoder workflows."""

from .groups import (
    DEFAULT_ROI_GROUPS,
    ROIGroupResolver,
    ROIGroupSpec,
    ResolvedROIGroup,
    normalize_roi_name,
    project_group_features,
    summarize_roi_groups,
)
from .materialize import (
    CanonicalROIPooler,
    ROIMask,
    discover_roi_masks,
    inspect_roi_materialization_inputs,
    materialize_roi_features,
)

__all__ = [
    "DEFAULT_ROI_GROUPS",
    "ROIGroupResolver",
    "ROIGroupSpec",
    "ResolvedROIGroup",
    "normalize_roi_name",
    "project_group_features",
    "summarize_roi_groups",
    "ROIMask",
    "CanonicalROIPooler",
    "discover_roi_masks",
    "materialize_roi_features",
    "inspect_roi_materialization_inputs",
]
