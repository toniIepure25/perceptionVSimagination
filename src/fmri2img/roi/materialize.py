from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from fmri2img.io.nsd_layout import NSDLayout
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem

from .groups import ROIGroupResolver, ROIGroupSpec, project_group_features, summarize_roi_groups

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised in lean runtime environments
    nib = None


def _require_nibabel(context: str) -> None:
    if nib is None:
        raise RuntimeError(
            f"nibabel is required to {context}, but it is not installed in the current environment. "
            "Install nibabel or materialize canonical ROI features from numpy volumes instead."
        )


def _normalize_decoder_index(df: pd.DataFrame, default_condition: str | None = None) -> pd.DataFrame:
    from fmri2img.data.canonical import normalize_decoder_index

    return normalize_decoder_index(df, default_condition=default_condition)


@dataclass(frozen=True)
class ROIMask:
    name: str
    path: str
    shape: tuple[int, int, int]
    voxel_count: int
    flat_indices: np.ndarray


def _normalize_subject(subject: str | int) -> str:
    text = str(subject)
    if text.startswith("subj"):
        return text
    return f"subj{int(text):02d}"


def _mask_name_from_path(path: str) -> str:
    name = Path(path).name
    name = re.sub(r"\.nii(\.gz)?$", "", name)
    return name


def _list_matching_files(pattern: str) -> list[str]:
    try:
        if pattern.startswith("s3://"):
            fs = get_s3_filesystem().fs
            return sorted(fs.glob(pattern))
        return sorted(str(Path(path)) for path in glob.glob(pattern))
    except Exception as exc:
        logger.debug("Failed to list ROI mask pattern %s: %s", pattern, exc)
        return []


def _discover_mask_candidates(
    *,
    subject: str,
    mask_root: str | None = None,
    mask_patterns: Sequence[str] | None = None,
    layout_config: str = "configs/data.yaml",
) -> list[str]:
    candidates: list[str] = []
    if mask_patterns:
        for pattern in mask_patterns:
            candidates.extend(_list_matching_files(pattern))
    if mask_root:
        root = Path(mask_root)
        if root.exists():
            candidates.extend(str(path) for path in sorted(root.rglob("*.nii*")) if path.is_file())
    if candidates:
        return sorted(dict.fromkeys(candidates))

    try:
        layout = NSDLayout(layout_config if Path(layout_config).exists() else None)
    except Exception as exc:
        logger.debug("Falling back to default NSDLayout for ROI discovery because %s could not be loaded: %s", layout_config, exc)
        layout = NSDLayout(None)
    for pattern in (
        layout.roi_masks_path(subject, full_url=True),
        layout.fsaverage_roi_masks_path(subject, full_url=True),
        layout.mni_roi_masks_path(subject, full_url=True),
    ):
        matches = _list_matching_files(pattern)
        if matches:
            return sorted(dict.fromkeys(matches))
    return []


def _load_nifti_header(path: str):
    if path.startswith("s3://"):
        loader = NIfTILoader()
        return loader.load(path, validate=True)
    _require_nibabel(f"read local NIfTI data from {path}")
    return nib.load(path)


def _resolve_row_data_path(
    row: pd.Series,
    *,
    index_path: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> str:
    if pd.notna(row.get("fmri_path")):
        raw = str(row["fmri_path"])
        path = Path(raw)
        if path.is_absolute() and path.exists():
            return str(path)
        if cache_root is not None:
            candidate = Path(cache_root) / path
            if candidate.exists():
                return str(candidate)
        if index_path is not None:
            candidate = Path(index_path).resolve().parent / path
            if candidate.exists():
                return str(candidate)
        return raw
    if pd.notna(row.get("beta_path")):
        return str(row["beta_path"])
    raise ValueError("Row does not contain fmri_path or beta_path.")


def _infer_reference_shape(
    df: pd.DataFrame,
    *,
    index_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    reference_volume_path: str | None = None,
) -> tuple[int, int, int] | None:
    candidates: list[str] = []
    if reference_volume_path:
        candidates.append(reference_volume_path)
    for _, row in df.iterrows():
        try:
            candidates.append(_resolve_row_data_path(row, index_path=index_path, cache_root=cache_root))
        except Exception:
            continue
        if len(candidates) >= 8:
            break

    for path in candidates:
        try:
            if path.endswith(".npy"):
                array = np.load(path, mmap_mode="r")
                if array.ndim == 3:
                    return tuple(int(dim) for dim in array.shape)
                if array.ndim == 4:
                    return tuple(int(dim) for dim in array.shape[:3])
                continue
            image = _load_nifti_header(path)
            if len(image.shape) >= 3:
                return tuple(int(dim) for dim in image.shape[:3])
        except Exception as exc:
            logger.debug("Failed to infer reference shape from %s: %s", path, exc)
    return None


def discover_roi_masks(
    *,
    subject: str,
    min_voxels: int = 50,
    mask_root: str | None = None,
    mask_patterns: Sequence[str] | None = None,
    layout_config: str = "configs/data.yaml",
    reference_shape: tuple[int, int, int] | None = None,
) -> list[ROIMask]:
    subject = _normalize_subject(subject)
    candidates = _discover_mask_candidates(
        subject=subject,
        mask_root=mask_root,
        mask_patterns=mask_patterns,
        layout_config=layout_config,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No ROI masks found for {subject}. "
            "Provide roi.mask_root or roi.mask_patterns, or make NSD ROI masks reachable."
        )

    by_shape: dict[tuple[int, int, int], list[ROIMask]] = {}
    for candidate in candidates:
        try:
            image = _load_nifti_header(candidate)
            data = image.get_fdata()
            shape = tuple(int(dim) for dim in data.shape[:3])
            mask = np.asarray(data > 0, dtype=bool)
            voxel_count = int(mask.sum())
            if voxel_count < min_voxels:
                continue
            entry = ROIMask(
                name=_mask_name_from_path(candidate),
                path=str(candidate),
                shape=shape,
                voxel_count=voxel_count,
                flat_indices=np.flatnonzero(mask.reshape(-1)),
            )
            by_shape.setdefault(shape, []).append(entry)
        except Exception as exc:
            logger.warning("Failed to load ROI mask %s: %s", candidate, exc)

    if not by_shape:
        raise RuntimeError(
            f"ROI masks were discovered for {subject}, but none could be loaded with >= {min_voxels} voxels."
        )

    if reference_shape is not None and reference_shape in by_shape:
        selected_shape = reference_shape
    else:
        selected_shape = max(by_shape.items(), key=lambda item: len(item[1]))[0]
        if reference_shape is not None and reference_shape != selected_shape:
            logger.warning(
                "Reference fMRI shape %s did not match any ROI mask set. Using the most common ROI mask shape %s.",
                reference_shape,
                selected_shape,
            )

    masks = by_shape[selected_shape]
    deduped: list[ROIMask] = []
    seen_names: set[str] = set()
    for mask in masks:
        if mask.name in seen_names:
            continue
        deduped.append(mask)
        seen_names.add(mask.name)
    return deduped


class CanonicalROIPooler:
    """Canonical ROI pooler for real-data shared/private workflows."""

    def __init__(self, masks: Sequence[ROIMask]):
        if not masks:
            raise ValueError("CanonicalROIPooler requires at least one ROI mask.")
        shapes = {mask.shape for mask in masks}
        if len(shapes) != 1:
            raise ValueError(f"ROI masks must share one spatial shape, got {sorted(shapes)}")
        self.masks = list(masks)
        self.shape = self.masks[0].shape

    @classmethod
    def from_index(
        cls,
        *,
        index: pd.DataFrame,
        subject: str,
        min_voxels: int = 50,
        mask_root: str | None = None,
        mask_patterns: Sequence[str] | None = None,
        layout_config: str = "configs/data.yaml",
        index_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        reference_volume_path: str | None = None,
    ) -> "CanonicalROIPooler":
        reference_shape = _infer_reference_shape(
            index,
            index_path=index_path,
            cache_root=cache_root,
            reference_volume_path=reference_volume_path,
        )
        masks = discover_roi_masks(
            subject=subject,
            min_voxels=min_voxels,
            mask_root=mask_root,
            mask_patterns=mask_patterns,
            layout_config=layout_config,
            reference_shape=reference_shape,
        )
        return cls(masks)

    @property
    def roi_names(self) -> list[str]:
        return [mask.name for mask in self.masks]

    def pool_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float32)
        if tuple(int(dim) for dim in volume.shape) != self.shape:
            raise ValueError(f"Volume shape {tuple(volume.shape)} does not match ROI mask shape {self.shape}")
        flat = volume.reshape(-1)
        values = np.empty((len(self.masks),), dtype=np.float32)
        for idx, mask in enumerate(self.masks):
            values[idx] = float(np.nanmean(flat[mask.flat_indices]))
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "mask_count": len(self.masks),
            "shape": list(self.shape),
            "roi_names": self.roi_names,
            "masks": [
                {
                    "name": mask.name,
                    "path": mask.path,
                    "voxel_count": mask.voxel_count,
                }
                for mask in self.masks
            ],
        }


def _load_volume(
    row: pd.Series,
    *,
    index_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    cache: dict[str, Any] | None = None,
) -> np.ndarray:
    cache = cache if cache is not None else {}
    path = _resolve_row_data_path(row, index_path=index_path, cache_root=cache_root)
    beta_index = int(row.get("beta_index", 0) or 0)

    if path.endswith(".npy"):
        if path not in cache:
            cache[path] = np.load(path, mmap_mode="r")
        data = cache[path]
        if data.ndim == 1:
            raise ValueError(
                f"Cannot materialize ROI features from 1D numpy vector {path}. "
                "Provide volumetric NIfTI/3D numpy data or precomputed ROI features."
            )
        if data.ndim == 3:
            return np.asarray(data, dtype=np.float32)
        if data.ndim == 4:
            return np.asarray(data[..., beta_index], dtype=np.float32)
        raise ValueError(f"Unsupported numpy fMRI shape {data.shape} for {path}")

    if path not in cache:
        if path.startswith("s3://"):
            loader = NIfTILoader()
            cache[path] = loader.load(path, validate=True)
        else:
            _require_nibabel(f"load local NIfTI volume from {path}")
            cache[path] = nib.load(path)
    image = cache[path]
    dataobj = image.dataobj
    if len(image.shape) == 3:
        return np.asarray(dataobj, dtype=np.float32)
    if len(image.shape) == 4:
        return np.asarray(dataobj[..., beta_index], dtype=np.float32)
    raise ValueError(f"Unsupported NIfTI shape {image.shape} for {path}")


def materialize_roi_features(
    *,
    index: str | Path | pd.DataFrame,
    subject: str,
    output_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    group_spec: ROIGroupSpec | None = None,
    min_voxels: int = 50,
    mask_root: str | None = None,
    mask_patterns: Sequence[str] | None = None,
    layout_config: str = "configs/data.yaml",
    cache_root: str | Path | None = None,
    reference_volume_path: str | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    if isinstance(index, (str, Path)):
        index_path = Path(index)
        df = pd.read_parquet(index_path)
    else:
        index_path = None
        df = index.copy()

    df = _normalize_decoder_index(df)
    subject = _normalize_subject(subject)
    subject_mask = df["subject"] == subject
    if not subject_mask.any():
        raise ValueError(f"Canonical ROI materialization found no rows for subject={subject}.")
    subject_df = df.loc[subject_mask].reset_index(drop=True)

    group_spec = group_spec or ROIGroupSpec()
    pooler = CanonicalROIPooler.from_index(
        index=subject_df,
        subject=subject,
        min_voxels=min_voxels,
        mask_root=mask_root,
        mask_patterns=mask_patterns,
        layout_config=layout_config,
        index_path=index_path,
        cache_root=cache_root,
        reference_volume_path=reference_volume_path,
    )
    resolved_groups = ROIGroupResolver(group_spec).resolve(pooler.roi_names)
    volume_cache: dict[str, Any] = {}

    materialized_rows = 0
    skipped_existing = 0
    roi_names_json = json.dumps(pooler.roi_names)
    for row_idx in range(len(subject_df)):
        row = subject_df.iloc[row_idx]
        if not overwrite_existing and pd.notna(row.get("roi_features_json")) and pd.notna(row.get("roi_values_json")):
            skipped_existing += 1
            if pd.isna(row.get("roi_names_json")):
                subject_df.at[row_idx, "roi_names_json"] = roi_names_json
            continue

        volume = _load_volume(row, index_path=index_path, cache_root=cache_root, cache=volume_cache)
        roi_values = pooler.pool_volume(volume)
        group_features = project_group_features(
            roi_values=roi_values,
            roi_names=pooler.roi_names,
            spec=group_spec,
            fallback_vector=None,
        )
        subject_df.at[row_idx, "roi_values_json"] = json.dumps(roi_values.astype(np.float32).tolist())
        subject_df.at[row_idx, "roi_names_json"] = roi_names_json
        subject_df.at[row_idx, "roi_features_json"] = json.dumps(
            {name: values.astype(np.float32).tolist() for name, values in group_features.items()}
        )
        materialized_rows += 1

    df.loc[subject_mask, subject_df.columns] = subject_df.to_numpy()

    output = Path(output_path) if output_path is not None else index_path
    if output is None:
        raise ValueError("Canonical ROI materialization needs an output path when the index is provided as a DataFrame.")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    summary = {
        "subject": subject,
        "input_index": None if index_path is None else str(index_path),
        "output_index": str(output),
        "rows": int(len(df)),
        "materialized_rows": int(materialized_rows),
        "skipped_existing_rows": int(skipped_existing),
        "pooler": pooler.describe(),
        "resolved_groups": summarize_roi_groups(resolved_groups),
        "group_spec": {
            "groups": {key: list(values) for key, values in group_spec.groups.items()},
            "missing_policy": group_spec.missing_policy,
            "fallback_policy": group_spec.fallback_policy,
        },
    }
    if provenance_path is not None:
        provenance = Path(provenance_path)
        provenance.parent.mkdir(parents=True, exist_ok=True)
        with open(provenance, "w") as handle:
            json.dump(summary, handle, indent=2)
    return summary


def inspect_roi_materialization_inputs(
    *,
    index: str | Path | pd.DataFrame,
    subject: str | None = None,
    cache_root: str | Path | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    if isinstance(index, (str, Path)):
        index_path = Path(index)
        df = pd.read_parquet(index_path)
    else:
        index_path = None
        df = index.copy()
    if subject is not None and "subject" in df.columns:
        df = df[df["subject"] == _normalize_subject(subject)].reset_index(drop=True)
    if df.empty:
        return {"rows": 0, "volumetric_rows": 0, "vector_rows": 0, "issues": ["empty index"]}

    volumetric_rows = 0
    vector_rows = 0
    issues: list[str] = []
    for _, row in df.head(sample_limit).iterrows():
        try:
            path = _resolve_row_data_path(row, index_path=index_path, cache_root=cache_root)
            if path.endswith(".npy"):
                array = np.load(path, mmap_mode="r")
                if array.ndim >= 3:
                    volumetric_rows += 1
                else:
                    vector_rows += 1
            else:
                image = _load_nifti_header(path)
                if len(image.shape) >= 3:
                    volumetric_rows += 1
                else:
                    issues.append(f"unsupported shape {image.shape} for {path}")
        except Exception as exc:
            issues.append(str(exc))
    return {
        "rows": int(len(df)),
        "sampled_rows": int(min(len(df), sample_limit)),
        "volumetric_rows": int(volumetric_rows),
        "vector_rows": int(vector_rows),
        "issues": issues[:10],
    }
