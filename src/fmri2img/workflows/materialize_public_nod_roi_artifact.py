from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import pandas as pd

from fmri2img.roi.groups import normalize_roi_name
from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.materialize_public_nod_payloads import (
    DEFAULT_OPENNEURO_S3_BASE,
    _download_to_path,
    _symlink_target_path,
)
from fmri2img.workflows.prepare_public_nod_roi_materialization_contract import (
    DEFAULT_OUTPUT as DEFAULT_CONTRACT,
    EXPECTED_JOIN_ROWS,
    EXPECTED_SOURCE_ROWS,
)
from fmri2img.workflows.prepare_public_nod_shared_only_join_contract import DEFAULT_OUTPUT as DEFAULT_JOIN


DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_roi_materialized.parquet"

_EARLY_VISUAL_FEATURES = {
    "early_visual_v1": ("V1",),
    "early_visual_v2": ("V2",),
    "early_visual_mt": ("MT",),
}
_VENTRAL_FEATURES = {
    "ventral_visual_faces": (),
    "ventral_visual_places": (),
}
_METACOGNITIVE_FEATURES = {
    "metacognitive_precuneus": ("precuneus",),
    "metacognitive_superiorparietal": ("superiorparietal",),
    "metacognitive_rostralmiddlefrontal": ("rostralmiddlefrontal",),
}
_ROI_VALUE_NAMES = list(_EARLY_VISUAL_FEATURES) + list(_VENTRAL_FEATURES) + list(_METACOGNITIVE_FEATURES)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _dataset_root(repo_root: Path) -> Path:
    return repo_root / "cache" / "public_datasets" / "ds004496"


def _atlas_relative_paths(subject: str) -> dict[str, str]:
    return {
        "early_visual": f"derivatives/ciftify/{subject}/standard_fsLR_surface/{subject}.BA_exvivo.32k_fs_LR.dlabel.nii",
        "ventral_faces": f"derivatives/ciftify/{subject}/results/ses-floc_task-floc/floc-faces.dlabel.nii",
        "ventral_places": f"derivatives/ciftify/{subject}/results/ses-floc_task-floc/floc-places.dlabel.nii",
        "metacognitive": f"derivatives/ciftify/{subject}/standard_fsLR_surface/{subject}.aparc.32k_fs_LR.dlabel.nii",
    }


def _ensure_dataset_payload(dataset_root: Path, relpath: str, base_url: str) -> tuple[Path, bool, int]:
    worktree_path = dataset_root / relpath
    if worktree_path.exists():
        return worktree_path, False, 0
    destination = _symlink_target_path(worktree_path)
    url = f"{base_url.rstrip('/')}/ds004496/{relpath}"
    downloaded_bytes = _download_to_path(url, destination)
    return worktree_path, True, downloaded_bytes


def _load_dlabel(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    img = nib.load(path)
    data = np.asarray(img.get_fdata()).reshape(-1).astype(np.int32)
    axis = img.header.get_axis(0)
    labels: dict[int, str] = {}
    for key, label in axis.label[0].items():
        name = label[0]
        if isinstance(name, bytes):
            name = name.decode()
        labels[int(key)] = str(name)
    return data, labels


def _mask_for_aliases(values: np.ndarray, labels: dict[int, str], aliases: tuple[str, ...]) -> np.ndarray:
    aliases_norm = tuple(normalize_roi_name(alias) for alias in aliases)
    matched = []
    for key, name in labels.items():
        if key == 0:
            continue
        normalized = normalize_roi_name(name)
        if any(alias in normalized for alias in aliases_norm):
            matched.append(key)
    mask = np.isin(values, matched)
    if not bool(mask.any()):
        raise ValueError(f"Could not resolve ROI aliases {aliases} from the available atlas labels.")
    return mask


def _nonzero_mask(values: np.ndarray) -> np.ndarray:
    mask = values != 0
    if not bool(mask.any()):
        raise ValueError("Expected a non-empty ROI mask, but the atlas has no non-zero vertices.")
    return mask


def _feature_masks_for_subject(
    *,
    subject: str,
    dataset_root: Path,
    base_url: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str], int, int]:
    atlas_paths = _atlas_relative_paths(subject)
    downloaded_files = 0
    downloaded_bytes = 0
    resolved_paths: dict[str, str] = {}
    for key, relpath in atlas_paths.items():
        _, downloaded, byte_count = _ensure_dataset_payload(dataset_root, relpath, base_url)
        resolved_paths[key] = relpath
        downloaded_files += int(downloaded)
        downloaded_bytes += byte_count

    early_values, early_labels = _load_dlabel(dataset_root / atlas_paths["early_visual"])
    faces_values, _ = _load_dlabel(dataset_root / atlas_paths["ventral_faces"])
    places_values, _ = _load_dlabel(dataset_root / atlas_paths["ventral_places"])
    meta_values, meta_labels = _load_dlabel(dataset_root / atlas_paths["metacognitive"])

    feature_masks: dict[str, np.ndarray] = {}
    for name, aliases in _EARLY_VISUAL_FEATURES.items():
        feature_masks[name] = _mask_for_aliases(early_values, early_labels, aliases)
    feature_masks["ventral_visual_faces"] = _nonzero_mask(faces_values)
    feature_masks["ventral_visual_places"] = _nonzero_mask(places_values)
    for name, aliases in _METACOGNITIVE_FEATURES.items():
        feature_masks[name] = _mask_for_aliases(meta_values, meta_labels, aliases)

    atlas_sources = {
        "early_visual": atlas_paths["early_visual"],
        "ventral_faces": atlas_paths["ventral_faces"],
        "ventral_places": atlas_paths["ventral_places"],
        "metacognitive": atlas_paths["metacognitive"],
    }
    return feature_masks, atlas_sources, resolved_paths, downloaded_files, downloaded_bytes


def _mean_or_raise(vector: np.ndarray, mask: np.ndarray, feature_name: str) -> float:
    selected = np.asarray(vector[mask], dtype=np.float32)
    if selected.size == 0:
        raise ValueError(f"NOD ROI materialization found an empty selection for feature {feature_name}.")
    return float(selected.mean(dtype=np.float32))


def build_public_nod_roi_materialized(
    contract_path: Path,
    join_contract_path: Path,
    *,
    base_url: str = DEFAULT_OPENNEURO_S3_BASE,
    feature_mask_builder: Callable[..., tuple[dict[str, np.ndarray], dict[str, str], dict[str, str], int, int]]
    | None = None,
) -> tuple[pd.DataFrame, dict]:
    contract_path = contract_path.resolve()
    join_contract_path = join_contract_path.resolve()
    repo_root = contract_path.parents[3]
    dataset_root = _dataset_root(repo_root)

    contract = pd.read_parquet(contract_path)
    join_df = pd.read_parquet(join_contract_path)
    if len(contract) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"NOD ROI materialization requires the fixed {EXPECTED_SOURCE_ROWS}-row contract, "
            f"but {contract_path} exposes {len(contract)} rows."
        )
    contract = contract.sort_values(["subject", "session", "run"]).reset_index(drop=True)
    if len(join_df) != EXPECTED_JOIN_ROWS or not bool(join_df["pair_id"].is_unique):
        raise ValueError(
            f"NOD ROI materialization requires the fixed {EXPECTED_JOIN_ROWS}-row unique-pair join contract."
        )
    join_df = join_df.sort_values(["pair_id"]).reset_index(drop=True)

    mask_builder = feature_mask_builder or _feature_masks_for_subject
    subject_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, str]]] = {}
    downloaded_files = 0
    downloaded_bytes = 0
    rows: list[dict] = []

    for source_row in contract.to_dict(orient="records"):
        subject = str(source_row["subject"])
        if subject not in subject_cache:
            feature_masks, atlas_sources, _, new_files, new_bytes = mask_builder(
                subject=subject,
                dataset_root=dataset_root,
                base_url=base_url,
            )
            subject_cache[subject] = (feature_masks, atlas_sources)
            downloaded_files += new_files
            downloaded_bytes += new_bytes
        else:
            feature_masks, atlas_sources = subject_cache[subject]

        subset = (
            join_df[join_df["adapter_row_id"] == source_row["adapter_row_id"]]
            .sort_values("trial_index")
            .reset_index(drop=True)
        )
        beta_path = dataset_root / source_row["source_ciftify_beta_path"]
        beta_data = np.asarray(nib.load(beta_path).get_fdata(), dtype=np.float32)
        if beta_data.ndim != 2:
            raise ValueError(f"NOD ROI materialization expected a 2D beta matrix at {beta_path}, got {beta_data.shape}.")
        if beta_data.shape[0] != len(subset):
            raise ValueError(
                f"NOD ROI materialization requires beta rows to match the join rows for {source_row['adapter_row_id']}."
            )

        for row_index, join_row in enumerate(subset.to_dict(orient="records")):
            vector = beta_data[row_index]
            early_values = [
                _mean_or_raise(vector, feature_masks[name], name) for name in _EARLY_VISUAL_FEATURES
            ]
            ventral_values = [
                _mean_or_raise(vector, feature_masks[name], name) for name in _VENTRAL_FEATURES
            ]
            meta_values = [
                _mean_or_raise(vector, feature_masks[name], name) for name in _METACOGNITIVE_FEATURES
            ]
            roi_values = np.asarray(early_values + ventral_values + meta_values, dtype=np.float32)
            roi_features = {
                "early_visual": early_values,
                "ventral_visual": ventral_values,
                "metacognitive": meta_values,
            }
            rows.append(
                {
                    "pair_id": int(join_row["pair_id"]),
                    "nsdId": int(join_row["pair_id"]),
                    "nsd_id": int(join_row["pair_id"]),
                    "subject": subject,
                    "session": source_row["session"],
                    "run": int(source_row["run"]),
                    "trial_index": int(join_row["trial_index"]),
                    "condition": "perception",
                    "task": "imagenet",
                    "target_identifier": join_row["target_identifier"],
                    "stimulus_path": join_row["stimulus_path"],
                    "source_beta_row_index": row_index,
                    "source_ciftify_beta_path": source_row["source_ciftify_beta_path"],
                    "source_ciftify_label_path": source_row["source_ciftify_label_path"],
                    "source_events_path": source_row["source_events_path"],
                    "roi_names_json": json.dumps(_ROI_VALUE_NAMES),
                    "roi_values_json": json.dumps(roi_values.tolist()),
                    "roi_features_json": json.dumps(roi_features),
                    "roi_feature_layout_version": "public_nod_imagenet_run10_v1",
                    "early_visual_atlas_path": atlas_sources["early_visual"],
                    "ventral_faces_atlas_path": atlas_sources["ventral_faces"],
                    "ventral_places_atlas_path": atlas_sources["ventral_places"],
                    "metacognitive_atlas_path": atlas_sources["metacognitive"],
                }
            )

    artifact = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    if len(artifact) != EXPECTED_JOIN_ROWS or not bool(artifact["pair_id"].is_unique):
        raise ValueError("NOD ROI materialization did not preserve the fixed 1:1 pair_id alignment.")

    report = {
        "source_roi_contract": str(contract_path),
        "source_join_contract": str(join_contract_path),
        "roi_rows": int(len(artifact)),
        "unique_pair_ids": int(artifact["pair_id"].nunique()),
        "downloaded_atlas_files": int(downloaded_files),
        "downloaded_atlas_bytes": int(downloaded_bytes),
        "downloaded_atlas_gib": round(downloaded_bytes / (1024 ** 3), 6),
        "roi_value_names": _ROI_VALUE_NAMES,
        "roi_feature_dimensions": {
            "early_visual": len(_EARLY_VISUAL_FEATURES),
            "ventral_visual": len(_VENTRAL_FEATURES),
            "metacognitive": len(_METACOGNITIVE_FEATURES),
        },
        "state": {
            "join_ready": True,
            "roi_ready": True,
            "downstream_prep_ready": False,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "dataset-side prepared dataset or loader path that consumes the fixed join contract, ROI artifact, and target cache",
            "checked-in shared-only train/eval config pointing to the fixed NOD prepared dataset and target cache",
        ],
    }
    return artifact, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.materialize_public_nod_roi_artifact")
    parser = argparse.ArgumentParser(
        description="Materialize the real ROI-side artifact for the fixed NOD shared-only slice."
    )
    parser.add_argument("--roi-contract", type=Path, default=_default_path(DEFAULT_CONTRACT))
    parser.add_argument("--join-contract", type=Path, default=_default_path(DEFAULT_JOIN))
    parser.add_argument("--output", type=Path, default=_default_path(DEFAULT_OUTPUT))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--openneuro-s3-base-url", default=DEFAULT_OPENNEURO_S3_BASE)
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact, report = build_public_nod_roi_materialized(
        args.roi_contract.resolve(),
        args.join_contract.resolve(),
        base_url=args.openneuro_s3_base_url,
    )
    artifact.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD ROI artifact: {output_path}")
    print(f"Rows: {len(artifact)}")
    print(f"Unique pair_ids: {report['unique_pair_ids']}")
    print(f"Downloaded atlas files: {report['downloaded_atlas_files']}")
    print(f"ROI ready: {report['state']['roi_ready']}")
    print(f"Downstream prep ready: {report['state']['downstream_prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
