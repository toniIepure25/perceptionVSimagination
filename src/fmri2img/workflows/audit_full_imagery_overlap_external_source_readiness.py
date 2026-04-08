from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_full_imagery_overlap_external_source_readiness")

from fmri2img.data.canonical import normalize_decoder_index  # noqa: E402
from fmri2img.data.nsd_imagery_metadata import assign_splits, parse_all_trials, trials_to_dataframe  # noqa: E402
from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.common import load_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/full_imagery_overlap_shared_only.yaml"
EXPECTED_EXPERIMENT_NAME = "full_imagery_overlap_shared_only"
DEFAULT_EXTERNAL_ROOT = "cache/nsd_imagery_external"
DEFAULT_EXTERNAL_METADATA_ROOT = "cache/nsd_imagery_external/metadata"
DEFAULT_EXTERNAL_BETA_ROOT = "cache/nsd_imagery_external/betas"
DEFAULT_PUBLIC_ROOT = "cache/nsd_imagery_full_all"
DEFAULT_PERCEPTION_FALLBACK_TEMPLATES = (
    "/home/jovyan/work/FMRI2images/data/indices/nsd_index/subject={subject}/index.parquet",
)
REQUIRED_PROVENANCE_FIELDS = (
    "source_dataset_name",
    "source_kind",
    "acquisition_date",
    "subjects",
    "total_size_bytes",
)
OPERATIONAL_BOUNDARY = [
    "this audit only prepares the existing full-overlap shared-only lane to consume a richer external NSD-style paired source without changing the readiness gate",
    "external_source_ready_for_rebuild requires a mounted source, explicit provenance, and measured paired-support gains over the current 5-total / 1-held-out ceiling",
    "this report does not claim a benchmark improvement, evidence-grade validation, production Animus readiness, or training_ready=true on its own",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _resolve_maybe_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "${" in text or "$" in text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_template(template: str, *, subject: str) -> Path:
    candidate = Path(template.format(subject=subject))
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _count_paired_groups(df: pd.DataFrame) -> int:
    if df.empty or "pair_id" not in df.columns or "condition" not in df.columns:
        return 0
    grouped = df.groupby("pair_id")["condition"].agg(lambda values: {str(value) for value in values})
    return int(sum({"perception", "imagery"}.issubset(values) for values in grouped))


def _validate_external_source_readiness_config(config) -> None:
    required_sections = ("experiment", "dataset", "targets", "evaluation", "preparation")
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise KeyError(f"External-source readiness config is missing sections: {missing_sections}")

    target_dim = int(config["targets"].get("dimension", 768))
    if target_dim != 768:
        raise ValueError(
            f"External-source readiness audit requires 768-D ViT-L/14 targets, got dimension={target_dim}."
        )


def _nsd_id_set(df: pd.DataFrame) -> set[int]:
    for column in ("nsdId", "nsd_id"):
        if column in df.columns:
            return {int(value) for value in df[column].dropna().astype(int).unique()}
    return set()


def _resolve_perception_source(subject: str, primary_template: str, fallback_templates: tuple[str, ...]) -> dict[str, Any]:
    candidates = [primary_template, *fallback_templates]
    searched = []
    for index, template in enumerate(candidates):
        path = _resolve_template(template, subject=subject)
        searched.append(str(path))
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        return {
            "path": str(path),
            "exists": True,
            "source_kind": "config_template" if index == 0 else f"fallback_{index}",
            "rows": int(len(df)),
            "unique_nsd_ids": int(len(_nsd_id_set(df))),
            "df": df,
            "searched_paths": searched,
        }
    return {
        "path": None,
        "exists": False,
        "source_kind": None,
        "rows": 0,
        "unique_nsd_ids": 0,
        "df": None,
        "searched_paths": searched,
    }


def _detect_condition(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    if "perception" in lowered_parts:
        return "perception"
    if "imagery" in lowered_parts:
        return "imagery"
    return "imagery"


def _discover_metadata_dir(data_root: Path | None, metadata_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    for base in (metadata_root, data_root):
        if base is None:
            continue
        candidates.extend([base, base / "metadata"])
    for candidate in candidates:
        if candidate.exists() and (
            (candidate / "designmatrixGLMsingle.mat").exists()
            or (candidate / "cue_pair_list.xlsx").exists()
            or any(candidate.glob("*_pair_list.mat"))
        ):
            return candidate.resolve()
    return None


def _discover_beta_path(subject: str, data_root: Path | None, beta_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if beta_root is not None:
        candidates.extend(
            [
                beta_root / subject / "betas_nsdimagery.nii.gz",
                beta_root / subject / "betas_nsdimagery.nii",
            ]
        )
        candidates.extend(sorted((beta_root / subject).glob("**/betas_nsdimagery.nii*")))
    if data_root is not None:
        candidates.extend(
            [
                data_root / "betas" / subject / "betas_nsdimagery.nii.gz",
                data_root / "betas" / subject / "betas_nsdimagery.nii",
                data_root / subject / "betas_nsdimagery.nii.gz",
                data_root / subject / "betas_nsdimagery.nii",
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _load_subject_rooted_trials(subject_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trial_id_counter = 0
    metadata_files = sorted(subject_root.glob("**/metadata.json"))
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text())
        trials_meta = payload.get("trials", payload if isinstance(payload, list) else [])
        if not isinstance(trials_meta, list):
            continue
        beta_dir = metadata_file.parent
        run_match = re.search(r"(?:run|session)_?(\d+)", beta_dir.name, re.IGNORECASE)
        run_id = int(run_match.group(1)) if run_match else -1
        condition = _detect_condition(metadata_file)
        directory_beta_files = sorted(list(beta_dir.glob("*.npy")) + list(beta_dir.glob("*.nii.gz")))
        for item_index, trial_meta in enumerate(trials_meta):
            trial_identifier = str(trial_meta.get("trial_id", f"trial_{trial_id_counter:05d}"))
            matching_beta = None
            for beta_file in directory_beta_files:
                if trial_identifier in beta_file.name:
                    matching_beta = beta_file
                    break
            if matching_beta is None and directory_beta_files:
                matching_beta = directory_beta_files[min(item_index, len(directory_beta_files) - 1)]
            if matching_beta is None:
                continue
            nsd_id = trial_meta.get("nsd_id", trial_meta.get("nsdId"))
            pair_id = trial_meta.get("pair_id", nsd_id)
            rows.append(
                {
                    "trial_id": trial_id_counter,
                    "subject": str(subject_root.name),
                    "condition": condition,
                    "stimulus_type": trial_meta.get("stimulus_type", "unknown"),
                    "task_type": condition,
                    "run_id": run_id,
                    "beta_index": int(trial_meta.get("beta_index", 0) or 0),
                    "fmri_path": str(matching_beta.resolve()),
                    "image_path": trial_meta.get("image_path"),
                    "text_prompt": trial_meta.get("text_prompt"),
                    "nsdId": None if nsd_id is None else int(nsd_id),
                    "nsd_id": None if nsd_id is None else int(nsd_id),
                    "pair_id": None if pair_id is None else int(pair_id),
                    "stimulus_set": trial_meta.get("stimulus_set"),
                    "shared_id": trial_meta.get("shared_id"),
                    "repeat_index": int(trial_meta.get("repeat_index", 0) or 0),
                    "meta_json": json.dumps(trial_meta),
                    "split": None,
                }
            )
            trial_id_counter += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return assign_splits(df)


def _load_external_imagery_df_for_subject(
    *,
    subject: str,
    data_root: Path | None,
    metadata_root: Path | None,
    beta_root: Path | None,
    imagery_conditions: list[str],
    stimulus_sets: list[str],
    require_nsd_id: bool,
) -> dict[str, Any]:
    subject_root = None if data_root is None else (data_root / subject).resolve()
    metadata_dir = _discover_metadata_dir(data_root, metadata_root)
    beta_path = _discover_beta_path(subject, data_root, beta_root)
    if metadata_dir is not None and beta_path is not None:
        trials = parse_all_trials(metadata_dir)
        raw_df = assign_splits(trials_to_dataframe(trials, subject=subject, beta_path=str(beta_path.resolve())))
        layout_kind = "split_metadata_beta"
    elif subject_root is not None and subject_root.exists():
        raw_df = _load_subject_rooted_trials(subject_root)
        layout_kind = "subject_rooted"
    else:
        return {
            "layout_kind": None,
            "mounted": False,
            "subject_root": None if subject_root is None else str(subject_root),
            "metadata_dir": None if metadata_dir is None else str(metadata_dir),
            "beta_path": None if beta_path is None else str(beta_path),
            "raw_rows": 0,
            "rows_after_filter": 0,
            "unique_nsd_ids_after_filter": 0,
            "filter_contract_ok": False,
            "missing_filter_support": [],
            "df": pd.DataFrame(),
        }

    filter_contract_ok = True
    missing_filter_support: list[str] = []
    filtered = raw_df.copy()
    if stimulus_sets:
        if "stimulus_set" not in filtered.columns:
            filter_contract_ok = False
            missing_filter_support.append("stimulus_set")
            filtered = filtered.iloc[0:0].copy()
        else:
            filtered = filtered[filtered["stimulus_set"].astype(str).isin({str(value) for value in stimulus_sets})]
    if require_nsd_id:
        if "nsdId" not in filtered.columns and "nsd_id" not in filtered.columns:
            filter_contract_ok = False
            missing_filter_support.append("nsdId")
            filtered = filtered.iloc[0:0].copy()
        else:
            nsd_mask = pd.Series(False, index=filtered.index)
            for column in ("nsdId", "nsd_id"):
                if column in filtered.columns:
                    nsd_mask = nsd_mask | filtered[column].notna()
            filtered = filtered[nsd_mask]

    if not filtered.empty:
        filtered = normalize_decoder_index(
            filtered.reset_index(drop=True),
            default_condition="imagery",
            allowed_conditions=imagery_conditions,
        )
    else:
        filtered = filtered.reset_index(drop=True)

    return {
        "layout_kind": layout_kind,
        "mounted": True,
        "subject_root": None if subject_root is None else str(subject_root),
        "metadata_dir": None if metadata_dir is None else str(metadata_dir),
        "beta_path": None if beta_path is None else str(beta_path),
        "raw_rows": int(len(raw_df)),
        "rows_after_filter": int(len(filtered)),
        "unique_nsd_ids_after_filter": int(len(_nsd_id_set(filtered))),
        "filter_contract_ok": filter_contract_ok,
        "missing_filter_support": missing_filter_support,
        "df": filtered,
    }


def _normalize_subject_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _extract_provenance_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, "", []):
            return payload[key]
    return None


def _summarize_provenance(contract_root: Path | None) -> dict[str, Any]:
    if contract_root is None:
        return {
            "manifest_path": None,
            "manifest_kind": None,
            "manifest_present": False,
            "required_fields_present": False,
            "missing_fields": list(REQUIRED_PROVENANCE_FIELDS),
            "subject_list": [],
            "source_dataset_name": None,
            "source_kind": None,
            "acquisition_date": None,
            "total_size_bytes": None,
        }

    candidate_files = [
        ("acquisition_provenance", contract_root / "acquisition_provenance.json"),
        ("download_manifest", contract_root / "download_manifest.json"),
    ]
    for manifest_kind, path in candidate_files:
        if not path.exists():
            continue
        payload = load_json(path)
        subject_list = _normalize_subject_list(_extract_provenance_value(payload, "subjects", "subject_list"))
        normalized = {
            "source_dataset_name": _extract_provenance_value(
                payload,
                "source_dataset_name",
                "dataset_name",
                "dataset",
                "dataset_id",
            ),
            "source_kind": _extract_provenance_value(payload, "source_kind", "source_type", "provenance_kind"),
            "acquisition_date": _extract_provenance_value(payload, "acquisition_date", "timestamp", "acquired_at"),
            "subjects": subject_list,
            "total_size_bytes": _extract_provenance_value(payload, "total_size_bytes", "total_bytes"),
        }
        missing_fields = [field for field in REQUIRED_PROVENANCE_FIELDS if normalized.get(field) in (None, "", [])]
        return {
            "manifest_path": str(path.resolve()),
            "manifest_kind": manifest_kind,
            "manifest_present": True,
            "required_fields_present": not missing_fields,
            "missing_fields": missing_fields,
            "subject_list": subject_list,
            "source_dataset_name": normalized["source_dataset_name"],
            "source_kind": normalized["source_kind"],
            "acquisition_date": normalized["acquisition_date"],
            "total_size_bytes": normalized["total_size_bytes"],
        }
    return {
        "manifest_path": None,
        "manifest_kind": None,
        "manifest_present": False,
        "required_fields_present": False,
        "missing_fields": list(REQUIRED_PROVENANCE_FIELDS),
        "subject_list": [],
        "source_dataset_name": None,
        "source_kind": None,
        "acquisition_date": None,
        "total_size_bytes": None,
    }


def _resolve_external_contract_paths(
    config,
    *,
    external_root: str | None,
    external_metadata_root: str | None,
    external_beta_root: str | None,
) -> dict[str, Any]:
    default_root = _default_path(DEFAULT_EXTERNAL_ROOT).resolve()
    default_metadata_root = _default_path(DEFAULT_EXTERNAL_METADATA_ROOT).resolve()
    default_beta_root = _default_path(DEFAULT_EXTERNAL_BETA_ROOT).resolve()
    current_public_root = _default_path(DEFAULT_PUBLIC_ROOT).resolve()

    configured = {
        "data_root": _resolve_maybe_path(config.get("preparation.imagery.data_root")),
        "metadata_root": _resolve_maybe_path(config.get("preparation.imagery.metadata_root")),
        "beta_root": _resolve_maybe_path(config.get("preparation.imagery.beta_root")),
    }
    explicit = {
        "data_root": _resolve_maybe_path(external_root),
        "metadata_root": _resolve_maybe_path(external_metadata_root),
        "beta_root": _resolve_maybe_path(external_beta_root),
    }

    selected_data_root = explicit["data_root"] or configured["data_root"] or default_root
    selected_metadata_root = explicit["metadata_root"] or configured["metadata_root"] or default_metadata_root
    selected_beta_root = explicit["beta_root"] or configured["beta_root"] or default_beta_root
    contract_root = selected_data_root
    if contract_root is None and selected_metadata_root and selected_beta_root and selected_metadata_root.parent == selected_beta_root.parent:
        contract_root = selected_metadata_root.parent

    configured_source_matches_current_public_root = any(
        path is not None and path == current_public_root
        for path in configured.values()
    ) or any(
        path is not None and current_public_root in path.parents
        for path in configured.values()
    )

    return {
        "env_vars": {
            "NSD_IMAGERY_ROOT": os.environ.get("NSD_IMAGERY_ROOT"),
            "NSD_IMAGERY_METADATA_ROOT": os.environ.get("NSD_IMAGERY_METADATA_ROOT"),
            "NSD_IMAGERY_BETA_ROOT": os.environ.get("NSD_IMAGERY_BETA_ROOT"),
        },
        "configured_paths": {key: None if value is None else str(value) for key, value in configured.items()},
        "explicit_paths": {key: None if value is None else str(value) for key, value in explicit.items()},
        "expected_canonical_external_paths": {
            "data_root": str(default_root),
            "metadata_root": str(default_metadata_root),
            "beta_root": str(default_beta_root),
        },
        "selected_paths": {
            "data_root": None if selected_data_root is None else str(selected_data_root),
            "metadata_root": None if selected_metadata_root is None else str(selected_metadata_root),
            "beta_root": None if selected_beta_root is None else str(selected_beta_root),
        },
        "selected_data_root": selected_data_root,
        "selected_metadata_root": selected_metadata_root,
        "selected_beta_root": selected_beta_root,
        "contract_root": contract_root,
        "current_public_root": str(current_public_root),
        "configured_source_matches_current_public_root": configured_source_matches_current_public_root,
    }


def build_full_imagery_overlap_external_source_readiness_audit(
    config,
    *,
    config_path: str | Path,
    external_root: str | None = None,
    external_metadata_root: str | None = None,
    external_beta_root: str | None = None,
    fallback_perception_templates: tuple[str, ...] = DEFAULT_PERCEPTION_FALLBACK_TEMPLATES,
) -> dict[str, Any]:
    _validate_external_source_readiness_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Full-overlap external-source readiness audit requires configs/canonical/full_imagery_overlap_shared_only.yaml."
        )

    eval_dir = Path(config["evaluation"]["output_dir"]).resolve()
    readiness_path = eval_dir / "readiness_audit.json"
    data_expansion_path = eval_dir / "data_expansion_audit.json"
    if not readiness_path.exists():
        raise FileNotFoundError(
            f"Full-overlap external-source readiness audit requires the current readiness artifact at {readiness_path}."
        )
    if not data_expansion_path.exists():
        raise FileNotFoundError(
            f"Full-overlap external-source readiness audit requires the current data-expansion artifact at {data_expansion_path}."
        )

    readiness = load_json(readiness_path)
    data_expansion = load_json(data_expansion_path)
    current_support = dict(readiness.get("heldout_support", {}))
    if not current_support:
        raise ValueError("Current readiness artifact is missing the heldout_support section.")

    primary_perception_template = config.get("preparation.overlap.perception_index_template")
    subjects = [str(subject) for subject in config.get("preparation.overlap.subjects", [])]
    imagery_conditions = list(config.get("preparation.imagery.conditions", config["dataset"].get("imagery_conditions", ["imagery"])))
    perception_conditions = list(config["dataset"].get("perception_conditions", ["perception"]))
    stimulus_sets = list(config.get("preparation.imagery.stimulus_sets", []))
    require_nsd_id = bool(config.get("preparation.imagery.require_nsd_id", True))
    if not primary_perception_template or not subjects:
        raise ValueError(
            "Full-overlap external-source readiness audit requires preparation.overlap.subjects and perception_index_template."
        )

    contract_paths = _resolve_external_contract_paths(
        config,
        external_root=external_root,
        external_metadata_root=external_metadata_root,
        external_beta_root=external_beta_root,
    )
    provenance = _summarize_provenance(contract_paths["contract_root"])

    subject_reports = []
    combined_rows: list[pd.DataFrame] = []
    subjects_with_detected_external_data: list[str] = []
    subjects_with_overlap: list[str] = []
    for subject in subjects:
        perception = _resolve_perception_source(subject, primary_perception_template, fallback_perception_templates)
        external = _load_external_imagery_df_for_subject(
            subject=subject,
            data_root=contract_paths["selected_data_root"],
            metadata_root=contract_paths["selected_metadata_root"],
            beta_root=contract_paths["selected_beta_root"],
            imagery_conditions=imagery_conditions,
            stimulus_sets=stimulus_sets,
            require_nsd_id=require_nsd_id,
        )
        if external["mounted"]:
            subjects_with_detected_external_data.append(subject)

        overlap_ids = []
        overlap_pair_group_count = 0
        overlap_split_pair_group_counts = {"train": 0, "val": 0, "test": 0}
        if perception["exists"] and external["rows_after_filter"] > 0 and external["filter_contract_ok"]:
            perception_df = normalize_decoder_index(
                perception["df"],
                default_condition="perception",
                allowed_conditions=perception_conditions,
            )
            imagery_df = external["df"]
            overlap_ids = sorted(perception_df["nsdId"].astype(int).unique().tolist() and list(_nsd_id_set(perception_df) & _nsd_id_set(imagery_df)))
            if overlap_ids:
                subjects_with_overlap.append(subject)
                overlap_perception = perception_df[perception_df["nsdId"].isin(overlap_ids)].reset_index(drop=True)
                overlap_imagery = imagery_df[imagery_df["nsdId"].isin(overlap_ids)].reset_index(drop=True)
                subject_mixed = normalize_decoder_index(
                    pd.concat([overlap_perception, overlap_imagery], ignore_index=True),
                    allowed_conditions=perception_conditions + imagery_conditions,
                )
                overlap_pair_group_count = _count_paired_groups(subject_mixed)
                overlap_split_pair_group_counts = {
                    split: _count_paired_groups(subject_mixed[subject_mixed["split"] == split].reset_index(drop=True))
                    for split in ("train", "val", "test")
                }
                combined_rows.append(subject_mixed)

        subject_reports.append(
            {
                "subject": subject,
                "perception_source": {
                    "path": perception["path"],
                    "exists": perception["exists"],
                    "source_kind": perception["source_kind"],
                    "rows": perception["rows"],
                    "unique_nsd_ids": perception["unique_nsd_ids"],
                    "searched_paths": perception["searched_paths"],
                },
                "external_source": {
                    "layout_kind": external["layout_kind"],
                    "mounted": external["mounted"],
                    "subject_root": external["subject_root"],
                    "metadata_dir": external["metadata_dir"],
                    "beta_path": external["beta_path"],
                    "raw_rows": external["raw_rows"],
                    "rows_after_filter": external["rows_after_filter"],
                    "unique_nsd_ids_after_filter": external["unique_nsd_ids_after_filter"],
                    "filter_contract_ok": external["filter_contract_ok"],
                    "missing_filter_support": external["missing_filter_support"],
                },
                "overlap_nsd_ids": overlap_ids,
                "overlap_pair_group_count": overlap_pair_group_count,
                "overlap_split_pair_group_counts": overlap_split_pair_group_counts,
            }
        )

    if combined_rows:
        combined_external_mixed = normalize_decoder_index(pd.concat(combined_rows, ignore_index=True))
        combined_overlap_ids = sorted(_nsd_id_set(combined_external_mixed))
        external_pair_group_count = _count_paired_groups(combined_external_mixed)
        external_split_pair_group_counts = {
            split: _count_paired_groups(combined_external_mixed[combined_external_mixed["split"] == split].reset_index(drop=True))
            for split in ("train", "val", "test")
        }
    else:
        combined_external_mixed = pd.DataFrame()
        combined_overlap_ids = []
        external_pair_group_count = 0
        external_split_pair_group_counts = {"train": 0, "val": 0, "test": 0}

    current_total_pairs = int(current_support.get("dataset_pair_group_count", 0) or 0)
    current_heldout_pairs = int(current_support.get("heldout_pair_count_from_metrics", 0) or 0)
    additional_total_pairs = max(0, external_pair_group_count - current_total_pairs)
    additional_heldout_pairs = max(0, external_split_pair_group_counts["test"] - current_heldout_pairs)

    external_source_mounted = bool(subjects_with_detected_external_data)
    external_source_not_mounted = not external_source_mounted
    external_source_preserves_contract = all(
        not item["external_source"]["mounted"] or item["external_source"]["filter_contract_ok"]
        for item in subject_reports
    )
    potential_support_exceeds_current_ceiling = (
        external_pair_group_count > current_total_pairs and external_split_pair_group_counts["test"] > current_heldout_pairs
    )
    external_source_ready_for_rebuild = bool(
        external_source_mounted
        and provenance["required_fields_present"]
        and external_source_preserves_contract
        and potential_support_exceeds_current_ceiling
    )

    blocked_reasons = []
    if not bool(data_expansion.get("state", {}).get("data_ceiling_confirmed")):
        blocked_reasons.append(
            "current full-overlap lane has not yet been confirmed as ceiling-blocked; external-source readiness should follow the existing ceiling proof"
        )
    if contract_paths["configured_source_matches_current_public_root"]:
        blocked_reasons.append(
            "configured imagery roots still point at the exhausted current public source rather than a richer external NSD-style mount"
        )
    if external_source_not_mounted:
        blocked_reasons.append(
            "no richer external NSD-style source is mounted at the canonical external layout or via the configured imagery env roots"
        )
    if external_source_mounted and not provenance["manifest_present"]:
        blocked_reasons.append(
            "mounted external source is missing explicit acquisition provenance or download manifest at the contract root"
        )
    if external_source_mounted and provenance["manifest_present"] and not provenance["required_fields_present"]:
        blocked_reasons.append(
            "mounted external source provenance is incomplete for rebuild auditing: "
            + ", ".join(provenance["missing_fields"])
        )
    if external_source_mounted and not external_source_preserves_contract:
        blocked_reasons.append(
            "mounted external source does not preserve the current lane filter contract cleanly for all detected subjects"
        )
    if external_source_mounted and not potential_support_exceeds_current_ceiling:
        blocked_reasons.append(
            "mounted external source does not exceed the current full-overlap ceiling of 5 total paired groups and 1 held-out paired group"
        )

    current_public_exhausted = bool(data_expansion.get("state", {}).get("data_ceiling_confirmed"))
    if current_public_exhausted and external_source_not_mounted:
        next_honest_move = "mount_richer_external_nsd_source"
    elif external_source_mounted and not provenance["required_fields_present"]:
        next_honest_move = "record_external_source_provenance"
    elif external_source_ready_for_rebuild:
        next_honest_move = "rebuild_full_overlap_with_external_source"
    else:
        next_honest_move = "resolve_external_source_contract_gaps"

    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {
            "current_readiness_audit": str(readiness_path),
            "current_data_expansion_audit": str(data_expansion_path),
            "current_mixed_index": str(current_support.get("mixed_index")),
            "current_public_root": contract_paths["current_public_root"],
            "external_contract_root": None if contract_paths["contract_root"] is None else str(contract_paths["contract_root"]),
            "external_provenance_manifest": provenance["manifest_path"],
        },
        "current_main_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "readiness_state": dict(readiness.get("state", {})),
            "heldout_support": current_support,
        },
        "external_source_contract": {
            "env_vars": contract_paths["env_vars"],
            "configured_paths": contract_paths["configured_paths"],
            "explicit_paths": contract_paths["explicit_paths"],
            "expected_canonical_external_paths": contract_paths["expected_canonical_external_paths"],
            "selected_paths": contract_paths["selected_paths"],
            "configured_source_matches_current_public_root": contract_paths["configured_source_matches_current_public_root"],
            "provenance": provenance,
        },
        "external_source_inventory": {
            "subjects_considered": subjects,
            "perception_primary_template": primary_perception_template,
            "perception_fallback_templates": list(fallback_perception_templates),
            "subjects_with_detected_external_data": subjects_with_detected_external_data,
            "subjects_with_overlap": subjects_with_overlap,
            "subjects": subject_reports,
        },
        "overlap_potential": {
            "combined_overlap_nsd_ids_estimate": combined_overlap_ids,
            "external_pair_group_count_estimate": external_pair_group_count,
            "external_split_pair_group_counts_estimate": external_split_pair_group_counts,
            "current_pair_group_count": current_total_pairs,
            "current_heldout_pair_group_count": current_heldout_pairs,
            "additional_pair_groups_vs_current_ceiling": additional_total_pairs,
            "additional_heldout_pair_groups_vs_current_ceiling": additional_heldout_pairs,
            "potential_support_exceeds_current_ceiling": potential_support_exceeds_current_ceiling,
        },
        "conclusion": {
            "external_source_ready_for_rebuild": external_source_ready_for_rebuild,
            "external_source_not_mounted": external_source_not_mounted,
            "current_public_source_exhausted": current_public_exhausted,
            "next_honest_move": next_honest_move,
        },
        "state": {
            "operational_ready": bool(readiness.get("state", {}).get("operational_ready")),
            "downstream_contract_ready": bool(readiness.get("state", {}).get("downstream_contract_ready")),
            "evidence_ready_candidate": bool(readiness.get("state", {}).get("evidence_ready_candidate")),
            "training_ready": bool(readiness.get("state", {}).get("training_ready")),
            "external_source_not_mounted": external_source_not_mounted,
            "external_source_mounted": external_source_mounted,
            "external_source_ready_for_rebuild": external_source_ready_for_rebuild,
            "provenance_recorded": bool(provenance["required_fields_present"]),
            "external_source_preserves_contract": external_source_preserves_contract,
            "potential_support_exceeds_current_ceiling": potential_support_exceeds_current_ceiling,
            "current_public_source_exhausted": current_public_exhausted,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def _blocked_report(config_path: str | Path, message: str) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {},
        "current_main_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "readiness_state": {},
            "heldout_support": {},
        },
        "external_source_contract": {
            "env_vars": {},
            "configured_paths": {},
            "explicit_paths": {},
            "expected_canonical_external_paths": {
                "data_root": str(_default_path(DEFAULT_EXTERNAL_ROOT).resolve()),
                "metadata_root": str(_default_path(DEFAULT_EXTERNAL_METADATA_ROOT).resolve()),
                "beta_root": str(_default_path(DEFAULT_EXTERNAL_BETA_ROOT).resolve()),
            },
            "selected_paths": {},
            "configured_source_matches_current_public_root": False,
            "provenance": {
                "manifest_present": False,
                "required_fields_present": False,
                "missing_fields": list(REQUIRED_PROVENANCE_FIELDS),
            },
        },
        "external_source_inventory": {
            "subjects_considered": [],
            "perception_primary_template": None,
            "perception_fallback_templates": [],
            "subjects_with_detected_external_data": [],
            "subjects_with_overlap": [],
            "subjects": [],
        },
        "overlap_potential": {
            "combined_overlap_nsd_ids_estimate": [],
            "external_pair_group_count_estimate": 0,
            "external_split_pair_group_counts_estimate": {"train": 0, "val": 0, "test": 0},
            "current_pair_group_count": 0,
            "current_heldout_pair_group_count": 0,
            "additional_pair_groups_vs_current_ceiling": 0,
            "additional_heldout_pair_groups_vs_current_ceiling": 0,
            "potential_support_exceeds_current_ceiling": False,
        },
        "conclusion": {
            "external_source_ready_for_rebuild": False,
            "external_source_not_mounted": False,
            "current_public_source_exhausted": False,
            "next_honest_move": "resolve_blocked_audit",
        },
        "state": {
            "operational_ready": False,
            "downstream_contract_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "external_source_not_mounted": False,
            "external_source_mounted": False,
            "external_source_ready_for_rebuild": False,
            "provenance_recorded": False,
            "external_source_preserves_contract": False,
            "potential_support_exceeds_current_ceiling": False,
            "current_public_source_exhausted": False,
        },
        "blocked_reasons": [message],
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether a richer external NSD-style imagery source is mounted and ready to rebuild the full-overlap shared-only lane."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--external-metadata-root", default=None)
    parser.add_argument("--external-beta-root", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_full_imagery_overlap_external_source_readiness_audit(
            config,
            config_path=args.config,
            external_root=args.external_root,
            external_metadata_root=args.external_metadata_root,
            external_beta_root=args.external_beta_root,
        )
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/full_imagery_overlap_shared_only/external_source_readiness_audit.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        return 1 if args.fail_on_blocked else 0

    output_path = args.output or (
        Path(config["evaluation"]["output_dir"]).resolve() / "external_source_readiness_audit.json"
    )
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    if args.fail_on_blocked:
        return 0 if report["state"]["external_source_ready_for_rebuild"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
