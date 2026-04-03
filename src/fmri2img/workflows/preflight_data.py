from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.roi import discover_roi_masks, inspect_roi_materialization_inputs
from fmri2img.targets import LatentTargetSpec, LatentTargetStore
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config
from fmri2img.workflows.prep_common import config_subject, json_safe, write_report


def _load_index_summary(
    path: str | Path,
    default_condition: str | None = None,
    allowed_conditions: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    target = Path(path)
    if not target.exists():
        return {"path": str(target), "exists": False}, None
    try:
        raw = pd.read_parquet(target)
        normalized = normalize_decoder_index(
            raw,
            default_condition=default_condition,
            allowed_conditions=allowed_conditions,
        )
        paired_groups = (
            normalized.groupby("pair_id")["condition"].nunique().ge(2).sum()
            if "pair_id" in normalized.columns
            else 0
        )
        summary = {
            "path": str(target),
            "exists": True,
            "rows": int(len(normalized)),
            "subjects": sorted(normalized["subject"].astype(str).unique().tolist()) if "subject" in normalized.columns else [],
            "conditions": normalized["condition"].value_counts().to_dict(),
            "unique_nsd_ids": int(normalized["nsdId"].nunique()),
            "paired_groups": int(paired_groups),
            "splits": normalized["split"].value_counts().to_dict() if "split" in normalized.columns else {},
        }
        if "vividness" in normalized.columns:
            summary["vividness_coverage"] = float(normalized["vividness"].notna().mean())
        if "confidence" in normalized.columns:
            summary["confidence_coverage"] = float(normalized["confidence"].notna().mean())
        return summary, normalized
    except Exception as exc:
        return {"path": str(target), "exists": True, "valid": False, "error": str(exc)}, None


def _target_summary(config, candidate_df: pd.DataFrame | None) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    spec = LatentTargetSpec(
        name=config["targets"].get("name", "vit_l14_image_768"),
        dimension=int(config["targets"].get("dimension", 768)),
        embedding_column=config["targets"].get("embedding_column"),
    )
    try:
        store = LatentTargetStore(
            cache_path=config["targets"]["cache_path"],
            spec=spec,
            id_column=config["targets"].get("id_column"),
        )
        summary = store.describe()
        if candidate_df is not None and not candidate_df.empty:
            unique_ids = sorted(candidate_df["nsdId"].dropna().astype(int).unique().tolist())
            covered = sum(store.has(nsd_id) for nsd_id in unique_ids)
            summary["coverage_count"] = int(covered)
            summary["coverage_ratio"] = float(covered / max(1, len(unique_ids)))
            if covered < len(unique_ids):
                problems.append(
                    f"Target cache covers {covered}/{len(unique_ids)} stimuli referenced by the current mixed-condition data."
                )
        return summary, problems
    except Exception as exc:
        problems.append(str(exc))
        return {"path": str(config["targets"]["cache_path"]), "valid": False, "error": str(exc)}, problems


def _current_train_index(config) -> tuple[str | None, pd.DataFrame | None]:
    dataset_cfg = config["dataset"]
    if dataset_cfg.get("mixed_index") and Path(dataset_cfg["mixed_index"]).exists():
        _, normalized = _load_index_summary(dataset_cfg["mixed_index"])
        return str(dataset_cfg["mixed_index"]), normalized
    if dataset_cfg.get("mixed_output_index") and Path(dataset_cfg["mixed_output_index"]).exists():
        _, normalized = _load_index_summary(dataset_cfg["mixed_output_index"])
        return str(dataset_cfg["mixed_output_index"]), normalized
    return None, None


def _roi_summary(config, candidate_df: pd.DataFrame | None) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    blocked: list[str] = []
    roi_cfg = config["roi"]
    fallback_policy = roi_cfg.get("fallback_policy", "error")
    summary: dict[str, Any] = {
        "fallback_policy": fallback_policy,
        "mask_root": roi_cfg.get("mask_root"),
        "mask_patterns": roi_cfg.get("mask_patterns", []),
    }

    if candidate_df is not None and not candidate_df.empty:
        summary["rows"] = int(len(candidate_df))
        features_ready = bool("roi_features_json" in candidate_df.columns and candidate_df["roi_features_json"].notna().all())
        values_ready = bool("roi_values_json" in candidate_df.columns and candidate_df["roi_values_json"].notna().all())
        names_ready = bool(
            ("roi_names_json" in candidate_df.columns and candidate_df["roi_names_json"].notna().all())
            or bool(roi_cfg.get("roi_names"))
        )
        summary["has_roi_features_json"] = features_ready
        summary["has_roi_values_json"] = values_ready
        summary["has_roi_names"] = names_ready
        if values_ready and not names_ready:
            blocked.append("ROI values are present, but ROI names are missing.")
        if features_ready or (values_ready and names_ready):
            summary["ready"] = True
            return summary, warnings, blocked

        volume_inspection = inspect_roi_materialization_inputs(
            index=candidate_df,
            subject=config_subject(config),
            cache_root=config.get("dataset.cache_root"),
        )
        summary["materialization_inputs"] = volume_inspection
        if volume_inspection["volumetric_rows"] <= 0:
            if fallback_policy == "full_feature_vector":
                warnings.append(
                    "Current index does not expose volumetric inputs for ROI pooling. The config can only run in full-feature-vector fallback mode."
                )
            else:
                blocked.append(
                    "Current mixed-condition data does not expose volumetric inputs for ROI pooling, and fallback_policy is not full_feature_vector."
                )

    try:
        masks = discover_roi_masks(
            subject=config_subject(config),
            min_voxels=int(roi_cfg.get("min_voxels", 1)),
            mask_root=roi_cfg.get("mask_root"),
            mask_patterns=roi_cfg.get("mask_patterns"),
            layout_config=roi_cfg.get("layout_config", "configs/data.yaml"),
        )
        summary["mask_count"] = len(masks)
        summary["roi_names"] = [mask.name for mask in masks]
    except Exception as exc:
        summary["mask_error"] = str(exc)
        if candidate_df is None or candidate_df.empty or not summary.get("ready", False):
            if fallback_policy == "full_feature_vector":
                warnings.append(
                    f"ROI masks are unavailable: {exc}. The config can still run only as a smoke/full-vector fallback."
                )
            else:
                blocked.append(f"ROI masks are unavailable: {exc}")
    return summary, warnings, blocked


def _readiness_status(
    *,
    blocked: list[str],
    warnings: list[str],
    roi_summary: dict[str, Any],
    labels: dict[str, Any],
    pairing_count: int,
    paper_pair_threshold: int,
) -> str:
    if blocked:
        return "blocked"
    if roi_summary.get("fallback_policy") == "full_feature_vector":
        return "smoke_only"
    vividness_enabled = labels.get("requested_vividness_head", False)
    if vividness_enabled and labels.get("vividness_coverage", 0.0) <= 0.0 and labels.get("confidence_coverage", 0.0) <= 0.0:
        return "bootstrap_ready"
    if pairing_count >= paper_pair_threshold:
        return "paper_ready"
    return "bootstrap_ready"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight the canonical data path before real decoder training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    config = load_workflow_config(args.config, args.override)
    subject = config_subject(config)
    dataset_cfg = config["dataset"]

    warnings: list[str] = []
    blocked: list[str] = []
    validation_notes: list[str] = []
    paper_pair_threshold = int(config.get("preparation.preflight.paper_pair_threshold", 32))
    try:
        validate_canonical_workflow_config(config)
        validation_notes.append("Canonical workflow config passed static validation.")
    except Exception as exc:
        blocked.append(str(exc))

    perception_path = dataset_cfg.get("perception_index")
    imagery_path = dataset_cfg.get("imagery_index")
    if perception_path:
        perception_summary, perception_df = _load_index_summary(
            perception_path,
            default_condition="perception",
            allowed_conditions=dataset_cfg.get("perception_conditions", ["perception"]),
        )
        if perception_df is None:
            blocked.append(f"Perception index is not ready: {perception_summary.get('error', 'missing file')}")
    else:
        perception_summary, perception_df = {"configured": False}, None
    if imagery_path:
        imagery_summary, imagery_df = _load_index_summary(
            imagery_path,
            default_condition="imagery",
            allowed_conditions=dataset_cfg.get("imagery_conditions", ["imagery"]),
        )
        if imagery_df is None:
            blocked.append(f"Imagery index is not ready: {imagery_summary.get('error', 'missing file')}")
    else:
        imagery_summary, imagery_df = {"configured": False}, None

    mixed_index_path, prepared_mixed_df = _current_train_index(config)
    mixed_summary: dict[str, Any] = {"path": mixed_index_path, "exists": mixed_index_path is not None}
    source_mixed_df = None
    if perception_df is not None and imagery_df is not None:
        source_mixed_df = pd.concat([perception_df, imagery_df], ignore_index=True)
        source_paired = source_mixed_df.groupby("pair_id")["condition"].nunique().ge(2).sum()
        mixed_summary["source_pairing_groups"] = int(source_paired)
        mixed_summary["source_rows"] = int(len(source_mixed_df))
        if source_paired <= 0:
            blocked.append("No shared perception/imagery nsdId pairs were found across the current source indices.")
    elif prepared_mixed_df is None:
        blocked.append(
            "No canonical mixed index is currently available. "
            "Provide dataset.mixed_index or prepare perception and imagery source indices."
        )
    if prepared_mixed_df is not None:
        mixed_summary["prepared_rows"] = int(len(prepared_mixed_df))
        mixed_summary["prepared_pairing_groups"] = int(
            prepared_mixed_df.groupby("pair_id")["condition"].nunique().ge(2).sum()
        )
        prepared_splits = prepared_mixed_df["split"].value_counts().to_dict() if "split" in prepared_mixed_df.columns else {}
        mixed_summary["prepared_splits"] = prepared_splits
        missing_splits = [split for split in ("train", "val", "test") if int(prepared_splits.get(split, 0)) <= 0]
        if missing_splits:
            blocked.append(
                "Canonical mixed index is missing required train/val/test coverage for "
                f"{missing_splits}. Rebuild the mixed index so global pair splits are reassigned jointly."
            )

    current_df = prepared_mixed_df if prepared_mixed_df is not None else source_mixed_df
    target_summary, target_problems = _target_summary(config, current_df)
    blocked.extend(target_problems)

    roi_summary, roi_warnings, roi_blocked = _roi_summary(config, current_df)
    warnings.extend(roi_warnings)
    blocked.extend(roi_blocked)

    labels = {
        "requested_vividness_head": bool(config["model"].get("use_vividness_head", True)),
        "vividness_coverage": 0.0,
        "confidence_coverage": 0.0,
    }
    if current_df is not None and not current_df.empty:
        if "vividness" in current_df.columns:
            labels["vividness_coverage"] = float(current_df["vividness"].notna().mean())
        if "confidence" in current_df.columns:
            labels["confidence_coverage"] = float(current_df["confidence"].notna().mean())
        if labels["requested_vividness_head"] and labels["vividness_coverage"] <= 0 and labels["confidence_coverage"] <= 0:
            warnings.append(
                "The vividness/confidence head is enabled in config, but the current data has no vividness or confidence labels. "
                "Training will disable that head automatically."
            )

    readiness = {
        "status": _readiness_status(
            blocked=blocked,
            warnings=warnings,
            roi_summary=roi_summary,
            labels=labels,
            pairing_count=int(mixed_summary.get("prepared_pairing_groups", mixed_summary.get("source_pairing_groups", 0))),
            paper_pair_threshold=paper_pair_threshold,
        ),
        "blocked_reasons": blocked,
        "warnings": warnings,
        "notes": validation_notes,
        "paper_pair_threshold": paper_pair_threshold,
    }

    report = {
        "subject": subject,
        "config": str(Path(args.config)),
        "perception_index": perception_summary,
        "imagery_index": imagery_summary,
        "mixed_index": mixed_summary,
        "target_cache": target_summary,
        "roi": roi_summary,
        "labels": labels,
        "readiness": readiness,
    }

    output_path = args.output
    if output_path is None:
        output_path = str(Path(config["training"].get("output_dir", "outputs/canonical/train")).parent / "preflight.json")
    write_report(output_path, report)

    print(f"Preflight status: {readiness['status']}")
    print(f"Report: {output_path}")
    if readiness["blocked_reasons"]:
        print("Blocked reasons:")
        for reason in readiness["blocked_reasons"]:
            print(f"- {reason}")
    if readiness["warnings"]:
        print("Warnings:")
        for warning in readiness["warnings"]:
            print(f"- {warning}")
    print(json.dumps(json_safe(report), indent=2))

    if args.fail_on_blocked and readiness["status"] == "blocked":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
