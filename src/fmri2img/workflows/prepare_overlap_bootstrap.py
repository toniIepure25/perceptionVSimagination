from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmri2img.data.canonical import build_mixed_condition_index, normalize_decoder_index
from fmri2img.roi import ROIGroupSpec, materialize_roi_features
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import json_safe, write_report


def _format_template(template: str | None, *, subject: str) -> str | None:
    if template is None:
        return None
    return template.format(subject=subject)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the canonical multi-subject overlap bootstrap dataset and ROI-ready mixed index."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    prep_cfg = config.get("preparation.overlap", {})
    if not prep_cfg:
        raise KeyError("prepare_overlap_bootstrap requires a preparation.overlap section in the config.")

    subjects = [str(value) for value in prep_cfg.get("subjects", [])]
    if not subjects:
        raise ValueError("prepare_overlap_bootstrap requires preparation.overlap.subjects.")

    perception_template = prep_cfg.get("perception_index_template")
    imagery_template = prep_cfg.get("imagery_index_template")
    if not perception_template or not imagery_template:
        raise KeyError(
            "prepare_overlap_bootstrap requires preparation.overlap.perception_index_template and "
            "preparation.overlap.imagery_index_template."
        )

    output_root = Path(prep_cfg.get("output_root", "outputs/canonical/prepared/overlap_bootstrap"))
    output_root.mkdir(parents=True, exist_ok=True)
    combined_output = Path(config["dataset"].get("mixed_index", output_root / "multisubj_overlap_mixed_with_roi.parquet"))
    report_path = Path(prep_cfg.get("report_path", output_root / "report.json"))
    overlap_ids_path = Path(prep_cfg.get("overlap_ids_path", output_root / "overlap_nsd_ids.json"))
    materialize_roi = bool(prep_cfg.get("materialize_roi", True))
    include_subjects_without_pairs = bool(prep_cfg.get("include_subjects_without_pairs", False))

    per_subject_indices: list[pd.DataFrame] = []
    subject_reports: list[dict] = []
    combined_overlap_ids: set[int] = set()
    allowed_perception = config["dataset"].get("perception_conditions", ["perception"])
    allowed_imagery = config["dataset"].get("imagery_conditions", ["imagery"])
    roi_cfg = config["roi"]

    for subject in subjects:
        perception_path = Path(_format_template(perception_template, subject=subject))
        imagery_path = Path(_format_template(imagery_template, subject=subject))
        if not perception_path.exists():
            raise FileNotFoundError(
                f"Perception index for {subject} was not found at {perception_path}. "
                "Prepare or normalize that subject before assembling the overlap bootstrap."
            )
        if not imagery_path.exists():
            raise FileNotFoundError(
                f"Imagery index for {subject} was not found at {imagery_path}. "
                "Run prepare_imagery_index for that subject before assembling the overlap bootstrap."
            )

        perception_df = normalize_decoder_index(
            pd.read_parquet(perception_path),
            default_condition="perception",
            allowed_conditions=allowed_perception,
        )
        imagery_df = normalize_decoder_index(
            pd.read_parquet(imagery_path),
            default_condition="imagery",
            allowed_conditions=allowed_imagery,
        )
        perception_df = perception_df[perception_df["subject"].astype(str) == subject].reset_index(drop=True)
        imagery_df = imagery_df[imagery_df["subject"].astype(str) == subject].reset_index(drop=True)

        overlap_ids = sorted(set(perception_df["nsdId"].tolist()) & set(imagery_df["nsdId"].tolist()))
        subject_report = {
            "subject": subject,
            "perception_index": str(perception_path),
            "imagery_index": str(imagery_path),
            "perception_rows": int(len(perception_df)),
            "imagery_rows": int(len(imagery_df)),
            "overlap_nsd_ids": [int(value) for value in overlap_ids],
            "overlap_count": int(len(overlap_ids)),
        }
        subject_dir = output_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        if not overlap_ids:
            subject_report["status"] = "skipped_no_overlap"
            subject_reports.append(subject_report)
            if include_subjects_without_pairs:
                continue
            continue

        combined_overlap_ids.update(int(value) for value in overlap_ids)
        perception_overlap = perception_df[perception_df["nsdId"].isin(overlap_ids)].reset_index(drop=True)
        imagery_overlap = imagery_df[imagery_df["nsdId"].isin(overlap_ids)].reset_index(drop=True)
        perception_overlap_path = subject_dir / "perception_overlap.parquet"
        imagery_overlap_path = subject_dir / "imagery_overlap.parquet"
        if args.overwrite_existing or not perception_overlap_path.exists():
            perception_overlap.to_parquet(perception_overlap_path, index=False)
        if args.overwrite_existing or not imagery_overlap_path.exists():
            imagery_overlap.to_parquet(imagery_overlap_path, index=False)

        mixed_path = subject_dir / "mixed.parquet"
        build_mixed_condition_index(
            perception_index=perception_overlap_path,
            imagery_index=imagery_overlap_path,
            output_path=mixed_path,
            subject=subject,
            perception_conditions=allowed_perception,
            imagery_conditions=allowed_imagery,
        )

        final_subject_path = mixed_path
        roi_summary = None
        if materialize_roi:
            final_subject_path = subject_dir / "mixed_with_roi.parquet"
            mask_root = _format_template(prep_cfg.get("mask_root_template"), subject=subject) or roi_cfg.get("mask_root")
            reference_volume_path = _format_template(prep_cfg.get("reference_volume_path_template"), subject=subject) or roi_cfg.get(
                "reference_volume_path"
            )
            roi_summary = materialize_roi_features(
                index=mixed_path,
                subject=subject,
                output_path=final_subject_path,
                provenance_path=subject_dir / "roi_summary.json",
                group_spec=ROIGroupSpec(
                    groups=roi_cfg.get("groups", {}),
                    missing_policy=roi_cfg.get("missing_policy", "error"),
                    fallback_policy=roi_cfg.get("fallback_policy", "error"),
                ),
                min_voxels=int(roi_cfg.get("min_voxels", 1)),
                mask_root=mask_root,
                mask_patterns=roi_cfg.get("mask_patterns"),
                layout_config=roi_cfg.get("layout_config", "configs/data.yaml"),
                cache_root=prep_cfg.get("cache_root", config.get("dataset.cache_root")),
                reference_volume_path=reference_volume_path,
                overwrite_existing=args.overwrite_existing,
            )

        subject_mixed_df = pd.read_parquet(final_subject_path)
        subject_report["status"] = "prepared"
        subject_report["mixed_index"] = str(final_subject_path)
        subject_report["mixed_rows"] = int(len(subject_mixed_df))
        subject_report["paired_groups"] = int(subject_mixed_df.groupby("pair_id")["condition"].nunique().ge(2).sum())
        if roi_summary is not None:
            subject_report["resolved_groups"] = roi_summary["resolved_groups"]
        subject_reports.append(subject_report)
        per_subject_indices.append(subject_mixed_df)

    if not per_subject_indices:
        raise ValueError(
            "prepare_overlap_bootstrap did not find any overlapping perception/imagery subjects to assemble."
        )

    combined = normalize_decoder_index(pd.concat(per_subject_indices, ignore_index=True))
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_output, index=False)

    report = {
        "config": str(Path(args.config)),
        "subjects": subject_reports,
        "combined_output_index": str(combined_output),
        "combined_rows": int(len(combined)),
        "combined_subjects": sorted(combined["subject"].astype(str).unique().tolist()),
        "combined_overlap_nsd_ids": sorted(int(value) for value in combined_overlap_ids),
        "combined_pairing_groups": int(combined.groupby("pair_id")["condition"].nunique().ge(2).sum()),
        "splits": combined["split"].value_counts().to_dict(),
        "materialize_roi": materialize_roi,
    }
    write_report(report_path, report)
    write_report(overlap_ids_path, {"nsd_ids": sorted(int(value) for value in combined_overlap_ids)})

    print(f"Prepared overlap bootstrap index: {combined_output}")
    print(f"Subjects included: {report['combined_subjects']}")
    print(f"Shared overlap nsdIds: {len(report['combined_overlap_nsd_ids'])}")
    print(f"Paired groups: {report['combined_pairing_groups']}")
    print(f"Report: {report_path}")
    print(json_safe(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
