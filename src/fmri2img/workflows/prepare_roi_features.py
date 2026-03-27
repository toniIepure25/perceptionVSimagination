from __future__ import annotations

import argparse
from pathlib import Path

from fmri2img.data.canonical import build_mixed_condition_index
from fmri2img.roi import ROIGroupSpec, materialize_roi_features
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import config_subject, get_preparation_section, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize canonical ROI features from real ROI masks and volumes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    dataset_cfg = config["dataset"]
    roi_cfg = config["roi"]
    prep_cfg = get_preparation_section(config, "roi")

    input_index = prep_cfg.get("input_index") or dataset_cfg.get("mixed_output_index") or dataset_cfg.get("mixed_index")
    if input_index is None:
        raise KeyError("Canonical ROI preparation requires dataset.mixed_output_index, dataset.mixed_index, or preparation.roi.input_index.")

    input_path = Path(input_index)
    if not input_path.exists():
        if dataset_cfg.get("perception_index") and dataset_cfg.get("imagery_index"):
            build_mixed_condition_index(
                perception_index=dataset_cfg["perception_index"],
                imagery_index=dataset_cfg["imagery_index"],
                output_path=input_path,
                subject=config_subject(config),
            )
        else:
            raise FileNotFoundError(
                f"ROI preparation input index does not exist: {input_path}. "
                "Run prepare_imagery_index and prepare_mixed_index first."
            )

    output_path = Path(prep_cfg.get("output_index", input_path))
    provenance_path = prep_cfg.get("provenance_path")
    if provenance_path is None:
        provenance_path = str(output_path.with_name(f"{output_path.stem}_roi_summary.json"))

    summary = materialize_roi_features(
        index=input_path,
        subject=config_subject(config),
        output_path=output_path,
        provenance_path=provenance_path,
        group_spec=ROIGroupSpec(
            groups=roi_cfg.get("groups", {}),
            missing_policy=roi_cfg.get("missing_policy", "error"),
            fallback_policy=roi_cfg.get("fallback_policy", "error"),
        ),
        min_voxels=int(roi_cfg.get("min_voxels", 1)),
        mask_root=roi_cfg.get("mask_root"),
        mask_patterns=roi_cfg.get("mask_patterns"),
        layout_config=roi_cfg.get("layout_config", "configs/data.yaml"),
        cache_root=prep_cfg.get("cache_root", config.get("dataset.cache_root")),
        reference_volume_path=roi_cfg.get("reference_volume_path"),
        overwrite_existing=args.overwrite_existing,
    )
    write_report(provenance_path, summary)

    print(f"Prepared ROI materialization index: {output_path}")
    print(f"ROI provenance: {provenance_path}")
    print(f"Materialized rows: {summary['materialized_rows']}")
    print(f"Resolved groups: {list(summary['resolved_groups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
