from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.data.nsd_imagery import build_nsd_imagery_index
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import config_subject, get_preparation_section, write_report


def _format_template(value: str | None, *, subject: str) -> str | None:
    if value is None:
        return None
    return str(value).format(subject=subject)


def _replace_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix("").with_name(path.with_suffix("").name + suffix)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical imagery index used by shared/private decoding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    subject = config_subject(config)
    prep_cfg = get_preparation_section(config, "imagery")
    data_root = prep_cfg.get("data_root") or config.get("dataset.imagery_data_root")
    metadata_root = prep_cfg.get("metadata_root")
    beta_root = prep_cfg.get("beta_root")
    beta_path = _format_template(prep_cfg.get("beta_path_template"), subject=subject) or prep_cfg.get("beta_path")
    if not any([data_root, metadata_root, beta_root, beta_path]):
        raise KeyError(
            "Canonical imagery preparation requires either preparation.imagery.data_root / dataset.imagery_data_root "
            "or split-layout paths such as preparation.imagery.metadata_root and beta_root/beta_path_template."
        )

    output_template = config["dataset"].get("imagery_index") or config.get("preparation.overlap.imagery_index_template")
    if not output_template:
        raise KeyError(
            "Canonical imagery preparation requires dataset.imagery_index or preparation.overlap.imagery_index_template."
        )
    output_path = Path(str(output_template).format(subject=subject))
    cache_root = Path(prep_cfg.get("cache_root", "cache"))
    stimulus_root = prep_cfg.get("stimulus_root")
    source_report_path = Path(_format_template(prep_cfg.get("source_report_template"), subject=subject) or _replace_suffix(output_path, ".source_report.json"))
    prepared_report_path = Path(_format_template(prep_cfg.get("report_template"), subject=subject) or _replace_suffix(output_path, ".report.json"))
    result_path = build_nsd_imagery_index(
        data_root=None if data_root is None else Path(data_root),
        subject=subject,
        cache_root=cache_root,
        output_path=output_path,
        stimulus_root=None if stimulus_root is None else Path(stimulus_root),
        metadata_root=None if metadata_root is None else Path(metadata_root),
        beta_root=None if beta_root is None else Path(beta_root),
        beta_path=None if beta_path is None else Path(beta_path),
        report_path=source_report_path,
        dry_run=args.dry_run,
        verbose=True,
    )
    if args.dry_run:
        return 0

    df = pd.read_parquet(result_path)
    rows_before_filter = int(len(df))
    allowed_conditions = prep_cfg.get("conditions", config["dataset"].get("imagery_conditions", ["imagery"]))
    stimulus_sets = prep_cfg.get("stimulus_sets")
    if stimulus_sets:
        allowed_sets = {str(value) for value in stimulus_sets}
        if "stimulus_set" not in df.columns:
            raise ValueError(
                "preparation.imagery.stimulus_sets was configured, but the imagery builder did not emit a stimulus_set column."
            )
        df = df[df["stimulus_set"].astype(str).isin(allowed_sets)].reset_index(drop=True)
    require_nsd_id = bool(prep_cfg.get("require_nsd_id", True))
    if require_nsd_id:
        nsd_mask = pd.Series(False, index=df.index)
        for column in ("nsdId", "nsd_id"):
            if column in df.columns:
                nsd_mask = nsd_mask | df[column].notna()
        df = df[nsd_mask].reset_index(drop=True)
    if df.empty:
        details = {
            "allowed_conditions": list(allowed_conditions or []),
            "stimulus_sets": list(stimulus_sets or []),
            "require_nsd_id": require_nsd_id,
            "source_index": str(result_path),
        }
        raise ValueError(
            "Canonical imagery preparation produced no rows after filtering. "
            f"Check the metadata layout or relax preparation.imagery filters: {json.dumps(details)}"
        )
    prepared = normalize_decoder_index(
        df,
        default_condition="imagery",
        allowed_conditions=allowed_conditions,
    )
    prepared.to_parquet(result_path, index=False)
    write_report(
        prepared_report_path,
        {
            "subject": subject,
            "output_path": str(result_path),
            "source_report_path": str(source_report_path),
            "rows_before_filter": rows_before_filter,
            "rows_after_filter": int(len(prepared)),
            "conditions": prepared["condition"].value_counts().to_dict(),
            "stimulus_sets": prepared["stimulus_set"].value_counts().to_dict() if "stimulus_set" in prepared.columns else {},
            "unique_nsd_ids": int(prepared["nsdId"].nunique()),
            "paired_groups": int(prepared["pair_id"].nunique()),
            "splits": prepared["split"].value_counts().to_dict(),
            "filters": {
                "allowed_conditions": list(allowed_conditions or []),
                "stimulus_sets": list(stimulus_sets or []),
                "require_nsd_id": require_nsd_id,
            },
        },
    )

    print(f"Prepared imagery index: {result_path}")
    print(f"Rows: {len(prepared)}")
    print(f"Unique nsdId: {prepared['nsdId'].nunique()}")
    print(f"Paired groups: {prepared['pair_id'].nunique()}")
    print(f"Splits: {prepared['split'].value_counts().to_dict()}")
    print(f"Source report: {source_report_path}")
    print(f"Prepared report: {prepared_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
