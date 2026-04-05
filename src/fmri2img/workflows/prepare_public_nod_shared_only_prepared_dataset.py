from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.materialize_public_nod_roi_artifact import DEFAULT_OUTPUT as DEFAULT_ROI
from fmri2img.workflows.prepare_public_nod_shared_only_join_contract import (
    DEFAULT_OUTPUT as DEFAULT_JOIN,
    EXPECTED_JOIN_ROWS,
)
from fmri2img.workflows.build_public_nod_target_embedding_cache import _default_output_path as _default_target_cache_path


DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def build_public_nod_shared_only_prepared_dataset(
    join_contract_path: Path,
    roi_artifact_path: Path,
    target_cache_path: Path,
) -> tuple[pd.DataFrame, dict]:
    join_contract_path = join_contract_path.resolve()
    roi_artifact_path = roi_artifact_path.resolve()
    target_cache_path = target_cache_path.resolve()

    join_df = pd.read_parquet(join_contract_path).sort_values("pair_id").reset_index(drop=True)
    roi_df = pd.read_parquet(roi_artifact_path).sort_values("pair_id").reset_index(drop=True)
    target_cache = pd.read_parquet(target_cache_path).sort_values("pair_id").reset_index(drop=True)

    for name, df in {
        "join contract": join_df,
        "ROI artifact": roi_df,
        "target cache": target_cache,
    }.items():
        if len(df) != EXPECTED_JOIN_ROWS:
            raise ValueError(f"NOD prepared dataset requires the fixed {EXPECTED_JOIN_ROWS}-row {name}.")
        if not bool(df["pair_id"].is_unique):
            raise ValueError(f"NOD prepared dataset requires unique pair_id rows in the {name}.")

    merged = join_df.merge(
        roi_df[
            [
                "pair_id",
                "nsdId",
                "nsd_id",
                "roi_names_json",
                "roi_values_json",
                "roi_features_json",
                "source_beta_row_index",
                "roi_feature_layout_version",
            ]
        ],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        target_cache[
            [
                "pair_id",
                "target_identifier",
                "embedding_model_id",
                "embedding_dimension",
                "clip_target_768",
            ]
        ],
        on=["pair_id", "target_identifier", "embedding_model_id", "embedding_dimension"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_JOIN_ROWS:
        raise ValueError("NOD prepared dataset could not preserve the fixed 3600-row slice after merging all sources.")

    prepared = merged[
        [
            "subject",
            "session",
            "run",
            "trial_index",
            "condition",
            "task",
            "pair_id",
            "nsdId",
            "nsd_id",
            "target_identifier",
            "stimulus_path",
            "roi_names_json",
            "roi_values_json",
            "roi_features_json",
            "source_beta_row_index",
            "source_ciftify_beta_path",
            "source_ciftify_label_path",
            "source_events_path",
            "roi_feature_layout_version",
        ]
    ].copy()
    prepared["target_cache_path"] = str(target_cache_path)
    prepared["target_embedding_column"] = "clip_target_768"
    prepared["dataset_contract_version"] = "public_nod_imagenet_run10_shared_only_prepared_v1"

    prepared = normalize_decoder_index(prepared, default_condition="perception", allowed_conditions=["perception"])
    prepared = prepared.sort_values("pair_id").reset_index(drop=True)

    dims = {
        "early_visual": len(json.loads(prepared.loc[0, "roi_features_json"])["early_visual"]),
        "ventral_visual": len(json.loads(prepared.loc[0, "roi_features_json"])["ventral_visual"]),
        "metacognitive": len(json.loads(prepared.loc[0, "roi_features_json"])["metacognitive"]),
    }

    report = {
        "source_join_contract": str(join_contract_path),
        "source_roi_artifact": str(roi_artifact_path),
        "source_target_cache": str(target_cache_path),
        "dataset_rows": int(len(prepared)),
        "unique_pair_ids": int(prepared["pair_id"].nunique()),
        "split_counts": {key: int(value) for key, value in prepared["split"].value_counts().to_dict().items()},
        "required_dataset_columns": prepared.columns.tolist(),
        "roi_feature_dimensions": dims,
        "state": {
            "join_ready": True,
            "roi_ready": True,
            "downstream_prep_ready": True,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "checked-in shared-only train/eval config pointing to the fixed NOD prepared dataset and target cache",
            "preflight and canonical trainer validation for the fixed NOD prepared dataset",
        ],
    }
    return prepared, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_shared_only_prepared_dataset")
    parser = argparse.ArgumentParser(
        description="Build the fixed prepared dataset artifact for the NOD shared-only slice."
    )
    parser.add_argument("--join-contract", type=Path, default=_default_path(DEFAULT_JOIN))
    parser.add_argument("--roi-artifact", type=Path, default=_default_path(DEFAULT_ROI))
    parser.add_argument("--target-cache", type=Path, default=_default_target_cache_path())
    parser.add_argument("--output", type=Path, default=_default_path(DEFAULT_OUTPUT))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared, report = build_public_nod_shared_only_prepared_dataset(
        args.join_contract.resolve(),
        args.roi_artifact.resolve(),
        args.target_cache.resolve(),
    )
    prepared.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD shared-only dataset: {output_path}")
    print(f"Rows: {len(prepared)}")
    print(f"Unique pair_ids: {report['unique_pair_ids']}")
    print(f"Downstream prep ready: {report['state']['downstream_prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
