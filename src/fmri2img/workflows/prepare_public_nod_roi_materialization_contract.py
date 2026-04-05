from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv


DEFAULT_JOIN = "cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet"
DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_roi_materialization_contract.parquet"

EXPECTED_SOURCE_ROWS = 36
EXPECTED_JOIN_ROWS = 3600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def build_public_nod_roi_materialization_contract(join_contract_path: Path) -> tuple[pd.DataFrame, dict]:
    join_contract_path = join_contract_path.resolve()
    join_df = pd.read_parquet(join_contract_path).sort_values(["pair_id"]).reset_index(drop=True)
    if len(join_df) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            f"NOD ROI contract requires the fixed {EXPECTED_JOIN_ROWS}-row join contract, "
            f"but {join_contract_path} exposes {len(join_df)} rows."
        )
    if not bool(join_df["pair_id"].is_unique):
        raise ValueError("NOD ROI contract requires one unique pair_id per row in the fixed join contract.")

    dataset_root = join_contract_path.parents[3] / "cache" / "public_datasets" / "ds004496"
    source_rows = (
        join_df[
            [
                "adapter_row_id",
                "subject",
                "session",
                "run",
                "events_path",
                "preproc_bold_path",
                "confounds_path",
                "ciftify_beta_path",
                "ciftify_label_path",
            ]
        ]
        .drop_duplicates()
        .sort_values(["subject", "session", "run"])
        .reset_index(drop=True)
    )
    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"NOD ROI contract requires the fixed {EXPECTED_SOURCE_ROWS}-row adapter/run source set, "
            f"but {join_contract_path} exposes {len(source_rows)} unique source rows."
        )

    contract_rows = []
    for row in source_rows.to_dict(orient="records"):
        subset = join_df[join_df["adapter_row_id"] == row["adapter_row_id"]].sort_values("trial_index").reset_index(drop=True)
        beta_path = dataset_root / row["ciftify_beta_path"]
        label_path = dataset_root / row["ciftify_label_path"]
        events_path = dataset_root / row["events_path"]
        preproc_bold_path = dataset_root / row["preproc_bold_path"]
        confounds_path = dataset_root / row["confounds_path"]

        beta_img = nib.load(beta_path)
        beta_shape = tuple(int(value) for value in beta_img.shape)
        label_count = len([line.strip() for line in label_path.read_text().splitlines() if line.strip()])
        join_row_count = int(len(subset))
        if beta_shape[0] != join_row_count or label_count != join_row_count:
            raise ValueError(
                "NOD ROI contract requires per-run beta rows, label rows, and join rows to match. "
                f"Found beta rows={beta_shape[0]}, labels={label_count}, join rows={join_row_count} "
                f"for {row['adapter_row_id']}."
            )

        contract_rows.append(
            {
                "adapter_row_id": row["adapter_row_id"],
                "subject": row["subject"],
                "session": row["session"],
                "run": int(row["run"]),
                "condition": "perception",
                "source_events_path": row["events_path"],
                "source_preproc_bold_path": row["preproc_bold_path"],
                "source_confounds_path": row["confounds_path"],
                "source_ciftify_beta_path": row["ciftify_beta_path"],
                "source_ciftify_label_path": row["ciftify_label_path"],
                "source_beta_rows": beta_shape[0],
                "source_beta_features": beta_shape[1] if len(beta_shape) > 1 else None,
                "source_label_rows": label_count,
                "join_rows": join_row_count,
                "pair_id_start": int(subset["pair_id"].min()),
                "pair_id_end": int(subset["pair_id"].max()),
                "beta_row_alignment_rule": "pair_id rows for this adapter_row_id are ordered by trial_index and map to beta row index = trial_index - 1",
                "required_output_artifact": "cache/indices/public_nod/imagenet_run10_roi_materialized.parquet",
                "required_output_columns_json": json.dumps(
                    [
                        "pair_id",
                        "subject",
                        "session",
                        "run",
                        "trial_index",
                        "condition",
                        "roi_values_json",
                        "source_beta_row_index",
                        "source_ciftify_beta_path",
                    ]
                ),
                "roi_materialized": False,
            }
        )

    artifact = pd.DataFrame(contract_rows).sort_values(["subject", "session", "run"]).reset_index(drop=True)
    report = {
        "source_join_contract": str(join_contract_path),
        "contract": {
            "dataset_id": "ds004496",
            "lane": "practical_animus",
            "task": "imagenet",
            "condition": "perception",
            "source_rows": EXPECTED_SOURCE_ROWS,
            "join_rows": EXPECTED_JOIN_ROWS,
        },
        "source_rows": int(len(artifact)),
        "verified_join_rows": int(sum(artifact["join_rows"].tolist())),
        "all_beta_alignment_verified": bool((artifact["source_beta_rows"] == artifact["join_rows"]).all()),
        "all_label_alignment_verified": bool((artifact["source_label_rows"] == artifact["join_rows"]).all()),
        "required_output_artifact": "cache/indices/public_nod/imagenet_run10_roi_materialized.parquet",
        "state": {
            "join_ready": True,
            "roi_ready": False,
            "downstream_prep_ready": False,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "materialize the ROI-side output artifact keyed by pair_id for the fixed NOD slice",
            "validate that the ROI materialized artifact preserves 1:1 alignment with the join contract",
            "dataset-side loader path from join contract + ROI artifact into the canonical shared-only dataset",
            "checked-in shared-only train/eval config pointing to the NOD join contract, ROI artifact, and target cache",
        ],
    }
    return artifact, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_roi_materialization_contract")
    parser = argparse.ArgumentParser(
        description="Build the fixed ROI materialization contract for the NOD shared-only slice."
    )
    parser.add_argument("--join-contract", type=Path, default=_default_path(DEFAULT_JOIN))
    parser.add_argument("--output", type=Path, default=_default_path(DEFAULT_OUTPUT))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact, report = build_public_nod_roi_materialization_contract(args.join_contract.resolve())
    artifact.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD ROI materialization contract: {output_path}")
    print(f"Source rows: {len(artifact)}")
    print(f"Verified join rows: {report['verified_join_rows']}")
    print(f"Join ready: {report['state']['join_ready']}")
    print(f"ROI ready: {report['state']['roi_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
