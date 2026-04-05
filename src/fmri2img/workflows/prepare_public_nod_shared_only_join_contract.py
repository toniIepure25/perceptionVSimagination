from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv


DEFAULT_ADAPTER = "cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet"
DEFAULT_SELECTION = "cache/indices/public_nod/imagenet_run10_target_selection.parquet"
DEFAULT_TARGET_CACHE = "cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet"
DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet"

EXPECTED_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 10)]
EXPECTED_SESSIONS = [f"ses-imagenet{index:02d}" for index in range(1, 5)]
EXPECTED_RUN = 10
EXPECTED_ADAPTER_ROWS = len(EXPECTED_SUBJECTS) * len(EXPECTED_SESSIONS)
EXPECTED_JOIN_ROWS = EXPECTED_ADAPTER_ROWS * 100


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _validate_adapter(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = df[
        (df["task"] == "imagenet")
        & (df["run"] == EXPECTED_RUN)
        & (df["contract_scope"] == "imagenet_multisession_common_sessions")
        & df["subject"].isin(EXPECTED_SUBJECTS)
        & df["session"].isin(EXPECTED_SESSIONS)
        & df["usable_for_later_shared_only_prep"]
    ].copy()
    required = required.sort_values(["subject", "session", "run"]).reset_index(drop=True)
    if len(required) != EXPECTED_ADAPTER_ROWS:
        raise ValueError(
            f"NOD join contract requires the fixed {EXPECTED_ADAPTER_ROWS}-row adapter slice, "
            f"but {input_path} exposes {len(required)} qualifying rows."
        )
    return required


def _validate_selection(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = df[
        (df["task"] == "imagenet")
        & (df["run"] == EXPECTED_RUN)
        & (df["lane"] == "practical_animus")
        & (df["target_domain"] == "imagenet")
        & (df["adapter_scope"] == "public_nod_imagenet_run10_shared_only")
        & df["subject"].isin(EXPECTED_SUBJECTS)
        & df["session"].isin(EXPECTED_SESSIONS)
    ].copy()
    required = required.sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)
    if len(required) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            f"NOD join contract requires the fixed {EXPECTED_JOIN_ROWS}-row target-selection slice, "
            f"but {input_path} exposes {len(required)} qualifying rows."
        )
    if int(required["target_identifier"].nunique()) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            f"NOD join contract requires one unique target identifier per trial, but {input_path} does not satisfy that."
        )
    if "stimulus_path" not in required.columns:
        if "stim_file" not in required.columns:
            raise ValueError(
                f"NOD join contract requires either stimulus_path or stim_file in {input_path} to connect target-selection rows to the target cache."
            )
        required["stimulus_path"] = required["stim_file"].map(
            lambda value: f"cache/public_datasets/ds004496/stimuli/{value}"
        )
    return required


def _validate_target_cache(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = df.copy().sort_values("pair_id").reset_index(drop=True)
    if len(required) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            f"NOD join contract requires the fixed {EXPECTED_JOIN_ROWS}-row target cache, "
            f"but {input_path} exposes {len(required)} rows."
        )
    if int(required["target_identifier"].nunique()) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            f"NOD join contract requires one unique target identifier per cached embedding, but {input_path} does not satisfy that."
        )
    return required


def build_public_nod_shared_only_join_contract(
    adapter_path: Path,
    selection_path: Path,
    target_cache_path: Path,
) -> tuple[pd.DataFrame, dict]:
    adapter = _validate_adapter(pd.read_parquet(adapter_path), adapter_path)
    selection = _validate_selection(pd.read_parquet(selection_path), selection_path)
    target_cache = _validate_target_cache(pd.read_parquet(target_cache_path), target_cache_path)

    adapter = adapter.copy()
    adapter["adapter_row_id"] = [
        f"{row.subject}|{row.session}|run-{int(row.run)}" for row in adapter.itertuples(index=False)
    ]

    joined = selection.merge(
        adapter[
            [
                "subject",
                "session",
                "run",
                "adapter_row_id",
                "events_path",
                "preproc_bold_path",
                "confounds_path",
                "ciftify_beta_path",
                "ciftify_label_path",
            ]
        ].rename(
            columns={
                "events_path": "source_events_path",
                "preproc_bold_path": "source_preproc_bold_path",
                "confounds_path": "source_confounds_path",
                "ciftify_beta_path": "source_ciftify_beta_path",
                "ciftify_label_path": "source_ciftify_label_path",
            }
        ),
        on=["subject", "session", "run"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            "NOD join contract could not preserve the fixed 3600-row selection slice after attaching adapter metadata."
        )

    joined = joined.merge(
        target_cache[
            [
                "pair_id",
                "target_identifier",
                "stimulus_path",
                "embedding_model_id",
                "embedding_dimension",
                "clip_target_768",
            ]
        ],
        on=["target_identifier", "stimulus_path"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != EXPECTED_JOIN_ROWS:
        raise ValueError(
            "NOD join contract could not preserve the fixed 3600-row slice after joining the target cache."
        )

    joined = joined.sort_values(["pair_id"]).reset_index(drop=True)
    joined["condition"] = "perception"
    joined["join_contract_version"] = "public_nod_imagenet_run10_v1"
    joined["target_cache_path"] = str(target_cache_path.resolve())
    joined["target_embedding_column"] = "clip_target_768"
    joined["target_ready"] = True
    joined["roi_contract_key"] = joined["pair_id"]
    joined["roi_ready"] = False
    joined["training_ready"] = False

    artifact = joined[
        [
            "pair_id",
            "adapter_row_id",
            "subject",
            "session",
            "run",
            "trial_index",
            "condition",
            "task",
            "target_identifier",
            "stimulus_path",
            "embedding_model_id",
            "embedding_dimension",
            "target_embedding_column",
            "source_events_path",
            "source_preproc_bold_path",
            "source_confounds_path",
            "source_ciftify_beta_path",
            "source_ciftify_label_path",
            "target_cache_path",
            "roi_contract_key",
            "join_contract_version",
            "target_ready",
            "roi_ready",
            "training_ready",
        ]
    ].copy()

    report = {
        "source_adapter": str(adapter_path.resolve()),
        "source_target_selection": str(selection_path.resolve()),
        "source_target_cache": str(target_cache_path.resolve()),
        "contract": {
            "dataset_id": "ds004496",
            "lane": "practical_animus",
            "task": "imagenet",
            "subjects": EXPECTED_SUBJECTS,
            "sessions": EXPECTED_SESSIONS,
            "run": EXPECTED_RUN,
        },
        "adapter_rows": int(len(adapter)),
        "join_rows": int(len(artifact)),
        "unique_pair_ids": int(artifact["pair_id"].nunique()),
        "primary_row_id": "pair_id",
        "required_downstream_columns": artifact.columns.tolist(),
        "state": {
            "join_ready": True,
            "roi_ready": False,
            "downstream_prep_ready": False,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "ROI materialization artifact keyed by pair_id for the fixed NOD slice",
            "dataset-side loader/join path that consumes this join contract into the canonical shared-only dataset",
            "checked-in shared-only train/eval config pointing to the adapter, join contract, ROI artifact, and target cache",
        ],
    }
    return artifact, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_shared_only_join_contract")
    parser = argparse.ArgumentParser(
        description="Build the fixed dataset-side join contract for the NOD shared-only slice."
    )
    parser.add_argument("--adapter", type=Path, default=_default_path(DEFAULT_ADAPTER))
    parser.add_argument("--target-selection", type=Path, default=_default_path(DEFAULT_SELECTION))
    parser.add_argument("--target-cache", type=Path, default=_default_path(DEFAULT_TARGET_CACHE))
    parser.add_argument("--output", type=Path, default=_default_path(DEFAULT_OUTPUT))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact, report = build_public_nod_shared_only_join_contract(
        args.adapter.resolve(),
        args.target_selection.resolve(),
        args.target_cache.resolve(),
    )
    artifact.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD join contract: {output_path}")
    print(f"Rows: {len(artifact)}")
    print(f"Unique pair_ids: {report['unique_pair_ids']}")
    print(f"Join ready: {report['state']['join_ready']}")
    print(f"ROI ready: {report['state']['roi_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
