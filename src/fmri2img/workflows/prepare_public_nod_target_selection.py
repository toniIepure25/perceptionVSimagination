from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv


DEFAULT_INPUT = "cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet"
DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_target_selection.parquet"

EXPECTED_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 10)]
EXPECTED_SESSIONS = [f"ses-imagenet{index:02d}" for index in range(1, 5)]
EXPECTED_RUN = 10
EXPECTED_ROWS = len(EXPECTED_SUBJECTS) * len(EXPECTED_SESSIONS)


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_INPUT


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_OUTPUT


def _validate_adapter(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = df.copy()
    required = required[
        (required["task"] == "imagenet")
        & (required["run"] == EXPECTED_RUN)
        & (required["contract_scope"] == "imagenet_multisession_common_sessions")
        & required["subject"].isin(EXPECTED_SUBJECTS)
        & required["session"].isin(EXPECTED_SESSIONS)
        & required["usable_for_later_shared_only_prep"]
    ].sort_values(["subject", "session", "run"]).reset_index(drop=True)

    if len(required) != EXPECTED_ROWS:
        raise ValueError(
            f"NOD target selection requires the fixed {EXPECTED_ROWS}-row adapter slice, "
            f"but {input_path} exposes {len(required)} qualifying rows."
        )
    if not bool(
        required[
            [
                "events_resolved",
                "preproc_bold_resolved",
                "confounds_resolved",
                "ciftify_beta_resolved",
                "ciftify_label_resolved",
            ]
        ]
        .all()
        .all()
    ):
        raise ValueError(
            f"NOD target selection requires fully resolved adapter rows, but {input_path} still contains unresolved payload flags."
        )
    return required


def build_public_nod_target_selection(input_path: Path) -> tuple[pd.DataFrame, dict]:
    input_path = input_path.resolve()
    adapter = pd.read_parquet(input_path)
    adapter = _validate_adapter(adapter, input_path)

    selection_rows: list[dict] = []
    per_run_counts: dict[str, int] = {}
    for row in adapter.to_dict(orient="records"):
        events_path = Path(row["events_path"])
        label_path = Path(row["ciftify_label_path"])
        dataset_root = input_path.parents[3] / "cache" / "public_datasets" / "ds004496"
        events = pd.read_csv(dataset_root / events_path, sep="\t")
        labels = [line.strip() for line in (dataset_root / label_path).read_text().splitlines() if line.strip()]

        if len(events) != len(labels):
            raise ValueError(
                "NOD target selection requires a one-to-one mapping between events.tsv and label.txt entries, "
                f"but {events_path} has {len(events)} rows and {label_path} has {len(labels)} labels."
            )

        key = f"{row['subject']}|{row['session']}|run-{row['run']}"
        per_run_counts[key] = len(events)
        for trial_index, (event_row, label_name) in enumerate(zip(events.to_dict(orient="records"), labels), start=1):
            stim_file = str(event_row["stim_file"])
            stim_basename = Path(stim_file).name
            if stim_basename != label_name:
                raise ValueError(
                    "NOD target selection requires deterministic agreement between events.tsv stim_file and label.txt. "
                    f"Found mismatch for {key}: stim_file={stim_basename}, label={label_name}."
                )
            selection_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_label": row["dataset_label"],
                    "lane": row["lane"],
                    "task": row["task"],
                    "subject": row["subject"],
                    "session": row["session"],
                    "run": int(row["run"]),
                    "trial_index": trial_index,
                    "onset": float(event_row["onset"]),
                    "duration": float(event_row["duration"]),
                    "trial_type": str(event_row["trial_type"]),
                    "response_time": None if pd.isna(event_row["response_time"]) else float(event_row["response_time"]),
                    "stim_file": stim_file,
                    "target_identifier": label_name,
                    "target_source": "events_stim_file_and_ciftify_label_match",
                    "target_domain": "imagenet",
                    "adapter_scope": row["adapter_scope"],
                    "adapter_status": row["adapter_status"],
                    "events_path": row["events_path"],
                    "ciftify_label_path": row["ciftify_label_path"],
                }
            )

    selection = pd.DataFrame(selection_rows).sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)
    report = {
        "source_adapter": str(input_path),
        "contract": {
            "dataset_id": "ds004496",
            "lane": "practical_animus",
            "task": "imagenet",
            "subjects": EXPECTED_SUBJECTS,
            "sessions": EXPECTED_SESSIONS,
            "run": EXPECTED_RUN,
        },
        "adapter_row_count": int(len(adapter)),
        "target_selection_rows": int(len(selection)),
        "unique_target_identifiers": int(selection["target_identifier"].nunique()),
        "per_run_target_counts": per_run_counts,
        "target_source": "events_stim_file_and_ciftify_label_match",
        "state": {
            "target_selection_ready": True,
            "downstream_prep_ready": True,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "canonical target embedding cache built from this target-selection artifact",
            "ROI materialization contract aligned to the NOD derivatives",
            "shared-only training/eval config that points to the NOD adapter and target-selection outputs",
        ],
    }
    return selection, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_target_selection")
    parser = argparse.ArgumentParser(
        description="Build the fixed target-selection artifact for the resolved NOD shared-only adapter slice."
    )
    parser.add_argument("--input", type=Path, default=_default_input_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selection, report = build_public_nod_target_selection(input_path)
    selection.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD target selection: {output_path}")
    print(f"Rows: {len(selection)}")
    print(f"Unique target identifiers: {report['unique_target_identifiers']}")
    print(f"Target-selection ready: {report['state']['target_selection_ready']}")
    print(f"Downstream prep ready: {report['state']['downstream_prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
