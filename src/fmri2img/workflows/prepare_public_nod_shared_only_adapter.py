from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.prepare_public_nod_index import DEFAULT_OUTPUT as DEFAULT_INDEX


DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet"

EXPECTED_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 10)]
EXPECTED_SESSIONS = [f"ses-imagenet{index:02d}" for index in range(1, 5)]
EXPECTED_RUN = 10
EXPECTED_TASK = "imagenet"
EXPECTED_SCOPE = "imagenet_multisession_common_sessions"


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_INDEX


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_OUTPUT


def build_public_nod_shared_only_adapter(index_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(index_path)
    filtered = df[df["usable_for_later_shared_only_prep"]].copy()
    filtered = filtered[
        (filtered["task"] == EXPECTED_TASK)
        & (filtered["run"] == EXPECTED_RUN)
        & (filtered["contract_scope"] == EXPECTED_SCOPE)
        & filtered["subject"].isin(EXPECTED_SUBJECTS)
        & filtered["session"].isin(EXPECTED_SESSIONS)
    ].sort_values(["subject", "session", "run"]).reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            "The NOD prepared index contains no rows that satisfy the fixed shared-only adapter contract "
            f"({EXPECTED_TASK}, run-{EXPECTED_RUN}, {EXPECTED_SCOPE})."
        )

    unexpected_subjects = sorted(set(filtered["subject"]) - set(EXPECTED_SUBJECTS))
    unexpected_sessions = sorted(set(filtered["session"]) - set(EXPECTED_SESSIONS))
    unexpected_runs = sorted(set(filtered["run"]) - {EXPECTED_RUN})
    if unexpected_subjects or unexpected_sessions or unexpected_runs:
        details = {
            "unexpected_subjects": unexpected_subjects,
            "unexpected_sessions": unexpected_sessions,
            "unexpected_runs": unexpected_runs,
        }
        raise ValueError(
            "The NOD shared-only adapter surface drifted outside the fixed resolved subset contract: "
            f"{json.dumps(details)}"
        )

    expected_rows = len(EXPECTED_SUBJECTS) * len(EXPECTED_SESSIONS)
    if len(filtered) != expected_rows:
        raise ValueError(
            "The NOD shared-only adapter requires the full resolved run-10 slice "
            f"({expected_rows} rows), but the prepared index only exposes {len(filtered)} rows."
        )

    filtered["adapter_scope"] = "public_nod_imagenet_run10_shared_only"
    filtered["adapter_status"] = "adapter_ready_not_training_ready"

    report = {
        "source_index": str(index_path.resolve()),
        "contract": {
            "dataset_id": "ds004496",
            "lane": "practical_animus",
            "task": EXPECTED_TASK,
            "subjects": EXPECTED_SUBJECTS,
            "sessions": EXPECTED_SESSIONS,
            "run": EXPECTED_RUN,
            "contract_scope": EXPECTED_SCOPE,
        },
        "row_count": int(len(filtered)),
        "usable_rows": int(filtered["usable_for_later_shared_only_prep"].sum()),
        "subjects": filtered["subject"].value_counts().sort_index().to_dict(),
        "sessions": filtered["session"].value_counts().sort_index().to_dict(),
        "all_payloads_resolved": bool(
            filtered[
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
        ),
        "state": {
            "adapter_ready": True,
            "prep_ready": True,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "canonical target-selection contract for NOD stimuli",
            "ROI materialization contract aligned to the NOD derivatives",
            "shared-only training/eval config that points to this adapter output",
        ],
    }
    return filtered, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_shared_only_adapter")
    parser = argparse.ArgumentParser(
        description="Build the fixed run-10 shared-only adapter surface for the resolved NOD imagenet subset."
    )
    parser.add_argument("--input", type=Path, default=_default_input_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared, report = build_public_nod_shared_only_adapter(input_path)
    prepared.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD shared-only adapter: {output_path}")
    print(f"Rows: {len(prepared)}")
    print(f"Adapter ready: {report['state']['adapter_ready']}")
    print(f"Prep ready: {report['state']['prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
