from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.inspect_public_nod import _default_dataset_root, summarize_nod_layout


DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_multisession_common_sessions.parquet"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_OUTPUT


def _resolve_path_state(path: Path) -> tuple[bool, bool]:
    return path.is_symlink() or path.exists(), path.exists()


def _run_artifact_paths(dataset_root: Path, subject: str, session: str, run: int) -> dict[str, Path]:
    run_name = f"{run}"
    raw_prefix = f"{subject}_{session}_task-imagenet_run-{run_name}"
    ciftify_run_dir = dataset_root / "derivatives" / "ciftify" / subject / "results" / f"{session}_task-imagenet_run-{run_name}"
    fmriprep_func_dir = dataset_root / "derivatives" / "fmriprep" / subject / session / "func"
    raw_func_dir = dataset_root / subject / session / "func"
    return {
        "events_path": raw_func_dir / f"{raw_prefix}_events.tsv",
        "preproc_bold_path": fmriprep_func_dir / f"{raw_prefix}_space-T1w_desc-preproc_bold.nii.gz",
        "confounds_path": fmriprep_func_dir / f"{raw_prefix}_desc-confounds_timeseries.tsv",
        "ciftify_beta_path": ciftify_run_dir / f"{session}_task-imagenet_run-{run_name}_beta.dscalar.nii",
        "ciftify_label_path": ciftify_run_dir / f"{session}_task-imagenet_run-{run_name}_label.txt",
    }


def build_public_nod_index(dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    summary = summarize_nod_layout(dataset_root)
    contract = summary["prepared_index_contract"]
    subjects = contract["subjects"]
    sessions = contract["common_sessions"]

    rows: list[dict] = []
    for subject in subjects:
        for session in sessions:
            for run in range(1, 11):
                paths = _run_artifact_paths(dataset_root, subject, session, run)
                visible = {}
                resolved = {}
                for key, path in paths.items():
                    path_visible, path_resolved = _resolve_path_state(path)
                    visible[key] = path_visible
                    resolved[key] = path_resolved

                required_keys = list(paths.keys())
                visible_count = sum(int(visible[key]) for key in required_keys)
                resolved_count = sum(int(resolved[key]) for key in required_keys)
                if visible_count == 0:
                    status = "missing"
                elif resolved_count == len(required_keys):
                    status = "resolved"
                elif visible_count == len(required_keys):
                    status = "missing_payload"
                else:
                    status = "incomplete"

                rows.append(
                    {
                        "dataset_id": "ds004496",
                        "dataset_label": "Natural Object Dataset (NOD)",
                        "lane": "practical_animus",
                        "task": "imagenet",
                        "subject": subject,
                        "session": session,
                        "run": run,
                        "contract_scope": "imagenet_multisession_common_sessions",
                        "events_path": str(paths["events_path"].relative_to(dataset_root)),
                        "preproc_bold_path": str(paths["preproc_bold_path"].relative_to(dataset_root)),
                        "confounds_path": str(paths["confounds_path"].relative_to(dataset_root)),
                        "ciftify_beta_path": str(paths["ciftify_beta_path"].relative_to(dataset_root)),
                        "ciftify_label_path": str(paths["ciftify_label_path"].relative_to(dataset_root)),
                        "events_visible": visible["events_path"],
                        "preproc_bold_visible": visible["preproc_bold_path"],
                        "confounds_visible": visible["confounds_path"],
                        "ciftify_beta_visible": visible["ciftify_beta_path"],
                        "ciftify_label_visible": visible["ciftify_label_path"],
                        "events_resolved": resolved["events_path"],
                        "preproc_bold_resolved": resolved["preproc_bold_path"],
                        "confounds_resolved": resolved["confounds_path"],
                        "ciftify_beta_resolved": resolved["ciftify_beta_path"],
                        "ciftify_label_resolved": resolved["ciftify_label_path"],
                        "row_status": status,
                        "usable_for_later_shared_only_prep": status == "resolved",
                    }
                )

    df = pd.DataFrame(rows).sort_values(["subject", "session", "run"]).reset_index(drop=True)
    report = {
        "dataset_root": str(dataset_root.resolve()),
        "output_contract": contract,
        "row_count": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in df["row_status"].value_counts().to_dict().items()},
        "usable_rows": int(df["usable_for_later_shared_only_prep"].sum()),
        "subjects": subjects,
        "sessions": sessions,
    }
    return df, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_index")
    parser = argparse.ArgumentParser(description="Build the first prepared index for the NOD imagenet multi-session common-session subset.")
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, report = build_public_nod_index(dataset_root)
    df.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD index: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Status counts: {report['status_counts']}")
    print(f"Usable rows: {report['usable_rows']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
