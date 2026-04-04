from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv


DEFAULT_DATASET_ROOT = "cache/public_datasets/ds004496"


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_DATASET_ROOT


def _collect_subjects(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.glob("sub-*") if path.is_dir())


def _read_multi_session_subjects(dataset_root: Path) -> list[str]:
    participants_path = dataset_root / "participants.tsv"
    if not participants_path.exists():
        return []
    lines = participants_path.read_text().splitlines()
    if not lines:
        return []
    multi_session: list[str] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "multi-session":
            multi_session.append(parts[0])
    return sorted(multi_session)


def summarize_nod_layout(dataset_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    subjects = _collect_subjects(dataset_root)
    multi_session_subjects = _read_multi_session_subjects(dataset_root)

    raw_event_files = sorted(dataset_root.glob("sub-*/ses-*/func/*_events.tsv"))
    raw_bold_sidecars = sorted(dataset_root.glob("sub-*/ses-*/func/*_bold.json"))
    fmriprep_bolds = sorted(
        dataset_root.glob("derivatives/fmriprep/sub-*/ses-*/func/*_space-T1w_desc-preproc_bold.nii.gz")
    )
    confounds = sorted(
        dataset_root.glob("derivatives/fmriprep/sub-*/ses-*/func/*_desc-confounds_timeseries.tsv")
    )
    ciftify_beta = sorted(dataset_root.glob("derivatives/ciftify/sub-*/results/**/*_beta.dscalar.nii"))
    ciftify_labels = sorted(dataset_root.glob("derivatives/ciftify/sub-*/results/**/*_label.txt"))
    ciftify_dtseries = sorted(dataset_root.glob("derivatives/ciftify/sub-*/results/**/*_Atlas.dtseries.nii"))
    floc_labels = sorted(dataset_root.glob("derivatives/ciftify/sub-*/results/ses-floc_task-floc/*.dlabel.nii"))

    sessions = sorted(
        {
            path.parent.parent.name
            for path in raw_event_files
        }
    )
    tasks = sorted(
        {
            path.name.split("_task-")[1].split("_")[0]
            for path in raw_event_files
            if "_task-" in path.name
        }
    )

    readiness = {
        "metadata_clone_present": (dataset_root / "dataset_description.json").exists()
        and (dataset_root / "participants.tsv").exists(),
        "raw_bids_visible": bool(raw_event_files and raw_bold_sidecars),
        "fmriprep_volume_visible": bool(fmriprep_bolds and confounds),
        "ciftify_surface_visible": bool(ciftify_dtseries),
        "surface_glm_visible": bool(ciftify_beta and ciftify_labels),
        "floc_roi_assets_visible": bool(floc_labels),
        "animus_shared_only_training_ready": False,
    }

    missing_for_first_runnable_path = [
        "canonical target-selection contract for NOD stimuli",
        "checked-in perception-only prepared-index adapter",
        "ROI materialization contract aligned to NOD derivatives",
        "shared-only training/eval config that points to a real prepared NOD index",
    ]

    recommended_first_contract = {
        "lane": "practical Animus lane",
        "task_family": "imagenet perception-only",
        "subject_tier": "multi-session subjects first",
        "subjects": multi_session_subjects,
        "derivative_source": "ciftify beta.dscalar.nii with paired label.txt",
        "status": "inspection_ready_not_training_ready",
    }

    return {
        "dataset_root": str(dataset_root),
        "subject_count": len(subjects),
        "subjects": subjects,
        "multi_session_subjects": multi_session_subjects,
        "task_sessions_visible": sessions,
        "task_types_visible": tasks,
        "counts": {
            "raw_event_files": len(raw_event_files),
            "raw_bold_sidecars": len(raw_bold_sidecars),
            "fmriprep_bold_files": len(fmriprep_bolds),
            "confound_tsv_files": len(confounds),
            "ciftify_beta_files": len(ciftify_beta),
            "ciftify_label_files": len(ciftify_labels),
            "ciftify_dtseries_files": len(ciftify_dtseries),
            "floc_dlabel_files": len(floc_labels),
        },
        "readiness": readiness,
        "recommended_first_contract": recommended_first_contract,
        "missing_for_first_runnable_path": missing_for_first_runnable_path,
    }


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.inspect_public_nod")
    parser = argparse.ArgumentParser(description="Inspect the checked-in ds004496 (NOD) metadata/derivative layout.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help=f"Path to the ds004496 clone (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON.",
    )
    args = parser.parse_args(argv)

    summary = summarize_nod_layout(args.dataset_root)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"dataset_root: {summary['dataset_root']}")
    print(f"subject_count: {summary['subject_count']}")
    print(f"multi_session_subjects: {', '.join(summary['multi_session_subjects'])}")
    print(f"task_types_visible: {', '.join(summary['task_types_visible'])}")
    print("readiness:")
    for key, value in summary["readiness"].items():
        print(f"  {key}: {value}")
    print("recommended_first_contract:")
    for key, value in summary["recommended_first_contract"].items():
        if isinstance(value, list):
            value = ", ".join(value)
        print(f"  {key}: {value}")
    print("missing_for_first_runnable_path:")
    for item in summary["missing_for_first_runnable_path"]:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
