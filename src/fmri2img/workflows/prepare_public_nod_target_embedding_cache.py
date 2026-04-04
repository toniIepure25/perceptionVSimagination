from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv


DEFAULT_INPUT = "cache/indices/public_nod/imagenet_run10_target_selection.parquet"
DEFAULT_OUTPUT = "cache/indices/public_nod/imagenet_run10_target_embedding_manifest.parquet"

EXPECTED_SUBJECTS = [f"sub-{index:02d}" for index in range(1, 10)]
EXPECTED_SESSIONS = [f"ses-imagenet{index:02d}" for index in range(1, 5)]
EXPECTED_RUN = 10
EXPECTED_SELECTION_ROWS = len(EXPECTED_SUBJECTS) * len(EXPECTED_SESSIONS) * 100
EXPECTED_UNIQUE_TARGETS = EXPECTED_SELECTION_ROWS
EMBEDDING_MODEL_ID = "openai/clip-vit-large-patch14"
EMBEDDING_DIMENSION = 768
EMBEDDING_COLUMN = "clip_target_768"


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_INPUT


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_OUTPUT


def _validate_target_selection(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    required = df.copy()
    required = required[
        (required["task"] == "imagenet")
        & (required["run"] == EXPECTED_RUN)
        & (required["lane"] == "practical_animus")
        & (required["target_domain"] == "imagenet")
        & (required["adapter_scope"] == "public_nod_imagenet_run10_shared_only")
        & required["subject"].isin(EXPECTED_SUBJECTS)
        & required["session"].isin(EXPECTED_SESSIONS)
    ].sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)

    if len(required) != EXPECTED_SELECTION_ROWS:
        raise ValueError(
            f"NOD target-embedding preparation requires the fixed {EXPECTED_SELECTION_ROWS}-row target-selection slice, "
            f"but {input_path} exposes {len(required)} qualifying rows."
        )
    unique_targets = int(required["target_identifier"].nunique())
    if unique_targets != EXPECTED_UNIQUE_TARGETS:
        raise ValueError(
            "NOD target-embedding preparation requires one unique target identifier per selected trial in the fixed slice, "
            f"but {input_path} exposes {unique_targets} unique ids for {len(required)} rows."
        )
    return required


def build_public_nod_target_embedding_manifest(input_path: Path) -> tuple[pd.DataFrame, dict]:
    input_path = input_path.resolve()
    selection = pd.read_parquet(input_path)
    selection = _validate_target_selection(selection, input_path)

    dataset_root = input_path.parents[3] / "cache" / "public_datasets" / "ds004496"
    stimuli_root = dataset_root / "stimuli"

    manifest = (
        selection[
            [
                "dataset_id",
                "dataset_label",
                "lane",
                "task",
                "subject",
                "session",
                "run",
                "trial_index",
                "stim_file",
                "target_identifier",
                "target_source",
                "target_domain",
                "adapter_scope",
                "adapter_status",
            ]
        ]
        .copy()
        .sort_values(["subject", "session", "run", "trial_index"])
        .reset_index(drop=True)
    )

    local_paths = [stimuli_root / stim_file for stim_file in manifest["stim_file"].tolist()]
    manifest["stimulus_path"] = [str(path.relative_to(input_path.parents[3])) for path in local_paths]
    manifest["stimulus_payload_visible"] = [os.path.lexists(path) for path in local_paths]
    manifest["stimulus_payload_resolved"] = [path.exists() for path in local_paths]
    manifest["stimulus_is_symlink"] = [path.is_symlink() for path in local_paths]
    manifest["embedding_model_id"] = EMBEDDING_MODEL_ID
    manifest["embedding_dimension"] = EMBEDDING_DIMENSION
    manifest["embedding_column"] = EMBEDDING_COLUMN
    manifest["embedding_materialized"] = False
    manifest["cache_kind"] = "manifest"
    manifest["cache_key"] = manifest["target_identifier"]
    manifest["embedding_status"] = [
        "embedding_pending"
        if resolved
        else ("missing_image_payload" if visible else "missing_image_reference")
        for visible, resolved in zip(
            manifest["stimulus_payload_visible"].tolist(),
            manifest["stimulus_payload_resolved"].tolist(),
        )
    ]

    visible_count = int(manifest["stimulus_payload_visible"].sum())
    resolved_count = int(manifest["stimulus_payload_resolved"].sum())
    report = {
        "source_target_selection": str(input_path),
        "contract": {
            "dataset_id": "ds004496",
            "lane": "practical_animus",
            "task": "imagenet",
            "subjects": EXPECTED_SUBJECTS,
            "sessions": EXPECTED_SESSIONS,
            "run": EXPECTED_RUN,
            "selection_rows": EXPECTED_SELECTION_ROWS,
            "unique_targets": EXPECTED_UNIQUE_TARGETS,
        },
        "output_kind": "target_embedding_manifest",
        "target_selection_rows": int(len(selection)),
        "unique_target_identifiers": int(manifest["target_identifier"].nunique()),
        "visible_stimulus_payloads": visible_count,
        "resolved_stimulus_payloads": resolved_count,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_column": EMBEDDING_COLUMN,
        "state": {
            "target_embedding_ready": resolved_count == len(manifest),
            "downstream_prep_ready": False,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "materialize the exact NOD stimulus JPEG payloads referenced by this manifest",
            "compute the canonical 768-D ViT-L/14 embeddings for the fixed NOD target slice",
            "ROI materialization contract aligned to the NOD derivatives",
            "shared-only training/eval config that points to the NOD adapter, target-selection artifact, and target cache",
        ],
    }
    return manifest, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.prepare_public_nod_target_embedding_cache")
    parser = argparse.ArgumentParser(
        description="Build the fixed target-embedding manifest for the NOD shared-only target-selection slice."
    )
    parser.add_argument("--input", type=Path, default=_default_input_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest, report = build_public_nod_target_embedding_manifest(input_path)
    manifest.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD target-embedding manifest: {output_path}")
    print(f"Rows: {len(manifest)}")
    print(f"Unique target identifiers: {report['unique_target_identifiers']}")
    print(f"Target-embedding ready: {report['state']['target_embedding_ready']}")
    print(f"Downstream prep ready: {report['state']['downstream_prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
