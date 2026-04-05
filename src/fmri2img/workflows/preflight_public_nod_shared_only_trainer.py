from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.preflight_public_nod_shared_only_trainer")

from fmri2img.workflows.common import (  # noqa: E402
    build_datasets,
    build_loaders,
    instantiate_model_from_dataset,
    load_workflow_config,
    resolve_runtime_device,
    validate_canonical_workflow_config,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_required_path(config, key: str) -> Path:
    value = config.get(f"public_nod.{key}")
    if value is None:
        raise KeyError(f"Missing public_nod.{key} in config.")
    path = _default_path(str(value)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required fixed NOD artifact is missing: public_nod.{key}={path}")
    return path


def _validate_fixed_slice(prepared_df: pd.DataFrame, config) -> dict[str, Any]:
    expected_subjects = list(config.get("public_nod.subjects", []))
    expected_sessions = list(config.get("public_nod.sessions", []))
    expected_run = int(config.get("public_nod.run", 10))
    expected_rows = int(config.get("public_nod.pair_rows", 3600))

    if len(prepared_df) != expected_rows or not bool(prepared_df["pair_id"].is_unique):
        raise ValueError(
            f"NOD trainer preflight requires the fixed {expected_rows}-row prepared dataset with unique pair_id rows."
        )
    if sorted(prepared_df["subject"].astype(str).unique().tolist()) != sorted(expected_subjects):
        raise ValueError("NOD trainer preflight detected subject drift outside the fixed slice contract.")
    if sorted(prepared_df["session"].astype(str).unique().tolist()) != sorted(expected_sessions):
        raise ValueError("NOD trainer preflight detected session drift outside the fixed slice contract.")
    if sorted(prepared_df["run"].astype(int).unique().tolist()) != [expected_run]:
        raise ValueError("NOD trainer preflight detected run drift outside the fixed slice contract.")
    if sorted(prepared_df["task"].astype(str).unique().tolist()) != [str(config.get("public_nod.task", "imagenet"))]:
        raise ValueError("NOD trainer preflight detected task drift outside the fixed slice contract.")
    return {
        "dataset_rows": int(len(prepared_df)),
        "unique_pair_ids": int(prepared_df["pair_id"].nunique()),
        "subjects": sorted(prepared_df["subject"].astype(str).unique().tolist()),
        "sessions": sorted(prepared_df["session"].astype(str).unique().tolist()),
        "runs": sorted(prepared_df["run"].astype(int).unique().tolist()),
        "split_counts": {key: int(value) for key, value in prepared_df["split"].value_counts().to_dict().items()},
    }


def build_public_nod_shared_only_trainer_preflight(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)

    target_cache_report_path = _resolve_required_path(config, "target_cache_report")
    join_report_path = _resolve_required_path(config, "join_report")
    roi_report_path = _resolve_required_path(config, "roi_report")
    prepared_report_path = _resolve_required_path(config, "prepared_report")
    prepared_dataset_path = Path(config["dataset"]["mixed_index"]).resolve()
    target_cache_path = Path(config["targets"]["cache_path"]).resolve()
    roi_artifact_path = _resolve_required_path(config, "roi_artifact")

    prepared_df = pd.read_parquet(prepared_dataset_path).sort_values("pair_id").reset_index(drop=True)
    target_cache = pd.read_parquet(target_cache_path).sort_values("pair_id").reset_index(drop=True)
    roi_df = pd.read_parquet(roi_artifact_path).sort_values("pair_id").reset_index(drop=True)
    slice_summary = _validate_fixed_slice(prepared_df, config)

    expected_rows = int(config.get("public_nod.pair_rows", 3600))
    for name, df in {
        "target cache": target_cache,
        "ROI artifact": roi_df,
    }.items():
        if len(df) != expected_rows or not bool(df["pair_id"].is_unique):
            raise ValueError(f"NOD trainer preflight requires the fixed {expected_rows}-row unique-pair {name}.")

    prepared_ids = prepared_df["pair_id"].astype(int).tolist()
    if prepared_ids != target_cache["pair_id"].astype(int).tolist():
        raise ValueError("NOD trainer preflight found pair_id drift between the prepared dataset and target cache.")
    if prepared_ids != roi_df["pair_id"].astype(int).tolist():
        raise ValueError("NOD trainer preflight found pair_id drift between the prepared dataset and ROI artifact.")

    train_ds, val_ds, test_ds, _, roi_summary, target_summary = build_datasets(config)
    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    train_batch = next(iter(train_loader))
    runtime_device = resolve_runtime_device(config["training"].get("device", "cpu"))
    model = instantiate_model_from_dataset(config, train_ds).to(runtime_device)

    with torch.no_grad():
        outputs = model(train_batch.to_device(runtime_device))

    target_report = _load_json(target_cache_report_path)
    join_report = _load_json(join_report_path)
    roi_report = _load_json(roi_report_path)
    prepared_report = _load_json(prepared_report_path)
    roi_feature_dims = {name: int(values.shape[-1]) for name, values in train_batch.roi_features.items()}

    report = {
        "config": str(Path(config_path).resolve()),
        "artifacts": {
            "prepared_dataset": str(prepared_dataset_path),
            "target_cache": str(target_cache_path),
            "roi_artifact": str(roi_artifact_path),
            "join_report": str(join_report_path),
            "roi_report": str(roi_report_path),
            "prepared_report": str(prepared_report_path),
            "target_cache_report": str(target_cache_report_path),
        },
        "fixed_contract": {
            "dataset_id": config.get("public_nod.dataset_id"),
            "lane": config.get("public_nod.lane"),
            "task": config.get("public_nod.task"),
            "subjects": list(config.get("public_nod.subjects", [])),
            "sessions": list(config.get("public_nod.sessions", [])),
            "run": int(config.get("public_nod.run", 10)),
            "adapter_rows": int(config.get("public_nod.adapter_rows", 36)),
            "pair_rows": expected_rows,
        },
        "prepared_dataset": slice_summary,
        "target_cache": {
            "rows": int(len(target_cache)),
            "unique_pair_ids": int(target_cache["pair_id"].nunique()),
            "embedding_model_id": target_report["embedding_model_id"],
            "embedding_dimension": int(target_report["embedding_dimension"]),
            "embedding_column": target_report["embedding_column"],
        },
        "roi_artifact": {
            "rows": int(len(roi_df)),
            "unique_pair_ids": int(roi_df["pair_id"].nunique()),
            "roi_feature_dimensions": roi_report["roi_feature_dimensions"],
            "excluded_subject_specific_features": roi_report.get("excluded_subject_specific_features", []),
        },
        "trainer_packet": {
            "runtime_device": runtime_device,
            "train_rows": int(len(train_ds)),
            "val_rows": int(len(val_ds)),
            "test_rows": int(len(test_ds)),
            "train_batch_size": int(train_batch.condition.shape[0]),
            "roi_feature_dims": roi_feature_dims,
            "target_shape": list(train_batch.targets.clip_target_768.shape),
            "content_pred_shape": list(outputs.content_pred.shape),
            "paired_group_count": int(train_ds.capabilities.paired_group_count),
        },
        "roi_summary": roi_summary,
        "target_summary": target_summary,
        "state": {
            "join_ready": bool(join_report["state"]["join_ready"]),
            "roi_ready": bool(roi_report["state"]["roi_ready"]),
            "downstream_prep_ready": bool(prepared_report["state"]["downstream_prep_ready"]),
            "trainer_config_ready": True,
            "preflight_ready": True,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "a controlled trainer smoke run using python -m fmri2img.workflows.train_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml",
            "checked-in eval/export surfaces for the fixed NOD slice only if later needed",
            "any benchmark or evidence-facing interpretation remains out of scope for this practical Animus path",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that the fixed NOD shared-only slice is consumable by the canonical trainer path."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_public_nod_shared_only_trainer_preflight(config, config_path=args.config)
    except Exception as exc:
        report = {
            "config": str(Path(args.config).resolve()),
            "state": {
                "join_ready": False,
                "roi_ready": False,
                "downstream_prep_ready": False,
                "trainer_config_ready": False,
                "preflight_ready": False,
                "training_ready": False,
            },
            "blocked_reasons": [str(exc)],
        }
        output_path = args.output or "outputs/public_nod/preflight/imagenet_run10_shared_only/trainer_preflight.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output
    if output_path is None:
        output_path = (
            Path(config["training"]["output_dir"]).parent / "trainer_preflight.json"
        )
    write_report(output_path, report)
    print(f"Trainer preflight ready: {report['state']['preflight_ready']}")
    print(f"Report: {output_path}")
    print(json.dumps(json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
