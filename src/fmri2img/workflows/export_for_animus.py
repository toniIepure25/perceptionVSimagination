from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.export_for_animus")

from fmri2img.evaluation import normalize_condition_semantics_payload
from fmri2img.export import export_decoder_bundle
from fmri2img.training.canonical import inspect_canonical_checkpoint
from fmri2img.workflows.common import build_datasets, checkpoint_artifact_spec, load_workflow_config


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _resolve_export_condition_semantics(config) -> dict | None:
    eval_output_dir = Path(config["evaluation"].get("output_dir", "outputs/canonical/eval"))
    transfer_output_dir = Path(config["evaluation"].get("transfer_output_dir", "outputs/canonical/transfer"))
    eval_metrics = _load_optional_json(eval_output_dir / "metrics.json")
    transfer_metrics = _load_optional_json(transfer_output_dir / "transfer_metrics.json")
    normalized_eval = normalize_condition_semantics_payload(eval_metrics)
    normalized_transfer = normalize_condition_semantics_payload(transfer_metrics)
    informative = [
        item
        for item in (normalized_eval, normalized_transfer)
        if item["present_conditions"] or item["pair_metrics_available_from_payload"] is not None
    ]
    if not informative:
        return None
    shared = informative[0]
    if any(item != shared for item in informative[1:]):
        return {
            "present_conditions": shared["present_conditions"],
            "missing_conditions": shared["missing_conditions"],
            "paired_metrics_available": shared["paired_metrics_available"],
            "paired_metrics_reason": shared["paired_metrics_reason"],
            "pair_metrics_available_from_payload": shared["pair_metrics_available_from_payload"],
            "source": "eval_transfer_artifacts",
            "consistent_across_eval_and_transfer": False,
        }
    return {
        **shared,
        "source": "eval_transfer_artifacts",
        "consistent_across_eval_and_transfer": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export canonical decoder artifacts for Animus.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    config = load_workflow_config(args.config, args.override)
    _, _, _, _, roi_summary, target_summary = build_datasets(config)
    checkpoint_metadata = inspect_canonical_checkpoint(args.checkpoint, map_location="cpu")
    effective_config = checkpoint_metadata.get("config", config.to_dict())
    artifact_spec = checkpoint_artifact_spec(
        config,
        args.checkpoint,
        target_summary,
        roi_summary,
        effective_config=effective_config,
    )
    condition_semantics = _resolve_export_condition_semantics(config)
    if condition_semantics is not None:
        artifact_spec.setdefault("metadata", {})["condition_semantics"] = condition_semantics
    output_dir = Path(config["export"].get("output_dir", "outputs/canonical/export"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_dir / "config_snapshot.json"
    with open(config_snapshot_path, "w") as f:
        json.dump(effective_config, f, indent=2)
    export_decoder_bundle(
        output_dir,
        args.checkpoint,
        artifact_spec,
        extra_files={"config_snapshot": config_snapshot_path},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
