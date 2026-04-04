from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.export_for_animus")

from fmri2img.export import export_decoder_bundle
from fmri2img.training.canonical import inspect_canonical_checkpoint
from fmri2img.workflows.common import build_datasets, checkpoint_artifact_spec, load_workflow_config


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
