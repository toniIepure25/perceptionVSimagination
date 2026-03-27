from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def export_decoder_bundle(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    artifact_spec: dict[str, Any],
    extra_files: dict[str, str | Path] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot export missing checkpoint: {checkpoint_path}")
    export_checkpoint = output_dir / checkpoint_path.name
    if checkpoint_path.resolve() != export_checkpoint.resolve():
        shutil.copy2(checkpoint_path, export_checkpoint)
    manifest = dict(artifact_spec)
    manifest["checkpoint"] = export_checkpoint.name
    if extra_files:
        manifest["extra_files"] = {}
        for key, value in extra_files.items():
            value = Path(value)
            target = output_dir / value.name
            if value.exists() and value.resolve() != target.resolve():
                shutil.copy2(value, target)
            manifest["extra_files"][key] = target.name
    required_manifest_keys = {"artifact_version", "target_spec", "preprocessing_spec", "roi_spec", "metadata"}
    missing_keys = required_manifest_keys - set(manifest)
    if missing_keys:
        raise ValueError(f"Animus export manifest is missing keys: {sorted(missing_keys)}")
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return output_dir
