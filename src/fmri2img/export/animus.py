from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _build_decoder_card(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata", {})
    experiment = metadata.get("experiment", {})
    animus = metadata.get("animus", {})
    interfaces = animus.get("interfaces", {})
    heads = metadata.get("heads", {})
    roi_spec = manifest.get("roi_spec", {})
    resolved_roi = roi_spec.get("resolved", {}) or {}
    roi_groups = roi_spec.get("groups", {}) or {}

    return {
        "project": metadata.get("project"),
        "workflow": metadata.get("workflow"),
        "compatibility_version": metadata.get("compatibility_version"),
        "experiment": {
            "name": experiment.get("name"),
            "benchmark_role": experiment.get("benchmark_role"),
            "evidence_tier": experiment.get("evidence_tier"),
            "description": experiment.get("description"),
        },
        "animus": {
            "subproject": animus.get("subproject"),
            "decoder_role": animus.get("decoder_role"),
            "stability_tier": animus.get("stability_tier"),
            "intended_use": animus.get("intended_use"),
        },
        "interfaces": interfaces,
        "heads": heads,
        "target": {
            "name": manifest.get("target_spec", {}).get("name"),
            "dimension": manifest.get("target_spec", {}).get("dimension"),
        },
        "roi": {
            "group_names": sorted(roi_groups.keys()),
            "resolved_group_count": len(resolved_roi),
        },
        "artifacts": {
            "checkpoint": manifest.get("checkpoint"),
            "config_snapshot": manifest.get("extra_files", {}).get("config_snapshot"),
        },
    }


def _write_decoder_card(output_dir: Path, decoder_card: dict[str, Any]) -> None:
    with open(output_dir / "decoder_card.json", "w") as f:
        json.dump(decoder_card, f, indent=2)

    experiment = decoder_card.get("experiment", {})
    animus = decoder_card.get("animus", {})
    target = decoder_card.get("target", {})
    roi = decoder_card.get("roi", {})
    artifacts = decoder_card.get("artifacts", {})
    interfaces = decoder_card.get("interfaces", {})
    heads = decoder_card.get("heads", {})

    lines = [
        "# Decoder Card",
        "",
        f"- Experiment: `{experiment.get('name')}`",
        f"- Benchmark role: `{experiment.get('benchmark_role')}`",
        f"- Evidence tier: `{experiment.get('evidence_tier')}`",
        f"- Subproject: `{animus.get('subproject')}`",
        f"- Decoder role: `{animus.get('decoder_role')}`",
        f"- Stability tier: `{animus.get('stability_tier')}`",
        f"- Target: `{target.get('name')}` ({target.get('dimension')}-D)",
        f"- ROI groups: `{roi.get('resolved_group_count')}` resolved groups from `{len(roi.get('group_names', []))}` configured groups",
        f"- Checkpoint: `{artifacts.get('checkpoint')}`",
        f"- Config snapshot: `{artifacts.get('config_snapshot')}`",
        "",
        "## Interfaces",
        "",
        f"- content: `{interfaces.get('content', {}).get('status')}`",
        f"- source: `{interfaces.get('source', {}).get('status')}`",
        f"- confidence: `{interfaces.get('confidence', {}).get('status')}`",
        "",
        "## Heads",
        "",
        f"- disentanglement mode: `{heads.get('disentanglement', {}).get('mode')}`",
        f"- domain head enabled: `{heads.get('domain', {}).get('enabled')}`",
        f"- vividness head enabled: `{heads.get('vividness', {}).get('enabled')}`",
        "",
        "## Intended Use",
        "",
        str(animus.get("intended_use")),
    ]
    (output_dir / "decoder_card.md").write_text("\n".join(lines) + "\n")


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
    _write_decoder_card(output_dir, _build_decoder_card(manifest))
    return output_dir
