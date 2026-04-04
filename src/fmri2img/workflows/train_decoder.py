from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.train_decoder")

import numpy as np
import torch
from fmri2img.training.canonical import CanonicalLossWeights, SharedPrivateTrainer
from fmri2img.workflows.common import (
    build_datasets,
    build_loaders,
    instantiate_model_from_dataset,
    load_workflow_config,
    resolve_runtime_device,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the canonical shared/private decoder.")
    parser.add_argument("--config", required=True, help="Path to canonical workflow config.")
    parser.add_argument("--override", action="append", default=[], help="Config override KEY=VALUE")
    args = parser.parse_args(argv)

    config = load_workflow_config(args.config, args.override)
    runtime_device = resolve_runtime_device(config["training"].get("device", "cpu"))
    train_ds, val_ds, test_ds, _, roi_summary, target_summary = build_datasets(config)
    train_loader, val_loader, _ = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    loss_weights = CanonicalLossWeights(
        domain=float(config["training"].get("domain_weight", 0.1)),
        vividness=float(config["training"].get("vividness_weight", 0.1)),
        confidence=float(config["training"].get("confidence_weight", 0.05)),
        reconstruction=float(config["training"].get("reconstruction_weight", 0.1)),
    )
    trainer = SharedPrivateTrainer(
        model=model,
        optimizer=optimizer,
        loss_weights=loss_weights,
        device=runtime_device,
    )
    output_dir = Path(config["training"].get("output_dir", "outputs/canonical/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = config.to_dict()
    config_snapshot.setdefault("model", {})
    model_snapshot = _json_safe(asdict(model.config))
    config_snapshot["model"].update(model_snapshot)
    config_snapshot["dataset_capabilities"] = {
        "has_pairing": bool(train_ds.capabilities.has_pairing),
        "paired_group_count": train_ds.capabilities.paired_group_count,
        "has_vividness": bool(train_ds.capabilities.has_vividness),
        "has_confidence": bool(train_ds.capabilities.has_confidence),
    }
    config_snapshot["runtime"] = {
        "requested_device": config["training"].get("device", "cpu"),
        "resolved_device": runtime_device,
    }
    with open(output_dir / "config_snapshot.json", "w") as f:
        json.dump(_json_safe(config_snapshot), f, indent=2)
    with open(output_dir / "roi_summary.json", "w") as f:
        json.dump(roi_summary, f, indent=2)
    with open(output_dir / "target_summary.json", "w") as f:
        json.dump(target_summary, f, indent=2)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["training"].get("epochs", 5)),
        output_dir=output_dir,
        config_snapshot=_json_safe(config_snapshot),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
