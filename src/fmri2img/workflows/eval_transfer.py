from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.eval_transfer")

import numpy as np
import pandas as pd

from fmri2img.evaluation import collect_predictions, compute_decoder_metrics, compute_pair_metrics
from fmri2img.evaluation.decoder import _json_safe
from fmri2img.training.canonical import load_canonical_checkpoint
from fmri2img.workflows.common import (
    build_datasets,
    build_loaders,
    instantiate_model_from_dataset,
    load_workflow_config,
    resolve_runtime_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate perception-to-imagery transfer for the canonical decoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    runtime_device = resolve_runtime_device(config["training"].get("device", "cpu"))
    train_ds, val_ds, test_ds, _, _, _ = build_datasets(config)
    _, _, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    load_canonical_checkpoint(model, args.checkpoint, map_location=runtime_device, device=runtime_device)
    bundle = collect_predictions(model, test_loader, device=runtime_device)
    metrics = compute_decoder_metrics(bundle)
    pred_norm = bundle["pred"] / (np.linalg.norm(bundle["pred"], axis=1, keepdims=True) + 1e-8)
    target_norm = bundle["target"] / (np.linalg.norm(bundle["target"], axis=1, keepdims=True) + 1e-8)
    per_trial = pd.DataFrame(
        {
            "nsd_id": bundle["nsd_id"],
            "pair_id": bundle["pair_id"],
            "condition": ["perception" if value == 0 else "imagery" for value in bundle["condition"]],
            "cosine": np.sum(pred_norm * target_norm, axis=1),
        }
    )
    pair_metrics = compute_pair_metrics(per_trial)
    metrics["pair_metrics"] = pair_metrics
    output_dir = Path(config["evaluation"].get("transfer_output_dir", "outputs/canonical/transfer"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "transfer_metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    per_trial.to_csv(output_dir / "per_trial_pairs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
