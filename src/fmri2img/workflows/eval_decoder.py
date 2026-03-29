from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.evaluation import (
    collect_predictions,
    compute_decoder_metrics,
    compute_roi_summary,
    write_evaluation_bundle,
)
from fmri2img.training.canonical import load_canonical_checkpoint
from fmri2img.workflows.common import (
    build_datasets,
    build_loaders,
    instantiate_model_from_dataset,
    load_workflow_config,
    resolve_runtime_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the canonical shared/private decoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    runtime_device = resolve_runtime_device(config["training"].get("device", "cpu"))
    train_ds, val_ds, test_ds, _, roi_summary, _ = build_datasets(config)
    _, _, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    load_canonical_checkpoint(model, args.checkpoint, map_location=runtime_device, device=runtime_device)
    bundle = collect_predictions(model, test_loader, device=runtime_device)
    metrics = compute_decoder_metrics(bundle)
    output_dir = Path(config["evaluation"].get("output_dir", "outputs/canonical/eval"))
    write_evaluation_bundle(output_dir, metrics, compute_roi_summary(bundle))
    with open(output_dir / "resolved_roi_groups.json", "w") as f:
        json.dump(roi_summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
