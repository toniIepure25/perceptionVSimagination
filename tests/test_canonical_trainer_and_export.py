import json

import pytest
import torch

from fmri2img.evaluation import collect_predictions, compute_decoder_metrics, compute_roi_summary
from fmri2img.export import export_decoder_bundle
from fmri2img.training import CanonicalLossWeights, SharedPrivateTrainer, load_canonical_checkpoint
from fmri2img.workflows.common import (
    build_datasets,
    build_loaders,
    instantiate_model_from_dataset,
    load_workflow_config,
    resolve_runtime_device,
)


def test_trainer_fit_and_eval_bundle(canonical_fixture_dir):
    config = load_workflow_config(str(canonical_fixture_dir["config_path"]))
    train_ds, val_ds, test_ds, _, roi_summary, target_summary = build_datasets(config)
    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    trainer = SharedPrivateTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_weights=CanonicalLossWeights(),
        device="cpu",
    )
    best = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        output_dir=config["training"]["output_dir"],
        config_snapshot=config.to_dict(),
    )
    assert "val_content_cosine" in best
    assert -1.0 <= best["val_content_cosine"] <= 1.0
    checkpoint = canonical_fixture_dir["root"] / "train_outputs" / "best_decoder.pt"
    assert checkpoint.exists()

    original_to = model.to
    seen_devices = []

    def tracking_to(device, *args, **kwargs):
        seen_devices.append(str(device))
        return original_to(device, *args, **kwargs)

    model.to = tracking_to  # type: ignore[assignment]
    bundle = collect_predictions(model, test_loader, device="cpu")
    assert "cpu" in seen_devices
    metrics = compute_decoder_metrics(bundle)
    roi_table = compute_roi_summary(bundle)
    assert metrics["target_space"] == "vit_l14_image_768"
    assert isinstance(roi_table, list)

    export_dir = canonical_fixture_dir["root"] / "manual_export"
    export_decoder_bundle(
        output_dir=export_dir,
        checkpoint_path=checkpoint,
        artifact_spec={
            "artifact_version": "1.0",
            "target_spec": target_summary,
            "preprocessing_spec": {"enabled": False},
            "roi_spec": {"resolved": roi_summary},
            "checkpoint_path": str(checkpoint),
            "metadata": {"project": "fmri2img"},
        },
    )
    manifest = json.loads((export_dir / "manifest.json").read_text())
    assert manifest["artifact_version"] == "1.0"
    assert manifest["target_spec"]["dimension"] == 768


def test_target_dimension_mismatch_is_rejected(canonical_fixture_dir, tmp_path):
    rows = [{"nsdId": 101, "clip_target_768": [0.0] * 512}]
    wrong_targets = tmp_path / "wrong_targets.parquet"
    import pandas as pd

    pd.DataFrame(rows).to_parquet(wrong_targets, index=False)
    config = load_workflow_config(
        str(canonical_fixture_dir["config_path"]),
        [f"targets.cache_path={json.dumps(str(wrong_targets))}"],
    )
    with pytest.raises(ValueError, match="Target dimension mismatch"):
        build_datasets(config)[0][0]


def test_checkpoint_reload_detects_model_mismatch(canonical_fixture_dir):
    config = load_workflow_config(str(canonical_fixture_dir["config_path"]))
    train_ds, val_ds, test_ds, _, _, _ = build_datasets(config)
    train_loader, val_loader, _ = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    trainer = SharedPrivateTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_weights=CanonicalLossWeights(),
        device="cpu",
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        output_dir=config["training"]["output_dir"],
        config_snapshot={
            "model": {
                "target_dim": config["targets"]["dimension"],
                "branch_embedding_dim": config["model"]["branch_embedding_dim"],
                "shared_dim": config["model"]["shared_dim"],
                "private_dim": config["model"]["private_dim"],
                "use_domain_head": True,
                "use_vividness_head": True,
            }
        },
    )
    checkpoint = canonical_fixture_dir["root"] / "train_outputs" / "best_decoder.pt"
    mismatched_model = instantiate_model_from_dataset(
        load_workflow_config(str(canonical_fixture_dir["config_path"]), ["model.private_dim=16"]),
        train_ds,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_canonical_checkpoint(mismatched_model, checkpoint)


def test_export_manifest_requires_canonical_keys(canonical_fixture_dir, tmp_path):
    checkpoint = tmp_path / "dummy.pt"
    torch.save({"state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="missing keys"):
        export_decoder_bundle(
            output_dir=tmp_path / "export",
            checkpoint_path=checkpoint,
            artifact_spec={"artifact_version": "1.0"},
        )


def test_resolve_runtime_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_runtime_device("cuda") == "cpu"
