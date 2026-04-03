import json
from pathlib import Path

import pytest
import torch
import yaml
import numpy as np
import pandas as pd

from fmri2img.evaluation import collect_predictions, compute_decoder_metrics, compute_roi_summary
from fmri2img.export import export_decoder_bundle
from fmri2img.training import CanonicalLossWeights, SharedPrivateTrainer, load_canonical_checkpoint
from fmri2img.workflows.common import (
    build_datasets,
    build_loaders,
    checkpoint_artifact_spec,
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


def test_checkpoint_artifact_spec_carries_animus_core_metadata(canonical_fixture_dir):
    config = load_workflow_config(str(canonical_fixture_dir["config_path"]))
    config["experiment"] = {
        "name": "animus_core_decoder",
        "description": "Shared-only practical decoder.",
        "benchmark_role": "canonical_neural_baseline",
        "evidence_tier": "validated",
    }
    config["animus"] = {
        "subproject": "animus_core_decoder",
        "decoder_role": "practical_content_decoder",
        "stability_tier": "current_default",
        "intended_use": "content decoding",
        "source_interface_status": "scaffolded",
        "confidence_interface_status": "scaffolded",
    }
    config["model"]["disentanglement_mode"] = "shared_only"
    config["model"]["use_domain_head"] = False
    config["model"]["use_vividness_head"] = False
    _, _, _, _, roi_summary, target_summary = build_datasets(config)
    artifact = checkpoint_artifact_spec(
        config,
        checkpoint_path="outputs/example/best_decoder.pt",
        target_spec=target_summary,
        roi_summary=roi_summary,
        effective_config=config.to_dict(),
    )
    assert artifact["metadata"]["experiment"]["name"] == "animus_core_decoder"
    assert artifact["metadata"]["animus"]["subproject"] == "animus_core_decoder"
    assert artifact["metadata"]["animus"]["decoder_role"] == "practical_content_decoder"
    assert artifact["metadata"]["animus"]["interfaces"]["content"]["status"] == "active"
    assert artifact["metadata"]["animus"]["interfaces"]["source"]["enabled"] is False
    assert artifact["metadata"]["animus"]["interfaces"]["source"]["status"] == "scaffolded"
    assert artifact["metadata"]["animus"]["interfaces"]["confidence"]["status"] == "scaffolded"
    assert artifact["metadata"]["heads"]["disentanglement"]["mode"] == "shared_only"


def test_resolve_runtime_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_runtime_device("cuda") == "cpu"


def test_trainer_supports_multisubject_roi_materialized_batches_with_unequal_raw_shapes(tmp_path):
    root = Path(tmp_path)
    fmri_dir = root / "fmri"
    fmri_dir.mkdir()

    rows = []
    target_rows = []
    target_ids = [9001, 9002, 9003]
    raw_lengths = {"subj02": 11, "subj03": 17}

    for nsd_id in target_ids:
        target = np.zeros(768, dtype=np.float32)
        target[nsd_id % 7] = 1.0
        target_rows.append({"nsdId": nsd_id, "clip_target_768": target.tolist()})

    assignments = [
        ("subj02", "perception", 9001, "train"),
        ("subj03", "imagery", 9001, "train"),
        ("subj02", "perception", 9002, "val"),
        ("subj03", "imagery", 9002, "val"),
        ("subj02", "perception", 9003, "test"),
        ("subj03", "imagery", 9003, "test"),
    ]

    for idx, (subject, condition, nsd_id, split) in enumerate(assignments):
        raw = np.linspace(0.0, 1.0, raw_lengths[subject], dtype=np.float32) + idx
        fmri_path = fmri_dir / f"{subject}_{condition}_{nsd_id}.npy"
        np.save(fmri_path, raw)
        rows.append(
            {
                "trial_id": idx,
                "subject": subject,
                "condition": condition,
                "nsdId": nsd_id,
                "pair_id": nsd_id,
                "split": split,
                "fmri_path": str(fmri_path),
                "roi_features_json": json.dumps(
                    {
                        "early_visual": [0.1 + idx, 0.2 + idx],
                        "ventral_visual": [0.3 + idx, 0.4 + idx, 0.5 + idx],
                        "metacognitive": [0.6 + idx, 0.7 + idx],
                    }
                ),
            }
        )

    mixed_index = root / "mixed.parquet"
    targets_path = root / "targets.parquet"
    pd.DataFrame(rows).to_parquet(mixed_index, index=False)
    pd.DataFrame(target_rows).to_parquet(targets_path, index=False)

    config = {
        "dataset": {
            "subject": "subj02",
            "mixed_index": str(mixed_index),
        },
        "preprocessing": {"enabled": False},
        "roi": {
            "groups": {
                "early_visual": ["V1", "V2", "V3"],
                "ventral_visual": ["V4", "LO", "FFA", "PPA"],
                "metacognitive": ["mPFC", "Precuneus", "IPS"],
            },
            "roi_names": [],
            "missing_policy": "warn",
            "fallback_policy": "full_feature_vector",
        },
        "targets": {
            "name": "vit_l14_image_768",
            "dimension": 768,
            "cache_path": str(targets_path),
            "embedding_column": "clip_target_768",
            "id_column": "nsdId",
        },
        "model": {
            "branch_embedding_dim": 8,
            "shared_dim": 8,
            "private_dim": 4,
            "dropout": 0.0,
            "use_domain_head": True,
            "use_vividness_head": False,
            "vividness_mode": "evidential",
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cpu",
            "output_dir": str(root / "train_outputs"),
            "seed": 0,
        },
        "evaluation": {
            "batch_size": 2,
            "output_dir": str(root / "eval_outputs"),
            "transfer_output_dir": str(root / "transfer_outputs"),
        },
        "analysis": {"output_dir": str(root / "analysis_outputs")},
        "export": {"output_dir": str(root / "export_outputs")},
    }
    config_path = root / "config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle)

    loaded = load_workflow_config(str(config_path))
    train_ds, val_ds, test_ds, _, _, _ = build_datasets(loaded)
    train_loader, val_loader, _ = build_loaders(loaded, train_ds, val_ds, test_ds)
    batch = next(iter(train_loader))
    assert batch.fmri is None

    model = instantiate_model_from_dataset(loaded, train_ds)
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
        output_dir=root / "train_outputs",
        config_snapshot=loaded.to_dict(),
    )
    assert "val_content_cosine" in best
    assert (root / "train_outputs" / "best_decoder.pt").exists()
