import pandas as pd
import pytest
import torch
import json

from fmri2img.data.canonical import CanonicalDecoderDataset, PairedConditionBatchSampler, decoder_collate_fn
from fmri2img.models.canonical_decoder import SharedPrivateMultitaskDecoder
from fmri2img.targets import LatentTargetStore
from fmri2img.workflows.common import build_datasets, build_loaders, instantiate_model_from_dataset, load_workflow_config


def test_canonical_dataset_pairing_and_capabilities(canonical_fixture_dir):
    config = load_workflow_config(str(canonical_fixture_dir["config_path"]))
    train_ds, val_ds, test_ds, _, _, _ = build_datasets(config)
    assert train_ds.capabilities.has_pairing
    assert train_ds.capabilities.has_vividness
    assert train_ds.capabilities.has_confidence
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0


def test_canonical_forward_shapes(canonical_fixture_dir):
    config = load_workflow_config(str(canonical_fixture_dir["config_path"]))
    train_ds, _, _, _, _, _ = build_datasets(config)
    model = instantiate_model_from_dataset(config, train_ds)
    samples = [train_ds[0], train_ds[1]]
    batch = decoder_collate_fn(samples)
    outputs = model(batch)
    assert outputs.z_shared.shape == (2, config["model"]["shared_dim"])
    assert outputs.z_perception_private.shape == (2, config["model"]["private_dim"])
    assert outputs.z_imagery_private.shape == (2, config["model"]["private_dim"])
    assert outputs.content_pred.shape == (2, config["targets"]["dimension"])
    assert outputs.domain_logits.shape == (2, 2)
    assert outputs.vividness_pred.shape == (2,)
    assert outputs.confidence_pred.shape == (2,)


def test_canonical_batch_without_vividness_still_works(canonical_fixture_dir):
    df = pd.read_parquet(canonical_fixture_dir["mixed_index"])
    df["vividness"] = None
    no_vivid_path = canonical_fixture_dir["root"] / "no_vivid.parquet"
    df.to_parquet(no_vivid_path, index=False)
    store = LatentTargetStore(canonical_fixture_dir["targets_path"])

    def resolver(_fmri, row):
        import json
        import numpy as np

        raw = json.loads(row["roi_features_json"])
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    dataset = CanonicalDecoderDataset(no_vivid_path, target_store=store, roi_feature_resolver=resolver, split="train")
    batch = decoder_collate_fn([dataset[0], dataset[1]])
    model = SharedPrivateMultitaskDecoder(
        roi_input_dims={name: len(value) for name, value in dataset[0]["roi_features"].items()}
    )
    outputs = model(batch)
    assert batch.targets.vividness is None
    assert outputs.vividness_pred is not None


def test_partial_vividness_labels_are_masked_not_dropped(canonical_fixture_dir):
    df = pd.read_parquet(canonical_fixture_dir["mixed_index"])
    df.loc[df.index[0], "vividness"] = None
    partial_path = canonical_fixture_dir["root"] / "partial_vivid.parquet"
    df.to_parquet(partial_path, index=False)
    store = LatentTargetStore(canonical_fixture_dir["targets_path"])

    def resolver(_fmri, row):
        import json
        import numpy as np

        raw = json.loads(row["roi_features_json"])
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    dataset = CanonicalDecoderDataset(partial_path, target_store=store, roi_feature_resolver=resolver, split="train")
    batch = decoder_collate_fn([dataset[0], dataset[1]])
    assert batch.targets.vividness is not None
    assert torch.isnan(batch.targets.vividness).sum() == 1


def test_pair_sampler_requires_true_cross_condition_pairs(canonical_fixture_dir, tmp_path):
    df = pd.read_parquet(canonical_fixture_dir["mixed_index"])
    df["pair_id"] = range(len(df))
    broken_path = tmp_path / "unpaired.parquet"
    df.to_parquet(broken_path, index=False)
    store = LatentTargetStore(canonical_fixture_dir["targets_path"])

    def resolver(_fmri, row):
        import json
        import numpy as np

        raw = json.loads(row["roi_features_json"])
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    dataset = CanonicalDecoderDataset(broken_path, target_store=store, roi_feature_resolver=resolver, split="train")
    assert not dataset.capabilities.has_pairing
    with pytest.raises(ValueError, match="requires at least one pair_id"):
        PairedConditionBatchSampler(dataset, batch_size=2)


def test_decoder_collate_drops_raw_fmri_when_shapes_differ_but_keeps_roi_features():
    samples = [
        {
            "fmri": torch.arange(12, dtype=torch.float32).numpy(),
            "roi_features": {
                "early_visual": torch.tensor([1.0, 2.0, 3.0]).numpy(),
                "ventral_visual": torch.tensor([4.0, 5.0, 6.0, 7.0]).numpy(),
                "metacognitive": torch.tensor([8.0, 9.0]).numpy(),
            },
            "condition": "perception",
            "nsd_id": 1,
            "pair_id": 1,
            "clip_target_768": torch.zeros(768).numpy(),
            "vividness": None,
            "confidence": None,
            "metadata": {"subject": "subj02"},
        },
        {
            "fmri": torch.arange(15, dtype=torch.float32).numpy(),
            "roi_features": {
                "early_visual": torch.tensor([1.5, 2.5, 3.5]).numpy(),
                "ventral_visual": torch.tensor([4.5, 5.5, 6.5, 7.5]).numpy(),
                "metacognitive": torch.tensor([8.5, 9.5]).numpy(),
            },
            "condition": "imagery",
            "nsd_id": 1,
            "pair_id": 1,
            "clip_target_768": torch.ones(768).numpy(),
            "vividness": None,
            "confidence": None,
            "metadata": {"subject": "subj03"},
        },
    ]

    batch = decoder_collate_fn(samples)

    assert batch.fmri is None
    assert batch.roi_features["early_visual"].shape == (2, 3)
    assert batch.targets.clip_target_768.shape == (2, 768)


def test_decoder_collate_preserves_raw_fmri_when_shapes_match():
    samples = [
        {
            "fmri": torch.arange(12, dtype=torch.float32).numpy(),
            "roi_features": {
                "early_visual": torch.tensor([1.0, 2.0, 3.0]).numpy(),
                "ventral_visual": torch.tensor([4.0, 5.0, 6.0, 7.0]).numpy(),
                "metacognitive": torch.tensor([8.0, 9.0]).numpy(),
            },
            "condition": "perception",
            "nsd_id": 1,
            "pair_id": 1,
            "clip_target_768": torch.zeros(768).numpy(),
            "vividness": None,
            "confidence": None,
            "metadata": {"subject": "subj01"},
        },
        {
            "fmri": (torch.arange(12, dtype=torch.float32) + 1).numpy(),
            "roi_features": {
                "early_visual": torch.tensor([1.5, 2.5, 3.5]).numpy(),
                "ventral_visual": torch.tensor([4.5, 5.5, 6.5, 7.5]).numpy(),
                "metacognitive": torch.tensor([8.5, 9.5]).numpy(),
            },
            "condition": "imagery",
            "nsd_id": 1,
            "pair_id": 1,
            "clip_target_768": torch.ones(768).numpy(),
            "vividness": None,
            "confidence": None,
            "metadata": {"subject": "subj01"},
        },
    ]

    batch = decoder_collate_fn(samples)

    assert batch.fmri is not None
    assert batch.fmri.shape == (2, 12)


def test_materialized_roi_features_do_not_require_raw_fmri_paths(tmp_path):
    mixed_index = tmp_path / "mixed.parquet"
    targets_path = tmp_path / "targets.parquet"
    df = pd.DataFrame(
        [
            {
                "subject": "subj02",
                "condition": "perception",
                "nsdId": 1001,
                "pair_id": 1001,
                "split": "train",
                "roi_features_json": json.dumps(
                    {
                        "early_visual": [0.1, 0.2],
                        "ventral_visual": [0.3, 0.4, 0.5],
                        "metacognitive": [0.6],
                    }
                ),
            }
        ]
    )
    df.to_parquet(mixed_index, index=False)
    pd.DataFrame([{"nsdId": 1001, "clip_target_768": [0.0] * 768}]).to_parquet(targets_path, index=False)
    store = LatentTargetStore(targets_path)

    def resolver(_fmri, row):
        import json
        import numpy as np

        raw = json.loads(row["roi_features_json"])
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    dataset = CanonicalDecoderDataset(mixed_index, target_store=store, roi_feature_resolver=resolver, split="train")
    sample = dataset[0]

    assert sample["fmri"] is None
    assert set(sample["roi_features"]) == {"early_visual", "ventral_visual", "metacognitive"}


def test_shared_only_ablation_disables_domain_head_and_zeros_private_latents(canonical_fixture_dir):
    config = load_workflow_config(
        str(canonical_fixture_dir["config_path"]),
        ["model.disentanglement_mode=\"shared_only\"", "model.use_domain_head=true"],
    )
    train_ds, _, _, _, _, _ = build_datasets(config)
    model = instantiate_model_from_dataset(config, train_ds)
    samples = [train_ds[0], train_ds[1]]
    batch = decoder_collate_fn(samples)
    outputs = model(batch)

    assert model.config.disentanglement_mode == "shared_only"
    assert model.config.use_domain_head is False
    assert outputs.domain_logits is None
    assert torch.count_nonzero(outputs.z_perception_private) == 0
    assert torch.count_nonzero(outputs.z_imagery_private) == 0
