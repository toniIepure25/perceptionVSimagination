import pandas as pd
import pytest
import torch

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
