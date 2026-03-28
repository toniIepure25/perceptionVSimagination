from pathlib import Path

import pandas as pd
import pytest

from fmri2img.roi import ROIGroupResolver, ROIGroupSpec
from fmri2img.utils.config_loader import load_config
from fmri2img.data.canonical import build_mixed_condition_index
from fmri2img.workflows.common import build_datasets, load_workflow_config, validate_canonical_workflow_config


def test_canonical_config_schema(canonical_fixture_dir):
    config = load_config(canonical_fixture_dir["config_path"])
    for section in (
        "dataset",
        "preprocessing",
        "roi",
        "targets",
        "model",
        "training",
        "evaluation",
        "analysis",
        "export",
    ):
        assert section in config


def test_roi_group_resolution():
    resolver = ROIGroupResolver(
        ROIGroupSpec(
            groups={
                "early_visual": ["V1", "V2", "V3"],
                "ventral_visual": ["V4", "LO", "FFA", "PPA"],
                "metacognitive": ["mPFC", "Precuneus", "IPS"],
            },
            missing_policy="error",
        )
    )
    resolved = resolver.resolve(
        ["lh.V1v", "lh.V2v", "lh.V3v", "rh.V4", "LO", "FFA-1", "PPA", "mPFC", "Precuneus", "IPS"]
    )
    assert set(resolved) == {"early_visual", "ventral_visual", "metacognitive"}
    assert resolved["early_visual"].input_dim >= 3


def test_roi_alias_resolution_does_not_confuse_v1_with_v10():
    resolver = ROIGroupResolver(ROIGroupSpec(groups={"early_visual": ["V1"]}, missing_policy="error"))
    resolved = resolver.resolve(["V10", "V1d"])
    assert resolved["early_visual"].roi_names == ("V1d",)


def test_workflow_config_resolves_repo_relative_paths():
    config = load_workflow_config("configs/canonical/shared_private_smoke.yaml")
    assert Path(config["dataset"]["mixed_index"]).exists()
    assert Path(config["targets"]["cache_path"]).exists()


def test_validate_canonical_workflow_config_surfaces_missing_inputs(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset:
  mixed_index: missing.parquet
roi:
  groups:
    early_visual: [V1]
targets:
  name: vit_l14_image_768
  dimension: 768
  cache_path: missing_targets.parquet
model: {}
training: {}
evaluation: {}
analysis: {}
export: {}
"""
    )
    config = load_workflow_config(str(config_path))
    with pytest.raises(FileNotFoundError):
        validate_canonical_workflow_config(config)


def test_malformed_nsdid_is_rejected(canonical_fixture_dir, tmp_path):
    df = pd.read_parquet(canonical_fixture_dir["mixed_index"])
    df.loc[0, "nsdId"] = None
    broken = tmp_path / "broken.parquet"
    df.to_parquet(broken, index=False)
    config = load_config(str(canonical_fixture_dir["config_path"]), overrides={"dataset.mixed_index": str(broken)})
    with pytest.raises(ValueError, match="missing nsdId"):
        build_datasets(config)


def test_build_mixed_condition_index_reassigns_splits_jointly(tmp_path):
    perception = pd.DataFrame(
        [
            {"subject": "subj01", "condition": "perception", "nsdId": 1, "pair_id": 1, "split": "train", "beta_path": "/tmp/a.nii.gz", "beta_index": 0},
            {"subject": "subj01", "condition": "perception", "nsdId": 2, "pair_id": 2, "split": "val", "beta_path": "/tmp/a.nii.gz", "beta_index": 1},
        ]
    )
    imagery = pd.DataFrame(
        [
            {"subject": "subj01", "condition": "imagery", "nsdId": 1, "pair_id": 1, "split": "test", "fmri_path": "/tmp/b.nii.gz"},
            {"subject": "subj01", "condition": "imagery", "nsdId": 2, "pair_id": 2, "split": "train", "fmri_path": "/tmp/c.nii.gz"},
        ]
    )
    perception_path = tmp_path / "perception.parquet"
    imagery_path = tmp_path / "imagery.parquet"
    perception.to_parquet(perception_path, index=False)
    imagery.to_parquet(imagery_path, index=False)

    mixed = build_mixed_condition_index(perception_path, imagery_path)
    per_pair_splits = mixed.groupby("pair_id")["split"].nunique().to_dict()
    assert per_pair_splits == {1: 1, 2: 1}


def test_build_mixed_condition_index_filters_noncanonical_imagery_conditions(tmp_path):
    perception = pd.DataFrame(
        [
            {"subject": "subj01", "condition": "perception", "nsdId": 1, "pair_id": 1, "beta_path": "/tmp/a.nii.gz", "beta_index": 0},
        ]
    )
    imagery = pd.DataFrame(
        [
            {"subject": "subj01", "condition": "imagery", "nsdId": 1, "pair_id": 1, "fmri_path": "/tmp/b.nii.gz"},
            {"subject": "subj01", "condition": "attention", "nsdId": 1, "pair_id": 1, "fmri_path": "/tmp/c.nii.gz"},
            {"subject": "subj01", "condition": "perception", "nsdId": 1, "pair_id": 1, "fmri_path": "/tmp/d.nii.gz"},
        ]
    )
    perception_path = tmp_path / "perception.parquet"
    imagery_path = tmp_path / "imagery.parquet"
    perception.to_parquet(perception_path, index=False)
    imagery.to_parquet(imagery_path, index=False)

    mixed = build_mixed_condition_index(
        perception_path,
        imagery_path,
        perception_conditions=["perception"],
        imagery_conditions=["imagery"],
    )

    assert mixed["condition"].value_counts().to_dict() == {"perception": 1, "imagery": 1}
