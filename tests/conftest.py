"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import yaml
import nibabel as nib


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output


@pytest.fixture
def sample_fmri():
    """Generate sample fMRI data for testing."""
    # 10 samples x 1000 voxels
    return np.random.randn(10, 1000).astype(np.float32)


@pytest.fixture
def sample_clip_embeddings():
    """Generate sample CLIP embeddings for testing."""
    # 10 samples x 768 dimensions (ViT-L/14)
    embeddings = np.random.randn(10, 768).astype(np.float32)
    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return embeddings


@pytest.fixture
def sample_nsd_ids():
    """Generate sample NSD IDs for testing."""
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture
def mock_encoding_model():
    """Mock encoding model for testing."""
    class MockEncodingModel:
        def __init__(self):
            self.n_voxels = 1000
            
        def predict(self, images):
            """Generate fake fMRI predictions."""
            batch_size = len(images) if isinstance(images, list) else images.shape[0]
            return np.random.randn(batch_size, self.n_voxels).astype(np.float32)
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
    
    return MockEncodingModel()


@pytest.fixture
def canonical_fixture_dir(tmp_path):
    """Build a tiny paired perception/imagery fixture for canonical workflow tests."""
    root = tmp_path / "canonical_fixture"
    root.mkdir()

    data_dir = root / "fmri"
    data_dir.mkdir()

    rows = []
    target_rows = []
    rng = np.random.default_rng(0)
    nsd_ids = [101, 102, 103, 104]
    conditions = ["perception", "imagery"]

    for pair_idx, nsd_id in enumerate(nsd_ids):
        target = rng.standard_normal(768).astype(np.float32)
        target = target / (np.linalg.norm(target) + 1e-8)
        target_rows.append({"nsdId": nsd_id, "clip_target_768": target.tolist()})
        for cond_idx, condition in enumerate(conditions):
            fmri = rng.standard_normal(12).astype(np.float32)
            fmri_path = data_dir / f"{condition}_{nsd_id}.npy"
            np.save(fmri_path, fmri)
            roi_features = {
                "early_visual": fmri[:3].tolist(),
                "ventral_visual": fmri[3:7].tolist(),
                "metacognitive": fmri[7:10].tolist(),
            }
            rows.append(
                {
                    "trial_id": len(rows),
                    "subject": "subj01",
                    "condition": condition,
                    "nsdId": nsd_id,
                    "nsd_id": nsd_id,
                    "pair_id": nsd_id,
                    "split": "train" if pair_idx < 2 else ("val" if pair_idx == 2 else "test"),
                    "fmri_path": str(fmri_path),
                    "roi_features_json": json_dump(roi_features),
                    "vividness": float(0.2 * (pair_idx + cond_idx + 1)),
                    "confidence": float(0.5 + 0.1 * cond_idx),
                }
            )

    mixed_index = root / "mixed.parquet"
    pd.DataFrame(rows).to_parquet(mixed_index, index=False)
    targets_path = root / "targets.parquet"
    pd.DataFrame(target_rows).to_parquet(targets_path, index=False)

    config = {
        "dataset": {
            "subject": "subj01",
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
            "branch_embedding_dim": 16,
            "shared_dim": 16,
            "private_dim": 8,
            "dropout": 0.0,
            "use_domain_head": True,
            "use_vividness_head": True,
            "vividness_mode": "evidential",
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "domain_weight": 0.1,
            "vividness_weight": 0.1,
            "reconstruction_weight": 0.1,
            "device": "cpu",
            "output_dir": str(root / "train_outputs"),
            "seed": 0,
        },
        "evaluation": {
            "batch_size": 4,
            "output_dir": str(root / "eval_outputs"),
            "transfer_output_dir": str(root / "transfer_outputs"),
        },
        "analysis": {"output_dir": str(root / "analysis_outputs")},
        "export": {"output_dir": str(root / "export_outputs")},
    }
    config_path = root / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return {
        "root": root,
        "config_path": config_path,
        "mixed_index": mixed_index,
        "targets_path": targets_path,
    }


def json_dump(value):
    import json

    return json.dumps(value)


@pytest.fixture
def canonical_volume_fixture(tmp_path):
    """Create a tiny volumetric fixture for canonical artifact prep and bootstrap tests."""
    root = tmp_path / "canonical_volume_fixture"
    root.mkdir()

    subject = "subj01"
    shape = (4, 4, 4)
    rng = np.random.default_rng(42)

    # Perception 4D NIfTI referenced via beta_path
    perception_dir = root / "perception"
    perception_dir.mkdir()
    perception_data = rng.standard_normal(shape + (4,)).astype(np.float32)
    perception_nii = perception_dir / "betas_session01.nii.gz"
    nib.save(nib.Nifti1Image(perception_data, np.eye(4)), perception_nii)

    perception_rows = []
    nsd_ids = [2001, 2002, 2003, 2004]
    for idx, nsd_id in enumerate(nsd_ids):
        perception_rows.append(
            {
                "subject": subject,
                "session": 1,
                "trial_in_session": idx,
                "global_trial_index": idx,
                "nsdId": nsd_id,
                "beta_path": str(perception_nii.resolve()),
                "beta_index": idx,
            }
        )
    raw_perception_index = root / "perception_raw.parquet"
    pd.DataFrame(perception_rows).to_parquet(raw_perception_index, index=False)

    # Imagery dataset root with one NIfTI per trial plus metadata.json
    imagery_root = root / "imagery_raw"
    imagery_run = imagery_root / subject / "imagery" / "run01"
    imagery_run.mkdir(parents=True)
    imagery_trials = []
    for idx, nsd_id in enumerate(nsd_ids):
        trial_id = f"trial_{idx + 1:03d}"
        trial_path = imagery_run / f"{trial_id}_beta.nii.gz"
        trial_volume = rng.standard_normal(shape).astype(np.float32)
        nib.save(nib.Nifti1Image(trial_volume, np.eye(4)), trial_path)
        imagery_trials.append(
            {
                "trial_id": trial_id,
                "stimulus_type": "complex",
                "nsdId": nsd_id,
                "pair_id": nsd_id,
                "text_prompt": f"Stimulus {nsd_id}",
            }
        )
    with open(imagery_run / "metadata.json", "w") as handle:
        import json

        json.dump({"trials": imagery_trials}, handle, indent=2)

    # ROI masks
    mask_root = root / "roi_masks"
    mask_root.mkdir()
    roi_coords = {
        "lh.V1v_roi": (0, 0, 0),
        "lh.V2v_roi": (0, 1, 0),
        "lh.V3v_roi": (0, 2, 0),
        "rh.V4_roi": (1, 0, 0),
        "LO_roi": (1, 1, 0),
        "FFA_roi": (1, 2, 0),
        "PPA_roi": (1, 3, 0),
        "mPFC_roi": (2, 0, 0),
        "Precuneus_roi": (2, 1, 0),
        "IPS_roi": (2, 2, 0),
    }
    for name, coord in roi_coords.items():
        mask = np.zeros(shape, dtype=np.float32)
        mask[coord] = 1.0
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_root / f"{name}.nii.gz")

    # Canonicalizable target cache input
    raw_target_rows = []
    for nsd_id in nsd_ids:
        emb = rng.standard_normal(768).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        raw_target_rows.append({"nsdId": nsd_id, "embedding": emb.tolist()})
    raw_targets = root / "targets_raw.parquet"
    pd.DataFrame(raw_target_rows).to_parquet(raw_targets, index=False)

    prepared_dir = root / "prepared"
    config = {
        "dataset": {
            "subject": subject,
            "perception_index": str(raw_perception_index),
            "imagery_index": str(prepared_dir / "imagery.parquet"),
            "mixed_output_index": str(prepared_dir / "mixed.parquet"),
        },
        "preprocessing": {"enabled": False},
        "roi": {
            "groups": {
                "early_visual": ["V1", "V2", "V3"],
                "ventral_visual": ["V4", "LO", "FFA", "PPA"],
                "metacognitive": ["mPFC", "Precuneus", "IPS"],
            },
            "missing_policy": "error",
            "fallback_policy": "error",
            "mask_root": str(mask_root),
            "min_voxels": 1,
        },
        "targets": {
            "name": "vit_l14_image_768",
            "dimension": 768,
            "cache_path": str(prepared_dir / "targets.parquet"),
            "id_column": "nsdId",
        },
        "model": {
            "branch_embedding_dim": 16,
            "shared_dim": 16,
            "private_dim": 8,
            "dropout": 0.0,
            "use_domain_head": True,
            "use_vividness_head": True,
            "vividness_mode": "evidential",
        },
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "domain_weight": 0.1,
            "vividness_weight": 0.1,
            "confidence_weight": 0.05,
            "reconstruction_weight": 0.1,
            "device": "cpu",
            "output_dir": str(root / "train_outputs"),
            "seed": 0,
        },
        "evaluation": {
            "batch_size": 4,
            "output_dir": str(root / "eval_outputs"),
            "transfer_output_dir": str(root / "transfer_outputs"),
        },
        "analysis": {"output_dir": str(root / "analysis_outputs")},
        "export": {"output_dir": str(root / "export_outputs")},
        "preparation": {
            "imagery": {
                "data_root": str(imagery_root),
                "cache_root": str(root / "cache"),
            },
            "targets": {
                "input_cache": str(raw_targets),
            },
            "roi": {
                "provenance_path": str(prepared_dir / "roi_summary.json"),
            },
        },
    }
    config_path = root / "bootstrap_config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle)

    return {
        "root": root,
        "config_path": config_path,
        "raw_perception_index": raw_perception_index,
        "imagery_root": imagery_root,
        "mask_root": mask_root,
        "raw_targets": raw_targets,
        "prepared_dir": prepared_dir,
    }
