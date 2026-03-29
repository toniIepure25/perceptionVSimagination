import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image


def _workflow_env():
    env = os.environ.copy()
    src_path = os.path.join(os.getcwd(), "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_preflight_reports_blocked_before_prep(canonical_volume_fixture):
    env = _workflow_env()
    report_path = canonical_volume_fixture["root"] / "preflight_blocked.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.preflight_data",
            "--config",
            str(canonical_volume_fixture["config_path"]),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["readiness"]["status"] == "blocked"
    blocked_reasons = "\n".join(report["readiness"]["blocked_reasons"])
    assert "imagery" in blocked_reasons.lower()
    assert "target" in blocked_reasons.lower()


def test_canonical_prep_pipeline_reaches_bootstrap_ready_and_trains(canonical_volume_fixture):
    env = _workflow_env()
    config_path = str(canonical_volume_fixture["config_path"])

    commands = [
        ["-m", "fmri2img.workflows.prepare_perception_index", "--config", config_path],
        ["-m", "fmri2img.workflows.prepare_imagery_index", "--config", config_path],
        ["-m", "fmri2img.workflows.prepare_targets", "--config", config_path],
        ["-m", "fmri2img.workflows.prepare_mixed_index", "--config", config_path],
        ["-m", "fmri2img.workflows.prepare_roi_features", "--config", config_path],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, *command], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr

    imagery_index = canonical_volume_fixture["prepared_dir"] / "imagery.parquet"
    imagery_df = pd.read_parquet(imagery_index)
    assert sorted(imagery_df["condition"].unique().tolist()) == ["imagery"]

    mixed_index = canonical_volume_fixture["prepared_dir"] / "mixed.parquet"
    mixed_df = pd.read_parquet(mixed_index)
    assert mixed_df["roi_features_json"].notna().all()
    assert mixed_df["roi_values_json"].notna().all()
    assert mixed_df["roi_names_json"].notna().all()

    report_path = canonical_volume_fixture["root"] / "preflight_ready.json"
    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.preflight_data",
            "--config",
            config_path,
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert preflight.returncode == 0, preflight.stderr
    report = json.loads(report_path.read_text())
    assert report["readiness"]["status"] == "bootstrap_ready"

    train = subprocess.run(
        [sys.executable, "-m", "fmri2img.workflows.train_decoder", "--config", config_path],
        capture_output=True,
        text=True,
        env=env,
    )
    assert train.returncode == 0, train.stderr
    checkpoint = canonical_volume_fixture["root"] / "train_outputs" / "best_decoder.pt"
    assert checkpoint.exists()

    eval_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.eval_decoder",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert eval_run.returncode == 0, eval_run.stderr

    export = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.export_for_animus",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert export.returncode == 0, export.stderr


def test_multisubject_overlap_bootstrap_workflow_builds_real_ready_index(canonical_multisubj_overlap_fixture):
    env = _workflow_env()
    config_path = str(canonical_multisubj_overlap_fixture["config_path"])

    for subject in canonical_multisubj_overlap_fixture["subjects"]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fmri2img.workflows.prepare_imagery_index",
                "--config",
                config_path,
                "--override",
                f"dataset.subject={json.dumps(subject)}",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    overlap = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.prepare_overlap_bootstrap",
            "--config",
            config_path,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert overlap.returncode == 0, overlap.stderr

    mixed_index = canonical_multisubj_overlap_fixture["prepared_root"] / "multisubj_overlap_mixed_with_roi.parquet"
    mixed_df = pd.read_parquet(mixed_index)
    assert sorted(mixed_df["subject"].unique().tolist()) == ["subj02", "subj05"]
    assert mixed_df["pair_id"].nunique() == 2
    assert mixed_df["roi_features_json"].notna().all()

    targets = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.prepare_targets",
            "--config",
            config_path,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert targets.returncode == 0, targets.stderr

    report_path = canonical_multisubj_overlap_fixture["root"] / "multisubj_preflight.json"
    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.preflight_data",
            "--config",
            config_path,
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert preflight.returncode == 0, preflight.stderr
    report = json.loads(report_path.read_text())
    assert report["readiness"]["status"] == "bootstrap_ready"
    assert report["readiness"]["paper_pair_threshold"] == 8


def test_prepare_targets_rejects_noncanonical_dimension(canonical_volume_fixture, tmp_path):
    env = _workflow_env()
    wrong_targets = tmp_path / "wrong_targets.parquet"
    pd.DataFrame([{"nsdId": 1, "embedding": [0.0] * 512}]).to_parquet(wrong_targets, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.prepare_targets",
            "--config",
            str(canonical_volume_fixture["config_path"]),
            "--override",
            f"preparation.targets.input_cache={json.dumps(str(wrong_targets))}",
            "--override",
            f"targets.cache_path={json.dumps(str(tmp_path / 'prepared_targets.parquet'))}",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "not canonical 768-D" in result.stderr


def test_optional_nibabel_reports_clear_runtime_errors(canonical_volume_fixture, monkeypatch):
    from fmri2img.data.canonical import CanonicalDecoderDataset, normalize_decoder_index
    from fmri2img.data import canonical as canonical_data
    from fmri2img.roi import materialize as roi_materialize

    nifti_path = canonical_volume_fixture["mask_root"] / "lh.V1v_roi.nii.gz"
    row = normalize_decoder_index(
        pd.DataFrame(
            [
                {
                    "subject": "subj01",
                    "condition": "perception",
                    "nsdId": 999,
                    "fmri_path": str(nifti_path),
                }
            ]
        )
    ).iloc[0]

    dataset = CanonicalDecoderDataset.__new__(CanonicalDecoderDataset)
    monkeypatch.setattr(canonical_data, "nib", None)
    with pytest.raises(RuntimeError, match="nibabel"):
        dataset._load_fmri(row)

    monkeypatch.setattr(roi_materialize, "nib", None)
    inspection = roi_materialize.inspect_roi_materialization_inputs(
        index=pd.DataFrame([row.to_dict()]),
        subject="subj01",
    )
    assert any("nibabel" in issue for issue in inspection["issues"])


def test_compute_embeddings_batch_accepts_structured_clip_outputs():
    from fmri2img.data.build_target_clip_cache_robust import compute_embeddings_batch

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            assert return_tensors == "pt"
            batch = len(images)
            return {"pixel_values": torch.ones(batch, 3, 2, 2)}

    class StructuredOutput:
        def __init__(self, tensor):
            self.pooler_output = tensor

    class DummyModel:
        def get_image_features(self, **inputs):
            batch = inputs["pixel_values"].shape[0]
            base = torch.tensor([[3.0, 4.0]] * batch)
            return StructuredOutput(base)

    images = {123: Image.new("RGB", (2, 2), color="white")}
    embeddings = compute_embeddings_batch(
        images=images,
        clip_model=DummyModel(),
        processor=DummyProcessor(),
        device="cpu",
        batch_size=1,
    )

    assert set(embeddings) == {123}
    vector = embeddings[123]
    assert vector.shape == (2,)
    assert pytest.approx(float((vector ** 2).sum()), rel=1e-6) == 1.0
