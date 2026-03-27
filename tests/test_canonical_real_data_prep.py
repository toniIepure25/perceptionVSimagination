import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


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
