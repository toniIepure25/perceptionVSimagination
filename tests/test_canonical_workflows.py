import os
import subprocess
import sys


def _workflow_env():
    env = os.environ.copy()
    src_path = os.path.join(os.getcwd(), "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_canonical_workflows_smoke(canonical_fixture_dir):
    env = _workflow_env()
    config_path = str(canonical_fixture_dir["config_path"])
    train = subprocess.run(
        [sys.executable, "-m", "fmri2img.workflows.train_decoder", "--config", config_path],
        capture_output=True,
        text=True,
        env=env,
    )
    assert train.returncode == 0, train.stderr

    checkpoint = canonical_fixture_dir["root"] / "train_outputs" / "best_decoder.pt"
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

    transfer = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.eval_transfer",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert transfer.returncode == 0, transfer.stderr

    analysis = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.run_analysis",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert analysis.returncode == 0, analysis.stderr

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


def test_docs_reference_canonical_workflows():
    readme = open("README.md").read()
    start_here = open("START_HERE.md").read()
    for command in (
        "fmri2img.workflows.prepare_perception_index",
        "fmri2img.workflows.prepare_imagery_index",
        "fmri2img.workflows.prepare_targets",
        "fmri2img.workflows.prepare_mixed_index",
        "fmri2img.workflows.prepare_roi_features",
        "fmri2img.workflows.preflight_data",
        "fmri2img.workflows.train_decoder",
        "fmri2img.workflows.eval_decoder",
        "fmri2img.workflows.eval_transfer",
        "fmri2img.workflows.run_analysis",
        "fmri2img.workflows.export_for_animus",
    ):
        assert command in readme
        assert command in start_here


def test_bootstrap_docs_reference_ridge_comparison_workflow():
    reproducibility = open("docs/REPRODUCIBILITY.md").read()
    validation = open("docs/VALIDATION.md").read()
    comparison = open("docs/TINY_OVERLAP_BASELINE_COMPARISON.md").read()
    command = "fmri2img.workflows.run_legacy_ridge_baseline"
    assert command in reproducibility
    assert command in validation
    assert command in comparison


def test_expanded_overlap_docs_reference_max_available_config():
    reproducibility = open("docs/REPRODUCIBILITY.md").read()
    expanded = open("docs/EXPANDED_OVERLAP_COMPARISON.md").read()
    config_name = "configs/canonical/max_available_overlap.yaml"
    assert config_name in reproducibility
    assert config_name in expanded


def test_scaling_audit_doc_exists_and_references_overlap_ceiling():
    scaling = open("docs/EXPANDED_OVERLAP_COMPARISON.md").read()
    assert "4 shared `nsdId` pairs" in scaling
    assert "further data expansion" in scaling


def test_checked_in_smoke_config_runs(tmp_path):
    env = _workflow_env()
    output_dir = tmp_path / "smoke_outputs"
    cmd = [
        sys.executable,
        "-m",
        "fmri2img.workflows.train_decoder",
        "--config",
        "configs/canonical/shared_private_smoke.yaml",
        "--override",
        f"training.output_dir={output_dir.as_posix()}",
        "--override",
        f"evaluation.output_dir={(tmp_path / 'eval').as_posix()}",
        "--override",
        f"evaluation.transfer_output_dir={(tmp_path / 'transfer').as_posix()}",
        "--override",
        f"analysis.output_dir={(tmp_path / 'analysis').as_posix()}",
        "--override",
        f"export.output_dir={(tmp_path / 'export').as_posix()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert (output_dir / "best_decoder.pt").exists()


def test_checked_in_mvp_config_fails_with_actionable_missing_artifact_message():
    env = _workflow_env()
    result = subprocess.run(
        [sys.executable, "-m", "fmri2img.workflows.train_decoder", "--config", "configs/canonical/shared_private_mvp.yaml"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "Canonical workflow prerequisites are missing" in result.stderr
