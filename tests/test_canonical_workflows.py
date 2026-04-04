import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_animus_core_workflow_wrappers_smoke(canonical_fixture_dir, tmp_path):
    env = _workflow_env()
    config_path = str(canonical_fixture_dir["config_path"])
    preflight_output = tmp_path / "preflight.json"
    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.preflight_animus_core_decoder",
            "--config",
            config_path,
            "--output",
            str(preflight_output),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert preflight_output.exists()

    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri2img.workflows.train_animus_core_decoder",
            "--config",
            config_path,
        ],
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
            "fmri2img.workflows.eval_animus_core_decoder",
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
            "fmri2img.workflows.export_animus_core_decoder",
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


def test_animus_core_wrapper_injects_default_config_when_omitted():
    env = _workflow_env()
    result = subprocess.run(
        [sys.executable, "-m", "fmri2img.workflows.train_animus_core_decoder"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "Canonical workflow prerequisites are missing" in result.stderr
    assert "required: --config" not in result.stderr


def test_docs_reference_canonical_workflows():
    readme = open("README.md").read()
    start_here = open("START_HERE.md").read()
    agents = open("AGENTS.md").read()
    documentation = open("Documentation.md").read()
    assert "START_HERE.md" in readme
    assert "docs/REPRODUCIBILITY.md" in readme
    assert "docs/EXPERIMENT_REGISTRY.md" in start_here
    assert "docs/EXPERIMENT_REGISTRY.md" in agents
    assert "docs/EXPERIMENT_REGISTRY.md" in documentation
    for command in (
        "fmri2img.workflows.acquire_public_nsd_imagery",
        "fmri2img.workflows.acquire_public_nod",
        "fmri2img.workflows.inspect_public_nod",
        "fmri2img.workflows.prepare_public_nod_index",
        "fmri2img.workflows.preflight_animus_core_decoder",
        "fmri2img.workflows.train_animus_core_decoder",
        "fmri2img.workflows.eval_animus_core_decoder",
        "fmri2img.workflows.export_animus_core_decoder",
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
    ladder = open("docs/BENCHMARK_LADDER.md").read()
    config_name = "configs/canonical/max_available_overlap.yaml"
    assert config_name in reproducibility
    assert config_name in expanded
    assert "configs/canonical/animus_core_decoder.yaml" in ladder
    assert "configs/canonical/threshold_shared_private_p16.yaml" in ladder


def test_external_data_docs_exist_and_reference_canonical_acquisition_plan():
    acquisition = open("docs/DATA_ACQUISITION_PROGRAM.md").read()
    integration = open("docs/EXTERNAL_DATA_INTEGRATION_PLAN.md").read()
    assert "fmri2img.workflows.acquire_public_nsd_imagery" in acquisition
    assert "openneuro.org/datasets/ds004937" in acquisition
    assert "natural-scenes-dataset" in acquisition
    assert "train_animus_core_decoder" in integration
    assert "threshold_shared_private_p16.yaml" in integration


def test_acquire_public_nsd_imagery_wrapper_invokes_official_script(monkeypatch):
    import fmri2img.workflows.acquire_public_nsd_imagery as workflow

    seen = {}

    class Result:
        returncode = 0

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        seen["check"] = check
        return Result()

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    rc = workflow.main(["--subjects", "subj01", "--dry-run"])
    assert rc == 0
    assert seen["cmd"][0] == sys.executable
    assert seen["cmd"][1].endswith("scripts/download_nsd_imagery.py")
    assert seen["cmd"][2:] == ["--subjects", "subj01", "--dry-run"]
    assert seen["check"] is False


def test_acquire_public_nod_wrapper_invokes_official_script(monkeypatch):
    import fmri2img.workflows.acquire_public_nod as workflow

    seen = {}

    class Result:
        returncode = 0

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        seen["check"] = check
        return Result()

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    rc = workflow.main(["--dry-run"])
    assert rc == 0
    assert seen["cmd"][0] == sys.executable
    assert seen["cmd"][1].endswith("scripts/download_nod_metadata.py")
    assert seen["cmd"][2:] == ["--dry-run"]
    assert seen["check"] is False


def test_inspect_public_nod_summarizes_minimal_layout(tmp_path):
    from fmri2img.workflows.inspect_public_nod import summarize_nod_layout

    root = tmp_path / "ds004496"
    root.mkdir()
    (root / "dataset_description.json").write_text("{}\n")
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\tgroup\n"
        "sub-01\t22\tF\tmulti-session\n"
        "sub-02\t21\tF\tmulti-session\n"
        "sub-10\t21\tM\tsingle-session\n"
    )
    (root / "sub-01" / "ses-imagenet01" / "func").mkdir(parents=True)
    (root / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-1_events.tsv").write_text("onset\tduration\n")
    (root / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-1_bold.json").write_text("{}\n")
    (root / "sub-01" / "ses-imagenet02" / "func").mkdir(parents=True)
    (root / "sub-01" / "ses-imagenet02" / "func" / "sub-01_ses-imagenet02_task-imagenet_run-1_events.tsv").write_text("onset\tduration\n")
    (root / "sub-01" / "ses-imagenet02" / "func" / "sub-01_ses-imagenet02_task-imagenet_run-1_bold.json").write_text("{}\n")
    (root / "sub-02" / "ses-imagenet01" / "func").mkdir(parents=True)
    (root / "sub-02" / "ses-imagenet01" / "func" / "sub-02_ses-imagenet01_task-imagenet_run-1_events.tsv").write_text("onset\tduration\n")
    (root / "sub-02" / "ses-imagenet01" / "func" / "sub-02_ses-imagenet01_task-imagenet_run-1_bold.json").write_text("{}\n")
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func").mkdir(parents=True)
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-1_space-T1w_desc-preproc_bold.nii.gz").write_text("x")
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-1_desc-confounds_timeseries.tsv").write_text("x")
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet02" / "func").mkdir(parents=True)
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet02" / "func" / "sub-01_ses-imagenet02_task-imagenet_run-1_space-T1w_desc-preproc_bold.nii.gz").write_text("x")
    (root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet02" / "func" / "sub-01_ses-imagenet02_task-imagenet_run-1_desc-confounds_timeseries.tsv").write_text("x")
    (root / "derivatives" / "fmriprep" / "sub-02" / "ses-imagenet01" / "func").mkdir(parents=True)
    (root / "derivatives" / "fmriprep" / "sub-02" / "ses-imagenet01" / "func" / "sub-02_ses-imagenet01_task-imagenet_run-1_space-T1w_desc-preproc_bold.nii.gz").write_text("x")
    (root / "derivatives" / "fmriprep" / "sub-02" / "ses-imagenet01" / "func" / "sub-02_ses-imagenet01_task-imagenet_run-1_desc-confounds_timeseries.tsv").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-1").mkdir(parents=True)
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_Atlas.dtseries.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_beta.dscalar.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_label.txt").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet02_task-imagenet_run-1").mkdir(parents=True)
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet02_task-imagenet_run-1" / "ses-imagenet02_task-imagenet_run-1_Atlas.dtseries.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet02_task-imagenet_run-1" / "ses-imagenet02_task-imagenet_run-1_beta.dscalar.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet02_task-imagenet_run-1" / "ses-imagenet02_task-imagenet_run-1_label.txt").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-02" / "results" / "ses-imagenet01_task-imagenet_run-1").mkdir(parents=True)
    (root / "derivatives" / "ciftify" / "sub-02" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_Atlas.dtseries.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-02" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_beta.dscalar.nii").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-02" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_label.txt").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-floc_task-floc").mkdir(parents=True)
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-floc_task-floc" / "floc-faces.dlabel.nii").write_text("x")

    summary = summarize_nod_layout(root)
    assert summary["subject_count"] == 2
    assert summary["multi_session_subjects"] == ["sub-01", "sub-02"]
    assert summary["readiness"]["metadata_clone_present"] is True
    assert summary["readiness"]["surface_glm_visible"] is True
    assert summary["readiness"]["animus_shared_only_training_ready"] is False
    assert summary["recommended_first_contract"]["task_family"] == "imagenet perception-only"
    assert summary["prepared_index_contract"]["common_sessions"] == ["ses-imagenet01"]
    assert summary["prepared_index_contract"]["per_subject_common_session_run_counts"] == {
        "sub-01": 1,
        "sub-02": 1,
    }
    assert summary["prepared_index_contract"]["expected_common_session_runs_per_subject"] == 1


def test_prepare_public_nod_index_marks_resolved_and_missing_payload(tmp_path):
    from fmri2img.workflows.prepare_public_nod_index import build_public_nod_index

    root = tmp_path / "ds004496"
    root.mkdir()
    (root / "dataset_description.json").write_text("{}\n")
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\tgroup\n"
        "sub-01\t22\tF\tmulti-session\n"
        "sub-02\t21\tF\tmulti-session\n"
    )
    for subject in ("sub-01", "sub-02"):
        (root / subject / "ses-imagenet01" / "func").mkdir(parents=True)
        (root / subject / "ses-imagenet01" / "func" / f"{subject}_ses-imagenet01_task-imagenet_run-1_events.tsv").write_text("onset\tduration\n")
        (root / subject / "ses-imagenet01" / "func" / f"{subject}_ses-imagenet01_task-imagenet_run-1_bold.json").write_text("{}\n")
        (root / "derivatives" / "fmriprep" / subject / "ses-imagenet01" / "func").mkdir(parents=True)
        (root / "derivatives" / "fmriprep" / subject / "ses-imagenet01" / "func" / f"{subject}_ses-imagenet01_task-imagenet_run-1_space-T1w_desc-preproc_bold.nii.gz").write_text("x")
        (root / "derivatives" / "fmriprep" / subject / "ses-imagenet01" / "func" / f"{subject}_ses-imagenet01_task-imagenet_run-1_desc-confounds_timeseries.tsv").write_text("x")
        run_dir = root / "derivatives" / "ciftify" / subject / "results" / "ses-imagenet01_task-imagenet_run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "ses-imagenet01_task-imagenet_run-1_Atlas.dtseries.nii").write_text("x")
        (run_dir / "ses-imagenet01_task-imagenet_run-1_label.txt").write_text("x")
    (root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_beta.dscalar.nii").write_text("x")
    broken_target = root / "missing_beta.nii"
    (root / "derivatives" / "ciftify" / "sub-02" / "results" / "ses-imagenet01_task-imagenet_run-1" / "ses-imagenet01_task-imagenet_run-1_beta.dscalar.nii").symlink_to(broken_target)

    df, report = build_public_nod_index(root)
    row1 = df[(df["subject"] == "sub-01") & (df["run"] == 1)].iloc[0]
    row2 = df[(df["subject"] == "sub-02") & (df["run"] == 1)].iloc[0]
    assert row1["row_status"] == "resolved"
    assert bool(row1["usable_for_later_shared_only_prep"]) is True
    assert row2["row_status"] == "missing_payload"
    assert bool(row2["ciftify_beta_visible"]) is True
    assert bool(row2["ciftify_beta_resolved"]) is False
    assert report["status_counts"]["resolved"] == 1
    assert report["status_counts"]["missing_payload"] == 1


def test_materialize_public_nod_payloads_builds_exact_manifest(tmp_path):
    from fmri2img.workflows.materialize_public_nod_payloads import build_missing_payload_manifest
    import pandas as pd

    root = tmp_path / "ds004496"
    root.mkdir()
    index_path = tmp_path / "index.parquet"

    bold_target = "../../.git/annex/objects/aa/bb/SHA256E-s100--bold.nii.gz/SHA256E-s100--bold.nii.gz"
    confounds_target = "../../.git/annex/objects/aa/bb/SHA256E-s20--confounds.tsv/SHA256E-s20--confounds.tsv"
    beta_target = "../../.git/annex/objects/aa/bb/SHA256E-s30--beta.nii/SHA256E-s30--beta.nii"
    label_target = "../../.git/annex/objects/aa/bb/SHA256E-s4--label.txt/SHA256E-s4--label.txt"

    events = root / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-10_events.tsv"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("onset\tduration\n")

    preproc = root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-10_space-T1w_desc-preproc_bold.nii.gz"
    confounds = root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sub-01_ses-imagenet01_task-imagenet_run-10_desc-confounds_timeseries.tsv"
    beta = root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-10" / "ses-imagenet01_task-imagenet_run-10_beta.dscalar.nii"
    label = root / "derivatives" / "ciftify" / "sub-01" / "results" / "ses-imagenet01_task-imagenet_run-10" / "ses-imagenet01_task-imagenet_run-10_label.txt"
    preproc.parent.mkdir(parents=True, exist_ok=True)
    beta.parent.mkdir(parents=True, exist_ok=True)
    preproc.symlink_to(bold_target)
    confounds.symlink_to(confounds_target)
    beta.symlink_to(beta_target)
    label.symlink_to(label_target)

    pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "session": "ses-imagenet01",
                "run": 10,
                "task": "imagenet",
                "row_status": "missing_payload",
                "usable_for_later_shared_only_prep": False,
                "events_path": str(events.relative_to(root)),
                "preproc_bold_path": str(preproc.relative_to(root)),
                "confounds_path": str(confounds.relative_to(root)),
                "ciftify_beta_path": str(beta.relative_to(root)),
                "ciftify_label_path": str(label.relative_to(root)),
                "events_visible": True,
                "preproc_bold_visible": True,
                "confounds_visible": True,
                "ciftify_beta_visible": True,
                "ciftify_label_visible": True,
                "events_resolved": True,
                "preproc_bold_resolved": False,
                "confounds_resolved": False,
                "ciftify_beta_resolved": False,
                "ciftify_label_resolved": False,
            }
        ]
    ).to_parquet(index_path, index=False)

    manifest, report = build_missing_payload_manifest(root, index_path)
    assert report["entry_count"] == 1
    assert report["runs"] == [10]
    assert report["total_estimated_bytes"] == 154
    assert report["bytes_by_class"] == {
        "preproc_bold": 100,
        "confounds": 20,
        "ciftify_beta": 30,
        "ciftify_label": 4,
    }
    entry = manifest["entries"][0]
    assert entry["subject"] == "sub-01"
    assert entry["files"]["preproc_bold"]["resolved"] is False
    assert entry["files"]["preproc_bold"]["estimated_bytes"] == 100


def test_materialize_public_nod_payloads_refuses_without_git_annex(monkeypatch, tmp_path, capsys):
    import fmri2img.workflows.materialize_public_nod_payloads as workflow

    dataset_root = tmp_path / "ds004496"
    dataset_root.mkdir()
    manifest = {
        "entries": [
            {
                "files": {
                    "preproc_bold": {"path": "a.nii.gz", "visible": True, "resolved": False},
                    "confounds": {"path": "a.tsv", "visible": True, "resolved": False},
                    "ciftify_beta": {"path": "a_beta.nii", "visible": True, "resolved": False},
                    "ciftify_label": {"path": "a_label.txt", "visible": True, "resolved": False},
                }
            }
        ]
    }
    monkeypatch.setattr(workflow.shutil, "which", lambda name: None)
    rc = workflow._materialize_via_annex(dataset_root, manifest)
    captured = capsys.readouterr()
    assert rc == 2
    assert "git-annex is not available" in captured.err


def test_materialize_public_nod_payloads_reports_unretrievable_annex_payloads(monkeypatch, tmp_path, capsys):
    import fmri2img.workflows.materialize_public_nod_payloads as workflow

    dataset_root = tmp_path / "ds004496"
    dataset_root.mkdir()
    manifest = {
        "entries": [
            {
                "files": {
                    "preproc_bold": {"path": "a.nii.gz", "visible": True, "resolved": False},
                    "confounds": {"path": "a.tsv", "visible": True, "resolved": False},
                    "ciftify_beta": {"path": "a_beta.nii", "visible": True, "resolved": False},
                    "ciftify_label": {"path": "a_label.txt", "visible": True, "resolved": False},
                }
            }
        ]
    }

    class Result:
        returncode = 1

    monkeypatch.setattr(workflow.shutil, "which", lambda name: "/usr/bin/git-annex")
    monkeypatch.setattr(workflow.subprocess, "run", lambda *args, **kwargs: Result())
    rc = workflow._materialize_via_annex(dataset_root, manifest)
    captured = capsys.readouterr()
    assert rc == 1
    assert "no usable annex source" in captured.err


def test_materialize_public_nod_payloads_direct_openneuro_s3_writes_to_annex_targets(monkeypatch, tmp_path):
    import json
    from urllib.error import URLError

    from fmri2img.workflows.materialize_public_nod_payloads import _materialize_via_openneuro_s3

    dataset_root = tmp_path / "ds004496"
    dataset_root.mkdir()
    retrieval_report = tmp_path / "retrieval.json"
    target = dataset_root / ".git" / "annex" / "objects" / "aa" / "bb" / "payload.nii.gz"
    worktree_path = dataset_root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sample.nii.gz"
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    worktree_path.symlink_to(Path(os.path.relpath(target, worktree_path.parent)))

    manifest = {
        "dataset_id": "ds004496",
        "entries": [
            {
                "files": {
                    "preproc_bold": {"path": str(worktree_path.relative_to(dataset_root)), "visible": True, "resolved": False},
                    "confounds": {"path": "ignore.tsv", "visible": False, "resolved": False},
                    "ciftify_beta": {"path": "ignore_beta.nii", "visible": False, "resolved": False},
                    "ciftify_label": {"path": "ignore_label.txt", "visible": False, "resolved": False},
                }
            }
        ],
    }

    def fake_download(url, destination, chunk_size=1024 * 1024):
        raise URLError("offline-test")

    monkeypatch.setattr(
        "fmri2img.workflows.materialize_public_nod_payloads._download_to_path",
        fake_download,
    )

    rc = _materialize_via_openneuro_s3(
        dataset_root,
        manifest,
        retrieval_report,
        base_url="https://example.invalid/openneuro",
    )
    assert rc == 1
    report = json.loads(retrieval_report.read_text())
    assert report["downloaded_files"] == 0
    assert len(report["failures"]) == 1


def test_materialize_public_nod_payloads_uses_direct_openneuro_s3_by_default(monkeypatch, tmp_path):
    import fmri2img.workflows.materialize_public_nod_payloads as workflow

    dataset_root = tmp_path / "ds004496"
    dataset_root.mkdir()
    index_path = tmp_path / "index.parquet"
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    retrieval_report_path = tmp_path / "retrieval.json"

    import pandas as pd

    pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "session": "ses-imagenet01",
                "run": 10,
                "task": "imagenet",
                "row_status": "missing_payload",
                "usable_for_later_shared_only_prep": False,
                "events_path": "sub-01/ses-imagenet01/func/sample_events.tsv",
                "preproc_bold_path": "derivatives/fmriprep/sub-01/ses-imagenet01/func/sample_bold.nii.gz",
                "confounds_path": "derivatives/fmriprep/sub-01/ses-imagenet01/func/sample_confounds.tsv",
                "ciftify_beta_path": "derivatives/ciftify/sub-01/results/run/sample_beta.nii",
                "ciftify_label_path": "derivatives/ciftify/sub-01/results/run/sample_label.txt",
                "events_visible": True,
                "preproc_bold_visible": True,
                "confounds_visible": False,
                "ciftify_beta_visible": False,
                "ciftify_label_visible": False,
                "events_resolved": True,
                "preproc_bold_resolved": False,
                "confounds_resolved": False,
                "ciftify_beta_resolved": False,
                "ciftify_label_resolved": False,
            }
        ]
    ).to_parquet(index_path, index=False)
    (dataset_root / "sub-01" / "ses-imagenet01" / "func").mkdir(parents=True)
    (dataset_root / "sub-01" / "ses-imagenet01" / "func" / "sample_events.tsv").write_text("x")
    target = dataset_root / ".git" / "annex" / "objects" / "aa" / "bb" / "payload.nii.gz"
    symlink_path = dataset_root / "derivatives" / "fmriprep" / "sub-01" / "ses-imagenet01" / "func" / "sample_bold.nii.gz"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    symlink_path.symlink_to(Path(os.path.relpath(target, symlink_path.parent)))

    seen = {}

    def fake_direct(dataset_root_arg, manifest_arg, retrieval_report_arg, base_url_arg):
        seen["dataset_root"] = dataset_root_arg
        seen["retrieval_report"] = retrieval_report_arg
        seen["base_url"] = base_url_arg
        seen["entry_count"] = len(manifest_arg["entries"])
        return 0

    monkeypatch.setattr(workflow, "_materialize_via_openneuro_s3", fake_direct)
    rc = workflow.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--index",
            str(index_path),
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
            "--retrieval-report",
            str(retrieval_report_path),
            "--materialize",
        ]
    )
    assert rc == 0
    assert seen["dataset_root"] == dataset_root.resolve()
    assert seen["retrieval_report"] == retrieval_report_path.resolve()
    assert seen["base_url"] == workflow.DEFAULT_OPENNEURO_S3_BASE
    assert seen["entry_count"] == 1


def test_scaling_audit_doc_exists_and_references_overlap_ceiling():
    scaling = open("docs/EXPANDED_OVERLAP_COMPARISON.md").read()
    assert "shared overlap ids: `5`" in scaling
    assert "shared-only" in scaling


def test_experiment_registry_exists_and_uses_compact_ledger_format():
    registry = Path("docs/EXPERIMENT_REGISTRY.md").read_text()
    assert "# Experiment Registry" in registry
    assert "Documentation.md" in registry
    assert "docs/PROJECT_MASTER_LOG.md" in registry
    assert "## EXP-2026-04-02-RIDGE-MAX-OVERLAP" in registry
    assert "Promoted to evidence?: yes" in registry


def test_project_venv_guard_helper_accepts_project_venv_and_rejects_system_python():
    from fmri2img.workflows._venv_guard import ensure_project_venv, is_running_in_project_venv, project_root

    assert is_running_in_project_venv(executable=sys.executable, repo_root=project_root())
    with pytest.raises(SystemExit) as excinfo:
        ensure_project_venv(
            "fmri2img.workflows.train_decoder",
            executable="/usr/bin/python3",
            repo_root=project_root(),
        )
    assert "must be run from the project .venv" in str(excinfo.value)


def test_train_decoder_fails_fast_outside_project_venv():
    system_python = Path("/usr/bin/python3")
    if not system_python.exists() or system_python.parent == Path(sys.executable).parent:
        pytest.skip("A distinct system Python interpreter is not available in this environment.")

    env = _workflow_env()
    result = subprocess.run(
        [str(system_python), "-m", "fmri2img.workflows.train_decoder", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "must be run from the project .venv" in result.stderr
    assert "./.venv/bin/python -m fmri2img.workflows.train_decoder" in result.stderr


def test_public_dataset_program_docs_and_catalog_exist():
    opportunity = Path("docs/PUBLIC_DATASET_OPPORTUNITY_MAP.md").read_text()
    plan = Path("docs/PUBLIC_DATASET_INTEGRATION_PLAN.md").read_text()
    acquisition = Path("docs/DATA_ACQUISITION_PROGRAM.md").read_text()
    nod_note = Path("docs/NOD_PUBLIC_DATASET.md").read_text()
    catalog = json.loads(Path("configs/public_datasets/catalog.json").read_text())
    assert "ds004496" in opportunity
    assert "ds000203" in opportunity
    assert "show_public_dataset_options" in opportunity
    assert "ds004496" in plan
    assert "orchestraiq-jupyter-75555bb5f5-hxwp5" in plan
    assert "kubectl exec" in acquisition
    assert "repo `.venv`" in acquisition
    assert "fmri2img.workflows.acquire_public_nod" in acquisition
    assert "metadata_only_git_clone" in nod_note
    assert "fmri2img.workflows.inspect_public_nod" in nod_note
    assert "prepared_index_contract" in nod_note
    assert "fmri2img.workflows.prepare_public_nod_index" in nod_note
    assert "fmri2img.workflows.materialize_public_nod_payloads" in nod_note
    assert any(item["id"] == "ds004496" for item in catalog["datasets"])


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
