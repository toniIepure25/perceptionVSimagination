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


def test_prepare_public_nod_shared_only_adapter_keeps_only_fixed_resolved_subset(tmp_path):
    from fmri2img.workflows.prepare_public_nod_shared_only_adapter import (
        EXPECTED_RUN,
        build_public_nod_shared_only_adapter,
    )
    import pandas as pd

    index_path = tmp_path / "prepared_index.parquet"
    rows = []
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            rows.append(
                {
                    "task": "imagenet",
                    "subject": subject,
                    "session": session,
                    "run": EXPECTED_RUN,
                    "contract_scope": "imagenet_multisession_common_sessions",
                    "usable_for_later_shared_only_prep": True,
                    "events_resolved": True,
                    "preproc_bold_resolved": True,
                    "confounds_resolved": True,
                    "ciftify_beta_resolved": True,
                    "ciftify_label_resolved": True,
                }
            )
    rows.append(
        {
            "task": "imagenet",
            "subject": "sub-01",
            "session": "ses-imagenet01",
            "run": 9,
            "contract_scope": "imagenet_multisession_common_sessions",
            "usable_for_later_shared_only_prep": True,
            "events_resolved": True,
            "preproc_bold_resolved": True,
            "confounds_resolved": True,
            "ciftify_beta_resolved": True,
            "ciftify_label_resolved": True,
        }
    )
    rows.append(
        {
            "task": "imagenet",
            "subject": "sub-01",
            "session": "ses-imagenet01",
            "run": EXPECTED_RUN,
            "contract_scope": "imagenet_multisession_common_sessions",
            "usable_for_later_shared_only_prep": False,
            "events_resolved": True,
            "preproc_bold_resolved": True,
            "confounds_resolved": True,
            "ciftify_beta_resolved": True,
            "ciftify_label_resolved": True,
        }
    )
    pd.DataFrame(rows).to_parquet(index_path, index=False)

    prepared, report = build_public_nod_shared_only_adapter(index_path)
    assert len(prepared) == 36
    assert sorted(prepared["run"].unique().tolist()) == [10]
    assert report["row_count"] == 36
    assert report["usable_rows"] == 36
    assert report["state"] == {
        "adapter_ready": True,
        "prep_ready": True,
        "training_ready": False,
    }
    assert prepared["adapter_scope"].nunique() == 1
    assert prepared["adapter_status"].nunique() == 1


def test_prepare_public_nod_shared_only_adapter_requires_full_fixed_slice(tmp_path):
    from fmri2img.workflows.prepare_public_nod_shared_only_adapter import build_public_nod_shared_only_adapter
    import pandas as pd

    index_path = tmp_path / "prepared_index.parquet"
    pd.DataFrame(
        [
            {
                "task": "imagenet",
                "subject": "sub-01",
                "session": "ses-imagenet01",
                "run": 10,
                "contract_scope": "imagenet_multisession_common_sessions",
                "usable_for_later_shared_only_prep": True,
                "events_resolved": True,
                "preproc_bold_resolved": True,
                "confounds_resolved": True,
                "ciftify_beta_resolved": True,
                "ciftify_label_resolved": True,
            }
        ]
    ).to_parquet(index_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_shared_only_adapter(index_path)
    assert "requires the full resolved run-10 slice" in str(excinfo.value)


def test_prepare_public_nod_target_selection_builds_deterministic_trial_table(tmp_path):
    from fmri2img.workflows.prepare_public_nod_target_selection import build_public_nod_target_selection
    import pandas as pd

    repo_root = tmp_path
    input_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_shared_only_adapter.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496"

    adapter_rows = []
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            events_rel = Path(subject) / session / "func" / f"{subject}_{session}_task-imagenet_run-10_events.tsv"
            label_rel = (
                Path("derivatives")
                / "ciftify"
                / subject
                / "results"
                / f"{session}_task-imagenet_run-10"
                / f"{session}_task-imagenet_run-10_label.txt"
            )
            events_abs = dataset_root / events_rel
            label_abs = dataset_root / label_rel
            events_abs.parent.mkdir(parents=True, exist_ok=True)
            label_abs.parent.mkdir(parents=True, exist_ok=True)
            events_abs.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tstim_file\n"
                "0\t1\t101\t0.5\timagenet/n00000001/sample_a.JPEG\n"
                "4\t1\t102\t0.6\timagenet/n00000002/sample_b.JPEG\n"
            )
            label_abs.write_text("sample_a.JPEG\nsample_b.JPEG\n")
            adapter_rows.append(
                {
                    "dataset_id": "ds004496",
                    "dataset_label": "Natural Object Dataset (NOD)",
                    "lane": "practical_animus",
                    "task": "imagenet",
                    "subject": subject,
                    "session": session,
                    "run": 10,
                    "contract_scope": "imagenet_multisession_common_sessions",
                    "usable_for_later_shared_only_prep": True,
                    "events_resolved": True,
                    "preproc_bold_resolved": True,
                    "confounds_resolved": True,
                    "ciftify_beta_resolved": True,
                    "ciftify_label_resolved": True,
                    "events_path": str(events_rel),
                    "ciftify_label_path": str(label_rel),
                    "adapter_scope": "public_nod_imagenet_run10_shared_only",
                    "adapter_status": "adapter_ready_not_training_ready",
                }
            )
    pd.DataFrame(adapter_rows).to_parquet(input_path, index=False)

    selection, report = build_public_nod_target_selection(input_path)
    assert len(selection) == 72
    assert selection["target_identifier"].nunique() == 2
    assert report["adapter_row_count"] == 36
    assert report["target_selection_rows"] == 72
    assert report["state"] == {
        "target_selection_ready": True,
        "downstream_prep_ready": True,
        "training_ready": False,
    }
    assert report["per_run_target_counts"]["sub-01|ses-imagenet01|run-10"] == 2


def test_prepare_public_nod_target_selection_rejects_label_mismatch(tmp_path):
    from fmri2img.workflows.prepare_public_nod_target_selection import build_public_nod_target_selection
    import pandas as pd

    repo_root = tmp_path
    input_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_shared_only_adapter.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496"

    adapter_rows = []
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            events_rel = Path(subject) / session / "func" / f"{subject}_{session}_task-imagenet_run-10_events.tsv"
            label_rel = (
                Path("derivatives")
                / "ciftify"
                / subject
                / "results"
                / f"{session}_task-imagenet_run-10"
                / f"{session}_task-imagenet_run-10_label.txt"
            )
            events_abs = dataset_root / events_rel
            label_abs = dataset_root / label_rel
            events_abs.parent.mkdir(parents=True, exist_ok=True)
            label_abs.parent.mkdir(parents=True, exist_ok=True)
            events_abs.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tstim_file\n"
                "0\t1\t101\t0.5\timagenet/n00000001/sample_a.JPEG\n"
            )
            label_abs.write_text(("wrong_name.JPEG\n" if subject == "sub-01" and session == "ses-imagenet01" else "sample_a.JPEG\n"))
            adapter_rows.append(
                {
                    "dataset_id": "ds004496",
                    "dataset_label": "Natural Object Dataset (NOD)",
                    "lane": "practical_animus",
                    "task": "imagenet",
                    "subject": subject,
                    "session": session,
                    "run": 10,
                    "contract_scope": "imagenet_multisession_common_sessions",
                    "usable_for_later_shared_only_prep": True,
                    "events_resolved": True,
                    "preproc_bold_resolved": True,
                    "confounds_resolved": True,
                    "ciftify_beta_resolved": True,
                    "ciftify_label_resolved": True,
                    "events_path": str(events_rel),
                    "ciftify_label_path": str(label_rel),
                    "adapter_scope": "public_nod_imagenet_run10_shared_only",
                    "adapter_status": "adapter_ready_not_training_ready",
                }
            )
    pd.DataFrame(adapter_rows).to_parquet(input_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_target_selection(input_path)
    assert "deterministic agreement between events.tsv stim_file and label.txt" in str(excinfo.value)


def test_prepare_public_nod_target_embedding_manifest_reports_missing_stimulus_payloads(tmp_path):
    from fmri2img.workflows.prepare_public_nod_target_embedding_cache import build_public_nod_target_embedding_manifest
    import pandas as pd

    repo_root = tmp_path
    input_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_selection.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496" / "stimuli"

    selection_rows = []
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            for trial_index in range(1, 101):
                stim_rel = Path("imagenet") / f"class_{trial_index:04d}" / f"{subject}_{session}_{trial_index:03d}.JPEG"
                stim_path = dataset_root / stim_rel
                stim_path.parent.mkdir(parents=True, exist_ok=True)
                if trial_index == 1:
                    stim_path.write_bytes(b"jpeg-bytes")
                else:
                    target = (
                        repo_root
                        / "cache"
                        / "public_datasets"
                        / "ds004496"
                        / ".git"
                        / "annex"
                        / "objects"
                        / f"{subject}_{session}_{trial_index:03d}.JPEG"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stim_path.symlink_to(target)
                selection_rows.append(
                    {
                        "dataset_id": "ds004496",
                        "dataset_label": "Natural Object Dataset (NOD)",
                        "lane": "practical_animus",
                        "task": "imagenet",
                        "subject": subject,
                        "session": session,
                        "run": 10,
                        "trial_index": trial_index,
                        "stim_file": str(stim_rel),
                        "target_identifier": f"{subject}_{session}_{trial_index:03d}.JPEG",
                        "target_source": "events_stim_file_and_ciftify_label_match",
                        "target_domain": "imagenet",
                        "adapter_scope": "public_nod_imagenet_run10_shared_only",
                        "adapter_status": "adapter_ready_not_training_ready",
                    }
                )

    pd.DataFrame(selection_rows).to_parquet(input_path, index=False)

    manifest, report = build_public_nod_target_embedding_manifest(input_path)
    assert len(manifest) == 3600
    assert report["target_selection_rows"] == 3600
    assert report["unique_target_identifiers"] == 3600
    assert report["visible_stimulus_payloads"] == 3600
    assert report["resolved_stimulus_payloads"] == 36
    assert report["state"] == {
        "target_embedding_ready": False,
        "downstream_prep_ready": False,
        "training_ready": False,
    }
    assert manifest.loc[0, "embedding_status"] == "embedding_pending"
    assert manifest.loc[1, "embedding_status"] == "missing_image_payload"


def test_prepare_public_nod_target_embedding_manifest_rejects_slice_drift(tmp_path):
    from fmri2img.workflows.prepare_public_nod_target_embedding_cache import build_public_nod_target_embedding_manifest
    import pandas as pd

    repo_root = tmp_path
    input_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_selection.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            for trial_index in range(1, 101):
                selection_rows.append(
                    {
                        "dataset_id": "ds004496",
                        "dataset_label": "Natural Object Dataset (NOD)",
                        "lane": "practical_animus",
                        "task": "imagenet",
                        "subject": subject,
                        "session": session,
                        "run": 11 if subject == "sub-01" and session == "ses-imagenet01" and trial_index == 1 else 10,
                        "trial_index": trial_index,
                        "stim_file": f"imagenet/class_{trial_index:04d}/{subject}_{session}_{trial_index:03d}.JPEG",
                        "target_identifier": f"{subject}_{session}_{trial_index:03d}.JPEG",
                        "target_source": "events_stim_file_and_ciftify_label_match",
                        "target_domain": "imagenet",
                        "adapter_scope": "public_nod_imagenet_run10_shared_only",
                        "adapter_status": "adapter_ready_not_training_ready",
                    }
                )

    pd.DataFrame(selection_rows).to_parquet(input_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_target_embedding_manifest(input_path)
    assert "requires the fixed 3600-row target-selection slice" in str(excinfo.value)


def test_materialize_public_nod_stimuli_downloads_exact_fixed_slice(tmp_path, monkeypatch):
    from fmri2img.workflows.materialize_public_nod_stimuli import materialize_public_nod_stimuli
    import pandas as pd

    repo_root = tmp_path
    manifest_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_embedding_manifest.parquet"
    report_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_embedding_retrieval_report.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496"
    for index in range(3600):
        class_name = f"class_{index:04d}"
        rel = Path("cache") / "public_datasets" / "ds004496" / "stimuli" / "imagenet" / class_name / f"sample_{index:04d}.JPEG"
        worktree_path = repo_root / rel
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        target = dataset_root / ".git" / "annex" / "objects" / f"SHA256E-s12--sample_{index:04d}.JPEG"
        target.parent.mkdir(parents=True, exist_ok=True)
        worktree_path.symlink_to(target)
        rows.append(
            {
                "dataset_id": "ds004496",
                "dataset_label": "Natural Object Dataset (NOD)",
                "lane": "practical_animus",
                "task": "imagenet",
                "subject": f"sub-{index // 400 + 1:02d}",
                "session": f"ses-imagenet{(index % 400) // 100 + 1:02d}",
                "run": 10,
                "trial_index": index % 100 + 1,
                "stim_file": f"imagenet/{class_name}/sample_{index:04d}.JPEG",
                "target_identifier": f"sample_{index:04d}.JPEG",
                "target_source": "events_stim_file_and_ciftify_label_match",
                "target_domain": "imagenet",
                "adapter_scope": "public_nod_imagenet_run10_shared_only",
                "adapter_status": "adapter_ready_not_training_ready",
                "stimulus_path": str(rel),
                "stimulus_payload_visible": True,
                "stimulus_payload_resolved": False,
                "stimulus_is_symlink": True,
                "embedding_model_id": "openai/clip-vit-large-patch14",
                "embedding_dimension": 768,
                "embedding_column": "clip_target_768",
                "embedding_materialized": False,
                "cache_kind": "manifest",
                "cache_key": f"sample_{index:04d}.JPEG",
                "embedding_status": "missing_image_payload",
            }
        )
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    def _fake_download(url, destination, chunk_size=1024 * 1024):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg-bytes")
        return len(b"jpeg-bytes")

    monkeypatch.setattr("fmri2img.workflows.materialize_public_nod_stimuli._download_to_path", _fake_download)

    report = materialize_public_nod_stimuli(manifest_path, report_path)
    refreshed = pd.read_parquet(manifest_path)
    assert report["downloaded_files"] == 3600
    assert report["resolved_stimulus_payloads_after"] == 3600
    assert int(refreshed["stimulus_payload_resolved"].sum()) == 3600
    assert report_path.exists()


def test_build_public_nod_target_embedding_cache_materializes_real_vectors(tmp_path, monkeypatch):
    from fmri2img.workflows.build_public_nod_target_embedding_cache import build_public_nod_target_embedding_cache
    import pandas as pd
    from PIL import Image
    import numpy as np

    repo_root = tmp_path
    manifest_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_embedding_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    stimuli_root = repo_root / "cache" / "public_datasets" / "ds004496" / "stimuli"
    for index in range(3600):
        class_name = f"class_{index:04d}"
        rel = Path("cache") / "public_datasets" / "ds004496" / "stimuli" / "imagenet" / class_name / f"sample_{index:04d}.JPEG"
        image_path = repo_root / rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(index % 255, 0, 0)).save(image_path)
        rows.append(
            {
                "dataset_id": "ds004496",
                "dataset_label": "Natural Object Dataset (NOD)",
                "lane": "practical_animus",
                "task": "imagenet",
                "subject": f"sub-{index // 400 + 1:02d}",
                "session": f"ses-imagenet{(index % 400) // 100 + 1:02d}",
                "run": 10,
                "trial_index": index % 100 + 1,
                "stim_file": f"imagenet/{class_name}/sample_{index:04d}.JPEG",
                "target_identifier": f"sample_{index:04d}.JPEG",
                "target_source": "events_stim_file_and_ciftify_label_match",
                "target_domain": "imagenet",
                "adapter_scope": "public_nod_imagenet_run10_shared_only",
                "adapter_status": "adapter_ready_not_training_ready",
                "stimulus_path": str(rel),
                "stimulus_payload_visible": True,
                "stimulus_payload_resolved": True,
                "stimulus_is_symlink": False,
                "embedding_model_id": "openai/clip-vit-large-patch14",
                "embedding_dimension": 768,
                "embedding_column": "clip_target_768",
                "embedding_materialized": False,
                "cache_kind": "manifest",
                "cache_key": f"sample_{index:04d}.JPEG",
                "embedding_status": "embedding_pending",
            }
        )
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    def _fake_load_clip_encoder(model_id, device):
        return object(), object(), 768

    def _fake_compute_embeddings_batch(images, clip_model, processor, device, batch_size=32):
        return {
            pair_id: (np.arange(768, dtype=np.float32) + pair_id).astype(np.float32)
            for pair_id in images
        }

    monkeypatch.setattr(
        "fmri2img.workflows.build_public_nod_target_embedding_cache.load_clip_encoder",
        _fake_load_clip_encoder,
    )
    monkeypatch.setattr(
        "fmri2img.workflows.build_public_nod_target_embedding_cache.compute_embeddings_batch",
        _fake_compute_embeddings_batch,
    )

    cache, report = build_public_nod_target_embedding_cache(manifest_path, device="cpu", inference_batch_size=8)
    assert len(cache) == 3600
    assert report["embeddings_materialized"] == 3600
    assert report["state"] == {
        "target_embedding_ready": True,
        "downstream_prep_ready": True,
        "training_ready": False,
    }
    assert cache.columns.tolist() == [
        "pair_id",
        "target_identifier",
        "stimulus_path",
        "embedding_model_id",
        "embedding_dimension",
        "clip_target_768",
    ]
    assert len(cache.loc[0, "clip_target_768"]) == 768


def test_build_public_nod_target_embedding_cache_rejects_unresolved_manifest(tmp_path):
    from fmri2img.workflows.build_public_nod_target_embedding_cache import build_public_nod_target_embedding_cache
    import pandas as pd

    repo_root = tmp_path
    manifest_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_target_embedding_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(3600):
        rows.append(
            {
                "dataset_id": "ds004496",
                "dataset_label": "Natural Object Dataset (NOD)",
                "lane": "practical_animus",
                "task": "imagenet",
                "subject": f"sub-{index // 400 + 1:02d}",
                "session": f"ses-imagenet{(index % 400) // 100 + 1:02d}",
                "run": 10,
                "trial_index": index % 100 + 1,
                "stim_file": f"imagenet/class_{index:04d}/sample_{index:04d}.JPEG",
                "target_identifier": f"sample_{index:04d}.JPEG",
                "target_source": "events_stim_file_and_ciftify_label_match",
                "target_domain": "imagenet",
                "adapter_scope": "public_nod_imagenet_run10_shared_only",
                "adapter_status": "adapter_ready_not_training_ready",
                "stimulus_path": f"cache/public_datasets/ds004496/stimuli/imagenet/class_{index:04d}/sample_{index:04d}.JPEG",
                "stimulus_payload_visible": True,
                "stimulus_payload_resolved": False,
                "stimulus_is_symlink": True,
                "embedding_model_id": "openai/clip-vit-large-patch14",
                "embedding_dimension": 768,
                "embedding_column": "clip_target_768",
                "embedding_materialized": False,
                "cache_kind": "manifest",
                "cache_key": f"sample_{index:04d}.JPEG",
                "embedding_status": "missing_image_payload",
            }
        )
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_target_embedding_cache(manifest_path, device="cpu")
    assert "still has 3600 unresolved payloads" in str(excinfo.value)


def test_prepare_public_nod_shared_only_join_contract_builds_fixed_machine_readable_join(tmp_path):
    from fmri2img.workflows.prepare_public_nod_shared_only_join_contract import build_public_nod_shared_only_join_contract
    import pandas as pd

    repo_root = tmp_path
    base = repo_root / "cache" / "indices" / "public_nod"
    base.mkdir(parents=True, exist_ok=True)
    adapter_path = base / "imagenet_run10_shared_only_adapter.parquet"
    selection_path = base / "imagenet_run10_target_selection.parquet"
    cache_path = base / "imagenet_run10_target_embedding_cache.parquet"

    adapter_rows = []
    selection_rows = []
    cache_rows = []
    pair_id = 1
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session_idx, session in enumerate([f"ses-imagenet{index:02d}" for index in range(1, 5)], start=1):
            adapter_rows.append(
                {
                    "dataset_id": "ds004496",
                    "dataset_label": "Natural Object Dataset (NOD)",
                    "lane": "practical_animus",
                    "task": "imagenet",
                    "subject": subject,
                    "session": session,
                    "run": 10,
                    "contract_scope": "imagenet_multisession_common_sessions",
                    "usable_for_later_shared_only_prep": True,
                    "events_path": f"{subject}/{session}/func/events.tsv",
                    "preproc_bold_path": f"{subject}/{session}/func/bold.nii.gz",
                    "confounds_path": f"{subject}/{session}/func/confounds.tsv",
                    "ciftify_beta_path": f"derivatives/ciftify/{subject}/{session}/beta.dscalar.nii",
                    "ciftify_label_path": f"derivatives/ciftify/{subject}/{session}/labels.txt",
                }
            )
            for trial_index in range(1, 101):
                stim_rel = f"cache/public_datasets/ds004496/stimuli/imagenet/{subject}/{session}_{trial_index:03d}.JPEG"
                target_identifier = f"{subject}_{session}_{trial_index:03d}.JPEG"
                selection_rows.append(
                    {
                        "dataset_id": "ds004496",
                        "dataset_label": "Natural Object Dataset (NOD)",
                        "lane": "practical_animus",
                        "task": "imagenet",
                        "subject": subject,
                        "session": session,
                        "run": 10,
                        "trial_index": trial_index,
                        "stimulus_path": stim_rel,
                        "target_identifier": target_identifier,
                        "target_source": "events_stim_file_and_ciftify_label_match",
                        "target_domain": "imagenet",
                        "adapter_scope": "public_nod_imagenet_run10_shared_only",
                        "adapter_status": "adapter_ready_not_training_ready",
                    }
                )
                cache_rows.append(
                    {
                        "pair_id": pair_id,
                        "target_identifier": target_identifier,
                        "stimulus_path": stim_rel,
                        "embedding_model_id": "openai/clip-vit-large-patch14",
                        "embedding_dimension": 768,
                        "clip_target_768": [0.0] * 768,
                    }
                )
                pair_id += 1

    pd.DataFrame(adapter_rows).to_parquet(adapter_path, index=False)
    pd.DataFrame(selection_rows).to_parquet(selection_path, index=False)
    pd.DataFrame(cache_rows).to_parquet(cache_path, index=False)

    artifact, report = build_public_nod_shared_only_join_contract(adapter_path, selection_path, cache_path)
    assert len(artifact) == 3600
    assert artifact["pair_id"].nunique() == 3600
    assert report["state"] == {
        "join_ready": True,
        "roi_ready": False,
        "downstream_prep_ready": False,
        "training_ready": False,
    }
    assert "pair_id" in report["required_downstream_columns"]


def test_prepare_public_nod_shared_only_join_contract_rejects_selection_drift(tmp_path):
    from fmri2img.workflows.prepare_public_nod_shared_only_join_contract import build_public_nod_shared_only_join_contract
    import pandas as pd

    repo_root = tmp_path
    base = repo_root / "cache" / "indices" / "public_nod"
    base.mkdir(parents=True, exist_ok=True)
    adapter_path = base / "imagenet_run10_shared_only_adapter.parquet"
    selection_path = base / "imagenet_run10_target_selection.parquet"
    cache_path = base / "imagenet_run10_target_embedding_cache.parquet"

    pd.DataFrame(
        [
            {
                "dataset_id": "ds004496",
                "dataset_label": "Natural Object Dataset (NOD)",
                "lane": "practical_animus",
                "task": "imagenet",
                "subject": "sub-01",
                "session": "ses-imagenet01",
                "run": 10,
                "contract_scope": "imagenet_multisession_common_sessions",
                "usable_for_later_shared_only_prep": True,
                "events_path": "a",
                "preproc_bold_path": "b",
                "confounds_path": "c",
                "ciftify_beta_path": "d",
                "ciftify_label_path": "e",
            }
        ]
    ).to_parquet(adapter_path, index=False)
    pd.DataFrame([{"task": "imagenet", "run": 10, "lane": "practical_animus", "target_domain": "imagenet", "adapter_scope": "public_nod_imagenet_run10_shared_only", "subject": "sub-01", "session": "ses-imagenet01", "trial_index": 1, "target_identifier": "x", "stimulus_path": "y", "target_source": "z", "adapter_status": "adapter_ready_not_training_ready"}]).to_parquet(selection_path, index=False)
    pd.DataFrame([{"pair_id": 1, "target_identifier": "x", "stimulus_path": "y", "embedding_model_id": "openai/clip-vit-large-patch14", "embedding_dimension": 768, "clip_target_768": [0.0] * 768}]).to_parquet(cache_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_shared_only_join_contract(adapter_path, selection_path, cache_path)
    assert "requires the fixed 36-row adapter slice" in str(excinfo.value)


def test_prepare_public_nod_roi_materialization_contract_builds_verified_source_manifest(tmp_path, monkeypatch):
    from fmri2img.workflows.prepare_public_nod_roi_materialization_contract import build_public_nod_roi_materialization_contract
    import pandas as pd

    repo_root = tmp_path
    join_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_shared_only_join_contract.parquet"
    join_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496"

    rows = []
    pair_id = 1
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            for trial_index in range(1, 101):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "adapter_row_id": f"{subject}|{session}|run-10",
                        "subject": subject,
                        "session": session,
                        "run": 10,
                        "trial_index": trial_index,
                        "condition": "perception",
                        "task": "imagenet",
                        "target_identifier": f"{subject}_{session}_{trial_index:03d}.JPEG",
                        "stimulus_path": f"cache/public_datasets/ds004496/stimuli/imagenet/{subject}/{session}_{trial_index:03d}.JPEG",
                        "embedding_model_id": "openai/clip-vit-large-patch14",
                        "embedding_dimension": 768,
                        "target_embedding_column": "clip_target_768",
                        "events_path": f"{subject}/{session}/func/events.tsv",
                        "preproc_bold_path": f"{subject}/{session}/func/bold.nii.gz",
                        "confounds_path": f"{subject}/{session}/func/confounds.tsv",
                        "ciftify_beta_path": f"derivatives/ciftify/{subject}/{session}/beta.dscalar.nii",
                        "ciftify_label_path": f"derivatives/ciftify/{subject}/{session}/labels.txt",
                        "target_cache_path": "cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet",
                        "roi_contract_key": pair_id,
                        "join_contract_version": "public_nod_imagenet_run10_v1",
                        "target_ready": True,
                        "roi_ready": False,
                        "training_ready": False,
                    }
                )
                pair_id += 1
            label = dataset_root / f"derivatives/ciftify/{subject}/{session}/labels.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join([f"{subject}_{session}_{idx:03d}.JPEG" for idx in range(1, 101)]) + "\n")
    pd.DataFrame(rows).to_parquet(join_path, index=False)

    class _FakeImg:
        shape = (100, 59412)

    monkeypatch.setattr("fmri2img.workflows.prepare_public_nod_roi_materialization_contract.nib.load", lambda path: _FakeImg())

    artifact, report = build_public_nod_roi_materialization_contract(join_path)
    assert len(artifact) == 36
    assert report["verified_join_rows"] == 3600
    assert report["state"] == {
        "join_ready": True,
        "roi_ready": False,
        "downstream_prep_ready": False,
        "training_ready": False,
    }


def test_prepare_public_nod_roi_materialization_contract_rejects_alignment_drift(tmp_path, monkeypatch):
    from fmri2img.workflows.prepare_public_nod_roi_materialization_contract import build_public_nod_roi_materialization_contract
    import pandas as pd

    repo_root = tmp_path
    join_path = repo_root / "cache" / "indices" / "public_nod" / "imagenet_run10_shared_only_join_contract.parquet"
    join_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "cache" / "public_datasets" / "ds004496"
    rows = []
    pair_id = 1
    for subject in [f"sub-{index:02d}" for index in range(1, 10)]:
        for session in [f"ses-imagenet{index:02d}" for index in range(1, 5)]:
            label = dataset_root / f"derivatives/ciftify/{subject}/{session}/labels.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("one\n")
            for trial_index in range(1, 101):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "adapter_row_id": f"{subject}|{session}|run-10",
                        "subject": subject,
                        "session": session,
                        "run": 10,
                        "trial_index": trial_index,
                        "condition": "perception",
                        "task": "imagenet",
                        "target_identifier": f"id_{pair_id}",
                        "stimulus_path": f"path_{pair_id}",
                        "embedding_model_id": "openai/clip-vit-large-patch14",
                        "embedding_dimension": 768,
                        "target_embedding_column": "clip_target_768",
                        "events_path": "a",
                        "preproc_bold_path": "b",
                        "confounds_path": "c",
                        "ciftify_beta_path": f"derivatives/ciftify/{subject}/{session}/beta.dscalar.nii",
                        "ciftify_label_path": f"derivatives/ciftify/{subject}/{session}/labels.txt",
                        "target_cache_path": "cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet",
                        "roi_contract_key": pair_id,
                        "join_contract_version": "public_nod_imagenet_run10_v1",
                        "target_ready": True,
                        "roi_ready": False,
                        "training_ready": False,
                    }
                )
                pair_id += 1
    pd.DataFrame(rows).to_parquet(join_path, index=False)

    class _FakeImg:
        shape = (99, 59412)

    monkeypatch.setattr("fmri2img.workflows.prepare_public_nod_roi_materialization_contract.nib.load", lambda path: _FakeImg())

    with pytest.raises(ValueError) as excinfo:
        build_public_nod_roi_materialization_contract(join_path)
    assert "requires per-run beta rows, label rows, and join rows to match" in str(excinfo.value)


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
    assert "fmri2img.workflows.prepare_public_nod_shared_only_adapter" in nod_note
    assert "fmri2img.workflows.prepare_public_nod_target_selection" in nod_note
    assert "fmri2img.workflows.prepare_public_nod_target_embedding_cache" in nod_note
    assert "fmri2img.workflows.materialize_public_nod_stimuli" in nod_note
    assert "fmri2img.workflows.build_public_nod_target_embedding_cache" in nod_note
    assert "fmri2img.workflows.prepare_public_nod_shared_only_join_contract" in nod_note
    assert "fmri2img.workflows.prepare_public_nod_roi_materialization_contract" in nod_note
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
