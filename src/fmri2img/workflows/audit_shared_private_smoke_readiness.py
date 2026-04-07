from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_shared_private_smoke_readiness")

from fmri2img.workflows._downstream_contract_audit import (  # noqa: E402
    load_json,
    normalize_decoder_card_target,
    normalize_manifest_target,
    surface_condition_semantics,
)
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/shared_private_smoke.yaml"
EXPECTED_EXPERIMENT_NAME = "shared_private_smoke"
TRAIN_FILES = ("best_decoder.pt", "config_snapshot.json", "roi_summary.json", "target_summary.json", "train_history.json")
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json", "downstream_contract_audit.json")
TRANSFER_FILES = ("transfer_metrics.json", "per_trial_pairs.csv")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")
OPERATIONAL_BOUNDARY = [
    "this readiness audit promotes a bundle only to candidate status based on real post-train artifacts and explicit gate checks",
    "evidence_ready_candidate does not imply benchmark success, paper evidence, or production readiness by itself",
    "training_ready only becomes true when the candidate also stops being smoke-scoped and fixture-backed",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _missing_files(root: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def _looks_fixture_backed(path_value: Any) -> bool:
    posix = Path(str(path_value)).as_posix()
    return posix.startswith("tests/fixtures/") or "/tests/fixtures/" in posix


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_summary(payload: dict[str, Any], *, expected_target_space: str) -> dict[str, Any]:
    overall = payload.get("overall", {})
    pair_metrics = payload.get("pair_metrics", {})
    condition_semantics = surface_condition_semantics(payload)
    try:
        pair_count = int(pair_metrics.get("n_pairs", 0) or 0)
    except (TypeError, ValueError):
        pair_count = 0
    overall_cosine_present = _is_finite_number(overall.get("cosine"))
    overall_mse_present = _is_finite_number(overall.get("mse"))
    return {
        "target_space": payload.get("target_space"),
        "target_space_matches_config": payload.get("target_space") == expected_target_space,
        "overall_cosine": overall.get("cosine"),
        "overall_mse": overall.get("mse"),
        "overall_cosine_present": overall_cosine_present,
        "overall_mse_present": overall_mse_present,
        "metrics_finite": overall_cosine_present and overall_mse_present,
        "by_condition_count": len(payload.get("by_condition", [])),
        "pair_count": pair_count,
        "pair_count_positive": pair_count > 0,
        "paired_metrics_available": bool(condition_semantics["paired_metrics_available"]),
    }


def _normalize_shared_target(
    *,
    manifest: dict[str, Any],
    decoder_card: dict[str, Any],
    downstream_report: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    manifest_target = normalize_manifest_target(
        manifest.get("metadata", {}).get("target_spec_normalized"),
        fallback_target_spec=manifest.get("target_spec"),
    )
    decoder_target = normalize_decoder_card_target(decoder_card.get("target"))
    config_target = normalize_manifest_target(
        None,
        fallback_target_spec={
            "target_name": config["targets"].get("name"),
            "dimension": config["targets"].get("dimension"),
        },
    )
    downstream_target = downstream_report.get("target_spec", {}).get("shared", {})
    views = [manifest_target, decoder_target, config_target]
    if downstream_target:
        views.append(downstream_target)
    return manifest_target, all(view == manifest_target for view in views[1:])


def _normalize_shared_conditions(
    *,
    manifest: dict[str, Any],
    eval_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any],
    downstream_report: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    manifest_condition = surface_condition_semantics(manifest.get("metadata", {}).get("condition_semantics"))
    eval_condition = surface_condition_semantics(eval_metrics)
    transfer_condition = surface_condition_semantics(transfer_metrics)
    downstream_condition = downstream_report.get("condition_semantics", {}).get("shared", {})
    views = [manifest_condition, eval_condition, transfer_condition]
    if downstream_condition:
        views.append(downstream_condition)
    return manifest_condition, all(view == manifest_condition for view in views[1:])


def build_shared_private_smoke_readiness_audit(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError("Shared-private readiness audit requires configs/canonical/shared_private_smoke.yaml.")

    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_output_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()
    downstream_contract_path = eval_output_dir / "downstream_contract_audit.json"

    missing = {
        "train": _missing_files(train_output_dir, TRAIN_FILES),
        "eval": _missing_files(eval_output_dir, EVAL_FILES),
        "transfer": _missing_files(transfer_output_dir, TRANSFER_FILES),
        "export": _missing_files(export_output_dir, EXPORT_FILES),
    }
    missing_artifacts = {section: values for section, values in missing.items() if values}
    if missing_artifacts:
        problems = []
        for section, values in missing_artifacts.items():
            problems.append(f"{section}: {', '.join(values)}")
        raise FileNotFoundError("Shared-private readiness audit requires real bundle artifacts: " + "; ".join(problems))

    config_snapshot = load_json(train_output_dir / "config_snapshot.json")
    train_history = load_json(train_output_dir / "train_history.json")
    eval_metrics = load_json(eval_output_dir / "metrics.json")
    transfer_metrics = load_json(transfer_output_dir / "transfer_metrics.json")
    manifest = load_json(export_output_dir / "manifest.json")
    decoder_card = load_json(export_output_dir / "decoder_card.json")
    downstream_report = load_json(downstream_contract_path)

    shared_target, target_metadata_consistent = _normalize_shared_target(
        manifest=manifest,
        decoder_card=decoder_card,
        downstream_report=downstream_report,
        config=config,
    )
    shared_condition, condition_semantics_consistent = _normalize_shared_conditions(
        manifest=manifest,
        eval_metrics=eval_metrics,
        transfer_metrics=transfer_metrics,
        downstream_report=downstream_report,
    )
    eval_summary = _metric_summary(eval_metrics, expected_target_space=str(config["targets"].get("name", "")))
    transfer_summary = _metric_summary(transfer_metrics, expected_target_space=str(config["targets"].get("name", "")))

    train_history_nonempty = isinstance(train_history, list) and len(train_history) > 0
    config_snapshot_matches_experiment = (
        str(config_snapshot.get("experiment", {}).get("name", "")) == EXPECTED_EXPERIMENT_NAME
    )
    artifact_availability = {
        "train_bundle_complete": not missing["train"],
        "eval_bundle_complete": not missing["eval"],
        "transfer_bundle_complete": not missing["transfer"],
        "export_bundle_complete": not missing["export"],
        "downstream_contract_present": downstream_contract_path.exists(),
        "train_history_nonempty": train_history_nonempty,
        "config_snapshot_matches_experiment": config_snapshot_matches_experiment,
        "missing_files": missing_artifacts,
    }

    downstream_state = dict(downstream_report.get("state", {}))
    operational_ready = (
        artifact_availability["train_bundle_complete"]
        and artifact_availability["eval_bundle_complete"]
        and artifact_availability["transfer_bundle_complete"]
        and artifact_availability["export_bundle_complete"]
        and artifact_availability["downstream_contract_present"]
        and artifact_availability["train_history_nonempty"]
        and artifact_availability["config_snapshot_matches_experiment"]
        and bool(downstream_state.get("smoke_ready", True))
        and bool(downstream_state.get("eval_smoke_ready"))
        and bool(downstream_state.get("transfer_smoke_ready"))
        and bool(downstream_state.get("export_smoke_ready"))
    )
    downstream_contract_ready = bool(downstream_state.get("downstream_contract_ready"))
    paired_conditions_present = set(shared_condition.get("present_conditions", [])) >= {"perception", "imagery"}
    paired_metrics_available = bool(shared_condition.get("paired_metrics_available"))
    metric_target_spaces_match_config = bool(
        eval_summary["target_space_matches_config"] and transfer_summary["target_space_matches_config"]
    )
    eval_metrics_finite = bool(eval_summary["metrics_finite"] and eval_summary["pair_count_positive"])
    transfer_metrics_finite = bool(transfer_summary["metrics_finite"] and transfer_summary["pair_count_positive"])

    evidence_checks = {
        "operational_ready": operational_ready,
        "downstream_contract_ready": downstream_contract_ready,
        "target_metadata_consistent": target_metadata_consistent,
        "condition_semantics_consistent": condition_semantics_consistent,
        "paired_conditions_present": paired_conditions_present,
        "paired_metrics_available": paired_metrics_available,
        "metric_target_spaces_match_config": metric_target_spaces_match_config,
        "eval_metrics_finite": eval_metrics_finite,
        "transfer_metrics_finite": transfer_metrics_finite,
    }
    evidence_ready_candidate = all(evidence_checks.values())

    experiment_description = str(config.get("experiment.description", ""))
    smoke_scoped_experiment = "smoke" in EXPECTED_EXPERIMENT_NAME.lower() or "smoke" in experiment_description.lower()
    fixture_backed_inputs = _looks_fixture_backed(config["dataset"].get("mixed_index")) or _looks_fixture_backed(
        config["targets"].get("cache_path")
    )
    training_checks = {
        "evidence_ready_candidate": evidence_ready_candidate,
        "non_smoke_experiment": not smoke_scoped_experiment,
        "non_fixture_inputs": not fixture_backed_inputs,
    }
    training_ready = all(training_checks.values())

    evidence_blockers: list[str] = []
    if not evidence_checks["operational_ready"]:
        evidence_blockers.append("candidate bundle is not fully operationally ready from real post-train artifacts")
    if not evidence_checks["downstream_contract_ready"]:
        evidence_blockers.append("downstream contract audit is not ready for the candidate bundle")
    if not evidence_checks["target_metadata_consistent"]:
        evidence_blockers.append("normalized target metadata is not consistent across config, export, and downstream audit surfaces")
    if not evidence_checks["condition_semantics_consistent"]:
        evidence_blockers.append("normalized condition semantics are not consistent across export, eval/transfer metrics, and downstream audit surfaces")
    if not evidence_checks["paired_conditions_present"]:
        evidence_blockers.append("candidate bundle does not expose both perception and imagery conditions")
    if not evidence_checks["paired_metrics_available"]:
        evidence_blockers.append("paired metrics are not available for the candidate bundle")
    if not evidence_checks["metric_target_spaces_match_config"]:
        evidence_blockers.append("eval/transfer target_space values do not match the checked-in config target name")
    if not evidence_checks["eval_metrics_finite"]:
        evidence_blockers.append("eval metrics are incomplete or non-finite for the candidate bundle")
    if not evidence_checks["transfer_metrics_finite"]:
        evidence_blockers.append("transfer metrics are incomplete or non-finite for the candidate bundle")

    training_blockers = list(evidence_blockers)
    if not training_checks["non_smoke_experiment"]:
        training_blockers.append("candidate config remains smoke-scoped and is not yet eligible for training-ready promotion")
    if not training_checks["non_fixture_inputs"]:
        training_blockers.append("candidate inputs remain fixture-backed and are not yet eligible for training-ready promotion")

    report = {
        "config": str(Path(config_path).resolve()),
        "bundle_identity": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "bundle_family": "shared_private_smoke",
            "description": experiment_description,
            "benchmark_role": manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role"),
            "selection_reason": "best current paired canonical post-train bundle with real train/eval/transfer/export/downstream audit artifacts",
        },
        "artifact_paths": {
            "train_dir": str(train_output_dir),
            "eval_dir": str(eval_output_dir),
            "transfer_dir": str(transfer_output_dir),
            "export_dir": str(export_output_dir),
            "downstream_contract_audit": str(downstream_contract_path.resolve()),
            "train_history": str((train_output_dir / "train_history.json").resolve()),
            "config_snapshot": str((train_output_dir / "config_snapshot.json").resolve()),
            "eval_metrics": str((eval_output_dir / "metrics.json").resolve()),
            "transfer_metrics": str((transfer_output_dir / "transfer_metrics.json").resolve()),
            "export_manifest": str((export_output_dir / "manifest.json").resolve()),
            "export_decoder_card": str((export_output_dir / "decoder_card.json").resolve()),
        },
        "artifact_availability": artifact_availability,
        "target_spec": shared_target,
        "condition_semantics": shared_condition,
        "metrics": {
            "eval": eval_summary,
            "transfer": transfer_summary,
        },
        "downstream_contract": {
            "path": str(downstream_contract_path.resolve()),
            "state": downstream_state,
            "blocked_reasons": list(downstream_report.get("blocked_reasons", [])),
        },
        "evidence_checks": evidence_checks,
        "training_checks": training_checks,
        "state": {
            "operational_ready": operational_ready,
            "downstream_contract_ready": downstream_contract_ready,
            "evidence_ready_candidate": evidence_ready_candidate,
            "training_ready": training_ready,
        },
        "blocked_reasons": {
            "evidence_ready_candidate": evidence_blockers,
            "training_ready": training_blockers,
        },
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }
    return report


def _blocked_report(config_path: str | Path, message: str) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve()),
        "bundle_identity": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "bundle_family": "shared_private_smoke",
        },
        "artifact_paths": {},
        "artifact_availability": {},
        "target_spec": {},
        "condition_semantics": {},
        "metrics": {},
        "downstream_contract": {},
        "evidence_checks": {},
        "training_checks": {},
        "state": {
            "operational_ready": False,
            "downstream_contract_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "blocked_reasons": {
            "evidence_ready_candidate": [message],
            "training_ready": [message],
        },
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit readiness-promotion status for the canonical shared-private smoke bundle."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_shared_private_smoke_readiness_audit(config, config_path=args.config)
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/shared_private_smoke/readiness_audit.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["evaluation"]["output_dir"]).resolve() / "readiness_audit.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    print(f"Operational ready: {report['state']['operational_ready']}")
    print(f"Evidence-ready candidate: {report['state']['evidence_ready_candidate']}")
    print(f"Training ready: {report['state']['training_ready']}")
    if args.fail_on_blocked:
        return 0 if report["state"]["training_ready"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
