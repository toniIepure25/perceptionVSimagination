from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_full_imagery_overlap_shared_only_readiness")

from fmri2img.workflows._downstream_contract_audit import (  # noqa: E402
    build_downstream_contract_audit_report,
    load_json,
    merge_condition_semantics,
    normalize_decoder_card_target,
    normalize_manifest_target,
    surface_condition_semantics,
)
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/full_imagery_overlap_shared_only.yaml"
EXPECTED_EXPERIMENT_NAME = "full_imagery_overlap_shared_only"
TRAIN_FILES = ("best_decoder.pt", "config_snapshot.json", "roi_summary.json", "target_summary.json", "train_history.json")
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json")
TRANSFER_FILES = ("transfer_metrics.json", "per_trial_pairs.csv")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")
OPERATIONAL_BOUNDARY = [
    "this readiness audit promotes the checked-in shared-only max-overlap bundle only to evidence-ready candidate status based on real post-train artifacts and explicit gates",
    "training_ready requires stronger held-out paired evidence and dedicated checked-in training provenance, not just successful export or internal contract consistency",
    "neither evidence_ready_candidate nor training_ready implies ridge has been surpassed or that production Animus deployment is already justified",
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


def _train_provenance_snapshot(config_snapshot: dict[str, Any], config) -> dict[str, Any]:
    snapshot_model = config_snapshot.get("model", {})
    expected_model = config["model"]
    snapshot_dataset = config_snapshot.get("dataset", {})
    expected_dataset = config["dataset"]
    snapshot_targets = config_snapshot.get("targets", {})
    expected_targets = config["targets"]
    model_keys = ("disentanglement_mode", "use_domain_head", "use_vividness_head", "private_dim", "shared_dim")
    dataset_keys = ("subject", "mixed_index", "perception_conditions", "imagery_conditions")
    target_keys = ("name", "dimension", "cache_path", "id_column")
    return {
        "config_snapshot_experiment_name": str(config_snapshot.get("experiment", {}).get("name", "")),
        "config_snapshot_matches_expected_experiment": (
            str(config_snapshot.get("experiment", {}).get("name", "")) == EXPECTED_EXPERIMENT_NAME
        ),
        "model_matches_expected": all(snapshot_model.get(key) == expected_model.get(key) for key in model_keys),
        "dataset_matches_expected": all(snapshot_dataset.get(key) == expected_dataset.get(key) for key in dataset_keys),
        "targets_match_expected": all(snapshot_targets.get(key) == expected_targets.get(key) for key in target_keys),
        "dataset_capabilities": dict(config_snapshot.get("dataset_capabilities", {})),
    }


def build_full_imagery_overlap_shared_only_readiness_audit(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError("Shared-only max-overlap readiness audit requires configs/canonical/full_imagery_overlap_shared_only.yaml.")

    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_output_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()

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
        raise FileNotFoundError(
            "Shared-only max-overlap readiness audit requires real bundle artifacts: " + "; ".join(problems)
        )

    config_snapshot = load_json(train_output_dir / "config_snapshot.json")
    train_history = load_json(train_output_dir / "train_history.json")
    eval_metrics = load_json(eval_output_dir / "metrics.json")
    transfer_metrics = load_json(transfer_output_dir / "transfer_metrics.json")
    manifest = load_json(export_output_dir / "manifest.json")
    decoder_card = load_json(export_output_dir / "decoder_card.json")

    merged_conditions = merge_condition_semantics(eval_metrics, transfer_metrics)
    downstream_contract = build_downstream_contract_audit_report(
        config_path=config_path,
        artifact_paths={
            "train_dir": str(train_output_dir),
            "eval_dir": str(eval_output_dir),
            "transfer_dir": str(transfer_output_dir),
            "export_dir": str(export_output_dir),
            "eval_metrics": str((eval_output_dir / "metrics.json").resolve()),
            "transfer_metrics": str((transfer_output_dir / "transfer_metrics.json").resolve()),
            "export_manifest": str((export_output_dir / "manifest.json").resolve()),
            "export_decoder_card": str((export_output_dir / "decoder_card.json").resolve()),
        },
        target_spec={
            "shared": normalize_manifest_target(
                manifest.get("metadata", {}).get("target_spec_normalized"),
                fallback_target_spec=manifest.get("target_spec"),
            ),
            "decoder_card": normalize_decoder_card_target(decoder_card.get("target")),
            "expected_config": normalize_manifest_target(
                None,
                fallback_target_spec={
                    "target_name": config["targets"].get("name"),
                    "dimension": config["targets"].get("dimension"),
                },
            ),
        },
        condition_semantics={
            "shared": surface_condition_semantics(manifest.get("metadata", {}).get("condition_semantics")),
            "decoder_card": surface_condition_semantics(decoder_card.get("condition_semantics")),
            "eval": merged_conditions["payload_0"],
            "transfer": merged_conditions["payload_1"],
        },
        identity={
            "experiment_name": {
                "manifest": manifest.get("metadata", {}).get("experiment", {}).get("name"),
                "decoder_card": decoder_card.get("experiment", {}).get("name"),
                "expected_config": EXPECTED_EXPERIMENT_NAME,
            },
            "benchmark_role": {
                "manifest": manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role"),
                "decoder_card": decoder_card.get("experiment", {}).get("benchmark_role"),
                "expected_config": config.get("experiment.benchmark_role"),
            },
        },
        state={
            "eval_smoke_ready": True,
            "transfer_smoke_ready": True,
            "export_smoke_ready": True,
            "training_ready": False,
        },
        target_checks=[
            {
                "surface_key": "decoder_card",
                "check_name": "target_manifest_vs_decoder_card",
                "shared_label": "export manifest",
                "surface_label": "decoder card",
            },
            {
                "surface_key": "expected_config",
                "check_name": "target_manifest_vs_expected_config",
                "shared_label": "export manifest",
                "surface_label": "checked-in shared-only config",
            },
        ],
        condition_checks=[
            {
                "surface_key": "decoder_card",
                "check_name": "condition_manifest_vs_decoder_card",
                "shared_label": "export manifest",
                "surface_label": "decoder card",
            },
            {
                "surface_key": "eval",
                "check_name": "condition_manifest_vs_eval_metrics",
                "shared_label": "export manifest",
                "surface_label": "eval metrics",
            },
            {
                "surface_key": "transfer",
                "check_name": "condition_manifest_vs_transfer_metrics",
                "shared_label": "export manifest",
                "surface_label": "transfer metrics",
            },
        ],
        extra_sections={
            "bundle": {
                "experiment_name": EXPECTED_EXPERIMENT_NAME,
                "report_surface": "eval_transfer_metrics_plus_export_bundle",
                "normalized_condition_semantics": merged_conditions,
            }
        },
        operational_boundary=[
            "this contract verdict only verifies internal consistency across the checked-in shared-only full-overlap eval, transfer, and export artifacts",
            "this is an operational contract verdict for a real non-smoke bundle, not a benchmark-winning or paper-superseding claim",
            "training_ready remains false here until stronger held-out paired evidence and dedicated checked-in training provenance are both real",
        ],
    )

    shared_target = dict(downstream_contract.get("target_spec", {}).get("shared", {}))
    shared_condition = dict(downstream_contract.get("condition_semantics", {}).get("shared", {}))
    eval_summary = _metric_summary(eval_metrics, expected_target_space=str(config["targets"].get("name", "")))
    transfer_summary = _metric_summary(transfer_metrics, expected_target_space=str(config["targets"].get("name", "")))
    train_provenance = _train_provenance_snapshot(config_snapshot, config)

    manifest_metadata = manifest.get("metadata", {})
    export_bundle_modernized = bool(
        manifest_metadata.get("target_spec_normalized")
        and manifest_metadata.get("condition_semantics")
        and decoder_card.get("target")
        and decoder_card.get("condition_semantics")
    )
    artifact_availability = {
        "train_bundle_complete": not missing["train"],
        "eval_bundle_complete": not missing["eval"],
        "transfer_bundle_complete": not missing["transfer"],
        "export_bundle_complete": not missing["export"],
        "export_bundle_modernized": export_bundle_modernized,
        "train_history_nonempty": isinstance(train_history, list) and len(train_history) > 0,
        "missing_files": missing_artifacts,
    }

    operational_ready = all(
        [
            artifact_availability["train_bundle_complete"],
            artifact_availability["eval_bundle_complete"],
            artifact_availability["transfer_bundle_complete"],
            artifact_availability["export_bundle_complete"],
            artifact_availability["export_bundle_modernized"],
            artifact_availability["train_history_nonempty"],
        ]
    )
    downstream_contract_ready = bool(downstream_contract.get("state", {}).get("downstream_contract_ready"))
    experiment_description = str(config.get("experiment.description", ""))
    non_smoke_experiment = "smoke" not in EXPECTED_EXPERIMENT_NAME.lower() and "smoke" not in experiment_description.lower()
    non_fixture_inputs = not (
        _looks_fixture_backed(config["dataset"].get("mixed_index"))
        or _looks_fixture_backed(config["targets"].get("cache_path"))
    )
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
        "non_smoke_experiment": non_smoke_experiment,
        "non_fixture_inputs": non_fixture_inputs,
        "target_metadata_consistent": bool(downstream_contract.get("consistency", {}).get("target_manifest_vs_decoder_card"))
        and bool(downstream_contract.get("consistency", {}).get("target_manifest_vs_expected_config")),
        "condition_semantics_consistent": bool(
            downstream_contract.get("consistency", {}).get("condition_manifest_vs_decoder_card")
        )
        and bool(downstream_contract.get("consistency", {}).get("condition_manifest_vs_eval_metrics"))
        and bool(downstream_contract.get("consistency", {}).get("condition_manifest_vs_transfer_metrics")),
        "paired_conditions_present": paired_conditions_present,
        "paired_metrics_available": paired_metrics_available,
        "metric_target_spaces_match_config": metric_target_spaces_match_config,
        "eval_metrics_finite": eval_metrics_finite,
        "transfer_metrics_finite": transfer_metrics_finite,
    }
    evidence_ready_candidate = all(evidence_checks.values())

    training_pair_threshold = int(config.get("preparation.preflight.paper_pair_threshold", 32))
    heldout_pair_count = min(eval_summary["pair_count"], transfer_summary["pair_count"])
    training_checks = {
        "evidence_ready_candidate": evidence_ready_candidate,
        "train_provenance_matches_checked_in_config": bool(
            train_provenance["config_snapshot_matches_expected_experiment"]
        ),
        "heldout_pair_count_meets_threshold": heldout_pair_count >= training_pair_threshold,
    }
    training_ready = all(training_checks.values())

    evidence_blockers: list[str] = []
    if not evidence_checks["operational_ready"]:
        evidence_blockers.append("candidate bundle is not fully operationally ready from real post-train artifacts")
    if not evidence_checks["downstream_contract_ready"]:
        evidence_blockers.append("downstream contract checks are not ready for the candidate bundle")
    if not evidence_checks["non_smoke_experiment"]:
        evidence_blockers.append("candidate config is still smoke-scoped")
    if not evidence_checks["non_fixture_inputs"]:
        evidence_blockers.append("candidate inputs remain fixture-backed")
    if not evidence_checks["target_metadata_consistent"]:
        evidence_blockers.append("normalized target metadata is not consistent across config, export, and contract surfaces")
    if not evidence_checks["condition_semantics_consistent"]:
        evidence_blockers.append("normalized condition semantics are not consistent across export, eval, and transfer surfaces")
    if not evidence_checks["paired_conditions_present"]:
        evidence_blockers.append("candidate bundle does not expose both perception and imagery conditions")
    if not evidence_checks["paired_metrics_available"]:
        evidence_blockers.append("paired metrics are not available for the candidate bundle")
    if not evidence_checks["metric_target_spaces_match_config"]:
        evidence_blockers.append("eval/transfer target_space values do not match the checked-in shared-only config target")
    if not evidence_checks["eval_metrics_finite"]:
        evidence_blockers.append("eval metrics are incomplete or non-finite for the candidate bundle")
    if not evidence_checks["transfer_metrics_finite"]:
        evidence_blockers.append("transfer metrics are incomplete or non-finite for the candidate bundle")

    training_blockers = list(evidence_blockers)
    if not training_checks["train_provenance_matches_checked_in_config"]:
        training_blockers.append(
            "train bundle provenance still points to a max_available_overlap override run rather than a dedicated checked-in full_imagery_overlap_shared_only training run"
        )
    if not training_checks["heldout_pair_count_meets_threshold"]:
        training_blockers.append(
            f"held-out paired evaluation support remains too small for training-ready promotion ({heldout_pair_count}/{training_pair_threshold} paired groups)"
        )

    report = {
        "config": str(Path(config_path).resolve()),
        "bundle_identity": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "bundle_family": "full_imagery_overlap_shared_only",
            "description": experiment_description,
            "benchmark_role": manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role"),
            "selection_reason": "best current non-smoke canonical neural baseline with real paired train/eval/transfer/export artifacts and practical Animus relevance",
        },
        "artifact_paths": {
            "train_dir": str(train_output_dir),
            "eval_dir": str(eval_output_dir),
            "transfer_dir": str(transfer_output_dir),
            "export_dir": str(export_output_dir),
            "train_history": str((train_output_dir / "train_history.json").resolve()),
            "config_snapshot": str((train_output_dir / "config_snapshot.json").resolve()),
            "eval_metrics": str((eval_output_dir / "metrics.json").resolve()),
            "transfer_metrics": str((transfer_output_dir / "transfer_metrics.json").resolve()),
            "export_manifest": str((export_output_dir / "manifest.json").resolve()),
            "export_decoder_card": str((export_output_dir / "decoder_card.json").resolve()),
        },
        "artifact_availability": artifact_availability,
        "train_provenance": train_provenance,
        "target_spec": shared_target,
        "condition_semantics": shared_condition,
        "metrics": {
            "eval": eval_summary,
            "transfer": transfer_summary,
        },
        "downstream_contract": {
            "state": dict(downstream_contract.get("state", {})),
            "consistency": dict(downstream_contract.get("consistency", {})),
            "blocked_reasons": list(downstream_contract.get("blocked_reasons", [])),
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
            "bundle_family": "full_imagery_overlap_shared_only",
        },
        "artifact_paths": {},
        "artifact_availability": {},
        "train_provenance": {},
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
        description="Audit readiness-promotion status for the non-smoke shared-only full-overlap canonical bundle."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_full_imagery_overlap_shared_only_readiness_audit(config, config_path=args.config)
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/full_imagery_overlap_shared_only/readiness_audit.json"
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
