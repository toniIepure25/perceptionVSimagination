from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_shared_private_smoke_downstream_contract")

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


DEFAULT_CONFIG = "configs/canonical/shared_private_smoke.yaml"
TRAIN_FILES = ("best_decoder.pt", "config_snapshot.json", "roi_summary.json", "target_summary.json", "train_history.json")
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json")
TRANSFER_FILES = ("transfer_metrics.json", "per_trial_pairs.csv")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def build_shared_private_smoke_downstream_contract_audit(config, *, config_path: str | Path) -> dict:
    validate_canonical_workflow_config(config)
    expected_name = "shared_private_smoke"
    if str(config.get("experiment.name", "")) != expected_name:
        raise ValueError("Shared-private downstream contract audit requires configs/canonical/shared_private_smoke.yaml.")

    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_output_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()

    missing_train_files = [name for name in TRAIN_FILES if not (train_output_dir / name).exists()]
    missing_eval_files = [name for name in EVAL_FILES if not (eval_output_dir / name).exists()]
    missing_transfer_files = [name for name in TRANSFER_FILES if not (transfer_output_dir / name).exists()]
    missing_export_files = [name for name in EXPORT_FILES if not (export_output_dir / name).exists()]
    missing = {
        "train": missing_train_files,
        "eval": missing_eval_files,
        "transfer": missing_transfer_files,
        "export": missing_export_files,
    }
    if any(missing.values()):
        problems = []
        for section, values in missing.items():
            if values:
                problems.append(f"{section}: {', '.join(values)}")
        raise FileNotFoundError("Shared-private downstream contract audit requires real smoke artifacts: " + "; ".join(problems))

    manifest_path = export_output_dir / "manifest.json"
    decoder_card_path = export_output_dir / "decoder_card.json"
    eval_metrics_path = eval_output_dir / "metrics.json"
    transfer_metrics_path = transfer_output_dir / "transfer_metrics.json"

    manifest = load_json(manifest_path)
    decoder_card = load_json(decoder_card_path)
    eval_metrics = load_json(eval_metrics_path)
    transfer_metrics = load_json(transfer_metrics_path)
    merged_conditions = merge_condition_semantics(eval_metrics, transfer_metrics)

    report = build_downstream_contract_audit_report(
        config_path=config_path,
        artifact_paths={
            "train_dir": str(train_output_dir),
            "eval_dir": str(eval_output_dir),
            "transfer_dir": str(transfer_output_dir),
            "export_dir": str(export_output_dir),
            "eval_metrics": str(eval_metrics_path.resolve()),
            "transfer_metrics": str(transfer_metrics_path.resolve()),
            "export_manifest": str(manifest_path.resolve()),
            "export_decoder_card": str(decoder_card_path.resolve()),
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
                "expected_config": expected_name,
            },
            "benchmark_role": {
                "manifest": manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role"),
                "decoder_card": decoder_card.get("experiment", {}).get("benchmark_role"),
                "expected_config": config.get("experiment.benchmark_role"),
            },
        },
        state={
            "smoke_ready": True,
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
                "surface_label": "checked-in smoke config",
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
                "experiment_name": expected_name,
                "report_surface": "eval_transfer_metrics_plus_export_bundle",
                "normalized_condition_semantics": merged_conditions,
            }
        },
        operational_boundary=[
            "this audit only verifies internal consistency across the canonical shared-private smoke eval, transfer, and export artifacts",
            "this is an operational downstream contract verdict, not a benchmark or evidence-facing result",
            "training_ready remains false even when downstream_contract_ready is true",
        ],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit downstream export/report contract consistency for the canonical shared-private smoke bundle."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_shared_private_smoke_downstream_contract_audit(config, config_path=args.config)
    except Exception as exc:
        report = {
            "config": str(Path(args.config).resolve()),
            "state": {
                "smoke_ready": False,
                "eval_smoke_ready": False,
                "transfer_smoke_ready": False,
                "export_smoke_ready": False,
                "training_ready": False,
                "downstream_contract_ready": False,
            },
            "blocked_reasons": [str(exc)],
        }
        output_path = args.output or "outputs/canonical/eval/shared_private_smoke/downstream_contract_audit.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["evaluation"]["output_dir"]).resolve() / "downstream_contract_audit.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    print(f"Downstream contract ready: {report['state']['downstream_contract_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
