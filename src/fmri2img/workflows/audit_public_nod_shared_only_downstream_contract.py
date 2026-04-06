from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_public_nod_shared_only_downstream_contract")

from fmri2img.evaluation import normalize_condition_semantics_payload  # noqa: E402
from fmri2img.export.animus import normalize_target_spec_payload  # noqa: E402
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _surface_condition_semantics(payload: dict[str, Any] | None) -> dict[str, Any]:
    return normalize_condition_semantics_payload(payload or {})


def _normalize_decoder_card_target(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return {
        "target_name_normalized": payload.get("name"),
        "target_dimension_normalized": payload.get("dimension"),
        "source_field_shape": payload.get("source_field_shape"),
        "target_name_from_payload": payload.get("target_name_from_payload", payload.get("name")),
    }


def _bool(value: Any) -> bool:
    return bool(value)


def build_public_nod_shared_only_downstream_contract_audit(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    expected_name = "public_nod_imagenet_run10_shared_only_smoke"
    if str(config.get("experiment.name", "")) != expected_name:
        raise ValueError("Downstream contract audit requires the checked-in fixed-slice NOD smoke config.")

    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()
    manifest_path = export_output_dir / "manifest.json"
    decoder_card_path = export_output_dir / "decoder_card.json"
    combined_report_path = eval_output_dir / "eval_export_smoke_report.json"

    missing = [str(path) for path in (manifest_path, decoder_card_path, combined_report_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Downstream contract audit requires real smoke artifacts: " + ", ".join(missing))

    manifest = _load_json(manifest_path)
    decoder_card = _load_json(decoder_card_path)
    combined_report = _load_json(combined_report_path)

    manifest_target = manifest.get("metadata", {}).get("target_spec_normalized")
    if not isinstance(manifest_target, dict):
        manifest_target = normalize_target_spec_payload(manifest.get("target_spec", {}))
    decoder_target = _normalize_decoder_card_target(decoder_card.get("target"))
    report_target = combined_report.get("target_spec")
    if not isinstance(report_target, dict):
        report_target = normalize_target_spec_payload(manifest.get("target_spec", {}))

    manifest_condition = _surface_condition_semantics(manifest.get("metadata", {}).get("condition_semantics"))
    decoder_condition = _surface_condition_semantics(decoder_card.get("condition_semantics"))
    report_condition = combined_report.get("condition_semantics", {}).get("shared")
    if not isinstance(report_condition, dict):
        report_condition = normalize_condition_semantics_payload({})

    experiment_name_manifest = manifest.get("metadata", {}).get("experiment", {}).get("name")
    experiment_name_card = decoder_card.get("experiment", {}).get("name")
    benchmark_role_manifest = manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role")
    benchmark_role_card = decoder_card.get("experiment", {}).get("benchmark_role")
    benchmark_role_report = combined_report.get("export_smoke", {}).get("decoder_card_benchmark_role")

    blocked_reasons: list[str] = []

    checks = {
        "target_manifest_vs_decoder_card": manifest_target == decoder_target,
        "target_manifest_vs_combined_report": manifest_target == report_target,
        "condition_manifest_vs_decoder_card": manifest_condition == decoder_condition,
        "condition_manifest_vs_combined_report": manifest_condition == report_condition,
        "experiment_name_consistent": experiment_name_manifest == experiment_name_card == expected_name,
        "benchmark_role_consistent": benchmark_role_manifest == benchmark_role_card == benchmark_role_report,
        "target_dimension_consistent": (
            manifest_target.get("target_dimension_normalized")
            == decoder_target.get("target_dimension_normalized")
            == report_target.get("target_dimension_normalized")
        ),
        "source_field_shape_explicit": all(
            item.get("source_field_shape") in {"name", "target_name"}
            for item in (manifest_target, decoder_target, report_target)
        ),
        "readiness_operational_only": (
            _bool(combined_report.get("state", {}).get("eval_smoke_ready"))
            and _bool(combined_report.get("state", {}).get("transfer_smoke_ready"))
            and _bool(combined_report.get("state", {}).get("export_smoke_ready"))
            and not _bool(combined_report.get("state", {}).get("training_ready"))
        ),
    }

    if not checks["target_manifest_vs_decoder_card"]:
        blocked_reasons.append("normalized target metadata differs between export manifest and decoder card")
    if not checks["target_manifest_vs_combined_report"]:
        blocked_reasons.append("normalized target metadata differs between export manifest and combined smoke report")
    if not checks["condition_manifest_vs_decoder_card"]:
        blocked_reasons.append("normalized condition semantics differ between export manifest and decoder card")
    if not checks["condition_manifest_vs_combined_report"]:
        blocked_reasons.append("normalized condition semantics differ between export manifest and combined smoke report")
    if not checks["experiment_name_consistent"]:
        blocked_reasons.append("experiment name drift detected across manifest, decoder card, or fixed smoke config")
    if not checks["benchmark_role_consistent"]:
        blocked_reasons.append("benchmark role drift detected across manifest, decoder card, and combined smoke report")
    if not checks["target_dimension_consistent"]:
        blocked_reasons.append("target dimension drift detected across manifest, decoder card, and combined smoke report")
    if not checks["source_field_shape_explicit"]:
        blocked_reasons.append("normalized target metadata is missing an explicit source_field_shape")
    if not checks["readiness_operational_only"]:
        blocked_reasons.append("combined smoke report readiness is not in the expected operational-only state")

    downstream_contract_ready = len(blocked_reasons) == 0
    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {
            "export_manifest": str(manifest_path.resolve()),
            "export_decoder_card": str(decoder_card_path.resolve()),
            "combined_report": str(combined_report_path.resolve()),
        },
        "fixed_contract": {
            "dataset_id": config.get("public_nod.dataset_id"),
            "task": config.get("public_nod.task"),
            "subjects": list(config.get("public_nod.subjects", [])),
            "sessions": list(config.get("public_nod.sessions", [])),
            "run": int(config.get("public_nod.run", 10)),
            "adapter_rows": int(config.get("public_nod.adapter_rows", 36)),
            "pair_rows": int(config.get("public_nod.pair_rows", 3600)),
        },
        "target_spec": {
            "shared": manifest_target,
            "decoder_card": decoder_target,
            "combined_report": report_target,
        },
        "condition_semantics": {
            "shared": manifest_condition,
            "decoder_card": decoder_condition,
            "combined_report": report_condition,
        },
        "identity": {
            "experiment_name": {
                "manifest": experiment_name_manifest,
                "decoder_card": experiment_name_card,
                "expected_config": expected_name,
            },
            "benchmark_role": {
                "manifest": benchmark_role_manifest,
                "decoder_card": benchmark_role_card,
                "combined_report": benchmark_role_report,
            },
        },
        "consistency": checks,
        "state": {
            "downstream_contract_ready": downstream_contract_ready,
            "eval_smoke_ready": _bool(combined_report.get("state", {}).get("eval_smoke_ready")),
            "transfer_smoke_ready": _bool(combined_report.get("state", {}).get("transfer_smoke_ready")),
            "export_smoke_ready": _bool(combined_report.get("state", {}).get("export_smoke_ready")),
            "training_ready": False,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": [
            "this audit only verifies internal consistency across the fixed NOD smoke export and report artifacts",
            "this is an operational downstream contract verdict, not a benchmark or evidence-facing result",
            "training_ready remains false even when downstream_contract_ready is true",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit downstream export/report contract consistency for the fixed NOD smoke bundle.")
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_public_nod_shared_only_downstream_contract_audit(config, config_path=args.config)
    except Exception as exc:
        report = {
            "config": str(Path(args.config).resolve()),
            "state": {
                "downstream_contract_ready": False,
                "eval_smoke_ready": False,
                "transfer_smoke_ready": False,
                "export_smoke_ready": False,
                "training_ready": False,
            },
            "blocked_reasons": [str(exc)],
        }
        output_path = args.output or "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json"
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
