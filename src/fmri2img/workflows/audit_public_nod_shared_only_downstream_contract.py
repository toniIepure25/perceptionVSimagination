from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_public_nod_shared_only_downstream_contract")

from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows._downstream_contract_audit import (  # noqa: E402
    build_downstream_contract_audit_report,
    load_json,
    normalize_decoder_card_target,
    normalize_manifest_target,
    surface_condition_semantics,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


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

    manifest = load_json(manifest_path)
    decoder_card = load_json(decoder_card_path)
    combined_report = load_json(combined_report_path)

    report = build_downstream_contract_audit_report(
        config_path=config_path,
        artifact_paths={
            "export_manifest": str(manifest_path.resolve()),
            "export_decoder_card": str(decoder_card_path.resolve()),
            "combined_report": str(combined_report_path.resolve()),
        },
        fixed_contract={
            "dataset_id": config.get("public_nod.dataset_id"),
            "task": config.get("public_nod.task"),
            "subjects": list(config.get("public_nod.subjects", [])),
            "sessions": list(config.get("public_nod.sessions", [])),
            "run": int(config.get("public_nod.run", 10)),
            "adapter_rows": int(config.get("public_nod.adapter_rows", 36)),
            "pair_rows": int(config.get("public_nod.pair_rows", 3600)),
        },
        target_spec={
            "shared": normalize_manifest_target(
                manifest.get("metadata", {}).get("target_spec_normalized"),
                fallback_target_spec=manifest.get("target_spec"),
            ),
            "decoder_card": normalize_decoder_card_target(decoder_card.get("target")),
            "combined_report": combined_report.get("target_spec", {}),
        },
        condition_semantics={
            "shared": surface_condition_semantics(manifest.get("metadata", {}).get("condition_semantics")),
            "decoder_card": surface_condition_semantics(decoder_card.get("condition_semantics")),
            "combined_report": surface_condition_semantics(combined_report.get("condition_semantics", {}).get("shared")),
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
                "combined_report": combined_report.get("export_smoke", {}).get("decoder_card_benchmark_role"),
            },
        },
        state={
            "eval_smoke_ready": bool(combined_report.get("state", {}).get("eval_smoke_ready")),
            "transfer_smoke_ready": bool(combined_report.get("state", {}).get("transfer_smoke_ready")),
            "export_smoke_ready": bool(combined_report.get("state", {}).get("export_smoke_ready")),
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
                "surface_key": "combined_report",
                "check_name": "target_manifest_vs_combined_report",
                "shared_label": "export manifest",
                "surface_label": "combined smoke report",
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
                "surface_key": "combined_report",
                "check_name": "condition_manifest_vs_combined_report",
                "shared_label": "export manifest",
                "surface_label": "combined smoke report",
            },
        ],
        target_dimension_reason="target dimension drift detected across manifest, decoder card, and combined smoke report",
        benchmark_role_reason="benchmark role drift detected across manifest, decoder card, and combined smoke report",
        experiment_name_reason="experiment name drift detected across manifest, decoder card, or fixed smoke config",
        operational_boundary=[
            "this audit only verifies internal consistency across the fixed NOD smoke export and report artifacts",
            "this is an operational downstream contract verdict, not a benchmark or evidence-facing result",
            "training_ready remains false even when downstream_contract_ready is true",
        ],
    )
    return report


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
