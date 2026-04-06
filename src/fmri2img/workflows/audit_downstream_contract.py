from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_downstream_contract")

from fmri2img.workflows.audit_public_nod_shared_only_downstream_contract import (  # noqa: E402
    build_public_nod_shared_only_downstream_contract_audit,
)
from fmri2img.workflows.audit_shared_private_smoke_downstream_contract import (  # noqa: E402
    build_shared_private_smoke_downstream_contract_audit,
)
from fmri2img.workflows.common import load_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/shared_private_smoke.yaml"

AUDIT_BUILDERS: dict[str, Callable] = {
    "public_nod_imagenet_run10_shared_only_smoke": build_public_nod_shared_only_downstream_contract_audit,
    "shared_private_smoke": build_shared_private_smoke_downstream_contract_audit,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def resolve_downstream_contract_audit_strategy(config) -> str:
    experiment_name = str(config.get("experiment.name", ""))
    if experiment_name in AUDIT_BUILDERS:
        return experiment_name
    raise ValueError(
        "No generic downstream contract audit strategy is registered for "
        f"experiment.name={experiment_name!r}. Supported experiments: {sorted(AUDIT_BUILDERS)}"
    )


def build_downstream_contract_audit(config, *, config_path: str | Path) -> dict:
    strategy = resolve_downstream_contract_audit_strategy(config)
    report = AUDIT_BUILDERS[strategy](config, config_path=config_path)
    report.setdefault("bundle_family", strategy)
    return report


def _blocked_report(config_path: str | Path, message: str) -> dict:
    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {},
        "target_spec": {},
        "condition_semantics": {},
        "identity": {},
        "consistency": {},
        "state": {
            "downstream_contract_ready": False,
            "eval_smoke_ready": False,
            "transfer_smoke_ready": False,
            "export_smoke_ready": False,
            "training_ready": False,
        },
        "blocked_reasons": [message],
        "operational_boundary": [
            "this generic dispatcher only supports bundle families with a registered downstream audit strategy",
            "unsupported configs are reported as blocked instead of being treated as implicitly auditable",
            "training_ready remains false in blocked dispatcher reports",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch to the canonical downstream contract audit for a supported bundle.")
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_downstream_contract_audit(config, config_path=args.config)
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/downstream_contract_audit.json"
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
