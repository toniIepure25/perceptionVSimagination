from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.plan_full_imagery_overlap_external_rebuild")

from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.audit_full_imagery_overlap_external_source_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_EXPERIMENT_NAME,
    OPERATIONAL_BOUNDARY as EXTERNAL_SOURCE_OPERATIONAL_BOUNDARY,
    REQUIRED_PROVENANCE_FIELDS,
)
from fmri2img.workflows.common import load_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_MOUNT_REQUEST = "configs/acquisition/full_overlap_external_mount_request.json"
DEFAULT_MANIFEST_TEMPLATE = "configs/external_sources/nsd_imagery_external_manifest.template.json"
OPERATIONAL_BOUNDARY = [
    "this plan only hardens operator handoff for the existing full-overlap shared-only lane and does not change the 32-pair readiness gate",
    "rebuild_should_proceed becomes true only when the external-source readiness audit has already verified a mounted source, explicit provenance, and measured paired-support gains over the current 5-total / 1-held-out ceiling",
    "this plan does not claim benchmark progress, evidence-grade validation, production Animus readiness, or training_ready=true on its own",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return (_repo_root() / relative).resolve()


def _validate_config(config) -> None:
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Full-overlap external rebuild plan requires configs/canonical/full_imagery_overlap_shared_only.yaml."
        )


def _build_prepare_imagery_commands(config_path: str | Path, subjects: list[str]) -> list[str]:
    return [
        "./.venv/bin/python -m fmri2img.workflows.prepare_imagery_index "
        f"--config {config_path} --override dataset.subject='\"{subject}\"'"
        for subject in subjects
    ]


def _build_rebuild_commands(config_path: str | Path, *, subjects: list[str]) -> list[str]:
    checkpoint = "outputs/canonical/train/full_imagery_overlap_shared_only/best_decoder.pt"
    return [
        "./.venv/bin/python -m fmri2img.workflows.audit_full_imagery_overlap_external_source_readiness "
        f"--config {config_path}",
        *_build_prepare_imagery_commands(config_path, subjects),
        "./.venv/bin/python -m fmri2img.workflows.prepare_overlap_bootstrap "
        f"--config {config_path} --overwrite-existing",
        "./.venv/bin/python -m fmri2img.workflows.prepare_targets "
        f"--config {config_path}",
        "./.venv/bin/python -m fmri2img.workflows.preflight_data "
        f"--config {config_path} --output outputs/canonical/eval/full_imagery_overlap_shared_only/preflight_data_full_imagery_overlap_shared_only_external.json",
        "./.venv/bin/python -m fmri2img.workflows.train_decoder "
        f"--config {config_path}",
        "./.venv/bin/python -m fmri2img.workflows.eval_decoder "
        f"--config {config_path} --checkpoint {checkpoint}",
        "./.venv/bin/python -m fmri2img.workflows.eval_transfer "
        f"--config {config_path} --checkpoint {checkpoint}",
        "./.venv/bin/python -m fmri2img.workflows.export_for_animus "
        f"--config {config_path} --checkpoint {checkpoint}",
        "./.venv/bin/python -m fmri2img.workflows.audit_full_imagery_overlap_shared_only_readiness "
        f"--config {config_path} --output outputs/canonical/eval/full_imagery_overlap_shared_only/readiness_audit.json",
    ]


def _build_blocked_report(
    *,
    config_path: str | Path,
    mount_request_path: str | Path,
    manifest_template_path: str | Path,
    message: str,
) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {
            "mount_request": str(Path(mount_request_path).resolve()),
            "manifest_template": str(Path(manifest_template_path).resolve()),
        },
        "handoff_contract": {
            "request": {},
            "manifest_template": {},
        },
        "current_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "state": {},
            "heldout_support": {},
        },
        "mount_validation": {
            "mount_contract_satisfied": False,
            "provenance_complete": False,
            "overlap_gain_measured": False,
            "current_public_source_exhausted": False,
            "external_source_not_mounted": False,
            "external_source_ready_for_rebuild": False,
        },
        "rebuild_plan": {
            "rebuild_should_proceed": False,
            "commands_when_ready": [],
            "commands_before_retry": [],
            "next_honest_move": "resolve_blocked_plan",
        },
        "blocked_reasons": [message],
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def build_full_imagery_overlap_external_rebuild_plan(
    config,
    *,
    config_path: str | Path,
    mount_request_path: str | Path = DEFAULT_MOUNT_REQUEST,
    manifest_template_path: str | Path = DEFAULT_MANIFEST_TEMPLATE,
) -> dict[str, Any]:
    _validate_config(config)

    config_path = Path(config_path).resolve()
    mount_request_path = Path(mount_request_path).resolve()
    manifest_template_path = Path(manifest_template_path).resolve()

    eval_dir = Path(config["evaluation"]["output_dir"]).resolve()
    readiness_path = eval_dir / "readiness_audit.json"
    data_expansion_path = eval_dir / "data_expansion_audit.json"
    external_readiness_path = eval_dir / "external_source_readiness_audit.json"

    for path, label in (
        (mount_request_path, "mount request"),
        (manifest_template_path, "manifest template"),
        (readiness_path, "readiness audit"),
        (data_expansion_path, "data-expansion audit"),
        (external_readiness_path, "external-source readiness audit"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Full-overlap external rebuild plan requires the current {label} at {path}.")

    mount_request = load_json(mount_request_path)
    manifest_template = load_json(manifest_template_path)
    readiness = load_json(readiness_path)
    data_expansion = load_json(data_expansion_path)
    external_readiness = load_json(external_readiness_path)

    request_subjects = [str(subject) for subject in mount_request.get("subjects_requested", [])]
    if mount_request.get("lane", {}).get("experiment_name") != EXPECTED_EXPERIMENT_NAME:
        raise ValueError("Mount request lane does not match full_imagery_overlap_shared_only.")
    if not request_subjects:
        raise ValueError("Mount request must include at least one requested subject.")
    request_required_fields = list(
        mount_request.get("external_source_contract", {})
        .get("required_provenance_manifest", {})
        .get("required_fields", REQUIRED_PROVENANCE_FIELDS)
    )
    missing_template_fields = [
        field for field in request_required_fields if manifest_template.get(field) in (None, "", [])
    ]
    template_contract_ok = not missing_template_fields

    external_state = dict(external_readiness.get("state", {}))
    external_conclusion = dict(external_readiness.get("conclusion", {}))
    external_provenance = (
        external_readiness.get("external_source_contract", {}).get("provenance", {})
        if isinstance(external_readiness.get("external_source_contract"), dict)
        else {}
    )
    overlap_potential = dict(external_readiness.get("overlap_potential", {}))
    heldout_support = dict(readiness.get("heldout_support", {}))

    mount_contract_satisfied = bool(
        external_state.get("external_source_mounted") and external_state.get("external_source_preserves_contract")
    )
    provenance_complete = bool(external_state.get("provenance_recorded")) and template_contract_ok
    overlap_gain_measured = bool(external_state.get("potential_support_exceeds_current_ceiling"))
    rebuild_should_proceed = bool(external_state.get("external_source_ready_for_rebuild")) and template_contract_ok

    blocked_reasons = list(external_readiness.get("blocked_reasons", []))
    if not template_contract_ok:
        blocked_reasons.append(
            "manifest template is missing required provenance placeholders: " + ", ".join(missing_template_fields)
        )

    commands_before_retry = [
        "./.venv/bin/python -m fmri2img.workflows.audit_full_imagery_overlap_external_source_readiness "
        f"--config {config_path}"
    ]
    commands_when_ready = _build_rebuild_commands(config_path, subjects=request_subjects)

    return {
        "config": str(config_path),
        "artifact_paths": {
            "mount_request": str(mount_request_path),
            "manifest_template": str(manifest_template_path),
            "readiness_audit": str(readiness_path),
            "data_expansion_audit": str(data_expansion_path),
            "external_source_readiness_audit": str(external_readiness_path),
            "planned_preflight_output": str(
                (_repo_root() / "outputs/canonical/eval/full_imagery_overlap_shared_only/preflight_data_full_imagery_overlap_shared_only_external.json").resolve()
            ),
        },
        "handoff_contract": {
            "request": mount_request,
            "manifest_template": {
                "path": str(manifest_template_path),
                "required_fields": request_required_fields,
                "template_contract_ok": template_contract_ok,
                "missing_required_placeholders": missing_template_fields,
            },
            "requested_subjects": request_subjects,
        },
        "current_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "state": dict(readiness.get("state", {})),
            "heldout_support": heldout_support,
            "data_ceiling_confirmed": bool(data_expansion.get("state", {}).get("data_ceiling_confirmed")),
        },
        "mount_validation": {
            "mount_contract_satisfied": mount_contract_satisfied,
            "provenance_complete": provenance_complete,
            "overlap_gain_measured": overlap_gain_measured,
            "current_public_source_exhausted": bool(external_conclusion.get("current_public_source_exhausted")),
            "external_source_not_mounted": bool(external_conclusion.get("external_source_not_mounted")),
            "external_source_ready_for_rebuild": bool(external_conclusion.get("external_source_ready_for_rebuild")),
            "detected_subjects_with_external_data": list(
                external_readiness.get("external_source_inventory", {}).get("subjects_with_detected_external_data", [])
            ),
            "provenance_manifest_path": external_provenance.get("manifest_path"),
            "external_pair_group_count_estimate": int(overlap_potential.get("external_pair_group_count_estimate", 0) or 0),
            "external_heldout_pair_group_count_estimate": int(
                overlap_potential.get("external_split_pair_group_counts_estimate", {}).get("test", 0) or 0
            ),
        },
        "rebuild_plan": {
            "rebuild_should_proceed": rebuild_should_proceed,
            "commands_before_retry": commands_before_retry,
            "commands_when_ready": commands_when_ready,
            "next_honest_move": external_conclusion.get("next_honest_move", "resolve_blocked_plan"),
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": [
            *OPERATIONAL_BOUNDARY,
            *[item for item in EXTERNAL_SOURCE_OPERATIONAL_BOUNDARY if item not in OPERATIONAL_BOUNDARY],
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan the exact full-overlap shared-only external-source rebuild once the richer NSD-style mount contract is satisfied."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--mount-request", default=str(_default_path(DEFAULT_MOUNT_REQUEST)))
    parser.add_argument("--manifest-template", default=str(_default_path(DEFAULT_MANIFEST_TEMPLATE)))
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_full_imagery_overlap_external_rebuild_plan(
            config,
            config_path=args.config,
            mount_request_path=args.mount_request,
            manifest_template_path=args.manifest_template,
        )
    except Exception as exc:
        report = _build_blocked_report(
            config_path=args.config,
            mount_request_path=args.mount_request,
            manifest_template_path=args.manifest_template,
            message=str(exc),
        )
        output_path = args.output or "outputs/canonical/eval/full_imagery_overlap_shared_only/external_rebuild_plan.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        return 1 if args.fail_on_blocked else 0

    output_path = args.output or (
        Path(config["evaluation"]["output_dir"]).resolve() / "external_rebuild_plan.json"
    )
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    if args.fail_on_blocked:
        return 0 if report["rebuild_plan"]["rebuild_should_proceed"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
