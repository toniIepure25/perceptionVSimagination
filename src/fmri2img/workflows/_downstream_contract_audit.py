from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fmri2img.evaluation import normalize_condition_semantics_payload
from fmri2img.export.animus import normalize_target_spec_payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_decoder_card_target(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return {
        "target_name_normalized": payload.get("name"),
        "target_dimension_normalized": payload.get("dimension"),
        "source_field_shape": payload.get("source_field_shape"),
        "target_name_from_payload": payload.get("target_name_from_payload", payload.get("name")),
    }


def normalize_manifest_target(payload: dict[str, Any] | None, *, fallback_target_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    if payload:
        return normalize_target_spec_payload(
            {
                "target_name": payload.get("target_name_normalized"),
                "name": payload.get("target_name_from_payload"),
                "dimension": payload.get("target_dimension_normalized"),
            }
        ) | {
            "source_field_shape": payload.get("source_field_shape", "missing"),
            "target_name_from_payload": payload.get("target_name_from_payload", payload.get("target_name_normalized")),
        }
    return normalize_target_spec_payload(fallback_target_spec or {})


def surface_condition_semantics(payload: dict[str, Any] | None) -> dict[str, Any]:
    return normalize_condition_semantics_payload(payload or {})


def merge_condition_semantics(*payloads: dict[str, Any]) -> dict[str, Any]:
    normalized = [normalize_condition_semantics_payload(payload) for payload in payloads]
    informative = [
        item
        for item in normalized
        if item["present_conditions"] or item["pair_metrics_available_from_payload"] is not None
    ]
    if not informative:
        informative = normalized
    shared = informative[0] if informative else normalize_condition_semantics_payload({})
    consistent = all(item == shared for item in informative[1:]) if len(informative) > 1 else True
    merged = {"shared": shared, "consistent_across_sources": consistent}
    for index, item in enumerate(normalized):
        merged[f"payload_{index}"] = item
    return merged


def build_downstream_contract_audit_report(
    *,
    config_path: str | Path,
    artifact_paths: dict[str, str],
    target_spec: dict[str, dict[str, Any]],
    condition_semantics: dict[str, dict[str, Any]],
    identity: dict[str, dict[str, Any]],
    state: dict[str, Any],
    target_checks: list[dict[str, str]],
    condition_checks: list[dict[str, str]],
    fixed_contract: dict[str, Any] | None = None,
    extra_sections: dict[str, Any] | None = None,
    operational_boundary: list[str] | None = None,
    target_dimension_reason: str = "target dimension drift detected across downstream contract surfaces",
    benchmark_role_reason: str = "benchmark role drift detected across downstream contract surfaces",
    experiment_name_reason: str = "experiment name drift detected across downstream contract surfaces",
) -> dict[str, Any]:
    shared_target = target_spec["shared"]
    shared_condition = condition_semantics["shared"]
    blocked_reasons: list[str] = []
    checks: dict[str, bool] = {}

    for item in target_checks:
        payload = target_spec[item["surface_key"]]
        ok = shared_target == payload
        checks[item["check_name"]] = ok
        if not ok:
            blocked_reasons.append(
                f"normalized target metadata differs between {item['shared_label']} and {item['surface_label']}"
            )

    for item in condition_checks:
        payload = condition_semantics[item["surface_key"]]
        ok = shared_condition == payload
        checks[item["check_name"]] = ok
        if not ok:
            blocked_reasons.append(
                f"normalized condition semantics differ between {item['shared_label']} and {item['surface_label']}"
            )

    target_views = [payload for payload in target_spec.values() if isinstance(payload, dict)]
    target_dimensions = {payload.get("target_dimension_normalized") for payload in target_views}
    checks["target_dimension_consistent"] = len(target_dimensions) == 1
    if not checks["target_dimension_consistent"]:
        blocked_reasons.append(target_dimension_reason)

    checks["source_field_shape_explicit"] = all(
        payload.get("source_field_shape") in {"name", "target_name"} for payload in target_views
    )
    if not checks["source_field_shape_explicit"]:
        blocked_reasons.append("normalized target metadata is missing an explicit source_field_shape")

    experiment_name_values = list(identity.get("experiment_name", {}).values())
    checks["experiment_name_consistent"] = len({value for value in experiment_name_values}) == 1
    if not checks["experiment_name_consistent"]:
        blocked_reasons.append(experiment_name_reason)

    benchmark_role_values = list(identity.get("benchmark_role", {}).values())
    checks["benchmark_role_consistent"] = len({value for value in benchmark_role_values}) == 1
    if not checks["benchmark_role_consistent"]:
        blocked_reasons.append(benchmark_role_reason)

    checks["readiness_operational_only"] = (
        bool(state.get("eval_smoke_ready"))
        and bool(state.get("transfer_smoke_ready"))
        and bool(state.get("export_smoke_ready"))
        and not bool(state.get("training_ready"))
    )
    if not checks["readiness_operational_only"]:
        blocked_reasons.append("downstream readiness is not in the expected operational-only state")

    downstream_contract_ready = len(blocked_reasons) == 0
    report = {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": artifact_paths,
        "target_spec": target_spec,
        "condition_semantics": condition_semantics,
        "identity": identity,
        "consistency": checks,
        "state": {
            **state,
            "downstream_contract_ready": downstream_contract_ready,
            "training_ready": False,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": operational_boundary or [],
    }
    if fixed_contract is not None:
        report["fixed_contract"] = fixed_contract
    if extra_sections:
        report.update(extra_sections)
    return report
