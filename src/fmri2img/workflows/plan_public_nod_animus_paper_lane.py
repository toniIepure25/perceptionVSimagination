from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.plan_public_nod_animus_paper_lane")

from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only.yaml"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/paper_lane_plan.json"
EXPECTED_EXPERIMENT_NAME = "public_nod_imagenet_run10_shared_only"
PAPER_LANE_NAME = "paper2_public_nod_imagenet_run10_shared_only_animus"
SMOKE_ARTIFACTS = {
    "trainer_preflight_report": "outputs/public_nod/train/trainer_preflight.json",
    "preflight_data_report": "outputs/public_nod/train/imagenet_run10_shared_only_preflight/preflight_data.json",
    "smoke_report": "outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json",
    "eval_export_smoke_report": "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json",
    "downstream_contract_audit": "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json",
}
OPERATING_BOUNDARY = [
    "this report defines a separate paper-2 planning surface for the fixed public NOD shared-only lane and does not modify the full_imagery_overlap_shared_only paper direction",
    "current public-NOD smoke artifacts remain operational surfaces only; this plan does not convert them into benchmark evidence",
    "training_ready remains false and evidence_ready_candidate remains false for this public-data paper lane until dedicated non-smoke research artifacts exist",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return (_repo_root() / relative).resolve()


def _resolve_required_public_nod_path(config, key: str) -> Path:
    value = config.get(f"public_nod.{key}")
    if value is None:
        raise KeyError(f"Missing public_nod.{key} in config.")
    path = _default_path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Required public-NOD artifact is missing: public_nod.{key}={path}")
    return path


def _resolve_required_smoke_path(key: str) -> Path:
    path = _default_path(SMOKE_ARTIFACTS[key])
    if not path.exists():
        raise FileNotFoundError(f"Required public-NOD smoke artifact is missing: {path}")
    return path


def _canonical_preflight_status(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if status is None and isinstance(payload.get("readiness"), dict):
        status = payload["readiness"].get("status")
    return status


def _load_frame_summary(path: Path, *, split_column: str | None = None) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    summary: dict[str, Any] = {"rows": int(len(frame))}
    if "pair_id" in frame.columns:
        summary["unique_pair_ids"] = int(frame["pair_id"].nunique())
    if split_column and split_column in frame.columns:
        summary["split_counts"] = {str(key): int(value) for key, value in frame[split_column].value_counts().to_dict().items()}
    return summary


def _lane_identity(config) -> dict[str, Any]:
    return {
        "paper_lane_name": PAPER_LANE_NAME,
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "source_config": "configs/canonical/public_nod_imagenet_run10_shared_only.yaml",
        "separate_from_full_overlap_lane": True,
        "separation_reasons": [
            "full_imagery_overlap_shared_only is the current paired benchmark lane and paper-1 reference, while this new lane is public-data-only and perception-only",
            "the public NOD lane is better suited to Animus-facing reliability and deployment questions than to the current paired threshold hypothesis",
            "the new paper thread should reuse the existing fixed public slice without rewriting the full-overlap evidence narrative",
        ],
        "animus_subsystem_value": [
            "the fixed NOD slice already exercises the shared-only decoder, export manifest, decoder card, and downstream contract surfaces on public data",
            "a public-data reliability paper can strengthen confidence-bearing export and deployment trust for Animus without claiming paired-threshold evidence",
        ],
        "public_data_feasible_now": True,
        "public_data_feasible_reasons": [
            "the fixed NOD public slice already has a checked-in config and reproducible prepared dataset contract",
            "target cache, ROI artifact, trainer preflight, smoke train/eval/transfer/export, and downstream contract artifacts already exist on the live pod",
            "the lane stays entirely inside public-data boundaries and does not depend on the richer external paired source required by the full-overlap lane",
        ],
    }


def _paper_direction() -> dict[str, Any]:
    return {
        "direction_id": "reliability_aware_public_shared_only_decoder",
        "headline": "Reliability-aware shared-only decoder for public natural-image fMRI with Animus-facing confidence export",
        "why_this_is_stronger_than_plain_baseline": [
            "a plain shared-only public baseline paper would largely restate the existing operational smoke story without adding a distinct scientific contribution",
            "the public NOD lane is perception-only, so it is better positioned to study trust, confidence, ROI stability, and session robustness than the paired threshold question",
            "Animus relevance is strongest when the paper asks which public shared-only outputs are reliable enough to export and trust, not merely whether a baseline can run",
        ],
        "why_public_data_fits_this_direction": [
            "the fixed public NOD slice already supports repeated-session, multi-subject, perception-only decoding on public data",
            "the existing export and downstream contract surfaces provide a natural foundation for confidence-bearing public-data outputs",
        ],
    }


def _research_questions() -> list[dict[str, str]]:
    return [
        {
            "id": "public_confidence_signal",
            "question": "Can a public-data shared-only decoder expose calibrated confidence signals that improve downstream Animus trust decisions over raw cosine alone?",
        },
        {
            "id": "roi_reliability",
            "question": "Which ROI groups contribute most to stable public-NOD decoding quality and confidence on the fixed Imagenet run-10 slice?",
        },
        {
            "id": "session_subject_stability",
            "question": "How stable are decoding outputs and confidence estimates across sessions and subjects on the fixed public slice?",
        },
        {
            "id": "low_trust_detection",
            "question": "Can uncertainty-aware scores identify low-trust outputs better than uncalibrated similarity metrics on the public NOD lane?",
        },
        {
            "id": "export_reliability_contract",
            "question": "Can the current Animus export contract be extended with reliability metadata without breaking the existing public shared-only downstream surface?",
        },
    ]


def _experiment_ladder() -> dict[str, list[dict[str, Any]]]:
    return {
        "baseline": [
            {
                "stage_id": "paper2_non_smoke_shared_only_baseline",
                "goal": "Create the first dedicated non-smoke public NOD shared-only baseline with separated paper-2 outputs.",
                "required_surfaces": [
                    "dedicated checked-in paper-2 baseline config derived from the fixed public NOD slice",
                    "non-smoke train/eval/transfer/export artifacts under outputs/public_nod/paper2/",
                ],
            }
        ],
        "ablations": [
            {
                "stage_id": "paper2_shared_capacity_ablation",
                "goal": "Measure whether the current shared-only width is over- or under-sized for the fixed public NOD slice.",
                "examples": ["shared_dim down", "shared_dim up", "reconstruction_weight control"],
            }
        ],
        "uncertainty_reliability": [
            {
                "stage_id": "paper2_posthoc_confidence_baseline",
                "goal": "Add a post-hoc reliability baseline that calibrates confidence from held-out validation behavior without changing the public-data contract.",
                "examples": ["cosine-to-risk calibration", "residual-norm confidence", "temperature-scaled low-trust flags"],
            }
        ],
        "roi_ablations": [
            {
                "stage_id": "paper2_roi_group_ablation",
                "goal": "Quantify which ROI groups matter most for stable public shared-only decoding and reliability.",
                "examples": ["early_visual only", "metacognitive only", "drop-one-group controls"],
            }
        ],
        "session_subject_robustness": [
            {
                "stage_id": "paper2_session_subject_robustness",
                "goal": "Measure subject-level and session-level robustness on the fixed public NOD slice.",
                "examples": ["leave-one-subject summaries", "per-session stability tables", "cross-session confidence drift"],
            }
        ],
        "downstream_export_confidence_audit": [
            {
                "stage_id": "paper2_confidence_export_audit",
                "goal": "Extend the Animus export inspection path with reliability metadata and verify downstream compatibility.",
                "examples": ["confidence-bearing manifest metadata", "decoder card reliability block", "export inspector trust summary"],
            }
        ],
    }


def _repo_surfaces() -> dict[str, list[str]]:
    return {
        "configs": [
            "configs/canonical/public_nod_imagenet_run10_shared_only.yaml",
            "configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml",
        ],
        "workflows": [
            "fmri2img.workflows.preflight_public_nod_shared_only_trainer",
            "fmri2img.workflows.train_decoder",
            "fmri2img.workflows.eval_decoder",
            "fmri2img.workflows.eval_transfer",
            "fmri2img.workflows.export_for_animus",
            "fmri2img.workflows.report_public_nod_shared_only_smoke",
            "fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke",
            "fmri2img.workflows.audit_public_nod_shared_only_downstream_contract",
            "fmri2img.workflows.plan_public_nod_animus_paper_lane",
        ],
        "outputs": [
            "cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet",
            "cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet",
            "cache/indices/public_nod/imagenet_run10_roi_materialized.parquet",
            "outputs/public_nod/train/trainer_preflight.json",
            "outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json",
            "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json",
            "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json",
            "outputs/public_nod/paper2/imagenet_run10_shared_only/paper_lane_plan.json",
        ],
        "docs": [
            "docs/PAPER_2_PUBLIC_NOD_ANIMUS_LANE.md",
            "docs/NOD_PUBLIC_DATASET.md",
            "docs/ANIMUS_CORE_DECODER.md",
        ],
    }


def _immediate_next_step() -> dict[str, str]:
    return {
        "step_id": "materialize_public_nod_paper2_plan",
        "description": "Write the separate machine-readable paper-2 plan artifact for the public NOD shared-only lane.",
        "command": "./.venv/bin/python -m fmri2img.workflows.plan_public_nod_animus_paper_lane --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml",
        "why_this_is_first": "it freezes the new lane identity, publication gaps, and experiment ladder without modifying the existing full-overlap paper lane or pretending the smoke bundle is evidence",
    }


def _build_publication_assessment() -> dict[str, list[str]]:
    return {
        "already_strong": [
            "the fixed public NOD slice is already reproducibly prepared, ROI-materialized, target-cached, and trainer-preflighted on public data",
            "the public shared-only smoke bundle already proves train/eval/transfer/export execution and a stable downstream contract on the live pod",
            "normalized target metadata and normalized condition semantics are already stable across the current public-NOD downstream surfaces",
        ],
        "operational_only": [
            "current train/eval/transfer/export outputs are smoke-only and remain practical Animus operational surfaces rather than benchmark evidence",
            "the fixed public NOD slice is perception-only, so it cannot answer the paired threshold question that drives the full-overlap paper lane",
            "the current public-NOD export contract is strong enough for subsystem hardening, but not yet for a publishable confidence or robustness claim",
        ],
        "missing_before_publishable": [
            "a dedicated non-smoke paper-2 baseline run with separated outputs",
            "a reliability or uncertainty surface stronger than raw smoke metrics alone",
            "ROI ablations and session-or-subject robustness analyses on the fixed public slice",
            "a confidence-bearing downstream export or audit surface that stays compatible with Animus consumption",
        ],
    }


def _expected_artifact_paths(config: Any | None) -> dict[str, str]:
    artifact_paths = {
        "trainer_preflight_report": str(_default_path(SMOKE_ARTIFACTS["trainer_preflight_report"])),
        "preflight_data_report": str(_default_path(SMOKE_ARTIFACTS["preflight_data_report"])),
        "smoke_report": str(_default_path(SMOKE_ARTIFACTS["smoke_report"])),
        "eval_export_smoke_report": str(_default_path(SMOKE_ARTIFACTS["eval_export_smoke_report"])),
        "downstream_contract_audit": str(_default_path(SMOKE_ARTIFACTS["downstream_contract_audit"])),
    }
    if config is None:
        return artifact_paths
    artifact_paths.update(
        {
            "prepared_dataset": str(Path(config["dataset"]["mixed_index"]).resolve()),
            "prepared_report": str(_default_path(str(config.get("public_nod.prepared_report")))),
            "target_cache": str(Path(config["targets"]["cache_path"]).resolve()),
            "target_cache_report": str(_default_path(str(config.get("public_nod.target_cache_report")))),
            "roi_artifact": str(_default_path(str(config.get("public_nod.roi_artifact")))),
            "roi_report": str(_default_path(str(config.get("public_nod.roi_report")))),
        }
    )
    return artifact_paths


def _build_blocked_report(*, config_path: str | Path, artifact_paths: dict[str, str], blocked_reasons: list[str]) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve()),
        "lane_identity": _lane_identity(None),
        "artifact_paths": artifact_paths,
        "repo_grounded_current_state": {},
        "publication_assessment": _build_publication_assessment(),
        "paper_direction": _paper_direction(),
        "research_questions": _research_questions(),
        "experiment_ladder": _experiment_ladder(),
        "repo_surfaces": _repo_surfaces(),
        "immediate_next_step": _immediate_next_step(),
        "state": {
            "operational_ready": False,
            "downstream_contract_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "paper_lane_plan_ready": False,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": OPERATING_BOUNDARY,
    }


def build_public_nod_animus_paper_lane_plan(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Public-NOD paper lane plan requires configs/canonical/public_nod_imagenet_run10_shared_only.yaml."
        )

    prepared_report_path = _resolve_required_public_nod_path(config, "prepared_report")
    target_cache_report_path = _resolve_required_public_nod_path(config, "target_cache_report")
    roi_report_path = _resolve_required_public_nod_path(config, "roi_report")
    roi_artifact_path = _resolve_required_public_nod_path(config, "roi_artifact")
    prepared_dataset_path = Path(config["dataset"]["mixed_index"]).resolve()
    target_cache_path = Path(config["targets"]["cache_path"]).resolve()
    if not prepared_dataset_path.exists():
        raise FileNotFoundError(f"Required public-NOD prepared dataset is missing: {prepared_dataset_path}")
    if not target_cache_path.exists():
        raise FileNotFoundError(f"Required public-NOD target cache is missing: {target_cache_path}")
    smoke_paths = {key: _resolve_required_smoke_path(key) for key in SMOKE_ARTIFACTS}

    prepared_report = load_json(prepared_report_path)
    target_cache_report = load_json(target_cache_report_path)
    roi_report = load_json(roi_report_path)
    trainer_preflight = load_json(smoke_paths["trainer_preflight_report"])
    preflight_data = load_json(smoke_paths["preflight_data_report"])
    smoke_report = load_json(smoke_paths["smoke_report"])
    eval_export_smoke = load_json(smoke_paths["eval_export_smoke_report"])
    downstream_contract = load_json(smoke_paths["downstream_contract_audit"])

    prepared_summary = _load_frame_summary(prepared_dataset_path, split_column="split")
    target_cache_summary = _load_frame_summary(target_cache_path)
    roi_summary = _load_frame_summary(roi_artifact_path)
    expected_pair_rows = int(config.get("public_nod.pair_rows", 3600))

    operational_ready = bool(
        prepared_report.get("state", {}).get("join_ready")
        and prepared_report.get("state", {}).get("roi_ready")
        and prepared_report.get("state", {}).get("downstream_prep_ready")
        and trainer_preflight.get("state", {}).get("trainer_config_ready")
        and trainer_preflight.get("state", {}).get("preflight_ready")
        and smoke_report.get("state", {}).get("smoke_ready")
        and eval_export_smoke.get("state", {}).get("eval_smoke_ready")
        and eval_export_smoke.get("state", {}).get("transfer_smoke_ready")
        and eval_export_smoke.get("state", {}).get("export_smoke_ready")
        and downstream_contract.get("state", {}).get("downstream_contract_ready")
    )
    downstream_contract_ready = bool(downstream_contract.get("state", {}).get("downstream_contract_ready"))

    return {
        "config": str(Path(config_path).resolve()),
        "lane_identity": _lane_identity(config),
        "artifact_paths": {
            "prepared_dataset": str(prepared_dataset_path),
            "prepared_report": str(prepared_report_path),
            "target_cache": str(target_cache_path),
            "target_cache_report": str(target_cache_report_path),
            "roi_artifact": str(roi_artifact_path),
            "roi_report": str(roi_report_path),
            "trainer_preflight_report": str(smoke_paths["trainer_preflight_report"]),
            "preflight_data_report": str(smoke_paths["preflight_data_report"]),
            "smoke_report": str(smoke_paths["smoke_report"]),
            "eval_export_smoke_report": str(smoke_paths["eval_export_smoke_report"]),
            "downstream_contract_audit": str(smoke_paths["downstream_contract_audit"]),
        },
        "repo_grounded_current_state": {
            "fixed_slice_contract": {
                "dataset_id": config.get("public_nod.dataset_id"),
                "lane": config.get("public_nod.lane"),
                "task": config.get("public_nod.task"),
                "run": int(config.get("public_nod.run", 10)),
                "subjects": list(config.get("public_nod.subjects", [])),
                "sessions": list(config.get("public_nod.sessions", [])),
                "adapter_rows": int(config.get("public_nod.adapter_rows", 36)),
                "pair_rows": expected_pair_rows,
            },
            "prepared_dataset": {
                **prepared_summary,
                "matches_fixed_pair_row_contract": prepared_summary["rows"] == expected_pair_rows,
                "state": prepared_report.get("state", {}),
            },
            "target_cache": {
                **target_cache_summary,
                "matches_fixed_pair_row_contract": target_cache_summary["rows"] == expected_pair_rows,
                "embedding_dimension": int(target_cache_report.get("embedding_dimension", -1)),
                "embedding_model_id": target_cache_report.get("embedding_model_id"),
                "state": target_cache_report.get("state", {}),
            },
            "roi_artifact": {
                **roi_summary,
                "matches_fixed_pair_row_contract": roi_summary["rows"] == expected_pair_rows,
                "roi_feature_dimensions": roi_report.get("roi_feature_dimensions", {}),
                "state": roi_report.get("state", {}),
            },
            "trainer_preflight": {
                "state": trainer_preflight.get("state", {}),
                "trainer_packet": trainer_preflight.get("trainer_packet", {}),
                "prepared_dataset": trainer_preflight.get("prepared_dataset", {}),
            },
            "preflight_data": {
                "status": _canonical_preflight_status(preflight_data),
                "readiness": preflight_data.get("readiness"),
            },
            "smoke_train": {
                "state": smoke_report.get("state", {}),
                "smoke_run": smoke_report.get("smoke_run", {}),
            },
            "smoke_eval_transfer_export": {
                "state": eval_export_smoke.get("state", {}),
                "condition_semantics": eval_export_smoke.get("condition_semantics", {}),
                "target_spec": eval_export_smoke.get("target_spec", {}),
                "eval_smoke": eval_export_smoke.get("eval_smoke", {}),
                "transfer_smoke": eval_export_smoke.get("transfer_smoke", {}),
                "export_smoke": eval_export_smoke.get("export_smoke", {}),
            },
            "downstream_contract": {
                "state": downstream_contract.get("state", {}),
                "condition_semantics": downstream_contract.get("condition_semantics", {}),
                "target_spec": downstream_contract.get("target_spec", {}),
                "consistency": downstream_contract.get("consistency", {}),
            },
        },
        "publication_assessment": _build_publication_assessment(),
        "paper_direction": _paper_direction(),
        "research_questions": _research_questions(),
        "experiment_ladder": _experiment_ladder(),
        "repo_surfaces": _repo_surfaces(),
        "immediate_next_step": _immediate_next_step(),
        "state": {
            "operational_ready": operational_ready,
            "downstream_contract_ready": downstream_contract_ready,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "paper_lane_plan_ready": True,
        },
        "blocked_reasons": [],
        "operational_boundary": OPERATING_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a separate paper-2 planning report for the public NOD shared-only Animus lane."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    config = None
    try:
        config = load_workflow_config(args.config, args.override)
        report = build_public_nod_animus_paper_lane_plan(config, config_path=args.config)
    except Exception as exc:
        report = _build_blocked_report(
            config_path=args.config,
            artifact_paths=_expected_artifact_paths(config),
            blocked_reasons=[str(exc)],
        )
        output_path = args.output or DEFAULT_OUTPUT
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or DEFAULT_OUTPUT
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    print(f"Paper lane plan ready: {report['state']['paper_lane_plan_ready']}")
    print(f"Operational ready: {report['state']['operational_ready']}")
    print(f"Downstream contract ready: {report['state']['downstream_contract_ready']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
