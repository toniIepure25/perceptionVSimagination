from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_full_imagery_overlap_promotion_path")

from fmri2img.data.canonical import normalize_decoder_index  # noqa: E402
from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/full_imagery_overlap_shared_only.yaml"
EXPECTED_EXPERIMENT_NAME = "full_imagery_overlap_shared_only"
DEFAULT_CANDIDATE_CONFIGS = (
    "configs/canonical/animus_core_decoder.yaml",
    "configs/canonical/threshold_shared_private_p16.yaml",
    "configs/canonical/max_available_overlap.yaml",
    "configs/canonical/multisubj_overlap_bootstrap.yaml",
)
TRAIN_FILES = ("best_decoder.pt", "config_snapshot.json", "roi_summary.json", "target_summary.json", "train_history.json")
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json")
TRANSFER_FILES = ("transfer_metrics.json", "per_trial_pairs.csv")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")
OPERATIONAL_BOUNDARY = [
    "this promotion-path audit compares only checked-in canonical lanes and real mounted artifacts that are already available in the current environment",
    "it does not weaken the full-overlap readiness gate or reinterpret the current held-out support ceiling as success",
    "a stronger promotion lane requires genuinely larger paired support or a stronger real candidate bundle, not wording changes",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _count_paired_groups(df: pd.DataFrame) -> int:
    if df.empty or "pair_id" not in df.columns or "condition" not in df.columns:
        return 0
    grouped = df.groupby("pair_id")["condition"].agg(lambda values: {str(value) for value in values})
    return int(sum({"perception", "imagery"}.issubset(values) for values in grouped))


def _heldout_pair_count_from_current_support(current_support: dict[str, Any]) -> int:
    if "heldout_pair_group_count" in current_support:
        return int(current_support.get("heldout_pair_group_count", 0) or 0)
    if "heldout_pair_count_from_metrics" in current_support:
        return int(current_support.get("heldout_pair_count_from_metrics", 0) or 0)
    split_counts = current_support.get("split_pair_group_counts", {})
    return int(split_counts.get("test", 0) or 0)


def _looks_fixture_backed(path_value: Any) -> bool:
    posix = Path(str(path_value)).as_posix()
    return posix.startswith("tests/fixtures/") or "/tests/fixtures/" in posix


def _support_summary(config) -> dict[str, Any]:
    allowed_conditions = list(config["dataset"].get("perception_conditions", ["perception"])) + list(
        config["dataset"].get("imagery_conditions", ["imagery"])
    )
    mixed_index_path = Path(config["dataset"]["mixed_index"]).resolve()
    mixed_df = normalize_decoder_index(pd.read_parquet(mixed_index_path), allowed_conditions=allowed_conditions)
    split_pair_counts = {
        split: _count_paired_groups(mixed_df[mixed_df["split"] == split].reset_index(drop=True))
        for split in ("train", "val", "test")
    }
    return {
        "mixed_index": str(mixed_index_path),
        "dataset_rows": int(len(mixed_df)),
        "split_row_counts": {split: int((mixed_df["split"] == split).sum()) for split in ("train", "val", "test")},
        "dataset_pair_group_count": _count_paired_groups(mixed_df),
        "split_pair_group_counts": split_pair_counts,
        "heldout_pair_group_count": int(split_pair_counts["test"]),
    }


def _missing_files(root: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def _bundle_status(config) -> dict[str, Any]:
    train_dir = Path(config["training"]["output_dir"]).resolve()
    eval_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_dir = Path(config["export"]["output_dir"]).resolve()
    missing = {
        "train": _missing_files(train_dir, TRAIN_FILES),
        "eval": _missing_files(eval_dir, EVAL_FILES),
        "transfer": _missing_files(transfer_dir, TRANSFER_FILES),
        "export": _missing_files(export_dir, EXPORT_FILES),
    }
    metrics_summary = {}
    if not missing["eval"]:
        metrics = load_json(eval_dir / "metrics.json")
        metrics_summary = {
            "target_space": metrics.get("target_space"),
            "overall_cosine": metrics.get("overall", {}).get("cosine"),
            "overall_mse": metrics.get("overall", {}).get("mse"),
            "pair_count": int(metrics.get("pair_metrics", {}).get("n_pairs", 0) or 0),
        }
    return {
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "transfer_dir": str(transfer_dir),
        "export_dir": str(export_dir),
        "missing_files": missing,
        "train_bundle_complete": not missing["train"],
        "eval_bundle_complete": not missing["eval"],
        "transfer_bundle_complete": not missing["transfer"],
        "export_bundle_complete": not missing["export"],
        "full_post_train_bundle_ready": all(not values for values in missing.values()),
        "metrics": metrics_summary,
    }


def _candidate_report(config, *, config_path: Path, current_support: dict[str, Any], current_mixed_index: str) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    support = _support_summary(config)
    artifacts = _bundle_status(config)
    experiment_name = str(config.get("experiment.name", ""))
    description = str(config.get("experiment.description", ""))
    non_smoke_experiment = "smoke" not in experiment_name.lower() and "smoke" not in description.lower()
    non_fixture_inputs = not (
        _looks_fixture_backed(config["dataset"].get("mixed_index"))
        or _looks_fixture_backed(config["targets"].get("cache_path"))
    )
    paired_conditions_present = set(config["dataset"].get("perception_conditions", [])) >= {"perception"} and set(
        config["dataset"].get("imagery_conditions", [])
    ) >= {"imagery"}
    stronger_total_pairs = support["dataset_pair_group_count"] > current_support["dataset_pair_group_count"]
    stronger_heldout_pairs = support["heldout_pair_group_count"] > _heldout_pair_count_from_current_support(current_support)
    stronger_paired_support = stronger_total_pairs or stronger_heldout_pairs
    eligible_as_stronger_real_candidate = bool(
        stronger_paired_support
        and artifacts["full_post_train_bundle_ready"]
        and non_smoke_experiment
        and non_fixture_inputs
        and paired_conditions_present
    )
    blocked_reasons: list[str] = []
    if not stronger_paired_support:
        blocked_reasons.append("does not improve paired support over the current full-overlap shared-only lane")
    if not artifacts["full_post_train_bundle_ready"]:
        blocked_reasons.append("does not currently expose a complete real train/eval/transfer/export bundle on the pod")
    if not non_smoke_experiment:
        blocked_reasons.append("is still smoke-scoped")
    if not non_fixture_inputs:
        blocked_reasons.append("still points at fixture-backed inputs")
    if not paired_conditions_present:
        blocked_reasons.append("does not expose both perception and imagery conditions")
    return {
        "experiment_name": experiment_name,
        "config": str(config_path.resolve()),
        "benchmark_role": config.get("experiment.benchmark_role"),
        "evidence_tier": config.get("experiment.evidence_tier"),
        "non_smoke_experiment": non_smoke_experiment,
        "non_fixture_inputs": non_fixture_inputs,
        "paired_conditions_present": paired_conditions_present,
        "same_prepared_dataset_as_current": support["mixed_index"] == current_mixed_index,
        "support": support,
        "artifacts": artifacts,
        "stronger_than_current_on_total_pairs": stronger_total_pairs,
        "stronger_than_current_on_heldout_pairs": stronger_heldout_pairs,
        "stronger_paired_support": stronger_paired_support,
        "eligible_as_stronger_real_candidate": eligible_as_stronger_real_candidate,
        "blocked_reasons": blocked_reasons,
    }


def build_full_imagery_overlap_promotion_path_audit(
    config,
    *,
    config_path: str | Path,
    candidate_config_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Full-overlap promotion-path audit requires configs/canonical/full_imagery_overlap_shared_only.yaml."
        )

    current_eval_dir = Path(config["evaluation"]["output_dir"]).resolve()
    readiness_path = current_eval_dir / "readiness_audit.json"
    if not readiness_path.exists():
        raise FileNotFoundError(
            f"Full-overlap promotion-path audit requires the current readiness artifact at {readiness_path}."
        )
    current_readiness = load_json(readiness_path)
    current_state = dict(current_readiness.get("state", {}))
    current_support = dict(current_readiness.get("heldout_support", {}))
    if not current_support:
        raise ValueError("Current full-overlap readiness artifact is missing the heldout_support section.")

    candidate_reports = []
    candidate_config_paths = candidate_config_paths or list(DEFAULT_CANDIDATE_CONFIGS)
    current_mixed_index = str(Path(config["dataset"]["mixed_index"]).resolve())
    for candidate_path in candidate_config_paths:
        resolved_candidate_path = Path(candidate_path)
        if not resolved_candidate_path.is_absolute():
            resolved_candidate_path = (_repo_root() / resolved_candidate_path).resolve()
        candidate_config = load_workflow_config(str(resolved_candidate_path))
        candidate_reports.append(
            _candidate_report(
                candidate_config,
                config_path=resolved_candidate_path,
                current_support=current_support,
                current_mixed_index=current_mixed_index,
            )
        )

    stronger_paired_support_available = any(item["stronger_paired_support"] for item in candidate_reports)
    stronger_real_candidate_available = any(item["eligible_as_stronger_real_candidate"] for item in candidate_reports)
    if stronger_real_candidate_available:
        best_candidate = max(
            [item for item in candidate_reports if item["eligible_as_stronger_real_candidate"]],
            key=lambda item: (
                item["support"]["dataset_pair_group_count"],
                item["support"]["heldout_pair_group_count"],
            ),
        )
        selected_main_lane = str(best_candidate["experiment_name"])
        selection_reason = (
            "a checked-in canonical lane with a real pod bundle improves paired support over the current "
            "full-overlap shared-only lane under the unchanged readiness philosophy"
        )
        next_move = "promote_selected_stronger_candidate"
    else:
        selected_main_lane = EXPECTED_EXPERIMENT_NAME
        selection_reason = (
            "no checked-in canonical lane in the current mounts improves paired support beyond the current "
            "full-overlap shared-only lane while also exposing a real complete post-train bundle"
        )
        next_move = "increase_paired_overlap_for_current_main_lane"

    blocked_reasons = []
    if bool(current_support.get("dataset_ceiling_blocks_training")):
        blocked_reasons.append(str(current_support.get("ceiling_blocked_reason")))
    if not stronger_paired_support_available:
        blocked_reasons.append("no checked-in canonical config currently exposes stronger paired support than 5 total / 1 held-out pair")
    if not stronger_real_candidate_available:
        blocked_reasons.append(
            "no stronger real mounted canonical promotion candidate is currently available; the next honest move is paired-data expansion"
        )

    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {
            "current_readiness_audit": str(readiness_path),
            "candidate_configs": [str((Path(path) if Path(path).is_absolute() else (_repo_root() / path)).resolve()) for path in candidate_config_paths],
        },
        "current_main_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "config": str(Path(config_path).resolve()),
            "benchmark_role": config.get("experiment.benchmark_role"),
            "evidence_tier": config.get("experiment.evidence_tier"),
            "readiness_state": current_state,
            "heldout_support": current_support,
        },
        "candidate_lanes": candidate_reports,
        "selection": {
            "selected_main_promotion_lane": selected_main_lane,
            "stronger_paired_support_available": stronger_paired_support_available,
            "stronger_real_candidate_available": stronger_real_candidate_available,
            "selection_reason": selection_reason,
            "next_honest_move": next_move,
        },
        "state": {
            "operational_ready": bool(current_state.get("operational_ready")),
            "downstream_contract_ready": bool(current_state.get("downstream_contract_ready")),
            "evidence_ready_candidate": bool(current_state.get("evidence_ready_candidate")),
            "training_ready": bool(current_state.get("training_ready")),
            "current_lane_ceiling_blocked": bool(current_support.get("dataset_ceiling_blocks_training")),
            "stronger_paired_support_available": stronger_paired_support_available,
            "stronger_real_candidate_available": stronger_real_candidate_available,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def _blocked_report(config_path: str | Path, message: str) -> dict[str, Any]:
    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {},
        "current_main_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
        },
        "candidate_lanes": [],
        "selection": {
            "selected_main_promotion_lane": EXPECTED_EXPERIMENT_NAME,
            "stronger_paired_support_available": False,
            "stronger_real_candidate_available": False,
            "selection_reason": "promotion-path audit is blocked",
            "next_honest_move": "resolve_blocked_audit",
        },
        "state": {
            "operational_ready": False,
            "downstream_contract_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "current_lane_ceiling_blocked": False,
            "stronger_paired_support_available": False,
            "stronger_real_candidate_available": False,
        },
        "blocked_reasons": [message],
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether any checked-in canonical lane honestly exceeds the current full-overlap shared-only promotion path."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--candidate-config", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_full_imagery_overlap_promotion_path_audit(
            config,
            config_path=args.config,
            candidate_config_paths=args.candidate_config or None,
        )
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/full_imagery_overlap_shared_only/promotion_path_audit.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["evaluation"]["output_dir"]).resolve() / "promotion_path_audit.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    if args.fail_on_blocked:
        return 0 if not report["state"]["stronger_real_candidate_available"] and report["state"]["operational_ready"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
