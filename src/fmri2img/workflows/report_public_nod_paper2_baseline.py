from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.report_public_nod_paper2_baseline")

from fmri2img.workflows._downstream_contract_audit import (  # noqa: E402
    load_json,
    normalize_decoder_card_target,
    normalize_manifest_target,
    surface_condition_semantics,
)
from fmri2img.workflows.common import load_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only_paper2_baseline.yaml"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/paper2_baseline_report.json"
DEFAULT_RELIABILITY_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/reliability_seed_report.json"
)
EXPECTED_EXPERIMENT_NAME = "public_nod_imagenet_run10_shared_only_paper2_baseline"
EXPECTED_EXPERIMENT_PREFIX = "public_nod_imagenet_run10_shared_only_paper2_"
TRAIN_FILES = ("best_decoder.pt", "config_snapshot.json", "roi_summary.json", "target_summary.json", "train_history.json")
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json")
TRANSFER_FILES = ("transfer_metrics.json", "per_trial_pairs.csv")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")
OPERATIONAL_BOUNDARY = [
    "this report summarizes the first dedicated paper-2 public-NOD baseline bundle and keeps it separate from the full_imagery_overlap_shared_only lane",
    "operational success here does not imply evidence-grade calibration, benchmark superiority, or production readiness",
    "training_ready remains false for this report because this first paper-2 baseline is still a non-smoke operational research artifact, not a promoted evidence gate",
]
RELIABILITY_BOUNDARY = [
    "this reliability seed report is exploratory paper-support only",
    "the low-trust heuristic is a simple score-ranking aid, not a calibrated uncertainty claim",
    "missing subject or session metadata is left unresolved rather than invented",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return (_repo_root() / relative).resolve()


def _missing_files(root: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_summary(payload: dict[str, Any], *, expected_target_space: str) -> dict[str, Any]:
    overall = payload.get("overall", {})
    condition_semantics = surface_condition_semantics(payload)
    pair_metrics = payload.get("pair_metrics", {})
    try:
        pair_count = int(pair_metrics.get("n_pairs", 0) or 0)
    except (TypeError, ValueError):
        pair_count = 0
    return {
        "target_space": payload.get("target_space"),
        "target_space_matches_config": payload.get("target_space") == expected_target_space,
        "overall_cosine": overall.get("cosine"),
        "overall_mse": overall.get("mse"),
        "overall_cosine_std": overall.get("cosine_std"),
        "metrics_finite": _is_finite_number(overall.get("cosine")) and _is_finite_number(overall.get("mse")),
        "by_condition": payload.get("by_condition", []),
        "condition_semantics": condition_semantics,
        "pair_count": pair_count,
    }


def _load_prepared_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _experiment_allowed(name: str) -> bool:
    return name == EXPECTED_EXPERIMENT_NAME or name.startswith(EXPECTED_EXPERIMENT_PREFIX)


def _classify_run(config) -> str:
    benchmark_role = str(config.get("experiment.benchmark_role", ""))
    if "ablation" in benchmark_role:
        return "paper2_operational_ablation"
    return "first_non_smoke_operational_baseline"


def _build_blocked_report(config, *, config_path: str | Path, output_path: str | Path, blocked_reasons: list[str]) -> dict[str, Any]:
    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_output_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()
    prepared_dataset_path = Path(config["dataset"]["mixed_index"]).resolve()
    return {
        "config": str(Path(config_path).resolve()),
        "report_path": str(Path(output_path).resolve()),
        "experiment": {
            "name": str(config.get("experiment.name", "")),
            "benchmark_role": config.get("experiment.benchmark_role"),
            "evidence_tier": config.get("experiment.evidence_tier"),
        },
        "artifact_paths": {
            "prepared_dataset": str(prepared_dataset_path),
            "train_dir": str(train_output_dir),
            "eval_dir": str(eval_output_dir),
            "transfer_dir": str(transfer_output_dir),
            "export_dir": str(export_output_dir),
        },
        "state": {
            "baseline_bundle_exists": False,
            "train_bundle_complete": False,
            "eval_bundle_complete": False,
            "transfer_bundle_complete": False,
            "export_bundle_complete": False,
            "operational_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "reliability_seed_written": False,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def _merge_subject_session_counts(
    pairs_df: pd.DataFrame,
    prepared_df: pd.DataFrame | None,
) -> dict[str, Any]:
    if prepared_df is None or "pair_id" not in prepared_df.columns:
        return {
            "prepared_dataset_present": False,
            "per_subject_counts_recoverable": False,
            "per_session_counts_recoverable": False,
        }
    index_cols = [col for col in ("pair_id", "subject", "session", "split") if col in prepared_df.columns]
    merged = pairs_df.merge(prepared_df[index_cols].drop_duplicates(), on="pair_id", how="left")
    result: dict[str, Any] = {
        "prepared_dataset_present": True,
        "prepared_dataset_rows": int(len(prepared_df)),
        "merged_rows": int(len(merged)),
    }
    if "subject" in merged.columns:
        result["per_subject_counts_recoverable"] = True
        result["per_subject_counts"] = {
            str(key): int(value) for key, value in merged["subject"].fillna("unknown").value_counts().to_dict().items()
        }
    else:
        result["per_subject_counts_recoverable"] = False
    if "session" in merged.columns:
        result["per_session_counts_recoverable"] = True
        result["per_session_counts"] = {
            str(key): int(value) for key, value in merged["session"].fillna("unknown").value_counts().to_dict().items()
        }
    else:
        result["per_session_counts_recoverable"] = False
    return result


def _build_low_trust_summary(pairs_df: pd.DataFrame) -> dict[str, Any]:
    if "cosine" not in pairs_df.columns or pairs_df.empty:
        return {
            "available": False,
            "reason": "per_trial_pairs.csv does not expose cosine scores",
        }
    ranked = pairs_df.sort_values("cosine", ascending=True).reset_index(drop=True)
    bucket_size = max(1, int(math.ceil(len(ranked) * 0.1)))
    threshold = float(ranked.iloc[bucket_size - 1]["cosine"])
    columns = [col for col in ("pair_id", "nsd_id", "condition", "subject", "session", "split", "cosine") if col in ranked.columns]
    return {
        "available": True,
        "heuristic": "bottom_decile_by_cosine",
        "bucket_size": bucket_size,
        "low_trust_threshold_cosine_lte": threshold,
        "lowest_examples": json_safe(ranked.loc[: min(9, len(ranked) - 1), columns].to_dict(orient="records")),
    }


def _build_reliability_seed_report(
    *,
    config,
    config_path: str | Path,
    output_path: str | Path,
    artifact_paths: dict[str, str],
    eval_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any],
    prepared_df: pd.DataFrame | None,
) -> dict[str, Any]:
    per_trial_pairs_path = Path(artifact_paths["per_trial_pairs"])
    pairs_df = pd.read_csv(per_trial_pairs_path)
    if "pair_id" in pairs_df.columns:
        pairs_df["pair_id"] = pairs_df["pair_id"].astype(int)
    metadata_counts = _merge_subject_session_counts(pairs_df, prepared_df)
    if prepared_df is not None and "pair_id" in prepared_df.columns and "pair_id" in pairs_df.columns:
        merged_cols = [col for col in ("pair_id", "subject", "session", "split") if col in prepared_df.columns]
        pairs_df = pairs_df.merge(prepared_df[merged_cols].drop_duplicates(), on="pair_id", how="left")

    by_condition_eval = {
        str(item.get("condition")): int(item.get("count", 0))
        for item in eval_metrics.get("by_condition", [])
        if isinstance(item, dict) and item.get("condition") is not None
    }
    by_condition_transfer = {
        str(item.get("condition")): int(item.get("count", 0))
        for item in transfer_metrics.get("by_condition", [])
        if isinstance(item, dict) and item.get("condition") is not None
    }
    return {
        "config": str(Path(config_path).resolve()),
        "report_path": str(Path(output_path).resolve()),
        "experiment": {
            "name": str(config.get("experiment.name", "")),
            "benchmark_role": config.get("experiment.benchmark_role"),
            "evidence_tier": config.get("experiment.evidence_tier"),
        },
        "artifact_paths": artifact_paths,
        "condition_semantics": {
            "eval": surface_condition_semantics(eval_metrics),
            "transfer": surface_condition_semantics(transfer_metrics),
        },
        "overall_metrics": {
            "eval": eval_metrics.get("overall", {}),
            "transfer": transfer_metrics.get("overall", {}),
        },
        "sample_counts": {
            "eval_by_condition": by_condition_eval,
            "transfer_by_condition": by_condition_transfer,
            "per_trial_pairs_rows": int(len(pairs_df)),
            **metadata_counts,
        },
        "low_trust_candidates": _build_low_trust_summary(pairs_df),
        "state": {
            "reliability_seed_ready": True,
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": RELIABILITY_BOUNDARY,
    }


def build_public_nod_paper2_baseline_report(
    config,
    *,
    config_path: str | Path,
    output_path: str | Path,
    reliability_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    experiment_name = str(config.get("experiment.name", ""))
    if not _experiment_allowed(experiment_name):
        raise ValueError(
            "Public-NOD paper-2 run report requires a fixed-slice paper-2 experiment name "
            "starting with public_nod_imagenet_run10_shared_only_paper2_."
        )

    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    transfer_output_dir = Path(config["evaluation"]["transfer_output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()
    prepared_dataset_path = Path(config["dataset"]["mixed_index"]).resolve()

    missing = {
        "train": _missing_files(train_output_dir, TRAIN_FILES),
        "eval": _missing_files(eval_output_dir, EVAL_FILES),
        "transfer": _missing_files(transfer_output_dir, TRANSFER_FILES),
        "export": _missing_files(export_output_dir, EXPORT_FILES),
    }
    missing_artifacts = {section: values for section, values in missing.items() if values}
    baseline_bundle_exists = any(root.exists() and any(root.iterdir()) for root in (train_output_dir, eval_output_dir, transfer_output_dir, export_output_dir))
    if missing_artifacts:
        blocked_reasons = [
            "paper-2 baseline bundle is incomplete: "
            + "; ".join(f"{section}: {', '.join(values)}" for section, values in missing_artifacts.items())
        ]
        report = _build_blocked_report(config, config_path=config_path, output_path=output_path, blocked_reasons=blocked_reasons)
        report["state"]["baseline_bundle_exists"] = baseline_bundle_exists
        return report, None

    config_snapshot = load_json(train_output_dir / "config_snapshot.json")
    train_history = load_json(train_output_dir / "train_history.json")
    eval_metrics = load_json(eval_output_dir / "metrics.json")
    transfer_metrics = load_json(transfer_output_dir / "transfer_metrics.json")
    manifest = load_json(export_output_dir / "manifest.json")
    decoder_card = load_json(export_output_dir / "decoder_card.json")
    prepared_df = _load_prepared_dataset(prepared_dataset_path)

    manifest_target = normalize_manifest_target(
        manifest.get("metadata", {}).get("target_spec_normalized"),
        fallback_target_spec=manifest.get("target_spec"),
    )
    decoder_target = normalize_decoder_card_target(decoder_card.get("target"))
    target_metadata_consistent = manifest_target == decoder_target if decoder_target else False

    manifest_condition = surface_condition_semantics(manifest.get("metadata", {}).get("condition_semantics"))
    eval_condition = surface_condition_semantics(eval_metrics)
    transfer_condition = surface_condition_semantics(transfer_metrics)
    condition_semantics_consistent = manifest_condition == eval_condition == transfer_condition

    eval_summary = _metric_summary(eval_metrics, expected_target_space=str(config["targets"].get("name", "")))
    transfer_summary = _metric_summary(transfer_metrics, expected_target_space=str(config["targets"].get("name", "")))

    train_history_nonempty = isinstance(train_history, list) and len(train_history) > 0
    latest_epoch = train_history[-1] if train_history_nonempty else {}
    config_snapshot_name = str(config_snapshot.get("experiment", {}).get("name", ""))
    manifest_name = str(manifest.get("metadata", {}).get("experiment", {}).get("name", ""))
    decoder_card_name = str(decoder_card.get("experiment", {}).get("name", ""))
    benchmark_roles = {
        "config": config.get("experiment.benchmark_role"),
        "config_snapshot": config_snapshot.get("experiment", {}).get("benchmark_role"),
        "manifest": manifest.get("metadata", {}).get("experiment", {}).get("benchmark_role"),
        "decoder_card": decoder_card.get("experiment", {}).get("benchmark_role"),
    }
    benchmark_role_consistent = len({str(value) for value in benchmark_roles.values()}) == 1
    experiment_identity_consistent = {config_snapshot_name, manifest_name, decoder_card_name, experiment_name} == {
        experiment_name
    }

    operational_ready = bool(
        train_history_nonempty
        and eval_summary["metrics_finite"]
        and transfer_summary["metrics_finite"]
        and eval_summary["target_space_matches_config"]
        and transfer_summary["target_space_matches_config"]
        and target_metadata_consistent
        and condition_semantics_consistent
        and experiment_identity_consistent
        and benchmark_role_consistent
    )

    artifact_paths = {
        "prepared_dataset": str(prepared_dataset_path),
        "train_dir": str(train_output_dir),
        "train_history": str((train_output_dir / "train_history.json").resolve()),
        "eval_dir": str(eval_output_dir),
        "eval_metrics": str((eval_output_dir / "metrics.json").resolve()),
        "transfer_dir": str(transfer_output_dir),
        "transfer_metrics": str((transfer_output_dir / "transfer_metrics.json").resolve()),
        "per_trial_pairs": str((transfer_output_dir / "per_trial_pairs.csv").resolve()),
        "export_dir": str(export_output_dir),
        "export_manifest": str((export_output_dir / "manifest.json").resolve()),
        "export_decoder_card": str((export_output_dir / "decoder_card.json").resolve()),
    }

    reliability_seed = _build_reliability_seed_report(
        config=config,
        config_path=config_path,
        output_path=reliability_output_path,
        artifact_paths=artifact_paths,
        eval_metrics=eval_metrics,
        transfer_metrics=transfer_metrics,
        prepared_df=prepared_df,
    )

    report = {
        "config": str(Path(config_path).resolve()),
        "report_path": str(Path(output_path).resolve()),
        "experiment": {
            "name": experiment_name,
            "benchmark_role": config.get("experiment.benchmark_role"),
            "evidence_tier": config.get("experiment.evidence_tier"),
            "description": config.get("experiment.description"),
        },
        "fixed_contract": {
            "dataset_id": config.get("public_nod.dataset_id"),
            "task": config.get("public_nod.task"),
            "run": int(config.get("public_nod.run", 10)),
            "subjects": list(config.get("public_nod.subjects", [])),
            "sessions": list(config.get("public_nod.sessions", [])),
            "pair_rows": int(config.get("public_nod.pair_rows", 3600)),
            "target_space": config["targets"].get("name"),
        },
        "artifact_paths": artifact_paths,
        "train_summary": {
            "train_history_nonempty": train_history_nonempty,
            "epoch_count": len(train_history) if train_history_nonempty else 0,
            "latest_epoch": latest_epoch,
        },
        "eval_summary": eval_summary,
        "transfer_summary": transfer_summary,
        "export_summary": {
            "target_metadata_consistent": target_metadata_consistent,
            "condition_semantics_consistent": condition_semantics_consistent,
            "normalized_target_spec": manifest_target,
            "condition_semantics": manifest_condition,
            "experiment_identity": {
                "config_snapshot": config_snapshot_name,
                "manifest": manifest_name,
                "decoder_card": decoder_card_name,
            },
            "benchmark_roles": benchmark_roles,
        },
        "publication_assessment": {
            "classification": _classify_run(config),
            "already_strong": [
                "the separate paper-2 lane now has a dedicated non-smoke paper-2 run bundle under its own output root",
                "the run bundle can be summarized without borrowing claims from the full-overlap lane",
                "the reliability seed report can rank low-trust candidates from real eval outputs without pretending calibration",
            ],
            "still_missing_before_publishable": [
                "replicated baseline runs or controlled ablations",
                "calibration beyond score ranking",
                "ROI and session-or-subject robustness analyses",
                "evidence-grade interpretation criteria for this separate paper-2 lane",
            ],
        },
        "state": {
            "baseline_bundle_exists": baseline_bundle_exists,
            "train_bundle_complete": True,
            "eval_bundle_complete": True,
            "transfer_bundle_complete": True,
            "export_bundle_complete": True,
            "operational_ready": operational_ready,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "reliability_seed_written": True,
        },
        "blocked_reasons": [],
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }
    return report, reliability_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the first dedicated paper-2 public-NOD baseline bundle and write a small reliability seed report when eval outputs exist."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--reliability-output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    config = load_workflow_config(args.config, args.override)
    output_path = args.output or DEFAULT_OUTPUT
    reliability_output_path = args.reliability_output or DEFAULT_RELIABILITY_OUTPUT

    try:
        report, reliability_seed = build_public_nod_paper2_baseline_report(
            config,
            config_path=args.config,
            output_path=output_path,
            reliability_output_path=reliability_output_path,
        )
    except Exception as exc:
        report = _build_blocked_report(
            config,
            config_path=args.config,
            output_path=output_path,
            blocked_reasons=[str(exc)],
        )
        reliability_seed = None

    write_report(output_path, report)
    if reliability_seed is not None:
        write_report(reliability_output_path, reliability_seed)

    print(json.dumps(json_safe(report), indent=2))
    if reliability_seed is not None:
        print(json.dumps(json_safe(reliability_seed), indent=2))
    if args.fail_on_blocked and report["blocked_reasons"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
