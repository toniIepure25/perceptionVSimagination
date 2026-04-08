from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.audit_full_imagery_overlap_data_expansion")

from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/full_imagery_overlap_shared_only.yaml"
EXPECTED_EXPERIMENT_NAME = "full_imagery_overlap_shared_only"
DEFAULT_PERCEPTION_FALLBACK_TEMPLATES = (
    "/home/jovyan/work/FMRI2images/data/indices/nsd_index/subject={subject}/index.parquet",
)
DEFAULT_LEGACY_IMAGERY_TEMPLATE = "cache/indices/imagery/{subject}.parquet"
OPERATIONAL_BOUNDARY = [
    "this audit only answers whether the current environment can produce more honest paired overlap for the existing full-overlap shared-only lane",
    "it does not weaken the 32-pair readiness gate or reinterpret the current five-pair benchmark ceiling as success",
    "if no stronger mounted source exists, the next honest move is external paired-data expansion rather than lane switching or report inflation",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _resolve_template(template: str, *, subject: str) -> Path:
    candidate = Path(template.format(subject=subject))
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _count_paired_groups(df: pd.DataFrame) -> int:
    if df.empty or "pair_id" not in df.columns or "condition" not in df.columns:
        return 0
    grouped = df.groupby("pair_id")["condition"].agg(lambda values: {str(value) for value in values})
    return int(sum({"perception", "imagery"}.issubset(values) for values in grouped))


def _nsd_id_set(df: pd.DataFrame) -> set[int]:
    for column in ("nsdId", "nsd_id"):
        if column in df.columns:
            return {int(value) for value in df[column].dropna().astype(int).unique()}
    return set()


def _resolve_perception_source(subject: str, primary_template: str, fallback_templates: tuple[str, ...]) -> dict[str, Any]:
    candidates = [primary_template, *fallback_templates]
    searched = []
    for index, template in enumerate(candidates):
        path = _resolve_template(template, subject=subject)
        searched.append(str(path))
        if not path.exists():
            continue
        source_kind = "config_template" if index == 0 else f"fallback_{index}"
        df = pd.read_parquet(path)
        ids = _nsd_id_set(df)
        return {
            "path": str(path),
            "exists": True,
            "source_kind": source_kind,
            "rows": int(len(df)),
            "unique_nsd_ids": int(len(ids)),
            "nsd_ids": ids,
            "searched_paths": searched,
        }
    return {
        "path": None,
        "exists": False,
        "source_kind": None,
        "rows": 0,
        "unique_nsd_ids": 0,
        "nsd_ids": set(),
        "searched_paths": searched,
    }


def _resolve_optional_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": 0,
            "unique_nsd_ids": 0,
            "nsd_ids": set(),
        }
    df = pd.read_parquet(path)
    ids = _nsd_id_set(df)
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "unique_nsd_ids": int(len(ids)),
        "nsd_ids": ids,
    }


def _current_lane_summary(config, readiness: dict[str, Any]) -> dict[str, Any]:
    mixed_index_path = Path(config["dataset"]["mixed_index"]).resolve()
    mixed_df = pd.read_parquet(mixed_index_path)
    subject_pair_counts = {
        str(subject): _count_paired_groups(subject_df.reset_index(drop=True))
        for subject, subject_df in mixed_df.groupby(mixed_df["subject"].astype(str))
    }
    subject_nsd_ids = {
        str(subject): sorted(_nsd_id_set(subject_df))
        for subject, subject_df in mixed_df.groupby(mixed_df["subject"].astype(str))
    }
    return {
        "mixed_index": str(mixed_index_path),
        "dataset_rows": int(len(mixed_df)),
        "dataset_pair_group_count": int(readiness.get("heldout_support", {}).get("dataset_pair_group_count", _count_paired_groups(mixed_df))),
        "subjects": sorted(subject_pair_counts),
        "subject_pair_group_counts": subject_pair_counts,
        "subject_nsd_ids": subject_nsd_ids,
    }


def _subject_source_report(
    *,
    subject: str,
    primary_perception_template: str,
    fallback_perception_templates: tuple[str, ...],
    imagery_template: str,
    legacy_imagery_template: str | None,
    current_lane_subject_pair_counts: dict[str, int],
) -> dict[str, Any]:
    perception = _resolve_perception_source(subject, primary_perception_template, fallback_perception_templates)
    imagery = _resolve_optional_source(_resolve_template(imagery_template, subject=subject))
    legacy_imagery = None
    if legacy_imagery_template:
        legacy_imagery = _resolve_optional_source(_resolve_template(legacy_imagery_template, subject=subject))

    mounted_overlap_ids = sorted(perception["nsd_ids"] & imagery["nsd_ids"])
    mounted_pair_group_count = int(len(mounted_overlap_ids))
    current_pair_groups = int(current_lane_subject_pair_counts.get(subject, 0))
    return {
        "subject": subject,
        "current_lane_included": current_pair_groups > 0,
        "current_lane_pair_group_count": current_pair_groups,
        "perception_source": {
            key: value
            for key, value in perception.items()
            if key != "nsd_ids"
        },
        "imagery_full_all_source": {
            key: value
            for key, value in imagery.items()
            if key != "nsd_ids"
        },
        "legacy_imagery_source": None
        if legacy_imagery is None
        else {key: value for key, value in legacy_imagery.items() if key != "nsd_ids"},
        "mounted_overlap_ids": mounted_overlap_ids,
        "mounted_pair_group_count": mounted_pair_group_count,
        "unused_pair_groups_vs_current_lane": max(0, mounted_pair_group_count - current_pair_groups),
    }


def _scan_prepared_mixed_indices(prepared_root: Path) -> dict[str, Any]:
    candidates = []
    for path in prepared_root.glob("**/*.parquet"):
        if "mixed" not in path.name:
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if not {"pair_id", "condition"}.issubset(df.columns):
            continue
        pair_groups = _count_paired_groups(df)
        if pair_groups <= 0:
            continue
        split_pair_groups = {}
        if "split" in df.columns:
            for split, split_df in df.groupby("split"):
                split_pair_groups[str(split)] = _count_paired_groups(split_df.reset_index(drop=True))
        candidates.append(
            {
                "path": str(path.resolve()),
                "rows": int(len(df)),
                "pair_groups": int(pair_groups),
                "split_pair_groups": split_pair_groups,
                "subjects": sorted(df["subject"].astype(str).unique().tolist()) if "subject" in df.columns else [],
            }
        )
    candidates.sort(key=lambda item: (item["pair_groups"], item["rows"], item["path"]), reverse=True)
    return {
        "prepared_root": str(prepared_root.resolve()),
        "candidate_count": len(candidates),
        "largest_pair_groups_found": int(candidates[0]["pair_groups"]) if candidates else 0,
        "top_candidates": candidates[:5],
    }


def build_full_imagery_overlap_data_expansion_audit(
    config,
    *,
    config_path: str | Path,
    fallback_perception_templates: tuple[str, ...] = DEFAULT_PERCEPTION_FALLBACK_TEMPLATES,
    legacy_imagery_template: str | None = DEFAULT_LEGACY_IMAGERY_TEMPLATE,
    prepared_root: str | Path | None = None,
) -> dict[str, Any]:
    validate_canonical_workflow_config(config)
    if str(config.get("experiment.name", "")) != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            "Full-overlap data-expansion audit requires configs/canonical/full_imagery_overlap_shared_only.yaml."
        )

    eval_dir = Path(config["evaluation"]["output_dir"]).resolve()
    readiness_path = eval_dir / "readiness_audit.json"
    promotion_path = eval_dir / "promotion_path_audit.json"
    if not readiness_path.exists():
        raise FileNotFoundError(
            f"Full-overlap data-expansion audit requires the current readiness artifact at {readiness_path}."
        )
    readiness = load_json(readiness_path)
    current_lane = _current_lane_summary(config, readiness)
    current_support = dict(readiness.get("heldout_support", {}))
    if not current_support:
        raise ValueError("Current full-overlap readiness artifact is missing the heldout_support section.")

    primary_perception_template = config.get("preparation.overlap.perception_index_template")
    imagery_template = config.get("preparation.overlap.imagery_index_template")
    subjects = [str(subject) for subject in config.get("preparation.overlap.subjects", [])]
    if not primary_perception_template or not imagery_template or not subjects:
        raise ValueError(
            "Full-overlap data-expansion audit requires preparation.overlap.subjects, perception_index_template, and imagery_index_template."
        )

    subject_reports = [
        _subject_source_report(
            subject=subject,
            primary_perception_template=primary_perception_template,
            fallback_perception_templates=fallback_perception_templates,
            imagery_template=imagery_template,
            legacy_imagery_template=legacy_imagery_template,
            current_lane_subject_pair_counts=current_lane["subject_pair_group_counts"],
        )
        for subject in subjects
    ]

    max_pair_groups_from_mounted_sources = int(sum(item["mounted_pair_group_count"] for item in subject_reports))
    unused_subjects_with_overlap = [
        item["subject"]
        for item in subject_reports
        if item["unused_pair_groups_vs_current_lane"] > 0 or (
            item["mounted_pair_group_count"] > 0 and not item["current_lane_included"]
        )
    ]
    additional_pair_groups_available = int(
        max(0, max_pair_groups_from_mounted_sources - current_lane["dataset_pair_group_count"])
    )
    mounted_source_can_increase_support = additional_pair_groups_available > 0

    prepared_root_path = Path(prepared_root).resolve() if prepared_root is not None else (_repo_root() / "outputs/canonical/prepared").resolve()
    prepared_scan = _scan_prepared_mixed_indices(prepared_root_path)
    larger_prepared_mixed_index_available = prepared_scan["largest_pair_groups_found"] > current_lane["dataset_pair_group_count"]

    promotion_summary = None
    if promotion_path.exists():
        promotion_payload = load_json(promotion_path)
        promotion_summary = {
            "path": str(promotion_path),
            "selected_main_promotion_lane": promotion_payload.get("selection", {}).get("selected_main_promotion_lane"),
            "stronger_real_candidate_available": bool(
                promotion_payload.get("selection", {}).get("stronger_real_candidate_available")
            ),
            "stronger_paired_support_available": bool(
                promotion_payload.get("selection", {}).get("stronger_paired_support_available")
            ),
        }

    current_lane_uses_max_mounted_support = (
        max_pair_groups_from_mounted_sources == current_lane["dataset_pair_group_count"]
        and not larger_prepared_mixed_index_available
    )
    data_ceiling_confirmed = not mounted_source_can_increase_support and not larger_prepared_mixed_index_available

    blocked_reasons = []
    if current_support.get("dataset_ceiling_blocks_training"):
        blocked_reasons.append(str(current_support.get("ceiling_blocked_reason")))
    if not mounted_source_can_increase_support:
        blocked_reasons.append(
            "mounted subject-level perception and imagery indices do not expose any additional paired groups beyond the current full-overlap lane"
        )
    if not larger_prepared_mixed_index_available:
        blocked_reasons.append(
            "no prepared mixed index under outputs/canonical/prepared exceeds the current 5-pair full-overlap dataset"
        )
    if promotion_summary and not promotion_summary["stronger_real_candidate_available"]:
        blocked_reasons.append(
            "no stronger mounted canonical lane exists either; the blocker is paired-data availability, not lane choice"
        )

    conclusion = {
        "can_materially_increase_paired_support_for_current_lane": bool(
            mounted_source_can_increase_support or larger_prepared_mixed_index_available
        ),
        "selected_main_promotion_lane": EXPECTED_EXPERIMENT_NAME,
        "current_lane_uses_max_mounted_pair_support": current_lane_uses_max_mounted_support,
        "next_honest_move": (
            "rebuild_full_overlap_with_stronger_mounted_sources"
            if mounted_source_can_increase_support or larger_prepared_mixed_index_available
            else "acquire_or_mount_larger_paired_source"
        ),
        "selection_reason": (
            "the current lane can be expanded with already mounted sources"
            if mounted_source_can_increase_support or larger_prepared_mixed_index_available
            else "the current lane already exhausts the paired overlap available from mounted canonical-compatible sources"
        ),
    }

    return {
        "config": str(Path(config_path).resolve()),
        "artifact_paths": {
            "current_readiness_audit": str(readiness_path),
            "current_promotion_path_audit": None if promotion_summary is None else promotion_summary["path"],
            "current_mixed_index": current_lane["mixed_index"],
            "prepared_root": prepared_scan["prepared_root"],
        },
        "current_main_lane": {
            "experiment_name": EXPECTED_EXPERIMENT_NAME,
            "config": str(Path(config_path).resolve()),
            "readiness_state": dict(readiness.get("state", {})),
            "heldout_support": current_support,
            "current_mixed_support": current_lane,
        },
        "mounted_source_inventory": {
            "subjects_considered": subjects,
            "perception_primary_template": primary_perception_template,
            "perception_fallback_templates": list(fallback_perception_templates),
            "imagery_full_all_template": imagery_template,
            "legacy_imagery_template": legacy_imagery_template,
            "subjects": subject_reports,
        },
        "prepared_inventory": prepared_scan,
        "promotion_path_summary": promotion_summary,
        "conclusion": conclusion,
        "state": {
            "operational_ready": bool(readiness.get("state", {}).get("operational_ready")),
            "downstream_contract_ready": bool(readiness.get("state", {}).get("downstream_contract_ready")),
            "evidence_ready_candidate": bool(readiness.get("state", {}).get("evidence_ready_candidate")),
            "training_ready": bool(readiness.get("state", {}).get("training_ready")),
            "mounted_source_can_increase_support": mounted_source_can_increase_support,
            "larger_prepared_mixed_index_available": larger_prepared_mixed_index_available,
            "current_lane_uses_max_mounted_pair_support": current_lane_uses_max_mounted_support,
            "data_ceiling_confirmed": data_ceiling_confirmed,
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
        "mounted_source_inventory": {
            "subjects_considered": [],
            "subjects": [],
        },
        "prepared_inventory": {
            "candidate_count": 0,
            "largest_pair_groups_found": 0,
            "top_candidates": [],
        },
        "promotion_path_summary": None,
        "conclusion": {
            "can_materially_increase_paired_support_for_current_lane": False,
            "selected_main_promotion_lane": EXPECTED_EXPERIMENT_NAME,
            "current_lane_uses_max_mounted_pair_support": False,
            "next_honest_move": "resolve_blocked_audit",
            "selection_reason": "data-expansion audit is blocked",
        },
        "state": {
            "operational_ready": False,
            "downstream_contract_ready": False,
            "evidence_ready_candidate": False,
            "training_ready": False,
            "mounted_source_can_increase_support": False,
            "larger_prepared_mixed_index_available": False,
            "current_lane_uses_max_mounted_pair_support": False,
            "data_ceiling_confirmed": False,
        },
        "blocked_reasons": [message],
        "operational_boundary": OPERATIONAL_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether the current environment can honestly expand paired overlap for the full-overlap shared-only lane."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_full_imagery_overlap_data_expansion_audit(config, config_path=args.config)
    except Exception as exc:
        report = _blocked_report(args.config, str(exc))
        output_path = args.output or "outputs/canonical/eval/full_imagery_overlap_shared_only/data_expansion_audit.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["evaluation"]["output_dir"]).resolve() / "data_expansion_audit.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    if args.fail_on_blocked:
        return 0 if report["state"]["data_ceiling_confirmed"] else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
