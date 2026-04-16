from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.compare_public_nod_paper2_robustness")

from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _rank_runs,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/robustness_comparison.json"
DEFAULT_TAIL_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/robustness_tail_comparison.json"
DEFAULT_RUNS = ("baseline", "early_visual_only", "metacognitive_only")


def _load_run_payload(run_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = run_root / "robustness_report.json"
    if not path.exists():
        return None, None
    return _load_optional_json(path), path


def _metric(report: dict[str, Any] | None, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("overall_metrics", {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dispersion(report: dict[str, Any] | None, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("dispersion_indicators", {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stability_ranking(run_rows: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    ranked = [
        {"run_id": row["run_id"], "experiment_name": row["experiment_name"], metric_key: row.get(metric_key)}
        for row in run_rows
        if row.get(metric_key) is not None
    ]
    return sorted(ranked, key=lambda item: float(item[metric_key]))


def _top_cluster(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    cluster = row.get(key)
    if not isinstance(cluster, dict):
        return None
    return {
        "group": cluster.get("group"),
        "low_trust_share_of_bucket": cluster.get("low_trust_share_of_bucket"),
        "concentration_flag": cluster.get("concentration_flag"),
    }


def _interpret(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_eval = _rank_runs(run_rows, "eval_cosine")
    best_transfer = _rank_runs(run_rows, "transfer_cosine")
    most_stable_subject = _stability_ranking(run_rows, "subject_mean_std")
    most_stable_session = _stability_ranking(run_rows, "session_mean_std")
    by_run = {row["run_id"]: row for row in run_rows}
    meta = by_run.get("metacognitive_only")

    notes: list[str] = []
    if best_eval:
        notes.append(f"best average eval run in the robustness pack is {best_eval[0]['run_id']}")
    if most_stable_subject:
        notes.append(f"most stable subject-level run by group-mean std is {most_stable_subject[0]['run_id']}")
    if most_stable_session:
        notes.append(f"most stable session-level run by group-mean std is {most_stable_session[0]['run_id']}")
    if best_eval and most_stable_subject:
        if best_eval[0]["run_id"] == most_stable_subject[0]["run_id"]:
            notes.append("the best average eval run is also the most stable by subject dispersion under the current heuristic")
        else:
            notes.append("the best average eval run is not the same run as the subject-dispersion winner")
    if meta is not None and meta.get("eval_cosine") is not None:
        baseline = by_run.get("baseline")
        if baseline is not None and baseline.get("eval_cosine") is not None:
            if float(meta["eval_cosine"]) >= float(baseline["eval_cosine"]):
                notes.append("metacognitive_only remains scientifically interesting because it matches or exceeds the baseline on average eval cosine in this fixed-slice breakdown")
            else:
                notes.append("metacognitive_only remains scientifically interesting because it preserves substantial signal even when isolated from the other ROI groups")
    notes.append("this robustness pack improves paper-2 stability support but still does not create publication-grade evidence")
    return {
        "best_eval_run": best_eval[0] if best_eval else None,
        "best_transfer_run": best_transfer[0] if best_transfer else None,
        "most_stable_subject_run": most_stable_subject[0] if most_stable_subject else None,
        "most_stable_session_run": most_stable_session[0] if most_stable_session else None,
        "notes": notes,
        "enough_for_publication_evidence": False,
    }


def build_public_nod_paper2_robustness_comparison(
    *,
    root_dir: str | Path,
    run_ids: list[str],
    comparison_output_path: str | Path,
    tail_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root_dir).resolve()
    run_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_root = root / run_id
        report, report_path = _load_run_payload(run_root)
        if report is None:
            run_rows.append(
                {
                    "run_id": run_id,
                    "run_root": str(run_root),
                    "report_path": None,
                    "robustness_report_ready": False,
                }
            )
            continue
        subject_dispersion = report.get("per_subject_cosine", {}).get("dispersion", {})
        session_dispersion = report.get("per_session_cosine", {}).get("dispersion", {})
        subject_cluster = report.get("low_trust_concentration", {}).get("top_subject")
        session_cluster = report.get("low_trust_concentration", {}).get("top_session")
        run_rows.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "report_path": str(report_path) if report_path is not None else None,
                "experiment_name": report.get("run_identity", {}).get("experiment_name"),
                "eval_cosine": _metric(report, "eval_cosine"),
                "transfer_cosine": _metric(report, "transfer_cosine"),
                "low_trust_threshold_cosine_lte": _metric(report, "low_trust_threshold_cosine_lte"),
                "subject_mean_std": _dispersion(report, "subject_mean_std"),
                "subject_mean_range": _dispersion(report, "subject_mean_range"),
                "session_mean_std": _dispersion(report, "session_mean_std"),
                "session_mean_range": _dispersion(report, "session_mean_range"),
                "worst_subject": report.get("per_subject_cosine", {}).get("worst_subject"),
                "worst_session": report.get("per_session_cosine", {}).get("worst_session"),
                "top_subject_low_trust_cluster": subject_cluster,
                "top_session_low_trust_cluster": session_cluster,
                "prepared_dataset_join_succeeded": report.get("prepared_dataset_join", {}).get("join_succeeded"),
                "robustness_report_ready": report.get("state", {}).get("robustness_report_ready"),
                "instability_flags": report.get("instability_flags", {}),
                "blocked_reasons": report.get("blocked_reasons", []),
            }
        )

    comparison = {
        "root_dir": str(root),
        "report_path": str(Path(comparison_output_path).resolve()),
        "runs": run_rows,
        "rankings": {
            "eval_cosine_desc": _rank_runs(run_rows, "eval_cosine"),
            "transfer_cosine_desc": _rank_runs(run_rows, "transfer_cosine"),
            "subject_dispersion_asc": _stability_ranking(run_rows, "subject_mean_std"),
            "session_dispersion_asc": _stability_ranking(run_rows, "session_mean_std"),
        },
        "interpretation": _interpret(run_rows),
        "state": {
            "run_count": len(run_rows),
            "all_reports_present": all(row.get("robustness_report_ready") for row in run_rows),
            "all_prepared_dataset_joins_succeeded": all(row.get("prepared_dataset_join_succeeded") for row in run_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this robustness comparison summarizes the paper-2 public-NOD stability pack only",
            "subject/session dispersion and tail clustering are descriptive support, not a publication gate",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }

    tail_rows = [
        {
            "run_id": row["run_id"],
            "experiment_name": row.get("experiment_name"),
            "low_trust_threshold_cosine_lte": row.get("low_trust_threshold_cosine_lte"),
            "top_subject_cluster": _top_cluster(row, "top_subject_low_trust_cluster"),
            "top_session_cluster": _top_cluster(row, "top_session_low_trust_cluster"),
            "worst_subject": row.get("worst_subject"),
            "worst_session": row.get("worst_session"),
        }
        for row in run_rows
    ]
    tail_comparison = {
        "root_dir": str(root),
        "report_path": str(Path(tail_output_path).resolve()),
        "runs": tail_rows,
        "tail_ranking_desc": sorted(
            [row for row in tail_rows if row.get("low_trust_threshold_cosine_lte") is not None],
            key=lambda row: float(row["low_trust_threshold_cosine_lte"]),
            reverse=True,
        ),
        "interpretation": {
            "notes": [
                "higher bottom-decile cosine threshold is treated here as a cleaner low-score tail",
                "subject/session clustering summaries are descriptive trust-drift indicators, not calibration",
                "this artifact does not imply evidence-grade robustness or publication readiness",
            ],
        },
        "state": {
            "all_reports_present": all(row.get("low_trust_threshold_cosine_lte") is not None for row in tail_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this robustness-tail comparison belongs only to the paper-2 public-NOD lane",
            "tail clustering is a heuristic trust-drift summary, not calibrated uncertainty evidence",
            "training_ready remains false",
        ],
    }
    return json_safe(comparison), json_safe(tail_comparison)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare subject-level and session-level robustness across existing paper-2 public-NOD runs."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    parser.add_argument("--tail-output", default=str(_default_path(DEFAULT_TAIL_OUTPUT)))
    args = parser.parse_args(argv)

    run_ids = args.run_ids or list(DEFAULT_RUNS)
    comparison, tail_comparison = build_public_nod_paper2_robustness_comparison(
        root_dir=args.root_dir,
        run_ids=run_ids,
        comparison_output_path=args.output,
        tail_output_path=args.tail_output,
    )
    write_report(args.output, comparison)
    write_report(args.tail_output, tail_comparison)
    print(json.dumps(json_safe(comparison), indent=2))
    print(json.dumps(json_safe(tail_comparison), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
