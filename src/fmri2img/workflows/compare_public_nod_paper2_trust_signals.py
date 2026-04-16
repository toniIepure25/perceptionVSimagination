from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.compare_public_nod_paper2_trust_signals")

from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _rank_runs,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/trust_signal_comparison.json"
DEFAULT_INSTABILITY_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/trust_instability_comparison.json"
)
DEFAULT_RUNS = ("baseline", "early_visual_only", "metacognitive_only")
PRIMARY_LABEL = "bottom_10_pct"


def _load_run_payload(run_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = run_root / "trust_signal_report.json"
    if not path.exists():
        return None, None
    return _load_optional_json(path), path


def _stability_ranking(run_rows: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    ranked = [
        {"run_id": row["run_id"], "experiment_name": row["experiment_name"], metric_key: row.get(metric_key)}
        for row in run_rows
        if row.get(metric_key) is not None
    ]
    return sorted(ranked, key=lambda item: float(item[metric_key]))


def _threshold_metric(report: dict[str, Any] | None, label: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("threshold_analyses", {}).get(label, {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trust_score(report: dict[str, Any] | None, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("descriptive_trust_scores", {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare_rows_desc(run_rows: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    ranked = [
        {"run_id": row["run_id"], "experiment_name": row["experiment_name"], metric_key: row.get(metric_key)}
        for row in run_rows
        if row.get(metric_key) is not None
    ]
    return sorted(ranked, key=lambda item: float(item[metric_key]), reverse=True)


def _interpret(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_eval = _rank_runs(run_rows, "eval_cosine")
    best_transfer = _rank_runs(run_rows, "transfer_cosine")
    subject_stability = _stability_ranking(run_rows, "subject_mean_std")
    session_stability = _stability_ranking(run_rows, "session_mean_std")
    cleanest_tail = _compare_rows_desc(run_rows, "bottom_10_threshold_cosine_lte")
    strongest_alignment = _compare_rows_desc(run_rows, "bottom_10_tail_instability_enrichment")
    by_run = {row["run_id"]: row for row in run_rows}
    meta = by_run.get("metacognitive_only")

    heuristic_support = any(
        row.get("bottom_10_tail_instability_enrichment") is not None and float(row["bottom_10_tail_instability_enrichment"]) > 1.25
        for row in run_rows
    )
    notes: list[str] = []
    if best_eval:
        notes.append(f"most accurate run by eval cosine is {best_eval[0]['run_id']}")
    if subject_stability:
        notes.append(f"most stable subject-level run is {subject_stability[0]['run_id']}")
    if session_stability:
        notes.append(f"most stable session-level run is {session_stability[0]['run_id']}")
    if cleanest_tail:
        notes.append(f"cleanest trust tail by bottom-10% threshold is {cleanest_tail[0]['run_id']}")
    if strongest_alignment:
        notes.append(f"strongest instability-linked tail by bottom-10% enrichment is {strongest_alignment[0]['run_id']}")
    if best_eval and cleanest_tail:
        if best_eval[0]["run_id"] == cleanest_tail[0]["run_id"]:
            notes.append("the best average run is also the cleanest trust-tail run under the primary bottom-10% heuristic")
        else:
            notes.append("the best average run is not the same run as the cleanest trust-tail run")
    if heuristic_support:
        notes.append("low-score tails add information beyond mean cosine because bottom-tail samples are enriched in unstable subject/session groups relative to their base rate")
    else:
        notes.append("the current exploratory tail metrics do not show clear instability enrichment beyond mean cosine alone")
    if meta is not None and meta.get("eval_cosine") is not None:
        if cleanest_tail and strongest_alignment and cleanest_tail[0]["run_id"] == "metacognitive_only":
            notes.append("metacognitive_only remains the most interesting paper-2 run because it combines the best average cosine with the cleanest trust tail")
        elif strongest_alignment and strongest_alignment[0]["run_id"] == "metacognitive_only":
            notes.append("metacognitive_only remains scientifically interesting because its low-tail behavior is strongly aligned to instability even when the cleanest-tail ranking differs")
        else:
            notes.append("metacognitive_only remains scientifically interesting because it preserves the strongest average signal under the trust analysis")
    notes.append("this trust-signal pack improves paper-2 publishability as exploratory trust support but does not establish uncertainty calibration or publication-grade evidence")
    return {
        "most_accurate_run": best_eval[0] if best_eval else None,
        "best_transfer_run": best_transfer[0] if best_transfer else None,
        "most_stable_subject_run": subject_stability[0] if subject_stability else None,
        "most_stable_session_run": session_stability[0] if session_stability else None,
        "cleanest_trust_tail_run": cleanest_tail[0] if cleanest_tail else None,
        "strongest_instability_alignment_run": strongest_alignment[0] if strongest_alignment else None,
        "best_average_run_is_also_best_trust_run": bool(best_eval and cleanest_tail and best_eval[0]["run_id"] == cleanest_tail[0]["run_id"]),
        "heuristic_support_for_tail_signal_beyond_mean_cosine": heuristic_support,
        "metacognitive_only_remains_most_interesting": bool(
            meta is not None and best_eval and best_eval[0]["run_id"] == "metacognitive_only"
        ),
        "notes": notes,
        "enough_for_publication_evidence": False,
    }


def build_public_nod_paper2_trust_signal_comparison(
    *,
    root_dir: str | Path,
    run_ids: list[str],
    comparison_output_path: str | Path,
    instability_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root_dir).resolve()
    run_rows: list[dict[str, Any]] = []
    instability_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_root = root / run_id
        report, report_path = _load_run_payload(run_root)
        if report is None:
            run_rows.append({"run_id": run_id, "run_root": str(run_root), "report_path": None, "trust_signal_report_ready": False})
            continue

        run_rows.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "report_path": str(report_path) if report_path is not None else None,
                "experiment_name": report.get("run_identity", {}).get("experiment_name"),
                "eval_cosine": report.get("comparison_context", {}).get("eval_cosine"),
                "transfer_cosine": report.get("comparison_context", {}).get("transfer_cosine"),
                "subject_mean_std": report.get("comparison_context", {}).get("subject_mean_std"),
                "session_mean_std": report.get("comparison_context", {}).get("session_mean_std"),
                "bottom_5_threshold_cosine_lte": _threshold_metric(report, "bottom_5_pct", "threshold_cosine_lte"),
                "bottom_10_threshold_cosine_lte": _threshold_metric(report, "bottom_10_pct", "threshold_cosine_lte"),
                "bottom_20_threshold_cosine_lte": _threshold_metric(report, "bottom_20_pct", "threshold_cosine_lte"),
                "bottom_10_tail_instability_enrichment": _threshold_metric(report, "bottom_10_pct", "tail_instability_enrichment"),
                "bottom_10_worst_subject_enrichment": _threshold_metric(report, "bottom_10_pct", "worst_subject_enrichment"),
                "bottom_10_worst_session_enrichment": _threshold_metric(report, "bottom_10_pct", "worst_session_enrichment"),
                "bottom_10_instability_flag_rate": _trust_score(report, "instability_flag_rate"),
                "bottom_10_subject_association": _trust_score(report, "subject_low_tail_rate_association"),
                "bottom_10_session_association": _trust_score(report, "session_low_tail_rate_association"),
                "worst_subject": report.get("instability_reference", {}).get("worst_subject"),
                "worst_session": report.get("instability_reference", {}).get("worst_session"),
                "trust_signal_report_ready": report.get("state", {}).get("trust_signal_report_ready"),
                "blocked_reasons": report.get("blocked_reasons", []),
            }
        )
        instability_rows.append(
            {
                "run_id": run_id,
                "experiment_name": report.get("run_identity", {}).get("experiment_name"),
                "bottom_5_pct": report.get("threshold_analyses", {}).get("bottom_5_pct"),
                "bottom_10_pct": report.get("threshold_analyses", {}).get("bottom_10_pct"),
                "bottom_20_pct": report.get("threshold_analyses", {}).get("bottom_20_pct"),
                "worst_subject": report.get("instability_reference", {}).get("worst_subject"),
                "worst_session": report.get("instability_reference", {}).get("worst_session"),
            }
        )

    comparison = {
        "root_dir": str(root),
        "report_path": str(Path(comparison_output_path).resolve()),
        "runs": run_rows,
        "rankings": {
            "eval_cosine_desc": _rank_runs(run_rows, "eval_cosine"),
            "transfer_cosine_desc": _rank_runs(run_rows, "transfer_cosine"),
            "subject_stability_asc": _stability_ranking(run_rows, "subject_mean_std"),
            "session_stability_asc": _stability_ranking(run_rows, "session_mean_std"),
            "cleanest_trust_tail_desc": _compare_rows_desc(run_rows, "bottom_10_threshold_cosine_lte"),
            "instability_alignment_desc": _compare_rows_desc(run_rows, "bottom_10_tail_instability_enrichment"),
        },
        "interpretation": _interpret(run_rows),
        "state": {
            "run_count": len(run_rows),
            "all_reports_present": all(row.get("trust_signal_report_ready") for row in run_rows),
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this trust comparison summarizes the paper-2 public-NOD trust-signal pack only",
            "tail enrichment and instability alignment are exploratory trust support, not uncertainty calibration",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }

    instability_comparison = {
        "root_dir": str(root),
        "report_path": str(Path(instability_output_path).resolve()),
        "runs": instability_rows,
        "rankings": {
            "bottom_5_instability_enrichment_desc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "tail_instability_enrichment": row.get("bottom_5_pct", {}).get("tail_instability_enrichment"),
                    }
                    for row in instability_rows
                    if row.get("bottom_5_pct", {}).get("tail_instability_enrichment") is not None
                ],
                key=lambda item: float(item["tail_instability_enrichment"]),
                reverse=True,
            ),
            "bottom_10_instability_enrichment_desc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "tail_instability_enrichment": row.get("bottom_10_pct", {}).get("tail_instability_enrichment"),
                    }
                    for row in instability_rows
                    if row.get("bottom_10_pct", {}).get("tail_instability_enrichment") is not None
                ],
                key=lambda item: float(item["tail_instability_enrichment"]),
                reverse=True,
            ),
            "bottom_20_instability_enrichment_desc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "tail_instability_enrichment": row.get("bottom_20_pct", {}).get("tail_instability_enrichment"),
                    }
                    for row in instability_rows
                    if row.get("bottom_20_pct", {}).get("tail_instability_enrichment") is not None
                ],
                key=lambda item: float(item["tail_instability_enrichment"]),
                reverse=True,
            ),
        },
        "interpretation": {
            "notes": [
                "higher tail-instability enrichment means low-score samples are more concentrated in unstable groups than their base rate",
                "bottom-5/10/20 tail comparisons are exploratory trust-signal summaries, not calibrated risk estimates",
                "this artifact does not establish uncertainty calibration or publication-grade evidence",
            ],
        },
        "state": {
            "all_reports_present": all(row.get("bottom_10_pct", {}).get("available") for row in instability_rows),
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this trust-instability comparison belongs only to the paper-2 public-NOD lane",
            "tail-instability enrichment is an exploratory trust heuristic, not calibrated uncertainty",
            "training_ready remains false",
        ],
    }
    return json_safe(comparison), json_safe(instability_comparison)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare exploratory trust-signal behavior across existing paper-2 public-NOD runs."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    parser.add_argument("--instability-output", default=str(_default_path(DEFAULT_INSTABILITY_OUTPUT)))
    args = parser.parse_args(argv)

    run_ids = args.run_ids or list(DEFAULT_RUNS)
    comparison, instability = build_public_nod_paper2_trust_signal_comparison(
        root_dir=args.root_dir,
        run_ids=run_ids,
        comparison_output_path=args.output,
        instability_output_path=args.instability_output,
    )
    write_report(args.output, comparison)
    write_report(args.instability_output, instability)
    print(json.dumps(json_safe(comparison), indent=2))
    print(json.dumps(json_safe(instability), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
