from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.compare_public_nod_paper2_risk_buckets")

from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _rank_runs,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/risk_bucket_comparison.json"
DEFAULT_MONOTONICITY_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/risk_monotonicity_comparison.json"
)
DEFAULT_RUNS = ("baseline", "early_visual_only", "metacognitive_only")
COARSE_STRATEGIES = ("tertiles", "quintiles")


def _load_run_payload(run_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    path = run_root / "risk_bucket_report.json"
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


def _metric(report: dict[str, Any] | None, section: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.get(section, {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strategy_metric(report: dict[str, Any] | None, strategy_name: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("risk_tables", {}).get(strategy_name, {}).get("monotonicity_diagnostics", {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strategy_flag(report: dict[str, Any] | None, strategy_name: str, key: str) -> bool | None:
    if report is None:
        return None
    value = report.get("risk_tables", {}).get(strategy_name, {}).get("monotonicity_diagnostics", {}).get(key)
    if value is None:
        return None
    return bool(value)


def _conditioning_metric(report: dict[str, Any] | None, strategy_name: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("low_performing_group_conditioning", {}).get("global_vs_conditioned_separation", {}).get(strategy_name, {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _conditioning_summary_metric(report: dict[str, Any] | None, key: str) -> float | None:
    if report is None:
        return None
    value = report.get("low_performing_group_conditioning", {}).get("global_vs_conditioned_separation", {}).get(key)
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


def _best_monotonic_run(run_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in run_rows if row.get("decile_monotonic_instability_signal_present")]
    if not eligible:
        eligible = [row for row in run_rows if row.get("decile_bucket_mean_cosine_vs_instability_share_spearman") is not None]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda row: (
            0 if row.get("decile_monotonic_instability_signal_present") else 1,
            float(row.get("decile_bucket_mean_cosine_vs_instability_share_spearman", 0.0)),
            -float(row.get("decile_risk_gap_lowest_vs_highest", 0.0)),
        ),
    )
    best = ordered[0]
    return {
        "run_id": best["run_id"],
        "experiment_name": best.get("experiment_name"),
        "bucket_mean_cosine_vs_instability_share_spearman": best.get("decile_bucket_mean_cosine_vs_instability_share_spearman"),
        "risk_gap_lowest_vs_highest": best.get("decile_risk_gap_lowest_vs_highest"),
        "monotonic_instability_signal_present": best.get("decile_monotonic_instability_signal_present"),
    }


def _strategy_improves_over_deciles(row: dict[str, Any], strategy_name: str) -> bool:
    decile_flag = bool(row.get("decile_monotonic_instability_signal_present"))
    strategy_flag = bool(row.get(f"{strategy_name}_monotonic_instability_signal_present"))
    decile_corr = row.get("decile_bucket_mean_cosine_vs_instability_share_spearman")
    strategy_corr = row.get(f"{strategy_name}_bucket_mean_cosine_vs_instability_share_spearman")
    decile_gap = row.get("decile_risk_gap_lowest_vs_highest")
    strategy_gap = row.get(f"{strategy_name}_risk_gap_lowest_vs_highest")
    if strategy_flag and not decile_flag:
        return True
    if strategy_corr is not None and decile_corr is not None and float(strategy_corr) < float(decile_corr):
        return True
    if strategy_gap is not None and decile_gap is not None and float(strategy_gap) > float(decile_gap):
        return True
    return False


def _best_coarse_run(run_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in run_rows:
        for strategy_name in COARSE_STRATEGIES:
            corr = row.get(f"{strategy_name}_bucket_mean_cosine_vs_instability_share_spearman")
            gap = row.get(f"{strategy_name}_risk_gap_lowest_vs_highest")
            flag = row.get(f"{strategy_name}_monotonic_instability_signal_present")
            if corr is None and gap is None and flag is None:
                continue
            candidates.append(
                {
                    "run_id": row["run_id"],
                    "experiment_name": row.get("experiment_name"),
                    "strategy_name": strategy_name,
                    "monotonic_instability_signal_present": flag,
                    "bucket_mean_cosine_vs_instability_share_spearman": corr,
                    "risk_gap_lowest_vs_highest": gap,
                }
            )
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda item: (
            0 if item.get("monotonic_instability_signal_present") else 1,
            float(item.get("bucket_mean_cosine_vs_instability_share_spearman", 0.0)),
            -float(item.get("risk_gap_lowest_vs_highest", 0.0)),
        ),
    )
    return ordered[0]


def _interpret(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_eval = _rank_runs(run_rows, "eval_cosine")
    best_transfer = _rank_runs(run_rows, "transfer_cosine")
    subject_stability = _stability_ranking(run_rows, "subject_mean_std")
    session_stability = _stability_ranking(run_rows, "session_mean_std")
    cleanest_monotonic = _best_monotonic_run(run_rows)
    cleanest_coarse = _best_coarse_run(run_rows)
    cleanest_tail = _compare_rows_desc(run_rows, "bottom_10_threshold_cosine_lte")
    by_run = {row["run_id"]: row for row in run_rows}
    meta = by_run.get("metacognitive_only")

    bucketed_signal = any(row.get("decile_monotonic_instability_signal_present") for row in run_rows)
    tertile_improvement = any(row.get("tertiles_improve_over_deciles") for row in run_rows)
    quintile_improvement = any(row.get("quintiles_improve_over_deciles") for row in run_rows)
    coarse_signal = tertile_improvement or quintile_improvement
    notes: list[str] = []
    if best_eval:
        notes.append(f"most accurate run by eval cosine is {best_eval[0]['run_id']}")
    if subject_stability:
        notes.append(f"most stable subject-level run is {subject_stability[0]['run_id']}")
    if session_stability:
        notes.append(f"most stable session-level run is {session_stability[0]['run_id']}")
    if cleanest_monotonic:
        notes.append(f"cleanest monotonic risk stratification is currently {cleanest_monotonic['run_id']}")
    if cleanest_coarse:
        notes.append(
            f"cleanest coarse-bin risk stratification is currently {cleanest_coarse['run_id']} using {cleanest_coarse['strategy_name']}"
        )
    if cleanest_tail:
        notes.append(f"cleanest bottom-10 trust tail remains {cleanest_tail[0]['run_id']}")
    if bucketed_signal:
        notes.append("instability rises as cosine falls for at least one real run under the ordered bucket analysis")
    else:
        notes.append("the bucketed analysis does not yet show a convincing monotonic risk signal beyond the simpler trust-tail heuristics")
    if tertile_improvement:
        notes.append("tertiles strengthen the exploratory risk signal over deciles for at least one real run")
    else:
        notes.append("tertiles do not cleanly strengthen the exploratory risk signal over deciles in the current pack")
    if quintile_improvement:
        notes.append("quintiles strengthen the exploratory risk signal over deciles for at least one real run")
    else:
        notes.append("quintiles do not cleanly strengthen the exploratory risk signal over deciles in the current pack")
    if coarse_signal:
        notes.append("coarse-bin risk adds information beyond mean cosine and the existing bottom-tail heuristic because at least one coarser view strengthens the graded-risk signal")
    else:
        notes.append("the current coarse-bin follow-up mainly confirms that the risk signal is fragile or noisy beyond the existing bottom-tail heuristic")
    if best_eval and cleanest_monotonic:
        if best_eval[0]["run_id"] == cleanest_monotonic["run_id"]:
            notes.append("the best average run is also the cleanest risk-stratified run under the current bucketed heuristic")
        else:
            notes.append("the best average run is not the same run as the cleanest risk-stratified run")
    if meta is not None and best_eval and best_eval[0]["run_id"] == "metacognitive_only":
        if cleanest_monotonic and cleanest_monotonic["run_id"] == "metacognitive_only":
            notes.append("metacognitive_only remains the strongest paper-2 run because it leads on average accuracy and on bucketed risk stratification")
        else:
            notes.append("metacognitive_only remains the strongest paper-2 run because it preserves the best average cosine and trust-tail cleanliness even when bucketed risk is considered")
    notes.append("this risk-bucket pack improves publishability as exploratory risk stratification, not calibrated uncertainty or publication-grade evidence")
    return {
        "most_accurate_run": best_eval[0] if best_eval else None,
        "best_transfer_run": best_transfer[0] if best_transfer else None,
        "most_stable_subject_run": subject_stability[0] if subject_stability else None,
        "most_stable_session_run": session_stability[0] if session_stability else None,
        "cleanest_monotonic_risk_run": cleanest_monotonic,
        "cleanest_coarse_risk_run": cleanest_coarse,
        "cleanest_trust_tail_run": cleanest_tail[0] if cleanest_tail else None,
        "best_average_run_is_also_best_risk_stratified_run": bool(
            best_eval and cleanest_monotonic and best_eval[0]["run_id"] == cleanest_monotonic["run_id"]
        ),
        "tertiles_improve_monotonicity_over_deciles": tertile_improvement,
        "quintiles_improve_monotonicity_over_deciles": quintile_improvement,
        "bucketed_risk_adds_information_beyond_mean_cosine_and_simple_tail_heuristics": bucketed_signal,
        "coarse_bin_risk_adds_information_beyond_mean_cosine_and_bottom_tail_heuristics": coarse_signal,
        "metacognitive_only_remains_strongest": bool(meta is not None and best_eval and best_eval[0]["run_id"] == "metacognitive_only"),
        "notes": notes,
        "enough_for_publication_evidence": False,
    }


def build_public_nod_paper2_risk_bucket_comparison(
    *,
    root_dir: str | Path,
    run_ids: list[str],
    comparison_output_path: str | Path,
    monotonicity_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root_dir).resolve()
    run_rows: list[dict[str, Any]] = []
    monotonicity_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_root = root / run_id
        report, report_path = _load_run_payload(run_root)
        if report is None:
            run_rows.append({"run_id": run_id, "run_root": str(run_root), "report_path": None, "risk_bucket_report_ready": False})
            continue

        diagnostics = report.get("monotonicity_diagnostics", {})
        comparison_context = report.get("comparison_context", {})
        run_rows.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "report_path": str(report_path) if report_path is not None else None,
                "experiment_name": report.get("run_identity", {}).get("experiment_name"),
                "eval_cosine": comparison_context.get("eval_cosine"),
                "transfer_cosine": comparison_context.get("transfer_cosine"),
                "subject_mean_std": comparison_context.get("subject_mean_std"),
                "session_mean_std": comparison_context.get("session_mean_std"),
                "primary_tail_threshold_label": comparison_context.get("primary_tail_threshold_label"),
                "bottom_10_threshold_cosine_lte": comparison_context.get("primary_tail_threshold_cosine_lte"),
                "primary_tail_instability_enrichment": comparison_context.get("primary_tail_instability_enrichment"),
                "decile_lowest_bucket_unstable_share": diagnostics.get("lowest_bucket_unstable_share"),
                "decile_highest_bucket_unstable_share": diagnostics.get("highest_bucket_unstable_share"),
                "decile_risk_gap_lowest_vs_highest": diagnostics.get("risk_gap_lowest_vs_highest"),
                "decile_monotonic_instability_signal_present": diagnostics.get("monotonic_instability_signal_present"),
                "decile_bucket_mean_cosine_vs_instability_share_spearman": diagnostics.get("bucket_mean_cosine_vs_instability_share_spearman"),
                "decile_bucket_mean_cosine_vs_worst_subject_share_spearman": diagnostics.get("bucket_mean_cosine_vs_worst_subject_share_spearman"),
                "decile_bucket_mean_cosine_vs_worst_session_share_spearman": diagnostics.get("bucket_mean_cosine_vs_worst_session_share_spearman"),
                "tertiles_risk_gap_lowest_vs_highest": _strategy_metric(report, "tertiles", "risk_gap_lowest_vs_highest"),
                "tertiles_monotonic_instability_signal_present": _strategy_flag(report, "tertiles", "monotonic_instability_signal_present"),
                "tertiles_bucket_mean_cosine_vs_instability_share_spearman": _strategy_metric(
                    report,
                    "tertiles",
                    "bucket_mean_cosine_vs_instability_share_spearman",
                ),
                "quintiles_risk_gap_lowest_vs_highest": _strategy_metric(report, "quintiles", "risk_gap_lowest_vs_highest"),
                "quintiles_monotonic_instability_signal_present": _strategy_flag(report, "quintiles", "monotonic_instability_signal_present"),
                "quintiles_bucket_mean_cosine_vs_instability_share_spearman": _strategy_metric(
                    report,
                    "quintiles",
                    "bucket_mean_cosine_vs_instability_share_spearman",
                ),
                "inside_low_performing_groups_mean_cosine": _conditioning_summary_metric(
                    report,
                    "inside_low_performing_groups_mean_cosine",
                ),
                "outside_low_performing_groups_mean_cosine": _conditioning_summary_metric(
                    report,
                    "outside_low_performing_groups_mean_cosine",
                ),
                "tertiles_conditioned_risk_gap": _conditioning_metric(
                    report,
                    "tertiles",
                    "inside_low_performing_groups_risk_gap",
                ),
                "quintiles_conditioned_risk_gap": _conditioning_metric(
                    report,
                    "quintiles",
                    "inside_low_performing_groups_risk_gap",
                ),
                "risk_bucket_report_ready": report.get("state", {}).get("risk_bucket_report_ready"),
                "blocked_reasons": report.get("blocked_reasons", []),
            }
        )
        run_rows[-1]["tertiles_improve_over_deciles"] = _strategy_improves_over_deciles(run_rows[-1], "tertiles")
        run_rows[-1]["quintiles_improve_over_deciles"] = _strategy_improves_over_deciles(run_rows[-1], "quintiles")
        monotonicity_rows.append(
            {
                "run_id": run_id,
                "experiment_name": report.get("run_identity", {}).get("experiment_name"),
                "deciles": report.get("risk_tables", {}).get("deciles"),
                "tertiles": report.get("risk_tables", {}).get("tertiles"),
                "quintiles": report.get("risk_tables", {}).get("quintiles"),
                "low_performing_group_conditioning": report.get("low_performing_group_conditioning"),
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
            "decile_risk_gap_desc": _compare_rows_desc(run_rows, "decile_risk_gap_lowest_vs_highest"),
            "decile_risk_monotonicity_corr_asc": _stability_ranking(run_rows, "decile_bucket_mean_cosine_vs_instability_share_spearman"),
            "tertile_risk_gap_desc": _compare_rows_desc(run_rows, "tertiles_risk_gap_lowest_vs_highest"),
            "tertile_risk_monotonicity_corr_asc": _stability_ranking(run_rows, "tertiles_bucket_mean_cosine_vs_instability_share_spearman"),
            "quintile_risk_gap_desc": _compare_rows_desc(run_rows, "quintiles_risk_gap_lowest_vs_highest"),
            "quintile_risk_monotonicity_corr_asc": _stability_ranking(run_rows, "quintiles_bucket_mean_cosine_vs_instability_share_spearman"),
        },
        "interpretation": _interpret(run_rows),
        "state": {
            "run_count": len(run_rows),
            "all_reports_present": all(row.get("risk_bucket_report_ready") for row in run_rows),
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this risk comparison summarizes the paper-2 public-NOD risk-stratification pack only",
            "bucketed and coarse-bin instability behavior here is exploratory risk support, not uncertainty calibration",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }

    monotonicity_comparison = {
        "root_dir": str(root),
        "report_path": str(Path(monotonicity_output_path).resolve()),
        "runs": monotonicity_rows,
        "rankings": {
            "decile_monotonic_signal_present": [
                {
                    "run_id": row["run_id"],
                    "experiment_name": row["experiment_name"],
                    "monotonic_instability_signal_present": row.get("deciles", {}).get("monotonicity_diagnostics", {}).get("monotonic_instability_signal_present"),
                }
                for row in monotonicity_rows
            ],
            "tertile_monotonic_signal_present": [
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "monotonic_instability_signal_present": row.get("tertiles", {}).get("monotonicity_diagnostics", {}).get("monotonic_instability_signal_present"),
                    }
                    for row in monotonicity_rows
                ]
            ][0],
            "quintile_monotonic_signal_present": [
                {
                    "run_id": row["run_id"],
                    "experiment_name": row["experiment_name"],
                    "monotonic_instability_signal_present": row.get("quintiles", {}).get("monotonicity_diagnostics", {}).get("monotonic_instability_signal_present"),
                }
                for row in monotonicity_rows
            ],
            "decile_bucket_instability_corr_asc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "bucket_mean_cosine_vs_instability_share_spearman": row.get("deciles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman"),
                    }
                    for row in monotonicity_rows
                    if row.get("deciles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman") is not None
                ],
                key=lambda item: float(item["bucket_mean_cosine_vs_instability_share_spearman"]),
            ),
            "tertile_bucket_instability_corr_asc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "bucket_mean_cosine_vs_instability_share_spearman": row.get("tertiles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman"),
                    }
                    for row in monotonicity_rows
                    if row.get("tertiles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman") is not None
                ],
                key=lambda item: float(item["bucket_mean_cosine_vs_instability_share_spearman"]),
            ),
            "quintile_bucket_instability_corr_asc": sorted(
                [
                    {
                        "run_id": row["run_id"],
                        "experiment_name": row["experiment_name"],
                        "bucket_mean_cosine_vs_instability_share_spearman": row.get("quintiles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman"),
                    }
                    for row in monotonicity_rows
                    if row.get("quintiles", {}).get("monotonicity_diagnostics", {}).get("bucket_mean_cosine_vs_instability_share_spearman") is not None
                ],
                key=lambda item: float(item["bucket_mean_cosine_vs_instability_share_spearman"]),
            ),
        },
        "interpretation": {
            "notes": [
                "more negative bucket_mean_cosine_vs_instability_share_spearman indicates cleaner exploratory risk stratification",
                "the decile, tertile, and quintile views are descriptive coarse-bin diagnostics rather than calibration metrics",
                "low-performing-group-conditioned tables are provided as exploratory subgroup summaries only",
                "monotonicity here is a descriptive risk-stratification property, not probabilistic calibration",
                "this artifact does not establish publication-grade evidence",
            ],
        },
        "state": {
            "all_reports_present": all(
                row.get("deciles") is not None and row.get("tertiles") is not None and row.get("quintiles") is not None
                for row in monotonicity_rows
            ),
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this risk-monotonicity comparison belongs only to the paper-2 public-NOD lane",
            "bucket-level and coarse-bin monotonicity are exploratory risk stratification, not calibrated uncertainty",
            "training_ready remains false",
        ],
    }
    return json_safe(comparison), json_safe(monotonicity_comparison)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare cosine-bucketed exploratory risk stratification across existing paper-2 public-NOD runs."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    parser.add_argument("--monotonicity-output", default=str(_default_path(DEFAULT_MONOTONICITY_OUTPUT)))
    args = parser.parse_args(argv)

    run_ids = args.run_ids or list(DEFAULT_RUNS)
    comparison, monotonicity = build_public_nod_paper2_risk_bucket_comparison(
        root_dir=args.root_dir,
        run_ids=run_ids,
        comparison_output_path=args.output,
        monotonicity_output_path=args.monotonicity_output,
    )
    write_report(args.output, comparison)
    write_report(args.monotonicity_output, monotonicity)
    print(json.dumps(json_safe(comparison), indent=2))
    print(json.dumps(json_safe(monotonicity), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
