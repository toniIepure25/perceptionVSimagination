from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.analyze_public_nod_paper2_risk_buckets")

from fmri2img.workflows.analyze_public_nod_paper2_robustness import (  # noqa: E402
    _coerce_pair_ids,
    _load_prepared_dataset,
    _resolve_csv,
)
from fmri2img.workflows.analyze_public_nod_paper2_trust_signals import (  # noqa: E402
    _safe_spearman,
)
from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _resolve_run_report,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_RUN = "baseline"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/risk_bucket_report.json"
REQUESTED_BUCKET_COUNT = 10
BINNING_STRATEGIES = {
    "deciles": 10,
    "quintiles": 5,
    "tertiles": 3,
}


def _resolve_json(run_root: Path, relative_path: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = run_root / relative_path
    if not path.exists():
        return None, None
    return path, _load_optional_json(path)


def _resolve_trust_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    return _resolve_json(run_root, "trust_signal_report.json")


def _resolve_robustness_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    return _resolve_json(run_root, "robustness_report.json")


def _share(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def _enrichment(observed_share: float | None, base_share: float | None) -> float | None:
    if observed_share is None or base_share is None or base_share <= 0:
        return None
    return float(observed_share / base_share)


def _assign_quantile_buckets(frame: pd.DataFrame, requested_bucket_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = frame.copy()
    valid["cosine"] = pd.to_numeric(valid["cosine"], errors="coerce")
    valid = valid.loc[valid["cosine"].notna()].sort_values(["cosine", "pair_id"], ascending=[True, True]).reset_index(drop=True)
    if valid.empty:
        definition = {
            "strategy": "equal_frequency_buckets_by_cosine_rank",
            "requested_bucket_count": int(requested_bucket_count),
            "actual_bucket_count": 0,
            "ordering": "bucket_01_lowest has the lowest cosine scores",
        }
        return valid, definition
    actual_bucket_count = max(1, min(requested_bucket_count, len(valid)))
    valid["bucket_index"] = (valid.index * actual_bucket_count) // len(valid)
    valid["bucket_rank_from_low_end"] = valid["bucket_index"] + 1
    valid["bucket_rank_from_high_end"] = actual_bucket_count - valid["bucket_index"]

    labels: dict[int, str] = {}
    for bucket_index in range(actual_bucket_count):
        if bucket_index == 0:
            suffix = "_lowest"
        elif bucket_index == actual_bucket_count - 1:
            suffix = "_highest"
        else:
            suffix = ""
        labels[bucket_index] = f"bucket_{bucket_index + 1:02d}{suffix}"
    valid["bucket_label"] = valid["bucket_index"].map(labels)
    definition = {
        "strategy": "equal_frequency_buckets_by_cosine_rank",
        "requested_bucket_count": int(requested_bucket_count),
        "actual_bucket_count": int(actual_bucket_count),
        "ordering": "bucket_01_lowest has the lowest cosine scores",
    }
    return valid, definition


def _series_mask(frame: pd.DataFrame, column: str, values: list[str]) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].isin(values)


def _is_nonincreasing(values: list[float | None]) -> bool:
    cleaned = [float(value) for value in values if value is not None]
    if len(cleaned) < 2:
        return False
    return all(lhs >= rhs for lhs, rhs in zip(cleaned, cleaned[1:]))


def _bucket_rows(
    bucketed: pd.DataFrame,
    *,
    worst_subject: str | None,
    worst_session: str | None,
    low_subjects: list[str],
    low_sessions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    low_subject_mask = bucketed["subject"].isin(low_subjects) if "subject" in bucketed.columns else pd.Series(False, index=bucketed.index)
    low_session_mask = bucketed["session"].isin(low_sessions) if "session" in bucketed.columns else pd.Series(False, index=bucketed.index)
    unstable_mask = low_subject_mask | low_session_mask
    worst_subject_mask = (
        bucketed["subject"].eq(worst_subject) if worst_subject is not None and "subject" in bucketed.columns else pd.Series(False, index=bucketed.index)
    )
    worst_session_mask = (
        bucketed["session"].eq(worst_session) if worst_session is not None and "session" in bucketed.columns else pd.Series(False, index=bucketed.index)
    )

    base_worst_subject_share = _share(worst_subject_mask)
    base_worst_session_share = _share(worst_session_mask)
    base_low_subject_share = _share(low_subject_mask)
    base_low_session_share = _share(low_session_mask)
    base_unstable_share = _share(unstable_mask)

    rows: list[dict[str, Any]] = []
    for _, frame in bucketed.groupby("bucket_index", sort=True):
        share_worst_subject = _share(frame["subject"].eq(worst_subject)) if worst_subject is not None and "subject" in frame.columns else None
        share_worst_session = _share(frame["session"].eq(worst_session)) if worst_session is not None and "session" in frame.columns else None
        share_low_subjects = _share(frame["subject"].isin(low_subjects)) if "subject" in frame.columns else None
        share_low_sessions = _share(frame["session"].isin(low_sessions)) if "session" in frame.columns else None
        share_unstable = _share(frame["subject"].isin(low_subjects) | frame["session"].isin(low_sessions))
        rows.append(
            {
                "bucket_label": str(frame["bucket_label"].iloc[0]),
                "bucket_index": int(frame["bucket_index"].iloc[0]),
                "bucket_rank_from_low_end": int(frame["bucket_rank_from_low_end"].iloc[0]),
                "bucket_rank_from_high_end": int(frame["bucket_rank_from_high_end"].iloc[0]),
                "cosine_min": float(frame["cosine"].min()),
                "cosine_max": float(frame["cosine"].max()),
                "sample_count": int(len(frame)),
                "mean_cosine": float(frame["cosine"].mean()),
                "share_in_worst_subject": share_worst_subject,
                "share_in_worst_session": share_worst_session,
                "share_in_low_performing_subjects": share_low_subjects,
                "share_in_low_performing_sessions": share_low_sessions,
                "unstable_group_share": share_unstable,
                "worst_subject_enrichment": _enrichment(share_worst_subject, base_worst_subject_share),
                "worst_session_enrichment": _enrichment(share_worst_session, base_worst_session_share),
                "low_performing_subject_enrichment": _enrichment(share_low_subjects, base_low_subject_share),
                "low_performing_session_enrichment": _enrichment(share_low_sessions, base_low_session_share),
                "unstable_group_enrichment": _enrichment(share_unstable, base_unstable_share),
            }
        )

    unstable_shares = [row["unstable_group_share"] for row in rows]
    worst_subject_enrichments = [row["worst_subject_enrichment"] for row in rows]
    worst_session_enrichments = [row["worst_session_enrichment"] for row in rows]
    mean_cosines = pd.Series([row["mean_cosine"] for row in rows], dtype="float64")

    diagnostics = {
        "unstable_group_share_rises_as_cosine_falls": _is_nonincreasing(unstable_shares),
        "worst_subject_enrichment_rises_as_cosine_falls": _is_nonincreasing(worst_subject_enrichments),
        "worst_session_enrichment_rises_as_cosine_falls": _is_nonincreasing(worst_session_enrichments),
        "bucket_mean_cosine_vs_instability_share_spearman": _safe_spearman(mean_cosines, pd.Series(unstable_shares, dtype="float64")),
        "bucket_mean_cosine_vs_worst_subject_share_spearman": _safe_spearman(
            mean_cosines,
            pd.Series(worst_subject_enrichments, dtype="float64"),
        ),
        "bucket_mean_cosine_vs_worst_session_share_spearman": _safe_spearman(
            mean_cosines,
            pd.Series(worst_session_enrichments, dtype="float64"),
        ),
        "lowest_bucket_unstable_share": unstable_shares[0] if unstable_shares else None,
        "highest_bucket_unstable_share": unstable_shares[-1] if unstable_shares else None,
        "risk_gap_lowest_vs_highest": (
            float(unstable_shares[0] - unstable_shares[-1])
            if unstable_shares and unstable_shares[0] is not None and unstable_shares[-1] is not None
            else None
        ),
    }
    diagnostics["monotonic_instability_signal_present"] = bool(
        diagnostics["unstable_group_share_rises_as_cosine_falls"]
        and diagnostics["risk_gap_lowest_vs_highest"] is not None
        and diagnostics["risk_gap_lowest_vs_highest"] > 0
    )
    return rows, diagnostics


def _build_strategy_report(
    frame: pd.DataFrame,
    *,
    strategy_name: str,
    bucket_count: int,
    worst_subject: str | None,
    worst_session: str | None,
    low_subjects: list[str],
    low_sessions: list[str],
) -> dict[str, Any]:
    bucketed, definition = _assign_quantile_buckets(frame, bucket_count)
    rows, diagnostics = _bucket_rows(
        bucketed,
        worst_subject=worst_subject,
        worst_session=worst_session,
        low_subjects=low_subjects,
        low_sessions=low_sessions,
    )
    definition["strategy_name"] = strategy_name
    return {
        "bucket_definition": definition,
        "bins": rows,
        "monotonicity_diagnostics": diagnostics,
    }


def _subset_summary(
    frame: pd.DataFrame,
    *,
    worst_subject: str | None,
    worst_session: str | None,
    low_subjects: list[str],
    low_sessions: list[str],
) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "sample_share_of_run": 0.0,
            "mean_cosine": None,
            "worst_subject_share": None,
            "worst_session_share": None,
            "low_performing_subject_share": None,
            "low_performing_session_share": None,
        }
    return {
        "sample_count": int(len(frame)),
        "mean_cosine": float(pd.to_numeric(frame["cosine"], errors="coerce").mean()),
        "worst_subject_share": _share(frame["subject"].eq(worst_subject)) if worst_subject is not None and "subject" in frame.columns else None,
        "worst_session_share": _share(frame["session"].eq(worst_session)) if worst_session is not None and "session" in frame.columns else None,
        "low_performing_subject_share": _share(frame["subject"].isin(low_subjects)) if "subject" in frame.columns else None,
        "low_performing_session_share": _share(frame["session"].isin(low_sessions)) if "session" in frame.columns else None,
    }


def _conditioned_tables(
    merged: pd.DataFrame,
    *,
    worst_subject: str | None,
    worst_session: str | None,
    low_subjects: list[str],
    low_sessions: list[str],
) -> dict[str, Any]:
    low_subject_mask = _series_mask(merged, "subject", low_subjects)
    low_session_mask = _series_mask(merged, "session", low_sessions)
    inside_mask = low_subject_mask | low_session_mask
    outside_mask = ~inside_mask
    total_count = int(len(merged))

    inside = merged.loc[inside_mask].copy()
    outside = merged.loc[outside_mask].copy()
    inside_summary = _subset_summary(
        inside,
        worst_subject=worst_subject,
        worst_session=worst_session,
        low_subjects=low_subjects,
        low_sessions=low_sessions,
    )
    outside_summary = _subset_summary(
        outside,
        worst_subject=worst_subject,
        worst_session=worst_session,
        low_subjects=low_subjects,
        low_sessions=low_sessions,
    )
    inside_summary["sample_share_of_run"] = float(len(inside) / total_count) if total_count else 0.0
    outside_summary["sample_share_of_run"] = float(len(outside) / total_count) if total_count else 0.0

    conditioned = {
        "inside_low_performing_groups": {
            **inside_summary,
            "tertiles": _build_strategy_report(
                inside,
                strategy_name="inside_low_performing_groups_tertiles",
                bucket_count=BINNING_STRATEGIES["tertiles"],
                worst_subject=worst_subject,
                worst_session=worst_session,
                low_subjects=low_subjects,
                low_sessions=low_sessions,
            ),
            "quintiles": _build_strategy_report(
                inside,
                strategy_name="inside_low_performing_groups_quintiles",
                bucket_count=BINNING_STRATEGIES["quintiles"],
                worst_subject=worst_subject,
                worst_session=worst_session,
                low_subjects=low_subjects,
                low_sessions=low_sessions,
            ),
        },
        "outside_low_performing_groups": {
            **outside_summary,
            "tertiles": _build_strategy_report(
                outside,
                strategy_name="outside_low_performing_groups_tertiles",
                bucket_count=BINNING_STRATEGIES["tertiles"],
                worst_subject=worst_subject,
                worst_session=worst_session,
                low_subjects=low_subjects,
                low_sessions=low_sessions,
            ),
            "quintiles": _build_strategy_report(
                outside,
                strategy_name="outside_low_performing_groups_quintiles",
                bucket_count=BINNING_STRATEGIES["quintiles"],
                worst_subject=worst_subject,
                worst_session=worst_session,
                low_subjects=low_subjects,
                low_sessions=low_sessions,
            ),
        },
    }

    base_inside_share = inside_summary["sample_share_of_run"]
    separation_summary: dict[str, Any] = {
        "inside_low_performing_groups_mean_cosine": inside_summary["mean_cosine"],
        "outside_low_performing_groups_mean_cosine": outside_summary["mean_cosine"],
        "inside_minus_outside_mean_cosine": (
            float(inside_summary["mean_cosine"] - outside_summary["mean_cosine"])
            if inside_summary["mean_cosine"] is not None and outside_summary["mean_cosine"] is not None
            else None
        ),
    }
    for strategy_name in ("tertiles", "quintiles"):
        strategy = conditioned["inside_low_performing_groups"][strategy_name]
        rows = strategy["bins"]
        lowest_share = rows[0]["unstable_group_share"] if rows else None
        highest_share = rows[-1]["unstable_group_share"] if rows else None
        separation_summary[strategy_name] = {
            "inside_low_performing_groups_lowest_bin_unstable_share": lowest_share,
            "inside_low_performing_groups_highest_bin_unstable_share": highest_share,
            "inside_low_performing_groups_risk_gap": (
                float(lowest_share - highest_share)
                if lowest_share is not None and highest_share is not None
                else None
            ),
            "inside_low_performing_groups_lowest_bin_enrichment_vs_base_share": _enrichment(lowest_share, base_inside_share),
            "inside_low_performing_groups_highest_bin_enrichment_vs_base_share": _enrichment(highest_share, base_inside_share),
            "inside_low_performing_groups_correlation": strategy["monotonicity_diagnostics"].get(
                "bucket_mean_cosine_vs_instability_share_spearman"
            ),
        }
    conditioned["global_vs_conditioned_separation"] = separation_summary
    return conditioned


def analyze_public_nod_paper2_risk_buckets(*, run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    output_path = Path(output_path).resolve()
    report_path, report = _resolve_run_report(run_root)
    robustness_path, robustness = _resolve_robustness_report(run_root)
    trust_path, trust = _resolve_trust_report(run_root)
    per_trial_pairs_path = _resolve_csv(run_root, "transfer/per_trial_pairs.csv", "per_trial_pairs.csv")

    blocked_reasons: list[str] = []
    if report is None:
        blocked_reasons.append("run summary report is missing")
    if robustness is None:
        blocked_reasons.append("robustness_report.json is missing")
    if trust is None:
        blocked_reasons.append("trust_signal_report.json is missing")
    if per_trial_pairs_path is None:
        blocked_reasons.append("per_trial_pairs.csv is missing")

    prepared_path, prepared_df, prepared_blockers = _load_prepared_dataset(run_root, report)
    blocked_reasons.extend(prepared_blockers)

    experiment = report.get("experiment", {}) if report is not None else {}
    artifact_paths = {
        "run_root": str(run_root),
        "summary_report": str(report_path) if report_path is not None else None,
        "robustness_report": str(robustness_path) if robustness_path is not None else None,
        "trust_signal_report": str(trust_path) if trust_path is not None else None,
        "prepared_dataset": str(prepared_path) if prepared_path is not None else None,
        "per_trial_pairs": str(per_trial_pairs_path) if per_trial_pairs_path is not None else None,
    }

    if per_trial_pairs_path is None:
        return {
            "report_path": str(output_path),
            "run_identity": {
                "run_id": run_root.name,
                "experiment_name": experiment.get("name"),
                "benchmark_role": experiment.get("benchmark_role"),
                "evidence_tier": experiment.get("evidence_tier"),
            },
            "artifact_paths_consumed": artifact_paths,
            "blocked_reasons": blocked_reasons,
            "state": {
                "risk_bucket_report_ready": False,
                "exploratory_only": True,
                "evidence_ready_candidate": False,
                "training_ready": False,
            },
            "interpretation": [
                "the risk-bucket analyzer could not run because per_trial_pairs.csv was missing",
                "this lane remains exploratory only",
            ],
        }

    pairs_df = pd.read_csv(per_trial_pairs_path)
    pairs_df["cosine"] = pd.to_numeric(pairs_df["cosine"], errors="coerce")
    join_succeeded = False
    if prepared_df is not None and "pair_id" in pairs_df.columns and "pair_id" in prepared_df.columns:
        pairs_df, prepared_df = _coerce_pair_ids(pairs_df, prepared_df)
        join_cols = [col for col in ("pair_id", "subject", "session", "split", "nsd_id", "condition") if col in prepared_df.columns]
        merged = pairs_df.merge(prepared_df[join_cols].drop_duplicates(), on="pair_id", how="left")
        join_succeeded = True
    else:
        merged = pairs_df.copy()
        blocked_reasons.append("prepared dataset join unavailable for risk-bucket analysis")

    instability_reference = trust.get("instability_reference", {}) if trust is not None else {}
    worst_subject = instability_reference.get("worst_subject", {}).get("subject")
    worst_session = instability_reference.get("worst_session", {}).get("session")
    low_subjects = list(instability_reference.get("low_performing_subjects", {}).get("selected_groups", []))
    low_sessions = list(instability_reference.get("low_performing_sessions", {}).get("selected_groups", []))

    strategy_reports = {
        strategy_name: _build_strategy_report(
            merged,
            strategy_name=strategy_name,
            bucket_count=bucket_count,
            worst_subject=worst_subject,
            worst_session=worst_session,
            low_subjects=low_subjects,
            low_sessions=low_sessions,
        )
        for strategy_name, bucket_count in BINNING_STRATEGIES.items()
    }
    bucket_definition = strategy_reports["deciles"]["bucket_definition"]
    bucket_rows = strategy_reports["deciles"]["bins"]
    diagnostics = strategy_reports["deciles"]["monotonicity_diagnostics"]
    conditioned_group_tables = _conditioned_tables(
        merged,
        worst_subject=worst_subject,
        worst_session=worst_session,
        low_subjects=low_subjects,
        low_sessions=low_sessions,
    )

    notes: list[str] = []
    if join_succeeded:
        notes.append("risk-bucket analysis joined per_trial_pairs.csv to the fixed prepared dataset on pair_id")
    else:
        notes.append("risk-bucket analysis remains partially blocked because the prepared-dataset join did not complete")
    if diagnostics.get("monotonic_instability_signal_present"):
        notes.append(
            f"unstable-group share rises as cosine falls across the ordered buckets, with a lowest-vs-highest risk gap of {diagnostics['risk_gap_lowest_vs_highest']:.3f}"
        )
    else:
        notes.append("the ordered bucket shares do not show a clean monotonic instability signal under the current heuristic")
    if diagnostics.get("bucket_mean_cosine_vs_instability_share_spearman") is not None:
        notes.append(
            "bucket-level cosine and unstable-group share have "
            f"Spearman correlation {diagnostics['bucket_mean_cosine_vs_instability_share_spearman']:.3f}"
        )
    tertile_diag = strategy_reports["tertiles"]["monotonicity_diagnostics"]
    quintile_diag = strategy_reports["quintiles"]["monotonicity_diagnostics"]
    if tertile_diag.get("monotonic_instability_signal_present"):
        notes.append("tertiles do show a clean monotonic instability signal under the current heuristic")
    elif tertile_diag.get("bucket_mean_cosine_vs_instability_share_spearman") is not None:
        notes.append(
            "tertiles remain exploratory only, with bucket-level cosine versus instability Spearman "
            f"{tertile_diag['bucket_mean_cosine_vs_instability_share_spearman']:.3f}"
        )
    if quintile_diag.get("monotonic_instability_signal_present"):
        notes.append("quintiles do show a clean monotonic instability signal under the current heuristic")
    elif quintile_diag.get("bucket_mean_cosine_vs_instability_share_spearman") is not None:
        notes.append(
            "quintiles remain exploratory only, with bucket-level cosine versus instability Spearman "
            f"{quintile_diag['bucket_mean_cosine_vs_instability_share_spearman']:.3f}"
        )
    notes.append("this report is exploratory risk stratification only and does not establish probabilistic calibration")

    ready = bool(
        report is not None
        and robustness is not None
        and trust is not None
        and per_trial_pairs_path is not None
        and join_succeeded
        and all(len(strategy_reports[name]["bins"]) > 1 for name in ("deciles", "tertiles", "quintiles"))
    )

    payload = {
        "report_path": str(output_path),
        "run_identity": {
            "run_id": run_root.name,
            "experiment_name": experiment.get("name"),
            "benchmark_role": experiment.get("benchmark_role"),
            "evidence_tier": experiment.get("evidence_tier"),
        },
        "artifact_paths_consumed": artifact_paths,
        "prepared_dataset_join": {
            "join_succeeded": join_succeeded,
            "prepared_dataset_rows": int(len(prepared_df)) if prepared_df is not None else None,
            "rows_with_subject_metadata": int(merged["subject"].notna().sum()) if "subject" in merged.columns else 0,
            "rows_with_session_metadata": int(merged["session"].notna().sum()) if "session" in merged.columns else 0,
        },
        "comparison_context": {
            "eval_cosine": report.get("eval_summary", {}).get("overall_cosine") if report is not None else None,
            "transfer_cosine": report.get("transfer_summary", {}).get("overall_cosine") if report is not None else None,
            "subject_mean_std": robustness.get("dispersion_indicators", {}).get("subject_mean_std") if robustness is not None else None,
            "session_mean_std": robustness.get("dispersion_indicators", {}).get("session_mean_std") if robustness is not None else None,
            "primary_tail_threshold_cosine_lte": trust.get("low_score_thresholds", {}).get(
                trust.get("descriptive_trust_scores", {}).get("primary_threshold_label", ""),
                {},
            ).get("threshold_cosine_lte")
            if trust is not None
            else None,
            "primary_tail_instability_enrichment": trust.get("descriptive_trust_scores", {}).get("tail_instability_enrichment") if trust is not None else None,
            "primary_tail_threshold_label": trust.get("descriptive_trust_scores", {}).get("primary_threshold_label") if trust is not None else None,
        },
        "risk_tables": strategy_reports,
        "bucket_definition": bucket_definition,
        "buckets": bucket_rows,
        "monotonicity_diagnostics": diagnostics,
        "low_performing_group_conditioning": conditioned_group_tables,
        "blocked_reasons": blocked_reasons,
        "interpretation": notes,
        "state": {
            "risk_bucket_report_ready": ready,
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this risk-bucket report analyzes existing paper-2 public-NOD artifacts only",
            "bucketed instability behavior here is exploratory risk stratification, not calibrated uncertainty",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }
    return json_safe(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cosine-bucketed exploratory risk stratification for one existing paper-2 public-NOD run."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve() if args.run_root else Path(args.root_dir).resolve() / args.run_id
    output = args.output or str(run_root / "risk_bucket_report.json")
    payload = analyze_public_nod_paper2_risk_buckets(run_root=run_root, output_path=output)
    write_report(output, payload)
    print(json.dumps(json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
