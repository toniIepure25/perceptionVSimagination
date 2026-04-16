from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.analyze_public_nod_paper2_trust_signals")

from fmri2img.workflows.analyze_public_nod_paper2_robustness import (  # noqa: E402
    _coerce_pair_ids,
    _load_prepared_dataset,
    _resolve_csv,
)
from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _resolve_reliability_report,
    _resolve_run_report,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_RUN = "baseline"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/trust_signal_report.json"
LOW_TAIL_SPECS = (
    ("bottom_5_pct", 0.05),
    ("bottom_10_pct", 0.10),
    ("bottom_20_pct", 0.20),
)
PRIMARY_TAIL_LABEL = "bottom_10_pct"
LOW_PERFORMING_GROUP_FRACTION = 0.25


def _resolve_robustness_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = run_root / "robustness_report.json"
    if not path.exists():
        return None, None
    return path, _load_optional_json(path)


def _safe_spearman(lhs: pd.Series, rhs: pd.Series) -> float | None:
    paired = pd.concat([pd.to_numeric(lhs, errors="coerce"), pd.to_numeric(rhs, errors="coerce")], axis=1).dropna()
    if len(paired) < 2:
        return None
    if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return None
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
    if pd.isna(value):
        return None
    return float(value)


def _group_mean_series(merged: pd.DataFrame, group_key: str) -> pd.Series:
    return merged.groupby(group_key)["cosine"].mean().sort_values()


def _low_performing_groups(group_means: pd.Series) -> list[str]:
    if group_means.empty:
        return []
    count = max(1, int(math.ceil(len(group_means) * LOW_PERFORMING_GROUP_FRACTION)))
    return [str(value) for value in group_means.head(count).index.tolist()]


def _enrichment_ratio(observed_share: float | None, base_share: float | None) -> float | None:
    if observed_share is None or base_share is None or base_share <= 0:
        return None
    return float(observed_share / base_share)


def _share(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def _group_low_tail_association(merged: pd.DataFrame, group_key: str, tail_mask: pd.Series) -> float | None:
    if group_key not in merged.columns:
        return None
    frame = merged.assign(low_tail=tail_mask.values).groupby(group_key, dropna=False).agg(
        cosine_mean=("cosine", "mean"),
        low_tail_rate=("low_tail", "mean"),
    )
    return _safe_spearman(frame["cosine_mean"], frame["low_tail_rate"])


def _threshold_analysis(
    merged: pd.DataFrame,
    *,
    label: str,
    fraction: float,
    worst_subject: str | None,
    worst_session: str | None,
    low_subjects: list[str],
    low_sessions: list[str],
) -> dict[str, Any]:
    cosine = pd.to_numeric(merged["cosine"], errors="coerce")
    ranked = cosine.dropna().sort_values()
    if ranked.empty:
        return {
            "available": False,
            "reason": "cosine values are not recoverable from per_trial_pairs.csv",
        }

    sample_count = max(1, int(math.ceil(len(ranked) * fraction)))
    threshold = float(ranked.iloc[sample_count - 1])
    tail_mask = cosine <= threshold
    tail = merged.loc[tail_mask].copy()

    subject_mask = merged["subject"].isin(low_subjects) if "subject" in merged.columns else pd.Series(False, index=merged.index)
    session_mask = merged["session"].isin(low_sessions) if "session" in merged.columns else pd.Series(False, index=merged.index)
    unstable_mask = subject_mask | session_mask

    worst_subject_mask = merged["subject"].eq(worst_subject) if worst_subject is not None and "subject" in merged.columns else pd.Series(False, index=merged.index)
    worst_session_mask = merged["session"].eq(worst_session) if worst_session is not None and "session" in merged.columns else pd.Series(False, index=merged.index)

    worst_subject_base_share = _share(worst_subject_mask)
    worst_session_base_share = _share(worst_session_mask)
    low_subject_base_share = _share(subject_mask)
    low_session_base_share = _share(session_mask)
    unstable_base_share = _share(unstable_mask)

    tail_worst_subject_share = _share(tail["subject"].eq(worst_subject)) if worst_subject is not None and "subject" in tail.columns else None
    tail_worst_session_share = _share(tail["session"].eq(worst_session)) if worst_session is not None and "session" in tail.columns else None
    tail_low_subject_share = _share(tail["subject"].isin(low_subjects)) if "subject" in tail.columns else None
    tail_low_session_share = _share(tail["session"].isin(low_sessions)) if "session" in tail.columns else None
    tail_unstable_share = _share(tail["subject"].isin(low_subjects) | tail["session"].isin(low_sessions))

    return {
        "available": True,
        "tail_fraction": fraction,
        "sample_count": int(len(tail)),
        "threshold_cosine_lte": threshold,
        "tail_mean_cosine": float(pd.to_numeric(tail["cosine"], errors="coerce").mean()),
        "tail_cosine_std": float(pd.to_numeric(tail["cosine"], errors="coerce").std(ddof=0)),
        "tail_instability_overlap": {
            "count": int((tail["subject"].isin(low_subjects) | tail["session"].isin(low_sessions)).sum()),
            "share_of_tail": tail_unstable_share,
            "base_unstable_share": unstable_base_share,
        },
        "tail_instability_enrichment": _enrichment_ratio(tail_unstable_share, unstable_base_share),
        "worst_subject_enrichment": _enrichment_ratio(tail_worst_subject_share, worst_subject_base_share),
        "worst_session_enrichment": _enrichment_ratio(tail_worst_session_share, worst_session_base_share),
        "low_performing_subject_overlap": {
            "share_of_tail": tail_low_subject_share,
            "base_share": low_subject_base_share,
        },
        "low_performing_subject_enrichment": _enrichment_ratio(tail_low_subject_share, low_subject_base_share),
        "low_performing_session_overlap": {
            "share_of_tail": tail_low_session_share,
            "base_share": low_session_base_share,
        },
        "low_performing_session_enrichment": _enrichment_ratio(tail_low_session_share, low_session_base_share),
        "instability_flag_rate": tail_unstable_share,
        "tracks_worst_subject": bool(_enrichment_ratio(tail_worst_subject_share, worst_subject_base_share) and _enrichment_ratio(tail_worst_subject_share, worst_subject_base_share) > 1.0),
        "tracks_worst_session": bool(_enrichment_ratio(tail_worst_session_share, worst_session_base_share) and _enrichment_ratio(tail_worst_session_share, worst_session_base_share) > 1.0),
        "rank_associations": {
            "subject_mean_vs_low_tail_rate_spearman": _group_low_tail_association(merged, "subject", tail_mask),
            "session_mean_vs_low_tail_rate_spearman": _group_low_tail_association(merged, "session", tail_mask),
        },
    }


def analyze_public_nod_paper2_trust_signals(*, run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    output_path = Path(output_path).resolve()
    report_path, report = _resolve_run_report(run_root)
    reliability_path, reliability = _resolve_reliability_report(run_root)
    robustness_path, robustness = _resolve_robustness_report(run_root)
    per_trial_pairs_path = _resolve_csv(run_root, "transfer/per_trial_pairs.csv", "per_trial_pairs.csv")

    blocked_reasons: list[str] = []
    if report is None:
        blocked_reasons.append("run summary report is missing")
    if reliability is None:
        blocked_reasons.append("reliability_seed_report.json is missing")
    if robustness is None:
        blocked_reasons.append("robustness_report.json is missing")
    if per_trial_pairs_path is None:
        blocked_reasons.append("per_trial_pairs.csv is missing")

    prepared_path, prepared_df, prepared_blockers = _load_prepared_dataset(run_root, report)
    blocked_reasons.extend(prepared_blockers)

    experiment = report.get("experiment", {}) if report is not None else {}
    artifact_paths = {
        "run_root": str(run_root),
        "summary_report": str(report_path) if report_path is not None else None,
        "reliability_report": str(reliability_path) if reliability_path is not None else None,
        "robustness_report": str(robustness_path) if robustness_path is not None else None,
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
                "trust_signal_report_ready": False,
                "exploratory_only": True,
                "evidence_ready_candidate": False,
                "training_ready": False,
            },
            "interpretation": [
                "the trust-signal analyzer could not run because per_trial_pairs.csv was missing",
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
        blocked_reasons.append("prepared dataset join unavailable for trust analysis")

    worst_subject = robustness.get("per_subject_cosine", {}).get("worst_subject", {}).get("subject") if robustness is not None else None
    worst_session = robustness.get("per_session_cosine", {}).get("worst_session", {}).get("session") if robustness is not None else None

    if "subject" in merged.columns:
        subject_means = _group_mean_series(merged, "subject")
        low_subjects = _low_performing_groups(subject_means)
        merged["subject_mean_cosine"] = merged["subject"].map(subject_means)
    else:
        subject_means = pd.Series(dtype="float64")
        low_subjects = []
    if "session" in merged.columns:
        session_means = _group_mean_series(merged, "session")
        low_sessions = _low_performing_groups(session_means)
        merged["session_mean_cosine"] = merged["session"].map(session_means)
    else:
        session_means = pd.Series(dtype="float64")
        low_sessions = []

    threshold_analyses = {
        label: _threshold_analysis(
            merged,
            label=label,
            fraction=fraction,
            worst_subject=worst_subject,
            worst_session=worst_session,
            low_subjects=low_subjects,
            low_sessions=low_sessions,
        )
        for label, fraction in LOW_TAIL_SPECS
    }
    primary = threshold_analyses[PRIMARY_TAIL_LABEL]

    overall_sample_associations = {
        "sample_cosine_vs_subject_mean_spearman": _safe_spearman(merged["cosine"], merged.get("subject_mean_cosine", pd.Series(dtype="float64"))),
        "sample_cosine_vs_session_mean_spearman": _safe_spearman(merged["cosine"], merged.get("session_mean_cosine", pd.Series(dtype="float64"))),
    }

    notes: list[str] = []
    if join_succeeded:
        notes.append("trust-signal analysis joined per_trial_pairs.csv to the fixed prepared dataset on pair_id")
    else:
        notes.append("trust-signal analysis could not complete the prepared-dataset join and therefore remains partially blocked")
    if primary.get("available"):
        notes.append(
            f"the primary bottom-10% tail threshold is {primary['threshold_cosine_lte']:.6f} across {primary['sample_count']} samples"
        )
        if primary.get("worst_subject_enrichment") is not None and worst_subject is not None:
            notes.append(
                f"bottom-10% samples are {primary['worst_subject_enrichment']:.2f}x enriched in worst subject {worst_subject}"
            )
        if primary.get("worst_session_enrichment") is not None and worst_session is not None:
            notes.append(
                f"bottom-10% samples are {primary['worst_session_enrichment']:.2f}x enriched in worst session {worst_session}"
            )
        if primary.get("tail_instability_enrichment") is not None:
            notes.append(
                f"bottom-10% samples overlap the low-performing subject/session groups at {primary['tail_instability_overlap']['share_of_tail']:.3f}, "
                f"which is {primary['tail_instability_enrichment']:.2f}x the base unstable-group share"
            )
    notes.append("this report is exploratory trust-signal analysis only and does not establish probabilistic calibration")

    ready = bool(
        report is not None
        and reliability is not None
        and robustness is not None
        and per_trial_pairs_path is not None
        and join_succeeded
        and all(payload.get("available") for payload in threshold_analyses.values())
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
        "pair_counts": {
            "pair_count": int(len(merged)),
            "rows_with_subject_metadata": int(merged["subject"].notna().sum()) if "subject" in merged.columns else 0,
            "rows_with_session_metadata": int(merged["session"].notna().sum()) if "session" in merged.columns else 0,
        },
        "prepared_dataset_join": {
            "join_succeeded": join_succeeded,
            "prepared_dataset_rows": int(len(prepared_df)) if prepared_df is not None else None,
            "rows_with_subject_metadata": int(merged["subject"].notna().sum()) if "subject" in merged.columns else 0,
            "rows_with_session_metadata": int(merged["session"].notna().sum()) if "session" in merged.columns else 0,
        },
        "instability_reference": {
            "worst_subject": robustness.get("per_subject_cosine", {}).get("worst_subject") if robustness is not None else None,
            "worst_session": robustness.get("per_session_cosine", {}).get("worst_session") if robustness is not None else None,
            "low_performing_subjects": {
                "rule": "bottom_quartile_by_subject_mean_cosine",
                "selected_groups": low_subjects,
                "group_means": {str(key): float(value) for key, value in subject_means.items()},
            },
            "low_performing_sessions": {
                "rule": "bottom_quartile_by_session_mean_cosine",
                "selected_groups": low_sessions,
                "group_means": {str(key): float(value) for key, value in session_means.items()},
            },
        },
        "comparison_context": {
            "eval_cosine": report.get("eval_summary", {}).get("overall_cosine") if report is not None else None,
            "transfer_cosine": report.get("transfer_summary", {}).get("overall_cosine") if report is not None else None,
            "subject_mean_std": robustness.get("dispersion_indicators", {}).get("subject_mean_std") if robustness is not None else None,
            "session_mean_std": robustness.get("dispersion_indicators", {}).get("session_mean_std") if robustness is not None else None,
            "primary_threshold_label": PRIMARY_TAIL_LABEL,
        },
        "low_score_thresholds": {
            label: {
                "tail_fraction": payload["tail_fraction"],
                "sample_count": payload["sample_count"],
                "threshold_cosine_lte": payload["threshold_cosine_lte"],
            }
            for label, payload in threshold_analyses.items()
            if payload.get("available")
        },
        "threshold_analyses": threshold_analyses,
        "rank_based_instability_associations": {
            "sample_cosine_vs_subject_mean_spearman": overall_sample_associations["sample_cosine_vs_subject_mean_spearman"],
            "sample_cosine_vs_session_mean_spearman": overall_sample_associations["sample_cosine_vs_session_mean_spearman"],
        },
        "descriptive_trust_scores": {
            "primary_threshold_label": PRIMARY_TAIL_LABEL,
            "tail_instability_overlap": primary.get("tail_instability_overlap"),
            "tail_instability_enrichment": primary.get("tail_instability_enrichment"),
            "worst_subject_enrichment": primary.get("worst_subject_enrichment"),
            "worst_session_enrichment": primary.get("worst_session_enrichment"),
            "instability_flag_rate": primary.get("instability_flag_rate"),
            "subject_low_tail_rate_association": primary.get("rank_associations", {}).get("subject_mean_vs_low_tail_rate_spearman"),
            "session_low_tail_rate_association": primary.get("rank_associations", {}).get("session_mean_vs_low_tail_rate_spearman"),
        },
        "blocked_reasons": blocked_reasons,
        "interpretation": notes,
        "state": {
            "trust_signal_report_ready": ready,
            "exploratory_only": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this trust-signal report analyzes existing paper-2 public-NOD artifacts only",
            "low-score enrichment here is exploratory trust support, not calibrated uncertainty evidence",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }
    return json_safe(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exploratory trust-signal behavior for one existing paper-2 public-NOD run."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve() if args.run_root else Path(args.root_dir).resolve() / args.run_id
    output = args.output or str(run_root / "trust_signal_report.json")
    payload = analyze_public_nod_paper2_trust_signals(run_root=run_root, output_path=output)
    write_report(output, payload)
    print(json.dumps(json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
