from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.analyze_public_nod_paper2_robustness")

from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _resolve_reliability_report,
    _resolve_run_report,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_RUN = "baseline"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/robustness_report.json"
LOW_TRUST_FRACTION = 0.1
SUBJECT_CONCENTRATION_MULTIPLIER = 2.0
SESSION_CONCENTRATION_MULTIPLIER = 1.6


def _resolve_json(run_root: Path, *relative_candidates: str) -> tuple[Path | None, dict[str, Any] | None]:
    for relative in relative_candidates:
        path = run_root / relative
        if path.exists():
            return path, _load_optional_json(path)
    return None, None


def _resolve_csv(run_root: Path, *relative_candidates: str) -> Path | None:
    for relative in relative_candidates:
        path = run_root / relative
        if path.exists():
            return path
    return None


def _coerce_pair_ids(pairs_df: pd.DataFrame, prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pairs_df.copy()
    prepared = prepared_df.copy()
    if "pair_id" in pairs.columns:
        pairs["pair_id"] = pd.to_numeric(pairs["pair_id"], errors="coerce").astype("Int64")
    if "pair_id" in prepared.columns:
        prepared["pair_id"] = pd.to_numeric(prepared["pair_id"], errors="coerce").astype("Int64")
    return pairs, prepared


def _load_prepared_dataset(run_root: Path, report: dict[str, Any] | None) -> tuple[Path | None, pd.DataFrame | None, list[str]]:
    blocked_reasons: list[str] = []
    prepared_path: Path | None = None
    if report is not None:
        candidate = report.get("artifact_paths", {}).get("prepared_dataset")
        if candidate:
            prepared_path = Path(candidate).resolve()
    if prepared_path is None:
        for relative in ("train/config_snapshot.json", "export/config_snapshot.json", "config_snapshot.json"):
            snapshot_path = run_root / relative
            if not snapshot_path.exists():
                continue
            snapshot = _load_optional_json(snapshot_path)
            if snapshot is None:
                continue
            candidate = snapshot.get("dataset", {}).get("mixed_index")
            if candidate:
                prepared_path = Path(candidate).resolve()
                break
    if prepared_path is None:
        blocked_reasons.append("prepared dataset path could not be recovered from the run report or config snapshot")
        return None, None, blocked_reasons
    if not prepared_path.exists():
        blocked_reasons.append(f"prepared dataset missing: {prepared_path}")
        return prepared_path, None, blocked_reasons
    prepared_df = pd.read_parquet(prepared_path)
    return prepared_path, prepared_df, blocked_reasons


def _dispersion_summary(group_means: pd.Series) -> dict[str, Any] | None:
    if group_means.empty:
        return None
    return {
        "group_count": int(len(group_means)),
        "mean_of_group_means": float(group_means.mean()),
        "std_of_group_means": float(group_means.std(ddof=0)),
        "min_group_mean": float(group_means.min()),
        "max_group_mean": float(group_means.max()),
        "range_of_group_means": float(group_means.max() - group_means.min()),
    }


def _group_summary(merged: pd.DataFrame, group_key: str, low_trust_mask: pd.Series | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if group_key not in merged.columns:
        return None, None
    rows: dict[str, Any] = {}
    overall_count = int(len(merged))
    low_trust_total = int(low_trust_mask.sum()) if low_trust_mask is not None else 0
    for group_value, frame in merged.groupby(group_key, dropna=False):
        key = str(group_value) if pd.notna(group_value) else "unknown"
        cosine = pd.to_numeric(frame["cosine"], errors="coerce")
        low_trust_count = int(low_trust_mask.loc[frame.index].sum()) if low_trust_mask is not None else None
        rows[key] = {
            "count": int(len(frame)),
            "share_of_samples": float(len(frame) / overall_count) if overall_count else None,
            "cosine_mean": float(cosine.mean()) if not cosine.empty else None,
            "cosine_std": float(cosine.std(ddof=0)) if len(cosine) > 0 else None,
            "low_trust_count": low_trust_count,
            "low_trust_share_within_group": float(low_trust_count / len(frame)) if low_trust_count is not None and len(frame) else None,
            "low_trust_share_of_bucket": (
                float(low_trust_count / low_trust_total) if low_trust_count is not None and low_trust_total else None
            ),
        }
    group_means = (
        pd.Series({key: payload["cosine_mean"] for key, payload in rows.items() if payload["cosine_mean"] is not None}, dtype="float64")
        if rows
        else pd.Series(dtype="float64")
    )
    return rows, _dispersion_summary(group_means)


def _low_trust_bucket(pairs_df: pd.DataFrame, reliability: dict[str, Any] | None) -> tuple[pd.Series | None, dict[str, Any]]:
    if "cosine" not in pairs_df.columns or pairs_df.empty:
        return None, {
            "available": False,
            "reason": "per_trial_pairs.csv does not expose cosine values",
        }
    cosine = pd.to_numeric(pairs_df["cosine"], errors="coerce")
    threshold = None
    if reliability is not None:
        threshold = reliability.get("low_trust_candidates", {}).get("low_trust_threshold_cosine_lte")
    if threshold is None:
        ranked = cosine.dropna().sort_values()
        if ranked.empty:
            return None, {
                "available": False,
                "reason": "cosine column is present but contains no numeric values",
            }
        bucket_size = max(1, int(math.ceil(len(ranked) * LOW_TRUST_FRACTION)))
        threshold = float(ranked.iloc[bucket_size - 1])
    mask = cosine <= float(threshold)
    return mask.fillna(False), {
        "available": True,
        "heuristic": "bottom_decile_by_cosine",
        "low_trust_threshold_cosine_lte": float(threshold),
        "low_trust_count": int(mask.sum()),
    }


def _top_concentration(group_rows: dict[str, Any] | None, *, uniform_fraction: float | None, multiplier: float) -> dict[str, Any] | None:
    if not group_rows:
        return None
    ranked = sorted(
        (
            {
                "group": group,
                "low_trust_count": payload.get("low_trust_count"),
                "low_trust_share_of_bucket": payload.get("low_trust_share_of_bucket"),
                "low_trust_share_within_group": payload.get("low_trust_share_within_group"),
                "count": payload.get("count"),
                "cosine_mean": payload.get("cosine_mean"),
            }
            for group, payload in group_rows.items()
            if payload.get("low_trust_share_of_bucket") is not None
        ),
        key=lambda item: float(item["low_trust_share_of_bucket"]),
        reverse=True,
    )
    if not ranked:
        return None
    top = ranked[0]
    threshold = uniform_fraction * multiplier if uniform_fraction is not None else None
    top["concentration_flag"] = bool(
        threshold is not None and top["low_trust_share_of_bucket"] is not None and top["low_trust_share_of_bucket"] > threshold
    )
    top["uniform_share_reference"] = uniform_fraction
    top["concentration_flag_threshold"] = threshold
    return top


def analyze_public_nod_paper2_robustness(*, run_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    output_path = Path(output_path).resolve()
    report_path, report = _resolve_run_report(run_root)
    reliability_path, reliability = _resolve_reliability_report(run_root)
    transfer_metrics_path, transfer_metrics = _resolve_json(run_root, "transfer/transfer_metrics.json", "transfer_metrics.json")
    eval_metrics_path, eval_metrics = _resolve_json(run_root, "eval/metrics.json", "metrics.json")
    per_trial_pairs_path = _resolve_csv(run_root, "transfer/per_trial_pairs.csv", "per_trial_pairs.csv")

    blocked_reasons: list[str] = []
    if report is None:
        blocked_reasons.append("run summary report is missing")
    if per_trial_pairs_path is None:
        blocked_reasons.append("per_trial_pairs.csv is missing")

    prepared_path, prepared_df, prepared_blockers = _load_prepared_dataset(run_root, report)
    blocked_reasons.extend(prepared_blockers)

    experiment = report.get("experiment", {}) if report is not None else {}
    artifact_paths = {
        "run_root": str(run_root),
        "summary_report": str(report_path) if report_path is not None else None,
        "reliability_report": str(reliability_path) if reliability_path is not None else None,
        "prepared_dataset": str(prepared_path) if prepared_path is not None else None,
        "per_trial_pairs": str(per_trial_pairs_path) if per_trial_pairs_path is not None else None,
        "eval_metrics": str(eval_metrics_path) if eval_metrics_path is not None else None,
        "transfer_metrics": str(transfer_metrics_path) if transfer_metrics_path is not None else None,
    }

    if per_trial_pairs_path is None:
        payload = {
            "report_path": str(output_path),
            "run_identity": {
                "run_id": run_root.name,
                "experiment_name": experiment.get("name"),
                "benchmark_role": experiment.get("benchmark_role"),
                "evidence_tier": experiment.get("evidence_tier"),
            },
            "artifact_paths_used": artifact_paths,
            "prepared_dataset_join": {
                "join_succeeded": False,
                "rows_with_metadata": 0,
                "rows_missing_metadata": None,
            },
            "blocked_reasons": blocked_reasons,
            "state": {
                "robustness_report_ready": False,
                "evidence_ready_candidate": False,
                "training_ready": False,
            },
            "interpretation": [
                "the robustness analyzer could not run because per_trial_pairs.csv was missing",
                "evidence_ready_candidate remains false",
            ],
        }
        return payload

    pairs_df = pd.read_csv(per_trial_pairs_path)
    pairs_df["cosine"] = pd.to_numeric(pairs_df.get("cosine"), errors="coerce")
    if prepared_df is not None and "pair_id" in pairs_df.columns and "pair_id" in prepared_df.columns:
        pairs_df, prepared_df = _coerce_pair_ids(pairs_df, prepared_df)
        join_cols = [col for col in ("pair_id", "subject", "session", "split", "condition", "nsd_id") if col in prepared_df.columns]
        merged = pairs_df.merge(prepared_df[join_cols].drop_duplicates(), on="pair_id", how="left")
        join_succeeded = True
    else:
        merged = pairs_df.copy()
        join_succeeded = False
        if prepared_df is None:
            blocked_reasons.append("prepared dataset join unavailable")
        else:
            blocked_reasons.append("prepared dataset is present but lacks pair_id join compatibility")

    low_trust_mask, low_trust_summary = _low_trust_bucket(merged, reliability)
    subject_rows, subject_dispersion = _group_summary(merged, "subject", low_trust_mask)
    session_rows, session_dispersion = _group_summary(merged, "session", low_trust_mask)

    worst_subject = None
    if subject_rows:
        worst_subject_key = min(
            (key for key, payload in subject_rows.items() if payload.get("cosine_mean") is not None),
            key=lambda key: float(subject_rows[key]["cosine_mean"]),
            default=None,
        )
        if worst_subject_key is not None:
            worst_subject = {"subject": worst_subject_key, **subject_rows[worst_subject_key]}

    worst_session = None
    if session_rows:
        worst_session_key = min(
            (key for key, payload in session_rows.items() if payload.get("cosine_mean") is not None),
            key=lambda key: float(session_rows[key]["cosine_mean"]),
            default=None,
        )
        if worst_session_key is not None:
            worst_session = {"session": worst_session_key, **session_rows[worst_session_key]}

    subject_concentration = _top_concentration(
        subject_rows,
        uniform_fraction=(1.0 / len(subject_rows)) if subject_rows else None,
        multiplier=SUBJECT_CONCENTRATION_MULTIPLIER,
    )
    session_concentration = _top_concentration(
        session_rows,
        uniform_fraction=(1.0 / len(session_rows)) if session_rows else None,
        multiplier=SESSION_CONCENTRATION_MULTIPLIER,
    )

    instability_flags = {
        "subject_dispersion_high": bool(
            subject_dispersion is not None and subject_dispersion.get("range_of_group_means", 0.0) > 0.02
        ),
        "session_dispersion_high": bool(
            session_dispersion is not None and session_dispersion.get("range_of_group_means", 0.0) > 0.01
        ),
        "low_trust_subject_concentrated": bool(subject_concentration and subject_concentration.get("concentration_flag")),
        "low_trust_session_concentrated": bool(session_concentration and session_concentration.get("concentration_flag")),
    }

    notes: list[str] = []
    if join_succeeded:
        notes.append("subject/session robustness was recovered by joining per_trial_pairs.csv to the fixed prepared dataset on pair_id")
    else:
        notes.append("subject/session robustness could not be fully recovered because the prepared-dataset join did not succeed")
    if worst_subject is not None:
        notes.append(
            f"worst subject by transfer cosine is {worst_subject['subject']} at {worst_subject['cosine_mean']:.6f}"
        )
    if worst_session is not None:
        notes.append(
            f"worst session by transfer cosine is {worst_session['session']} at {worst_session['cosine_mean']:.6f}"
        )
    if subject_concentration and subject_concentration.get("concentration_flag"):
        notes.append(
            f"low-trust examples are concentrated in subject {subject_concentration['group']} "
            f"({subject_concentration['low_trust_share_of_bucket']:.3f} of the bottom-decile bucket)"
        )
    else:
        notes.append("low-trust examples do not show a severe subject-level concentration flag under the current heuristic")
    if session_concentration and session_concentration.get("concentration_flag"):
        notes.append(
            f"low-trust examples are concentrated in session {session_concentration['group']} "
            f"({session_concentration['low_trust_share_of_bucket']:.3f} of the bottom-decile bucket)"
        )
    else:
        notes.append("low-trust examples do not show a severe session-level concentration flag under the current heuristic")
    notes.append("this robustness report is descriptive support for paper 2, not calibration proof or a publication gate")

    payload = {
        "report_path": str(output_path),
        "run_identity": {
            "run_id": run_root.name,
            "experiment_name": experiment.get("name"),
            "benchmark_role": experiment.get("benchmark_role"),
            "evidence_tier": experiment.get("evidence_tier"),
        },
        "artifact_paths_used": artifact_paths,
        "overall_metrics": {
            "eval_cosine": report.get("eval_summary", {}).get("overall_cosine") if report is not None else None,
            "eval_mse": report.get("eval_summary", {}).get("overall_mse") if report is not None else None,
            "transfer_cosine": report.get("transfer_summary", {}).get("overall_cosine") if report is not None else None,
            "transfer_mse": report.get("transfer_summary", {}).get("overall_mse") if report is not None else None,
            "low_trust_threshold_cosine_lte": low_trust_summary.get("low_trust_threshold_cosine_lte"),
        },
        "prepared_dataset_join": {
            "join_succeeded": join_succeeded,
            "prepared_dataset_rows": int(len(prepared_df)) if prepared_df is not None else None,
            "rows_with_metadata": int(
                merged["subject"].notna().sum()
            )
            if "subject" in merged.columns
            else 0,
            "rows_missing_metadata": int(
                merged["subject"].isna().sum()
            )
            if "subject" in merged.columns
            else int(len(merged)),
        },
        "sample_counts": {
            "pair_count": int(len(merged)),
            "per_subject_counts": {key: int(value["count"]) for key, value in (subject_rows or {}).items()},
            "per_session_counts": {key: int(value["count"]) for key, value in (session_rows or {}).items()},
        },
        "per_subject_cosine": {
            "available": subject_rows is not None,
            "groups": subject_rows,
            "dispersion": subject_dispersion,
            "worst_subject": worst_subject,
        },
        "per_session_cosine": {
            "available": session_rows is not None,
            "groups": session_rows,
            "dispersion": session_dispersion,
            "worst_session": worst_session,
        },
        "low_trust_concentration": {
            "available": bool(low_trust_summary.get("available")),
            "summary": low_trust_summary,
            "top_subject": subject_concentration,
            "top_session": session_concentration,
        },
        "dispersion_indicators": {
            "subject_mean_range": subject_dispersion.get("range_of_group_means") if subject_dispersion is not None else None,
            "subject_mean_std": subject_dispersion.get("std_of_group_means") if subject_dispersion is not None else None,
            "session_mean_range": session_dispersion.get("range_of_group_means") if session_dispersion is not None else None,
            "session_mean_std": session_dispersion.get("std_of_group_means") if session_dispersion is not None else None,
        },
        "instability_flags": instability_flags,
        "blocked_reasons": blocked_reasons,
        "interpretation": notes,
        "state": {
            "robustness_report_ready": join_succeeded and bool(subject_rows) and bool(session_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this robustness report analyzes existing paper-2 public-NOD run artifacts only",
            "subject/session breakdowns here are descriptive stability summaries, not publication-grade proof",
            "evidence_ready_candidate remains false and training_ready remains false",
        ],
    }
    return json_safe(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze subject-level and session-level robustness for one existing paper-2 public-NOD run."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve() if args.run_root else Path(args.root_dir).resolve() / args.run_id
    output = args.output or str(run_root / "robustness_report.json")
    payload = analyze_public_nod_paper2_robustness(run_root=run_root, output_path=output)
    write_report(output, payload)
    print(json.dumps(json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
