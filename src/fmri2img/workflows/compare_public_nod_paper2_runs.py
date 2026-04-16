from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.compare_public_nod_paper2_runs")

from fmri2img.workflows._downstream_contract_audit import load_json  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/shared_capacity_comparison.json"
DEFAULT_RELIABILITY_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/reliability_comparison.json"
)
DEFAULT_RUNS = ("baseline", "shared_dim32", "shared_dim128")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return (_repo_root() / relative).resolve()


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _resolve_run_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in ("paper2_baseline_report.json", "paper2_ablation_report.json"):
        path = run_root / candidate
        if path.exists():
            return path, load_json(path)
    return None, None


def _resolve_reliability_report(run_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = run_root / "reliability_seed_report.json"
    if path.exists():
        return path, load_json(path)
    return None, None


def _metric_value(report: dict[str, Any] | None, section: str, key: str) -> float | None:
    if report is None:
        return None
    value = report.get(section, {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_runs(run_rows: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    ranked = [
        {"run_id": row["run_id"], "experiment_name": row["experiment_name"], metric_key: row.get(metric_key)}
        for row in run_rows
        if row.get(metric_key) is not None
    ]
    return sorted(ranked, key=lambda item: float(item[metric_key]), reverse=True)


def _interpret_capacity(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run = {row["run_id"]: row for row in run_rows}
    baseline = by_run.get("baseline")
    low = by_run.get("shared_dim32")
    high = by_run.get("shared_dim128")
    best_eval = _rank_runs(run_rows, "eval_cosine")
    best_transfer = _rank_runs(run_rows, "transfer_cosine")
    threshold_rows = [row for row in run_rows if row.get("low_trust_threshold_cosine_lte") is not None]
    cleanest_tail = None
    if threshold_rows:
        cleanest_tail = max(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"]))

    notes: list[str] = []
    if baseline and low and high and all(row.get("eval_cosine") is not None for row in (baseline, low, high)):
        if best_eval and best_eval[0]["run_id"] == "shared_dim128":
            notes.append("higher shared capacity helps eval cosine on the fixed public NOD slice relative to baseline and shared_dim32")
        elif best_eval and best_eval[0]["run_id"] == "shared_dim32":
            notes.append("lower shared capacity helps eval cosine on the fixed public NOD slice relative to baseline and shared_dim128")
        elif best_eval and best_eval[0]["run_id"] == "baseline":
            notes.append("the middle shared capacity remains the strongest eval setting among the three tested runs")
    if cleanest_tail is not None:
        notes.append(
            f"{cleanest_tail['run_id']} currently has the cleanest bottom-decile tail by cosine threshold "
            f"({cleanest_tail['low_trust_threshold_cosine_lte']:.6f})"
        )
    notes.append("this comparison pack is still operational-ablation status and is not enough for publication evidence by itself")
    return {
        "best_eval_run": best_eval[0] if best_eval else None,
        "best_transfer_run": best_transfer[0] if best_transfer else None,
        "cleanest_low_trust_tail": {
            "run_id": cleanest_tail["run_id"],
            "low_trust_threshold_cosine_lte": cleanest_tail["low_trust_threshold_cosine_lte"],
        }
        if cleanest_tail is not None
        else None,
        "notes": notes,
        "enough_for_publication_evidence": False,
    }


def build_public_nod_paper2_run_comparison(
    *,
    root_dir: str | Path,
    run_ids: list[str],
    comparison_output_path: str | Path,
    reliability_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root_dir).resolve()
    run_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_root = root / run_id
        report_path, report = _resolve_run_report(run_root)
        reliability_path, reliability = _resolve_reliability_report(run_root)
        eval_metrics = _load_optional_json(run_root / "eval" / "metrics.json")
        transfer_metrics = _load_optional_json(run_root / "transfer" / "transfer_metrics.json")
        train_history = _load_optional_json(run_root / "train" / "train_history.json")

        experiment_name = None
        if report is not None:
            experiment_name = report.get("experiment", {}).get("name")
        run_rows.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "experiment_name": experiment_name,
                "report_path": str(report_path) if report_path is not None else None,
                "reliability_report_path": str(reliability_path) if reliability_path is not None else None,
                "artifacts_present": {
                    "train": (run_root / "train").exists(),
                    "eval": (run_root / "eval").exists(),
                    "transfer": (run_root / "transfer").exists(),
                    "export": (run_root / "export").exists(),
                    "summary_report": report_path is not None,
                    "reliability_report": reliability_path is not None,
                },
                "eval_cosine": _metric_value(report, "eval_summary", "overall_cosine"),
                "eval_mse": _metric_value(report, "eval_summary", "overall_mse"),
                "transfer_cosine": _metric_value(report, "transfer_summary", "overall_cosine"),
                "transfer_mse": _metric_value(report, "transfer_summary", "overall_mse"),
                "epoch_count": len(train_history) if isinstance(train_history, list) else None,
                "target_metadata_consistent": report.get("export_summary", {}).get("target_metadata_consistent")
                if report is not None
                else None,
                "condition_semantics_consistent": report.get("export_summary", {}).get("condition_semantics_consistent")
                if report is not None
                else None,
                "condition_semantics": report.get("export_summary", {}).get("condition_semantics") if report is not None else None,
                "low_trust_threshold_cosine_lte": reliability.get("low_trust_candidates", {}).get("low_trust_threshold_cosine_lte")
                if reliability is not None
                else None,
                "overall_eval_metrics_present": bool(eval_metrics),
                "overall_transfer_metrics_present": bool(transfer_metrics),
                "state": report.get("state", {}) if report is not None else {},
            }
        )
        reliability_rows.append(
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "low_trust_threshold_cosine_lte": reliability.get("low_trust_candidates", {}).get("low_trust_threshold_cosine_lte")
                if reliability is not None
                else None,
                "worst_example_count": len(reliability.get("low_trust_candidates", {}).get("lowest_examples", []))
                if reliability is not None
                else None,
                "lowest_example_min_cosine": (
                    min(float(item["cosine"]) for item in reliability.get("low_trust_candidates", {}).get("lowest_examples", []))
                    if reliability is not None and reliability.get("low_trust_candidates", {}).get("lowest_examples")
                    else None
                ),
            }
        )

    comparison = {
        "root_dir": str(root),
        "report_path": str(Path(comparison_output_path).resolve()),
        "runs": run_rows,
        "rankings": {
            "eval_cosine_desc": _rank_runs(run_rows, "eval_cosine"),
            "transfer_cosine_desc": _rank_runs(run_rows, "transfer_cosine"),
        },
        "interpretation": _interpret_capacity(run_rows),
        "state": {
            "run_count": len(run_rows),
            "all_run_reports_present": all(row["artifacts_present"]["summary_report"] for row in run_rows),
            "all_reliability_reports_present": all(row["artifacts_present"]["reliability_report"] for row in run_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this comparison artifact summarizes the paper-2 public-NOD shared-capacity pack only",
            "rankings here are controlled operational comparisons, not evidence-grade claims",
            "training_ready remains false and no publication gate is implied by this artifact alone",
        ],
    }
    threshold_rows = [row for row in reliability_rows if row["low_trust_threshold_cosine_lte"] is not None]
    cleanest = max(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"])) if threshold_rows else None
    noisiest = min(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"])) if threshold_rows else None
    reliability_comparison = {
        "root_dir": str(root),
        "report_path": str(Path(reliability_output_path).resolve()),
        "runs": reliability_rows,
        "tail_ranking_desc": sorted(
            [row for row in reliability_rows if row["low_trust_threshold_cosine_lte"] is not None],
            key=lambda row: float(row["low_trust_threshold_cosine_lte"]),
            reverse=True,
        ),
        "interpretation": {
            "cleanest_tail_run": cleanest,
            "noisiest_tail_run": noisiest,
            "notes": [
                "higher bottom-decile cosine threshold is interpreted here as a cleaner low-score tail",
                "worst-example summaries are descriptive only and do not constitute calibration",
                "this comparison does not establish uncertainty calibration or publication-grade reliability evidence",
            ],
        },
        "state": {
            "all_reliability_reports_present": all(row["low_trust_threshold_cosine_lte"] is not None for row in threshold_rows)
            if threshold_rows
            else False,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this reliability comparison is a starter artifact for the paper-2 lane only",
            "low-trust threshold comparisons are heuristic score-tail summaries, not calibration claims",
            "training_ready remains false",
        ],
    }
    return comparison, reliability_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the paper-2 public-NOD baseline and shared-capacity ablations into one machine-readable comparison pack."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    parser.add_argument("--reliability-output", default=str(_default_path(DEFAULT_RELIABILITY_OUTPUT)))
    args = parser.parse_args(argv)

    run_ids = args.run_ids or list(DEFAULT_RUNS)
    comparison, reliability_comparison = build_public_nod_paper2_run_comparison(
        root_dir=args.root_dir,
        run_ids=run_ids,
        comparison_output_path=args.output,
        reliability_output_path=args.reliability_output,
    )
    write_report(args.output, comparison)
    write_report(args.reliability_output, reliability_comparison)
    print(comparison["state"])
    print(reliability_comparison["state"])
    print(json_safe(comparison))
    print(json_safe(reliability_comparison))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
