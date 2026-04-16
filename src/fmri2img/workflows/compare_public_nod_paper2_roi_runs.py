from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.compare_public_nod_paper2_roi_runs")

from fmri2img.workflows.compare_public_nod_paper2_runs import (  # noqa: E402
    _default_path,
    _load_optional_json,
    _metric_value,
    _rank_runs,
    _resolve_reliability_report,
    _resolve_run_report,
)
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only"
DEFAULT_OUTPUT = "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/roi_ablation_comparison.json"
DEFAULT_RELIABILITY_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/roi_reliability_comparison.json"
)
DEFAULT_RUNS = ("baseline", "early_visual_only", "metacognitive_only")


def _normalize_zero_out_groups(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if value and all(isinstance(item, str) and len(item) == 1 for item in value):
            return _normalize_zero_out_groups("".join(value))
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if not inner:
                return []
            return [part.strip().strip('"').strip("'") for part in inner.split(",")]
        return [stripped]
    return [str(value)]


def _interpret_roi(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_eval = _rank_runs(run_rows, "eval_cosine")
    best_transfer = _rank_runs(run_rows, "transfer_cosine")
    by_run = {row["run_id"]: row for row in run_rows}
    threshold_rows = [row for row in run_rows if row.get("low_trust_threshold_cosine_lte") is not None]
    cleanest_tail = max(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"])) if threshold_rows else None

    notes: list[str] = []
    early = by_run.get("early_visual_only")
    meta = by_run.get("metacognitive_only")
    if best_eval:
        top = best_eval[0]["run_id"]
        if top == "early_visual_only":
            notes.append("early_visual_only is the strongest eval-only ROI view in the current controlled pack")
        elif top == "metacognitive_only":
            notes.append("metacognitive_only retains measurable eval signal and is the strongest ROI-only view in the current pack")
        elif top == "baseline":
            notes.append("the full baseline remains stronger than either single-ROI-only view on eval cosine")
    if early and meta and early.get("eval_cosine") is not None and meta.get("eval_cosine") is not None:
        if float(early["eval_cosine"]) > float(meta["eval_cosine"]):
            notes.append("early_visual_only currently outperforms metacognitive_only on eval cosine")
        elif float(meta["eval_cosine"]) > float(early["eval_cosine"]):
            notes.append("metacognitive_only currently outperforms early_visual_only on eval cosine")
        else:
            notes.append("early_visual_only and metacognitive_only are currently tied on eval cosine")
    if cleanest_tail is not None:
        notes.append(
            f"{cleanest_tail['run_id']} has the cleanest current low-trust tail by bottom-decile cosine "
            f"({cleanest_tail['low_trust_threshold_cosine_lte']:.6f})"
        )
    notes.append("this ROI comparison pack improves interpretability for paper 2 but still does not establish publication-grade evidence")
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


def build_public_nod_paper2_roi_run_comparison(
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
        manifest = _load_optional_json(run_root / "export" / "manifest.json")

        experiment_name = report.get("experiment", {}).get("name") if report is not None else None
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
                "low_trust_threshold_cosine_lte": reliability.get("low_trust_candidates", {}).get("low_trust_threshold_cosine_lte")
                if reliability is not None
                else None,
                "zero_out_groups": _normalize_zero_out_groups(manifest.get("roi_spec", {}).get("zero_out_groups"))
                if manifest is not None
                else None,
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

    threshold_rows = [row for row in reliability_rows if row["low_trust_threshold_cosine_lte"] is not None]
    cleanest = max(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"])) if threshold_rows else None
    noisiest = min(threshold_rows, key=lambda row: float(row["low_trust_threshold_cosine_lte"])) if threshold_rows else None

    comparison = {
        "root_dir": str(root),
        "report_path": str(Path(comparison_output_path).resolve()),
        "runs": run_rows,
        "rankings": {
            "eval_cosine_desc": _rank_runs(run_rows, "eval_cosine"),
            "transfer_cosine_desc": _rank_runs(run_rows, "transfer_cosine"),
        },
        "interpretation": _interpret_roi(run_rows),
        "state": {
            "run_count": len(run_rows),
            "all_run_reports_present": all(row["artifacts_present"]["summary_report"] for row in run_rows),
            "all_reliability_reports_present": all(row["artifacts_present"]["reliability_report"] for row in run_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this ROI comparison artifact summarizes the paper-2 public-NOD ROI ablation pack only",
            "ROI-only rankings here are controlled operational comparisons, not evidence-grade claims",
            "training_ready remains false and no publication gate is implied by this artifact alone",
        ],
    }
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
                "this ROI comparison does not establish uncertainty calibration or publication-grade reliability evidence",
            ],
        },
        "state": {
            "all_reliability_reports_present": all(row["low_trust_threshold_cosine_lte"] is not None for row in reliability_rows),
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this ROI reliability comparison is a starter artifact for the paper-2 lane only",
            "low-trust threshold comparisons are heuristic score-tail summaries, not calibration claims",
            "training_ready remains false",
        ],
    }
    return comparison, reliability_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the paper-2 public-NOD baseline and ROI-only ablations into one machine-readable comparison pack."
    )
    parser.add_argument("--root-dir", default=str(_default_path(DEFAULT_ROOT)))
    parser.add_argument("--run-id", action="append", dest="run_ids", default=[])
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    parser.add_argument("--reliability-output", default=str(_default_path(DEFAULT_RELIABILITY_OUTPUT)))
    args = parser.parse_args(argv)

    run_ids = args.run_ids or list(DEFAULT_RUNS)
    comparison, reliability_comparison = build_public_nod_paper2_roi_run_comparison(
        root_dir=args.root_dir,
        run_ids=run_ids,
        comparison_output_path=args.output,
        reliability_output_path=args.reliability_output,
    )
    write_report(args.output, comparison)
    write_report(args.reliability_output, reliability_comparison)
    print(json_safe(comparison))
    print(json_safe(reliability_comparison))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
