from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke")

from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml"
EVAL_FILES = ("metrics.json", "roi_summary.json", "resolved_roi_groups.json")
EXPORT_FILES = ("best_decoder.pt", "config_snapshot.json", "manifest.json", "decoder_card.json", "decoder_card.md")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return _repo_root() / relative


def _resolve_required_path(config, key: str) -> Path:
    value = config.get(f"public_nod.{key}")
    if value is None:
        raise KeyError(f"Missing public_nod.{key} in config.")
    path = _default_path(str(value)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required fixed NOD artifact is missing: public_nod.{key}={path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canonical_preflight_status(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if status is None and isinstance(payload.get("readiness"), dict):
        status = payload["readiness"].get("status")
    return status


def build_public_nod_shared_only_eval_export_smoke_report(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)

    expected_name = "public_nod_imagenet_run10_shared_only_smoke"
    if str(config.get("experiment.name", "")) != expected_name:
        raise ValueError("NOD eval/export smoke report requires the checked-in fixed-slice smoke config.")

    train_output_dir = Path(config["training"]["output_dir"]).resolve()
    eval_output_dir = Path(config["evaluation"]["output_dir"]).resolve()
    export_output_dir = Path(config["export"]["output_dir"]).resolve()
    checkpoint_path = train_output_dir / "best_decoder.pt"
    if train_output_dir.name != "imagenet_run10_shared_only_smoke":
        raise ValueError("NOD eval/export smoke report requires the fixed smoke training output directory.")
    if eval_output_dir.name != "imagenet_run10_shared_only_smoke":
        raise ValueError("NOD eval/export smoke report requires the fixed smoke evaluation output directory.")
    if export_output_dir.name != "imagenet_run10_shared_only_smoke":
        raise ValueError("NOD eval/export smoke report requires the fixed smoke export output directory.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected smoke checkpoint is missing: {checkpoint_path}")

    prepared_report_path = _resolve_required_path(config, "prepared_report")
    target_cache_report_path = _resolve_required_path(config, "target_cache_report")
    roi_report_path = _resolve_required_path(config, "roi_report")
    join_report_path = _resolve_required_path(config, "join_report")
    trainer_preflight_path = _resolve_required_path(config, "trainer_preflight_report")
    preflight_data_path = _resolve_required_path(config, "preflight_data_report")
    smoke_report_path = _resolve_required_path(config, "smoke_report")

    missing_eval_files = [name for name in EVAL_FILES if not (eval_output_dir / name).exists()]
    missing_export_files = [name for name in EXPORT_FILES if not (export_output_dir / name).exists()]

    prepared_report = _load_json(prepared_report_path)
    target_cache_report = _load_json(target_cache_report_path)
    roi_report = _load_json(roi_report_path)
    join_report = _load_json(join_report_path)
    trainer_preflight = _load_json(trainer_preflight_path)
    preflight_data = _load_json(preflight_data_path)
    smoke_report = _load_json(smoke_report_path)
    if not bool(smoke_report.get("state", {}).get("smoke_ready")):
        raise ValueError("NOD eval/export smoke requires a completed fixed-slice trainer smoke run first.")

    blocked_reasons: list[str] = []
    eval_smoke_ready = not missing_eval_files
    export_smoke_ready = not missing_export_files
    if missing_eval_files:
        blocked_reasons.append(
            "canonical eval smoke did not produce the required evaluation artifacts: " + ", ".join(missing_eval_files)
        )
    if missing_export_files:
        blocked_reasons.append(
            "canonical export smoke did not produce the required export artifacts: " + ", ".join(missing_export_files)
        )

    eval_metrics = _load_json(eval_output_dir / "metrics.json") if eval_smoke_ready else {}
    export_manifest = _load_json(export_output_dir / "manifest.json") if export_smoke_ready else {}
    export_card = _load_json(export_output_dir / "decoder_card.json") if export_smoke_ready else {}

    report = {
        "config": str(Path(config_path).resolve()),
        "fixed_contract": {
            "dataset_id": config.get("public_nod.dataset_id"),
            "task": config.get("public_nod.task"),
            "subjects": list(config.get("public_nod.subjects", [])),
            "sessions": list(config.get("public_nod.sessions", [])),
            "run": int(config.get("public_nod.run", 10)),
            "adapter_rows": int(config.get("public_nod.adapter_rows", 36)),
            "pair_rows": int(config.get("public_nod.pair_rows", 3600)),
        },
        "artifact_paths": {
            "smoke_checkpoint": str(checkpoint_path),
            "smoke_report": str(smoke_report_path),
            "eval_dir": str(eval_output_dir),
            "export_dir": str(export_output_dir),
            "eval_metrics": str((eval_output_dir / "metrics.json").resolve()),
            "eval_roi_summary": str((eval_output_dir / "roi_summary.json").resolve()),
            "eval_resolved_roi_groups": str((eval_output_dir / "resolved_roi_groups.json").resolve()),
            "export_manifest": str((export_output_dir / "manifest.json").resolve()),
            "export_decoder_card": str((export_output_dir / "decoder_card.json").resolve()),
        },
        "upstream_state": {
            "join_ready": bool(join_report["state"]["join_ready"]),
            "roi_ready": bool(roi_report["state"]["roi_ready"]),
            "downstream_prep_ready": bool(prepared_report["state"]["downstream_prep_ready"]),
            "trainer_config_ready": bool(trainer_preflight["state"]["trainer_config_ready"]),
            "preflight_ready": bool(trainer_preflight["state"]["preflight_ready"]),
            "smoke_ready": bool(smoke_report["state"]["smoke_ready"]),
            "canonical_preflight_status": _canonical_preflight_status(preflight_data),
            "target_embedding_ready": bool(target_cache_report["state"]["target_embedding_ready"]),
        },
        "eval_smoke": {
            "artifacts_present": eval_smoke_ready,
            "target_space": eval_metrics.get("target_space"),
            "pair_metrics": eval_metrics.get("pair_metrics"),
            "by_condition_count": len(eval_metrics.get("by_condition", [])),
            "missing_files": missing_eval_files,
        },
        "export_smoke": {
            "artifacts_present": export_smoke_ready,
            "manifest_target_name": export_manifest.get("target_spec", {}).get("name"),
            "manifest_target_dim": export_manifest.get("target_spec", {}).get("dimension"),
            "decoder_card_experiment_name": export_card.get("experiment", {}).get("name"),
            "decoder_card_benchmark_role": export_card.get("experiment", {}).get("benchmark_role"),
            "missing_files": missing_export_files,
        },
        "state": {
            "join_ready": bool(join_report["state"]["join_ready"]),
            "roi_ready": bool(roi_report["state"]["roi_ready"]),
            "downstream_prep_ready": bool(prepared_report["state"]["downstream_prep_ready"]),
            "trainer_config_ready": bool(trainer_preflight["state"]["trainer_config_ready"]),
            "preflight_ready": bool(trainer_preflight["state"]["preflight_ready"]),
            "smoke_ready": bool(smoke_report["state"]["smoke_ready"]),
            "eval_smoke_ready": eval_smoke_ready,
            "export_smoke_ready": export_smoke_ready,
            "training_ready": False,
        },
        "blocked_reasons": blocked_reasons,
        "operational_boundary": [
            "this report only verifies that the canonical eval/export entrypoints can consume the fixed NOD smoke checkpoint and write smoke artifacts",
            "evaluation metrics in this smoke are operational outputs only and are not benchmark evidence",
            "training readiness and evidence-facing interpretation remain unchanged",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-slice eval/export smoke outputs for the NOD shared-only smoke checkpoint."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_public_nod_shared_only_eval_export_smoke_report(config, config_path=args.config)
    except Exception as exc:
        report = {
            "config": str(Path(args.config).resolve()),
            "state": {
                "join_ready": False,
                "roi_ready": False,
                "downstream_prep_ready": False,
                "trainer_config_ready": False,
                "preflight_ready": False,
                "smoke_ready": False,
                "eval_smoke_ready": False,
                "export_smoke_ready": False,
                "training_ready": False,
            },
            "blocked_reasons": [str(exc)],
        }
        output_path = args.output or "outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["evaluation"]["output_dir"]).resolve() / "eval_export_smoke_report.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    print(f"Eval smoke ready: {report['state']['eval_smoke_ready']}")
    print(f"Export smoke ready: {report['state']['export_smoke_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
