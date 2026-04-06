from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.report_public_nod_shared_only_smoke")

from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_CONFIG = "configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml"
REQUIRED_SMOKE_FILES = (
    "best_decoder.pt",
    "config_snapshot.json",
    "roi_summary.json",
    "target_summary.json",
    "train_history.json",
)


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


def build_public_nod_shared_only_smoke_report(config, *, config_path: str | Path) -> dict[str, Any]:
    validate_canonical_workflow_config(config)

    expected_name = "public_nod_imagenet_run10_shared_only_smoke"
    experiment_name = str(config.get("experiment.name", ""))
    output_dir = Path(config["training"]["output_dir"]).resolve()
    if experiment_name != expected_name:
        raise ValueError("NOD trainer smoke report requires the checked-in fixed-slice smoke config.")
    if output_dir.name != expected_name:
        raise ValueError("NOD trainer smoke report requires the fixed smoke output directory contract.")
    if int(config["training"].get("epochs", 1)) != 1:
        raise ValueError("NOD trainer smoke report requires epochs=1.")
    if int(config["training"].get("batch_size", 0)) != 2880:
        raise ValueError("NOD trainer smoke report requires the fixed 2880-row train-batch smoke contract.")
    if int(config["evaluation"].get("batch_size", 0)) != 360:
        raise ValueError("NOD trainer smoke report requires the fixed 360-row eval-batch smoke contract.")

    prepared_report_path = _resolve_required_path(config, "prepared_report")
    target_cache_report_path = _resolve_required_path(config, "target_cache_report")
    roi_report_path = _resolve_required_path(config, "roi_report")
    join_report_path = _resolve_required_path(config, "join_report")
    trainer_preflight_path = _resolve_required_path(config, "trainer_preflight_report")
    preflight_data_path = _resolve_required_path(config, "preflight_data_report")

    missing_files = [name for name in REQUIRED_SMOKE_FILES if not (output_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "NOD trainer smoke output is incomplete. Missing required smoke artifacts: "
            + ", ".join(missing_files)
        )

    train_history = json.loads((output_dir / "train_history.json").read_text())
    if not isinstance(train_history, list) or not train_history:
        raise ValueError("NOD trainer smoke report requires a non-empty train_history.json.")

    config_snapshot = json.loads((output_dir / "config_snapshot.json").read_text())
    if str(config_snapshot.get("experiment", {}).get("name", "")) != expected_name:
        raise ValueError("Smoke config snapshot does not match the fixed NOD smoke experiment name.")

    prepared_report = json.loads(prepared_report_path.read_text())
    target_cache_report = json.loads(target_cache_report_path.read_text())
    roi_report = json.loads(roi_report_path.read_text())
    join_report = json.loads(join_report_path.read_text())
    trainer_preflight = json.loads(trainer_preflight_path.read_text())
    preflight_data = json.loads(preflight_data_path.read_text())
    last_epoch = train_history[-1]

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
        "smoke_config": {
            "experiment_name": experiment_name,
            "device": str(config["training"].get("device", "cpu")),
            "epochs": int(config["training"].get("epochs", 1)),
            "train_batch_size": int(config["training"].get("batch_size", 0)),
            "eval_batch_size": int(config["evaluation"].get("batch_size", 0)),
            "output_dir": str(output_dir),
        },
        "artifact_paths": {
            name: str((output_dir / name).resolve()) for name in REQUIRED_SMOKE_FILES
        },
        "upstream_state": {
            "join_ready": bool(join_report["state"]["join_ready"]),
            "roi_ready": bool(roi_report["state"]["roi_ready"]),
            "downstream_prep_ready": bool(prepared_report["state"]["downstream_prep_ready"]),
            "trainer_config_ready": bool(trainer_preflight["state"]["trainer_config_ready"]),
            "preflight_ready": bool(trainer_preflight["state"]["preflight_ready"]),
            "canonical_preflight_status": preflight_data.get("status"),
            "target_embedding_ready": bool(target_cache_report["state"]["target_embedding_ready"]),
        },
        "smoke_run": {
            "epochs_completed": len(train_history),
            "last_epoch": {
                "epoch": int(last_epoch["epoch"]),
                "train_loss": float(last_epoch["train_loss"]),
                "val_loss": float(last_epoch["val_loss"]),
                "val_content_cosine": float(last_epoch["val_content_cosine"]),
            },
        },
        "state": {
            "join_ready": bool(join_report["state"]["join_ready"]),
            "roi_ready": bool(roi_report["state"]["roi_ready"]),
            "downstream_prep_ready": bool(prepared_report["state"]["downstream_prep_ready"]),
            "trainer_config_ready": bool(trainer_preflight["state"]["trainer_config_ready"]),
            "preflight_ready": bool(trainer_preflight["state"]["preflight_ready"]),
            "smoke_ready": True,
            "training_ready": False,
        },
        "operational_boundary": [
            "this run only validates trainer startup, one bounded epoch, and canonical artifact creation on the fixed NOD slice",
            "loss values and checkpoint outputs from this smoke are operational only and are not benchmark evidence",
            "full training, eval, and evidence-facing interpretation remain out of scope",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed NOD shared-only smoke run without promoting it to training readiness."
    )
    parser.add_argument("--config", default=str(_default_path(DEFAULT_CONFIG)))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_workflow_config(args.config, args.override)
        report = build_public_nod_shared_only_smoke_report(config, config_path=args.config)
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
                "training_ready": False,
            },
            "blocked_reasons": [str(exc)],
        }
        output_path = args.output or "outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json"
        write_report(output_path, report)
        print(json.dumps(json_safe(report), indent=2))
        if args.fail_on_blocked:
            return 1
        return 0

    output_path = args.output or (Path(config["training"]["output_dir"]).resolve() / "smoke_report.json")
    write_report(output_path, report)
    print(json.dumps(json_safe(report), indent=2))
    print(f"Smoke ready: {report['state']['smoke_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
