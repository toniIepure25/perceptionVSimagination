from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fmri2img.workflows._venv_guard import ensure_project_venv

ensure_project_venv("fmri2img.workflows.diagnose_public_nod_paper2_condition_semantics")

from fmri2img.workflows._downstream_contract_audit import load_json, surface_condition_semantics  # noqa: E402
from fmri2img.workflows.prep_common import json_safe, write_report  # noqa: E402


DEFAULT_RUN_ROOT = "outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim128"
DEFAULT_OUTPUT = (
    "outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim128/condition_semantics_diagnosis.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_path(relative: str) -> Path:
    return (_repo_root() / relative).resolve()


def _relative_to_run(run_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    return load_json(path.resolve())


def _normalized_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(Path(value).resolve())


def diagnose_public_nod_paper2_condition_semantics(
    *,
    run_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    eval_path = root / "eval" / "metrics.json"
    transfer_path = root / "transfer" / "transfer_metrics.json"
    manifest_path = root / "export" / "manifest.json"
    decoder_card_path = root / "export" / "decoder_card.json"
    config_snapshot_path = root / "export" / "config_snapshot.json"
    report_path = root / "paper2_ablation_report.json"

    eval_metrics = _read_json(eval_path)
    transfer_metrics = _read_json(transfer_path)
    manifest = _read_json(manifest_path)
    decoder_card = _read_json(decoder_card_path)
    config_snapshot = _read_json(config_snapshot_path)
    existing_report = _read_json(report_path) if report_path.exists() else None

    eval_condition = surface_condition_semantics(eval_metrics)
    transfer_condition = surface_condition_semantics(transfer_metrics)
    manifest_raw = manifest.get("metadata", {}).get("condition_semantics")
    manifest_condition = surface_condition_semantics(manifest_raw)
    decoder_condition = surface_condition_semantics(decoder_card.get("condition_semantics"))

    mismatches: list[dict[str, Any]] = []
    if eval_condition != transfer_condition:
        mismatches.append(
            {
                "surface_pair": ["eval.metrics", "transfer.transfer_metrics"],
                "left": eval_condition,
                "right": transfer_condition,
                "reason": "eval and transfer condition semantics disagree",
            }
        )
    if manifest_condition != eval_condition:
        mismatches.append(
            {
                "surface_pair": ["export.manifest.metadata.condition_semantics", "eval.metrics"],
                "left": manifest_condition,
                "right": eval_condition,
                "reason": "export manifest condition semantics do not match eval metrics",
            }
        )
    if decoder_condition != eval_condition:
        mismatches.append(
            {
                "surface_pair": ["export.decoder_card.condition_semantics", "eval.metrics"],
                "left": decoder_condition,
                "right": eval_condition,
                "reason": "decoder card condition semantics do not match eval metrics",
            }
        )

    config_eval_dir = config_snapshot.get("evaluation", {}).get("output_dir")
    config_transfer_dir = config_snapshot.get("evaluation", {}).get("transfer_output_dir")
    config_export_dir = config_snapshot.get("export", {}).get("output_dir")
    expected_eval_dir = str((root / "eval").resolve())
    expected_transfer_dir = str((root / "transfer").resolve())
    expected_export_dir = str((root / "export").resolve())
    config_eval_dir_norm = _normalized_path(config_eval_dir)
    config_transfer_dir_norm = _normalized_path(config_transfer_dir)
    config_export_dir_norm = _normalized_path(config_export_dir)
    config_paths_match_run_root = (
        config_eval_dir_norm == expected_eval_dir
        and config_transfer_dir_norm == expected_transfer_dir
        and config_export_dir_norm == expected_export_dir
    )

    root_cause = "condition_semantics_consistent"
    rerun_needed = False
    code_fix_needed = False
    archived_bundle_should_remain_inconsistent = False
    notes: list[str] = []

    if not mismatches:
        notes.append("eval, transfer, export manifest, and decoder card all agree on condition semantics")
    elif eval_condition == transfer_condition and config_paths_match_run_root:
        if manifest_raw is None:
            root_cause = "malformed_export_bundle_missing_condition_semantics"
            rerun_needed = True
            archived_bundle_should_remain_inconsistent = True
            notes.append(
                "eval and transfer agree, but the export manifest omitted metadata.condition_semantics and the decoder card no longer reflects the eval/transfer semantics"
            )
            notes.append(
                "the export config snapshot points at the correct run-local eval/transfer roots, so the mismatch is isolated to the written export bundle"
            )
        elif manifest_condition == eval_condition and decoder_condition != manifest_condition:
            root_cause = "decoder_card_export_logic_bug"
            rerun_needed = True
            code_fix_needed = True
            archived_bundle_should_remain_inconsistent = True
            notes.append("the manifest carries the correct semantics but the decoder card drops or rewrites them")
        else:
            root_cause = "export_surface_condition_semantics_mismatch"
            rerun_needed = True
            archived_bundle_should_remain_inconsistent = True
            notes.append("eval and transfer agree, but the export surfaces drift from that shared semantics")
    elif not config_paths_match_run_root:
        root_cause = "config_override_mismatch"
        archived_bundle_should_remain_inconsistent = True
        notes.append("the export config snapshot points to different eval/transfer/export roots than the inspected run root")
    else:
        root_cause = "upstream_eval_transfer_condition_mismatch"
        archived_bundle_should_remain_inconsistent = True
        notes.append("the inconsistency begins before export because eval and transfer themselves disagree")

    diagnosis = {
        "run_root": str(root),
        "report_path": str(Path(output_path).resolve()),
        "inspected_files": [
            _relative_to_run(root, eval_path),
            _relative_to_run(root, transfer_path),
            _relative_to_run(root, manifest_path),
            _relative_to_run(root, decoder_card_path),
            _relative_to_run(root, config_snapshot_path),
        ],
        "experiment_name": config_snapshot.get("experiment", {}).get("name"),
        "normalized_condition_semantics": {
            "eval": eval_condition,
            "transfer": transfer_condition,
            "manifest": manifest_condition,
            "decoder_card": decoder_condition,
        },
        "raw_field_presence": {
            "manifest_metadata_condition_semantics_present": manifest_raw is not None,
            "decoder_card_condition_semantics_present": "condition_semantics" in decoder_card,
        },
        "mismatching_fields": mismatches,
        "config_snapshot_alignment": {
            "config_paths_match_run_root": config_paths_match_run_root,
            "evaluation_output_dir": config_eval_dir_norm,
            "transfer_output_dir": config_transfer_dir_norm,
            "export_output_dir": config_export_dir_norm,
        },
        "existing_report_state": existing_report.get("state") if isinstance(existing_report, dict) else None,
        "root_cause_classification": root_cause,
        "rerun_needed": rerun_needed,
        "code_fix_needed": code_fix_needed,
        "existing_bundle_should_remain_archived_but_marked_inconsistent": archived_bundle_should_remain_inconsistent,
        "notes": notes,
        "state": {
            "diagnosis_ready": True,
            "evidence_ready_candidate": False,
            "training_ready": False,
        },
        "operational_boundary": [
            "this diagnosis artifact only audits the paper-2 public-NOD run-local condition-semantics contract",
            "it does not promote the run to evidence-grade status",
            "training_ready remains false",
        ],
    }
    return diagnosis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose paper-2 public-NOD condition-semantics mismatches across eval, transfer, and export surfaces."
    )
    parser.add_argument("--run-root", default=str(_default_path(DEFAULT_RUN_ROOT)))
    parser.add_argument("--output", default=str(_default_path(DEFAULT_OUTPUT)))
    args = parser.parse_args(argv)

    diagnosis = diagnose_public_nod_paper2_condition_semantics(run_root=args.run_root, output_path=args.output)
    write_report(args.output, diagnosis)
    print(json_safe(diagnosis))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
