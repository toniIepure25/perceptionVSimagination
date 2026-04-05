from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.materialize_public_nod_payloads import (
    DEFAULT_OPENNEURO_S3_BASE,
    _annex_target_size_bytes,
    _download_to_path,
    _symlink_target_path,
)
from fmri2img.workflows.prepare_public_nod_target_embedding_cache import DEFAULT_OUTPUT as DEFAULT_MANIFEST


_ANNEX_SIZE_RE = re.compile(r"SHA256E-s(\d+)--")


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_MANIFEST


def _default_report_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "cache/indices/public_nod/imagenet_run10_target_embedding_retrieval_report.json"
    )


def build_public_nod_stimulus_retrieval_plan(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = manifest_path.resolve()
    manifest = pd.read_parquet(manifest_path).sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)
    repo_root = manifest_path.parents[3]

    total_estimated_bytes = 0
    rows: list[dict] = []
    for row in manifest.to_dict(orient="records"):
        stimulus_path = repo_root / row["stimulus_path"]
        estimated_bytes = _annex_target_size_bytes(stimulus_path)
        if estimated_bytes is not None:
            total_estimated_bytes += estimated_bytes
        rows.append(
            {
                **row,
                "estimated_bytes": estimated_bytes,
                "download_url": f"{DEFAULT_OPENNEURO_S3_BASE.rstrip('/')}/ds004496/{row['stimulus_path'].split('ds004496/', 1)[-1]}",
            }
        )

    plan = pd.DataFrame(rows)
    report = {
        "source_manifest": str(manifest_path),
        "stimulus_rows": int(len(plan)),
        "unique_target_identifiers": int(plan["target_identifier"].nunique()),
        "estimated_total_bytes": total_estimated_bytes,
        "estimated_total_gib": round(total_estimated_bytes / (1024 ** 3), 3),
        "visible_stimulus_payloads": int(plan["stimulus_payload_visible"].sum()),
        "resolved_stimulus_payloads": int(plan["stimulus_payload_resolved"].sum()),
    }
    return plan, report


def materialize_public_nod_stimuli(
    manifest_path: Path,
    report_path: Path,
    base_url: str = DEFAULT_OPENNEURO_S3_BASE,
) -> dict:
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    manifest = pd.read_parquet(manifest_path).sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)
    repo_root = manifest_path.parents[3]

    downloaded = []
    failures = []
    total_downloaded_bytes = 0
    for row in manifest.to_dict(orient="records"):
        relpath = row["stimulus_path"].split("ds004496/", 1)[-1]
        source_url = f"{base_url.rstrip('/')}/ds004496/{relpath}"
        worktree_path = repo_root / row["stimulus_path"]
        destination = _symlink_target_path(worktree_path)
        try:
            downloaded_bytes = _download_to_path(source_url, destination)
        except (HTTPError, URLError) as exc:
            failures.append({"target_identifier": row["target_identifier"], "url": source_url, "error": str(exc)})
            continue
        downloaded.append(
            {
                "target_identifier": row["target_identifier"],
                "stimulus_path": row["stimulus_path"],
                "destination": str(destination),
                "bytes": downloaded_bytes,
            }
        )
        total_downloaded_bytes += downloaded_bytes

    refreshed_manifest = manifest.copy()
    resolved = []
    for relpath in refreshed_manifest["stimulus_path"].tolist():
        abspath = manifest_path.parents[3] / relpath
        resolved.append(abspath.exists())
    refreshed_manifest["stimulus_payload_resolved"] = resolved
    refreshed_manifest["embedding_status"] = [
        "embedding_pending" if is_resolved else "missing_image_payload" for is_resolved in resolved
    ]
    refreshed_manifest.to_parquet(manifest_path, index=False)

    report = {
        "source_manifest": str(manifest_path),
        "strategy": "direct_openneuro_s3",
        "stimulus_rows": int(len(manifest)),
        "downloaded_files": len(downloaded),
        "downloaded_bytes": total_downloaded_bytes,
        "downloaded_gib": round(total_downloaded_bytes / (1024 ** 3), 3),
        "resolved_stimulus_payloads_after": int(sum(resolved)),
        "downloaded": downloaded,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.materialize_public_nod_stimuli")
    parser = argparse.ArgumentParser(
        description="Materialize the exact fixed NOD stimulus JPEG subset referenced by the target-embedding manifest."
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--report", type=Path, default=_default_report_path())
    parser.add_argument("--openneuro-s3-base-url", default=DEFAULT_OPENNEURO_S3_BASE)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    plan, plan_report = build_public_nod_stimulus_retrieval_plan(manifest_path)
    print(f"Stimulus rows: {len(plan)}")
    print(f"Unique target identifiers: {plan_report['unique_target_identifiers']}")
    print(f"Estimated total GiB: {plan_report['estimated_total_gib']}")
    print(f"Visible payloads: {plan_report['visible_stimulus_payloads']}")
    print(f"Resolved payloads: {plan_report['resolved_stimulus_payloads']}")
    if not args.materialize:
        print("Dry run only. Pass --materialize to download the exact fixed JPEG slice.")
        return 0

    report = materialize_public_nod_stimuli(manifest_path, report_path, base_url=args.openneuro_s3_base_url)
    print(f"Downloaded files: {report['downloaded_files']}")
    print(f"Downloaded GiB: {report['downloaded_gib']}")
    print(f"Resolved payloads after: {report['resolved_stimulus_payloads_after']}")
    print(f"Retrieval report: {report_path}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
