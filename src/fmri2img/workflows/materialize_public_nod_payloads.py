from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.inspect_public_nod import _default_dataset_root
from fmri2img.workflows.prepare_public_nod_index import DEFAULT_OUTPUT as DEFAULT_INDEX


_ANNEX_SIZE_RE = re.compile(r"SHA256E-s(\d+)--")
_REQUIRED_COLUMNS = {
    "events": "events_path",
    "preproc_bold": "preproc_bold_path",
    "confounds": "confounds_path",
    "ciftify_beta": "ciftify_beta_path",
    "ciftify_label": "ciftify_label_path",
}


def _default_index_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_INDEX


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "cache/indices/public_nod/imagenet_missing_payload_manifest.json"


def _default_report_path() -> Path:
    return Path(__file__).resolve().parents[3] / "cache/indices/public_nod/imagenet_missing_payload_report.json"


def _annex_target_size_bytes(path: Path) -> int | None:
    if not path.is_symlink():
        return path.stat().st_size if path.exists() else None
    target = path.readlink().as_posix()
    match = _ANNEX_SIZE_RE.search(target)
    if match:
        return int(match.group(1))
    return None


def build_missing_payload_manifest(dataset_root: Path, index_path: Path, status_filter: str = "missing_payload") -> tuple[dict, dict]:
    dataset_root = dataset_root.resolve()
    index_path = index_path.resolve()
    df = pd.read_parquet(index_path)
    subset = df[df["row_status"] == status_filter].sort_values(["subject", "session", "run"]).reset_index(drop=True)

    entries = []
    total_bytes = 0
    bytes_by_class = {key: 0 for key in _REQUIRED_COLUMNS if key != "events"}
    resolved_by_class = {key: 0 for key in _REQUIRED_COLUMNS}
    visible_by_class = {key: 0 for key in _REQUIRED_COLUMNS}

    for row in subset.to_dict(orient="records"):
        files = {}
        for key, column in _REQUIRED_COLUMNS.items():
            relpath = row[column]
            abspath = dataset_root / relpath
            visible = bool(row[f"{key}_visible"])
            resolved = bool(row[f"{key}_resolved"])
            estimated_bytes = _annex_target_size_bytes(abspath)
            files[key] = {
                "path": relpath,
                "visible": visible,
                "resolved": resolved,
                "estimated_bytes": estimated_bytes,
            }
            visible_by_class[key] += int(visible)
            resolved_by_class[key] += int(resolved)
            if key != "events" and estimated_bytes is not None:
                bytes_by_class[key] += estimated_bytes
                total_bytes += estimated_bytes

        entries.append(
            {
                "subject": row["subject"],
                "session": row["session"],
                "run": int(row["run"]),
                "task": row["task"],
                "row_status": row["row_status"],
                "usable_for_later_shared_only_prep": bool(row["usable_for_later_shared_only_prep"]),
                "files": files,
            }
        )

    manifest = {
        "dataset_id": "ds004496",
        "dataset_root": str(dataset_root),
        "index_path": str(index_path),
        "status_filter": status_filter,
        "entry_count": len(entries),
        "entries": entries,
    }
    report = {
        "dataset_id": "ds004496",
        "dataset_root": str(dataset_root),
        "index_path": str(index_path),
        "status_filter": status_filter,
        "entry_count": len(entries),
        "subjects": sorted({entry["subject"] for entry in entries}),
        "sessions": sorted({entry["session"] for entry in entries}),
        "runs": sorted({entry["run"] for entry in entries}),
        "bytes_by_class": bytes_by_class,
        "total_estimated_bytes": total_bytes,
        "total_estimated_gib": round(total_bytes / (1024 ** 3), 3),
        "visible_by_class": visible_by_class,
        "resolved_by_class": resolved_by_class,
    }
    return manifest, report


def _materialize_paths(dataset_root: Path, manifest: dict) -> int:
    if shutil.which("git-annex") is None:
        print(
            "git-annex is not available on this host. Install git-annex on the live pod before "
            "running `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize`.",
            file=sys.stderr,
        )
        return 2

    wanted = []
    for entry in manifest["entries"]:
        for file_key in ("preproc_bold", "confounds", "ciftify_beta", "ciftify_label"):
            info = entry["files"][file_key]
            if info["visible"] and not info["resolved"]:
                wanted.append(info["path"])
    wanted = sorted(set(wanted))
    if not wanted:
        print("No unresolved annex-backed NOD payloads matched the selected manifest rows.")
        return 0

    result = subprocess.run(
        ["git-annex", "get", *wanted],
        cwd=dataset_root,
        check=False,
    )
    if result.returncode != 0:
        print(
            "git-annex ran, but one or more requested payloads were not retrievable. "
            "This usually means the current dataset clone has no usable annex source "
            "for the selected keys. Inspect `git config --get-regexp '^remote\\..*annex|^annex\\..*'` "
            "and `git-annex whereis <path>` inside the dataset clone before retrying.",
            file=sys.stderr,
        )
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.materialize_public_nod_payloads")
    parser = argparse.ArgumentParser(
        description="Build an exact manifest for the first unresolved NOD payload subset and optionally materialize it via git-annex."
    )
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--index", type=Path, default=_default_index_path())
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--report", type=Path, default=_default_report_path())
    parser.add_argument("--status-filter", default="missing_payload")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    report_path = args.report.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest, report = build_missing_payload_manifest(args.dataset_root, args.index, status_filter=args.status_filter)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(f"Entries: {report['entry_count']}")
    print(f"Estimated GiB: {report['total_estimated_gib']}")
    print(f"Runs: {report['runs']}")

    if not args.materialize:
        return 0
    return _materialize_paths(args.dataset_root.resolve(), manifest)


if __name__ == "__main__":
    raise SystemExit(main())
