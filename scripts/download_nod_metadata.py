#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/OpenNeuroDatasets/ds004496.git"
DEFAULT_OUTPUT = "cache/public_datasets/ds004496"
MIN_FREE_GB = 1.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_path() -> Path:
    return _repo_root() / DEFAULT_OUTPUT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire a metadata-only working copy of OpenNeuro ds004496 (NOD) "
            "for the practical Animus lane."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help=f"Target directory for the metadata clone (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git mirror to clone for metadata-only provenance acquisition.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without cloning anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing non-git output path before cloning.",
    )
    return parser.parse_args()


def _free_space_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _write_provenance(
    output_dir: Path,
    repo_url: str,
    free_space_gb: float,
    clone_performed: bool,
) -> Path:
    provenance_path = output_dir / "acquisition_provenance.json"
    payload = {
        "dataset_id": "ds004496",
        "label": "Natural Object Dataset (NOD)",
        "mode": "metadata_only_git_clone",
        "lane": "practical Animus lane",
        "classification": "perception-only Animus robustness dataset",
        "threshold_ladder_role": "not a replacement for the primary paired threshold ladder",
        "source_repo": repo_url,
        "output_dir": str(output_dir),
        "clone_performed": clone_performed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "free_space_gb_at_write": round(free_space_gb, 2),
        "notes": [
            "This acquisition mode only captures the OpenNeuro Git metadata surface.",
            "Full annexed imaging content is not downloaded by this command.",
            "Use this as the first safe remote integration step before larger acquisition planning.",
        ],
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n")
    return provenance_path


def main() -> int:
    args = _parse_args()
    output_dir = args.output.expanduser()
    parent_dir = output_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    free_space_gb = _free_space_gb(parent_dir)
    if free_space_gb < MIN_FREE_GB:
        raise SystemExit(
            f"Refusing ds004496 metadata acquisition: only {free_space_gb:.2f} GB free at {parent_dir}. "
            f"Need at least {MIN_FREE_GB:.1f} GB free."
        )

    existing_git_dir = output_dir / ".git"
    if output_dir.exists() and existing_git_dir.exists():
        provenance_path = _write_provenance(
            output_dir=output_dir,
            repo_url=args.repo_url,
            free_space_gb=free_space_gb,
            clone_performed=False,
        )
        print(f"ds004496 metadata clone already present at {output_dir}")
        print(f"Updated provenance: {provenance_path}")
        return 0

    if output_dir.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing non-git path: {output_dir}\n"
            "Remove it manually or rerun with --force."
        )

    if args.dry_run:
        print("DRY RUN — no files cloned.")
        print(f"Source repo: {args.repo_url}")
        print(f"Target path: {output_dir}")
        print(f"Free space at target parent: {free_space_gb:.2f} GB")
        print("Acquisition mode: metadata_only_git_clone")
        return 0

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", args.repo_url, str(output_dir)],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    provenance_path = _write_provenance(
        output_dir=output_dir,
        repo_url=args.repo_url,
        free_space_gb=free_space_gb,
        clone_performed=True,
    )
    print(f"Cloned ds004496 metadata to {output_dir}")
    print(f"Wrote provenance: {provenance_path}")
    print("Note: annexed imaging content is not downloaded by this command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
