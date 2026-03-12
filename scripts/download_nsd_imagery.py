#!/usr/bin/env python3
"""
Download NSD-Imagery Dataset from AWS S3
=========================================

Downloads the NSD-Imagery fMRI data (mental imagery betas, experiment metadata,
and imagery stimuli) from the public NSD S3 bucket. No credentials needed.

Data comes from three S3 prefixes:
    1. nsddata/experiments/nsdimagery/          — experiment design, trial info (~9 MB)
    2. nsddata_betas/ppdata/{subj}/func1pt8mm/
       nsdimagerybetas_fithrf_GLMdenoise_RR/    — fMRI betas (~1.5 GB per subject)
    3. nsddata_stimuli/stimuli/nsdimagery/       — cue images (~614 MB)

Total sizes:
    - Minimal (subj01 only):    ~2.2 GB
    - All 4 project subjects:   ~6.6 GB
    - All 8 NSD subjects:      ~12.6 GB

Usage:
    # Check disk space and do dry run
    python scripts/download_nsd_imagery.py --dry-run

    # Download subj01 only (minimal, ~2.2 GB)
    python scripts/download_nsd_imagery.py --subjects subj01

    # Download all 4 subjects used in project (~6.6 GB)
    python scripts/download_nsd_imagery.py --subjects subj01 subj02 subj05 subj07

    # Download everything (~12.6 GB)
    python scripts/download_nsd_imagery.py --subjects all

    # Custom output directory
    python scripts/download_nsd_imagery.py --subjects subj01 --output /home/jovyan/local-data/nsd_imagery

    # Skip stimuli (only betas + metadata)
    python scripts/download_nsd_imagery.py --subjects subj01 --skip-stimuli
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "natural-scenes-dataset"
REGION = "us-east-1"

# S3 prefixes
PREFIX_METADATA = "nsddata/experiments/nsdimagery/"
PREFIX_STIMULI = "nsddata_stimuli/stimuli/nsdimagery/"
PREFIX_BETAS_TEMPLATE = (
    "nsddata_betas/ppdata/{subject}/func1pt8mm/"
    "nsdimagerybetas_fithrf_GLMdenoise_RR/"
)

ALL_SUBJECTS = ["subj01", "subj02", "subj03", "subj04",
                "subj05", "subj06", "subj07", "subj08"]
PROJECT_SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]

# Minimum free space required (in GB) before we start downloading
MIN_FREE_SPACE_GB = 15.0


def get_free_space_gb(path: str) -> float:
    """Return free disk space in GB for the filesystem containing `path`."""
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def get_s3_client():
    """Create an anonymous S3 client for the NSD public bucket."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("ERROR: boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED),
    )


def list_s3_objects(s3, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns list of {Key, Size, ETag}."""
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({
                "Key": obj["Key"],
                "Size": obj["Size"],
                "ETag": obj.get("ETag", "").strip('"'),
            })
    return objects


def download_file(s3, key: str, dest: Path, expected_size: int,
                  retries: int = 3) -> bool:
    """
    Download a single file from S3 with retry and integrity check.

    Returns True on success, False on failure.
    """
    # Skip 0-byte "directory" objects (S3 folder markers)
    if expected_size == 0 and key.endswith("/"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        return True

    # Ensure parent exists (handle case where a "dir/" object was a file)
    parent = dest.parent
    if parent.exists() and parent.is_file():
        parent.unlink()
    parent.mkdir(parents=True, exist_ok=True)

    # If dest path is itself a directory (from a prev "dir/" download), remove it
    if dest.exists() and dest.is_dir():
        shutil.rmtree(dest)

    # Skip if already downloaded with correct size
    if dest.exists() and dest.stat().st_size == expected_size:
        return True

    for attempt in range(1, retries + 1):
        try:
            s3.download_file(BUCKET, key, str(dest))
            # Verify size
            actual = dest.stat().st_size
            if actual == expected_size:
                return True
            else:
                print(f"  WARNING: Size mismatch for {key}: "
                      f"expected {expected_size}, got {actual}")
                dest.unlink(missing_ok=True)
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed for {key}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    return False


def build_manifest(s3, subjects: list[str],
                   skip_stimuli: bool = False) -> list[dict]:
    """
    Build the full download manifest: list of {key, size, relative_path}.

    Structure on disk:
        output_dir/
        ├── metadata/           (from nsddata/experiments/nsdimagery/)
        ├── stimuli/            (from nsddata_stimuli/stimuli/nsdimagery/)
        └── betas/
            └── {subject}/      (from nsddata_betas/ppdata/{subject}/...)
    """
    manifest = []

    # 1. Metadata
    print("Scanning metadata...")
    for obj in list_s3_objects(s3, PREFIX_METADATA):
        # Strip prefix to get relative path
        rel = obj["Key"][len(PREFIX_METADATA):]
        manifest.append({
            "key": obj["Key"],
            "size": obj["Size"],
            "rel_path": f"metadata/{rel}",
            "category": "metadata",
        })

    # 2. Stimuli
    if not skip_stimuli:
        print("Scanning stimuli...")
        for obj in list_s3_objects(s3, PREFIX_STIMULI):
            rel = obj["Key"][len(PREFIX_STIMULI):]
            manifest.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "rel_path": f"stimuli/{rel}",
                "category": "stimuli",
            })

    # 3. Betas per subject
    for subj in subjects:
        prefix = PREFIX_BETAS_TEMPLATE.format(subject=subj)
        print(f"Scanning betas for {subj}...")
        for obj in list_s3_objects(s3, prefix):
            rel = obj["Key"][len(prefix):]
            manifest.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "rel_path": f"betas/{subj}/{rel}",
                "category": f"betas_{subj}",
            })

    # Filter out 0-byte directory markers (S3 "folder" objects ending in /)
    manifest = [m for m in manifest if not (m["size"] == 0 and m["key"].endswith("/"))]
    # Filter out entries with empty relative paths
    manifest = [m for m in manifest if m["rel_path"].rstrip("/")]

    return manifest


def print_manifest_summary(manifest: list[dict]) -> dict:
    """Print download summary grouped by category. Returns size totals."""
    categories: dict[str, dict] = {}
    for item in manifest:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "size": 0}
        categories[cat]["count"] += 1
        categories[cat]["size"] += item["size"]

    total_size = sum(c["size"] for c in categories.values())
    total_count = sum(c["count"] for c in categories.values())

    print()
    print("=" * 60)
    print("Download Manifest")
    print("=" * 60)
    for cat, info in sorted(categories.items()):
        print(f"  {cat:<20s}  {info['count']:>5d} files  {human_size(info['size']):>10s}")
    print(f"  {'─' * 48}")
    print(f"  {'TOTAL':<20s}  {total_count:>5d} files  {human_size(total_size):>10s}")
    print("=" * 60)

    return {"total_size": total_size, "total_count": total_count, "categories": categories}


def main():
    parser = argparse.ArgumentParser(
        description="Download NSD-Imagery dataset from AWS S3 (public, no auth needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                              # Check space, list files
  %(prog)s --subjects subj01                      # Minimal download (~2.2 GB)
  %(prog)s --subjects subj01 subj02 subj05 subj07 # Project subjects (~6.6 GB)
  %(prog)s --subjects all                          # All 8 subjects (~12.6 GB)
  %(prog)s --subjects subj01 --skip-stimuli        # Betas + metadata only (~1.6 GB)
        """,
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["subj01"],
        help="Subjects to download (e.g., subj01 subj02). Use 'all' for all 8. "
             "Default: subj01",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: {NSD_DATA}/nsdimagery/ where NSD_DATA "
             "is auto-detected from existing NSD data, or /home/jovyan/work/data/nsd/nsdimagery",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and check space without downloading",
    )
    parser.add_argument(
        "--skip-stimuli",
        action="store_true",
        help="Skip stimulus image download (saves ~614 MB)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_SPACE_GB,
        help=f"Minimum free disk space in GB (default: {MIN_FREE_SPACE_GB})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if free space is below threshold",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4)",
    )
    args = parser.parse_args()

    # Resolve subjects
    if args.subjects == ["all"]:
        subjects = ALL_SUBJECTS
    else:
        subjects = args.subjects
        for s in subjects:
            if s not in ALL_SUBJECTS:
                print(f"ERROR: Unknown subject '{s}'. Valid: {ALL_SUBJECTS}")
                sys.exit(1)

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Auto-detect from existing NSD data
        candidates = [
            Path("/home/jovyan/work/data/nsd/nsdimagery"),
            Path("/home/jovyan/local-data/nsd_imagery"),
            Path("cache/nsd_imagery"),
        ]
        # Check if NSD perception data exists somewhere
        for cand_parent in [Path("/home/jovyan/work/data/nsd"),
                            Path("/home/jovyan/local-data/data/nsd")]:
            if cand_parent.exists():
                output_dir = cand_parent / "nsdimagery"
                break
        else:
            output_dir = candidates[0]

    print(f"NSD-Imagery Downloader")
    print(f"  Subjects:  {', '.join(subjects)}")
    print(f"  Output:    {output_dir}")
    print(f"  Stimuli:   {'skip' if args.skip_stimuli else 'include'}")
    print()

    # -----------------------------------------------------------------------
    # Disk space check
    # -----------------------------------------------------------------------
    # Find the mount point for the output directory
    check_path = str(output_dir)
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
        if check_path == "/":
            break

    free_gb = get_free_space_gb(check_path)
    print(f"  Disk free: {free_gb:.1f} GB (on {check_path})")
    print(f"  Minimum:   {args.min_free_gb:.1f} GB")

    if free_gb < args.min_free_gb and not args.force:
        print()
        print(f"ERROR: Insufficient disk space!")
        print(f"  Available: {free_gb:.1f} GB")
        print(f"  Required:  {args.min_free_gb:.1f} GB minimum")
        print()
        print("Options:")
        print("  --force           Download anyway")
        print("  --skip-stimuli    Save ~614 MB")
        print("  --subjects subj01 Download only one subject (~2.2 GB)")
        print("  --output <path>   Use a different filesystem")
        sys.exit(1)

    print(f"  ✓ Sufficient disk space ({free_gb:.1f} GB >= {args.min_free_gb:.1f} GB)")
    print()

    # -----------------------------------------------------------------------
    # Connect to S3 and build manifest
    # -----------------------------------------------------------------------
    s3 = get_s3_client()
    manifest = build_manifest(s3, subjects, skip_stimuli=args.skip_stimuli)

    if not manifest:
        print("ERROR: No files found in S3. Check your network connection.")
        sys.exit(1)

    summary = print_manifest_summary(manifest)
    total_size = summary["total_size"]
    total_count = summary["total_count"]

    # Check if download would exceed available space (with 2GB buffer)
    needed_gb = total_size / (1024 ** 3)
    if needed_gb + 2.0 > free_gb and not args.force:
        print()
        print(f"ERROR: Download ({needed_gb:.1f} GB) would not fit with 2 GB buffer!")
        print(f"  Free: {free_gb:.1f} GB, Need: {needed_gb:.1f} GB + 2 GB buffer")
        sys.exit(1)

    if args.dry_run:
        print()
        print("DRY RUN — no files downloaded.")
        print(f"To download, run without --dry-run")
        print()
        # Show sample files
        print("Sample files:")
        for item in manifest[:10]:
            print(f"  {item['rel_path']} ({human_size(item['size'])})")
        if len(manifest) > 10:
            print(f"  ... and {len(manifest) - 10} more")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest for reproducibility
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "subjects": subjects,
            "skip_stimuli": args.skip_stimuli,
            "total_size_bytes": total_size,
            "total_files": total_count,
            "files": manifest,
        }, f, indent=2)

    print()
    print(f"Downloading {total_count} files ({human_size(total_size)})...")
    print()

    downloaded = 0
    downloaded_bytes = 0
    skipped = 0
    failed = 0
    failed_keys = []
    start_time = time.time()

    for i, item in enumerate(manifest, 1):
        dest = output_dir / item["rel_path"]

        # Skip already-downloaded files
        if dest.exists() and dest.stat().st_size == item["size"]:
            skipped += 1
            downloaded_bytes += item["size"]
            continue

        # Progress
        pct = (i / total_count) * 100
        elapsed = time.time() - start_time
        if downloaded > 0:
            rate = downloaded_bytes / elapsed  # bytes/sec
            remaining = (total_size - downloaded_bytes) / max(rate, 1)
            eta = f"ETA {remaining / 60:.0f}m"
        else:
            eta = ""

        print(f"  [{i}/{total_count}] ({pct:5.1f}%) {item['rel_path']:<60s} "
              f"{human_size(item['size']):>10s}  {eta}")

        ok = download_file(s3, item["key"], dest, item["size"])
        if ok:
            downloaded += 1
            downloaded_bytes += item["size"]
        else:
            failed += 1
            failed_keys.append(item["key"])

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"  Downloaded:  {downloaded} files ({human_size(downloaded_bytes)})")
    print(f"  Skipped:     {skipped} files (already existed)")
    print(f"  Failed:      {failed} files")
    print(f"  Duration:    {elapsed / 60:.1f} minutes")
    print(f"  Output:      {output_dir}")

    if failed_keys:
        print()
        print("FAILED files (re-run script to retry):")
        for k in failed_keys:
            print(f"  {k}")

    # Save completion marker
    marker = output_dir / ".download_complete"
    with open(marker, "w") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "subjects": subjects,
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "total_bytes": downloaded_bytes,
        }, indent=2))

    # Final disk check
    final_free = get_free_space_gb(str(output_dir))
    print(f"  Disk free:   {final_free:.1f} GB remaining")
    print("=" * 60)

    if failed:
        print()
        print(f"WARNING: {failed} files failed. Re-run the script to retry.")
        sys.exit(1)

    print()
    print("Next steps:")
    print(f"  1. Verify:  ls -lR {output_dir}/betas/")
    print(f"  2. Index:   python scripts/build_nsd_imagery_index.py \\")
    print(f"                --data-root {output_dir} --subject subj01 \\")
    print(f"                --cache-root cache/ --output cache/indices/imagery/subj01.parquet")
    print(f"  3. Preprocess: python scripts/fit_preprocessing.py --domain imagery")


if __name__ == "__main__":
    main()
