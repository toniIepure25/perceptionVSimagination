#!/usr/bin/env python3
"""
Build NSD-Imagery Index

Constructs canonical Parquet indices from raw NSD-Imagery data for use in
perception-to-imagery transfer evaluation.

Usage:
    # Build index for single subject
    python scripts/build_nsd_imagery_index.py \\
        --subject subj01 \\
        --cache-root cache/ \\
        --output cache/indices/imagery/subj01.parquet
    
    # Dry run (validate without writing)
    python scripts/build_nsd_imagery_index.py \\
        --subject subj01 \\
        --cache-root cache/ \\
        --output cache/indices/imagery/subj01.parquet \\
        --dry-run
    
    # Build for all available subjects
    for subj in subj01 subj02 subj05 subj07; do
        python scripts/build_nsd_imagery_index.py \\
            --subject $subj \\
            --cache-root cache/ \\
            --output cache/indices/imagery/${subj}.parquet
    done

Requirements:
    - NSD-Imagery data downloaded to cache/nsd_imagery/
    - NSD perception data available for cross-validation
    - Sufficient disk space for index files (~1-10MB per subject)

Output:
    Parquet index file with canonical ImageryTrial schema.
    See docs/technical/NSD_IMAGERY_DATASET_GUIDE.md for schema details.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build NSD-Imagery canonical index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for subject 01
  %(prog)s --subject subj01 --cache-root cache/ --output cache/indices/imagery/subj01.parquet
  
  # Dry run to validate data
  %(prog)s --subject subj01 --cache-root cache/ --output test.parquet --dry-run
  
  # Verbose output
  %(prog)s --subject subj01 --cache-root cache/ --output index.parquet --verbose

For more information, see:
  docs/technical/NSD_IMAGERY_DATASET_GUIDE.md
  docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=False,
        help='Root directory containing NSD-Imagery data. Supports subject-rooted or split metadata/beta layouts.'
    )
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Subject ID (e.g., subj01, subj02, subj05, subj07)'
    )
    parser.add_argument(
        '--cache-root',
        type=str,
        required=True,
        help='Root directory for cached data (e.g., cache/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for Parquet index (e.g., cache/indices/imagery/subj01.parquet)'
    )
    parser.add_argument(
        '--stimulus-root',
        type=str,
        default=None,
        help='Optional root directory for stimulus files (images/text)'
    )
    parser.add_argument(
        '--metadata-root',
        type=str,
        default=None,
        help='Optional metadata root for split layouts (contains designmatrixGLMsingle.mat / pair lists)'
    )
    parser.add_argument(
        '--beta-root',
        type=str,
        default=None,
        help='Optional beta root for split layouts (expects {beta_root}/{subject}/betas_nsdimagery.nii.gz)'
    )
    parser.add_argument(
        '--beta-path',
        type=str,
        default=None,
        help='Optional explicit beta NIfTI path for split layouts'
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default=None,
        help='Optional JSON sidecar describing the discovered imagery layout and source provenance'
    )
    
    # Optional arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate data and show preview without writing output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output file if present'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate NSD-Imagery data availability, do not build index'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not any([args.data_root, args.metadata_root, args.beta_root, args.beta_path]):
        print(
            "ERROR: NSD-Imagery data location (nsd_imagery) is required. Provide --data-root for a subject-rooted layout "
            "or --metadata-root with --beta-root/--beta-path for the split imagery layout.",
            file=sys.stderr,
        )
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}", file=sys.stderr)
        print(f"Please specify correct path with --data-root", file=sys.stderr)
        sys.exit(1)
    
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    
    # Check if NSD-Imagery data exists for this subject or split metadata/beta layout.
    if data_root is not None:
        subject_data_dir = data_root / args.subject
        if not subject_data_dir.exists():
            print(f"ERROR: Subject data directory not found: {subject_data_dir}", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(f"Available subjects:", file=sys.stderr)
            if data_root.exists():
                available = [d.name for d in data_root.iterdir() if d.is_dir()]
                for subj in available:
                    print(f"  - {subj}")
            else:
                print(f"  (data root directory not found)")
            sys.exit(1)
    elif args.metadata_root is None:
        print("ERROR: split-layout mode requires --metadata-root.", file=sys.stderr)
        sys.exit(1)
    
    if args.validate_only:
        print(f"✓ NSD-Imagery data found for {args.subject}")
        if data_root is not None:
            print(f"  Data directory: {subject_data_dir}")
        if args.metadata_root:
            print(f"  Metadata root: {args.metadata_root}")
        if args.beta_root:
            print(f"  Beta root: {args.beta_root}")
        if args.beta_path:
            print(f"  Beta path: {args.beta_path}")
        print(f"  Data appears valid (basic check only)")
        sys.exit(0)
    
    # Check if output already exists
    output_path = Path(args.output)
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"ERROR: Output file already exists: {output_path}", file=sys.stderr)
        print(f"Use --force to overwrite or --dry-run to preview", file=sys.stderr)
        sys.exit(1)
    
    # Import here to give helpful error if data is missing (checked above)
    try:
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index
    except ImportError as e:
        print(f"ERROR: Failed to import build_nsd_imagery_index: {e}", file=sys.stderr)
        print(f"Make sure fmri2img package is installed: pip install -e .", file=sys.stderr)
        sys.exit(1)
    
    # Call the index building function
    print(f"Building NSD-Imagery index for {args.subject}...")
    if data_root is not None:
        print(f"  Data root: {data_root}")
    if args.metadata_root:
        print(f"  Metadata root: {args.metadata_root}")
    if args.beta_root:
        print(f"  Beta root: {args.beta_root}")
    if args.beta_path:
        print(f"  Beta path: {args.beta_path}")
    print(f"  Cache root: {cache_root}")
    print(f"  Output: {output_path}")
    print(f"  Dry run: {args.dry_run}")
    print("")
    
    try:
        stimulus_root = Path(args.stimulus_root) if args.stimulus_root else None
        
        result_path = build_nsd_imagery_index(
            data_root=data_root,
            subject=args.subject,
            cache_root=cache_root,
            output_path=output_path,
            stimulus_root=stimulus_root,
            metadata_root=Path(args.metadata_root) if args.metadata_root else None,
            beta_root=Path(args.beta_root) if args.beta_root else None,
            beta_path=Path(args.beta_path) if args.beta_path else None,
            report_path=Path(args.report_path) if args.report_path else None,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        
        if args.dry_run:
            print("")
            print("Dry run completed successfully!")
            print("Run without --dry-run to write output")
        else:
            print("")
            print(f"✓ Index built successfully: {result_path}")
            print(f"")
            print(f"Next steps:")
            print(f"  1. Run evaluation:")
            print(f"     python scripts/eval_perception_to_imagery_transfer.py \\")
            print(f"       --index {result_path} \\")
            print(f"       --checkpoint checkpoints/two_stage/{args.subject}/best.pt \\")
            print(f"       --mode imagery \\")
            print(f"       --output-dir outputs/reports/imagery/")
        
        sys.exit(0)
    
    except NotImplementedError as e:
        print(f"", file=sys.stderr)
        print(f"=" * 80, file=sys.stderr)
        print(f"NOT YET IMPLEMENTED", file=sys.stderr)
        print(f"=" * 80, file=sys.stderr)
        print(f"", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"This is a placeholder script. The index building logic will be", file=sys.stderr)
        print(f"implemented in a follow-up commit.", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"Current status: Phase 1 (Documentation + Scaffolding)", file=sys.stderr)
        print(f"Next step: Implement build_nsd_imagery_index() in src/fmri2img/data/nsd_imagery.py", file=sys.stderr)
        print(f"", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Index building failed: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
