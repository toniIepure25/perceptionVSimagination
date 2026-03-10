#!/usr/bin/env python3
"""
Memory-efficient fMRI feature extraction.
==========================================

Pre-extracts fMRI features from NIfTI volumes to numpy arrays on disk,
avoiding repeated NIfTI loading during training. Groups by session file
to load each NIfTI exactly once.

Memory profile: ~4GB peak (one 4D NIfTI file + preprocessing artifacts)
Output: X.npy (N, D) and nsd_ids.npy (N,) saved to disk

Usage:
    python scripts/extract_features.py \\
        --subject subj01 \\
        --preproc-dir outputs/preproc/baseline \\
        --output-dir outputs/features/baseline/subj01
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract fMRI features to disk")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--preproc-dir", required=True, help="Preprocessing artifacts dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for features")
    parser.add_argument("--limit", type=int, help="Limit number of trials")
    args = parser.parse_args()

    from fmri2img.data.nsd_index_reader import read_subject_index
    from fmri2img.data.preprocess import NSDPreprocessor

    # Load index
    df = read_subject_index(args.index_root, args.subject)
    if args.limit:
        df = df.head(args.limit)
    logger.info(f"Index: {len(df)} trials for {args.subject}")

    # Load preprocessor
    prep = NSDPreprocessor(args.subject, args.preproc_dir)
    if not prep.load_artifacts():
        logger.error("Preprocessing artifacts not found!")
        return 1
    summary = prep.summary()
    logger.info(f"Preprocessor: {summary.get('n_voxels_kept', '?')} voxels, "
                f"PCA {summary.get('pca_components', 'N/A')} components")

    # Determine output dim from a dummy volume
    dummy_shape = tuple(prep.mask_.shape) if prep.mask_ is not None else None
    if dummy_shape:
        dummy = np.zeros(dummy_shape, dtype=np.float32)
        out = prep.transform(dummy)
        output_dim = out.shape[0]
        logger.info(f"Output feature dimension: {output_dim}")
    else:
        logger.error("Cannot determine output dim without mask")
        return 1

    # Group by beta_path (session file) to load each NIfTI once
    df["_row_idx"] = range(len(df))
    groups = df.groupby("beta_path")
    n_files = len(groups)
    logger.info(f"Processing {n_files} NIfTI files ({len(df)} trials total)")

    # Allocate output arrays
    X = np.zeros((len(df), output_dim), dtype=np.float32)
    nsd_ids = np.zeros(len(df), dtype=np.int64)
    valid_mask = np.zeros(len(df), dtype=bool)

    import nibabel as nib
    
    t0 = time.time()
    for file_idx, (beta_path, file_df) in enumerate(groups):
        t_file = time.time()
        
        # Load the 4D NIfTI file once
        try:
            img = nib.load(str(beta_path))
            data_4d = img.get_fdata().astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to load {beta_path}: {e}")
            continue

        # Extract all volumes from this file
        n_extracted = 0
        for _, row in file_df.iterrows():
            row_idx = int(row["_row_idx"])
            beta_index = int(row["beta_index"])
            nsd_id = int(row["nsdId"])

            try:
                vol = data_4d[..., beta_index]
                features = prep.transform(vol)
                X[row_idx] = features
                nsd_ids[row_idx] = nsd_id
                valid_mask[row_idx] = True
                n_extracted += 1
            except Exception as e:
                logger.warning(f"Failed row {row_idx} (nsd={nsd_id}): {e}")

        # Free memory
        del data_4d, img
        gc.collect()

        elapsed = time.time() - t_file
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (file_idx + 1)) * (n_files - file_idx - 1)
        logger.info(
            f"[{file_idx+1}/{n_files}] {Path(beta_path).name}: "
            f"{n_extracted}/{len(file_df)} volumes, "
            f"{elapsed:.1f}s, ETA {eta/60:.1f}min"
        )

    # Filter valid
    n_valid = valid_mask.sum()
    X_valid = X[valid_mask]
    nsd_ids_valid = nsd_ids[valid_mask]
    # Save original indices for mapping back to dataframe
    orig_indices = np.where(valid_mask)[0]

    logger.info(f"Extracted {n_valid}/{len(df)} valid features, shape={X_valid.shape}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X_valid)
    np.save(out_dir / "nsd_ids.npy", nsd_ids_valid)
    np.save(out_dir / "orig_indices.npy", orig_indices)

    meta = {
        "subject": args.subject,
        "n_trials": int(n_valid),
        "n_total": int(len(df)),
        "feature_dim": int(output_dim),
        "preproc_dir": args.preproc_dir,
        "index_root": args.index_root,
        "pca_components": summary.get("pca_components"),
        "n_voxels_kept": summary.get("n_voxels_kept"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    logger.info(f"Done in {total_time/60:.1f} minutes")
    logger.info(f"Saved: {out_dir}/X.npy ({X_valid.nbytes/1e6:.1f}MB)")
    logger.info(f"Saved: {out_dir}/nsd_ids.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
