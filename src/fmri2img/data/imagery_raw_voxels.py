"""
Raw-Voxel Imagery Dataset
=========================

Wraps NSDImageryDataset to produce raw nsdgeneral voxels (15,724-d) instead
of PCA-reduced features (3,072-d). This is the input format expected by
the FMRI2images model (825M param vMF decoder).

Usage:
    extractor = NSDGeneralExtractor.from_nsd_data("subj01", nsd_root)
    dataset = ImageryRawVoxelDataset(
        index_path="cache/indices/imagery/subj01.parquet",
        subject="subj01",
        extractor=extractor,
        condition="imagery",
        data_root="/path/to/nsd/nsdimagery",
    )
    for sample in dataset:
        voxels = sample["voxels"]  # (15724,) float32
        nsd_id = sample["nsd_id"]  # int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImageryRawVoxelDataset:
    """Dataset that yields raw nsdgeneral voxels from NSD-Imagery NIfTI files.

    Unlike NSDImageryDataset (which applies reliability mask + PCA), this
    dataset applies only the nsdgeneral ROI mask and per-volume z-score,
    matching FMRI2images preprocessing.

    Parameters
    ----------
    index_path : str
        Path to imagery index parquet file.
    subject : str
        Subject ID (e.g., "subj01").
    extractor : NSDGeneralExtractor
        Pre-configured extractor with nsdgeneral mask loaded.
    condition : str or None
        Filter by condition: "imagery", "perception", "attention", or None for all.
    data_root : str or None
        Root directory for resolving fMRI file paths in the index.
    clip_cache : object or None
        Optional CLIPCache for loading CLIP target embeddings.
    limit : int or None
        Maximum number of samples to yield.
    shuffle : bool
        Whether to shuffle trial order.
    seed : int
        Random seed for shuffling.
    """

    def __init__(
        self,
        index_path: str,
        subject: str,
        extractor: "NSDGeneralExtractor",
        condition: Optional[Literal["perception", "imagery", "attention"]] = None,
        data_root: Optional[str] = None,
        clip_cache: Optional[object] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        stimulus_type_filter: Optional[str] = None,
    ):
        self.index_path = Path(index_path)
        self.subject = subject
        self.extractor = extractor
        self.condition = condition
        self.data_root = Path(data_root) if data_root else None
        self.clip_cache = clip_cache
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed

        if not self.index_path.exists():
            raise FileNotFoundError(f"Imagery index not found: {self.index_path}")

        # Load and filter index
        df = pd.read_parquet(self.index_path)
        df = df[df["subject"] == subject]
        if condition is not None:
            df = df[df["condition"] == condition]
        if stimulus_type_filter is not None:
            df = df[df["stimulus_type"] == stimulus_type_filter]
        self.df = df.reset_index(drop=True)

        logger.info(
            f"ImageryRawVoxelDataset: {len(self.df)} trials "
            f"(subject={subject}, condition={condition})"
        )

    def __len__(self) -> int:
        if self.limit is not None:
            return min(len(self.df), self.limit)
        return len(self.df)

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over trials, yielding raw nsdgeneral voxels."""
        import random

        # NIfTI cache to avoid reloading 4D volumes
        _nifti_cache: Dict[str, object] = {}

        indices = list(range(len(self.df)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        if self.limit is not None:
            indices = indices[: self.limit]

        for idx in indices:
            row = self.df.iloc[idx]

            try:
                # Resolve fMRI path
                fmri_rel = row["fmri_path"]
                fmri_path = self._resolve_path(fmri_rel)
                beta_idx = row.get("beta_index", None)

                # Load 3D volume
                vol_3d = self._load_volume(fmri_path, beta_idx, _nifti_cache)

                # Extract raw nsdgeneral voxels
                voxels = self.extractor.extract(vol_3d)

                # Build sample
                sample = {
                    "voxels": voxels,
                    "condition": row["condition"],
                    "stimulus_type": row.get("stimulus_type", "unknown"),
                    "trial_id": row.get("trial_id", idx),
                    "nsd_id": None,
                    "run_number": row.get("run_number", None),
                    "target_image": None,
                    "target_text": None,
                }

                # Load target image if available
                image_path = row.get("image_path", None)
                if pd.notna(image_path):
                    try:
                        from PIL import Image
                        img_path = self._resolve_path(image_path)
                        if img_path.exists():
                            sample["target_image"] = Image.open(img_path).convert("RGB")
                    except Exception:
                        pass

                # Load target text if available
                text_prompt = row.get("text_prompt", None)
                if pd.notna(text_prompt):
                    sample["target_text"] = str(text_prompt)

                # Add nsd_id if available
                nsd_id = row.get("nsd_id", row.get("nsdId", None))
                if pd.notna(nsd_id):
                    sample["nsd_id"] = int(nsd_id)

                # Add CLIP target if cache available
                if self.clip_cache is not None and sample["nsd_id"] is not None:
                    try:
                        clip_emb = self.clip_cache.get_embedding(sample["nsd_id"])
                        sample["clip_target"] = clip_emb
                    except Exception:
                        pass

                yield sample

            except Exception as e:
                logger.warning(
                    f"Failed to load trial {row.get('trial_id', idx)}: {e}"
                )
                continue

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path against known roots."""
        if self.data_root is not None:
            p = self.data_root / rel_path
            if p.exists():
                return p
        p = Path(rel_path)
        if p.exists():
            return p
        # Return data_root version as best guess
        return (self.data_root / rel_path) if self.data_root else Path(rel_path)

    @staticmethod
    def _load_volume(
        fmri_path: Path,
        beta_idx: Optional[int],
        cache: Dict[str, object],
    ) -> np.ndarray:
        """Load a 3D volume from a NIfTI file."""
        import nibabel as nib

        key = str(fmri_path)
        if key not in cache:
            cache[key] = nib.load(fmri_path)
        img = cache[key]
        data = img.get_fdata()

        if data.ndim == 4 and beta_idx is not None:
            return data[..., int(beta_idx)].astype(np.float32)
        elif data.ndim == 3:
            return data.astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected NIfTI shape {data.shape} with beta_idx={beta_idx}"
            )

    def get_all_voxels(self) -> tuple:
        """Collect all voxels and metadata into arrays.

        Returns
        -------
        voxels : np.ndarray of shape (N, n_voxels)
        nsd_ids : np.ndarray of shape (N,) — -1 for missing
        conditions : list of str
        stimulus_types : list of str
        """
        all_voxels = []
        all_nsd_ids = []
        all_conditions = []
        all_stim_types = []

        for sample in self:
            all_voxels.append(sample["voxels"])
            all_nsd_ids.append(sample["nsd_id"] if sample["nsd_id"] is not None else -1)
            all_conditions.append(sample["condition"])
            all_stim_types.append(sample["stimulus_type"])

        return (
            np.stack(all_voxels),
            np.array(all_nsd_ids, dtype=np.int64),
            all_conditions,
            all_stim_types,
        )
