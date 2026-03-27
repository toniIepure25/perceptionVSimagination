"""
NSDGeneral Voxel Extractor
==========================

Extracts raw nsdgeneral ROI voxels from 3D fMRI volumes for use with the
FMRI2images model, which expects exactly 15,724 raw voxels as input.

The nsdgeneral mask defines the "general-purpose visual cortex" ROI used by
the FMRI2images project (825M param vMF decoder, ViT-bigG/14). This module
provides a clean extraction path that matches FMRI2images preprocessing:
    1. Load nsdgeneral boolean mask (81 x 104 x 83)
    2. Apply mask to 3D volume → flat vector
    3. Per-volume z-score normalization

Usage:
    extractor = NSDGeneralExtractor.from_nsd_data("subj01", "/path/to/nsd")
    voxels = extractor.extract(vol_3d)                     # (15724,)
    batch = extractor.extract_batch_from_nifti(path, [0,1]) # (2, 15724)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


class NSDGeneralExtractor:
    """Extract raw nsdgeneral voxels from 3D NIfTI volumes.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of shape (81, 104, 83) or similar 3D shape.
    expected_dim : int
        Expected number of True voxels. Raises if mismatch.
    zscore : bool
        If True, apply per-volume z-score after masking (default True,
        matching FMRI2images preprocessing).
    """

    def __init__(
        self,
        mask: np.ndarray,
        expected_dim: int = 15724,
        zscore: bool = True,
    ):
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, got shape {mask.shape}")
        self.mask = mask.astype(bool)
        self.n_voxels = int(self.mask.sum())
        self.zscore = zscore

        if expected_dim > 0 and self.n_voxels != expected_dim:
            raise ValueError(
                f"nsdgeneral mask has {self.n_voxels} voxels, "
                f"expected {expected_dim}. Check mask file."
            )
        logger.info(
            f"NSDGeneralExtractor: mask {self.mask.shape}, "
            f"{self.n_voxels} voxels, zscore={zscore}"
        )

    @classmethod
    def from_nsd_data(
        cls,
        subject: str,
        nsd_root: str,
        expected_dim: int = 15724,
        zscore: bool = True,
    ) -> "NSDGeneralExtractor":
        """Auto-discover nsdgeneral mask from standard NSD paths.

        Searches for the mask at:
            {nsd_root}/ppdata/{subject}/func1pt8mm/roi/nsdgeneral.nii.gz

        Falls back to common alternative paths if not found.
        """
        import nibabel as nib

        nsd_root = Path(nsd_root)
        candidates = [
            nsd_root / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
            nsd_root / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
            nsd_root / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii",
            nsd_root / subject / "roi" / "nsdgeneral.nii.gz",
        ]

        mask_path = None
        for p in candidates:
            if p.exists():
                mask_path = p
                break

        if mask_path is None:
            tried = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"nsdgeneral ROI mask not found for {subject}. Tried:\n  {tried}\n"
                f"Download from NSD S3 bucket or copy from FMRI2images project."
            )

        logger.info(f"Loading nsdgeneral mask from {mask_path}")
        img = nib.load(mask_path)
        mask_data = img.get_fdata()

        # Convert to boolean (nsdgeneral is typically integer labels)
        mask = mask_data > 0
        return cls(mask=mask, expected_dim=expected_dim, zscore=zscore)

    @classmethod
    def from_mask_file(
        cls,
        mask_path: str,
        expected_dim: int = 15724,
        zscore: bool = True,
    ) -> "NSDGeneralExtractor":
        """Load mask from an explicit file path (NIfTI or numpy)."""
        mask_path = Path(mask_path)

        if mask_path.suffix in (".npy", ".npz"):
            mask = np.load(mask_path)
            if isinstance(mask, np.lib.npyio.NpzFile):
                mask = mask[list(mask.keys())[0]]
        elif mask_path.suffix in (".nii", ".gz"):
            import nibabel as nib
            mask = nib.load(mask_path).get_fdata() > 0
        else:
            raise ValueError(f"Unsupported mask format: {mask_path.suffix}")

        return cls(mask=mask, expected_dim=expected_dim, zscore=zscore)

    def extract(self, vol_3d: np.ndarray) -> np.ndarray:
        """Extract masked voxels from a single 3D volume.

        Parameters
        ----------
        vol_3d : np.ndarray
            3D volume of shape matching self.mask.shape (e.g., 81x104x83).

        Returns
        -------
        np.ndarray
            Flat float32 vector of shape (n_voxels,).
        """
        if vol_3d.shape != self.mask.shape:
            raise ValueError(
                f"Volume shape {vol_3d.shape} does not match "
                f"mask shape {self.mask.shape}"
            )

        voxels = vol_3d[self.mask].astype(np.float32)

        if self.zscore:
            mu = voxels.mean()
            std = voxels.std()
            if std > 1e-8:
                voxels = (voxels - mu) / std
            else:
                voxels = voxels - mu

        return voxels

    def extract_batch_from_nifti(
        self,
        nifti_path: str,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Extract voxels for multiple volumes from a 4D NIfTI file.

        Parameters
        ----------
        nifti_path : str
            Path to 4D NIfTI file (e.g., 81x104x83x720).
        indices : sequence of int
            Volume indices to extract.

        Returns
        -------
        np.ndarray
            Array of shape (len(indices), n_voxels).
        """
        import nibabel as nib

        img = nib.load(nifti_path)
        data = img.get_fdata()

        if data.ndim != 4:
            raise ValueError(
                f"Expected 4D NIfTI, got shape {data.shape}"
            )

        result = np.empty((len(indices), self.n_voxels), dtype=np.float32)
        for i, idx in enumerate(indices):
            vol_3d = data[..., int(idx)]
            result[i] = self.extract(vol_3d)

        return result

    def extract_all_from_nifti(self, nifti_path: str) -> np.ndarray:
        """Extract voxels for ALL volumes from a 4D NIfTI file.

        Returns
        -------
        np.ndarray
            Array of shape (n_volumes, n_voxels).
        """
        import nibabel as nib

        img = nib.load(nifti_path)
        data = img.get_fdata()

        if data.ndim != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape {data.shape}")

        n_volumes = data.shape[-1]
        indices = list(range(n_volumes))
        return self.extract_batch_from_nifti(nifti_path, indices)

    @property
    def shape(self) -> tuple:
        """Mask spatial shape."""
        return self.mask.shape

    def summary(self) -> dict:
        """Return extractor metadata."""
        return {
            "mask_shape": list(self.mask.shape),
            "n_voxels": self.n_voxels,
            "zscore": self.zscore,
        }
