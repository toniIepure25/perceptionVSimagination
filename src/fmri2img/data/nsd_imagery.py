"""
NSD-Imagery Dataset Module

This module provides dataset classes and utilities for working with the NSD-Imagery
dataset, which contains fMRI recordings from subjects mentally imagining visual stimuli.

Components:
- ImageryTrial: Dataclass for canonical trial representation
- NSDImageryDataset: PyTorch Dataset for imagery data
- build_nsd_imagery_index: Function to construct Parquet indices from raw imagery data
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class ImageryTrial:
    """
    Canonical representation of an NSD-Imagery trial.
    
    This dataclass defines the standard structure for imagery trials,
    ensuring consistency across the pipeline.
    
    Attributes:
        trial_id: Global unique trial identifier
        subject: Subject ID (e.g., "subj01")
        session: Session number
        trial_in_session: Trial index within session
        condition: Data source - "perception" or "imagery"
        nsd_id: NSD stimulus ID (matches perception dataset)
        coco_id: COCO dataset ID (if applicable)
        beta_path: Relative path to NIfTI file from cache root
        beta_index: Volume index within NIfTI file
        repeat_index: 0 for first presentation, 1+ for repeats
        caption: Optional image caption
        run_number: Optional run number within session
        is_valid: Quality control flag
        snr_estimate: Optional signal-to-noise ratio estimate
    """
    
    # Unique identifiers
    trial_id: int
    subject: str
    session: int
    trial_in_session: int
    
    # Condition and stimulus
    condition: Literal["perception", "imagery"]
    nsd_id: int
    
    # fMRI data
    beta_path: str
    beta_index: int
    
    # Optional fields (all have defaults)
    coco_id: Optional[int] = None
    roi_mask_path: Optional[str] = None
    
    # Optional metadata
    repeat_index: int = 0
    caption: Optional[str] = None
    run_number: Optional[int] = None
    
    # Quality control
    is_valid: bool = True
    snr_estimate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'trial_id': self.trial_id,
            'subject': self.subject,
            'session': self.session,
            'trial_in_session': self.trial_in_session,
            'condition': self.condition,
            'nsd_id': self.nsd_id,
            'coco_id': self.coco_id,
            'beta_path': self.beta_path,
            'beta_index': self.beta_index,
            'roi_mask_path': self.roi_mask_path,
            'repeat_index': self.repeat_index,
            'caption': self.caption,
            'run_number': self.run_number,
            'is_valid': self.is_valid,
            'snr_estimate': self.snr_estimate,
        }


class NSDImageryDataset(IterableDataset):
    """
    PyTorch Dataset for NSD-Imagery data.
    
    This dataset mirrors the interface of NSDIterableDataset but is specialized
    for imagery data. It supports the same preprocessing, caching, and batching
    operations as the perception dataset.
    
    Usage:
        >>> from fmri2img.data.nsd_imagery import NSDImageryDataset
        >>> dataset = NSDImageryDataset(
        ...     index_path="cache/indices/imagery/subj01.parquet",
        ...     subject="subj01",
        ...     preprocessor=preprocessor,
        ...     clip_cache=clip_cache
        ... )
        >>> for sample in dataset:
        ...     fmri = sample["fmri"]  # (n_voxels,) array
        ...     nsd_id = sample["nsdId"]  # int
        ...     # ... process sample
    
    Args:
        index_path: Path to Parquet index file
        subject: Subject ID (e.g., "subj01")
        condition: Filter by condition ("perception", "imagery", or None for both)
        session: Optional session filter
        shuffle: Whether to shuffle trials
        limit: Optional limit on number of trials
        seed: Random seed for shuffling
        preprocessor: Optional NSDPreprocessor for transformations
        clip_cache: Optional CLIPCache instance or path
    """
    
    def __init__(
        self,
        index_path: str,
        subject: str = "subj01",
        condition: Optional[Literal["perception", "imagery"]] = "imagery",
        session: Optional[int] = None,
        shuffle: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
        preprocessor: Optional["NSDPreprocessor"] = None,
        clip_cache: Optional["CLIPCache"] = None,
        cache_root: Optional[str] = None,
        data_root: Optional[str] = None,
        stimulus_type_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ):
        super().__init__()
        self.index_path = index_path
        self.subject = subject
        self.condition = condition
        self.session = session
        self.shuffle = shuffle
        self.limit = limit
        self.seed = seed
        self.preprocessor = preprocessor
        self.clip_cache = clip_cache
        self._cache_root = Path(cache_root) if cache_root else Path('cache')
        self._data_root = Path(data_root) if data_root else None
        
        # Validate index exists
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Imagery index not found: {index_path}\n"
                f"Please run: python scripts/build_nsd_imagery_index.py --subject {subject}"
            )
        
        # Load and filter index
        self.df = self._load_index(stimulus_type_filter, split_filter)
    
    def _load_index(self, stimulus_type_filter: Optional[str] = None, 
                    split_filter: Optional[str] = None) -> pd.DataFrame:
        """Load and filter the index based on parameters."""
        df = pd.read_parquet(self.index_path)
        
        # Filter by subject
        df = df[df["subject"] == self.subject]
        
        # Filter by condition if specified
        if self.condition is not None:
            df = df[df["condition"] == self.condition]
        
        # Filter by session if specified
        if self.session is not None and "session" in df.columns:
            df = df[df["session"] == self.session]
        
        # Filter by stimulus type if specified
        if stimulus_type_filter is not None:
            df = df[df["stimulus_type"] == stimulus_type_filter]
        
        # Filter by split if specified
        if split_filter is not None:
            df = df[df["split"] == split_filter]
        
        return df.reset_index(drop=True)
    
    def __len__(self) -> int:
        """Return number of trials in dataset."""
        if self.limit is not None:
            return min(len(self.df), self.limit)
        return len(self.df)
    
    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over dataset samples.
        
        Yields dictionaries with:
            - voxels: np.ndarray (float32) - fMRI data
            - condition: str - "imagery" or "perception"
            - stimulus_type: str - stimulus category
            - target_image: PIL.Image or None - target image if available
            - target_text: str or None - text prompt if available
            - meta: dict - additional metadata
            - trial_id: int - unique trial identifier
        """
        from PIL import Image
        from torch.utils.data import get_worker_info
        import random
        
        # File caches to avoid reloading 4D NIfTI on every trial
        _nifti_cache = {}  # path → nibabel image
        _hdf5_cache = {}   # path → h5py File
        
        # Worker-aware sharding
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))

        indices = list(range(len(self.df)))
        if self.shuffle:
            rng.shuffle(indices)

        if self.limit is not None:
            indices = indices[: self.limit]

        # Simple round-robin sharding across workers
        if info and info.num_workers > 1:
            indices = indices[info.id :: info.num_workers]
        
        # Get data root: prefer explicit data_root, then cache_root, then cwd
        data_root = getattr(self, '_data_root', None)
        cache_root = getattr(self, '_cache_root', Path('cache'))
        
        def _resolve_path(rel_path: str) -> Path:
            """Resolve a relative path from the index against known roots."""
            if data_root is not None:
                p = data_root / rel_path
                if p.exists():
                    return p
            p = cache_root / rel_path
            if p.exists():
                return p
            p = Path(rel_path)
            if p.exists():
                return p
            # Return data_root version as best guess even if missing
            return (data_root / rel_path) if data_root else Path(rel_path)
        
        for idx in indices:
            row = self.df.iloc[idx]
            
            try:
                # Load fMRI data
                fmri_path = _resolve_path(row['fmri_path'])
                beta_idx = row.get('beta_index', None)
                
                if fmri_path.suffix == '.npy':
                    vol_3d = np.load(fmri_path).astype(np.float32)
                elif fmri_path.suffix == '.gz':
                    # Load NIfTI (may be 4D with multiple volumes)
                    import nibabel as nib
                    fmri_key = str(fmri_path)
                    if fmri_key not in _nifti_cache:
                        _nifti_cache[fmri_key] = nib.load(fmri_path)
                    img = _nifti_cache[fmri_key]
                    data = img.get_fdata()
                    
                    if data.ndim == 4 and beta_idx is not None:
                        # Extract single 3D volume from 4D NIfTI
                        vol_3d = data[..., int(beta_idx)].astype(np.float32)
                    else:
                        # Already 3D or no specific beta index
                        vol_3d = data.astype(np.float32)
                elif fmri_path.suffix == '.hdf5':
                    import h5py
                    fmri_key = str(fmri_path)
                    if fmri_key not in _hdf5_cache:
                        _hdf5_cache[fmri_key] = h5py.File(fmri_path, 'r')
                    f = _hdf5_cache[fmri_key]
                    if beta_idx is not None:
                        vol_3d = f['betas'][int(beta_idx)].astype(np.float32)
                    else:
                        vol_3d = f['betas'][:].astype(np.float32)
                else:
                    raise ValueError(f"Unsupported fMRI file format: {fmri_path.suffix}")
                
                # Apply preprocessing if provided (expects 3D volume)
                if self.preprocessor is not None:
                    voxels = self.preprocessor.transform(vol_3d)
                else:
                    voxels = vol_3d.flatten()
                
                # Load target image if available
                target_image = None
                if pd.notna(row.get('image_path')):
                    img_path = _resolve_path(row['image_path'])
                    if img_path.exists():
                        target_image = Image.open(img_path).convert('RGB')
                
                # Get text prompt if available
                target_text = None
                if pd.notna(row.get('text_prompt')):
                    target_text = row['text_prompt']
                
                # Parse metadata
                meta = {}
                if pd.notna(row.get('meta_json')):
                    import json
                    meta = json.loads(row['meta_json'])
                
                # Build sample dict
                sample = {
                    'voxels': voxels,
                    'condition': row['condition'],
                    'stimulus_type': row['stimulus_type'],
                    'target_image': target_image,
                    'target_text': target_text,
                    'meta': meta,
                    'trial_id': row['trial_id'],
                    'split': row.get('split', 'unknown'),
                }
                
                # Add CLIP embedding if clip_cache is available
                nsd_id = row.get('nsd_id', row.get('nsdId', None))
                if self.clip_cache is not None and nsd_id is not None:
                    try:
                        clip_emb = self.clip_cache.get_embedding(int(nsd_id))
                        sample['clip_target'] = clip_emb
                        sample['nsd_id'] = int(nsd_id)
                    except Exception:
                        pass
                
                yield sample
                
            except Exception as e:
                logger.warning(f"Failed to load trial {row.get('trial_id', idx)}: {e}")
                continue
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample by index.
        
        TODO: Implement indexing support if needed for non-iterable use cases.
        """
        raise NotImplementedError(
            "NSDImageryDataset does not support indexing. Use iteration instead.\n"
            "If indexing is required, consider implementing a map-style Dataset variant."
        )


def build_nsd_imagery_index(
    data_root: Path,
    subject: str,
    cache_root: Path,
    output_path: Path,
    stimulus_root: Optional[Path] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Build canonical Parquet index from raw NSD-Imagery data.
    
    Parses the GLMsingle design matrices and experiment metadata to create
    a trial-level index for all 720 betas per subject. Each beta is labeled
    with its task type (imagery/perception/attention), stimulus identity,
    cue letter, NSD ID (for set B), and stimulus type.
    
    Args:
        data_root: Root directory of downloaded NSD-Imagery data.
            Expected structure:
                data_root/
                    betas/{subject}/betas_nsdimagery.nii.gz  (4D: 81×104×83×720)
                    metadata/designmatrixGLMsingle.mat
                    metadata/cue_pair_list.xlsx
                    metadata/*_dm.mat
                    stimuli/allstim/  (optional cue images)
        subject: Subject ID (e.g., "subj01")
        cache_root: Cache directory (unused, kept for API compat)
        output_path: Path to write Parquet index
        stimulus_root: Optional override for stimulus directory
        dry_run: If True, validate but don't write output
        verbose: If True, print progress messages
    
    Returns:
        Path to created index file
        
    Index Schema:
        - trial_id: int (= beta_index, 0-719)
        - subject: str
        - condition: str ("imagery", "perception", or "attention")
        - stimulus_type: str ("simple", "complex", "conceptual")
        - task_type: str ("imagery", "perception", "attention")
        - run_id: int (0-11)
        - beta_index: int (volume index in 4D NIfTI)
        - fmri_path: str (relative path to beta file from data_root)
        - image_path: str (nullable, path to cue/stimulus image)
        - text_prompt: str (nullable, for conceptual stimuli)
        - nsd_id: int (nullable, NSD stimulus ID for set B)
        - cue_letter: str (cue letter shown to subject)
        - stimulus_set: str ("A", "B", "C")
        - stimulus_name: str (full stimulus name)
        - shared_id: int (nullable, shared1000 index for set B)
        - repeat_index: int (0 or 1)
        - meta_json: str (nullable, serialized metadata dict)
        - split: str ("train", "val", "test")
    """
    from fmri2img.data.nsd_imagery_metadata import (
        parse_all_trials,
        trials_to_dataframe,
        assign_splits,
    )
    
    data_root = Path(data_root)
    output_path = Path(output_path)
    
    if verbose:
        logger.info("Building NSD-Imagery index for %s...", subject)
        logger.info("  Data root: %s", data_root)
    
    # Validate data structure
    metadata_dir = data_root / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_dir}\n"
            f"Expected: {data_root}/metadata/designmatrixGLMsingle.mat"
        )
    
    beta_file = data_root / "betas" / subject / "betas_nsdimagery.nii.gz"
    if not beta_file.exists():
        # Also check for HDF5
        beta_file_hdf5 = data_root / "betas" / subject / "betas_nsdimagery.hdf5"
        if beta_file_hdf5.exists():
            beta_file = beta_file_hdf5
        else:
            raise FileNotFoundError(
                f"Beta file not found for {subject}.\n"
                f"Expected: {beta_file}\n"
                f"Available subjects: {[d.name for d in (data_root / 'betas').iterdir() if d.is_dir()]}"
            )
    
    # Verify 4D shape
    if beta_file.suffix == '.gz':
        import nibabel as nib
        img = nib.load(str(beta_file))
        shape = img.shape
        if len(shape) != 4:
            raise ValueError(
                f"Expected 4D NIfTI but got shape {shape}: {beta_file}"
            )
        n_volumes = shape[3]
        if verbose:
            logger.info("  Beta file: %s (shape=%s, %d volumes)", 
                       beta_file.name, shape, n_volumes)
    elif beta_file.suffix == '.hdf5':
        import h5py
        with h5py.File(str(beta_file), 'r') as f:
            shape = f['betas'].shape
        n_volumes = shape[0]  # HDF5 is (volumes, z, y, x)
        if verbose:
            logger.info("  Beta file: %s (shape=%s, %d volumes)",
                       beta_file.name, shape, n_volumes)
    
    # Resolve stimulus directory
    if stimulus_root is None:
        stimulus_root = data_root / "stimuli"
    stimulus_root = Path(stimulus_root) if stimulus_root else None
    
    # Parse all trial metadata
    if verbose:
        logger.info("  Parsing experiment metadata...")
    
    trials = parse_all_trials(
        metadata_dir=metadata_dir,
        stimulus_root=stimulus_root,
    )
    
    if len(trials) != n_volumes:
        logger.warning(
            "Trial count (%d) != volume count (%d). "
            "Index will use trial count.", len(trials), n_volumes
        )
    
    # Build relative path for fmri_path column
    beta_rel_path = str(beta_file.relative_to(data_root))
    
    # Convert to DataFrame
    df = trials_to_dataframe(trials, subject=subject, beta_path=beta_rel_path)
    
    # Assign splits
    df = assign_splits(df, seed=42)
    
    # Print summary
    if verbose or dry_run:
        logger.info("=== Index Summary ===")
        logger.info("Total trials: %d", len(df))
        logger.info("By task type:")
        for task, count in df['task_type'].value_counts().items():
            logger.info("  %s: %d", task, count)
        logger.info("By stimulus type:")
        for stype, count in df['stimulus_type'].value_counts().items():
            logger.info("  %s: %d", stype, count)
        logger.info("By stimulus set:")
        for sset, count in df['stimulus_set'].value_counts().items():
            logger.info("  Set %s: %d", sset, count)
        logger.info("By split:")
        for split in ['train', 'val', 'test']:
            count = (df['split'] == split).sum()
            logger.info("  %s: %d", split, count)
        logger.info("With NSD IDs: %d", df['nsd_id'].notna().sum())
        logger.info("With images: %d", df['image_path'].notna().sum())
        
        # Print imagery-specific stats
        imagery_df = df[df['condition'] == 'imagery']
        perception_df = df[df['condition'] == 'perception']
        logger.info("Imagery trials: %d", len(imagery_df))
        logger.info("Perception trials: %d", len(perception_df))
        logger.info("NSD-linked imagery (set B): %d",
                    imagery_df['nsd_id'].notna().sum())
    
    if dry_run:
        if verbose:
            logger.info("[DRY RUN] Would write to: %s", output_path)
        return output_path
    
    # Write to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    if verbose:
        logger.info("Index saved to: %s", output_path)
    
    return output_path


# Export public API
__all__ = [
    'ImageryTrial',
    'NSDImageryDataset',
    'build_nsd_imagery_index',
]

# Also make metadata parser accessible
try:
    from fmri2img.data.nsd_imagery_metadata import (
        parse_all_trials,
        parse_cue_pair_list,
        trials_to_dataframe,
        assign_splits,
        TrialInfo,
        RUN_INFO,
    )
    __all__.extend([
        'parse_all_trials',
        'parse_cue_pair_list',
        'trials_to_dataframe',
        'assign_splits',
        'TrialInfo',
        'RUN_INFO',
    ])
except ImportError:
    pass
