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
    coco_id: Optional[int] = None
    
    # fMRI data
    beta_path: str
    beta_index: int
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
        
        # Get cache root if available
        cache_root = getattr(self, '_cache_root', Path('cache'))
        
        for idx in indices:
            row = self.df.iloc[idx]
            
            try:
                # Load fMRI data
                fmri_path = cache_root / row['fmri_path']
                if not fmri_path.exists():
                    # Try without cache_root prefix
                    fmri_path = Path(row['fmri_path'])
                
                if fmri_path.suffix == '.npy':
                    voxels = np.load(fmri_path).astype(np.float32)
                elif fmri_path.suffix == '.gz':
                    # Load NIfTI
                    import nibabel as nib
                    img = nib.load(fmri_path)
                    voxels = img.get_fdata().astype(np.float32).flatten()
                else:
                    raise ValueError(f"Unsupported fMRI file format: {fmri_path.suffix}")
                
                # Apply preprocessing if provided
                if self.preprocessor is not None:
                    voxels = self.preprocessor.transform(voxels)
                
                # Load target image if available
                target_image = None
                if pd.notna(row.get('image_path')):
                    img_path = Path(row['image_path'])
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
    
    Args:
        data_root: Root directory containing imagery fMRI data
        subject: Subject ID (e.g., "subj01")
        cache_root: Cache directory for intermediate files
        output_path: Path to write Parquet index
        stimulus_root: Optional root for stimulus files (images/text)
        dry_run: If True, validate but don't write output
        verbose: If True, print progress messages
    
    Returns:
        Path to created index file
        
    Index Schema:
        - trial_id: int (unique global ID)
        - subject: str
        - condition: str ("imagery" or "perception")
        - stimulus_type: str ("simple", "complex", "conceptual", "unknown")
        - run_id: int (optional, for session/run organization)
        - fmri_path: str (relative path to beta file)
        - image_path: str (nullable, for complex stimuli)
        - text_prompt: str (nullable, for conceptual stimuli)
        - meta_json: str (nullable, serialized metadata dict)
        - split: str ("train", "val", "test")
    """
    import json
    from collections import defaultdict
    
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    output_path = Path(output_path)
    
    if verbose:
        logger.info("Building NSD-Imagery index for %s...", subject)
        logger.info("  Data root: %s", data_root)
        logger.info("  Cache root: %s", cache_root)
    
    # Discover fMRI files for this subject
    subject_data_dir = data_root / subject
    if not subject_data_dir.exists():
        raise FileNotFoundError(
            f"Subject data directory not found: {subject_data_dir}\n"
            f"Expected structure: {data_root}/{subject}/"
        )
    
    # Scan for beta files (npy or nii.gz)
    beta_files = list(subject_data_dir.glob("**/*.npy")) + \
                 list(subject_data_dir.glob("**/*.nii.gz"))
    
    if not beta_files:
        raise FileNotFoundError(
            f"No beta files (*.npy or *.nii.gz) found in {subject_data_dir}"
        )
    
    if verbose:
        logger.info("  Found %d beta files", len(beta_files))
    
    # Parse metadata if available
    metadata_file = subject_data_dir / "metadata.json"
    metadata_dict = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata_dict = json.load(f)
        if verbose:
            logger.info("  Loaded metadata from %s", metadata_file)
    
    # Build trial records
    trials = []
    trial_id_counter = 0
    stimulus_stats = defaultdict(int)
    
    for beta_idx, beta_file in enumerate(sorted(beta_files)):
        # Determine relative path from cache_root
        try:
            rel_path = beta_file.relative_to(data_root)
        except ValueError:
            rel_path = beta_file
        
        # Infer stimulus type from filename or metadata
        filename = beta_file.stem
        stimulus_type = "unknown"
        image_path = None
        text_prompt = None
        run_id = None
        
        # Try to extract run_id from filename patterns like "run01", "r01", etc.
        import re
        run_match = re.search(r'(?:run|r)_?(\d+)', filename, re.IGNORECASE)
        if run_match:
            run_id = int(run_match.group(1))
        
        # Check metadata for this trial
        trial_meta = metadata_dict.get(str(beta_idx), {})
        if 'stimulus_type' in trial_meta:
            stimulus_type = trial_meta['stimulus_type']
        if 'image_path' in trial_meta:
            image_path = trial_meta['image_path']
            stimulus_type = "complex"
        if 'text_prompt' in trial_meta:
            text_prompt = trial_meta['text_prompt']
            stimulus_type = "conceptual"
        if 'run_id' in trial_meta:
            run_id = trial_meta['run_id']
        
        # Infer from stimulus_root if provided
        if stimulus_root and not image_path:
            potential_image = stimulus_root / f"{filename}.png"
            if potential_image.exists():
                image_path = str(potential_image)
                stimulus_type = "complex"
            else:
                potential_image = stimulus_root / f"{filename}.jpg"
                if potential_image.exists():
                    image_path = str(potential_image)
                    stimulus_type = "complex"
        
        # Extract nsd_id from metadata if available (critical for CLIP cache lookup)
        nsd_id = trial_meta.get('nsd_id', trial_meta.get('nsdId', None))
        coco_id = trial_meta.get('coco_id', trial_meta.get('cocoId', None))
        
        # Create trial record
        trial = {
            'trial_id': trial_id_counter,
            'subject': subject,
            'condition': 'imagery',
            'stimulus_type': stimulus_type,
            'run_id': run_id if run_id is not None else -1,
            'fmri_path': str(rel_path),
            'image_path': image_path,
            'text_prompt': text_prompt,
            'nsd_id': nsd_id,
            'coco_id': coco_id,
            'meta_json': json.dumps(trial_meta) if trial_meta else None,
            'split': None,  # Will be assigned below
        }
        
        trials.append(trial)
        stimulus_stats[stimulus_type] += 1
        trial_id_counter += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(trials)
    
    # Assign splits deterministically based on run_id or trial_id
    np.random.seed(42)
    if 'run_id' in df.columns and df['run_id'].nunique() > 3:
        # Split by runs if available
        unique_runs = sorted(df['run_id'].unique())
        n_runs = len(unique_runs)
        n_train = max(1, int(n_runs * 0.8))
        n_val = max(1, int(n_runs * 0.1))
        
        train_runs = unique_runs[:n_train]
        val_runs = unique_runs[n_train:n_train + n_val]
        test_runs = unique_runs[n_train + n_val:]
        
        df.loc[df['run_id'].isin(train_runs), 'split'] = 'train'
        df.loc[df['run_id'].isin(val_runs), 'split'] = 'val'
        df.loc[df['run_id'].isin(test_runs), 'split'] = 'test'
    else:
        # Split by trial_id
        shuffled_ids = df['trial_id'].values.copy()
        np.random.shuffle(shuffled_ids)
        
        n_total = len(shuffled_ids)
        n_train = max(1, int(n_total * 0.8))
        n_val = max(1, int(n_total * 0.1))
        
        train_ids = set(shuffled_ids[:n_train])
        val_ids = set(shuffled_ids[n_train:n_train + n_val])
        test_ids = set(shuffled_ids[n_train + n_val:])
        
        df.loc[df['trial_id'].isin(train_ids), 'split'] = 'train'
        df.loc[df['trial_id'].isin(val_ids), 'split'] = 'val'
        df.loc[df['trial_id'].isin(test_ids), 'split'] = 'test'
    
    if verbose or dry_run:
        logger.info("=== Index Summary ===")
        logger.info("Total trials: %d", len(df))
        logger.info("By stimulus type:")
        for stype, count in sorted(stimulus_stats.items()):
            logger.info("  %s: %d", stype, count)
        logger.info("By split:")
        for split in ['train', 'val', 'test']:
            count = (df['split'] == split).sum()
            logger.info("  %s: %d", split, count)
        logger.info("With images: %d", df['image_path'].notna().sum())
        logger.info("With text: %d", df['text_prompt'].notna().sum())
    
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
