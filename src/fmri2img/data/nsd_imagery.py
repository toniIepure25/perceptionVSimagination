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
from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, Iterator, Literal, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def _coerce_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


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
    beta_path: str = ""
    beta_index: int = 0
    roi_mask_path: Optional[str] = None
    coco_id: Optional[int] = None
    
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
        if "nsd_id" not in df.columns and "nsdId" in df.columns:
            df["nsd_id"] = df["nsdId"]
        if "nsdId" not in df.columns and "nsd_id" in df.columns:
            df["nsdId"] = df["nsd_id"]
        if "pair_id" not in df.columns and "nsdId" in df.columns:
            df["pair_id"] = df["nsdId"]
        
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
                if "fmri_path" not in row.index or pd.isna(row.get("fmri_path")):
                    raise NotImplementedError(
                        "NSDImageryDataset iteration for beta_path-only indices is not yet implemented.\n"
                        "TODO: rebuild the imagery index with scripts/build_nsd_imagery_index.py so it emits canonical fmri_path fields."
                    )

                # Load fMRI data
                fmri_path = Path(row['fmri_path'])
                if not fmri_path.is_absolute():
                    fmri_path = cache_root / fmri_path
                if not fmri_path.exists():
                    # Try resolving relative to the index location as a compatibility fallback.
                    fmri_path = Path(self.index_path).resolve().parent / Path(row['fmri_path'])
                
                if fmri_path.suffix == '.npy':
                    voxels = np.load(fmri_path).astype(np.float32)
                elif fmri_path.suffix == '.gz' or fmri_path.suffix == '.nii':
                    # Load NIfTI
                    import nibabel as nib
                    img = nib.load(fmri_path)
                    data = img.get_fdata().astype(np.float32)
                    beta_index = int(row.get("beta_index", 0))
                    if data.ndim == 4:
                        voxels = data[..., beta_index].flatten()
                    else:
                        voxels = data.flatten()
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
                    'nsd_id': row.get('nsdId'),
                    'nsdId': row.get('nsdId'),
                    'pair_id': row.get('pair_id'),
                    'vividness': row.get('vividness'),
                    'confidence': row.get('confidence'),
                    'split': row.get('split', 'unknown'),
                }
                
                # Add CLIP embedding if clip_cache is available
                if self.clip_cache is not None and 'nsdId' in row:
                    try:
                        clip_emb = self.clip_cache.get([int(row['nsdId'])]).get(int(row['nsdId']))
                        if clip_emb is not None:
                            sample['clip_target'] = clip_emb
                    except Exception:
                        pass  # CLIP embedding not available for this trial
                
                yield sample
                
            except NotImplementedError:
                raise
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
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
    *,
    data_root: Optional[Path] = None,
    subject: str,
    cache_root: Path,
    output_path: Path,
    stimulus_root: Optional[Path] = None,
    metadata_root: Optional[Path] = None,
    beta_root: Optional[Path] = None,
    beta_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
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
    from collections import defaultdict

    from fmri2img.data.nsd_imagery_metadata import assign_splits, parse_all_trials, trials_to_dataframe

    def _detect_condition(path: Path) -> str:
        lowered_parts = [part.lower() for part in path.parts]
        if "perception" in lowered_parts:
            return "perception"
        if "imagery" in lowered_parts:
            return "imagery"
        return "imagery"

    def _build_relative(path: Path) -> str:
        # Store absolute paths for robustness. Older code often resolved relative
        # imagery paths against cache roots that did not mirror the raw dataset.
        return str(path.resolve())

    def _maybe_int(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _infer_split(df: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(42)
        if "run_id" in df.columns and df["run_id"].nunique() > 3:
            unique_runs = sorted(df["run_id"].unique())
            n_runs = len(unique_runs)
            n_train = max(1, int(n_runs * 0.8))
            n_val = max(1, int(n_runs * 0.1))
            train_runs = unique_runs[:n_train]
            val_runs = unique_runs[n_train:n_train + n_val]
            test_runs = unique_runs[n_train + n_val:]
            df.loc[df["run_id"].isin(train_runs), "split"] = "train"
            df.loc[df["run_id"].isin(val_runs), "split"] = "val"
            df.loc[df["run_id"].isin(test_runs), "split"] = "test"
            return df

        split_key = "pair_id" if "pair_id" in df.columns and df["pair_id"].notna().any() else "trial_id"
        unique_ids = df[split_key].fillna(df["trial_id"]).drop_duplicates().to_numpy().copy()
        np.random.shuffle(unique_ids)
        n_total = len(unique_ids)
        n_train = max(1, int(n_total * 0.8))
        n_val = max(1, int(n_total * 0.1))
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train:n_train + n_val])
        test_ids = set(unique_ids[n_train + n_val:])
        values = df[split_key].fillna(df["trial_id"])
        df.loc[values.isin(train_ids), "split"] = "train"
        df.loc[values.isin(val_ids), "split"] = "val"
        df.loc[values.isin(test_ids), "split"] = "test"
        return df

    def _discover_metadata_dir(data_root_path: Path | None, metadata_root_path: Path | None) -> Path | None:
        candidates: list[Path] = []
        for base in (metadata_root_path, data_root_path):
            if base is None:
                continue
            candidates.extend([base, base / "metadata"])
        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "designmatrixGLMsingle.mat").exists()
                or (candidate / "cue_pair_list.xlsx").exists()
                or any(candidate.glob("*_pair_list.mat"))
            ):
                return candidate
        return None

    def _discover_beta_path(
        *,
        subject: str,
        data_root_path: Path | None,
        beta_root_path: Path | None,
        explicit_beta_path: Path | None,
    ) -> Path | None:
        candidates: list[Path] = []
        if explicit_beta_path is not None:
            candidates.append(explicit_beta_path)
        if beta_root_path is not None:
            if beta_root_path.is_file():
                candidates.append(beta_root_path)
            else:
                candidates.extend(
                    [
                        beta_root_path / subject / "betas_nsdimagery.nii.gz",
                        beta_root_path / subject / "betas_nsdimagery.nii",
                    ]
                )
                candidates.extend(sorted((beta_root_path / subject).glob("**/betas_nsdimagery.nii*")))
        if data_root_path is not None:
            candidates.extend(
                [
                    data_root_path / "betas" / subject / "betas_nsdimagery.nii.gz",
                    data_root_path / "betas" / subject / "betas_nsdimagery.nii",
                    data_root_path / subject / "betas_nsdimagery.nii.gz",
                    data_root_path / subject / "betas_nsdimagery.nii",
                ]
            )
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return None

    if data_root is None and metadata_root is None and beta_root is None and beta_path is None:
        raise NotImplementedError(
            f"build_nsd_imagery_index for {subject} is not yet implemented without an explicit data_root.\n"
            "TODO: pass a canonical NSD-Imagery data root or provide metadata_root/beta_root so the builder "
            "can discover real trials."
        )

    data_root = _coerce_optional_path(data_root)
    metadata_root = _coerce_optional_path(metadata_root)
    beta_root = _coerce_optional_path(beta_root)
    beta_path = _coerce_optional_path(beta_path)
    stimulus_root = _coerce_optional_path(stimulus_root)
    cache_root = Path(cache_root)
    output_path = Path(output_path)
    report_path = _coerce_optional_path(report_path)
    
    if verbose:
        print(f"Building NSD-Imagery index for {subject}...")
        if data_root is not None:
            print(f"  Data root: {data_root}")
        if metadata_root is not None:
            print(f"  Metadata root: {metadata_root}")
        if beta_root is not None:
            print(f"  Beta root: {beta_root}")
        if beta_path is not None:
            print(f"  Beta path: {beta_path}")
        print(f"  Cache root: {cache_root}")
    
    layout_summary: dict[str, Any] = {"subject": subject}

    # Discover fMRI files for this subject
    subject_data_dir = data_root / subject if data_root is not None else None
    subject_root_available = subject_data_dir is not None and subject_data_dir.exists()
    beta_files = []
    metadata_files = []
    if subject_root_available:
        beta_files = list(subject_data_dir.glob("**/*.npy")) + list(subject_data_dir.glob("**/*.nii.gz"))
        metadata_files = sorted(subject_data_dir.glob("**/metadata.json"))

    # Build trial records
    trials = []
    trial_id_counter = 0
    stimulus_stats = defaultdict(int)

    metadata_used = False
    if metadata_files:
        layout_summary["layout"] = "subject_rooted"
        layout_summary["data_root"] = str(data_root) if data_root is not None else None
        if verbose:
            print(f"  Found {len(beta_files)} beta files in subject-rooted layout")
        for metadata_file in metadata_files:
            with open(metadata_file) as f:
                metadata = json.load(f)
            trials_meta = metadata.get("trials", metadata if isinstance(metadata, list) else [])
            if not isinstance(trials_meta, list):
                continue
            beta_dir = metadata_file.parent
            run_match = re.search(r"(?:run|session)_?(\d+)", beta_dir.name, re.IGNORECASE)
            run_id = int(run_match.group(1)) if run_match else -1
            condition = _detect_condition(metadata_file)
            directory_beta_files = sorted(
                list(beta_dir.glob("*.npy")) + list(beta_dir.glob("*.nii.gz"))
            )
            for trial_meta in trials_meta:
                trial_identifier = str(trial_meta.get("trial_id", f"trial_{trial_id_counter:05d}"))
                matching_beta = None
                for beta_file in directory_beta_files:
                    if trial_identifier in beta_file.name:
                        matching_beta = beta_file
                        break
                if matching_beta is None and directory_beta_files:
                    matching_beta = directory_beta_files[min(trial_id_counter, len(directory_beta_files) - 1)]
                if matching_beta is None:
                    continue
                image_path = trial_meta.get("image_path")
                if image_path is not None:
                    image_path = str((beta_dir / image_path).resolve()) if not Path(image_path).is_absolute() else image_path
                elif stimulus_root is not None:
                    possible = stimulus_root / f"{trial_identifier}.png"
                    if possible.exists():
                        image_path = str(possible)
                if image_path is None:
                    sibling_candidates = (
                        sorted(beta_dir.glob(f"{trial_identifier}*image.png"))
                        + sorted(beta_dir.glob(f"{trial_identifier}*.png"))
                        + sorted(beta_dir.glob(f"{trial_identifier}*image.jpg"))
                        + sorted(beta_dir.glob(f"{trial_identifier}*.jpg"))
                    )
                    for candidate in sibling_candidates:
                        if candidate.is_file():
                            image_path = str(candidate.resolve())
                            break
                stimulus_type = trial_meta.get("stimulus_type", "unknown")
                if image_path and stimulus_type == "unknown":
                    stimulus_type = "complex"
                if trial_meta.get("text_prompt") and stimulus_type == "unknown":
                    stimulus_type = "conceptual"

                nsd_id = _maybe_int(trial_meta.get("nsd_id", trial_meta.get("nsdId")))
                pair_id = _maybe_int(trial_meta.get("pair_id"))
                if pair_id is None and nsd_id is not None:
                    pair_id = nsd_id

                trials.append(
                    {
                        "trial_id": trial_id_counter,
                        "subject": subject,
                        "condition": condition,
                        "stimulus_type": stimulus_type,
                        "run_id": _maybe_int(trial_meta.get("run_id")) or run_id,
                        "fmri_path": _build_relative(matching_beta),
                        "image_path": image_path,
                        "text_prompt": trial_meta.get("text_prompt"),
                        "meta_json": json.dumps(trial_meta) if trial_meta else None,
                        "split": None,
                        "nsd_id": nsd_id,
                        "nsdId": nsd_id,
                        "pair_id": pair_id,
                        "vividness": trial_meta.get("vividness"),
                        "confidence": trial_meta.get("confidence"),
                        "beta_index": _maybe_int(trial_meta.get("beta_index")) or 0,
                    }
                )
                stimulus_stats[stimulus_type] += 1
                trial_id_counter += 1
            metadata_used = True

    if not metadata_used:
        metadata_dir = _discover_metadata_dir(data_root, metadata_root)
        split_beta_path = _discover_beta_path(
            subject=subject,
            data_root_path=data_root,
            beta_root_path=beta_root,
            explicit_beta_path=beta_path,
        )
        if metadata_dir is not None and split_beta_path is not None:
            layout_summary["layout"] = "split_metadata_beta"
            layout_summary["metadata_root"] = str(metadata_dir.resolve())
            layout_summary["beta_path"] = str(split_beta_path.resolve())
            trials_df = trials_to_dataframe(
                parse_all_trials(metadata_dir, stimulus_root=stimulus_root),
                subject=subject,
                beta_path=str(split_beta_path.resolve()),
            )
            df = assign_splits(trials_df)
            for stimulus_type, count in df["stimulus_type"].value_counts().to_dict().items():
                stimulus_stats[stimulus_type] = int(count)
            metadata_used = True
        elif subject_root_available:
            if not beta_files:
                raise FileNotFoundError(
                    f"No beta files (*.npy or *.nii.gz) found in {subject_data_dir}"
                )
        else:
            details = []
            if data_root is not None:
                details.append(f"data_root={data_root}")
            if metadata_root is not None:
                details.append(f"metadata_root={metadata_root}")
            if beta_root is not None:
                details.append(f"beta_root={beta_root}")
            if beta_path is not None:
                details.append(f"beta_path={beta_path}")
            raise FileNotFoundError(
                f"Could not discover a supported NSD-Imagery layout for {subject}. "
                f"Tried subject-rooted and split metadata/beta layouts with: {', '.join(details) or 'no paths'}."
            )

    if not metadata_used:
        for beta_idx, beta_file in enumerate(sorted(beta_files)):
            rel_path = _build_relative(beta_file)
            filename = beta_file.stem
            run_match = re.search(r'(?:run|r|session)_?(\d+)', filename, re.IGNORECASE)
            run_id = int(run_match.group(1)) if run_match else -1
            condition = _detect_condition(beta_file)
            stimulus_type = "unknown"
            image_path = None
            text_prompt = None
            if stimulus_root:
                potential_image = stimulus_root / f"{filename}.png"
                if potential_image.exists():
                    image_path = str(potential_image)
                    stimulus_type = "complex"
            trials.append(
                {
                    "trial_id": trial_id_counter,
                    "subject": subject,
                    "condition": condition,
                    "stimulus_type": stimulus_type,
                    "run_id": run_id,
                    "fmri_path": rel_path,
                    "image_path": image_path,
                    "text_prompt": text_prompt,
                    "meta_json": None,
                    "split": None,
                    "nsd_id": None,
                    "nsdId": None,
                    "pair_id": None,
                    "vividness": None,
                    "confidence": None,
                    "beta_index": 0,
                }
            )
            stimulus_stats[stimulus_type] += 1
            trial_id_counter += 1

    # Convert to DataFrame
    if not metadata_used or trials:
        df = pd.DataFrame(trials)
        df = _infer_split(df)
    layout_summary.setdefault("layout", "beta_only_fallback")
    layout_summary["output_path"] = str(output_path)
    layout_summary["rows"] = int(len(df))
    if "condition" in df.columns:
        layout_summary["conditions"] = df["condition"].value_counts().to_dict()
    if "stimulus_set" in df.columns:
        layout_summary["stimulus_sets"] = df["stimulus_set"].value_counts().to_dict()
    if "nsdId" in df.columns:
        layout_summary["rows_with_nsd_id"] = int(df["nsdId"].notna().sum())
        layout_summary["unique_nsd_ids"] = int(df["nsdId"].dropna().nunique())
    
    if verbose or dry_run:
        print(f"\n=== Index Summary ===")
        print(f"Total trials: {len(df)}")
        print(f"\nBy stimulus type:")
        for stype, count in sorted(stimulus_stats.items()):
            print(f"  {stype}: {count}")
        print(f"\nBy split:")
        for split in ['train', 'val', 'test']:
            count = (df['split'] == split).sum()
            print(f"  {split}: {count}")
        print(f"\nWith images: {df['image_path'].notna().sum()}")
        print(f"With text: {df['text_prompt'].notna().sum()}")
        if "nsdId" in df.columns:
            paired = df["pair_id"].notna().sum()
            print(f"With pair ids: {paired}")
    
    if dry_run:
        if verbose:
            print(f"\n[DRY RUN] Would write to: {output_path}")
        return output_path
    
    # Write to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as handle:
            json.dump(layout_summary, handle, indent=2)
    
    if verbose:
        print(f"\n✓ Index saved to: {output_path}")
        if report_path is not None:
            print(f"✓ Layout report saved to: {report_path}")
    
    return output_path


# Export public API
__all__ = [
    'ImageryTrial',
    'NSDImageryDataset',
    'build_nsd_imagery_index',
]
