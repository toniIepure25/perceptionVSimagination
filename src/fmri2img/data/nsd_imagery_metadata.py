"""
NSD-Imagery Metadata Parser

Parses the NSD-Imagery experiment design matrices and metadata files
to produce a complete trial-level mapping for all 720 betas per subject.

The NSD-Imagery experiment has:
- 3 stimulus sets: A (oriented bars/crosses), B (NSD shared images), C (verbal concepts)
- 18 unique stimuli (6 per set), each paired with a cue letter
- 3 task types per set: imagery (img), perception (vis), attention (att)
- 12 runs total: 3 sessions × (img_1, att, vis) + 3 second imagery (img_2)
- 720 single-trial betas per subject (GLMsingle)

Key metadata files (in metadata/ directory):
- designmatrixGLMsingle.mat: trial onset → condition mapping for all 720 betas
- cue_pair_list.xlsx: cue letter → stimulus identity mapping
- {task}{set}_{rep}_dm.mat: individual design matrices per run type
- {set}_pair_list.mat: per-set stimulus pairing information

Run ordering (0-indexed):
  Run  0: imgA_1 (Set A imagery, first presentation)
  Run  1: attA   (Set A attention)
  Run  2: visA   (Set A visual perception)
  Run  3: imgB_1 (Set B imagery, first presentation)
  Run  4: attB   (Set B attention)
  Run  5: visB   (Set B visual perception)
  Run  6: imgC_1 (Set C imagery, first presentation)
  Run  7: attC   (Set C attention)
  Run  8: visC   (Set C visual perception)
  Run  9: imgA_2 (Set A imagery, second presentation)
  Run 10: imgB_2 (Set B imagery, second presentation)
  Run 11: imgC_2 (Set C imagery, second presentation)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Run → (task_type, stimulus_set, dm_file) mapping
# Derived from cross-referencing GLMsingle design matrix with individual DMs
RUN_INFO = {
    0:  ("imagery",    "A", "imgA_1"),
    1:  ("attention",  "A", "attA"),
    2:  ("perception", "A", "visA"),
    3:  ("imagery",    "B", "imgB_1"),
    4:  ("attention",  "B", "attB"),
    5:  ("perception", "B", "visB"),
    6:  ("imagery",    "C", "imgC_1"),
    7:  ("attention",  "C", "attC"),
    8:  ("perception", "C", "visC"),
    9:  ("imagery",    "A", "imgA_2"),
    10: ("imagery",    "B", "imgB_2"),
    11: ("imagery",    "C", "imgC_2"),
}

# Stimulus type labels per set
SET_STIMULUS_TYPE = {
    "A": "simple",       # oriented bars and crosses
    "B": "complex",      # real NSD photos (shared1000 subset)
    "C": "conceptual",   # verbal/semantic concepts
}


@dataclass
class TrialInfo:
    """Complete metadata for a single NSD-Imagery beta volume."""
    beta_index: int          # Volume index in 4D NIfTI (0-719)
    run_id: int              # Run number (0-11)
    tr: int                  # TR (time point) within run
    cond_col: int            # GLMsingle condition column (0-35)
    task_type: str           # "imagery", "perception", or "attention"
    stimulus_set: str        # "A", "B", or "C"
    stimulus_type: str       # "simple", "complex", or "conceptual"
    cue_letter: str          # Single-letter cue (e.g., "E", "B", "F")
    stimulus_name: str       # Full stimulus name (e.g., "bar_000.0deg_450L_43W.png")
    nsd_id: Optional[int]    # NSD stimulus ID (only for set B)
    shared_id: Optional[int] # Shared1000 index (only for set B)
    image_path: Optional[str]  # Path to stimulus/cue image if available
    repeat_index: int        # 0 for first presentation runs, 1 for second

    def to_dict(self) -> Dict:
        return asdict(self)


def _get_cell_str(cell) -> str:
    """Extract string from MATLAB cell array element."""
    if hasattr(cell, 'flat') and cell.size > 0:
        return str(cell.flat[0])
    return str(cell)


def _get_dm_onset_conds(dm: np.ndarray) -> List[Tuple[int, int]]:
    """Extract (onset_tr, cond_idx) pairs from an individual design matrix.
    
    Onset is detected as a transition from 0→nonzero in a condition column.
    """
    onsets = []
    for tr in range(dm.shape[0]):
        for col in range(dm.shape[1]):
            if dm[tr, col] != 0 and (tr == 0 or dm[tr - 1, col] == 0):
                onsets.append((tr, col))
    return sorted(onsets, key=lambda x: x[0])


def _get_glm_onset_conds(rd: np.ndarray) -> List[Tuple[int, int]]:
    """Extract (onset_tr, cond_col) pairs from a GLMsingle run matrix."""
    onsets = []
    for tr in range(rd.shape[0]):
        for col in range(rd.shape[1]):
            if rd[tr, col] != 0:
                onsets.append((tr, col))
    return sorted(onsets, key=lambda x: x[0])


def parse_cue_pair_list(metadata_dir: Path) -> pd.DataFrame:
    """Parse cue_pair_list.xlsx to get cue → stimulus mapping.
    
    Returns DataFrame with columns: stim_set, target, cue, nsd_id, shared_id
    """
    xlsx_path = metadata_dir / "cue_pair_list.xlsx"
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        # Fallback: reconstruct from .mat pair_lists
        rows = []
        for set_name in ["A", "B", "C"]:
            mat_path = metadata_dir / f"{set_name}_pair_list.mat"
            if not mat_path.exists():
                continue
            import scipy.io as sio
            mat = sio.loadmat(str(mat_path))
            pl = mat["pair_list"]
            for i in range(pl.shape[0]):
                target = _get_cell_str(pl[i, 1])
                cue = _get_cell_str(pl[i, 2])
                rows.append({"stim_set": set_name, "target": target, "cue": cue})
        df = pd.DataFrame(rows)
    
    # Extract NSD IDs from target names like "shared0385_nsd28752.png"
    df["nsd_id"] = None
    df["shared_id"] = None
    for idx, row in df.iterrows():
        target = str(row["target"])
        nsd_match = re.search(r'_nsd(\d+)', target)
        shared_match = re.search(r'shared(\d+)', target)
        if nsd_match:
            nsd_id = int(nsd_match.group(1))
            df.at[idx, "nsd_id"] = nsd_id if nsd_id > 0 else None
        if shared_match:
            df.at[idx, "shared_id"] = int(shared_match.group(1))
    
    return df


def _build_col_to_cue_map(
    metadata_dir: Path,
    dm_file: str,
    glm_run_data: np.ndarray,
) -> Dict[int, str]:
    """Map GLMsingle condition columns → cue letters for one run.
    
    Cross-references trial onset times between the GLMsingle design matrix
    and the individual design matrix to establish the mapping.
    """
    import scipy.io as sio
    
    # Handle attention DMs which are longer than needed
    # The attention DMs include NaN (catch) conditions that GLMsingle excludes
    dm_file_base = dm_file  # e.g., "attA"
    mat = sio.loadmat(str(metadata_dir / f"{dm_file_base}_dm.mat"))
    dm = mat["dm"]
    cl = mat["condit_list"]
    
    # Extract condition labels
    labels = []
    for i in range(cl.shape[0]):
        parts = []
        for j in range(cl.shape[1]):
            parts.append(_get_cell_str(cl[i, j]))
        labels.append("_".join(parts) if len(parts) > 1 else parts[0])
    
    # For attention DMs, exclude NaN conditions
    valid_cond_indices = []
    for i, label in enumerate(labels):
        if "NaN" not in label:
            valid_cond_indices.append(i)
    
    # Get onset times from both matrices
    glm_onsets = _get_glm_onset_conds(glm_run_data)
    dm_onsets = _get_dm_onset_conds(dm)
    
    # Build TR→dm_cond map (only for valid conditions)
    dm_by_tr = {}
    for tr, cond in dm_onsets:
        if cond in valid_cond_indices:
            dm_by_tr[tr] = cond
    
    # Match: for each GLM trial, find the DM condition at the same TR
    col_mapping = {}  # glm_col → set of dm_cond_indices
    for tr, glm_col in glm_onsets:
        if tr in dm_by_tr:
            dm_cond = dm_by_tr[tr]
            col_mapping.setdefault(glm_col, set()).add(dm_cond)
    
    # Verify consistency and extract cue letters
    result = {}
    for glm_col, dm_conds in col_mapping.items():
        if len(dm_conds) == 1:
            dm_idx = list(dm_conds)[0]
            label = labels[dm_idx]
            # Extract just the cue letter (first character before any underscore)
            cue_letter = label.split("_")[0]
            result[glm_col] = cue_letter
        else:
            # Multiple DM conditions mapped to same GLM column — 
            # pick the most common one
            logger.warning(
                f"Ambiguous mapping for col {glm_col}: DM conds {dm_conds}"
            )
            dm_idx = list(dm_conds)[0]
            result[glm_col] = labels[dm_idx].split("_")[0]
    
    return result


def parse_all_trials(
    metadata_dir: Path,
    stimulus_root: Optional[Path] = None,
) -> List[TrialInfo]:
    """Parse all 720 trial betas from NSD-Imagery metadata.
    
    Args:
        metadata_dir: Path to metadata/ directory containing .mat files
        stimulus_root: Optional path to stimuli/ directory for resolving image paths
    
    Returns:
        List of 720 TrialInfo objects, ordered by beta_index
    """
    import scipy.io as sio
    
    # 1. Load GLMsingle design matrix
    glm_path = metadata_dir / "designmatrixGLMsingle.mat"
    if not glm_path.exists():
        raise FileNotFoundError(
            f"designmatrixGLMsingle.mat not found in {metadata_dir}. "
            f"Make sure NSD-Imagery metadata is downloaded."
        )
    
    mat = sio.loadmat(str(glm_path))
    stimulus = mat["stimulus"]  # (1, 12) cell array of per-run design matrices
    
    # 2. Load cue→stimulus mapping
    cue_df = parse_cue_pair_list(metadata_dir)
    cue_to_info = {}
    for _, row in cue_df.iterrows():
        cue_to_info[row["cue"]] = {
            "stimulus_name": str(row["target"]),
            "stimulus_set": str(row["stim_set"]),
            "nsd_id": row["nsd_id"],
            "shared_id": row["shared_id"],
        }
    
    # 3. For each run, build GLMsingle_col → cue_letter mapping
    run_col_maps = {}
    for run_idx in range(12):
        task_type, stim_set, dm_file = RUN_INFO[run_idx]
        rd = stimulus[0, run_idx]
        
        col_to_cue = _build_col_to_cue_map(metadata_dir, dm_file, rd)
        run_col_maps[run_idx] = col_to_cue
    
    # 4. Build the complete trial list
    # GLMsingle produces betas in order: run0 trials (sorted by TR),
    # then run1 trials, etc.
    trials = []
    beta_idx = 0
    
    for run_idx in range(12):
        rd = stimulus[0, run_idx]
        task_type, stim_set, dm_file = RUN_INFO[run_idx]
        col_to_cue = run_col_maps[run_idx]
        
        # Determine repeat_index
        if dm_file.endswith("_2"):
            repeat_index = 1
        else:
            repeat_index = 0
        
        # Get trial onsets for this run, sorted by TR
        run_onsets = _get_glm_onset_conds(rd)
        
        for tr, cond_col in run_onsets:
            cue_letter = col_to_cue.get(cond_col, "?")
            cue_info = cue_to_info.get(cue_letter, {})
            
            stimulus_name = cue_info.get("stimulus_name", f"unknown_{cue_letter}")
            nsd_id = cue_info.get("nsd_id")
            shared_id = cue_info.get("shared_id")
            stim_type = SET_STIMULUS_TYPE.get(stim_set, "unknown")
            
            # Resolve image path if stimulus_root is available
            image_path = None
            if stimulus_root is not None:
                allstim_dir = stimulus_root / "allstim"
                if allstim_dir.exists():
                    # Try to find cue image matching this stimulus
                    # Pattern: {stimulus_base_name}_cue{letter}.png
                    stim_base = stimulus_name.replace(".png", "")
                    cue_img = allstim_dir / f"{stim_base}_cue{cue_letter}.png"
                    if cue_img.exists():
                        image_path = str(cue_img)
                    else:
                        # For conceptual stimuli, look for pattern like
                        # "banana0000_nsd00000_cueN.png"
                        matches = list(allstim_dir.glob(
                            f"{stimulus_name}*_cue{cue_letter}.png"
                        ))
                        if matches:
                            image_path = str(matches[0])
                
                # Also check rawtargetimages for the actual target
                if nsd_id is not None and nsd_id > 0:
                    raw_img = (metadata_dir / "rawtargetimages" / "setB" / 
                              stimulus_name)
                    if raw_img.exists():
                        image_path = str(raw_img)
            
            trial = TrialInfo(
                beta_index=beta_idx,
                run_id=run_idx,
                tr=tr,
                cond_col=cond_col,
                task_type=task_type,
                stimulus_set=stim_set,
                stimulus_type=stim_type,
                cue_letter=cue_letter,
                stimulus_name=stimulus_name,
                nsd_id=int(nsd_id) if nsd_id is not None else None,
                shared_id=int(shared_id) if shared_id is not None else None,
                image_path=image_path,
                repeat_index=repeat_index,
            )
            trials.append(trial)
            beta_idx += 1
    
    logger.info(
        f"Parsed {len(trials)} trials: "
        f"{sum(1 for t in trials if t.task_type == 'imagery')} imagery, "
        f"{sum(1 for t in trials if t.task_type == 'perception')} perception, "
        f"{sum(1 for t in trials if t.task_type == 'attention')} attention"
    )
    
    return trials


def trials_to_dataframe(
    trials: List[TrialInfo],
    subject: str,
    beta_path: str,
) -> pd.DataFrame:
    """Convert trial list to a DataFrame suitable for the index Parquet.
    
    Args:
        trials: List of TrialInfo from parse_all_trials()
        subject: Subject ID (e.g., "subj01")
        beta_path: Relative path to the betas_nsdimagery.nii.gz file
    
    Returns:
        DataFrame with canonical index schema
    """
    records = []
    for t in trials:
        # Map attention to its underlying task type for the condition column
        # (attention trials involve viewing stimuli, so label as perception)
        if t.task_type == "attention":
            condition = "attention"
        else:
            condition = t.task_type
        
        meta = {
            "cue_letter": t.cue_letter,
            "stimulus_set": t.stimulus_set,
            "tr": t.tr,
            "cond_col": t.cond_col,
            "repeat_index": t.repeat_index,
        }
        if t.nsd_id is not None:
            meta["nsd_id"] = t.nsd_id
        if t.shared_id is not None:
            meta["shared_id"] = t.shared_id
        
        records.append({
            "trial_id": t.beta_index,
            "subject": subject,
            "condition": condition,
            "stimulus_type": t.stimulus_type,
            "task_type": t.task_type,
            "run_id": t.run_id,
            "beta_index": t.beta_index,
            "fmri_path": beta_path,
            "image_path": t.image_path,
            "text_prompt": t.stimulus_name if t.stimulus_type == "conceptual" else None,
            "nsd_id": t.nsd_id,
            "cue_letter": t.cue_letter,
            "stimulus_set": t.stimulus_set,
            "stimulus_name": t.stimulus_name,
            "shared_id": t.shared_id,
            "repeat_index": t.repeat_index,
            "meta_json": json.dumps(meta),
            "split": None,  # Assigned downstream
        })
    
    return pd.DataFrame(records)


def assign_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> pd.DataFrame:
    """Assign train/val/test splits deterministically.
    
    Splits by run_id to avoid data leakage (same stimulus in train and test).
    """
    df = df.copy()
    rng = np.random.RandomState(seed)
    
    unique_runs = sorted(df["run_id"].unique())
    n_runs = len(unique_runs)
    
    if n_runs >= 4:
        # Split by runs
        shuffled_runs = unique_runs.copy()
        rng.shuffle(shuffled_runs)
        
        n_train = max(1, int(n_runs * train_frac))
        n_val = max(1, int(n_runs * val_frac))
        
        train_runs = set(shuffled_runs[:n_train])
        val_runs = set(shuffled_runs[n_train:n_train + n_val])
        test_runs = set(shuffled_runs[n_train + n_val:])
        
        df.loc[df["run_id"].isin(train_runs), "split"] = "train"
        df.loc[df["run_id"].isin(val_runs), "split"] = "val"
        df.loc[df["run_id"].isin(test_runs), "split"] = "test"
    else:
        # Split by trial_id
        shuffled_ids = df["trial_id"].values.copy()
        rng.shuffle(shuffled_ids)
        
        n_total = len(shuffled_ids)
        n_train = max(1, int(n_total * train_frac))
        n_val = max(1, int(n_total * val_frac))
        
        train_ids = set(shuffled_ids[:n_train])
        val_ids = set(shuffled_ids[n_train:n_train + n_val])
        
        df.loc[df["trial_id"].isin(train_ids), "split"] = "train"
        df.loc[df["trial_id"].isin(val_ids), "split"] = "val"
        df.loc[~df["trial_id"].isin(train_ids | val_ids), "split"] = "test"
    
    # Fill any remaining NaN splits
    df["split"] = df["split"].fillna("train")
    
    return df
