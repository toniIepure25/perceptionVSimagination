# NSD-Imagery Dataset Guide

**Technical Documentation for NSD-Imagery Integration**

---

## Overview

### What is NSD-Imagery?

The **NSD-Imagery** dataset is a complementary extension to the Natural Scenes Dataset (NSD) that captures fMRI brain activity while subjects **mentally imagine** visual scenes they previously viewed during the original NSD perception experiments. This dataset enables direct comparison of neural representations during perception versus imagery, providing a unique opportunity to study cross-domain generalization in brain decoding.

**Key Properties**:
- **Paired Design**: Subjects imagined the same stimuli they viewed in NSD perception sessions
- **Same Acquisition Protocol**: 7T fMRI with identical scanning parameters as NSD
- **Reduced SNR**: Imagery signals typically have lower signal-to-noise ratio than perception
- **Subset of Stimuli**: Not all NSD stimuli were used in imagery sessions (typically ~1000-3000 per subject)
- **Fewer Repeats**: Imagery trials often have fewer repetitions than perception trials

### Research Applications

1. **Cross-Domain Transfer Learning**: Test if decoders trained on perception generalize to imagery
2. **Shared Neural Representations**: Quantify overlap between perception and imagery activations
3. **Domain Adaptation**: Develop methods to bridge perception-imagery gaps
4. **Cognitive Neuroscience**: Study mental imagery mechanisms at the neural level

---

## Dataset Structure

### Expected Local Folder Structure

All NSD-Imagery data should be organized under the `cache/` directory to maintain consistency with the existing NSD perception pipeline:

```
cache/
├── nsd_imagery/                       # Root for NSD-Imagery data
│   ├── betas/                         # fMRI beta estimates
│   │   ├── subj01/
│   │   │   ├── imagery_betas_session01.nii.gz
│   │   │   ├── imagery_betas_session02.nii.gz
│   │   │   └── ...
│   │   ├── subj02/
│   │   └── ...
│   ├── stimuli/                       # Stimulus metadata and references
│   │   ├── imagery_stimuli_info.csv   # Trial-level metadata
│   │   └── presented_image_ids.txt    # List of image IDs used in imagery
│   └── metadata/
│       ├── trial_manifest_subj01.csv  # Per-subject trial manifests
│       └── subject_sessions.json      # Session counts and dates
│
├── nsd_hdf5/                          # Existing NSD perception data
│   └── nsd_stimuli.hdf5               # Shared stimulus images
│
├── clip_embeddings/                   # CLIP embeddings (shared across modalities)
│   └── imagery_cache/                 # Imagery-specific CLIP cache if needed
│
└── indices/                           # Canonical index files
    ├── imagery/                       # NSD-Imagery indices (Parquet format)
    │   ├── subj01.parquet
    │   ├── subj02.parquet
    │   └── full_index.parquet         # Combined across subjects
    └── perception/                    # Existing NSD perception indices
        └── ...
```

**Important Notes**:
- Do **NOT** hardcode absolute paths like `/home/user/...` in code
- Use relative paths from project root: `cache/nsd_imagery/...`
- Environment variables can override cache root if needed: `$NSD_CACHE_ROOT`

---

## Data Access / Download

### Prerequisites

1. **NSD Data Access Agreement**: You must have signed the NSD data sharing agreement
2. **NSD-Imagery Access**: Request access to the NSD-Imagery extension dataset from the NSD maintainers
3. **Storage**: Allocate ~50-100GB for NSD-Imagery data (varies by number of subjects)

### Download Steps (Generic)

**Note**: Specific download URLs and procedures depend on the NSD-Imagery data release mechanism. Check the official NSD website for the latest instructions.

```bash
# Step 1: Create directory structure
mkdir -p cache/nsd_imagery/{betas,stimuli,metadata}

# Step 2: Download data files (placeholder command)
# This will vary based on whether data is hosted on AWS S3, institutional server, etc.
# Example (adjust as needed):
# aws s3 sync s3://nsd-imagery/betas cache/nsd_imagery/betas --profile nsd

# Step 3: Download stimulus metadata
# wget <URL_TO_IMAGERY_STIMULI_INFO> -O cache/nsd_imagery/stimuli/imagery_stimuli_info.csv

# Step 4: Verify data integrity
python scripts/verify_nsd_imagery_data.py --cache-root cache/
```

### Data Validation

After downloading, verify:
- [ ] Beta files exist for all expected subjects and sessions
- [ ] Stimulus metadata file contains expected columns (see below)
- [ ] Image IDs in imagery match existing NSD stimuli (cross-reference with `nsd_stimuli.hdf5`)

---

## Canonical Internal Representation

### ImageryTrial Schema

All NSD-Imagery data will be represented internally using a standardized schema:

```python
from typing import Optional, Literal
from dataclasses import dataclass

@dataclass
class ImageryTrial:
    """Canonical representation of an NSD-Imagery trial."""
    
    # Unique identifiers
    trial_id: int                          # Global unique trial ID
    subject: str                           # e.g., "subj01"
    session: int                           # Session number
    trial_in_session: int                  # Trial index within session
    
    # Condition and stimulus
    condition: Literal["perception", "imagery"]  # Data source
    nsd_id: int                            # NSD stimulus ID (shared with perception)
    coco_id: Optional[int]                 # COCO dataset ID (if applicable)
    
    # fMRI data
    beta_path: str                         # Path to NIfTI file (relative to cache root)
    beta_index: int                        # Volume index within NIfTI file
    roi_mask_path: Optional[str]           # Path to ROI mask if using specific regions
    
    # Optional metadata
    repeat_index: int = 0                  # 0 for first presentation, 1+ for repeats
    caption: Optional[str] = None          # Image caption (if available)
    run_number: Optional[int] = None       # Run number within session
    
    # Quality control
    is_valid: bool = True                  # Flag for excluding corrupted trials
    snr_estimate: Optional[float] = None   # Signal-to-noise ratio if available
```

### Index File Format (Parquet)

Indices are stored as Parquet files with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `trial_id` | int64 | Global unique ID | 45231 |
| `subject` | str | Subject ID | "subj01" |
| `session` | int32 | Session number | 3 |
| `trial_in_session` | int32 | Trial index | 75 |
| `condition` | str | "perception" or "imagery" | "imagery" |
| `nsd_id` | int32 | NSD stimulus ID | 5234 |
| `coco_id` | int32 (nullable) | COCO ID | 187450 |
| `beta_path` | str | Relative path to beta file | "nsd_imagery/betas/subj01/session03.nii.gz" |
| `beta_index` | int32 | Volume index | 74 |
| `repeat_index` | int32 | Repeat count | 0 |
| `is_valid` | bool | Quality flag | True |

**Usage Example**:
```python
import pandas as pd

# Load imagery index
index = pd.read_parquet("cache/indices/imagery/subj01.parquet")

# Filter to imagery trials only
imagery_trials = index[index["condition"] == "imagery"]

# Get first 100 trials
subset = imagery_trials.head(100)
```

---

## Common Pitfalls

### 1. Mismatched Trial Indices

**Problem**: NSD-Imagery trial indices do not directly correspond to NSD perception trial indices.

**Solution**: Always join on `nsd_id` (stimulus ID) rather than trial indices when matching perception and imagery trials for the same stimulus.

```python
# ❌ WRONG: Assuming same trial_id means same stimulus
perception_trial = perception_index[perception_index["trial_id"] == 1000]
imagery_trial = imagery_index[imagery_index["trial_id"] == 1000]

# ✅ CORRECT: Match on nsd_id
nsd_id_of_interest = 5234
perception_trial = perception_index[perception_index["nsd_id"] == nsd_id_of_interest]
imagery_trial = imagery_index[imagery_index["nsd_id"] == nsd_id_of_interest]
```

---

### 2. Subject Alignment Issues

**Problem**: Not all NSD subjects participated in imagery sessions. Subject IDs may differ.

**Solution**: 
- Always check which subjects have imagery data before assuming availability
- Use `imagery_index["subject"].unique()` to get available subjects
- Document subject availability in experiment configs

```python
# Check available subjects in imagery
imagery_subjects = set(imagery_index["subject"].unique())
perception_subjects = set(perception_index["subject"].unique())

common_subjects = imagery_subjects & perception_subjects
print(f"Subjects with both perception and imagery: {common_subjects}")
```

---

### 3. Different Number of Repeats

**Problem**: Imagery trials typically have fewer repeats than perception trials, affecting noise ceiling estimates.

**Solution**:
- When comparing reliability, normalize by number of repeats
- Use `repeat_index` column to identify repeat trials
- Calculate reliability separately for perception and imagery

```python
# Count repeats per condition
perception_repeats = perception_index[perception_index["repeat_index"] > 0].groupby("nsd_id").size()
imagery_repeats = imagery_index[imagery_index["repeat_index"] > 0].groupby("nsd_id").size()

print(f"Avg perception repeats: {perception_repeats.mean():.1f}")
print(f"Avg imagery repeats: {imagery_repeats.mean():.1f}")
```

---

### 4. Lower SNR for Imagery

**Problem**: Mental imagery produces weaker and more variable fMRI signals than perception, leading to lower decoding accuracy.

**Solution**:
- Expect lower baseline performance on imagery (50-80% of perception performance is typical)
- Use more aggressive preprocessing (spatial smoothing, denoising) for imagery
- Consider ensemble methods or averaging over repeats to boost SNR
- Do not directly compare raw metrics between perception and imagery without normalization

```python
# Example: Apply stronger preprocessing for imagery
preprocessor_perception = NSDPreprocessor(smoothing_fwhm=3.0)
preprocessor_imagery = NSDPreprocessor(smoothing_fwhm=5.0)  # Stronger smoothing
```

---

### 5. Missing Stimuli in Imagery Set

**Problem**: Not all NSD stimuli were used in imagery experiments; attempting to decode imagery for perception-only stimuli will fail.

**Solution**:
- Build a set of valid `nsd_id` values present in imagery dataset
- Filter evaluation to only shared stimuli
- Document split clearly: perception-only stimuli vs. shared stimuli

```python
# Get valid imagery stimulus IDs
valid_imagery_nsd_ids = set(imagery_index["nsd_id"].unique())

# Filter perception data to shared stimuli
shared_perception = perception_index[
    perception_index["nsd_id"].isin(valid_imagery_nsd_ids)
]

print(f"Shared stimuli: {len(valid_imagery_nsd_ids)}")
print(f"Perception-only stimuli: {len(perception_index['nsd_id'].unique()) - len(valid_imagery_nsd_ids)}")
```

---

## Integration with Existing Pipeline

### Data Loading

The existing `NSDIterableDataset` and related loaders can be extended to support imagery:

```python
from fmri2img.data.nsd_imagery import NSDImageryDataset

# Load imagery data using same interface as perception
imagery_dataset = NSDImageryDataset(
    index_path="cache/indices/imagery/subj01.parquet",
    subject="subj01",
    condition="imagery",  # New parameter
    preprocessor=preprocessor,
    clip_cache=clip_cache
)

# Use with PyTorch DataLoader as usual
from torch.utils.data import DataLoader
loader = DataLoader(imagery_dataset, batch_size=32, num_workers=4)
```

### CLIP Embeddings

Imagery trials can share the same CLIP embeddings as perception trials (since they refer to the same images):

```bash
# CLIP embeddings are keyed by nsd_id, so existing cache works
python scripts/build_clip_cache.py --mode both  # Handles perception + imagery
```

### Evaluation

Extend existing evaluation scripts to support mixed evaluation:

```python
# Evaluate on both perception and imagery test sets
python scripts/eval_shared1000_full.py \
  --subject subj01 \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --test-sets perception,imagery \
  --output outputs/eval/cross_domain/
```

---

## Next Steps

1. **Data Acquisition**: Secure access to NSD-Imagery dataset
2. **Index Building**: Run `scripts/build_nsd_imagery_index.py` to create canonical indices
3. **Validation**: Verify data quality and alignment with perception data
4. **Evaluation**: Use `scripts/eval_perception_to_imagery_transfer.py` to establish baselines

For research workflow, see: [`docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md`](../research/PERCEPTION_VS_IMAGERY_ROADMAP.md)

---

## References

- Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116-126.
- NSD Data Manual: https://cvnlab.slite.page/p/NKalgWd_qQ/NSD-Data-Manual

---

**Last Updated**: January 10, 2026  
**Maintainer**: fMRI2Image Research Team
