# System Configurations

Infrastructure-level configs for data access and NSD dataset structure.

---

## Available Configs

| Config | Purpose |
|--------|---------|
| `data.yaml` | NSD dataset: S3 paths, preprocessing pipelines, subjects, splits, ablation grids |

---

## `data.yaml`

Comprehensive NSD dataset configuration:

- **S3 access**: bucket name, region, anonymous access settings
- **Cache**: local cache directories for S3 files, stimuli, and indices
- **fMRI pipelines**: `betas_fithrf_GLMdenoise_RR` (recommended), `betas_fithrf`, `betas_assumehrf`
- **Subjects**: all 8 NSD subjects with session counts
- **Preprocessing**: reliability threshold, PCA dimensionality, variance filtering
- **Ablation grids**: reliability sweep and PCA sweep values

This config is referenced by data loaders (`NSDLayout`) and preprocessing modules. The lighter `configs/data.yaml` (root-level) provides minimal split ratios used as defaults by training scripts.

---

## Notes

- Logging is configured in `configs/base.yaml` under the `logging:` section.
- CLIP model settings live in `configs/clip.yaml` (project root), not in this directory.
