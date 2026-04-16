# Anonymous Supplementary Material

## Benchmark Reproducibility Package

This supplementary material accompanies the NeurIPS 2026 E&D submission:
*Benchmarking Paired fMRI Perception–Imagery Decoding Under Overlap Scarcity*.

### What this package contains

| Item | Description | Included? |
|------|-------------|-----------|
| Canonical benchmark configs (YAML) | Exact configs for every ladder rung | Yes |
| Workflow entry points | Exact `python -m` commands for every stage | Yes |
| Expected artifact tree | Directory structure for all benchmark outputs | Yes |
| External asset inventory | Public datasets, models, and their licenses | Yes |
| Artifact manifest | Expected output files with descriptions | Yes |
| Seed-stability protocol | Instructions for running seed sweeps | Yes |
| Prepared dataset | 94-row mixed overlap parquet | Not bundled (derived from public NSD data; see reproduction instructions) |
| Target cache | CLIP ViT-L/14 embeddings parquet | Not bundled (derived; see reproduction instructions) |
| Model checkpoints | Trained `.pt` files | Not bundled (reproducible via configs + commands) |
| Source code | Full `fmri2img` package | Not bundled in anonymous submission; will be released upon acceptance |

### What a reviewer can verify from this package

1. **Config completeness**: Every hyperparameter, path, and role declaration
   referenced in the paper is present in checked-in YAML configs.
2. **Command reproducibility**: Every workflow stage has an exact `python -m`
   command with a named config.
3. **Artifact contract**: The expected output structure is fully documented,
   so a reviewer can confirm that the paper's claims map to specific artifacts.
4. **External dependency transparency**: All external data sources and their
   licenses are listed explicitly.
5. **Evaluation protocol fixedness**: Target space, ROI contract, metrics,
   and model roles are all specified in configs, not in ad-hoc code.

### How to reproduce the full benchmark

#### Prerequisites

- Python ≥ 3.10
- The `fmri2img` package installed in a virtual environment
- NSD perception data (public, see external assets below)
- NSD-Imagery beta data (public, see external assets below)
- NSD ROI masks (public, from NSD release)
- ~10 GB disk for prepared data + targets + model outputs
- GPU optional (workflows fall back to CPU automatically)

#### Step-by-step reproduction

```
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Set environment variables for NSD data paths
export NSD_ROI_MASK_ROOT=/path/to/nsd/roi_masks
export NSD_IMAGERY_ROOT=/path/to/nsd_imagery
export NSD_IMAGERY_METADATA_ROOT=/path/to/nsd_imagery/metadata
export NSD_IMAGERY_BETA_ROOT=/path/to/nsd_imagery/betas

# 3. Acquire imagery indices
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all --skip-stimuli --output cache/nsd_imagery_full_all

# 4. Prepare overlap dataset
python -m fmri2img.workflows.prepare_overlap_bootstrap \
  --config configs/canonical/max_available_overlap.yaml --overwrite-existing

# 5. Build target cache
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/max_available_overlap.yaml

# 6. Run preflight
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/max_available_overlap.yaml

# 7. Train Ridge baseline
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml

# 8. Train shared-only (canonical neural baseline)
python -m fmri2img.workflows.train_animus_core_decoder

# 9. Train shared-private p16 (best exploratory variant)
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml

# 10. Train shared-private default
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/max_available_overlap.yaml

# 11. Train shared-private p8 (recovery variant)
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/max_available_overlap.yaml \
  --override model.private_dim=8

# 12. Train shared-private no-domain (diagnostic control)
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/max_available_overlap.yaml \
  --override model.use_domain_head=false

# 13. Evaluate (repeat for each model)
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/animus_core_decoder.yaml \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

#### Seed-stability reproduction

To verify ordering stability across seeds, override the seed parameter:

```
# Shared-only, seeds 1 and 2
python -m fmri2img.workflows.train_animus_core_decoder \
  --override training.seed=1 training.output_dir=outputs/seed_stability/shared_only_s1
python -m fmri2img.workflows.train_animus_core_decoder \
  --override training.seed=2 training.output_dir=outputs/seed_stability/shared_only_s2

# Shared-private p16, seeds 1 and 2
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --override training.seed=1 training.output_dir=outputs/seed_stability/sp_p16_s1
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --override training.seed=2 training.output_dir=outputs/seed_stability/sp_p16_s2
```

---

## Canonical Configs Used in the Paper

### Primary benchmark configs

| Paper system | Config file | Role |
|---|---|---|
| Ridge | `configs/canonical/max_available_overlap.yaml` | External low-data reference |
| Shared-only | `configs/canonical/animus_core_decoder.yaml` | Canonical neural baseline |
| SP, d_priv=16 | `configs/canonical/threshold_shared_private_p16.yaml` | Best exploratory SP variant |

### Shared settings across all neural configs

| Parameter | Value | Notes |
|---|---|---|
| `training.batch_size` | 8 | Fixed across ladder |
| `training.epochs` | 5 | Fixed across ladder |
| `training.learning_rate` | 0.001 | Fixed across ladder |
| `training.weight_decay` | 0.0001 | Fixed across ladder |
| `training.seed` | 0 | Frozen benchmark seed |
| `targets.name` | `vit_l14_image_768` | Fixed target space |
| `targets.dimension` | 768 | Fixed dimensionality |
| `model.shared_dim` | 128 | Same encoder backbone |
| `model.branch_embedding_dim` | 128 | Same encoder backbone |

### Model-specific settings

| Config | `disentanglement_mode` | `private_dim` | `use_domain_head` |
|---|---|---|---|
| `animus_core_decoder.yaml` | `shared_only` | 64 (unused) | `false` |
| `threshold_shared_private_p16.yaml` | `shared_private` | 16 | `true` |
| `max_available_overlap.yaml` | (default) | 64 | `true` |

---

## Expected Artifact Tree

All 6 ladder rungs from Table 1 of the paper:

```
outputs/
├── canonical/
│   ├── prepared/
│   │   └── full_imagery_overlap/
│   │       ├── full_imagery_overlap_mixed_with_roi.parquet  # 94-row prepared dataset
│   │       ├── report.json
│   │       ├── overlap_nsd_ids.json                         # 5 shared paired IDs
│   │       └── preflight.json                               # readiness classification
│   ├── baselines/
│   │   └── full_imagery_overlap_ridge_legacy/
│   │       └── metrics.json                                 # Ridge results
│   └── train/
│       ├── full_imagery_overlap/                            # SP default
│       │   ├── best_decoder.pt
│       │   ├── config_snapshot.json
│       │   └── train_history.json
│       ├── full_imagery_overlap_priv8/                      # SP p8 (override)
│       │   ├── best_decoder.pt
│       │   ├── config_snapshot.json
│       │   └── train_history.json
│       └── full_imagery_overlap_nodomain/                   # SP no-domain (override)
│           ├── best_decoder.pt
│           ├── config_snapshot.json
│           └── train_history.json
├── targets/
│   └── full_imagery_overlap_vit_l14_image_768.parquet       # CLIP target cache
├── animus/
│   └── core_decoder/
│       ├── train/full_imagery_overlap_shared_only/          # Shared-only
│       │   ├── best_decoder.pt
│       │   ├── config_snapshot.json
│       │   └── train_history.json
│       └── eval/full_imagery_overlap_shared_only/
│           └── eval_metrics.json
└── research/
    └── threshold_shared_private_p16/
        ├── train/full_imagery_overlap/                      # SP p16
        │   ├── best_decoder.pt
        │   ├── config_snapshot.json
        │   └── train_history.json
        └── eval/full_imagery_overlap/
            └── eval_metrics.json
```

---

## Paper Claims → Artifact Provenance

This table maps every row in Table 1 to its config, command, and output artifact,
so a reviewer can trace each reported number to its source.

| Table 1 row | Cosine | MSE | Config | Override | Artifact |
|---|---|---|---|---|---|
| Ridge | 0.55199 | 0.001167 | `max_available_overlap.yaml` | — | `outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/metrics.json` |
| Shared-only | 0.13596 | 0.002250 | `animus_core_decoder.yaml` | — | `outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt` |
| SP, d_priv=16 | 0.10784 | 0.002323 | `threshold_shared_private_p16.yaml` | — | `outputs/research/threshold_shared_private_p16/train/full_imagery_overlap/best_decoder.pt` |
| SP, d_priv=8 | 0.09595 | 0.002354 | `max_available_overlap.yaml` | `model.private_dim=8` | `outputs/canonical/train/full_imagery_overlap_priv8/best_decoder.pt` |
| SP default | 0.06927 | 0.002424 | `max_available_overlap.yaml` | — | `outputs/canonical/train/full_imagery_overlap/best_decoder.pt` |
| SP, no domain | 0.05907 | 0.002450 | `max_available_overlap.yaml` | `model.use_domain_head=false` | `outputs/canonical/train/full_imagery_overlap_nodomain/best_decoder.pt` |

Table 2 (claim boundary) is derived from the same frozen evidence bundle.
Each claim maps to the metrics above plus the preflight status in
`outputs/canonical/prepared/full_imagery_overlap/preflight.json`.

---

## External Asset Inventory and Licenses

### Datasets

| Asset | Source | License | URL | Usage in paper |
|---|---|---|---|---|
| Natural Scenes Dataset (NSD) | Allen et al., 2022 | CC BY 4.0 (data terms at NSD site) | https://naturalscenesdataset.org | Perception indices, ROI masks, stimulus IDs |
| NSD-Imagery | NSD consortium | Same as NSD release terms | https://naturalscenesdataset.org | Imagery beta data, imagery conditions |
| COCO images (via NSD stimuli) | Microsoft COCO | CC BY 4.0 | https://cocodataset.org | Stimulus images for target embedding |

### Pretrained models

| Asset | Source | License | Usage in paper |
|---|---|---|---|
| CLIP ViT-L/14 | Radford et al., 2021 (OpenAI) | MIT License | Target embedding extraction (`vit_l14_image_768`) |

### Software dependencies

| Package | License | Role |
|---|---|---|
| PyTorch | BSD-3-Clause | Neural model training |
| scikit-learn | BSD-3-Clause | Ridge regression baseline |
| OpenCLIP | MIT | CLIP model loading for target extraction |
| pandas, numpy, scipy | BSD-3-Clause | Data handling and metrics |
| nibabel | MIT | NIfTI file loading for fMRI data |

---

## Artifact Scope Statement

This anonymous supplementary material provides the complete specification
needed to reproduce the paper's benchmark results. It does not bundle:

- Raw NSD data (publicly available, ~1 TB)
- Prepared intermediate datasets (derivable from public data + configs)
- Model checkpoints (reproducible from configs + commands)
- Full source code (will be released upon acceptance)

The configs, commands, artifact tree, and external asset inventory together
constitute a complete reproduction contract: a reviewer can verify that every
number in the paper maps to a specific config, command, and output path, and
that the external data sources are publicly accessible.

---

## Seed-Stability Results

A 3-seed stability check was run for the two primary neural models on an
NVIDIA H100 80GB GPU. The training pipeline sets all global random seeds
(`torch.manual_seed`, `numpy.random.seed`, `random.seed`,
`torch.cuda.manual_seed_all`) before model construction.

### Commands used

```bash
# Shared-only, 3 seeds
for SEED in 0 42 123; do
  ./.venv/bin/python -m fmri2img.workflows.train_decoder \
    --config configs/canonical/full_imagery_overlap_shared_only.yaml \
    --override "training.seed=${SEED}" \
    --override "training.output_dir=outputs/seed_stability/shared_only/seed${SEED}" \
    --override "training.device=cuda"
done

# SP p16, 3 seeds
for SEED in 0 42 123; do
  ./.venv/bin/python -m fmri2img.workflows.train_decoder \
    --config configs/canonical/threshold_shared_private_p16.yaml \
    --override "training.seed=${SEED}" \
    --override "training.output_dir=outputs/seed_stability/sp_p16/seed${SEED}" \
    --override "training.device=cuda"
done

# Evaluation for each
for MODEL_DIR in shared_only sp_p16; do
  CONFIG="configs/canonical/full_imagery_overlap_shared_only.yaml"
  [ "$MODEL_DIR" = "sp_p16" ] && CONFIG="configs/canonical/threshold_shared_private_p16.yaml"
  for SEED in 0 42 123; do
    ./.venv/bin/python -m fmri2img.workflows.eval_decoder \
      --config "$CONFIG" \
      --checkpoint "outputs/seed_stability/${MODEL_DIR}/seed${SEED}/best_decoder.pt" \
      --override "evaluation.output_dir=outputs/seed_stability/${MODEL_DIR}/seed${SEED}/eval" \
      --override "training.device=cuda"
  done
done
```

### Summary results

| Model | Seed | Test Cosine | Test MSE |
|---|---|---|---|
| Shared-only | 0 | 0.079 | 0.00240 |
| Shared-only | 42 | 0.073 | 0.00241 |
| Shared-only | 123 | 0.150 | 0.00221 |
| **Shared-only mean ± std** | | **0.101 ± 0.035** | **0.00234 ± 0.00009** |
| SP, d_priv=16 | 0 | 0.002 | 0.00260 |
| SP, d_priv=16 | 42 | 0.048 | 0.00248 |
| SP, d_priv=16 | 123 | 0.010 | 0.00258 |
| **SP p16 mean ± std** | | **0.020 ± 0.020** | **0.00255 ± 0.00005** |

The ordering (shared-only > SP p16) is consistent across all 3 seeds. The
worst shared-only seed (0.073) exceeds the best SP p16 seed (0.048).

Machine-readable summary: `seed_stability_summary.json`

---

## Bootstrap Confidence Intervals

A 10,000-resample bootstrap analysis was run over the 19-row test set to
assess whether pairwise ordering differences are statistically robust. Neural
model cosines are seed-averaged (over seeds 0, 42, 123) before bootstrapping.

### Pairwise ordering CIs (95%)

| Comparison | Mean diff | 95% CI | Excludes zero? |
|---|---|---|---|
| Ridge − Shared-only | +0.451 | [0.404, 0.495] | Yes |
| Shared-only − SP d₁₆ | +0.081 | [0.074, 0.087] | Yes |
| Ridge − SP d₁₆ | +0.532 | [0.487, 0.576] | Yes |

All pairwise ordering differences exclude zero at the 95% level.

### Per-seed ordering CIs (shared-only − SP d₁₆)

| Seed | Mean diff | 95% CI | Excludes zero? |
|---|---|---|---|
| 0 | +0.077 | [0.063, 0.094] | Yes |
| 42 | +0.026 | [0.020, 0.030] | Yes |
| 123 | +0.140 | [0.127, 0.151] | Yes |

Machine-readable results: `bootstrap_ci_analysis.json`

---

## Per-Condition Breakdown

The test set contains 3 perception and 16 imagery trials, all from subj02.

| Model | Overall | Perception (n=3) | Imagery (n=16) |
|---|---|---|---|
| Ridge | 0.552 | 0.554 | 0.552 |
| Shared-only (seed-avg) | 0.101 ± 0.035 | 0.108 ± 0.026 | 0.100 ± 0.037 |
| SP, d₁₆ (seed-avg) | 0.020 ± 0.020 | 0.050 ± 0.006 | 0.014 ± 0.024 |

Machine-readable results: `per_condition_breakdown.json`
