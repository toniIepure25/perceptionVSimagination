# Cluster Environment & Infrastructure

> Last Updated: March 12, 2026

## Hardware

| Component | Details |
|-----------|---------|
| **GPU** | 1× NVIDIA H100 80GB HBM3 |
| **CPU** | 32 cores |
| **RAM** | 100 GiB |
| **Shared Memory** | 8 GiB |
| **Local Storage** | `/home/jovyan/local-data` (hostPath, bypasses NFS) |

## Access

| Service | Address | Credentials |
|---------|---------|-------------|
| **Jupyter** | `http://10.130.123.131:8888` | token: `orchestraiq` |
| **SSH** | `ssh jovyan@10.130.123.131` | password: `orchestraiq` |
| **SSH (root)** | `ssh root@10.130.123.131` | password: `orchestraiq` |

> **Note**: Cluster restarts may change the host key. Clear with: `ssh-keygen -f ~/.ssh/known_hosts -R "10.130.123.131"`

## Software

| Package | Version |
|---------|---------|
| **Python** | 3.13.12 |
| **PyTorch** | 2.10.0+cu128 |
| **CUDA** | 12.8 |
| **conda** | `/opt/conda` |
| **open_clip** | installed |
| **diffusers** | installed |
| **transformers** | installed |

## Two Projects on the Cluster

### 1. perceptionVSimagination (this project)

**Location**: `/home/jovyan/local-data/perceptionVSimagination/`

**Persistent venv**: `/home/jovyan/local-data/venv`

```bash
# Activate
export PATH=/home/jovyan/local-data/venv/bin:$PATH
export VIRTUAL_ENV=/home/jovyan/local-data/venv
cd /home/jovyan/local-data/perceptionVSimagination
```

**Focus**: Cross-domain perception vs. imagery analysis. 28 trained models, 19 analysis modules. ViT-L/14, 768-d CLIP, PCA 3072 features.

### 2. FMRI2images (separate project)

**Location**: `/home/jovyan/work/FMRI2images/`

```bash
cd /home/jovyan/work/FMRI2images
```

**Focus**: State-of-the-art fMRI → image reconstruction.

| Aspect | Details |
|--------|---------|
| **Architecture** | 4-layer residual MLP [8192,8192,4096,2048] → vMF decoder |
| **CLIP backbone** | ViT-bigG/14 (LAION-2B), 1280-d × 257 tokens |
| **Input** | 15,724 raw voxels (nsdgeneral ROI) |
| **Model size** | ~825M params |
| **Training** | vMF-NCE + SoftCLIP + MixCo + EMA, bf16, queue 1024 |
| **Best checkpoint** | `experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt` |
| **Metrics** | R@1 ~58%, CSLS R@1 ~70% |

**Checkpoint structure**:
```python
# Load the FMRI2images best model
import torch
ckpt = torch.load('experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt',
                   map_location='cpu', weights_only=False)
# Keys: epoch, global_step, model_state_dict, optimizer_state_dict,
#        config, model_config, subject, ema_shadow
# Encoder input: 15724 → [8192, 8192, 4096, 2048]
# Decoder output: 2048 → 328960 (257 tokens × 1280 dim)
```

## NSD Data on Cluster

### Perception Data (`/home/jovyan/work/data/nsd/`)

- `nsddata/` — experiment metadata, ROI masks, design matrices
- `nsddata_betas/` — fMRI beta volumes (subj01, subj02, subj05, subj07 — 40 sessions)
- `nsddata_stimuli/` — stimulus images (`nsd_stimuli.hdf5`, ~20GB)

### Pre-computed Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| CLIP cache | `outputs/clip_cache/clip.parquet` | Complete (~30MB) |
| NSD indices | `data/indices/nsd_index/subj{01,02,05,07}.csv` | Complete |
| Preprocessing scalers | `cache/preproc/subj{01,02,05,07}_t1_scaler.pkl` | Complete |
| Preprocessing PCA | `cache/preproc/subj{01,02,05,07}_t2_pca_k512.npz` | Complete |
| Extracted features | `outputs/preproc/*/subj{01,02,05,07}/` | Complete (baseline/novel/soft_only/infonce_only) |

### NSD-Imagery Data

**Status**: NOT YET DOWNLOADED.

Must be acquired from OpenNeuro ds004937 or the NSD S3 bucket. See `docs/technical/NSD_IMAGERY_DATASET_GUIDE.md`.

## Storage Strategy

| Location | Type | Use For |
|----------|------|---------|
| `/home/jovyan/work/` | NFS (shared, persistent) | Code, configs, indices, checkpoints |
| `/home/jovyan/local-data/` | hostPath (fast I/O) | Venv, large cache files |
| `/dev/shm` | Shared memory (8 GiB) | Keep DataLoader `num_workers` ≤ 6 |

## Performance Notes

- H100 supports TF32, BF16, FP8 — always enable `mixed_precision: true`
- With 100 GiB RAM, all preprocessing fits in memory
- Shared memory is only 8 GiB — avoid `num_workers > 6` for large batches
- Use `tmux` or `screen` for long-running training jobs

## Quick Connection

```bash
# From local machine
sshpass -p orchestraiq ssh -o StrictHostKeyChecking=no jovyan@10.130.123.131

# Start persistent session
tmux new -s experiment

# Navigate to this project
cd /home/jovyan/local-data/perceptionVSimagination
export PATH=/home/jovyan/local-data/venv/bin:$PATH

# Or to FMRI2images
cd /home/jovyan/work/FMRI2images
```
