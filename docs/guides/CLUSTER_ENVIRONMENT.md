# Cluster Environment & Infrastructure

## Hardware Specification

| Component         | Details                                            |
| ----------------- | -------------------------------------------------- |
| **GPU**           | 1x NVIDIA H100 80GB HBM3                           |
| **CPU**           | 32 cores                                           |
| **RAM**           | 100 GiB                                            |
| **Shared Memory** | 8 GiB                                              |
| **Local Storage** | `/home/jovyan/local-data` (hostPath, bypasses NFS) |

## Access

| Service        | Address                      | Credentials             |
| -------------- | ---------------------------- | ----------------------- |
| **Jupyter**    | `http://10.130.123.131:8888` | token: `orchestraiq`    |
| **SSH**        | `ssh jovyan@10.130.123.131`  | password: `orchestraiq` |
| **SSH (root)** | `ssh root@10.130.123.131`    | password: `orchestraiq` |

## Pre-installed Tools

`openssh`, `htop`, `tmux`, `screen`, `vim`, `nano`, `git`, `build-essential`, `net-tools`, `curl`, `wget`

## Software Environment (as of March 2026)

| Package          | Version         |
| ---------------- | --------------- |
| **Python**       | 3.13.12         |
| **PyTorch**      | 2.10.0+cu128    |
| **CUDA**         | 12.8            |
| **conda**        | at `/opt/conda` |
| **open_clip**    | installed       |
| **diffusers**    | installed       |
| **transformers** | installed       |

## Existing Data on Cluster

The cluster already has the fmri2img project at `/home/jovyan/work/fmri2img/` with:

### NSD Perception Data (`/home/jovyan/work/data/nsd/`)

- `nsddata/` — experiment metadata, ROI masks, design matrices
- `nsddata_betas/` — fMRI beta volumes (subj01, subj02, subj05, subj07 — 40 sessions each)
- `nsddata_stimuli/` — stimulus images (`nsd_stimuli.hdf5`, ~20GB)

### Pre-computed Artifacts

| Artifact                 | Path                                              | Status                                             |
| ------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| CLIP cache               | `outputs/clip_cache/clip.parquet`                 | ~30MB, complete                                    |
| NSD indices              | `data/indices/nsd_index/subj{01,02,05,07}.csv`    | Complete                                           |
| Preprocessing T1 scalers | `cache/preproc/subj{01,02,05,07}_t1_scaler.pkl`   | Complete                                           |
| Preprocessing T2 PCA     | `cache/preproc/subj{01,02,05,07}_t2_pca_k512.npz` | Complete                                           |
| Extracted features       | `outputs/preproc/*/subj{01,02,05,07}/`            | Complete for baseline/novel/soft_only/infonce_only |

### NSD-Imagery Data

**Status**: Not yet downloaded. Must be acquired separately (OpenNeuro ds004937 or NSD S3 bucket).

## Storage Strategy

- **NFS** (`/home/jovyan/work/`): shared, persists across restarts — use for code, configs, indices, checkpoints
- **Local** (`/home/jovyan/local-data/`): hostPath, faster I/O — use for large cache files (S3 cache, beta volumes) if NFS is slow
- **Shared memory** (`/dev/shm`): 8 GiB — limited; keep DataLoader `num_workers` moderate (4-6)

## Performance Notes

- H100 with CUDA 12.8 supports TF32, BF16, FP8 — always enable `mixed_precision: true`
- With 100GiB RAM, all preprocessing fits in memory; no need for streaming
- Shared memory is only 8GiB (down from typical 64GiB), so avoid `num_workers > 6` for large batch sizes
- Use `tmux` or `screen` for long-running training jobs so they survive SSH disconnects

## Quick Connection

```bash
# From local machine
sshpass -p orchestraiq ssh -o StrictHostKeyChecking=no jovyan@10.130.123.131

# Start a persistent session
tmux new -s experiment

# Navigate to project
cd /home/jovyan/work/fmri2img
```
