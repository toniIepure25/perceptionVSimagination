# Project Status — Single Source of Truth

> **Version**: 0.3.0  
> **Last commit**: `0d162f4` (main)  
> **Last updated**: March 12, 2026

---

## Executive Summary

| Aspect | Status |
|--------|--------|
| **Perception pipeline** | ✅ Complete — 28 models trained, all 4 subjects |
| **Analysis modules** | ✅ 19 modules code-complete, **zero real results** |
| **NSD-Imagery data** | ❌ **NOT DOWNLOADED** — critical blocker |
| **Cross-project bridge** | 🔧 In progress — external loader for FMRI2images |
| **Tests** | ✅ 51 passing |
| **Docs** | ✅ Fully overhauled |

---

## Trained Models

### Perception Baselines (shared-1000 evaluation)

| Subject | Architecture | R@1 (%) | R@5 (%) | Median Rank | Status |
|---------|-------------|---------|---------|-------------|--------|
| subj01  | MLP         | 5.7     | 16.5    | ~90         | ✅ |
| subj01  | Ridge       | 4.2     | 13.1    | ~120        | ✅ |
| subj01  | CLIP Adapter| 3.8     | 11.9    | ~130        | ✅ |
| subj02  | MLP         | 4.9     | 14.2    | ~100        | ✅ |
| subj02  | Ridge       | 3.6     | 11.5    | ~125        | ✅ |
| subj05  | MLP         | 4.1     | 12.8    | ~110        | ✅ |
| subj07  | MLP         | 3.5     | 11.0    | ~135        | ✅ |

> **Note**: 28 total checkpoints across subjects/architectures/configs. Only top results shown.  
> All numbers from dry-run evaluation on shared-1000 perception set.

### FMRI2images Comparison (separate project)

| Metric | This Project (ViT-L/14) | FMRI2images (ViT-bigG/14) |
|--------|--------------------------|---------------------------|
| R@1    | ~5.7%                    | ~58%                      |
| CSLS R@1| —                       | ~70%                      |
| Params | 6.3M                     | 825M                      |
| CLIP   | ViT-L/14 768-d CLS      | ViT-bigG/14 1280-d × 257 tokens |
| Input  | PCA 3072                 | 15,724 raw voxels         |

The R@1 gap is explained by: target quality (bigG >> L/14), input representation (raw voxels vs PCA), model capacity (130× larger), and training recipe (vMF-NCE + SoftCLIP + MixCo + EMA vs InfoNCE alone).

---

## Analysis Modules (19 total)

All modules are code-complete but have only been tested with synthetic or dry-run data. **No real imagery results exist yet.**

### Tier 1 — Core Metrics

| # | Module | File | Synthetic Test | Real Data |
|---|--------|------|----------------|-----------|
| 1 | RSA (Representational Similarity) | `src/fmri2img/analysis/rsa.py` | ✅ | ❌ |
| 2 | CKA (Centered Kernel Alignment) | `src/fmri2img/analysis/cka.py` | ✅ | ❌ |
| 3 | Voxel Reliability | `src/fmri2img/analysis/reliability.py` | ✅ | ❌ |
| 4 | Noise-Ceiling Estimation | `src/fmri2img/analysis/noise_ceiling.py` | ✅ | ❌ |
| 5 | Domain Confusion / Classifier | `src/fmri2img/analysis/domain_confusion.py` | ✅ | ❌ |

### Tier 2 — Transfer & Adaptation

| # | Module | File | Synthetic Test | Real Data |
|---|--------|------|----------------|-----------|
| 6 | Cross-Domain Transfer Eval | `src/fmri2img/analysis/transfer_eval.py` | ✅ | ❌ |
| 7 | Domain Adaptation (DANN) | `src/fmri2img/analysis/domain_adaptation.py` | ✅ | ❌ |
| 8 | Imagery Adapter (LoRA / linear) | `src/fmri2img/adapters/imagery_adapter.py` | ✅ | ❌ |
| 9 | Soft Retrieval | `src/fmri2img/analysis/soft_retrieval.py` | ✅ | ❌ |
| 10 | Uncertainty Estimation | `src/fmri2img/analysis/uncertainty.py` | ✅ | ❌ |

### Tier 3 — Advanced

| # | Module | File | Synthetic Test | Real Data |
|---|--------|------|----------------|-----------|
| 11 | Barlow Twins Probe | `src/fmri2img/analysis/barlow_twins.py` | ✅ | ❌ |
| 12 | VICReg Probe | `src/fmri2img/analysis/vicreg.py` | ✅ | ❌ |
| 13 | Topographic Analysis | `src/fmri2img/analysis/topographic.py` | ✅ | ❌ |
| 14 | Temporal Dynamics | `src/fmri2img/analysis/temporal_dynamics.py` | ✅ | ❌ |
| 15 | Feature Attribution (Grad-CAM) | `src/fmri2img/analysis/feature_attribution.py` | ✅ | ❌ |

### Cross-Cutting

| # | Module | File | Synthetic Test | Real Data |
|---|--------|------|----------------|-----------|
| 16 | Composite Score | `src/fmri2img/analysis/composite_score.py` | ✅ | ❌ |
| 17 | Stats (permutation tests) | `src/fmri2img/analysis/stats.py` | ✅ | ❌ |
| 18 | Loss Functions (InfoNCE, SoftCLIP, MixCo) | `src/fmri2img/losses/` | ✅ | ❌ |
| 19 | Manifold Geometry | `src/fmri2img/analysis/manifold_geometry.py` | ✅ | ❌ |

---

## Critical Blockers

### 1. NSD-Imagery Data (BLOCKING)

The NSD-Imagery fMRI betas have **never been downloaded**. Without this data, none of the 19 analysis modules can produce real perception-vs-imagery results.

**Action required**:
1. Obtain access to OpenNeuro ds004937 (or NSD S3 bucket `nsdimagery/` prefix)
2. Download imagery betas for subj01 (at minimum)
3. Run `scripts/build_nsd_imagery_index.py` to build canonical indices
4. Run preprocessing via `scripts/fit_preprocessing.py --domain imagery`

See [NSD_IMAGERY_DATASET_GUIDE.md](../technical/NSD_IMAGERY_DATASET_GUIDE.md) for details.

### 2. Cross-Project Embedding Mismatch (DESIGN NEEDED)

This project uses ViT-L/14 (768-d CLS). FMRI2images uses ViT-bigG/14 (1280-d × 257 tokens). Direct checkpoint transfer is impossible. Planned approaches:
- **Prediction-level comparison**: Run both decoders on same stimuli, compare CLIP cosine similarity of predictions
- **Shared evaluation set**: shared-1000 perception images → decode with both systems → compare
- **Projection head**: Train a small linear layer to project bigG CLS (1280-d) → L/14 CLS (768-d)

---

## Test Suite

```
51 tests passing (pytest)
```

Key test files:
- `tests/test_adapters_simple.py` — adapter architecture tests
- `tests/test_brain_alignment.py` — CKA/RSA on synthetic data
- `tests/test_losses.py` — InfoNCE, SoftCLIP, MixCo losses
- `tests/test_imagery_adapter.py` — imagery adapter with mock data
- `tests/test_stats.py` — permutation tests, bootstrap CI
- `tests/test_uncertainty.py` — uncertainty estimation

---

## File Inventory

### Checkpoints (`checkpoints/`)

| Directory | Contents |
|-----------|----------|
| `clip_adapter/subj01/` | CLIP adapter models |
| `clip_to_fmri/subj01/` | CLIP-to-fMRI mapping models |
| `mlp/` | MLP encoder models |
| `ridge/` | Ridge regression baselines |
| `two_stage/` | Two-stage pipeline models |

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_full_pipeline.py` | End-to-end training pipeline |
| `scripts/eval_shared1000_full.py` | Shared-1000 evaluation |
| `scripts/eval_perception_to_imagery_transfer.py` | Cross-domain transfer (needs imagery data) |
| `scripts/fit_preprocessing.py` | Feature extraction & PCA |
| `scripts/build_nsd_imagery_index.py` | Build NSD-Imagery indices |
| `scripts/train_imagery_adapter.py` | Train imagery adapter (needs imagery data) |
| `scripts/run_novel_analyses.py` | Run all 19 analysis modules |
| `scripts/run_imagery_ablations.py` | Ablation studies |
| `scripts/make_paper_figures.py` | Generate publication figures |

---

## Version History

| Version | Commit | Description |
|---------|--------|-------------|
| 0.3.0 | `0d162f4` | 19 analysis modules, imagery adapter, full test suite |
| 0.2.x | — | Preprocessing pipeline, shared-1000 eval |
| 0.1.x | — | Initial perception training, Ridge + MLP baselines |

---

## Next Actions (Priority Order)

1. **Download NSD-Imagery data** → unblocks all 19 modules
2. **Run real cross-domain eval** → first genuine perception-vs-imagery results
3. **Build external model loader** → bridge to FMRI2images predictions
4. **Run full analysis battery** → populate this STATUS.md with real numbers
5. **Write paper** → structure already in PAPER_DRAFT_OUTLINE.md
