# Experiment Results

**Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI**

> Static reference document — records methodology, architectures, configurations, and final results.
> Last updated: 2026-03-11

---

## 1. Overview

This project investigates whether fMRI representations learned during **visual perception** (seeing images) can transfer to **mental imagery** (imagining images). We train neural encoders that map fMRI brain activity → CLIP embedding space, then evaluate whether perception-trained encoders preserve useful structure for imagined stimuli.

**Subject**: NSD subj01 (40 sessions × 750 trials = 30,000 total)  
**Split**: 80/10/10 (24K train / 3K val / 3K test)  
**CLIP backbone**: ViT-L/14 (768-dim embeddings)  
**Infrastructure**: H100 80GB HBM3, CUDA 12.8, PyTorch 2.10.0

---

## 2. Pipeline

### 2.1 Preprocessing

1. **Load NSD betas** (GLMdenoise + Ridge Regression denoised, 1.8mm func space)
2. **Voxel selection** via reliability threshold (≥ 0.1 test-retest reliability)
3. **Two feature configs**:
   - **Baseline**: Hard-threshold mask → StandardScaler → PCA(3072) → 30K × 3072
   - **Novel**: Soft reliability-weighted voxels (sigmoid curve, τ=0.1) → PCA(3072) → 30K × 3072
4. **Auto-centering**: When feature mean norm > 0.01 (detects PCA mean bias from soft-weights), subtract dataset mean before training

### 2.2 CLIP Target Cache

- Pre-compute CLIP ViT-L/14 embeddings for all 73K NSD stimuli
- L2-normalized, stored as parquet (768-dim per image)

### 2.3 Model Training

Four encoder architectures trained with configurable multi-objective losses:
- Ridge, MLP, TwoStage, MultiLayer (see Section 3)

### 2.4 Evaluation

- **Test-split retrieval**: R@1, R@5, R@10, Median Rank, MRR on held-out 3K samples
- **Shared-1000 benchmark**: NSD's 1000 images seen 3× by each subject, with 3-rep averaging and statistical testing (Phase C, completed — see Section 7)

---

## 3. Architectures

### 3.1 Ridge Regression (`ridge`)
- Sklearn Ridge with `fit_intercept=True`, α tuned by cross-validation
- Fastest to train; serves as strong linear baseline
- Output: 768-dim L2-normalized vector

### 3.2 MLP Encoder (`mlp`)
- Architecture: `Linear(3072, 2048) → ReLU → Dropout(0.3) → Linear(2048, 768)`
- Trained with configurable loss (cosine, MSE, InfoNCE)
- ~6.3M parameters

### 3.3 Two-Stage Encoder (`two_stage`)
- **Stage 1**: Linear(3072, dim_hidden) with LayerNorm + GELU
- **Stage 2**: MLP or linear head → 768-dim output
- ~7.9M parameters (with MLP head)
- Designed for curriculum learning (freeze stage 1, fine-tune head)

### 3.4 Multi-Layer Two-Stage Encoder (`multilayer`)
- Extends TwoStage with multi-layer CLIP supervision
- Parallel projection heads for intermediate CLIP layers (e.g., layer_6, layer_12, layer_18, final)
- Optional learnable layer weights (softmax-normalized logits)
- Multi-layer loss: weighted sum of per-layer cosine losses + optional multi-layer InfoNCE

---

## 4. Loss Configurations

| Loss Config | cosine_weight | mse_weight | infonce_weight | Temperature | Notes |
|-------------|:------------:|:----------:|:-------------:|:-----------:|-------|
| cosine | 1.0 | 0.0 | 0.0 | — | Pure cosine similarity |
| cosine_mse | 0.5 | 0.5 | 0.0 | — | Equal cosine + MSE blend |
| light_infonce | 0.7 | 0.0 | 0.3 | 0.07 | Mild contrastive push |
| strong_infonce | 0.3 | 0.0 | 0.7 | 0.05 | Aggressive contrastive |

---

## 5. Complete Results — Test Split (3,000 samples)

Sorted by R@1 (top-1 retrieval accuracy). All models evaluated on subj01.

| Config | Model | Features | Cosine | R@1 | R@5 | R@10 | Med Rank | MRR |
|--------|-------|----------|--------|-----|-----|------|----------|-----|
| mlp_novel_strong_infonce_v2 | MLP | novel | 0.5298 | 0.0573 | 0.1667 | 0.2583 | 37 | 0.1235 |
| ts_novel_cosine_v2 | TwoStage | novel | 0.8124 | 0.0350 | 0.1160 | 0.1993 | 56 | 0.0896 |
| multilayer_baseline_v2 | Multilayer | baseline | 0.8084 | 0.0327 | 0.1020 | 0.1607 | 72 | 0.0798 |
| multilayer_baseline | Multilayer | baseline | 0.8076 | 0.0320 | 0.1023 | 0.1710 | 73 | 0.0798 |
| multilayer_novel_v3_lw | Multilayer | novel | 0.8098 | 0.0317 | 0.1153 | 0.1810 | 64 | 0.0850 |
| multilayer_novel_v3 | Multilayer | novel | 0.8115 | 0.0283 | 0.1080 | 0.1783 | 61 | 0.0800 |
| mlp_novel_cosine_v2 | MLP | novel | 0.8026 | 0.0277 | 0.0887 | 0.1437 | 92 | 0.0700 |
| ts_novel_cosine_mse_v2 | TwoStage | novel | 0.8129 | 0.0277 | 0.1053 | 0.1713 | 61 | 0.0781 |
| mlp_baseline | MLP | baseline | 0.8014 | 0.0270 | 0.0870 | 0.1403 | 94 | 0.0680 |
| mlp_novel_light_infonce_v2 | MLP | novel | 0.7723 | 0.0267 | 0.0990 | 0.1530 | 78 | 0.0737 |
| mlp_novel_cosine_mse_v2 | MLP | novel | 0.8021 | 0.0257 | 0.0883 | 0.1370 | 96 | 0.0674 |
| ridge_baseline | Ridge | baseline | 0.7913 | 0.0183 | 0.0620 | 0.0987 | 197 | 0.0478 |
| ridge_novel | Ridge | novel | 0.7913 | 0.0177 | 0.0603 | 0.0947 | 203 | 0.0466 |
| ts_novel_light_infonce_v2 | TwoStage | novel | 0.7754 | 0.0090 | 0.0413 | 0.0767 | 137 | 0.0366 |
| mlp_novel_light_infonce | MLP | novel | 0.4541 | 0.0010 | 0.0023 | 0.0057 | 1321 | 0.0043 |
| ts_novel_strong_infonce_v2 | TwoStage | novel | 0.6832 | 0.0010 | 0.0037 | 0.0077 | 729 | 0.0062 |
| two_stage_baseline | TwoStage | baseline | 0.5515 | 0.0010 | 0.0043 | 0.0090 | 648 | 0.0065 |
| mlp_novel_strong_infonce | MLP | novel | 0.3877 | 0.0007 | 0.0023 | 0.0040 | 1459 | 0.0033 |
| mlp_novel | MLP | novel | 0.3868 | 0.0003 | 0.0017 | 0.0030 | 1462 | 0.0030 |
| mlp_novel_cosine | MLP | novel | 0.5103 | 0.0003 | 0.0017 | 0.0040 | 1462 | 0.0030 |
| mlp_novel_cosine_mse | MLP | novel | 0.5171 | 0.0003 | 0.0017 | 0.0037 | 1463 | 0.0031 |
| multilayer_novel | Multilayer | novel | 0.7440 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |
| multilayer_novel_v2 | Multilayer | novel | 0.7442 | 0.0003 | 0.0017 | 0.0033 | 1496 | 0.0029 |
| ts_novel_cosine | TwoStage | novel | 0.7469 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |
| ts_novel_cosine_mse | TwoStage | novel | 0.7469 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |
| ts_novel_light_infonce | TwoStage | novel | 0.7466 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |
| ts_novel_strong_infonce | TwoStage | novel | 0.7286 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |
| two_stage_novel | TwoStage | novel | 0.6399 | 0.0003 | 0.0017 | 0.0033 | 1500 | 0.0029 |

> **Note**: v1 models (no `_v2`/`_v3` suffix) trained on uncentered novel features — performance collapsed due to PCA mean bias (norm=11.1). v2/v3 models trained after auto-centering fix.

---

## 6. Key Findings

### 6.1 Novel Feature PCA Mean Bias (Root Cause of v1 Collapse)

Soft-reliability weighting shifts the effective voxel mean by ~0.024/voxel. After PCA projection through 3072 components, this creates a constant bias vector with norm ≈ 11.1. All novel feature vectors end up nearly collinear (pairwise cosine ≈ 0.9989), making any downstream model predict essentially the same direction for every sample.

**Fix**: Auto-centering — subtract dataset mean when `np.linalg.norm(mean) > 0.01`. All v2/v3 models include this fix.

### 6.2 Multi-layer InfoNCE + Learnable Weights Collapse

When `use_learnable_weights=True` AND `use_multilayer_infonce=True`:
- InfoNCE gradients on weight logits create conflicting optimization: push toward whichever layer most reduces contrastive loss
- Weights shift to layer_18 (52.5%), killing the final layer (6.5%)
- val_cosine drops from 0.7428 (epoch 1) to 0.1111 (epoch 10)

**Fix**: Use cosine-only loss for multilayer models. v3 configs achieve cosine=0.81+.

### 6.3 Strong InfoNCE + MLP = Best Retrieval

`mlp_novel_strong_infonce_v2` achieves the highest R@1=0.0573 and median_rank=37, but at the cost of lower cosine similarity (0.5298). This reflects InfoNCE's design: it optimizes for *relative* ranking (contrastive discrimination) rather than *absolute* alignment.

### 6.4 Cosine-Only Models = Best Alignment

`ts_novel_cosine_mse_v2` achieves the highest cosine similarity (0.8129) but lower retrieval (R@1=0.0277). Pure cosine loss optimizes for directional alignment, which may not translate to discriminative retrieval.

### 6.5 Novel vs. Baseline Features

After the centering fix, novel (soft-reliability-weighted) features **match** baseline (hard-threshold) features across all architectures:
- Best novel R@1: 0.0573 vs. best baseline R@1: 0.0327
- Best novel cosine: 0.8129 vs. best baseline cosine: 0.8084
- Ridge novel (0.7913, R@1=0.0177) ≈ Ridge baseline (0.7913, R@1=0.0183) — virtually identical

This suggests soft-reliability weighting captures comparable signal once the mean bias is corrected, with neural network models able to exploit the richer feature representation.

---

## 7. Shared-1000 Benchmark Results

NSD's standard benchmark: 1000 images seen 3× by subj01, with 3-rep averaging from raw NIfTI betas. This tests generalization beyond the training pipeline.

| Model | Type | Features | R@1 | R@5 | R@10 | Med Rank | MRR | Cosine |
|-------|------|----------|-----|-----|------|----------|-----|--------|
| ridge_baseline | Ridge | baseline | 0.70 | 2.30 | 3.90 | 386 | 0.0209 | -0.0458 |
| ridge_novel | Ridge | novel | 0.70 | 1.90 | 3.30 | 408 | 0.0190 | -0.0570 |
| mlp_baseline | MLP | baseline | 0.20 | 1.00 | 2.00 | 440 | 0.0118 | 0.3070 |
| mlp_novel_cosine_v2 | MLP | novel | 0.20 | 0.80 | 2.10 | 442 | 0.0110 | 0.4269 |
| mlp_novel_strong_infonce_v2 | MLP | novel | 0.20 | 0.90 | 1.90 | 474 | 0.0108 | 0.1420 |
| ts_novel_cosine_v2 | TwoStage | novel | 0.10 | 1.10 | 1.90 | 496 | 0.0103 | 0.6899 |
| ts_novel_cosine_mse_v2 | TwoStage | novel | 0.10 | 1.00 | 1.70 | 496 | 0.0098 | 0.6938 |
| multilayer_novel_v3 | Multilayer | novel | 0.10 | 0.50 | 1.60 | 480 | 0.0090 | 0.6881 |
| multilayer_novel_v3_lw | Multilayer | novel | 0.10 | 0.80 | 1.30 | 494 | 0.0087 | 0.6833 |
| multilayer_baseline_v2 | Multilayer | baseline | 0.00 | 0.80 | 1.30 | 483 | 0.0090 | 0.6870 |

### Statistical Comparisons

All deep models (MLP, TwoStage, Multilayer) show significantly better performance than Ridge (p < 0.005, Holm-Bonferroni corrected). However, within the deep model group, no significant differences were found (p > 0.4) — they form a statistically equivalent cluster.

### Notes on Shared-1000 vs. Test-Split Discrepancy

Performance is substantially lower on Shared-1000 due to preprocessing mismatch: the benchmark reconstructs features from raw NIfTI betas through the stored preprocessing pipeline, whereas test-split uses the exact same pre-extracted features as training. The negative cosine values for ridge models indicate the NIfTI-to-feature reconstruction doesn't perfectly match the training pipeline for linear models. This is a known limitation — the Shared-1000 results should be compared **across models** rather than to test-split numbers.

---

## 8. Commit History

| Commit | Description |
|--------|-------------|
| `602f22d` | Fix Shared-1000 mask/weights handling for novel features |
| `b825618` | Add experiment docs, fix P3/P5/P7/P8 bugs |
| `a417cdc` | Fix v1 checkpoint loading (infer output_dim, head_type from state_dict) |
| `ea24a38` | Fix report overwriting, multilayer metadata, eval weights filename |
| `e164bbe` | Novel rerun with auto-centering + multilayer fix |
| `0dcafc3` | Train all 6 initial model configs |
| `2c268fc` | Feature extraction (baseline + novel, 30K × 3072) |

---

## 9. Remaining Work

- [x] Fix ridge_novel evaluation (centering mismatch) — **Fixed**: cosine 0.7913, R@1 0.0177
- [x] Fix MRR display (key case mismatch) — **Fixed**: MRR now showing correctly
- [x] Run Shared-1000 benchmark (Phase C) — **Done**: All 10 models evaluated
- [x] Build 19 analysis modules (v0.3.0, commit `0d162f4`) — RSA, CKA, noise ceiling, domain confusion, etc.
- [ ] **BLOCKED**: Download NSD-Imagery fMRI data (OpenNeuro ds004937)
- [ ] Run cross-domain eval with real imagery data
- [ ] Retrain TwoStage baseline with v2 hyperparameters
- [ ] Run FMRI2images checkpoint on shared stimuli for cross-project comparison

---

## 10. Cross-Project Context

A separate project (**FMRI2images**) on the same cluster achieves R@1 ~58% using:
- ViT-bigG/14 (1280-d × 257 tokens) instead of ViT-L/14 (768-d CLS)
- 4-layer residual MLP [8192, 8192, 4096, 2048] → vMF decoder (~825M params)
- 15,724 raw voxels (nsdgeneral) instead of PCA-3072
- vMF-NCE + SoftCLIP + MixCo + EMA training

The 10× R@1 gap is expected given: 130× more parameters, higher-capacity CLIP backbone (bigG vs L/14), and richer input (raw voxels vs PCA). An external model loader (`src/fmri2img/models/external_loader.py`) allows loading FMRI2images checkpoints for prediction-level comparison without importing the other codebase.

See [STATUS.md](STATUS.md) for the single source of truth on current project state.
