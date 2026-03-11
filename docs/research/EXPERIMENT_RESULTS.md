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
- **Shared-1000 benchmark**: NSD's 1000 images seen 3× by each subject, with 3-rep averaging and statistical testing (Phase C, pending)

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

| Config | Model | Features | Cosine | R@1 | R@5 | R@10 | Med Rank |
|--------|-------|----------|--------|-----|-----|------|----------|
| mlp_novel_strong_infonce_v2 | MLP | novel | 0.5298 | 0.0573 | 0.1667 | 0.2583 | 37 |
| ts_novel_cosine_v2 | TwoStage | novel | 0.8124 | 0.0350 | 0.1160 | 0.1993 | 56 |
| multilayer_baseline_v2 | Multilayer | baseline | 0.8084 | 0.0327 | 0.1020 | 0.1607 | 72 |
| multilayer_baseline | Multilayer | baseline | 0.8076 | 0.0320 | 0.1023 | 0.1710 | 73 |
| multilayer_novel_v3_lw | Multilayer | novel | 0.8098 | 0.0317 | 0.1153 | 0.1810 | 64 |
| multilayer_novel_v3 | Multilayer | novel | 0.8115 | 0.0283 | 0.1080 | 0.1783 | 61 |
| mlp_novel_cosine_v2 | MLP | novel | 0.8026 | 0.0277 | 0.0887 | 0.1437 | 92 |
| ts_novel_cosine_mse_v2 | TwoStage | novel | 0.8129 | 0.0277 | 0.1053 | 0.1713 | 61 |
| mlp_baseline | MLP | baseline | 0.8014 | 0.0270 | 0.0870 | 0.1403 | 94 |
| mlp_novel_light_infonce_v2 | MLP | novel | 0.7723 | 0.0267 | 0.0990 | 0.1530 | 78 |
| mlp_novel_cosine_mse_v2 | MLP | novel | 0.8021 | 0.0257 | 0.0883 | 0.1370 | 96 |
| ridge_baseline | Ridge | baseline | 0.7913 | 0.0183 | 0.0620 | 0.0987 | 197 |
| ts_novel_light_infonce_v2 | TwoStage | novel | 0.7754 | 0.0090 | 0.0413 | 0.0767 | 137 |
| mlp_novel_light_infonce | MLP | novel | 0.4541 | 0.0010 | 0.0023 | 0.0057 | 1321 |
| ts_novel_strong_infonce_v2 | TwoStage | novel | 0.6832 | 0.0010 | 0.0037 | 0.0077 | 729 |
| two_stage_baseline | TwoStage | baseline | 0.5515 | 0.0010 | 0.0043 | 0.0090 | 648 |
| mlp_novel_strong_infonce | MLP | novel | 0.3877 | 0.0007 | 0.0023 | 0.0040 | 1459 |
| mlp_novel | MLP | novel | 0.3868 | 0.0003 | 0.0017 | 0.0030 | 1462 |
| mlp_novel_cosine | MLP | novel | 0.5103 | 0.0003 | 0.0017 | 0.0040 | 1462 |
| mlp_novel_cosine_mse | MLP | novel | 0.5171 | 0.0003 | 0.0017 | 0.0037 | 1463 |
| multilayer_novel | Multilayer | novel | 0.7440 | 0.0003 | 0.0017 | 0.0033 | 1500 |
| multilayer_novel_v2 | Multilayer | novel | 0.7442 | 0.0003 | 0.0017 | 0.0033 | 1496 |
| ridge_novel | Ridge | novel | -0.0036 | 0.0003 | 0.0020 | 0.0040 | 1404 |
| ts_novel_cosine | TwoStage | novel | 0.7469 | 0.0003 | 0.0017 | 0.0033 | 1500 |
| ts_novel_cosine_mse | TwoStage | novel | 0.7469 | 0.0003 | 0.0017 | 0.0033 | 1500 |
| ts_novel_light_infonce | TwoStage | novel | 0.7466 | 0.0003 | 0.0017 | 0.0033 | 1500 |
| ts_novel_strong_infonce | TwoStage | novel | 0.7286 | 0.0003 | 0.0017 | 0.0033 | 1500 |
| two_stage_novel | TwoStage | novel | 0.6399 | 0.0003 | 0.0017 | 0.0033 | 1500 |

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

After the centering fix, novel (soft-reliability-weighted) features generally **match or slightly exceed** baseline (hard-threshold) features:
- Best novel R@1: 0.0573 vs. best baseline R@1: 0.0327
- Best novel cosine: 0.8129 vs. best baseline cosine: 0.8084

This suggests soft-reliability weighting captures additional signal once the mean bias is corrected.

---

## 7. Commit History

| Commit | Description |
|--------|-------------|
| `a417cdc` | Fix v1 checkpoint loading (infer output_dim, head_type from state_dict) |
| `ea24a38` | Fix report overwriting, multilayer metadata, eval weights filename |
| `e164bbe` | Novel rerun with auto-centering + multilayer fix |
| `0dcafc3` | Train all 6 initial model configs |
| `2c268fc` | Feature extraction (baseline + novel, 30K × 3072) |

---

## 8. Remaining Work

- [ ] Fix ridge_novel evaluation (centering mismatch)
- [ ] Run Shared-1000 benchmark (Phase C)
- [ ] Retrain TwoStage baseline with v2 hyperparameters
- [ ] Imagery adapter training and cross-domain evaluation (Phase D)
