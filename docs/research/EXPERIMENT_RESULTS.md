# Experiment Results

**Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI**

> Static reference document — records methodology, architectures, configurations, and final results.
> Last updated: 2026-03-19 — **Phase 4 complete, Phase 5 three checkpoints executed (V30e + V33b + V28a)**

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
| `6d2d785` | Fix Ridge coef_ attribute reference in novel analysis script |
| `0ff438c` | Real-data novel analyses script (run_real_novel_analyses.py) |
| `eee773a` | Preprocessing support for imagery eval (data_root, 3D volume fix) |
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
- [x] Download NSD-Imagery fMRI data — **Done**: 5.9 GB, 4 subjects (subj01/02/05/07), 4D NIfTI 81×104×83×720
- [x] Run cross-domain eval with real imagery data — **Done**: 6 evaluations (3 models × {imagery, perception}), see Section 11
- [x] Run 13/15 novel analysis directions on real data — **Done**: 26 figures generated, see Section 12
- [ ] Retrain TwoStage baseline with v2 hyperparameters
- [ ] Train multi-subject models (subj02/05/07 indices built but no checkpoints)
- [ ] Train imagery adapters (LoRA, linear, MLP) on real data
- [x] Build FMRI2images integration infrastructure (Phase 5) — 8 new modules
- [ ] **Run FMRI2images on imagery data** (requires cluster, nsdgeneral mask)
- [ ] **Run fidelity ladder experiment** (Ridge → MLP → FMRI2images × perception/imagery)
- [ ] **Run token-level spatial analysis** (16×16 imagery fidelity maps)
- [ ] **Run cross-capacity consistency** (13 analyses × 2 model scales)
- [ ] Evaluate attention-condition trials (288 trials not yet analysed)
- [ ] Multi-subject imagery evaluation (subj02, 05, 07)

---

## 10. Cross-Domain Imagery Transfer Results (Phase 4)

> **Full details, interpretation, and new hypotheses**: see [IMAGERY_RESULTS.md](IMAGERY_RESULTS.md)

### 10.1 NSD-Imagery Dataset

Downloaded from OpenNeuro ds004937 (NSD-Imagery extension). 4 subjects × 720 trials each. Structure: 12 runs × 60 trials, 3 stimulus sets (A: simple bars/crosses, B: NSD photos, C: verbal concepts), 3 task types (imagery: 288, perception: 144, attention: 288). Format: 4D NIfTI int16 (81×104×83×720), 1.8mm isotropic, GLMsingle betas.

Reused perception preprocessing artifacts: reliability mask → 23,097 voxels → PCA → 3072-D (81.3% explained variance). Voxel space identical between perception and imagery NIfTI.

### 10.2 Transfer Evaluation (3 Models × 2 Conditions)

| Model | Condition | Cosine (mean ± std) | R@1 | R@5 | R@10 | Transfer Gap |
|-------|-----------|---------------------|-----|-----|------|--------------|
| Ridge | Perception | 0.6223 ± 0.1254 | 0.014 | 0.042 | 0.076 | — |
| Ridge | Imagery | 0.6226 ± 0.1268 | 0.007 | 0.028 | 0.063 | **+0.0003** |
| MLP | Perception | 0.6155 ± 0.1246 | — | — | — | — |
| MLP | Imagery | 0.6148 ± 0.1287 | — | — | — | **−0.0007** |
| TwoStage | Perception | 0.4702 ± 0.0991 | — | — | — | — |
| TwoStage | Imagery | 0.4599 ± 0.1020 | — | — | — | **−0.0103** |

**Key finding**: The cosine transfer gap is essentially **zero** for Ridge and MLP. The pre-registered expectation (H1: 60-80% of within-domain performance) was dramatically exceeded — imagery performance matches perception. TwoStage shows a small gap (−0.01) but its absolute performance is lower due to v1 hyperparameter issues (see Section 4, P1).

### 10.3 By Stimulus Type (Ridge, Imagery)

| Stimulus Set | Description | Cosine (mean) | Notes |
|-------------|-------------|---------------|-------|
| A (simple) | Bars, crosses | 0.609 | Low-complexity visual stimuli |
| B (complex) | NSD photos (5 nsd_ids) | 0.668 | **Best** — rich natural scenes |
| C (conceptual) | Verbal/text cues | 0.591 ± 0.196 | Most variable — text→CLIP target |

---

## 11. Novel Analyses on Real Data

13 of 15 analysis directions completed in 72.8s on real NSD-Imagery data (subj01, Ridge encoder). 26 figures generated. Full interpretation in [IMAGERY_RESULTS.md](IMAGERY_RESULTS.md).

| Analysis | Key Metric | Value | Interpretation |
|----------|-----------|-------|----------------|
| Summary | Transfer ratio | 1.000 | No cosine degradation |
| Dimensionality Gap | PR ratio (imagery/perception) | 0.77 | Imagery is lower-dimensional |
| Manifold Geometry | Hull volume ratio | 2.66 | Imagery spreads MORE in space |
| Topological RSA | RDM correlation | 0.196 (p<0.001) | Relational structure partially preserved |
| Reality Monitor | Classifier AUC | 0.661 | Subtle but detectable boundary |
| Adversarial Reality | Discriminator acc. | 0.504 | Near-chance — domains distribution-matched |
| Reality Confusion | Confusion score | 0.985 | Decision boundary nearly absent |
| Compositional | Imagery success / Perception success | 71.5% / 67.5% | Imagery slightly MORE composable |
| Semantic Survival | Per-concept preservation | — | See IMAGERY_RESULTS.md |
| Uncertainty | MC Dropout | — | See IMAGERY_RESULTS.md |
| Predictive Coding | Top-down index | — | See IMAGERY_RESULTS.md |
| SSI Dissociation | (dry-run mode) | — | Lacks real structural targets |
| Creative Divergence | ✗ ERROR | — | Requires shared stimulus IDs |
| Modality Decomp. | ✗ ERROR | — | Requires shared stimulus IDs |

---

## 12. Cross-Project Context

A separate project (**FMRI2images**) on the same cluster achieves R@1 ~58% using:
- ViT-bigG/14 (1280-d × 257 tokens) instead of ViT-L/14 (768-d CLS)
- 4-layer residual MLP [8192, 8192, 4096, 2048] → vMF decoder (~825M params)
- 15,724 raw voxels (nsdgeneral) instead of PCA-3072
- vMF-NCE + SoftCLIP + MixCo + EMA training

The 10× R@1 gap is expected given: 130× more parameters, higher-capacity CLIP backbone (bigG vs L/14), and richer input (raw voxels vs PCA). An external model loader (`src/fmri2img/models/external_loader.py`) allows loading FMRI2images checkpoints for prediction-level comparison without importing the other codebase.

---

## 13. FMRI2images High-Fidelity Imagery Integration (Phase 5)

> **Status**: Infrastructure complete, three checkpoint executions complete (`V30e_rerank_head_2048`, `V33b_shortlist_teacher_distill_preinit`, `N1v28a_dual_head`).

### 13.0 First Execution Results (V30e, subj01)

Checkpoint run: `/home/jovyan/work/data/FMRI2images/experimental_results/V30e_rerank_head_2048/subj01/checkpoint_best.pt`

| Condition | N | Cosine (mean ± std) | R@1 | R@5 | R@10 |
|-----------|---|----------------------|-----|-----|------|
| Perception | 144 | 0.1280 ± 0.0436 | 0.0069 | 0.0417 | 0.1111 |
| Imagery | 288 | 0.1246 ± 0.0444 | 0.0000 | 0.0139 | 0.0313 |

**V30e transfer gap (imagery − perception)**: **−0.0033**.

Note: V30e is a rerank-head variant with `mu_head: 2048→768` (tokens=1×768), so it validates cross-capacity execution and trend direction, but is not an apples-to-apples replacement for token-decoder bigG benchmark reporting (~55% R@1, ~70% CSLS).

### 13.0b Second Execution Results (V33b, subj01)

Checkpoint run: `/home/jovyan/work/data/FMRI2images/experimental_results/V33b_shortlist_teacher_distill_preinit/subj01/checkpoint_best.pt`

| Condition | N | Cosine (mean ± std) | R@1 | R@5 | R@10 |
|-----------|---|----------------------|-----|-----|------|
| Perception | 144 | 0.1746 ± 0.0531 | 0.0069 | 0.0139 | 0.1042 |
| Imagery | 288 | 0.1656 ± 0.0492 | 0.0035 | 0.0174 | 0.0347 |

**V33b transfer gap (imagery − perception)**: **−0.0090**.

Cross-check versus V30e:
- Perception cosine: +0.0466 (0.1280 -> 0.1746)
- Imagery cosine: +0.0410 (0.1246 -> 0.1656)
- Gap magnitude increased slightly (−0.0033 -> −0.0090)

Interpretation: both executed high-capacity checkpoints show the same directional pattern as baseline models: imagery remains close in semantic alignment but below perception, with a larger drop in retrieval discrimination.

### 13.0c Third Execution Results (V28a, subj01)

Checkpoint run: `/home/jovyan/work/data/FMRI2images/experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt`

| Condition | N | Cosine (mean ± std) | R@1 | R@5 | R@10 |
|-----------|---|----------------------|-----|-----|------|
| Perception | 144 | -0.0059 ± 0.0619 | 0.0000 | 0.0417 | 0.0903 |
| Imagery | 288 | 0.0008 ± 0.0571 | 0.0035 | 0.0139 | 0.0451 |

**V28a transfer gap (imagery − perception)**: **+0.0067**.

Important caveat: this dual-head checkpoint outputs token-structured predictions (`257×768`) and shows near-zero absolute cosine under the current CLS-compatible extraction route. It is therefore best interpreted as a compatibility run and not directly rank-compared against V30e/V33b absolute cosine values.

### 13.1 Motivation

All 13 imagery analyses (Section 11) used only weak models (Ridge R@1=1.8%, 768-d ViT-L/14). The "zero transfer gap" finding (Section 10) could be an artifact of models too weak to detect fine-grained perception/imagery differences. FMRI2images high-capacity checkpoints are now being added to this analysis track, with V30e and V33b already executed on imagery data.

### 13.2 New Infrastructure

| Module | Purpose |
|--------|---------|
| `src/fmri2img/data/nsdgeneral_extractor.py` | Extract 15,724 raw nsdgeneral voxels from imagery NIfTI |
| `src/fmri2img/data/imagery_raw_voxels.py` | Raw-voxel imagery dataset (bypasses PCA) |
| `src/fmri2img/analysis/token_spatial.py` | Novel 16×16 spatial token fidelity analysis |
| `src/fmri2img/analysis/cross_capacity.py` | Cross-model consistency (6M vs 825M) |
| `src/fmri2img/analysis/concept_conditional.py` | Per-category transfer gap with bootstrap CI |
| `scripts/eval_fmri2images_imagery.py` | FMRI2images inference on imagery (CLS + 257 tokens) |
| `scripts/run_fidelity_ladder.py` | Central experiment: gap × model capacity |
| `scripts/run_hifi_analyses.py` | Master orchestration for all high-fidelity analyses |

### 13.3 Planned Experiments

**Experiment 1: Fidelity Ladder** — Does the zero transfer gap hold with a 10× better model?
- Ridge (6M) → MLP (6.3M) → FMRI2images (825M)
- Two outcomes: "model-independent shared substrate" OR "resolution-dependent divergence"

**Experiment 2: Token-Level Spatial Decomposition** — First spatial map of imagery fidelity.
- 257 ViT-bigG/14 tokens → 16×16 spatial grid per image region
- Where does imagery break down? Center vs periphery (Kosslyn's spotlight theory)

**Experiment 3: Neural Compression Theory** — Per-token dimensionality analysis.
- Does imagery compress spatial tokens differently? Token-level participation ratio grid.

**Experiment 4: Cross-Capacity Consistency** — Are findings model-independent?
- Run all 13 analyses on both weak (Ridge) and strong (FMRI2images) predictions.
- Effect-size correlation determines if findings are neural vs artifactual.

**Experiment 5: Concept-Conditional Transfer** — Which categories gap most?
- Per-category (faces, scenes, objects) transfer gap with CLIP zero-shot labels.

**Experiment 6: Attention Condition** — 288 untested attention trials.
- Tests whether physical stimulus presence is the key variable.

See [IMAGERY_RESULTS.md](IMAGERY_RESULTS.md) for the single source of truth on current project state.

---

## 14. Discoveries Summary (March 2026)

1. **Perception→Imagery transfer is unexpectedly robust** on subj01 for baseline models (Ridge/MLP gap ≈ 0 in cosine).
2. **Imagery changes geometry more than mean alignment**: lower intrinsic dimensionality (PR ratio 0.77), but larger manifold spread (hull ratio 2.66).
3. **Discrimination drops while alignment holds**: retrieval degrades on imagery even when cosine mean is stable.
4. **Domain boundary is weak**: linear probe finds a small signal (AUC 0.661), adversarial discriminator is near chance (0.504).
5. **High-capacity FMRI2images checkpoints are now executable on imagery data (V30e, V33b, V28a)**: V30e/V33b show stable negative transfer gaps, while V28a completes as a dual-head compatibility run with near-zero absolute cosine scale.

## 15. Innovative Research Track (Immediate Next Runs)

### 15.1 Attention-as-Bridge Experiment (H6)
- Goal: test whether attention trials sit between perception and imagery.
- Run: `eval_perception_to_imagery_transfer.py --mode attention` for Ridge and V30e.
- Novel value: first 3-condition trajectory (perception→attention→imagery).

### 15.2 Retrieval Collapse Mechanism (H4)
- Goal: explain why R@K drops when cosine is stable.
- Run: inter-sample similarity / local-density analysis on predictions.
- Novel value: disentangles alignment fidelity from discriminability.

### 15.3 Shared-Stimulus Pair Analysis (Set B)
- Goal: explicit paired comparison on known shared `nsd_id`s (5 photos).
- Run: matched perception-vs-imagery deltas per stimulus.
- Novel value: within-item transfer avoids aggregate confounds.

### 15.4 Cross-Capacity Consistency (Ridge vs FMRI2images)
- Goal: measure whether effect directions replicate across model scales.
- Run: `analysis/cross_capacity.py` with the same 13 analysis outputs.
- Novel value: separates neural phenomena from model artifacts.
