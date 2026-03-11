# Experiment Context (Living Document)

**Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI**

> This is a **living document** — updated during experiments with current state, issues, decisions, and plans.  
> Last updated: 2026-03-11

---

## 1. Infrastructure

| Component | Value |
|-----------|-------|
| **GPU** | NVIDIA H100 80GB HBM3 |
| **CPU/RAM** | 32 cores, 100 GiB |
| **CUDA** | 12.8 |
| **PyTorch** | 2.10.0+cu128 |
| **Python** | 3.13.12 |
| **SSH** | `jovyan@10.130.123.131` (pw: `orchestraiq`) |
| **Persistent storage** | `/home/jovyan/local-data/` (hostPath, 878G total, ~207G free) |
| **Venv** | `/home/jovyan/local-data/venv` (`--system-site-packages`) |
| **Code deploy** | `git pull` + `pip install -e .` on cluster |

> **Note**: H100 pod restarts reset `/home/jovyan/work/` but `/home/jovyan/local-data/` persists. Host key changes on restart — clear with `ssh-keygen -f ~/.ssh/known_hosts -R "10.130.123.131"`.

### Venv Activation

```bash
export PATH=/home/jovyan/local-data/venv/bin:$PATH
export VIRTUAL_ENV=/home/jovyan/local-data/venv
```

---

## 2. Data Artifacts

| Artifact | Location (cluster) | Shape | Notes |
|----------|-------------------|-------|-------|
| NSD betas | `/home/jovyan/work/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/` | 40 sessions × 750 | GLMdenoise + RR |
| Baseline features (X) | `outputs/features/baseline/subj01/X.npy` | 30000 × 3072 | Hard-threshold + PCA |
| Novel features (X) | `outputs/features/novel/subj01/X.npy` | 30000 × 3072 | Soft-weighted + PCA |
| CLIP cache | `outputs/clip_cache/clip.parquet` | 73K × 768 | ViT-L/14 embeddings |
| NSD IDs | `outputs/features/*/subj01/nsd_ids.npy` | 30000 | Maps features → stimuli |
| PCA model (baseline) | `outputs/features/baseline/subj01/pca_model.pkl` | — | Fitted PCA(3072) |
| PCA model (novel) | `outputs/features/novel/subj01/pca_model.pkl` | — | Fitted PCA(3072) |

---

## 3. Training Status

### Checkpoints (28 total)

| Config | Architecture | Features | Status | Key Result |
|--------|-------------|----------|--------|------------|
| ridge_baseline | Ridge | baseline | ✅ Done | cosine=0.7913, R@1=0.0183 |
| ridge_novel | Ridge | novel | ✅ Fixed | cosine=0.7913, R@1=0.0177 |
| mlp_baseline | MLP | baseline | ✅ Done | cosine=0.8014, R@1=0.0270 |
| mlp_novel | MLP | novel | ❌ v1 collapsed | cosine=0.3868 |
| mlp_novel_cosine | MLP | novel | ❌ v1 collapsed | cosine=0.5103 |
| mlp_novel_cosine_mse | MLP | novel | ❌ v1 collapsed | cosine=0.5171 |
| mlp_novel_light_infonce | MLP | novel | ❌ v1 collapsed | cosine=0.4541 |
| mlp_novel_strong_infonce | MLP | novel | ❌ v1 collapsed | cosine=0.3877 |
| mlp_novel_cosine_v2 | MLP | novel | ✅ Done | cosine=0.8026, R@1=0.0277 |
| mlp_novel_cosine_mse_v2 | MLP | novel | ✅ Done | cosine=0.8021, R@1=0.0257 |
| mlp_novel_light_infonce_v2 | MLP | novel | ✅ Done | cosine=0.7723, R@1=0.0267 |
| mlp_novel_strong_infonce_v2 | MLP | novel | ✅ Best R@1 | **R@1=0.0573**, med_rank=37 |
| two_stage_baseline | TwoStage | baseline | ⚠️ v1 weak | cosine=0.5515, R@1=0.0010 |
| two_stage_novel | TwoStage | novel | ❌ v1 collapsed | cosine=0.6399 |
| ts_novel_cosine | TwoStage | novel | ❌ v1 collapsed | cosine=0.7469 |
| ts_novel_cosine_mse | TwoStage | novel | ❌ v1 collapsed | cosine=0.7469 |
| ts_novel_light_infonce | TwoStage | novel | ❌ v1 collapsed | cosine=0.7466 |
| ts_novel_strong_infonce | TwoStage | novel | ❌ v1 collapsed | cosine=0.7286 |
| ts_novel_cosine_v2 | TwoStage | novel | ✅ Done | cosine=0.8124, R@1=0.0350 |
| ts_novel_cosine_mse_v2 | TwoStage | novel | ✅ Best cosine | **cosine=0.8129** |
| ts_novel_light_infonce_v2 | TwoStage | novel | ✅ Done | cosine=0.7754, R@1=0.0090 |
| ts_novel_strong_infonce_v2 | TwoStage | novel | ⚠️ Partial collapse | cosine=0.6832, R@1=0.0010 |
| multilayer_baseline | Multilayer | baseline | ✅ Done | cosine=0.8076, R@1=0.0320 |
| multilayer_baseline_v2 | Multilayer | baseline | ✅ Done | cosine=0.8084, R@1=0.0327 |
| multilayer_novel | Multilayer | novel | ❌ LW+InfoNCE collapse | cosine=0.7440 |
| multilayer_novel_v2 | Multilayer | novel | ❌ LW+InfoNCE collapse | cosine=0.7442 |
| multilayer_novel_v3 | Multilayer | novel | ✅ Fixed (cosine-only) | cosine=0.8115, R@1=0.0283 |
| multilayer_novel_v3_lw | Multilayer | novel | ✅ Fixed (LW, no InfoNCE) | cosine=0.8098, R@1=0.0317 |

**Legend**: ✅ Working | ⚠️ Needs attention | ❌ Known failure (v1 or design issue)

---

## 4. Known Issues

### P1: TwoStage Baseline Weak (Medium Priority)

- **Symptom**: `two_stage_baseline` cosine=0.5515, far below MLP baseline (0.8014)
- **Cause**: v1 hyperparameters too conservative (lr=5e-5, temp=0.05, batch=48) for 7.9M parameter model
- **Fix**: Retrain with v2 hyperparameters (lr=1e-3, cosine scheduler, larger batch)
- **Status**: Not started

### P2: Strong InfoNCE + TwoStage Collapse (Medium Priority)

- **Symptom**: `ts_novel_strong_infonce_v2` cosine=0.6832, only R@1=0.0010
- **Cause**: 7.9M params + infonce_weight=0.7 + temp=0.05 → entropy collapse. Sharp InfoNCE gradients can overwhelm cosine loss when model is deep enough.
- **Fix**: Reduce infonce_weight to 0.4-0.5, raise temp to 0.07, or use gradient scaling
- **Status**: Not started

### P3: Ridge Novel Eval Broken (Resolved)

- **Symptom**: `ridge_novel` cosine=-0.0036 (essentially random)
- **Cause**: `eval_all_models.py` centered ALL features, but Ridge trained on uncentered features with `fit_intercept=True`
- **Fix**: Ridge models now use raw (uncentered) features during eval
- **Result**: cosine=0.7913, R@1=0.0177 — matches ridge_baseline
- **Status**: ✅ Resolved

### P4: Multilayer LW + InfoNCE Collapse (Resolved)

- **Symptom**: `multilayer_novel` and `multilayer_novel_v2` have near-zero retrieval
- **Cause**: Learnable weights + InfoNCE creates conflicting gradient paths on weight logits
- **Fix**: Use cosine-only loss for multilayer. v3 models work correctly.
- **Status**: ✅ Resolved

### P5: MRR Shows 0.0000 in Eval Table (Resolved)

- **Symptom**: MRR column always showed 0.0000 in comparison table
- **Cause**: Key case mismatch — `compute_ranking_metrics()` returns lowercase `'mrr'`, but `eval_all_models.py` reads `'MRR'` (uppercase)
- **Fix**: Changed `r.get('MRR', 0)` → `r.get('mrr', 0)` at 4 locations
- **Result**: MRR now shows correctly (top model: 0.1235)
- **Status**: ✅ Resolved

### P6: Shared-1000 Benchmark Not Run (High Priority)

- **Symptom**: Phase C deliverable incomplete
- **Cause**: Script exists (`eval_shared1000_benchmark.py`) but hadn't been executed; also had 3D array indexing bugs
- **Fix**: Rewrote `extract_shared1000_features()` to always use `reliability_mask.npy` for voxel selection (23097 voxels), with optional soft weights within mask. Fixed 3D→1D flattening for mask, scaler arrays.
- **Status**: ✅ Resolved (commits 72e18ec, 063c33a, 602f22d). 10 models evaluated.

### P7: CLIP Dimension Mismatch in Configs (Low Priority)

- **Symptom**: `base.yaml` says `model_name: "ViT-B/32"` and `embedding_dim: 512` but actual model used everywhere is ViT-L/14 (768-dim)
- **Cause**: Config was set up early in the project before switching to ViT-L/14
- **Fix**: Update `base.yaml` → `model_name: "ViT-L/14"`, `embedding_dim: 768`, `final: 768`
- **Status**: ✅ Resolved (commit b825618)

### P8: Imagery Adapter save_adapter NameError (High Priority)

- **Symptom**: `train_imagery_adapter.py` line 540 would crash when saving final model
- **Cause**: Calls `save_adapter()` but the imported function is `save_imagery_adapter()`
- **Fix**: Change `save_adapter(` → `save_imagery_adapter(` on line 540
- **Status**: ✅ Resolved (commit b825618)

---

## 5. Root Causes Discovered

### PCA Mean Bias from Soft-Reliability Weighting

The novel preprocessing (soft-reliability sigmoid weighting) shifts the effective voxel mean by ~0.024 per voxel. After PCA projection through 3072 components, this becomes a constant bias vector with norm ≈ 11.1. Every novel feature vector is dominated by this bias, making pairwise cosines ≈ 0.9989 (all vectors nearly identical direction). Any model trained on these features learns to output the same CLIP direction for all inputs.

**Solution**: Auto-centering in `train_from_features_v2.py` detects mean norm > 0.01 and subtracts dataset mean, restoring the centered distribution needed for discriminative learning.

### InfoNCE Gradient Conflict with Learnable Weights

In MultiLayerTwoStageEncoder, learnable layer weights (softmax logits) interact badly with InfoNCE loss. Each layer's InfoNCE contribution pushes the weight logits in a different direction: whichever layer happens to have the sharpest contrastive signal gets amplified, suppressing other layers. Within 10 epochs, layer_18 captures 52.5% weight while the final CLIP layer drops to 6.5%. With cosine-only loss, learnable weights remain more balanced and functional.

---

## 6. Next Steps

- [x] **P3**: Deploy ridge centering fix → re-evaluate ridge_novel ✅
- [x] **P5**: Deploy MRR key fix → re-run eval for correct MRR values ✅
- [x] **P8**: Deploy imagery save fix ✅
- [x] **P7**: Update config CLIP dimensions ✅
- [x] **P6**: Run Shared-1000 benchmark (10 models evaluated) ✅
- [ ] **P1**: Retrain TwoStage baseline with v2 hyperparameters
- [ ] **P2**: Retrain TwoStage strong_infonce with reduced weight
- [ ] Investigate Shared-1000 preprocessing mismatch (negative cosine for ridge)
- [ ] Begin imagery adapter experiments (Phase D)
- [ ] Paper figures: bar charts comparing baseline vs. novel across architectures

---

## 7. Session Log

### Session 1 — Initial Setup
- Project structure, configs, NSD data download
- Remote cluster SSH deployment

### Session 2 — Preprocessing & Training
- Python 3.13 fixes, NIfTI loader fixes
- Persistent venv at `/home/jovyan/local-data/venv`
- Feature extraction: baseline (30K × 3072) + novel (30K × 3072)
- Trained 6 initial models (Ridge, MLP, TwoStage × baseline/novel)
- CLIP cache: 73K × 768 (ViT-L/14)

### Session 3 — Ablation & Multi-Layer
- Phase A: Ablation grid (4 loss configs × 2 architectures = 8 novel models)
- Phase B: Multi-layer CLIP supervision (baseline + novel)
- Discovered novel collapse — all v1 novel models near-random

### Session 4 — Diagnosis & Fix
- Root cause: PCA mean bias from soft-reliability weights (norm=11.1)
- Implemented auto-centering in train_from_features_v2.py
- Retrained all novel models (v2) — performance restored
- Fixed multi-layer InfoNCE collapse (v3: cosine-only)
- Fixed model loading for v1 checkpoints (infer dims from state_dict)
- Fixed report overwriting (config_name in filename)
- Comprehensive evaluation: all 28 models evaluated successfully
- Best retrieval: mlp_novel_strong_infonce_v2 R@1=0.0573
- Best cosine: ts_novel_cosine_mse_v2 cosine=0.8129

### Session 5 — Documentation, Bug Fixes & Shared-1000
- Created EXPERIMENT_RESULTS.md and EXPERIMENT_CONTEXT.md
- Fixed P3 (ridge centering), P5 (MRR key), P7 (config dim), P8 (imagery save) — commit b825618
- Ridge_novel corrected: cosine 0.7913, R@1 0.0177 (was -0.0036 / 0.0003)
- Shared-1000 benchmark: 3 iterations fixing 3D array indexing in `extract_shared1000_features()`
  - Root cause: must use `reliability_mask.npy` for voxel selection (23097), not `weights > 0` (268K for novel)
- Shared-1000 results: Ridge R@1=0.70% best; deep models 0.10-0.20%; all much lower than test-split
- Statistical finding: Ridge significantly outperforms NNs on Shared-1000 (opposite of test-split ranking)
- Commits: b825618, 76a5294, 72e18ec, 063c33a, 602f22d
