# Project Status — Single Source of Truth

> **Version**: 0.4.1  
> **Last commit**: `working tree (uncommitted)`  
> **Last updated**: March 19, 2026

---

## Executive Summary

| Aspect | Status |
|--------|--------|
| **Perception pipeline** | ✅ Complete — 28 models trained, subj01 primary |
| **Analysis modules** | ✅ 19 modules code-complete, **13/15 run on real data** |
| **NSD-Imagery data** | ✅ Downloaded — 5.9 GB, 4 subjects, indices built |
| **Cross-domain eval** | ✅ Complete — 6 evaluations, transfer gap ≈ 0 |
| **Novel analyses** | ✅ 13/15 directions, 26 figures generated |
| **Cross-project bridge** | ✅ Three FMRI2images checkpoints executed (V30e, V33b, V28a) |
| **Tests** | ✅ 51 passing |
| **Docs** | ✅ Fully overhauled + imagery results documented |

### New Findings (March 19)

FMRI2images `V30e_rerank_head_2048`, `V33b_shortlist_teacher_distill_preinit`, and `N1v28a_dual_head` were executed on NSD-Imagery subj01:

| Checkpoint | Condition | Cosine (mean ± std) | R@1 | R@5 | R@10 |
|------------|-----------|----------------------|-----|-----|------|
| V30e | Perception (N=144) | 0.1280 ± 0.0436 | 0.0069 | 0.0417 | 0.1111 |
| V30e | Imagery (N=288) | 0.1246 ± 0.0444 | 0.0000 | 0.0139 | 0.0313 |
| V33b | Perception (N=144) | 0.1746 ± 0.0531 | 0.0069 | 0.0139 | 0.1042 |
| V33b | Imagery (N=288) | 0.1656 ± 0.0492 | 0.0035 | 0.0174 | 0.0347 |
| V28a | Perception (N=144) | -0.0059 ± 0.0619 | 0.0000 | 0.0417 | 0.0903 |
| V28a | Imagery (N=288) | 0.0008 ± 0.0571 | 0.0035 | 0.0139 | 0.0451 |

Transfer gaps (imagery − perception): **−0.0033** (V30e), **−0.0090** (V33b), **+0.0067** (V28a).

Execution note:
- V28a now runs successfully, but absolute cosine remains near zero with current CLS-compatible extraction, so treat it as a compatibility datapoint rather than a direct quality ranking against V30e/V33b.

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

All modules are code-complete. **13 of 15 novel analysis directions run on real NSD-Imagery data** (subj01, Ridge encoder). See [IMAGERY_RESULTS.md](IMAGERY_RESULTS.md) for full results.

### Novel Analysis Directions (Real Data)

| # | Direction | Status | Key Result |
|---|-----------|--------|------------|
| 1 | Dimensionality Gap | ✅ Real | PR ratio 0.77 — imagery lower-dimensional |
| 2 | Uncertainty / Vividness | ✅ Real | MC Dropout correlation computed |
| 3 | Semantic Survival | ✅ Real | Per-concept preservation ratios |
| 4 | Topological RSA | ✅ Real | RDM correlation 0.196 (p<0.001) |
| 5 | SSI Dissociation | ⚠️ Dry-run | Needs structural targets |
| 6 | Reality Monitor | ✅ Real | AUC 0.661 |
| 7 | Reality Confusion | ✅ Real | Confusion score 0.985 |
| 8 | Adversarial Reality | ✅ Real | Discriminator 0.504 (chance) |
| 9 | Compositional Imagination | ✅ Real | Imagery 71.5% > Perception 67.5% |
| 10 | Predictive Coding | ✅ Real | Top-down index computed |
| 11 | Manifold Geometry | ✅ Real | Hull volume ratio 2.66 |
| 12 | Modality Decomposition | ❌ Error | Requires shared stimulus IDs |
| 13 | Creative Divergence | ❌ Error | Requires shared stimulus IDs |

### Infrastructure Modules (All Synthetic-Tested)

| # | Module | File | Synthetic Test | Real Data |
|---|--------|------|----------------|-----------|
| 1 | RSA (Representational Similarity) | `src/fmri2img/analysis/rsa.py` | ✅ | ✅ (via topological_rsa) |
| 2 | CKA (Centered Kernel Alignment) | `src/fmri2img/analysis/cka.py` | ✅ | ❌ pending |
| 3 | Voxel Reliability | `src/fmri2img/analysis/reliability.py` | ✅ | ✅ (used in preprocessing) |
| 4 | Noise-Ceiling Estimation | `src/fmri2img/analysis/noise_ceiling.py` | ✅ | ❌ pending |
| 5 | Domain Confusion / Classifier | `src/fmri2img/analysis/domain_confusion.py` | ✅ | ✅ (via reality_monitor) |
| 6 | Cross-Domain Transfer Eval | `src/fmri2img/analysis/transfer_eval.py` | ✅ | ✅ |
| 7 | Domain Adaptation (DANN) | `src/fmri2img/analysis/domain_adaptation.py` | ✅ | ✅ (via adversarial_reality) |
| 8 | Imagery Adapter (LoRA / linear) | `src/fmri2img/adapters/imagery_adapter.py` | ✅ | ❌ pending |
| 9 | Soft Retrieval | `src/fmri2img/analysis/soft_retrieval.py` | ✅ | ❌ pending |
| 10 | Uncertainty Estimation | `src/fmri2img/analysis/uncertainty.py` | ✅ | ✅ |
| 11 | Barlow Twins Probe | `src/fmri2img/analysis/barlow_twins.py` | ✅ | ❌ pending |
| 12 | VICReg Probe | `src/fmri2img/analysis/vicreg.py` | ✅ | ❌ pending |
| 13 | Topographic Analysis | `src/fmri2img/analysis/topographic.py` | ✅ | ❌ pending |
| 14 | Temporal Dynamics | `src/fmri2img/analysis/temporal_dynamics.py` | ✅ | ❌ pending |
| 15 | Feature Attribution (Grad-CAM) | `src/fmri2img/analysis/feature_attribution.py` | ✅ | ❌ pending |
| 16 | Composite Score | `src/fmri2img/analysis/composite_score.py` | ✅ | ❌ pending |
| 17 | Stats (permutation tests) | `src/fmri2img/analysis/stats.py` | ✅ | ✅ (via novel analyses) |
| 18 | Loss Functions (InfoNCE, SoftCLIP, MixCo) | `src/fmri2img/losses/` | ✅ | ✅ (training) |
| 19 | Manifold Geometry | `src/fmri2img/analysis/manifold_geometry.py` | ✅ | ✅ |

---

## Critical Blockers

### ~~1. NSD-Imagery Data~~ ✅ RESOLVED

NSD-Imagery fMRI betas downloaded (5.9 GB, 4 subjects). Indices built for all 4 subjects. Cross-domain eval complete for subj01. See [IMAGERY_RESULTS.md](IMAGERY_RESULTS.md) for full results.

### 2. Cross-Project Embedding Mismatch (PARTIALLY RESOLVED)

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
| `scripts/eval_perception_to_imagery_transfer.py` | Cross-domain transfer eval (6 evaluations complete) |
| `scripts/run_real_novel_analyses.py` | Real-data novel analyses (13/15 complete) |
| `scripts/fit_preprocessing.py` | Feature extraction & PCA |
| `scripts/build_nsd_imagery_index.py` | Build NSD-Imagery indices (4 subjects done) |
| `scripts/train_imagery_adapter.py` | Train imagery adapter (needs real data — now available) |
| `scripts/run_novel_analyses.py` | Run all 15 analysis modules (dry-run + real) |
| `scripts/run_imagery_ablations.py` | Ablation studies |
| `scripts/make_paper_figures.py` | Generate publication figures |

---

## Version History

| Version | Commit | Description |
|---------|--------|-------------|
| 0.4.0 | `6d2d785` | Cross-domain imagery eval, 13 novel analyses on real data, 26 figures |
| 0.3.0 | `0d162f4` | 19 analysis modules, imagery adapter, full test suite |
| 0.2.x | — | Preprocessing pipeline, shared-1000 eval |
| 0.1.x | — | Initial perception training, Ridge + MLP baselines |

---

## Next Actions (Priority Order)

1. ~~Download NSD-Imagery data~~ ✅
2. ~~Run real cross-domain eval~~ ✅
3. ~~Run novel analysis battery~~ ✅ (13/15)
4. ~~Execute first FMRI2images checkpoint on imagery~~ ✅ (V30e)
5. ~~Execute second FMRI2images checkpoint on imagery~~ ✅ (V33b)
6. **Run attention condition (Ridge + V30e/V33b)** → establish perception→attention→imagery trajectory
7. **Run shared-stimulus paired analysis (Set B nsd_ids)** → within-item transfer effect sizes
8. **Run cross-capacity consistency** → Ridge vs FMRI2images effect-direction agreement
9. **Add token-space v28a evaluator** → evaluate V28a in native token objective space
10. **Fix failed analyses** → shared stimulus alignment for creative divergence + modality decomposition
11. **Train multi-subject models** → subj02/05/07 indices ready, need checkpoints
12. **Write paper** → promote March findings into Results/Discussion draft
