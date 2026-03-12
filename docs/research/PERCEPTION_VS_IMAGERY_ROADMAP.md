# Perception vs. Imagery: Cross-Domain fMRI Decoding

**Research Roadmap — v0.3.0**

> Last Updated: March 12, 2026

---

## Abstract

This research extends existing NSD perception-based fMRI-to-image decoding to investigate cross-domain generalization to mental imagery. Using the NSD-Imagery dataset, we evaluate how models trained on perception transfer to imagery, quantify what information survives, and apply 19 novel analysis directions to characterize the representational differences between perceiving and imagining.

---

## Research Question

**Can neural decoders trained on visual perception fMRI generalize to decode mental imagery fMRI, and what does the transfer gap reveal about the neural architecture of imagination?**

---

## Hypotheses

### H1: Perception-to-Imagery Generalization Gap
Models trained on perception show degraded but non-zero performance on imagery, indicating partial shared representations.

**Operationalization**: CLIP cosine similarity and R@K on imagery test sets. Expected: 60-80% of within-domain performance.

### H2: Mixed Training Improves Robustness
Models trained on perception + imagery data achieve better cross-domain generalization at minor cost to perception performance.

**Operationalization**: Compare mixed vs perception-only models on both test sets. Expect <5% perception drop with >15% imagery gain.

### H3: Lightweight Adapter Benefit
A lightweight adapter trained on small amounts of imagery data bridges the gap without full retraining.

**Operationalization**: Adapter on 10-20% of imagery data, frozen encoder. Expect 80-90% of fine-tuning performance at 10× less cost.

---

## Experimental Matrix

| Training Config | Training Data | Test: Perception | Test: Imagery |
|----------------|--------------|-----------------|---------------|
| **Perception-Only** (baseline) | NSD perception | Within-domain | Cross-domain |
| **Mixed Training** | Perception + Imagery | Within-domain | Mixed-domain |
| **Perception + Adapter** | Perception (frozen) + Imagery (adapter) | Within-domain | Adapted |
| **No-Adaptation Baseline** | None (direct CLIP) | Upper bound | Upper bound |

**Model Variants**: Ridge, MLP, TwoStage, CLIP Adapter, FMRI2images vMF-NCE (external)

**Subjects**: subj01 (primary), subj02, subj05, subj07 (NSD-Imagery availability)

---

## Evaluation Metrics

### Primary
1. **CLIP Cosine Similarity** — Semantic fidelity (ViT-L/14, 768-d)
2. **Retrieval R@K** — K = {1, 5, 10, 50}
3. **Noise-Ceiling Normalized** — Metrics as % of theoretical maximum

### Secondary
4. **CKA** — Centered Kernel Alignment between perception/imagery representations
5. **Dimensionality** — PCA participation ratio
6. **Transfer Ratio** — Imagery performance / Perception performance

---

## Phases

### Phase 0: Perception Models ✅ COMPLETE

**28 models trained** on H100 cluster. All evaluated on test-split and Shared-1000.

| Architecture | Best Metrics | Configs |
|-------------|-------------|---------|
| Ridge | cosine 0.79, R@1 1.8% | 1 |
| MLP | cosine 0.79, R@1 5.7% | 11 |
| TwoStage | cosine 0.81, R@1 3.2% | 10 |
| Multilayer | cosine 0.81, R@1 3.2% | 4 |
| Adapters | — (CLIP/fMRI direction) | 3 |

Key finding: Ridge outperforms deep models on Shared-1000, likely due to preprocessing mismatch.

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for full results.

### Phase 1: Analysis Infrastructure ✅ COMPLETE

All analysis modules implemented and tested (51+ tests passing):

- **15 neuroscience analysis directions** (code complete, dry-run validated for 11-15)
- **CKA analysis** — linear/RBF kernels, debiased HSIC (11 tests)
- **Advanced losses** — VICReg, Barlow Twins, Triplet+InfoNCE, Hard Negatives (17 tests)
- **LoRA adapters** — Multi-rank, save/load (12 tests)
- **Domain adversarial** — Gradient reversal + DANN
- **Noise-ceiling normalization** — Ceiling-relative metrics
- **UMAP/t-SNE** — Manifold visualization
- **ROI decoding** — Per-brain-region analysis
- **Interpretability** — Integrated Gradients, SmoothGrad, Grad×Input
- **SoTA comparison** — 8 published baselines, LaTeX tables
- **FDR correction** — Benjamini-Hochberg, Bonferroni (11 tests)

### Phase 2: Adapter Architecture ✅ COMPLETE

Code implemented and tested (no real imagery data yet):

- [x] `LinearAdapter`, `MLPAdapter`, `ConditionEmbedding` in `src/fmri2img/models/adapters.py`
- [x] `LoRAAdapter`, `MultiRankLoRA` in `src/fmri2img/models/lora_adapter.py`
- [x] Training: `scripts/train_imagery_adapter.py`
- [x] Ablation: `scripts/run_imagery_ablations.py`
- [x] Figure generation: `scripts/make_paper_figures.py`

### Phase 3: NSD-Imagery Data Acquisition ⚠️ BLOCKED

**Status: NOT STARTED — This is the critical blocker.**

NSD-Imagery fMRI data has never been downloaded. Everything downstream requires it.

- [ ] Download NSD-Imagery betas from OpenNeuro ds004937 or NSD S3 bucket
- [ ] Build imagery index (`scripts/build_nsd_imagery_index.py`)
- [ ] Preprocess imagery betas (same z-score + PCA pipeline)
- [ ] Validate alignment with perception stimulus IDs

See [NSD_IMAGERY_DATASET_GUIDE.md](../technical/NSD_IMAGERY_DATASET_GUIDE.md).

### Phase 4: Cross-Domain Discovery 🔮 PLANNED

Once imagery data is available:

- [ ] Run cross-domain transfer eval (H1) — all 28 models
- [ ] Run 15 analysis directions on real data
- [ ] Run CKA / UMAP / ROI analysis
- [ ] Generate publication figures
- [ ] Noise-ceiling normalize all results
- [ ] Compare against SoTA baselines

### Phase 5: Cross-Project Integration 🔮 PLANNED

Leverage the FMRI2images project (R@1~58%) for cross-validation:

- [ ] Build external model loader for FMRI2images vMF-NCE checkpoint
- [ ] Run same analyses with FMRI2images predictions
- [ ] Compare patterns: same findings with different model quality → robust phenomenon
- [ ] Document where model quality affects findings

---

## Cross-Project Context

### FMRI2images (`/home/jovyan/work/FMRI2images/`)

| Aspect | Details |
|--------|---------|
| Architecture | 4-layer residual MLP → vMF decoder (825M params) |
| CLIP backbone | ViT-bigG/14, 1280-d × 257 tokens |
| Input | 15,724 raw voxels (nsdgeneral) |
| Best metrics | R@1 ~58%, CSLS R@1 ~70% |
| Training | vMF-NCE + SoftCLIP + MixCo + EMA, bf16, queue 1024 |
| Checkpoint | `experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt` |

The R@1 gap (5.7% vs 58%) is primarily due to:
1. **Target quality**: ViT-bigG/14 (1280-d×257) vs ViT-L/14 (768-d×1)
2. **Input representation**: Raw 15724 voxels vs PCA-reduced 3072 features
3. **Model capacity**: 825M vs 6.3M params
4. **Training recipe**: vMF-NCE + SoftCLIP + MixCo + EMA vs InfoNCE alone

---

## Reproducibility

- All configs version-controlled in `configs/`
- Seeds fixed: 42 for splits, per-config for training
- Hardware: H100 80GB, CUDA 12.8, PyTorch 2.10.0
- Results JSON in `outputs/` for programmatic access
- 51+ tests passing locally and on cluster

---

## References

1. Allen et al. (2022). "A massive 7T fMRI dataset." *Nature Neuroscience*.
2. Ozcelik & VanRullen (2023). "Brain-Diffuser." *Scientific Reports*.
3. Takagi & Nishimoto (2023). "High-resolution reconstruction." *CVPR*.
4. Scotti et al. (2023). "MindEye." *NeurIPS*.
5. Hu et al. (2021). "LoRA." *ICLR*.
6. Dijkstra et al. (2019). "Shared neural mechanisms." *Trends in Cognitive Sciences*.
