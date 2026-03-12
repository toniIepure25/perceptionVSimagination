# Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)]()

> **Investigating how neural representations of visual perception and mental imagery diverge — and what that reveals about the architecture of imagination.**

This project compares fMRI-based neural decoding across two cognitive domains: *visual perception* (seeing an image) and *mental imagery* (imagining the same image). Using the Natural Scenes Dataset (NSD) and its imagery extension, we train models to reconstruct visual content from brain activity and measure how representations transfer — or fail to transfer — between perceiving and imagining.

---

## Current Status

> See [STATUS.md](docs/research/STATUS.md) for the full single-source-of-truth.

| What | State | Details |
|------|-------|---------|
| **Perception models** | ✅ Done | 28 configs trained on H100 — Ridge, MLP, TwoStage, Multilayer, Adapters |
| **Best perception R@1** | 5.7% | MLP `strong_infonce_v2` (test-split, gallery=3000) |
| **Analysis modules** | ✅ Code complete | 19 neuroscience directions + CKA, UMAP, ROI, interpretability |
| **Advanced losses** | ✅ Tested | VICReg, Barlow Twins, Triplet+InfoNCE, DANN, LoRA |
| **NSD-Imagery data** | ⚠️ Not downloaded | The single blocker for all cross-domain analyses |
| **Cross-project bridge** | 🔧 Planned | FMRI2images project (R@1~58%) for comparison |
| **Real analysis results** | ❌ None yet | All 19 modules awaiting imagery data |

---

## Research Overview

### Core Question

**Can neural decoders trained on visual perception generalize to mental imagery, and what does the transfer gap reveal about the neural architecture of imagination?**

### Hypotheses

| ID | Hypothesis | How We Test It |
|----|-----------|----------------|
| **H1** | Perception-trained decoders show degraded but non-zero imagery performance | CLIP cosine drops to 60-80% of within-domain |
| **H2** | Mixed perception+imagery training improves cross-domain robustness | <5% perception drop with >15% imagery gain |
| **H3** | Lightweight adapters bridge the gap efficiently | 80-90% of full fine-tuning at 10× less cost |

### 19 Novel Analysis Directions

Beyond standard transfer evaluation, the project implements neuroscience-driven analyses:

| # | Direction | Key Question |
|---|-----------|-------------|
| 1 | **Dimensionality Gap** | Does imagery compress the perceptual manifold? |
| 2 | **Uncertainty as Vividness** | Does MC Dropout uncertainty track imagery quality? |
| 3 | **Semantic Survival** | Which concepts survive the transition to imagination? |
| 4 | **Topological RSA** | Does imagination restructure representational geometry? |
| 5 | **Individual Fingerprints** | Is the imagery gap subject-specific and stable? |
| 6 | **Semantic-Structural Dissociation** | Do semantics survive while structure degrades? |
| 7 | **Reality Monitor** | Can PRM theory predict perception-imagery confusability? |
| 8 | **Reality Confusion** | Where is the boundary between perceived and imagined? |
| 9 | **Adversarial Reality** | Can a discriminator tell perception from imagery? |
| 10 | **Hierarchical Reality** | At which layer does the gap emerge? |
| 11 | **Compositional Imagination** | Can imagination compose novel concepts? |
| 12 | **Predictive Coding** | Does imagery follow top-down information flow? |
| 13 | **Manifold Geometry** | Is there a centrality bias in imagery? |
| 14 | **Modality Decomposition** | What's shared vs. unique between modalities? |
| 15 | **Creative Divergence** | What transformation rules govern imagination? |

Plus advanced research modules: **CKA** (representation comparison), **UMAP/t-SNE** (manifold visualization), **ROI Decoding** (per-brain-region accuracy), **Interpretability** (gradient attribution).

---

## Architecture

### Decoding Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  fMRI Data  │────▶│ Preprocessing│────▶│   Encoder   │────▶│    Target    │
│ (Perception │     │  (Z-score +  │     │  (Ridge/MLP/│     │  Embeddings  │
│ or Imagery) │     │  PCA k=3072) │     │  Two-Stage) │     │  (CLIP 768-d)│
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                     │
                              ┌───────────────────────────────────────┘
                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │     Cross-Domain Analysis Engine (19 dirs)       │
                    │  Dimensionality · Topology · Uncertainty · CKA   │
                    │  Semantics · ROI · Grad Attribution · UMAP/tSNE  │
                    └──────────────────────────────────────────────────┘
```

### Encoder Architectures (This Project)

| Architecture | Best Metric | Role |
|-------------|------------|------|
| **Ridge** | cosine 0.79 | Linear baseline |
| **MLP** | R@1 5.7% | Nonlinear baseline (best retrieval) |
| **TwoStage** | cosine 0.81 | Residual encoder (best cosine) |
| **Multilayer** | cosine 0.81 | Deep variant |
| **Adapters** | — | Lightweight perception→imagery bridge |

### Cross-Project: FMRI2images

A separate project at `/home/jovyan/work/FMRI2images/` achieves R@1~58% (CSLS~70%) using:
- vMF-NCE probabilistic posterior (825M params)
- ViT-bigG/14 token targets (1280-d × 257 tokens)
- 15,724 raw voxels (no PCA)
- MixCo augmentation + SoftCLIP + EMA

We plan to use its predictions alongside ours to validate that perception-vs-imagery findings are robust to model quality.

---

## Installation

```bash
git clone https://github.com/toniIepure25/perceptionVSimagination.git
cd perceptionVSimagination

# Option A: Conda
conda env create -f environment.yml
conda activate fmri2img

# Option B: pip
pip install -e ".[all]"

# Verify
python -c "import fmri2img; print(fmri2img.__version__)"
```

---

## Quick Start

### 1. Train Perception Encoder (on H100 cluster)

```bash
# Extract features + fit preprocessing
python scripts/fit_preprocessing.py
python scripts/extract_features.py

# Train (recommended: run_full_pipeline.py handles everything)
python scripts/run_full_pipeline.py

# Or train individual models
python scripts/train_from_features_v2.py --config configs/training/mlp.yaml
```

### 2. Evaluate Perception Models

```bash
# Evaluate all models on test split
python scripts/eval_all_models.py

# Shared-1000 benchmark
python scripts/eval_shared1000_full.py
```

### 3. Cross-Domain Transfer (requires NSD-Imagery data)

```bash
# Build imagery index
python scripts/build_nsd_imagery_index.py \
  --subject subj01 --data-root data/nsd_imagery \
  --cache-root cache/ --output cache/indices/imagery/subj01.parquet

# Evaluate transfer
python scripts/eval_perception_to_imagery_transfer.py \
  --checkpoint checkpoints/mlp/subj01/mlp.pt \
  --mode both --output-dir outputs/reports/imagery/
```

### 4. Novel Analyses

```bash
# Dry-run (validates code without real data)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml --dry-run

# Full run (requires imagery data)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml

# Publication figures
python scripts/make_novel_figures.py --results-dir outputs/novel_analyses/
```

---

## Evaluation Metrics

### Perception Models (actual results)

| Model | CLIP Cosine | R@1 | R@5 | Median Rank |
|-------|------------|-----|-----|-------------|
| Ridge | 0.79 | 1.8% | — | — |
| MLP (strong_infonce_v2) | 0.79 | **5.7%** | — | 37 |
| TwoStage | **0.81** | 3.2% | — | — |

### Cross-Domain Metrics (to be measured)

| Metric | What It Measures | Perception Target | Imagery Target |
|--------|-----------------|-------------------|----------------|
| CLIP Cosine | Semantic fidelity | >0.50 | >0.35 |
| R@1 | Exact identification | >5% | >2% |
| Transfer Ratio | Imagery / Perception | — | 0.60–0.80 |

---

## Repository Structure

```
perceptionVSimagination/
├── src/fmri2img/
│   ├── analysis/                   # 19 neuroscience analysis directions
│   │   ├── core.py                 # Shared utilities & embedding bundles
│   │   ├── dimensionality.py       # Dir 1: Dimensionality gap
│   │   ├── imagery_uncertainty.py  # Dir 2: Uncertainty as vividness
│   │   ├── semantic_decomposition.py # Dir 3: Semantic survival
│   │   ├── topological_rsa.py      # Dir 4: Topological signatures
│   │   ├── cross_subject.py        # Dir 5: Individual fingerprints
│   │   ├── semantic_structural_dissociation.py  # Dir 6
│   │   ├── reality_monitor.py      # Dir 7: PRM theory
│   │   ├── reality_confusion.py    # Dir 8: Confusion boundaries
│   │   ├── adversarial_reality.py  # Dir 9: Discriminator
│   │   ├── hierarchical_reality.py # Dir 10: Layer emergence
│   │   ├── compositional_imagination.py # Dir 11: Brain algebra
│   │   ├── predictive_coding.py    # Dir 12: Information flow
│   │   ├── manifold_geometry.py    # Dir 13: Centrality bias
│   │   ├── modality_decomposition.py # Dir 14: Shared vs unique
│   │   ├── creative_divergence.py  # Dir 15: Transformation rules
│   │   ├── cka.py                  # CKA: Representation similarity
│   │   ├── embedding_visualization.py # UMAP/t-SNE visualization
│   │   ├── roi_decoding.py         # Per-ROI decoding
│   │   └── interpretability.py     # Gradient attribution
│   ├── models/                     # Encoder architectures
│   │   ├── ridge.py                # Ridge regression
│   │   ├── mlp.py                  # MLP encoder
│   │   ├── encoders.py             # Two-stage residual encoder
│   │   ├── adapters.py             # Imagery adapters (Linear, MLP)
│   │   ├── lora_adapter.py         # LoRA (Low-Rank Adaptation)
│   │   ├── clip_adapter.py         # CLIP dimension adapter
│   │   ├── multi_target_decoder.py # CLIP/IP/SD prediction
│   │   └── encoding_model.py       # Image→fMRI encoding
│   ├── eval/                       # Evaluation & benchmarking
│   │   ├── retrieval.py            # R@K, MRR, median rank
│   │   ├── ceiling_normalized.py   # Noise-ceiling normalization
│   │   ├── sota_comparison.py      # Published baseline comparison
│   │   ├── uncertainty.py          # MC Dropout uncertainty
│   │   └── brain_alignment.py      # Neural alignment metrics
│   ├── training/                   # Training losses & loops
│   │   ├── losses.py               # InfoNCE, VICReg, Barlow, Triplet
│   │   ├── domain_adversarial.py   # DANN + gradient reversal
│   │   └── base.py                 # Training loop infrastructure
│   ├── stats/                      # Statistical inference
│   │   └── inference.py            # BH/Bonferroni FDR correction
│   ├── data/                       # Data loading
│   │   ├── nsd_imagery.py          # NSD-Imagery dataset
│   │   └── loaders.py              # DataLoaderFactory
│   └── utils/                      # Config, logging, checkpointing
│
├── scripts/
│   ├── run_full_pipeline.py        # End-to-end training pipeline
│   ├── train_from_features_v2.py   # Feature-based training (latest)
│   ├── eval_all_models.py          # Evaluate all perception models
│   ├── eval_shared1000_full.py     # Shared-1000 benchmark
│   ├── eval_perception_to_imagery_transfer.py  # Cross-domain eval
│   ├── build_nsd_imagery_index.py  # NSD-Imagery index builder
│   ├── run_novel_analyses.py       # 15-direction analysis orchestrator
│   ├── make_novel_figures.py       # Publication figure generator
│   ├── train_imagery_adapter.py    # Adapter training
│   ├── run_imagery_ablations.py    # Ablation suite
│   ├── fit_preprocessing.py        # Z-score + PCA preprocessing
│   └── extract_features.py         # Feature extraction from betas
│
├── configs/
│   ├── base.yaml                   # Base config (ViT-L/14, 768-d)
│   ├── experiments/
│   │   ├── novel_analyses.yaml     # 15-direction analysis config
│   │   ├── perception_to_imagery_eval.yaml
│   │   ├── ablation.yaml
│   │   └── reproducibility.yaml
│   └── training/                   # Per-model training configs
│
├── tests/                          # 51+ tests (all passing)
├── docs/                           # Research docs, guides, technical
├── checkpoints/                    # Trained model weights
├── outputs/                        # Evaluation results & figures
└── cache/                          # NSD data, CLIP embeddings
```

---

## Key References

- Allen et al. (2022). A massive 7T fMRI dataset. *Nature Neuroscience*.
- Ozcelik & VanRullen (2023). Brain-Diffuser. *Scientific Reports*.
- Takagi & Nishimoto (2023). High-resolution reconstruction from brain activity. *CVPR*.
- Scotti et al. (2023). MindEye: Reconstructing the mind's eye. *NeurIPS*.
- Pearson (2019). The human imagination. *Nature Reviews Neuroscience*.
- Dijkstra et al. (2019). Shared neural mechanisms of perception and imagery. *Trends in Cognitive Sciences*.

---

**Version**: 0.3.0 | **Last Updated**: March 12, 2026 | **License**: MIT
