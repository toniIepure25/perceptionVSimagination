# Perception vs. Imagination: Cross-Domain Neural Decoding from fMRI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Investigating how neural representations of visual perception and mental imagery diverge — and what that reveals about the architecture of imagination.**

This project compares fMRI-based neural decoding across two cognitive domains: *visual perception* (seeing an image) and *mental imagery* (imagining the same image). Using the Natural Scenes Dataset (NSD) and its imagery extension, we train models to reconstruct visual content from brain activity and then measure how those representations transfer — or fail to transfer — between perceiving and imagining.

The core research question: **Can neural decoders trained on visual perception generalize to mental imagery, and what does the transfer gap reveal about the neural architecture of imagination?**

---

## Research Overview

### Motivation

Perception and imagination share substantial neural substrate in visual cortex, yet they are not identical processes. How information degrades, transforms, or reorganizes when transitioning from external perception to internal imagery is a fundamental question in cognitive neuroscience. By training computational models on perception data and evaluating them on imagery data, we can precisely quantify *what* information survives the transition to imagination and *what* is lost.

### Hypotheses

| ID | Hypothesis | Operationalization |
|----|-----------|-------------------|
| **H1** | Perception-trained decoders show degraded but non-zero performance on imagery | CLIP cosine similarity drops to 60-80% of within-domain performance |
| **H2** | Mixed perception+imagery training improves cross-domain robustness | <5% perception drop with >15% imagery gain |
| **H3** | Lightweight adapters can bridge the perception-imagery gap efficiently | 80-90% of full fine-tuning performance at 10x less training cost |

### Novel Analysis Directions

Beyond standard cross-domain evaluation, this project implements six neuroscience-driven analyses:

1. **The Dimensionality Gap** — Imagery representations occupy a lower-dimensional manifold than perception, suggesting a lossy compression during internalization.
2. **Uncertainty as Vividness** — MC Dropout uncertainty in imagery decoding correlates with subjective vividness of mental images.
3. **Semantic Survival** — High-level semantic content (object identity, category) is better preserved in imagery than low-level visual features (texture, spatial layout).
4. **Topological Signatures** — Persistent homology reveals structural reorganization of representational geometry during imagination.
5. **Individual Imagery Fingerprints** — The pattern of perception-to-imagery degradation is subject-specific and stable, constituting a cognitive fingerprint.
6. **Semantic-Structural Dissociation** — Multi-target decoding (CLIP, IP-Adapter tokens, SD VAE latents) reveals differential preservation: semantics survive imagery while structural details degrade.

---

## Architecture

### Decoding Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  fMRI Data  │────▶│ Preprocessing│────▶│   Encoder   │────▶│    Target    │
│ (Perception │     │ (Z-score +   │     │  (Ridge/MLP/│     │ Embeddings   │
│ or Imagery) │     │  PCA k=512)  │     │  Two-Stage) │     │ (CLIP/IP/SD) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                     │
                              ┌───────────────────────────────────────┘
                              ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────┐
│Reconstructed│◀────│   Stable     │◀────│  Cross-Domain Analysis Engine   │
│   Image     │     │  Diffusion   │     │  (Dimensionality, Uncertainty,  │
│  (512×512)  │     │  (v1.5/2.1)  │     │   Topology, Dissociation, ...)  │
└─────────────┘     └──────────────┘     └─────────────────────────────────┘
```

### Encoder Architectures

| Architecture | Parameters | Description | Role in Project |
|-------------|-----------|-------------|-----------------|
| **Ridge** | — | Linear regression | Baseline: measures linearly decodable information |
| **MLP** | ~148K | Multi-layer perceptron | Standard nonlinear baseline |
| **Two-Stage** | ~413K | Residual blocks + projection | Primary encoder for cross-domain evaluation |
| **Adapters** | ~10-50K | Linear/MLP heads on frozen encoder | Lightweight perception→imagery bridge (H3) |
| **MultiTarget** | ~500K | Simultaneous CLIP + IP + SD prediction | Dissociation analysis (Direction 6) |

### Cross-Domain Transfer Strategies

| Strategy | Training Data | Frozen? | Use Case |
|----------|--------------|---------|----------|
| **Direct Transfer** | Perception only | N/A | Zero-shot imagery evaluation |
| **Mixed Training** | Perception + Imagery | No | Joint domain training |
| **Adapter Fine-tuning** | Imagery only | Encoder frozen | Efficient domain bridging |
| **Full Fine-tuning** | Imagery only | No | Upper bound on adaptation |

---

## Installation

### Prerequisites

- **Python**: 3.10+
- **CUDA**: 11.7+ (for GPU acceleration)
- **RAM**: 32GB recommended
- **Storage**: ~200GB for NSD data + models

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/perceptionVSimagination.git
cd perceptionVSimagination

# Create conda environment
conda env create -f environment.yml
conda activate fmri2img

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import fmri2img; print('Ready')"
```

---

## Quick Start

### 1. Data Preparation

```bash
# Build NSD perception index
python scripts/build_full_index.py \
  --cache-root cache --subject subj01 \
  --output data/indices/nsd_index/

# Build CLIP embedding cache (~2-3 hours, one-time)
python scripts/build_clip_cache.py \
  --cache-root cache --output outputs/clip_cache/clip.parquet \
  --batch-size 256

# Build NSD-Imagery index
python scripts/build_nsd_imagery_index.py \
  --subject subj01 --data-root data/nsd_imagery \
  --cache-root cache/ --output cache/indices/imagery/subj01.parquet
```

### 2. Train Perception Encoder

```bash
# Two-Stage encoder (recommended, ~6-8 hours)
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --subject subj01 \
  --output-dir checkpoints/two_stage/subj01
```

### 3. Evaluate Cross-Domain Transfer

```bash
# Test perception-trained model on imagery data
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --mode imagery --split test \
  --output-dir outputs/reports/imagery/perception_transfer

# Compare against within-domain baseline
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --mode perception --split test \
  --output-dir outputs/reports/imagery/perception_baseline
```

### 4. Train Imagery Adapter (H3)

```bash
python scripts/train_imagery_adapter.py \
  --perception-checkpoint checkpoints/two_stage/subj01/best.pt \
  --imagery-index cache/indices/imagery/subj01.parquet \
  --adapter-type mlp \
  --output-dir checkpoints/adapters/subj01
```

### 5. Run Novel Analyses

```bash
# All six analysis directions (dry-run for validation)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml --dry-run

# With real data and trained models
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml

# Generate publication-quality figures
python scripts/make_novel_figures.py --results-dir outputs/novel_analyses/
```

---

## Evaluation Metrics

### Cross-Domain Metrics

| Metric | What It Measures | Perception Target | Imagery Target |
|--------|-----------------|-------------------|----------------|
| **CLIP Cosine** | Semantic fidelity of decoding | >0.50 | >0.35 |
| **Retrieval R@1** | Exact image identification | >20% | >10% |
| **Retrieval R@10** | Neighborhood accuracy | >60% | >40% |
| **Transfer Ratio** | Imagery / Perception performance | — | 0.60–0.80 |

### Novel Analysis Metrics

| Analysis | Key Metric |
|----------|-----------|
| Dimensionality Gap | PCA participation ratio (perception vs. imagery) |
| Uncertainty-Vividness | Spearman correlation between MC Dropout variance and vividness |
| Semantic Survival | Per-concept preservation ratio across cognitive domains |
| Topological RSA | Contraction ratio and neighborhood preservation |
| Cross-Subject | Second-order RSA of degradation profiles |
| Dissociation | Semantic-Structural Index (CLIP gap vs. SD-latent gap) |

---

## Repository Structure

```
perceptionVSimagination/
├── src/fmri2img/
│   ├── models/                 # Encoder architectures
│   │   ├── ridge.py            # Ridge regression baseline
│   │   ├── mlp.py              # MLP encoder
│   │   ├── encoders.py         # Two-stage residual encoder
│   │   ├── clip_adapter.py     # CLIP dimension adapter
│   │   ├── adapters.py         # Imagery adapters (Linear, MLP, MultiTarget)
│   │   ├── multi_target_decoder.py  # Simultaneous CLIP/IP/SD prediction
│   │   └── encoding_model.py   # Image→fMRI encoding model
│   ├── analysis/               # Novel neuroscience analyses
│   │   ├── core.py             # Shared utilities & embedding bundles
│   │   ├── dimensionality.py   # Direction 1: Dimensionality gap
│   │   ├── imagery_uncertainty.py   # Direction 2: Uncertainty as vividness
│   │   ├── semantic_decomposition.py # Direction 3: Semantic survival
│   │   ├── topological_rsa.py  # Direction 4: Topological signatures
│   │   ├── cross_subject.py    # Direction 5: Individual fingerprints
│   │   └── semantic_structural_dissociation.py  # Direction 6: Dissociation
│   ├── data/                   # Data loading & preprocessing
│   │   ├── nsd_imagery.py      # NSD-Imagery dataset & index builder
│   │   ├── loaders.py          # DataLoaderFactory, FMRIDataset
│   │   └── nsd_index.py        # NSD perception index
│   ├── eval/                   # Evaluation & uncertainty
│   ├── training/               # Training infrastructure
│   ├── generation/             # Diffusion-based reconstruction
│   ├── inference/              # Inference pipelines
│   └── utils/                  # Config, logging, checkpointing
│
├── scripts/
│   ├── build_nsd_imagery_index.py          # Build imagery data index
│   ├── eval_perception_to_imagery_transfer.py  # Cross-domain evaluation
│   ├── train_imagery_adapter.py            # Train perception→imagery adapters
│   ├── run_imagery_ablations.py            # Systematic ablation studies
│   ├── run_novel_analyses.py               # Orchestrate all 6 analyses
│   ├── make_novel_figures.py               # Publication-quality figures
│   ├── train_two_stage.py                  # Train perception encoder
│   ├── build_clip_cache.py                 # CLIP embedding extraction
│   └── build_full_index.py                 # NSD perception index
│
├── configs/
│   ├── experiments/
│   │   ├── novel_analyses.yaml             # Novel analysis configuration
│   │   └── perception_to_imagery_eval.yaml # Transfer evaluation config
│   ├── two_stage_sota.yaml                 # SOTA encoder config
│   └── base.yaml                           # Base configuration
│
├── tests/
│   ├── test_imagery_adapter.py             # Adapter unit tests
│   ├── test_imagery_integration.py         # Integration tests
│   └── test_imagery_scaffold.py            # Scaffold validation
│
├── docs/
│   ├── research/
│   │   └── PERCEPTION_VS_IMAGERY_ROADMAP.md
│   ├── architecture/
│   │   └── IMAGERY_EXTENSION.md
│   └── technical/
│       └── NSD_IMAGERY_DATASET_GUIDE.md
│
├── START_HERE.md               # Step-by-step onboarding guide
├── docs/guides/ADAPTER_QUICK_START.md  # Adapter training quick reference
├── pyproject.toml              # Package configuration
└── environment.yml             # Conda environment
```

---

## Experimental Design

### Cross-Domain Transfer Matrix

| Training Config | Training Data | Test: Perception | Test: Imagery |
|----------------|--------------|-----------------|---------------|
| Perception-Only (baseline) | NSD perception | Within-domain | Cross-domain |
| Mixed Training | Perception + Imagery | Within-domain | Mixed-domain |
| Perception + Adapter | Perception (frozen) + Imagery (adapter) | Within-domain | Adapted |
| Imagery-Only | NSD imagery | Cross-domain | Within-domain |

### Subjects

NSD-Imagery includes subjects: subj01, subj02, subj05, subj07. All analyses support per-subject and cross-subject comparisons.

### Ablation Studies

```bash
# Adapter architecture ablations
python scripts/run_imagery_ablations.py \
  --perception-checkpoint checkpoints/two_stage/subj01/best.pt \
  --imagery-index cache/indices/imagery/subj01.parquet \
  --output-dir outputs/ablations/subj01

# Covers: adapter type, hidden dimensions, learning rate, freeze depth, data fraction
```

---

## Key References

- Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25(1), 116–126.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*.
- Pearson, J. (2019). The human imagination: the cognitive neuroscience of visual mental imagery. *Nature Reviews Neuroscience*, 20(10), 624–634.
- Kosslyn, S. M. (2005). Mental images and the brain. *Cognitive Neuropsychology*, 22(3-4), 333–347.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Natural Scenes Dataset**: Emily Allen and team, University of Minnesota
- **NSD-Imagery Extension**: For providing the mental imagery fMRI data that makes cross-domain comparison possible
- **OpenAI CLIP**: Semantic embedding backbone
- **Stability AI**: Stable Diffusion for visual reconstruction
- **Hugging Face**: `diffusers` and `transformers` libraries
