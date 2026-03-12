# Quick Start Guide

> **Onboarding for the Perception vs. Imagination neural decoding project (v0.3.0).**

For project overview, see [README.md](README.md). For full status, see [STATUS.md](docs/research/STATUS.md).

---

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.13+ (cluster) |
| **GPU** | 6 GB VRAM | H100 80 GB (cluster) |
| **RAM** | 16 GB | 100 GB (cluster) |
| **Storage** | 50 GB | 200 GB+ |

### Environment Setup

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

### H100 Cluster Access

```bash
# SSH into the cluster
sshpass -p orchestraiq ssh -o StrictHostKeyChecking=no jovyan@10.130.123.131

# Activate persistent venv
export PATH=/home/jovyan/local-data/venv/bin:$PATH
export VIRTUAL_ENV=/home/jovyan/local-data/venv

# Navigate to project
cd /home/jovyan/local-data/perceptionVSimagination
```

See [CLUSTER_ENVIRONMENT.md](docs/guides/CLUSTER_ENVIRONMENT.md) for full cluster details.

---

## Choose Your Path

### Path 1: Perception Baseline (already done)

All 28 perception models are trained. To reproduce or extend:

```bash
# View existing results
python scripts/eval_all_models.py
python scripts/eval_shared1000_full.py

# Train new model configs
python scripts/train_from_features_v2.py --config configs/training/mlp.yaml

# Or run the full pipeline
python scripts/run_full_pipeline.py
```

**Results**: Best R@1=5.7% (MLP), best cosine=0.81 (TwoStage). See [EXPERIMENT_RESULTS.md](docs/research/EXPERIMENT_RESULTS.md).

### Path 2: Cross-Domain Transfer (requires NSD-Imagery data)

⚠️ **BLOCKER**: NSD-Imagery fMRI data has not been downloaded. See [NSD_IMAGERY_DATASET_GUIDE.md](docs/technical/NSD_IMAGERY_DATASET_GUIDE.md).

```bash
# 1. Download NSD-Imagery data (on cluster)
# See docs/technical/NSD_IMAGERY_DATASET_GUIDE.md

# 2. Build imagery index
python scripts/build_nsd_imagery_index.py \
  --subject subj01 --data-root data/nsd_imagery \
  --cache-root cache/ --output cache/indices/imagery/subj01.parquet

# 3. Evaluate cross-domain transfer
python scripts/eval_perception_to_imagery_transfer.py \
  --checkpoint checkpoints/mlp/subj01/mlp.pt \
  --mode both --output-dir outputs/reports/imagery/
```

### Path 3: Imagery Adapter Training (requires imagery data)

```bash
python scripts/train_imagery_adapter.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --model-type two_stage --adapter mlp \
  --output-dir outputs/adapters/subj01/mlp --epochs 50

# Full ablation suite
python scripts/run_imagery_ablations.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --output-dir outputs/imagery_ablations/subj01
```

See [ADAPTER_QUICK_START.md](docs/guides/ADAPTER_QUICK_START.md) for details.

### Path 4: Novel Analyses (19 directions)

```bash
# Dry-run — validates all code without real data
python scripts/run_novel_analyses.py \
  --config configs/experiments/novel_analyses.yaml --dry-run

# Full run — requires imagery data for real results
python scripts/run_novel_analyses.py \
  --config configs/experiments/novel_analyses.yaml

# Generate publication figures
python scripts/make_novel_figures.py --results-dir outputs/novel_analyses/
```

### Path 5: Cross-Project Comparison

The [FMRI2images](docs/guides/CLUSTER_ENVIRONMENT.md) project at `/home/jovyan/work/FMRI2images/` has a stronger model (R@1~58%) that can serve as a comparison baseline. Running the same perception-vs-imagery analyses on both models' predictions tests whether findings are robust to model quality.

---

## Novel Analysis Directions

| # | Direction | Key Question | Module |
|---|-----------|-------------|--------|
| 1 | **Dimensionality Gap** | Does imagery compress perceptual space? | `analysis/dimensionality.py` |
| 2 | **Uncertainty as Vividness** | Does MC Dropout uncertainty track vividness? | `analysis/imagery_uncertainty.py` |
| 3 | **Semantic Survival** | Which concepts survive imagination? | `analysis/semantic_decomposition.py` |
| 4 | **Topological RSA** | Is representational topology restructured? | `analysis/topological_rsa.py` |
| 5 | **Individual Fingerprints** | Is the perception-imagery gap subject-specific? | `analysis/cross_subject.py` |
| 6 | **Semantic-Structural Dissociation** | Semantics survive, structure degrades? | `analysis/semantic_structural_dissociation.py` |
| 7 | **Reality Monitor** | PRM theory prediction | `analysis/reality_monitor.py` |
| 8 | **Reality Confusion** | Perception-imagery boundary | `analysis/reality_confusion.py` |
| 9 | **Adversarial Reality** | Discriminator detection | `analysis/adversarial_reality.py` |
| 10 | **Hierarchical Reality** | Layer-wise gap emergence | `analysis/hierarchical_reality.py` |
| 11 | **Compositional Imagination** | Novel concept composition | `analysis/compositional_imagination.py` |
| 12 | **Predictive Coding** | Top-down information flow | `analysis/predictive_coding.py` |
| 13 | **Manifold Geometry** | Centrality bias in imagery | `analysis/manifold_geometry.py` |
| 14 | **Modality Decomposition** | Shared vs unique components | `analysis/modality_decomposition.py` |
| 15 | **Creative Divergence** | Transformation rules of imagination | `analysis/creative_divergence.py` |

Additional: **CKA**, **UMAP/t-SNE**, **ROI Decoding**, **Interpretability** (Integrated Gradients, SmoothGrad).

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and architecture |
| [STATUS.md](docs/research/STATUS.md) | Single source of truth — what's done, what's blocked |
| [EXPERIMENT_RESULTS.md](docs/research/EXPERIMENT_RESULTS.md) | Actual perception model results |
| [EXPERIMENT_CONTEXT.md](docs/research/EXPERIMENT_CONTEXT.md) | Living experiment log |
| [PERCEPTION_VS_IMAGERY_ROADMAP.md](docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md) | Research plan (H1-H3) |
| [PAPER_DRAFT_OUTLINE.md](docs/research/PAPER_DRAFT_OUTLINE.md) | Paper narrative |
| [IMAGERY_EXTENSION.md](docs/architecture/IMAGERY_EXTENSION.md) | System architecture |
| [ADAPTER_QUICK_START.md](docs/guides/ADAPTER_QUICK_START.md) | Adapter training guide |
| [CLUSTER_ENVIRONMENT.md](docs/guides/CLUSTER_ENVIRONMENT.md) | H100 cluster setup |
| [NSD_IMAGERY_DATASET_GUIDE.md](docs/technical/NSD_IMAGERY_DATASET_GUIDE.md) | NSD-Imagery data access |

---

## Troubleshooting

**"Module not found"** — Run `pip install -e ".[all]"` to install in development mode.

**"CUDA out of memory"** — Reduce batch size in config YAML or use `--override "training.batch_size=64"`.

**SSH connection refused** — Cluster may have restarted. Clear host key: `ssh-keygen -f ~/.ssh/known_hosts -R "10.130.123.131"`.

**Tests fail locally** — Some tests need `torchvision`. Use: `pip install -e ".[dev]"`.

---

## Quick Commands

```bash
# Tests
pytest tests/ -v                  # All 51+ tests
pytest tests/test_cka.py -v       # CKA analysis tests
pytest tests/test_lora.py -v      # LoRA adapter tests

# Evaluation
python scripts/eval_all_models.py         # All perception models
python scripts/eval_shared1000_full.py    # Shared-1000 benchmark

# Analysis (dry-run)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml --dry-run
```
