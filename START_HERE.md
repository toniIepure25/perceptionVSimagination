# Quick Start Guide

> **Onboarding for the Perception vs. Imagination neural decoding project.**

For a project overview, hypotheses, and architecture details, see the [main README](README.md).

---

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **GPU** | 6 GB VRAM | 24 GB VRAM (RTX 3090/4090) |
| **RAM** | 16 GB | 32 GB+ |
| **Storage** | 50 GB free | 200 GB+ |
| **OS** | Linux / macOS | Ubuntu 20.04+ |

### Environment Setup

```bash
git clone https://github.com/yourusername/perceptionVSimagination.git
cd perceptionVSimagination

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate fmri2img

# Option B: pip
pip install -e ".[all]"

# Verify
python -c "import fmri2img; print(fmri2img.__version__)"
```

---

## Choose Your Path

### Path 1: Perception Baseline (start here)

Train an encoder on perception fMRI, then evaluate within-domain.

```bash
# 1. Build NSD perception index
python scripts/build_full_index.py \
  --cache-root cache --subject subj01 \
  --output data/indices/nsd_index/

# 2. Build CLIP embedding cache (~2-3 hours, one-time)
python scripts/build_clip_cache.py \
  --cache-root cache --output outputs/clip_cache/clip.parquet \
  --batch-size 256

# 3. Train Two-Stage encoder (~6-8 hours)
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --subject subj01 \
  --output-dir checkpoints/two_stage/subj01

# Or use the Makefile shortcuts:
make ridge          # Fast baseline (~5 min)
make mlp            # MLP encoder (~2 hours)
```

### Path 2: Cross-Domain Transfer Evaluation

Test whether perception-trained models generalize to mental imagery.

```bash
# 1. Build NSD-Imagery index
python scripts/build_nsd_imagery_index.py \
  --subject subj01 --data-root data/nsd_imagery \
  --cache-root cache/ \
  --output cache/indices/imagery/subj01.parquet --verbose

# 2. Evaluate perception model on imagery data
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --mode imagery --split test \
  --output-dir outputs/reports/imagery/perception_transfer

# 3. Compare against within-domain baseline
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --mode perception --split test \
  --output-dir outputs/reports/imagery/perception_baseline

# Or use Makefile:
make imagery-index
make imagery-eval
```

### Path 3: Imagery Adapter Training

Train a lightweight adapter to bridge the perception-imagery gap (Hypothesis H3).

```bash
python scripts/train_imagery_adapter.py \
  --perception-checkpoint checkpoints/two_stage/subj01/best.pt \
  --imagery-index cache/indices/imagery/subj01.parquet \
  --adapter-type mlp \
  --output-dir checkpoints/adapters/subj01

# Or: make imagery-adapter
```

See [Adapter Quick Start](docs/guides/ADAPTER_QUICK_START.md) for the full adapter guide.

### Path 4: Novel Neuroscience Analyses

Run all six research directions that go beyond standard transfer evaluation.

```bash
# Dry-run (validates everything works without real data)
python scripts/run_novel_analyses.py \
  --config configs/experiments/novel_analyses.yaml --dry-run

# Full run with trained models and real data
python scripts/run_novel_analyses.py \
  --config configs/experiments/novel_analyses.yaml

# Generate publication-quality figures
python scripts/make_novel_figures.py --results-dir outputs/novel_analyses/

# Or: make novel-analyses && make novel-figures
```

---

## Novel Analysis Directions

| # | Direction | What It Reveals | Key Metric |
|---|-----------|----------------|-----------|
| 1 | **Dimensionality Gap** | Imagery compresses perceptual space into a lower-dimensional manifold | PCA participation ratio |
| 2 | **Uncertainty as Vividness** | MC Dropout variance correlates with imagery quality | Spearman rho (uncertainty vs. accuracy) |
| 3 | **Semantic Survival** | High-level concepts survive imagery; low-level features degrade | Per-concept preservation ratio |
| 4 | **Topological Signatures** | Persistent homology reveals structural reorganization in imagery | Wasserstein distance between persistence diagrams |
| 5 | **Individual Fingerprints** | The perception-imagery gap has a subject-specific, stable pattern | Second-order RSA across subjects |
| 6 | **Semantic-Structural Dissociation** | CLIP (semantics) transfers better than SD-latents (structure) | Semantic-Structural Index |

Each direction has a dedicated module in `src/fmri2img/analysis/` and is configured via `configs/experiments/novel_analyses.yaml`.

---

## Evaluation Outputs

After running cross-domain evaluation, outputs land in:

```
outputs/reports/imagery/perception_transfer/
  metrics.json        # CLIP cosine similarity, retrieval@K
  per_trial.csv       # Per-trial results with stimulus_type breakdown
  README.md           # Human-readable summary
```

After running novel analyses:

```
outputs/novel_analyses/
  dimensionality/     # PCA curves, participation ratios
  uncertainty/        # MC Dropout distributions
  semantic_survival/  # Per-concept preservation profiles
  topological_rsa/    # Persistence diagrams, RDMs
  cross_subject/      # Degradation profiles, weight similarity
  dissociation/       # SSI index, three-target comparison
```

---

## Configuration

All experiments are driven by YAML configs in `configs/`:

```bash
# View the main experiment config
cat configs/experiments/novel_analyses.yaml

# Override parameters at runtime
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --override "training.learning_rate=5e-5"
```

See `configs/experiments/reproducibility.yaml` for the full experimental protocol (seeds, splits, metrics).

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview, architecture, hypotheses |
| [Adapter Quick Start](docs/guides/ADAPTER_QUICK_START.md) | Imagery adapter training guide |
| [docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md](docs/research/PERCEPTION_VS_IMAGERY_ROADMAP.md) | Full research plan with hypotheses H1-H3 |
| [docs/research/PAPER_DRAFT_OUTLINE.md](docs/research/PAPER_DRAFT_OUTLINE.md) | Paper structure and narrative |
| [docs/architecture/IMAGERY_EXTENSION.md](docs/architecture/IMAGERY_EXTENSION.md) | System design for imagery pipeline |
| [docs/technical/NSD_IMAGERY_DATASET_GUIDE.md](docs/technical/NSD_IMAGERY_DATASET_GUIDE.md) | NSD-Imagery data format and access |

---

## Troubleshooting

**"CLIP embedding missing for nsdId=XXX"** -- CLIP cache not built. Run `make build-clip-cache`.

**"CUDA out of memory"** -- Reduce batch size: `--override "training.batch_size=64"`.

**Slow training on CPU** -- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Quick Commands

```bash
# Full Makefile reference
make help

# Core workflow
make ridge                # Train Ridge baseline (~5 min)
make mlp                  # Train MLP encoder (~2 hours)
make imagery-index        # Build NSD-Imagery index
make imagery-eval         # Evaluate perception -> imagery transfer
make imagery-adapter      # Train imagery adapter
make novel-analyses       # Run all 6 novel analyses
make novel-figures        # Generate publication figures
make test                 # Run test suite
```
