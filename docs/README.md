# Documentation

**Perception vs. Imagination: Cross-Domain Neural Decoding**
*Research-Level Documentation*

---

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[Quick Start Guide](../START_HERE.md)** | Get started in <5 minutes | New users |
| **[Adapter Quick Start](../ADAPTER_QUICK_START.md)** | Imagery adapter training | Researchers |
| **[Project README](../README.md)** | Project overview & setup | Everyone |

---

## Documentation Categories

### 1. Research Documentation (`research/`)

Core research design and experimental protocols for the perception-vs-imagery investigation.

- **[Perception vs. Imagery Roadmap](research/PERCEPTION_VS_IMAGERY_ROADMAP.md)** — Comprehensive research plan: hypotheses (H1–H3), experimental matrix, evaluation metrics, and timeline.

### 2. Architecture Documentation (`architecture/`)

System design and component specifications.

- **[Imagery Extension Architecture](architecture/IMAGERY_EXTENSION.md)** — How the NSD-Imagery data pipeline, adapter framework, and cross-domain evaluation integrate with the base encoding system.
- Model architectures (Ridge, MLP, Two-Stage, Adapters, MultiTargetDecoder)
- Data pipeline design and preprocessing modules

### 3. Technical Documentation (`technical/`)

Implementation details, data formats, and troubleshooting.

- **[NSD-Imagery Dataset Guide](technical/NSD_IMAGERY_DATASET_GUIDE.md)** — Data structure, index format, train/val/test splits, and CLIP cache integration for the imagery domain.
- Configuration best practices
- Data validation procedures

### 4. User Guides (in project root)

Step-by-step onboarding and usage instructions.

- **[START_HERE.md](../START_HERE.md)** — Full walkthrough: installation, data preparation, training, evaluation, and the perception-vs-imagery track.
- **[ADAPTER_QUICK_START.md](../ADAPTER_QUICK_START.md)** — Quick reference for training imagery adapters on top of frozen perception-trained encoders.

---

## Documentation by Task

### I want to understand the research

1. Read [PERCEPTION_VS_IMAGERY_ROADMAP.md](research/PERCEPTION_VS_IMAGERY_ROADMAP.md) for hypotheses and experimental design
2. Review the [Project README](../README.md) for the six novel analysis directions
3. Check [IMAGERY_EXTENSION.md](architecture/IMAGERY_EXTENSION.md) for how the analysis modules are structured

### I want to train a model

1. Start with [START_HERE.md](../START_HERE.md) for environment setup and data preparation
2. Train a perception encoder (Ridge → MLP → Two-Stage)
3. Train an imagery adapter using [ADAPTER_QUICK_START.md](../ADAPTER_QUICK_START.md)

### I want to evaluate cross-domain transfer

1. Read the evaluation section in [START_HERE.md](../START_HERE.md)
2. Run `scripts/eval_perception_to_imagery_transfer.py` for transfer metrics
3. Run `scripts/run_novel_analyses.py` for the six neuroscience analyses

### I want to work with NSD-Imagery data

1. Read [NSD_IMAGERY_DATASET_GUIDE.md](technical/NSD_IMAGERY_DATASET_GUIDE.md) for data format and structure
2. Use `scripts/build_nsd_imagery_index.py` to create the data index
3. See `src/fmri2img/data/nsd_imagery.py` for the dataset API

### I want to run the novel analyses

1. Configure `configs/experiments/novel_analyses.yaml`
2. Run `scripts/run_novel_analyses.py` (supports `--dry-run` for validation)
3. Generate figures with `scripts/make_novel_figures.py`

---

## External References

- [Natural Scenes Dataset (NSD)](http://naturalscenesdataset.org/)
- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Key Papers

- Allen et al. (2022) — Natural Scenes Dataset
- Radford et al. (2021) — CLIP: Learning Transferable Visual Models
- Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion
- Pearson (2019) — The Human Imagination: Cognitive Neuroscience of Visual Mental Imagery

---

**Last Updated**: February 2026
**Status**: Active Development
