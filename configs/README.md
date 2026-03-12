# Configuration System

Configuration files for the **Perception vs. Imagination** cross-domain neural decoding project. All configs use YAML with a hierarchical inheritance model rooted at `base.yaml`.

---

## Directory Structure

```
configs/
├── base.yaml                    # Shared defaults (all experiments inherit from this)
├── clip.yaml                    # CLIP model settings (ViT-L/14, 768-D)
├── data.yaml                    # Default data splits (used by training scripts)
│
├── training/                    # Encoder training presets
│   ├── ridge_baseline.yaml      # Ridge regression (~5 min, CPU)
│   ├── mlp_standard.yaml        # MLP encoder (~2 hrs, GPU)
│   ├── two_stage_sota.yaml      # Two-Stage SOTA (~4 hrs, GPU)
│   ├── adapter_vitl14.yaml      # [DEPRECATED] CLIP 512→768 adapter
│   ├── clip2fmri.yaml           # Inverse mapping (research)
│   └── dev_fast.yaml            # Fast debugging (1K samples, 10 epochs)
│
├── inference/                   # Image generation presets
│   ├── production.yaml          # Balanced (512px, 50 steps, ~10s)
│   ├── fast_inference.yaml      # Speed-optimized (25 steps, ~5s)
│   └── highres_quality.yaml     # Publication-quality (1024px, 200 steps)
│
├── experiments/                 # Research experiment configs
│   ├── ablation.yaml            # Ablation study template
│   ├── novel_analyses.yaml      # 19 novel analysis directions
│   ├── perception_to_imagery_eval.yaml  # Cross-domain evaluation
│   └── reproducibility.yaml     # Experiment protocol & seeds
│
├── data/
│   └── nsd_imagery.yaml         # NSD-Imagery dataset config
│
└── system/
    └── data.yaml                # NSD dataset paths, S3 access, preprocessing
```

---

## Inheritance

All training/inference configs inherit from `base.yaml`:

```yaml
# In any config file:
_base_: ../base.yaml      # Inherit all defaults
model:
  hidden_dim: 512          # Override specific parameters
```

Runtime overrides take highest priority:

```bash
python scripts/train_two_stage.py \
    --config configs/training/two_stage_sota.yaml \
    --override "training.learning_rate=5e-5" \
    --override "training.batch_size=64"
```

Priority: runtime overrides > specific config > base.yaml

---

## Quick Reference

### Training

```bash
# Full pipeline (Ridge + MLP + TwoStage, all subjects)
python scripts/run_full_pipeline.py

# Train from pre-extracted features
python scripts/train_from_features_v2.py --config mlp_baseline --subject subj01

# Imagery adapter (requires NSD-Imagery data)
python scripts/train_imagery_adapter.py --subject subj01
```

### Evaluation

```bash
# Shared-1000 benchmark
python scripts/eval_shared1000_full.py

# Cross-domain transfer (requires NSD-Imagery data)
python scripts/eval_perception_to_imagery_transfer.py
```

### Research

```bash
# Novel analyses (all 19 directions — RSA, CKA, domain confusion, etc.)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml

# Imagery ablation studies (requires NSD-Imagery data)
python scripts/run_imagery_ablations.py

# Paper figures
python scripts/make_paper_figures.py
```

---

## CLIP Model Notes

Two CLIP models are used in the pipeline:

| Model | Dim | Config | Purpose |
|-------|-----|--------|---------|
| ViT-L/14 | 768 | `base.yaml` clip section | fMRI training target & CLIP cache |
| ViT-L/14 | 768 | `clip.yaml` | CLIP cache, diffusion conditioning |

The `adapter_vitl14.yaml` config is **deprecated** — encoders now train directly against ViT-L/14.

---

## See Also

- [training/README.md](training/README.md) -- Training config details
- [inference/README.md](inference/README.md) -- Generation config details
- [experiments/README.md](experiments/README.md) -- Research experiment configs
- [system/README.md](system/README.md) -- NSD dataset and system configs
