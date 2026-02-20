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
│   ├── adapter_vitl14.yaml      # CLIP 512→768 adapter (~30 min)
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
│   ├── novel_analyses.yaml      # Six novel analysis directions
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
# Ridge baseline (~5 min, CPU)
python scripts/train_ridge.py --config configs/training/ridge_baseline.yaml --subject subj01

# MLP encoder (~2 hrs)
python scripts/train_mlp.py --config configs/training/mlp_standard.yaml --subject subj01

# Two-Stage SOTA (~4 hrs, best results)
python scripts/train_two_stage.py --config configs/training/two_stage_sota.yaml --subject subj01

# CLIP adapter (~30 min, after training encoder)
python scripts/train_clip_adapter.py --config configs/training/adapter_vitl14.yaml \
    --pretrained checkpoints/two_stage/subj01/two_stage_best.pt
```

### Inference

```bash
# Balanced quality/speed
python scripts/decode_diffusion.py --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt

# Publication figures
python scripts/decode_diffusion.py --config configs/inference/highres_quality.yaml \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt
```

### Research

```bash
# Cross-domain transfer evaluation
python scripts/eval_perception_to_imagery_transfer.py \
    --config configs/experiments/perception_to_imagery_eval.yaml

# Novel analyses (all 6 directions)
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml

# Ablation study
python -m fmri2img.eval.ablation_driver --base-config configs/experiments/ablation.yaml
```

---

## CLIP Model Notes

Two CLIP models are used in the pipeline:

| Model | Dim | Config | Purpose |
|-------|-----|--------|---------|
| ViT-B/32 | 512 | `base.yaml` clip section | fMRI training target |
| ViT-L/14 | 768 | `clip.yaml` | CLIP cache, diffusion conditioning |

The `adapter_vitl14.yaml` config trains a lightweight adapter bridging 512-D to 768-D.

---

## See Also

- [training/README.md](training/README.md) -- Training config details
- [inference/README.md](inference/README.md) -- Generation config details
- [experiments/README.md](experiments/README.md) -- Research experiment configs
- [system/README.md](system/README.md) -- NSD dataset and system configs
