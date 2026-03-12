# Imagery Adapter Quick Start Guide

> **TL;DR**: Train a lightweight adapter (~1M params) on imagery data to improve perception→imagery transfer without retraining the base model.

⚠️ **Prerequisite**: NSD-Imagery fMRI data must be downloaded first. See [NSD_IMAGERY_DATASET_GUIDE.md](../technical/NSD_IMAGERY_DATASET_GUIDE.md).

---

## What Problem Does This Solve?

Models trained on **visual perception** fMRI don't perform as well on **mental imagery** fMRI (Hypothesis H1). Full retraining is expensive and risks degrading perception performance.

**Solution**: Train a lightweight adapter between the frozen perception encoder and output layer. The adapter learns to bridge the perception-imagery gap.

---

## Quick Start (3 Commands)

```bash
# 1. Train adapter (< 1 hour on H100)
python scripts/train_imagery_adapter.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --model-type two_stage \
  --adapter mlp \
  --output-dir outputs/adapters/subj01/mlp \
  --epochs 50

# 2. Evaluate with adapter
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --adapter-checkpoint outputs/adapters/subj01/mlp/checkpoints/adapter_best.pt \
  --mode imagery \
  --output-dir outputs/eval/mlp_adapter

# 3. View results
cat outputs/eval/mlp_adapter/README.md
```

---

## Expected Performance

> **Note**: These are projections based on cross-domain transfer literature. Actual numbers will be updated once NSD-Imagery data is acquired and experiments are run.

| Metric | Baseline (No Adapter) | With MLP Adapter | Expected Improvement |
|--------|----------------------|------------------|--------------------|
| CLIP Cosine | TBD | TBD | +15–30% relative |
| Retrieval R@1 | TBD | TBD | +40–70% relative |
| Retrieval R@5 | TBD | TBD | +20–40% relative |

**Training Efficiency**:

| Aspect | Full Retraining | Adapter |
|--------|----------------|---------|
| Time | 6-12 hours | **<1 hour** |
| GPU Memory | 12-24 GB | **4-8 GB** |
| Trainable Params | 100% | **<1% (~1M)** |
| Perception Performance | May degrade | **Preserved** |

---

## Full Ablation Suite

```bash
python scripts/run_imagery_ablations.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/two_stage/subj01/best.pt \
  --model-type two_stage \
  --output-dir outputs/imagery_ablations/subj01 \
  --epochs 50 \
  --with-condition
```

This automatically:
1. Evaluates baseline (no adapter)
2. Trains + evaluates linear adapter
3. Trains + evaluates MLP adapter
4. Trains + evaluates MLP + condition token
5. Generates comparison tables and figures

**Output**:
```
outputs/imagery_ablations/subj01/
├── adapters/
│   ├── linear/checkpoints/adapter_best.pt
│   ├── mlp/checkpoints/adapter_best.pt
│   └── mlp_condition/checkpoints/adapter_best.pt
├── eval/
│   ├── baseline/metrics.json
│   ├── linear_adapter/metrics.json
│   ├── mlp_adapter/metrics.json
│   └── mlp_adapter_condition/metrics.json
├── figures/
│   ├── bar_overall_metric.png
│   └── table.tex
├── results_table.csv
└── results_table.md
```

---

## Architecture

```
Input fMRI (voxels)
    ↓
┌───────────────────┐
│  Base Model       │  ← Frozen (no gradient updates)
│  (Ridge/MLP/      │
│   TwoStage)       │
└───────┬───────────┘
        │
        ▼  768-d CLIP embedding
┌───────────────────┐
│  Adapter          │  ← Trainable (~1M params)
│  (Linear or MLP)  │
│  + optional       │
│  condition token  │
└───────┬───────────┘
        │
        ▼  768-d adapted embedding
   Loss (cosine + InfoNCE)
```

### Adapter Types

| Type | Parameters | Description |
|------|-----------|-------------|
| **LinearAdapter** | ~590K | `W·x + b` — simple affine transform |
| **MLPAdapter** | ~1.2M | Two-layer MLP with residual connection |
| **ConditionEmbedding** | +768 | Learnable `[percept]` / `[imagery]` tokens added to input |

All adapters preserve embedding dimensionality (768-d CLIP ViT-L/14 space) and are initialized to identity for smooth training.

---

## Advanced: LoRA Adapters

For even more parameter-efficient adaptation:

```python
from fmri2img.models.lora_adapter import LoRAAdapter, MultiRankLoRA

# Single-rank LoRA
lora = LoRAAdapter(in_features=768, out_features=768, rank=4)

# Multi-rank with automatic selection
multi_lora = MultiRankLoRA(in_features=768, out_features=768, ranks=[2, 4, 8, 16])
```

---

## CLI Reference

```bash
python scripts/train_imagery_adapter.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--index` | required | Path to imagery Parquet index |
| `--checkpoint` | required | Path to perception model checkpoint |
| `--model-type` | `two_stage` | Model type: ridge, mlp, two_stage |
| `--adapter` | `mlp` | Adapter type: linear, mlp |
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 32 | Batch size |
| `--device` | `cuda` | Device |
| `--output-dir` | required | Output directory |

---

## References

- [IMAGERY_EXTENSION.md](../architecture/IMAGERY_EXTENSION.md) — Full architecture
- [PERCEPTION_VS_IMAGERY_ROADMAP.md](../research/PERCEPTION_VS_IMAGERY_ROADMAP.md) — Research plan
- [NSD_IMAGERY_DATASET_GUIDE.md](../technical/NSD_IMAGERY_DATASET_GUIDE.md) — Data access

---

**Status**: Code complete, awaiting NSD-Imagery data for real experiments  
**Last Updated**: March 12, 2026
