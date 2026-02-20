# Training Configurations

Encoder training presets for fMRI-to-CLIP embedding prediction. All configs inherit from `../base.yaml`.

---

## Available Configs

| Config | Architecture | Time | GPU Memory | Cosine Sim |
|--------|-------------|------|------------|------------|
| `ridge_baseline.yaml` | Ridge + PCA | ~5 min | CPU only | 0.25--0.35 |
| `mlp_standard.yaml` | MLP (2048-D hidden) | ~2 hrs | 8 GB | 0.35--0.45 |
| `two_stage_sota.yaml` | Residual blocks + multi-layer | ~4 hrs | 12 GB | 0.45--0.55 |
| `adapter_vitl14.yaml` | Linear adapter (512→768) | ~30 min | 6 GB | N/A |
| `clip2fmri.yaml` | Inverse CLIP→fMRI | ~1 hr | 8 GB | N/A |
| `dev_fast.yaml` | MLP (1K samples, 10 epochs) | ~3 min | 4 GB | N/A |

*Benchmarks: NSD subj01, 30K trials, NVIDIA A100 40GB.*

---

## Recommended Workflow

1. **Ridge baseline** -- validates the data pipeline and establishes a lower bound.
2. **MLP standard** -- confirms neural encoder improvement over linear.
3. **Two-Stage SOTA** -- final results for the paper.
4. **CLIP adapter** -- bridges ViT-B/32 (512-D) to ViT-L/14 (768-D) for diffusion generation.

```bash
# Step 1
python scripts/train_ridge.py --config configs/training/ridge_baseline.yaml --subject subj01

# Step 2
python scripts/train_mlp.py --config configs/training/mlp_standard.yaml --subject subj01

# Step 3
python scripts/train_two_stage.py --config configs/training/two_stage_sota.yaml --subject subj01

# Step 4
python scripts/train_clip_adapter.py --config configs/training/adapter_vitl14.yaml \
    --pretrained checkpoints/two_stage/subj01/two_stage_best.pt
```

---

## Common Overrides

```bash
# Adjust learning rate
--override "training.learning_rate=5e-5"

# Larger model capacity
--override "encoder.stage1.latent_dim=1024"

# Longer training with patience
--override "training.epochs=300" --override "training.early_stop_patience=20"

# Reduce batch size (low-memory GPUs)
--override "training.batch_size=16"
```

---

## Notes

- **`dev_fast.yaml`** is for debugging only -- do not use for final results.
- **`clip2fmri.yaml`** trains the inverse mapping (CLIP→fMRI) used for brain-consistency loss in ablation studies.
- All configs assume the NSD index is built at `data/indices/nsd_index/`. Run `make build-index` first.
