# Inference Configurations

Image generation presets using Stable Diffusion conditioned on decoded fMRI embeddings. All configs inherit from `../base.yaml`.

---

## Available Configs

| Config | Resolution | Steps | Time/image | Use Case |
|--------|-----------|-------|------------|----------|
| `production.yaml` | 512 px | 50 | ~10 s | Standard evaluation, demos |
| `fast_inference.yaml` | 512 px | 25 | ~5 s | Batch processing, previews |
| `highres_quality.yaml` | 1024 px | 200 | ~60 s | Publication figures |

*Benchmarks on NVIDIA A100 40GB.*

---

## Workflow

```bash
# 1. Preview with fast config
python scripts/decode_diffusion.py \
    --config configs/inference/fast_inference.yaml \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --samples 10

# 2. Main evaluation
python scripts/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --samples 500

# 3. Publication-quality for selected samples
python scripts/decode_diffusion.py \
    --config configs/inference/highres_quality.yaml \
    --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
    --samples 20
```

---

## Common Overrides

```bash
# More diffusion steps (higher quality)
--override "diffusion.num_inference_steps=100"

# Adjust guidance scale (higher = more faithful, lower = more creative)
--override "diffusion.guidance_scale=9.0"

# Reduce memory usage
--override "diffusion.enable_attention_slicing=true" \
--override "diffusion.enable_vae_slicing=true"

# Deterministic generation
--override "diffusion.eta=0.0"
```

---

## Notes

- **Adapter required**: Generation with ViT-L/14 conditioning requires a trained CLIP adapter. See `configs/training/adapter_vitl14.yaml`.
- **GPU memory**: 1024 px generation requires 16+ GB VRAM. Use attention slicing on smaller GPUs.
- **Seed control**: Set `training.seed` for reproducible generation across runs.
