# Reproducibility

## Environment

- Python `>=3.10`
- Install with `pip install -e ".[all]"`
- The canonical workflow expects a 768-D target cache and mixed perception/imagery index or separate condition indices
- The checked-in smoke fixture is runnable without external data:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
```

## Known-good workflow order

The official bootstrap sequence is now:

1. Normalize or rebuild the perception index if needed:

```bash
python -m fmri2img.workflows.prepare_perception_index \
  --config configs/canonical/shared_private_bootstrap.yaml
```

2. Build the imagery index from raw imagery data:

```bash
python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/shared_private_bootstrap.yaml
```

3. Prepare the canonical 768-D ViT-L/14 target cache:

```bash
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/shared_private_bootstrap.yaml
```

4. Build the mixed perception/imagery index:

```bash
python -m fmri2img.workflows.prepare_mixed_index \
  --config configs/canonical/shared_private_bootstrap.yaml
```

5. Materialize canonical ROI features from real ROI masks:

```bash
python -m fmri2img.workflows.prepare_roi_features \
  --config configs/canonical/shared_private_bootstrap.yaml
```

6. Preflight the run and classify whether it is blocked, smoke-only, bootstrap-ready, or paper-ready:

```bash
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/shared_private_bootstrap.yaml
```

7. Train:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml
```

8. Evaluate:

```bash
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

9. Export:

```bash
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

## Bootstrap artifact contract

The canonical bootstrap config expects or produces:

- `dataset.perception_index`
- `dataset.imagery_index`
- `dataset.mixed_output_index`
- `targets.cache_path`
- `roi.mask_root` or `roi.mask_patterns`
- per-row `roi_features_json`, `roi_values_json`, and `roi_names_json` after ROI preparation

`train_decoder` now automatically consumes `dataset.mixed_output_index` when it exists, so the ROI-enriched mixed index becomes the canonical train/eval/export input.

## Artifact contract

Canonical outputs include:

- config snapshot
- ROI summary
- target summary
- checkpoint
- metrics bundle
- export manifest for Animus

The canonical export manifest is the supported handoff surface for downstream systems.

## Scientific honesty rules

- Do not train vividness/confidence heads without real labels.
- Do not claim stimulus-vs-percept dissociation is solved by the current dataset.
- Do not silently swap between 512-D and 768-D target spaces.
- If ROI masks are absent, use explicit fallback only for smoke tests, not for bootstrap or benchmark claims.
- Do not treat the canonical ROI branch summaries as full ROI ablation evidence; that remains future work.
