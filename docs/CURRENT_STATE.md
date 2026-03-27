# Current State

`fmri2img` is now centered on a canonical research-grade MVP for:

- shared-vs-private latent disentanglement across perception and imagery
- 768-D ViT-L/14 content decoding
- optional vividness/confidence prediction when real labels exist
- ROI-grouped branch modeling
- reproducible train/eval/transfer/export workflows

## Canonical path

Use the workflow modules under `fmri2img.workflows`:

- `python -m fmri2img.workflows.prepare_perception_index --config ...`
- `python -m fmri2img.workflows.prepare_imagery_index --config ...`
- `python -m fmri2img.workflows.prepare_targets --config ...`
- `python -m fmri2img.workflows.prepare_mixed_index --config ...`
- `python -m fmri2img.workflows.prepare_roi_features --config ...`
- `python -m fmri2img.workflows.preflight_data --config ...`
- `python -m fmri2img.workflows.train_decoder --config ...`
- `python -m fmri2img.workflows.eval_decoder --config ...`
- `python -m fmri2img.workflows.eval_transfer --config ...`
- `python -m fmri2img.workflows.run_analysis --config ...`
- `python -m fmri2img.workflows.export_for_animus --config ...`

For a known-good repository-local smoke run, use `configs/canonical/shared_private_smoke.yaml`.
For real research runs, use `configs/canonical/shared_private_bootstrap.yaml` and the canonical prep sequence before training.

## Canonical modeling decisions

- Content target: `vit_l14_image_768`
- Pairing key: shared `nsdId`
- ROI streams: `early_visual`, `ventral_visual`, `metacognitive`
- Vividness/confidence: optional supervised head only
- Generation/reconstruction: supported downstream, not the primary repo identity

## Important caveat

The canonical workflow now has an official ROI materialization path, but it still requires:

- a reachable ROI mask source
- volumetric fMRI inputs or already materialized ROI features
- a prepared 768-D target cache

If those artifacts are missing, `preflight_data` should be treated as the canonical source of truth for whether the run is blocked, smoke-only, bootstrap-ready, or paper-ready.

## Legacy boundary

The following remain in the repo for comparison or historical reproduction, but are no longer the official entrypoint surface:

- multi-script feature training in `scripts/`
- older perception-only or reconstruction-first evaluation flows
- 512-D CLIP assumptions
- ad hoc analysis drivers that do not use the canonical data/model/export contract

Treat those surfaces as legacy unless a current doc explicitly calls them out.
