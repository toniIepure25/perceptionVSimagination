# Migration Guide

This guide explains what is now canonical, what remains legacy, and which old
capabilities still matter.

## Canonical now

- `fmri2img.workflows.prepare_perception_index`
- `fmri2img.workflows.prepare_imagery_index`
- `fmri2img.workflows.prepare_targets`
- `fmri2img.workflows.prepare_mixed_index`
- `fmri2img.workflows.prepare_roi_features`
- `fmri2img.workflows.preflight_data`
- `fmri2img.workflows.train_decoder`
- `fmri2img.workflows.eval_decoder`
- `fmri2img.workflows.eval_transfer`
- `fmri2img.workflows.run_analysis`
- `fmri2img.workflows.export_for_animus`

These workflows define the official shared/private decoder surface, including the
real-data preparation steps needed for the first bootstrap paper experiment.

## Legacy paths that still matter

### `fmri2img.data.roi.ROIPooler`

Why it still matters:
- it remains the historical implementation the new canonical ROI materializer was derived from

Migration status:
- the canonical ROI story now lives in `fmri2img.roi.materialize`
- keep `fmri2img.data.roi.ROIPooler` only as a legacy compatibility path

### `scripts/build_nsd_imagery_index.py`

Why it still matters:
- it is still useful as a historical wrapper around imagery index building

Migration status:
- the official path is now `fmri2img.workflows.prepare_imagery_index`
- new work should not depend on the script directly

### `scripts/train_imagery_adapter.py`

Why it still matters:
- it preserves the older adapter-based transfer baseline

Why it is legacy:
- it is a comparison path, not the canonical shared/private decoder

## Practical migration advice

If you are starting new work:

1. Run the canonical prep sequence:
   - `prepare_perception_index`
   - `prepare_imagery_index`
   - `prepare_targets`
   - `prepare_mixed_index`
   - `prepare_roi_features`
   - `preflight_data`
2. Train and evaluate through the canonical `fmri2img.workflows.*` surface.

If you need functionality from old code:

- use the legacy script explicitly
- document that it is a comparison/baseline path
- do not describe it as canonical until it has been migrated
