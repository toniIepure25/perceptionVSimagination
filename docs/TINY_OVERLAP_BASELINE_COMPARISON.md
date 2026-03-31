# Tiny Overlap Baseline Comparison

## Purpose

This document freezes the first apples-to-apples comparison between:

- the canonical shared-private decoder
- a simple legacy-style Ridge baseline

Both runs use the same rebuilt canonical overlap dataset:

- config: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- subjects: `subj02`, `subj05`, `subj07`
- shared overlap: 4 paired `nsdId`s
- target space: 768-D ViT-L/14 image embeddings
- split counts: `train=38`, `val=19`, `test=19`

This is a bootstrap-scale engineering comparison, not a paper-scale benchmark.

## Compared Runs

### Canonical shared-private decoder

- workflow: `python -m fmri2img.workflows.train_decoder --config configs/canonical/multisubj_overlap_bootstrap.yaml`
- evaluation: `python -m fmri2img.workflows.eval_decoder ...`
- checkpoint: `outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt`
- metrics: `outputs/canonical/eval/multisubj_overlap_bootstrap/metrics.json`

### Simple legacy baseline

- workflow: `python -m fmri2img.workflows.run_legacy_ridge_baseline --config configs/canonical/multisubj_overlap_bootstrap.yaml`
- model class: `fmri2img.models.ridge.RidgeEncoder`
- feature source: flat `roi_values_json` vectors from the canonical prepared mixed index
- target space: same 768-D ViT-L/14 cache as the canonical model
- artifacts: `outputs/canonical/baselines/multisubj_overlap_ridge_legacy/`

## Fresh Results

### Canonical shared-private decoder

- content cosine: `0.00685`
- content cosine std: `0.01977`
- content MSE: `0.002586`
- imagery cosine mean: `0.00574`
- perception cosine mean: `0.01275`
- pair metric: `1` paired eval group, imagery-minus-perception gap `-0.00700`
- domain accuracy: `0.5263`

### Legacy Ridge baseline

- selected alpha: `100.0`
- validation cosine: `0.62765`
- validation MSE: `0.000970`
- test content cosine: `0.42202`
- test content cosine std: `0.06391`
- test content MSE: `0.001505`
- imagery cosine mean: `0.43820`
- perception cosine mean: `0.33576`
- pair metric: `1` paired eval group, imagery-minus-perception gap `0.10244`

## Interpretation

The simple Ridge baseline substantially outperformed the canonical shared-private decoder on this tiny overlap dataset.

That does **not** mean the shared-private architecture is wrong. It means:

- the canonical model is now operationally validated on a real prepared overlap dataset
- this specific dataset is far too small to justify a higher-capacity disentanglement model
- a low-dimensional linear decoder has a major bias-variance advantage in this regime

The shared-private model currently demonstrates infrastructure validity more than performance advantage:

- canonical prep, preflight, train, eval, transfer, analysis, and export all work end to end
- `z_shared`-based decoding does not show suspiciously inflated performance
- the domain head is not interpretable at this scale
- there is not enough held-out paired overlap to say anything strong about transfer

The Ridge baseline should therefore be treated as the current tiny-data reference, not as a reason to abandon the canonical architecture.

## Fairness Notes

- The comparison uses the same rebuilt canonical overlap mixed index.
- The comparison uses the same target cache and split logic.
- The Ridge baseline does **not** have a domain head or shared/private structure, so there is no direct domain-accuracy comparison.
- The Ridge baseline operates on flat ROI values, while the canonical model routes those ROI signals through grouped branches and latent factorization.

## Conclusion

On the current 4-pair overlap bootstrap:

- the canonical shared-private model is **architecturally validated**
- the simple Ridge baseline is **much stronger as a tiny-data predictive baseline**
- the dataset is **still too small to judge the core paper hypothesis**

The single best next experiment is to expand the overlap mixed-condition dataset while keeping this exact comparison fixed.
