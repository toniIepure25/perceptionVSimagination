# Expanded Overlap Comparison

## Summary

This document records the first fixed-comparison scaling audit after the canonical bootstrap baseline.

Question:

- As overlap mixed-condition data grows, does the canonical shared-private model begin to become competitive with the simple Ridge baseline?

Current answer:

- not yet, because the current mounted environment does **not** actually provide a larger fully canonical overlap dataset than the existing bootstrap baseline

This was a real live-environment audit and rerun, not a speculative note.

## What Was Audited

The live `orchestraiq-jupyter` pod currently exposes:

- perception indices for `subj01`, `subj02`, `subj05`, `subj07`
- rebuildable imagery beta volumes only for `subj02`, `subj05`, `subj07`
- one stale cached imagery index for `subj01`

The stale `subj01` imagery parquet still contains many rows without usable `nsdId`, so it fails the canonical mixed-condition requirement. It also has no rebuildable imagery beta package mounted in the pod, so it cannot currently be promoted into the canonical overlap path.

## Maximum Overlap Currently Achievable

The largest fully canonical overlap dataset currently rebuildable in the pod is still:

- subjects: `subj02`, `subj05`, `subj07`
- shared overlap size: `4` paired `nsdId`s
- overlap ids: `30857`, `53882`, `61178`, `65873`
- total rows: `76`
- splits: `train=38`, `val=19`, `test=19`
- readiness: `bootstrap_ready`

Per-subject overlap breakdown:

- `subj02`: `2` shared `nsdId`s, `38` mixed rows
- `subj05`: `1` shared `nsdId`, `19` mixed rows
- `subj07`: `1` shared `nsdId`, `19` mixed rows

So the attempted scale-up did **not** increase the overlap dataset beyond the existing bootstrap ceiling.

## Fixed Comparison That Stayed The Same

The comparison remained fixed:

- canonical shared-private decoder
- simple legacy Ridge baseline
- same 768-D ViT-L/14 target space
- same canonical split logic
- same evaluation metrics

No architecture, target-space, or evaluation changes were introduced.

## Canonical Max-Available Run

Config:

- `configs/canonical/max_available_overlap.yaml`

Live runtime execution used the same content as that checked-in config, with CPU-safe execution because the H100 was occupied by another process.

Fresh canonical results:

- test cosine: `0.05192`
- test cosine std: `0.01507`
- test MSE: `0.002469`
- imagery cosine mean: `0.05224`
- perception cosine mean: `0.05021`
- paired eval groups: `1`
- imagery-minus-perception gap: `0.00204`
- domain accuracy: `0.63158`

Training history:

- epoch 1: train loss `25.57`, validation cosine `0.02743`
- epoch 2: train loss `23.71`, validation cosine `0.03307`
- epoch 3: train loss `21.48`, validation cosine `0.04055`
- epoch 4: train loss `21.20`, validation cosine `0.05150`
- epoch 5: train loss `20.45`, validation cosine `0.06211`

This is a healthier canonical run than the weakest earlier rerun, but it is still evaluated on the same 4-pair overlap set.

## Ridge Baseline On The Same Dataset

Artifacts:

- `outputs/canonical/baselines/max_available_overlap_ridge_legacy/metrics.json`

Results:

- selected alpha: `100.0`
- test cosine: `0.42202`
- test MSE: `0.001505`
- paired eval groups: `1`
- imagery-minus-perception gap: `0.10244`

This baseline remains dramatically stronger on the current tiny overlap dataset.

## Interpretation

The canonical shared-private model did recover a modest positive content signal on this rerun, which is encouraging in a narrow engineering sense. But this should **not** be interpreted as evidence that the model is catching up because of scale.

Why not:

- the overlap dataset did not actually grow
- the subject set stayed effectively the same
- the unique paired stimulus count stayed at `4`

So the correct interpretation is:

- the canonical model is operational and can produce a weak positive signal on the fully canonical overlap ceiling
- the simple Ridge baseline still dominates strongly
- the current environment is still too small to answer the data-scale question

## What This Means Scientifically

The current result supports three careful conclusions:

1. The canonical shared-private platform is stable enough to run real mixed-condition comparisons repeatedly.
2. The fixed tiny-data baseline still favors a simple linear decoder by a large margin.
3. The next justified move is still further overlap expansion, not architecture changes.

## Most Important Limiter

The bottleneck is now clearly dataset scale, not platform readiness.

In particular:

- `subj01` is not currently canonically rebuildable for overlap in the pod
- the mounted imagery-capable subject set does not yield more than `4` usable paired `nsdId`s

So the next scale-up requires new overlapable data, not a different decoder head or loss.

## Recommended Next Experiment

The single best next experiment is:

- further data expansion with the same canonical model and the same Ridge comparison baseline

Only after a meaningfully larger overlap set exists should the project consider:

- targeted regularization or loss tuning
- model simplification
- improved ROI decomposition
