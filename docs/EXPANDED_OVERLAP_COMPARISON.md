# Expanded Overlap Comparison

## Summary

This document records the first successful expansion of the canonical overlap dataset after acquiring the full public NSD-Imagery source.

Checked-in config for this scaling audit:

- `configs/canonical/max_available_overlap.yaml`

Question:

- As overlap mixed-condition data grows, does the canonical shared-private model begin to become competitive with the simple Ridge baseline?

Current answer:

- the overlap dataset did grow beyond the old 4-pair ceiling
- the canonical multi-subject batching bug has now been fixed in the ROI-first path
- the expanded fixed comparison can now be rerun without redesigning the model or benchmark

## What Changed

The live `orchestraiq-jupyter` pod now contains the **full public NSD-Imagery beta release** for all `8` subjects, downloaded with the repo's own acquisition logic:

- `scripts/download_nsd_imagery.py --subjects all --skip-stimuli`

Output root:

- `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery_full_all`

This download added:

- `96` files
- `11.2 GB`
- full imagery metadata
- full imagery beta bundles for `subj01..subj08`

It also became clear that the earlier “truncated source” hypothesis was wrong in one important sense:

- the currently mounted imagery source had missing subjects and missing derived files
- but the public full source still yields the same `720` source trials per subject and the same `5` shared `setB` `nsdId`s after canonical filtering

So the real upside came from **adding subjects**, not from increasing per-subject imagery coverage.

## Rebuilt Canonical Imagery Coverage

After rebuilding canonical imagery indices from the full source, every subject produced:

- `80` canonical imagery rows
- `5` unique `nsdId`s
- `5` paired groups

Those `5` ids are:

- `28752`
- `30857`
- `53882`
- `61178`
- `65873`

## Expanded Overlap Dataset

The overlap dataset is now:

- subjects included: `subj02`, `subj03`, `subj05`, `subj07`
- shared overlap ids: `5`
- total rows: `94`
- splits: `train=56`, `val=19`, `test=19`
- readiness: `bootstrap_ready`

Per-subject overlap breakdown:

- `subj02`: `2` ids, `38` mixed rows
- `subj03`: `1` id, `18` mixed rows
- `subj05`: `1` id, `19` mixed rows
- `subj07`: `1` id, `19` mixed rows

Subjects inspected but skipped:

- `subj01`: `0` overlap ids
- `subj04`: `0` overlap ids
- `subj06`: `0` overlap ids
- `subj08`: `0` overlap ids

So the ceiling was broken:

- previous ceiling: `4 shared `nsdId` pairs`
- current ceiling: `5` paired ids

## Fixed Comparison That Stayed The Same

The intended comparison remained fixed:

- canonical shared-private decoder
- simple legacy Ridge baseline
- same `768`-D ViT-L/14 target space
- same canonical split logic
- same evaluation metrics

No target-space or benchmark-surface changes were introduced.

## Ridge Baseline On The Expanded Dataset

Artifacts:

- `outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/metrics.json`

Results:

- selected alpha: `100.0`
- validation cosine: `0.47876`
- validation MSE: `0.001357`
- test cosine: `0.55199`
- test MSE: `0.001167`
- paired eval groups: `1`
- imagery cosine mean: `0.55152`
- perception cosine mean: `0.55446`

Compared with the earlier `4`-pair max-available run:

- previous Ridge test cosine: `0.42202`
- expanded Ridge test cosine: `0.55199`

So the simple baseline benefited materially from the small increase in overlap scale.

## Canonical Shared-Private Run Status

The earlier expanded run was blocked by a multi-subject raw-fMRI batching
assumption. That infrastructure issue is now fixed in the canonical codebase.

The fix is intentionally minimal:

- the shared/private model continues to train on canonical ROI branch inputs
- raw `fmri` is optional in `DecoderBatch`
- the dataset no longer forces a raw voxel load when serialized ROI features are
  already present
- the collate path stacks raw `fmri` only when every sample has the same shape;
  otherwise it keeps the ROI features and drops the incompatible stacked raw
  tensor

This preserves single-subject compatibility while making the official
multi-subject overlap path scientifically cleaner and operationally valid.

## Interpretation

The important scientific and engineering conclusions are:

1. The overlap ceiling is no longer `4`; it is now `5`.
2. The imagery acquisition path is real and reproducible.
3. The simple Ridge baseline improves on the larger set.
4. The earlier canonical failure was an infrastructure issue, not evidence
   against the model concept.
5. The expanded canonical comparison should now be interpreted from the fresh
   rerun metrics, not from the pre-fix batching failure.

This means the project has crossed an important threshold:

- the main limiting factor is no longer “can we get more imagery overlap at all?”
- the new limiting factor is “can the canonical trainer handle cross-subject raw-fMRI heterogeneity without changing the benchmark?”

## Recommended Next Step

The next justified move remains a fixed-comparison rerun on the expanded overlap
set, followed by a direct comparison against the already fresh Ridge baseline.
Beyond that immediate rerun, the larger strategic need is still further data expansion so the benchmark can move beyond this tiny overlap regime.
