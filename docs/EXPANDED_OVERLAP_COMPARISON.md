# Expanded Overlap Comparison

This report should now be read together with:

- [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md)
- [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md)
- [ANIMUS_CORE_DECODER.md](ANIMUS_CORE_DECODER.md)
- [TOP_LEVEL_RESEARCH_DOSSIER.md](TOP_LEVEL_RESEARCH_DOSSIER.md)

## Summary

This document records the first successful expansion of the canonical overlap dataset after acquiring the full public NSD-Imagery source.

Checked-in config for this scaling audit:

- `configs/canonical/max_available_overlap.yaml`

Question:

- As overlap mixed-condition data grows, does the canonical shared-private model begin to become competitive with the simple Ridge baseline?

Current answer:

- the overlap dataset did grow beyond the old 4-pair ceiling
- the canonical multi-subject batching bug has now been fixed in the ROI-first path
- the expanded fixed comparison has now been rerun successfully
- the canonical shared-private model improves modestly on the larger set, but Ridge still dominates by a wide margin
- a shared-only ablation materially outperforms the full shared-private model on this same dataset
- a narrow private-capacity sweep improves shared-private somewhat, but shared-only still remains the best canonical neural baseline

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
assumption. That infrastructure issue is now fixed in the canonical codebase,
and the expanded canonical rerun completed successfully.

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

Fresh canonical artifacts:

- `outputs/canonical/train/full_imagery_overlap/best_decoder.pt`
- `outputs/canonical/train/full_imagery_overlap/train_history.json`
- `outputs/canonical/eval/full_imagery_overlap/metrics.json`
- `outputs/canonical/transfer/full_imagery_overlap/transfer_metrics.json`
- `outputs/canonical/analysis/full_imagery_overlap/roi_resolved_summary.json`
- `outputs/canonical/export/full_imagery_overlap/manifest.json`

Fresh canonical results:

- validation cosine at best epoch: `0.06319`
- test cosine: `0.06927`
- test MSE: `0.002424`
- imagery cosine mean: `0.07151`
- perception cosine mean: `0.05735`
- paired eval groups: `1`
- imagery-minus-perception gap: `0.01415`
- domain accuracy: `0.89474`

Training history summary:

- epoch 1: train loss `31.58`, val cosine `0.02842`
- epoch 5: train loss `26.29`, val cosine `0.06319`

Transfer metrics matched the eval metrics exactly in this run because the held-out
paired evaluation set still contains only `1` usable pair group.

## Interpretation

The important scientific and engineering conclusions are:

1. The overlap ceiling is no longer `4`; it is now `5`.
2. The imagery acquisition path is real and reproducible.
3. The earlier canonical failure was an infrastructure issue, not evidence
   against the model concept.
4. The canonical model now trains and evaluates correctly on the expanded
   multi-subject set without relying on equal raw voxel dimensionality.
5. The canonical model shows a weak positive content signal on the larger set,
   but Ridge remains dramatically stronger.
6. This run is still too small to validate the paper hypothesis on performance.

This means the project has crossed an important threshold:

- the main limiting factor is no longer “can we get more imagery overlap at all?”
- the new limiting factor is dataset scale, not the basic viability of the
  canonical multi-subject training path

## Comparison Snapshot

Canonical shared-private vs refreshed Ridge on the same `5`-id expanded overlap set:

- canonical cosine: `0.06927`
- Ridge cosine: `0.55199`
- canonical MSE: `0.002424`
- Ridge MSE: `0.001167`
- canonical imagery mean: `0.07151`
- Ridge imagery mean: `0.55152`
- canonical perception mean: `0.05735`
- Ridge perception mean: `0.55446`

The gap remains very large. The canonical model is now operationally valid on
the expanded dataset, but there is no performance evidence yet that it is
catching up to a simple linear decoder at this scale.

The high canonical domain accuracy should not be overinterpreted. The held-out
test split is tiny and imbalanced (`16` imagery samples vs `3` perception
samples), so this metric is mainly a sanity check that the private-latent/domain
path is numerically active.

## Minimal Ablation Follow-Up

To avoid changing too many variables at once, the next pass kept the same:

- dataset
- target space
- split logic
- evaluation surface

and added only disciplined model ablations.

### Shared-only ablation

Overrides:

- `model.disentanglement_mode="shared_only"`
- `model.use_domain_head=false`

Artifacts:

- `outputs/canonical/train/full_imagery_overlap_shared_only/best_decoder.pt`
- `outputs/canonical/eval/full_imagery_overlap_shared_only/metrics.json`

Results:

- val cosine: `0.07430`
- test cosine: `0.13596`
- test MSE: `0.002250`
- imagery mean: `0.13422`
- perception mean: `0.14527`

### Shared-private with domain head disabled

Overrides:

- `model.use_domain_head=false`

Artifacts:

- `outputs/canonical/train/full_imagery_overlap_nodomain/best_decoder.pt`
- `outputs/canonical/eval/full_imagery_overlap_nodomain/metrics.json`

Results:

- val cosine: `0.04813`
- test cosine: `0.05907`
- test MSE: `0.002450`

### Ablation interpretation

Ordering on the same `5`-id dataset:

- Ridge: `0.55199`
- shared-only: `0.13596`
- shared-private: `0.06927`
- shared-private no-domain: `0.05907`

The most important conclusion is that shared-only is clearly stronger than full
shared-private on the current tiny overlap set. Disabling the domain head alone
does not recover that gap, so the main drag is not just the domain auxiliary.
The private/disentanglement structure itself is currently too costly relative to
the available data.

## Narrow Recovery Sweep

Because further overlap expansion was not possible from the current mounted
environment, the next disciplined step was a very small shared-private recovery
suite with the benchmark held fixed.

Variants tested:

- `private_dim=16`
- `private_dim=8`

Everything else was kept fixed:

- same expanded overlap dataset
- same `vit_l14_image_768` targets
- same canonical evaluation path

Results:

- shared-private, `private_dim=16`
  - test cosine: `0.10784`
  - test MSE: `0.002323`
- shared-private, `private_dim=8`
  - test cosine: `0.09595`
  - test MSE: `0.002354`

Interpretation:

- smaller private capacity does help shared-private
- `private_dim=16` is the best recovery variant tried so far
- neither recovery variant beats shared-only (`0.13596`)
- Ridge still dominates overwhelmingly

So the current evidence-based ordering is now:

1. Ridge
2. shared-only
3. shared-private (`private_dim=16`)
4. shared-private (`private_dim=8`)
5. shared-private
6. shared-private without domain head

## Recommended Next Step

The next justified move is further data expansion with the comparison held
fixed.

Only after a materially larger overlap set exists should the project consider:

- targeted regularization or loss tuning
- private-latent capacity reduction informed by the shared-only result
- improved ROI decomposition
