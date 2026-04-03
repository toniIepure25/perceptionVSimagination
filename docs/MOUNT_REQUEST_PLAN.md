# Mount Request Plan

> Historical note: this document reflects the earlier imagery-mount request
> phase. For the current next-step acquisition program after the public
> `5`-id ceiling was established, use
> [DATA_ACQUISITION_PROGRAM.md](DATA_ACQUISITION_PROGRAM.md) and
> [EXTERNAL_DATA_INTEGRATION_PLAN.md](EXTERNAL_DATA_INTEGRATION_PLAN.md).

## Status Update

The original request in this document has now been largely satisfied in the live pod:

- the full public NSD-Imagery source was downloaded successfully for `subj01..subj08`
- canonical imagery rebuild now works from that full source
- the overlap dataset increased from `4` to `5` shared `nsdId`s by adding `subj03`

So this document should now be read mainly as historical context.

The new blocking issue is different:

- the canonical multi-subject trainer still tries to stack raw full-fMRI vectors across subjects
- `subj03` introduces a different voxel dimensionality
- the expanded canonical train now fails with `ValueError: all input arrays must have the same shape`

So the next operational request is no longer “mount more imagery.”

It is:

- either provide a canonical subject-aligned fMRI representation for multi-subject training
- or harden the canonical trainer so its reconstruction target does not require identical raw voxel dimensionality across subjects

## Goal

Break the current `4`-pair overlap ceiling by acquiring the missing imagery-side data that the canonical prep path already knows how to consume.

## Best Practical Request

Request a mount or copy of the **full NSD-Imagery release** for the current imagery-capable project subjects:

- `subj01`
- `subj02`
- `subj05`
- `subj07`

Why this is the best first request:

- the pod already has canonical perception support for these subjects
- the currently mounted imagery bundles are clearly reduced bootstrap bundles
- canonical prep is already working for these subjects once the data exists

## Exact Paths To Acquire

The repo’s own downloader expects these source prefixes:

- metadata:
  - `nsddata/experiments/nsdimagery/`
- imagery betas:
  - `nsddata_betas/ppdata/subj01/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj02/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj05/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj07/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
- optional imagery stimuli:
  - `nsddata_stimuli/stimuli/nsdimagery/`

If mounting from an existing local or shared copy is easier, the canonical target layout can be normalized to:

- `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery/metadata/`
- `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery/betas/{subject}/`

or to the current split-layout beta root:

- `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/betas/{subject}/`

## What This Unlocks Immediately

Once those paths exist, the official canonical commands become:

```bash
python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --override dataset.subject="\"subj01\""

python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --override dataset.subject="\"subj02\""

python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --override dataset.subject="\"subj05\""

python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --override dataset.subject="\"subj07\""

python -m fmri2img.workflows.prepare_overlap_bootstrap \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml

python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

## Expected Payoff

### Option A: Full Imagery For The Current 4 Subjects

Expected payoff:

- highest practical payoff
- likely much more than the current `4` shared `nsdId`s
- should be the first move before any new modeling work

Effort:

- moderate

### Option B: Add `subj03` Imagery Only

Expected payoff:

- likely the best single-subject incremental gain if only one new subject can be added quickly
- modest if the imagery bundle is still tiny
- much better if the full imagery package exists

Effort:

- moderate

### Option C: Add Imagery For `subj03/04/06/08`

Expected payoff:

- best high-upside move if the full NSD-Imagery source truly covers all `8` subjects

Effort:

- moderate to high

## Suggested Handoff Message

Use this if you want to ask an ops owner or cluster admin for the missing data:

> Please mount or copy the full NSD-Imagery metadata and subject beta packages into the `orchestraiq-jupyter` environment for `subj01`, `subj02`, `subj05`, and `subj07` first, and `subj03/04/06/08` if available. The current pod only has reduced imagery bundles with about `720` rows per subject and about `0.36–0.45 GB` per subject, which caps the canonical overlap dataset at `4` shared `nsdId`s. The canonical prep pipeline is already working; it only needs the missing imagery data to expand overlap.

## Secondary Request

If someone can only provide one smaller addition first:

- request `subj03` imagery beta data and matching metadata

Why:

- perception support for `subj03` already exists
- it is the strongest candidate for a near-term incremental overlap gain after the current `subj02/subj05/subj07` set
