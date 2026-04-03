# External Data Integration Plan

This document is the ready-to-execute playbook for integrating the next larger
paired perception/imagery data source into the canonical benchmark ladder.

## Goal

Break out of the current `5`-id overlap regime without changing:

- the target space: `vit_l14_image_768`
- the practical Animus lane: shared-only
- the exploratory threshold lane: shared-private `private_dim=16`
- the external reference: Ridge

## Preferred data class

### Class A. Richer NSD-style paired perception/imagery data

This is the preferred next source class because it preserves the benchmark
semantics exactly.

Desired properties:

- perception and imagery from the same or directly alignable stimulus universe
- recoverable `nsdId` or an equivalent stable shared stimulus identifier
- subject-level beta files or equivalent single-trial response estimates
- enough repeated shared items to increase held-out paired evaluation strength

### Class B. Alternate public paired imagery/perception data

Use this only if Class A is unavailable.

It can support a secondary paper path, but it is weaker for the current
threshold ladder because task semantics may differ.

### Class C. Perception-only data

Useful for Animus Core Decoder robustness later, but not sufficient for the
paired threshold benchmark by itself.

## Phase 1: Acquire or mount the source

### Public NSD-Imagery baseline path

If the task is to reproduce or refresh the public imagery source, use:

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

Fallback:

```bash
python scripts/download_nsd_imagery.py \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

### Internal or mounted richer paired source

If a larger paired source is mounted or copied from external storage, normalize
it into one of these canonical layouts:

Subject-rooted layout:

- `cache/nsd_imagery_external/{subject}/...`

Split layout:

- metadata:
  - `cache/nsd_imagery_external/metadata/`
- subject betas:
  - `cache/nsd_imagery_external/betas/{subject}/betas_nsdimagery.nii.gz`

Required provenance to record:

- source dataset name
- snapshot / version / DOI if available
- acquisition date
- subject list
- file sizes
- whether source is public, mounted internal storage, or partner-provided

## Phase 2: Canonicalize imagery indices

Set the external-data env vars:

```bash
export PYTHONPATH=src
export NSD_IMAGERY_ROOT=/abs/path/to/nsd_imagery_external
export NSD_IMAGERY_METADATA_ROOT=/abs/path/to/nsd_imagery_external/metadata
export NSD_IMAGERY_BETA_ROOT=/abs/path/to/nsd_imagery_external/betas
export NSD_ROI_MASK_ROOT=/abs/path/to/roi_masks_parent
export NSD_HDF5=/abs/path/to/nsd_stimuli.hdf5
```

Then rebuild imagery indices through the canonical workflow surface:

```bash
for subject in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
  python -m fmri2img.workflows.prepare_imagery_index \
    --config configs/canonical/max_available_overlap.yaml \
    --override dataset.subject="\"${subject}\""
done
```

Success criteria for this phase:

- each subject report is written
- `nsdId` recovery succeeds where the source is truly pairable
- the subject-level imagery indices remain canonical and provenance-aware

## Phase 3: Rebuild the paired overlap benchmark

Once the imagery indices are refreshed:

```bash
python -m fmri2img.workflows.prepare_overlap_bootstrap \
  --config configs/canonical/max_available_overlap.yaml \
  --overwrite-existing
```

```bash
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/max_available_overlap.yaml
```

```bash
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/max_available_overlap.yaml
```

The benchmark is only worth rerunning if preflight and the overlap report show:

- more than `5` shared paired ids
- more than `1` meaningful held-out paired group

## Phase 4: Rerun the fixed ladder

Once the new overlap benchmark is materially larger, rerun:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
```

```bash
python -m fmri2img.workflows.train_animus_core_decoder
```

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

Then complete the corresponding eval/export steps:

```bash
python -m fmri2img.workflows.eval_animus_core_decoder --checkpoint ...
python -m fmri2img.workflows.export_animus_core_decoder --checkpoint ...
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --checkpoint ...
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --checkpoint ...
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --checkpoint ...
```

## Phase 5: Decision rule after the rerun

Interpret the rerun this way:

- if Ridge still dominates and shared-only still beats shared-private:
  - the threshold hypothesis remains unconfirmed
- if shared-private `private_dim=16` narrows the gap meaningfully:
  - the threshold hypothesis gains support
- if shared-private overtakes shared-only on materially larger overlap:
  - the exploratory lane becomes eligible for promotion

Do not change the ladder or target space before this rerun is completed.

## Alternate-source integration rule

If a non-NSD public paired dataset is used instead, do not silently merge it
into the current benchmark.

Instead:

- normalize it into the canonical schema first
- document the dataset-specific stimulus identifier
- declare whether it is a secondary paired benchmark or a future paper path
- keep the current NSD-style threshold ladder distinct unless the task/stimulus
  semantics are genuinely comparable
