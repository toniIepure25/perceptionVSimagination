# Public NOD Integration Note

This note defines the first checked-in integration surface for OpenNeuro
`ds004496` (NOD).

## Role in this repo

`ds004496` is:

- perception-only
- a practical Animus-lane strengthening dataset
- not a replacement for the primary paired threshold ladder

Use it to strengthen the shared-only practical subsystem. Do not present it as
direct evidence for the current threshold question.

## First safe acquisition mode

The first checked-in acquisition surface is intentionally small:

- metadata-only Git clone from the public OpenNeuro dataset mirror

Why start here:

- it is reproducible
- it is cheap in disk usage
- it gives the repo a durable remote provenance surface before larger annexed
  downloads are attempted

## Canonical command

Run from the project `.venv`, preferably on the live cluster pod:

```bash
./.venv/bin/python -m fmri2img.workflows.acquire_public_nod \
  --output cache/public_datasets/ds004496
```

Dry run:

```bash
./.venv/bin/python -m fmri2img.workflows.acquire_public_nod \
  --output cache/public_datasets/ds004496 \
  --dry-run
```

## What this command does

- clones `https://github.com/OpenNeuroDatasets/ds004496.git`
- writes `acquisition_provenance.json`
- records that the mode is `metadata_only_git_clone`

## What this command does not do

- it does not download annexed imaging content
- it does not canonicalize NOD into the current ROI-first training contract
- it does not change the primary benchmark ladder

## What the live metadata clone actually shows

Verified on the live pod at:

- `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`

The current clone exposes:

- `30` BIDS subjects
- a multi-session cohort:
  - `sub-01..sub-09`
- a single-session cohort:
  - `sub-10..sub-30`
- raw BIDS task/session structure for:
  - `imagenet`
  - `coco`
  - `floc`
  - `prf`
- `fmriprep` derivatives with:
  - `*_space-T1w_desc-preproc_bold.nii.gz`
  - `*_desc-confounds_timeseries.tsv`
- `ciftify` derivatives with:
  - `*_Atlas.dtseries.nii`
  - `*_beta.dscalar.nii`
  - `*_label.txt`
- floc surface ROI label assets such as:
  - `floc-faces.dlabel.nii`
  - `floc-places.dlabel.nii`
  - `floc-bodies.dlabel.nii`
  - `floc-words.dlabel.nii`

This means the clone is already stronger than a metadata-only manifest. It is
inspection-ready for a perception-only surface contract.

## Smallest viable NOD contract for this repo

The smallest honest contract is:

- lane: practical Animus only
- task family: `imagenet` perception-only
- subject tier: start with multi-session subjects `sub-01..sub-09`
- derivative source: `ciftify` `*_beta.dscalar.nii` with paired `*_label.txt`
- status: inspection-ready, not training-ready

Why this is the smallest viable contract:

- `imagenet` is the cleanest perception task visible across the dataset
- the multi-session subjects provide the richest repeated structure
- the surface GLM beta plus label pairing is already visible in the clone

What is still missing before a real shared-only run:

- a canonical target-selection contract for NOD stimuli
- a checked-in prepared-index adapter
- an ROI materialization contract aligned to NOD derivatives
- a real shared-only train/eval config that points to a prepared NOD index

## Canonical inspection command

Run from the project `.venv`, preferably on the live pod:

```bash
./.venv/bin/python -m fmri2img.workflows.inspect_public_nod
```

JSON summary:

```bash
./.venv/bin/python -m fmri2img.workflows.inspect_public_nod --json
```

## Expected remote path

On the verified live pod:

- repo path:
  `/home/jovyan/local-data/perceptionVSimagination`
- canonical NOD metadata target:
  `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`

## Next step after metadata acquisition

Once the metadata clone exists, the next practical move is:

1. inspect the dataset layout and derivative availability
2. define the first prepared-index adapter around `imagenet` plus the
   multi-session cohort
3. add a config skeleton only after the prepared-index contract is explicit
