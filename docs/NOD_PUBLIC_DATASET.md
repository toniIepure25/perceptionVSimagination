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
- common session set: `ses-imagenet01..04`
- expected common-session runs per subject: `40`
- derivative source: `ciftify` `*_beta.dscalar.nii` with paired `*_label.txt`
- status: inspection-ready, not training-ready

Why this is the smallest viable contract:

- `imagenet` is the cleanest perception task visible across the dataset
- the multi-session subjects provide the richest repeated structure
- `ses-imagenet01..04` is the shared session core across `sub-01..sub-09`
- `sub-01` has an extra `ses-imagenet05`, so it should stay outside the first
  prepared-index contract
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

This now includes a `prepared_index_contract` block that names:

- the common `imagenet` session set
- the multi-session subject subset
- the expected common-session run count per subject
- the exact raw and derivative file patterns the first adapter must rely on

## First prepared-index build surface

The first checked-in prepared-index workflow is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_index
```

Default output:

- `cache/indices/public_nod/imagenet_multisession_common_sessions.parquet`

Default report:

- `cache/indices/public_nod/imagenet_multisession_common_sessions.report.json`

## What the prepared index records

Each row captures:

- subject
- session
- run
- task
- path to `events.tsv`
- path to `fmriprep` preproc BOLD
- path to `fmriprep` confounds
- path to `ciftify` beta
- path to `ciftify` label
- `visible` flags
- `resolved` flags
- `row_status`
- `usable_for_later_shared_only_prep`

Current row-level status meanings:

- `resolved`: all required payloads are actually present
- `missing_payload`: all required paths are visible, but at least one payload is
  not materialized behind the visible path
- `incomplete`: only part of the row contract is visible
- `missing`: none of the required inputs are visible

## Current live-pod prepared-index result

Built on the live pod from:

- `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`

Outputs:

- prepared index:
  `/home/jovyan/local-data/perceptionVSimagination/cache/indices/public_nod/imagenet_multisession_common_sessions.parquet`
- report:
  `/home/jovyan/local-data/perceptionVSimagination/cache/indices/public_nod/imagenet_multisession_common_sessions.report.json`

Observed result on the current clone:

- rows: `360`
- row status counts:
  - `incomplete`: `324`
  - `missing_payload`: `36`
- usable rows for later shared-only prep: `0`

Current dominant blockers:

- only `36` rows expose visible `events.tsv`
- `fmriprep` BOLD and confounds are visible for all `360` rows, but not
  resolved as payloads
- `ciftify` beta and label paths are visible for all `360` rows, but not
  resolved as payloads

Interpretation:

- the first prepared index is real and useful
- it proves the subset contract and the visibility/resolution surface
- it does **not** justify shared-only training yet

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
