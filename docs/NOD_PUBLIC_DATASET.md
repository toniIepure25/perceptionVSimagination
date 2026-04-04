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
  - `resolved`: `36`
- usable rows for later shared-only prep: `36`

Current scope boundary:

- only the fixed `run-10` slice is resolved
- the remaining `324` rows are still incomplete
- this does not justify widening the NOD contract beyond the currently resolved
  subset

Interpretation:

- the first prepared index is real and useful
- it proves the subset contract and the visibility/resolution surface
- it now supports a narrow downstream shared-only adapter surface
- it does **not** justify shared-only training yet

## Exact missing-payload manifest surface

The next checked-in surface is a tight materialization manifest helper:

```bash
./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads
```

Default outputs:

- manifest:
  `cache/indices/public_nod/imagenet_missing_payload_manifest.json`
- report:
  `cache/indices/public_nod/imagenet_missing_payload_report.json`

What it does:

- reads the current prepared index
- selects rows with `row_status == missing_payload`
- records the exact unresolved file paths per row
- estimates payload size from the annex symlink targets
- optionally runs `git annex get` only if `git-annex` is actually available

Current live-pod result from the manifest audit:

- first materialization target rows:
  - `36` rows
  - exactly the `run-10` rows across `sub-01..sub-09` and
    `ses-imagenet01..04`
- estimated missing payload size:
  - `preproc_bold`: about `7.382 GiB`
  - `confounds`: about `0.040 GiB`
  - `ciftify_beta`: about `0.808 GiB`
  - `ciftify_label`: negligible
  - total: about `8.23 GiB`

Current operational result:

- the exact missing-payload subset can be retrieved directly from the official
  OpenNeuro public S3 bucket
- base URL:
  `https://s3.amazonaws.com/openneuro.org/ds004496/`
- the helper now supports a direct strategy that writes into the existing
  annex-object targets instead of broadening the dataset clone

Interpretation:

- this is a real readiness improvement for the practical Animus lane
- it is still **not** a claim that NOD is threshold evidence or that the whole
  NOD subset is training-ready

## Live pod annex-enablement result

Verified on `2026-04-04` on pod `orchestraiq-jupyter-75555bb5f5-hxwp5`:

- `git-annex` was safely enabled in place via:
  `apt-get install -y --no-install-recommends git-annex`
- package footprint was modest:
  - about `17.4 MB` download
  - about `105 MB` additional installed disk usage
- the helper now runs real `git annex get` attempts on the live pod

Materialization outcome for the current exact subset:

- exact target manifest:
  `cache/indices/public_nod/imagenet_missing_payload_manifest.json`
- retrieval strategy:
  `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize --strategy direct_openneuro_s3`
- retrieved subset:
  - `36` rows
  - `144` files total
  - `36` `fmriprep` preproc BOLD files
  - `36` `fmriprep` confounds TSVs
  - `36` `ciftify` beta files
  - `36` `ciftify` label files
  - total downloaded size: about `8.23 GiB`
- rerunning the prepared index now yields:
  - `324` `incomplete`
  - `36` `resolved`
  - `36` usable rows

This is the first real NOD payload-ready subset for the practical Animus lane.
It stays narrow:

- `run-10` only
- `sub-01..sub-09`
- `ses-imagenet01..04`
- still no claim that the wider NOD subset is ready

## Fixed shared-only adapter surface

The smallest downstream shared-only adapter surface for the resolved slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_adapter
```

Default outputs:

- adapter parquet:
  `cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet`
- adapter report:
  `cache/indices/public_nod/imagenet_run10_shared_only_adapter.report.json`

What it enforces:

- `task == imagenet`
- `subject in sub-01..sub-09`
- `session in ses-imagenet01..04`
- `run == 10`
- `usable_for_later_shared_only_prep == True`

Current meaning:

- adapter-ready: yes
- prep-ready: yes
- training-ready: no

Why it is not yet training-ready:

- the adapter only turns the resolved NOD slice into a stable downstream prep
  artifact
- the repo still lacks a canonical target-selection contract for NOD stimuli
- the repo still lacks an ROI materialization contract aligned to this NOD path
- no checked-in shared-only train/eval config should point here until those
  contracts are explicit

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
