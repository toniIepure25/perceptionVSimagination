# Public Dataset Integration Plan

This document turns the public dataset audit into concrete repo actions.

The current ordering remains:

- primary fixed ladder:
  - Ridge
  - shared-only
  - shared-private `private_dim=16`

No public dataset in this plan silently replaces that ladder.

## Remote execution surface of record

Real dataset acquisition and canonical prep should run on the remote cluster
environment, not from a local workstation shell.

Verified on `2026-04-04`:

- namespace: `runai-romania-dev`
- live pod: `orchestraiq-jupyter-75555bb5f5-hxwp5`
- preferred access path: `kubectl exec`
- repo path on pod: `/home/jovyan/local-data/perceptionVSimagination`
- existing data root: `/home/jovyan/work/data`
- existing NSD root: `/home/jovyan/work/data/nsd`
- repo cache/output roots already in use:
  - `/home/jovyan/local-data/perceptionVSimagination/cache`
  - `/home/jovyan/local-data/perceptionVSimagination/outputs`
- free space before any new download: about `215G`

Current remote constraint:

- the pod currently exposes `/opt/conda/bin/python` but does **not** yet have
  `/home/jovyan/local-data/perceptionVSimagination/.venv/bin/python`

Operational rule:

- inspect storage and mounted paths on the pod first
- verify whether the target dataset already exists before downloading
- do not start canonical acquisition, prep, or rerun commands on the pod until
  the repo `.venv` is provisioned there

Current remote status after environment provisioning:

- the repo `.venv` now exists on the live pod
- the first `ds004496` step has been completed as a metadata-only Git clone at
  `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`
- annexed imaging content has not been acquired yet

## Immediate priorities

### Priority 1. Practical Animus strengthening

Target:
- `ds004496` (NOD)

Why first:
- strongest immediate public perception-only dataset for the practical lane
- large subject count and image scale
- useful for robust shared-only subsystem strengthening without changing the
  threshold benchmark meaning

Expected role:
- Animus-only support dataset

What to implement next:
1. provision the repo `.venv` on the remote pod
2. use `fmri2img.workflows.acquire_public_nod` for the first metadata-only
   remote acquisition step
3. inspect the real NOD layout before attempting a larger annexed download
4. add a dataset-provenance doc block
5. add a perception-only canonical adapter plan
6. add a config skeleton for a perception-only shared-only training/eval path

Current status:

- steps `1` and `2` are now complete on the live pod
- the next minimal contract is `imagenet` + multi-session subjects
  `sub-01..sub-09` + common sessions `ses-imagenet01..04` +
  `ciftify` `*_beta.dscalar.nii` / `*_label.txt` inspection, still short of
  training readiness

Prepared-index contract now defined:

- task: `imagenet`
- subjects: `sub-01..sub-09`
- common sessions: `ses-imagenet01..04`
- expected runs per subject inside the first contract: `40`
- derivative dependence:
  - raw `events.tsv`
  - `fmriprep` preproc BOLD + confounds
  - `ciftify` beta + label pairs

First checked-in build surface:

- `fmri2img.workflows.prepare_public_nod_index`

This builds a row-level prepared index for the first NOD subset and records
payload visibility versus real payload resolution, without overstating
training readiness.

### Priority 2. Secondary imagery benchmark

Target:
- `ds000203`

Why second:
- best current public imagery-related fMRI candidate
- useful for a secondary benchmark or future reality-monitoring paper path

Expected role:
- secondary benchmark, not the primary ladder

What to implement next:
1. dataset-specific canonical identifier plan
2. explicit benchmark-boundary doc
3. remote metadata inspection on the live pod before any adapter stub
4. dataset adapter stub after metadata inspection

### Priority 3. Object-semantic public benchmark

Target:
- `ds004192` (THINGS-fMRI)

Expected role:
- Animus robustness dataset and future object-semantic benchmark

What to implement next:
1. remote acquisition note
2. adapter feasibility check against the existing ROI-first contract
3. config skeleton for a perception-only shared-only path

### Priority 4. Future internally generated state paper path

Target:
- `ds006623`

Expected role:
- future-paper dataset

What to implement next:
1. keep separate from the threshold ladder
2. document as a future source-routing / consciousness path
3. only integrate after the primary public-data program is clearer

## Lane assignment rules

- `ds004937`:
  canonical public NSD imagery source; already integrated baseline public path
- `ds000203`:
  secondary imagery benchmark
- `ds004496`:
  practical Animus strengthening dataset
- `ds004192`:
  practical Animus strengthening dataset plus future object-semantic benchmark
- `ds001499`:
  practical Animus strengthening backup / supporting dataset
- `ds006623`:
  future-paper dataset

## Next concrete repo work

The next implementation step should be:

1. provision the project `.venv` on
   `/home/jovyan/local-data/perceptionVSimagination`
2. verify remote download tooling and target storage on the live pod
3. run the first metadata-only remote acquisition for `ds004496`
4. add a minimal provenance-aware integration note for it
5. keep the threshold ladder untouched while the practical lane gains a public
   strengthening path
