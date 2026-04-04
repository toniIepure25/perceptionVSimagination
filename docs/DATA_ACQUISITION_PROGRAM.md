# Data Acquisition Program

This document defines the current external-data program for the repository.

## Remote execution rule

Real acquisition and prep work should use the remote cluster environment as the
execution surface of record.

Verified on `2026-04-04`:

- namespace: `runai-romania-dev`
- live pod: `orchestraiq-jupyter-75555bb5f5-hxwp5`
- preferred access path: `kubectl exec`
- repo path: `/home/jovyan/local-data/perceptionVSimagination`
- existing shared data root: `/home/jovyan/work/data`
- free space before any new download: about `215G`

Current blocker:

- the live pod does not yet have the repo `.venv`, so canonical workflow
  entrypoints should not be run there until that environment is provisioned

Operational consequence:

- check remote storage and dataset presence first
- do not begin a new public download on the pod until `.venv` parity is fixed
- keep local-only dry runs separate from remote acquisition of record

## Chosen track

The current program is on:

- `Track B — External Data Readiness / Acquisition Plan`

Why:

- the public NSD-Imagery source has already been integrated canonically
- that public source only expands the current benchmark to `5` shared paired
  `nsdId`s across `subj02`, `subj03`, `subj05`, and `subj07`
- no larger paired source is currently mounted in the accessible environment
- the next decisive bottleneck is therefore external paired-data acquisition,
  not more modeling on the same tiny benchmark

## Current ceiling

The current fixed real benchmark is:

- prepared mixed index:
  `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired groups: `1`
- official ladder:
  - Ridge
  - Animus Core Decoder (shared-only)
  - shared-private `private_dim=16`

This ceiling is already reflected in:

- [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md)
- [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md)
- [EXPANDED_OVERLAP_COMPARISON.md](EXPANDED_OVERLAP_COMPARISON.md)

## Source matrix

| Source | Contains | Truly paired / overlapable for current ladder? | Helps Animus lane? | Difficulty | Likely payoff | Current verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Public NSD-Imagery via NSD S3 / OpenNeuro `ds004937` | imagery metadata, imagery betas, optional stimuli | `yes`, but public release is already exhausted in the current program and still yields only `5` canonical overlapable ids | `yes` | low | already realized | keep as the canonical public source, but not the next decisive expansion |
| Unmounted internal / lab-local NSD-style paired imagery or richer paired caches | same-task or near-same-task perception/imagery data with more shared stimuli | `yes` | `yes` | moderate | highest immediate payoff | best practical next acquisition |
| Alternate public paired imagery/perception dataset | paired perception/imagery or reality-monitoring style data with explicit imagery/perception structure | `partial` | limited | moderate to high | modest for the threshold benchmark, useful for a secondary paper path | backup research option, not first choice |
| Large public perception-only natural image datasets | many perception trials, many subjects, strong visual targets | `no` for the threshold ladder | `yes` | moderate | high for Animus robustness, low for the paired threshold question | best practical complement, not the next paired-data answer |
| New custom paired perception/imagery data with subjective labels | richer paired content plus vividness/confidence/reality metadata | `yes` | `yes` | very high | highest long-term payoff | strongest future program path |

## Ranked next data options

### 1. Best practical immediate acquisition

Acquire or mount a richer paired NSD-style imagery/perception source than the
current public release.

Why this is rank 1:

- it preserves the benchmark ladder exactly
- it uses the current canonical schema and prep workflows
- it helps both the research lane and the practical Animus lane
- it is the only realistic near-term option that can directly answer the
  threshold question without changing the task definition

Remote note:

- this class still depends on the live pod gaining a repo `.venv` before any
  canonical rebuild or rerun work should begin

## 2. Best medium-effort acquisition

Integrate a public paired imagery/perception dataset as a secondary benchmark.

Concrete example:

- legacy OpenfMRI `ds000203` (“Visual imagery and false memory for pictures”)

Why this is rank 2:

- it is public and genuinely imagery-related
- it could support a second, smaller paired benchmark or future
  reality-monitoring-style paper path

Why it is not rank 1:

- it is not the same natural-image paired setting as the current ladder
- it would need dataset-specific canonicalization and careful claim boundaries

## 3. Best high-upside acquisition

Collect or obtain a larger custom paired perception/imagery dataset with a much
larger shared stimulus set.

Why this is rank 3 instead of rank 1:

- highest scientific upside
- highest operational cost
- depends on data collection or outside collaboration

## 4. Best practical Animus-only complement

Bring in a large public perception-only dataset such as:

- NOD (`ds004496`)
- BOLD5000 (`ds001499`)
- THINGS-fMRI (`ds004192`)

Why this matters:

- it can strengthen the shared-only Animus Core Decoder lane later
- it can improve robustness and export confidence for the practical subsystem

Why it is not the immediate next answer:

- it does not directly break the paired perception/imagery threshold benchmark
- on the current cluster pod, no candidate public datasets are already staged,
  so the first step is still remote environment alignment plus disciplined
  acquisition planning

Current first practical move:

```bash
./.venv/bin/python -m fmri2img.workflows.acquire_public_nod \
  --output cache/public_datasets/ds004496
```

This is a metadata-only Git acquisition surface for the practical Animus lane.
It is intentionally smaller than a full annex download and does not change the
primary threshold ladder.

Current status:

- completed on the live pod
- output path:
  `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`
- provenance file:
  `cache/public_datasets/ds004496/acquisition_provenance.json`

## Canonical public source path

The official public imagery acquisition command is now:

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

This thin wrapper delegates to:

- `scripts/download_nsd_imagery.py`

Canonical public source locations already used by the repo:

- NSD metadata:
  - `s3://natural-scenes-dataset/nsddata/experiments/nsdimagery/`
- NSD imagery betas:
  - `s3://natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
- OpenNeuro public dataset:
  - `https://openneuro.org/datasets/ds004937`

This public source is important because it remains the reproducible baseline
acquisition path. It is just no longer sufficient by itself to answer the main
threshold question.

## What counts as success for the next data move

The next acquisition should be considered scientifically successful only if it
produces:

- materially more than `5` shared paired `nsdId`s
- more than `1` meaningful held-out paired evaluation group
- a benchmark rerun where the fixed ladder can be executed without changing:
  - target space
  - evaluation surface
  - practical vs exploratory lane separation

## Decision rule once new data appears

As soon as a larger paired source is available, rerun the ladder unchanged:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

If shared-private `private_dim=16` still fails to close the gap on materially
larger paired data, the threshold hypothesis should be weakened accordingly.
