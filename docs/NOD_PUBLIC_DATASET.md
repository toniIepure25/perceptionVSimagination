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
- a canonical target-embedding cache contract over the fixed NOD target slice
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

## Fixed target-selection surface

The smallest canonical target-selection workflow for the resolved adapter slice
is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_selection
```

Default outputs:

- target-selection parquet:
  `cache/indices/public_nod/imagenet_run10_target_selection.parquet`
- target-selection report:
  `cache/indices/public_nod/imagenet_run10_target_selection.report.json`

What it uses:

- the fixed `36`-row shared-only adapter parquet
- `events.tsv` `stim_file` values
- `ciftify` `label.txt` values

Current target identifier contract:

- one trial-level row per selected image presentation
- deterministic target identifier:
  `Path(stim_file).name`
- validation rule:
  `Path(stim_file).name == label.txt entry`

Current meaning:

- target-selection-ready: yes
- downstream-prep-ready: yes
- training-ready: no

What still blocks training after target selection:

- a canonical target embedding cache built from this target-selection artifact
- an ROI materialization contract aligned to the NOD derivatives
- a shared-only training/eval config that points to both the adapter and the
  target-selection outputs

## Fixed target-embedding cache surface

The smallest canonical target-embedding workflow for the fixed target-selection
slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_embedding_cache
```

Default outputs:

- target-embedding manifest parquet:
  `cache/indices/public_nod/imagenet_run10_target_embedding_manifest.parquet`
- target-embedding report:
  `cache/indices/public_nod/imagenet_run10_target_embedding_manifest.report.json`

What it uses:

- the fixed `3600`-row target-selection parquet
- the expected stimulus paths under:
  `cache/public_datasets/ds004496/stimuli/`
- the canonical embedding target space:
  - model id: `openai/clip-vit-large-patch14`
  - dimension: `768`
  - embedding column: `clip_target_768`

Current live-pod interpretation:

- the target-selection slice is deterministic and fixed
- the manifest now records the exact stimulus JPEG paths and whether they are
  visible versus resolved
- before materialization it correctly exposes a manifest-only cache contract
- after the exact fixed-slice JPEG materialization succeeds, rerunning the
  workflow promotes the manifest to `target_embedding_ready: yes`

Current meaning:

- target-embedding-ready: no
- downstream-prep-ready: no
- training-ready: no

What still blocks training after the embedding-cache contract:

- exact NOD stimulus JPEG payloads must be materialized for the fixed `3600`
  target rows
- the canonical `768`-D ViT-L/14 embeddings must then be computed from those
  resolved images
- an ROI materialization contract aligned to the NOD derivatives is still
  required
- a shared-only training/eval config must point to the adapter,
  target-selection artifact, and real target cache

## Fixed stimulus materialization surface

The smallest exact-subset stimulus retrieval workflow for the same fixed slice
is:

```bash
./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_stimuli
```

Materialization run:

```bash
./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_stimuli --materialize
```

Default report:

- `cache/indices/public_nod/imagenet_run10_target_embedding_retrieval_report.json`

What it does:

- consumes the fixed target-embedding manifest
- keeps the exact `3600`-row `imagenet` / `sub-01..sub-09` /
  `ses-imagenet01..04` / `run-10` slice fixed
- downloads only the referenced JPEGs from the official OpenNeuro public S3
  path
- writes them into the existing annex-backed stimulus targets

## Fixed real target-cache build surface

Once the exact JPEG payloads are actually resolved, the next workflow is:

```bash
./.venv/bin/python -m fmri2img.workflows.build_public_nod_target_embedding_cache
```

Default outputs:

- target cache parquet:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- target cache report:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.report.json`

What it does:

- consumes the fixed target-embedding manifest
- verifies that all fixed-slice stimulus JPEGs are resolved
- computes the real canonical embeddings with:
  - model id: `openai/clip-vit-large-patch14`
  - dimension: `768`
  - output column: `clip_target_768`

Current live-pod result for the fixed slice:

- exact JPEG retrieval completed for all `3600` target rows
- the real target cache now exists at:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- report:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.report.json`
- current state:
  - `target_embedding_ready`: yes
  - `downstream_prep_ready`: yes
  - `training_ready`: no

Current training boundary after a real target cache exists:

- target-embedding-ready: yes
- downstream-prep-ready: yes
- training-ready: still no

What still blocks honest shared-only training even after the cache exists:

- an ROI materialization contract aligned to the NOD derivatives
- a dataset-side join contract from the fixed NOD slice into the canonical
  shared-only trainer
- a checked-in shared-only train/eval config that points to the adapter,
  target-selection artifact, and target cache

## Fixed dataset-side join contract

The smallest machine-readable downstream join surface for the fixed slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_join_contract
```

Default outputs:

- join contract parquet:
  `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet`
- join contract report:
  `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.report.json`

What it defines:

- primary row id: `pair_id`
- exact join from adapter row -> target-selection rows -> target cache rows
- canonical downstream columns for the fixed slice only
- the neural-side source paths that still need ROI-side materialization

Current meaning:

- join-ready: yes
- ROI-ready: no
- downstream-prep-ready: no
- training-ready: no

## Fixed ROI materialization contract

The smallest ROI-side contract for the same fixed slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_roi_materialization_contract
```

Default outputs:

- ROI contract parquet:
  `cache/indices/public_nod/imagenet_run10_roi_materialization_contract.parquet`
- ROI contract report:
  `cache/indices/public_nod/imagenet_run10_roi_materialization_contract.report.json`

What it defines:

- the exact neural source files expected per `adapter_row_id`
- the verified alignment requirement between:
  - `ciftify` `beta.dscalar.nii`
  - `label.txt`
  - join-contract rows
- the required future ROI output artifact:
  `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- the required future ROI output columns keyed by `pair_id`

Current meaning:

- join-ready: yes
- ROI-ready: no
- downstream-prep-ready: no
- training-ready: no

## Fixed ROI artifact

The smallest real ROI-side artifact for the same fixed slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_roi_artifact
```

Default outputs:

- ROI materialized parquet:
  `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- ROI materialized report:
  `cache/indices/public_nod/imagenet_run10_roi_materialized.report.json`

What it does:

- consumes the fixed ROI materialization contract plus the fixed join contract
- materializes the exact `3600` `pair_id` rows for the fixed slice only
- uses the existing resolved `beta.dscalar.nii` run payloads
- resolves only the subject-universal supporting atlas payloads needed for real
  pooled branch features
- writes `roi_values_json` and `roi_features_json` without widening the NOD
  slice

Current live-pod feature boundary:

- early visual features come from subject-specific `BA_exvivo` labels
- metacognitive features come from subject-specific `aparc` labels
- subject-specific `floc-faces` / `floc-places` masks are not universal across
  `sub-01..sub-09`, so they are excluded from the materialized fixed-slice ROI
  artifact rather than being faked or silently imputed

Current meaning after ROI materialization:

- join-ready: yes
- ROI-ready: yes
- downstream-prep-ready: no
- training-ready: no

## Fixed prepared dataset surface

The smallest dataset-side loader/prepared-dataset surface for the same fixed
slice is:

```bash
./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_prepared_dataset
```

Default outputs:

- prepared dataset parquet:
  `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`
- prepared dataset report:
  `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.report.json`

What it does:

- consumes the fixed join contract
- consumes the real ROI artifact
- validates full `pair_id` alignment against the real target cache
- emits the narrowest canonical prepared dataset parquet for later shared-only
  consumption

Current live-pod result:

- rows: `3600`
- unique `pair_id`: `3600`
- split counts:
  - `train`: `2880`
  - `val`: `360`
  - `test`: `360`
- ROI feature dimensions:
  - `early_visual`: `3`
  - `ventral_visual`: `0`
  - `metacognitive`: `3`

Current meaning after the prepared dataset exists:

- join-ready: yes
- ROI-ready: yes
- downstream-prep-ready: yes
- training-ready: still no

## Fixed trainer-preflight surface

The smallest checked-in trainer-facing config for the same fixed slice is:

- `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`

The narrow trainer-ingestion validation workflow is:

```bash
./.venv/bin/python -m fmri2img.workflows.preflight_public_nod_shared_only_trainer
```

Default output:

- trainer preflight report:
  `outputs/public_nod/train/trainer_preflight.json`

What it proves:

- the fixed prepared dataset loads through the canonical dataset path
- the fixed target cache aligns by `pair_id`
- the fixed ROI artifact aligns by `pair_id`
- train/val/test splits are present
- one real trainer batch can be constructed
- one real canonical forward packet can run without widening the slice

Current meaning after trainer preflight:

- join-ready: yes
- ROI-ready: yes
- downstream-prep-ready: yes
- preflight-ready: yes
- training-ready: still no

## Fixed trainer smoke surface

The smallest checked-in trainer smoke config for the same fixed slice is:

- `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`

Run it through the canonical trainer entrypoint:

```bash
./.venv/bin/python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml
```

Smoke intent:

- one epoch only
- one large train batch for the fixed `2880`-row train split
- one eval batch for each fixed `360`-row validation/test split
- smoke-only output path:
  `outputs/public_nod/train/imagenet_run10_shared_only_smoke/`

Current meaning after a successful smoke:

- trainer-config-ready: yes
- preflight-ready: yes
- smoke-ready: yes
- training-ready: still no

The checked-in smoke summarizer is:

```bash
./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_smoke \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml
```

Default smoke report:

- `outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json`

What it proves:

- the smoke output directory contains the canonical trainer artifacts
- the upstream fixed-slice readiness reports still align
- the smoke run completed operationally
- `training_ready` still remains `false`

## Fixed eval/export smoke surface

The fixed shared-only smoke config also serves as the narrow eval/export smoke
config:

- `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`

Run the canonical eval/export entrypoints against the smoke checkpoint:

```bash
./.venv/bin/python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml \
  --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt
./.venv/bin/python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml \
  --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt
./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml
./.venv/bin/python -m fmri2img.workflows.audit_public_nod_shared_only_downstream_contract \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml
```

Default smoke outputs:

- eval directory:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/`
- export directory:
  `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`
- eval/export smoke report:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- downstream contract audit:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`

What it proves:

- the canonical eval path can consume the fixed-slice smoke checkpoint
- the canonical export path can package the same smoke checkpoint
- both smoke output trees are written under fixed smoke-only namespaces
- the downstream export/report bundle can now be audited mechanically for
  normalized target and condition semantics consistency
- `training_ready` still remains `false`

Current live-pod status:

- export smoke: succeeded
- eval smoke: succeeded
- transfer smoke: succeeded
- downstream contract audit: ready
- normalized condition semantics:
  `present_conditions=["perception"]`,
  `missing_conditions=["imagery"]`,
  `paired_metrics_available=false`
- normalized target metadata:
  `target_name_normalized="vit_l14_image_768"`,
  `target_dimension_normalized=768`,
  `source_field_shape="target_name"`
- machine-readable status:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- machine-readable downstream contract verdict:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`

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
