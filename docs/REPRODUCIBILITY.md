# Reproducibility

## Environment

- Python `>=3.10`
- Install with `pip install -e ".[all]"`
- The canonical shared/private path expects 768-D ViT-L/14 targets
- Runtime device selection is now automatic: if a config requests `cuda` but no GPU is available, canonical workflows fall back to `cpu`

Useful environment variables for real runs:

```bash
export PYTHONPATH=src
export NSD_IMAGERY_ROOT=/abs/path/to/nsd_imagery            # subject-rooted layout, if available
export NSD_IMAGERY_METADATA_ROOT=/abs/path/to/nsd_imagery/metadata
export NSD_IMAGERY_BETA_ROOT=/abs/path/to/nsd_imagery_betas
export NSD_ROI_MASK_ROOT=/abs/path/to/roi_masks_parent
export NSD_HDF5=/abs/path/to/nsd_stimuli.hdf5
```

`NSD_HDF5` is optional if `cache/nsd_hdf5/nsd_stimuli.hdf5` already exists or remote NSD stimulus access is available.

For the live `orchestraiq-jupyter` pod, the canonical imagery prep path now supports the split layout that is actually mounted there:

- shared GLMsingle metadata under `NSD_IMAGERY_METADATA_ROOT`
- subject beta volumes under `NSD_IMAGERY_BETA_ROOT/{subject}/betas_nsdimagery.nii.gz`

`prepare_imagery_index` automatically detects this split layout and writes both a source-layout report and a filtered canonical report per subject.

## External data acquisition

The canonical public imagery acquisition entrypoint is now:

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

This delegates to the repo's official public downloader script and is the
canonical way to refresh the public NSD-Imagery source before running imagery
prep.

Use the acquisition program docs when the goal is to break beyond the current
public-data ceiling:

- `docs/DATA_ACQUISITION_PROGRAM.md`
- `docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`

The canonical pre-rebuild audit for a richer external NSD-style source is now:

```bash
python -m fmri2img.workflows.audit_full_imagery_overlap_external_source_readiness \
  --config configs/canonical/full_imagery_overlap_shared_only.yaml
```

This writes:

- `outputs/canonical/eval/full_imagery_overlap_shared_only/external_source_readiness_audit.json`

The audit does not fabricate data or rerun the ladder. It only checks whether
the canonical external layout is present under `cache/nsd_imagery_external/`
or the configured imagery env roots, whether provenance is explicit, which
subjects are actually covered, and whether the mounted source would exceed the
current `5` total / `1` held-out paired ceiling.

## Smoke Fixture

The checked-in smoke fixture remains the fastest sanity check:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
```

## Practical Animus Core Decoder

The current practical shared-only subsystem is:

- config: `configs/canonical/animus_core_decoder.yaml`

Dedicated wrapper commands:

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.eval_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
python -m fmri2img.workflows.export_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

These wrappers default to the Animus Core config, but still allow `--config`
and `--override` when a controlled variant is needed.

## Official Real Bootstrap Baseline

The official checked-in real-data baseline is:

- config: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- run type: real mixed-condition bootstrap
- subjects: `subj02`, `subj05`, `subj07`
- ROI grouping: atlas-union bootstrap fallback groups
- vividness head: disabled, because this bootstrap dataset has no vividness/confidence labels

This is a real bootstrap baseline, not a paper-scale result.

### Bootstrap Prep Order

1. Prepare imagery indices for each subject:

```bash
for subject in subj02 subj05 subj07; do
  python -m fmri2img.workflows.prepare_imagery_index \
    --config configs/canonical/multisubj_overlap_bootstrap.yaml \
    --override dataset.subject="\"${subject}\""
done
```

This step now rebuilds the subject imagery indices from the canonical raw metadata/beta layout rather than relying on stale cached overlap-era parquet files. The checked-in config filters to the true MVP slice:

- condition: `imagery`
- stimulus set: `B`
- `nsdId` required
- output: `cache/indices/imagery/{subject}.parquet`
- reports:
  - `outputs/canonical/prepared/imagery/{subject}.source_report.json`
  - `outputs/canonical/prepared/imagery/{subject}.report.json`

2. Assemble the overlap-only mixed-condition bootstrap index and materialize ROI features:

```bash
python -m fmri2img.workflows.prepare_overlap_bootstrap \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

3. Build the exact 768-D target cache for the prepared overlap dataset:

```bash
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

4. Preflight the real run:

```bash
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

5. Train:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

6. Evaluate:

```bash
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --checkpoint outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt
```

7. Evaluate transfer:

```bash
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --checkpoint outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt
```

8. Export for Animus:

```bash
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml \
  --checkpoint outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt
```

### Simple Legacy Comparison Baseline

To reproduce the tiny-overlap comparison baseline on the same prepared dataset and target cache:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

This writes the comparison baseline to:

- `outputs/canonical/baselines/multisubj_overlap_ridge_legacy/`

The current scaling status of that comparison is documented in:

- `docs/EXPANDED_OVERLAP_COMPARISON.md`

## Maximum Available Overlap Audit

The repository also includes a checked-in scaling-audit config:

- `configs/canonical/max_available_overlap.yaml`

This config is meant to answer a specific question:

- what is the largest fully canonical overlap dataset currently rebuildable in the mounted environment?

In the live pod where this audit was executed, that checked-in config intentionally resolves to the current fully canonical ceiling:

- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- rows: `94`
- shared paired `nsdId`s: `5`

The broader audit also confirmed that the full public NSD-Imagery source is now
mounted and canonicalized, but only these four subjects contribute usable
overlap on the current benchmark. The remaining subjects are excluded because
their recoverable imagery ids still do not overlap the perception indices.

The official fixed-ladder commands are now:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

The shared-private threshold hypothesis is now formalized as:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

## Artifact Contract

The official real bootstrap path now expects or produces:

- imagery indices at `cache/indices/imagery/{subject}.parquet`
- imagery provenance reports at `outputs/canonical/prepared/imagery/{subject}.source_report.json`
- filtered imagery prep reports at `outputs/canonical/prepared/imagery/{subject}.report.json`
- combined ROI-ready mixed index at `outputs/canonical/prepared/overlap_bootstrap/multisubj_overlap_mixed_with_roi.parquet`
- overlap report at `outputs/canonical/prepared/overlap_bootstrap/report.json`
- overlap stimulus ids at `outputs/canonical/prepared/overlap_bootstrap/overlap_nsd_ids.json`
- target cache at `outputs/targets/overlap_multisubj_vit_l14_image_768.parquet`
- preflight report at `outputs/canonical/prepared/overlap_bootstrap/preflight.json` unless `--output` is provided
- train/eval/transfer/export artifacts under `outputs/canonical/*/multisubj_overlap_bootstrap/`
- comparison baseline artifacts under `outputs/canonical/baselines/multisubj_overlap_ridge_legacy/`

## Readiness Labels

- `smoke_only`: synthetic or fallback-only path
- `bootstrap_ready`: real-data path is runnable, but still too small or provisional for paper-scale claims
- `paper_ready`: real-data path exceeds the configured pair threshold and does not depend on smoke fallbacks
- `blocked`: required artifacts or metadata are missing

The checked-in real bootstrap config sets `preparation.preflight.paper_pair_threshold: 32`, so small overlap runs remain honestly classified as `bootstrap_ready`.

## Single-Subject Canonical Prep

The single-subject config remains useful when you already have one subject's artifacts and want to build a canonical subject-local baseline:

```bash
python -m fmri2img.workflows.prepare_perception_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_mixed_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_roi_features \
  --config configs/canonical/shared_private_bootstrap.yaml
```

## Scientific Honesty Rules

- Do not claim paper-scale performance from the overlap bootstrap baseline.
- Do not enable vividness/confidence supervision without real labels.
- Do not silently mix 512-D and 768-D target caches.
- Do not treat atlas-union bootstrap ROI groups as the final paper-grade ROI decomposition.
- Do not interpret the tiny-overlap Ridge comparison as a final model ranking; it is a low-data sanity baseline.
