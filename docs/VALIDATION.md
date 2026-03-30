# Validation

This document records the post-refactor validation and hardening pass for the
canonical shared/private decoder platform.

## What was validated end-to-end

The checked-in smoke fixture now runs through the full official workflow surface:

```bash
python -m fmri2img.workflows.train_decoder --config configs/canonical/shared_private_smoke.yaml
python -m fmri2img.workflows.eval_decoder --config configs/canonical/shared_private_smoke.yaml --checkpoint outputs/canonical/hardening_smoke/train/best_decoder.pt
python -m fmri2img.workflows.eval_transfer --config configs/canonical/shared_private_smoke.yaml --checkpoint outputs/canonical/hardening_smoke/train/best_decoder.pt
python -m fmri2img.workflows.run_analysis --config configs/canonical/shared_private_smoke.yaml --checkpoint outputs/canonical/hardening_smoke/train/best_decoder.pt
python -m fmri2img.workflows.export_for_animus --config configs/canonical/shared_private_smoke.yaml --checkpoint outputs/canonical/hardening_smoke/train/best_decoder.pt
```

Validated outputs were created under `outputs/canonical/hardening_smoke/`.

## What is now validated for real-data readiness

The canonical platform now has a first-class preparation surface for real bootstrap runs:

```bash
python -m fmri2img.workflows.prepare_perception_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_imagery_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_mixed_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_roi_features --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.preflight_data --config configs/canonical/shared_private_bootstrap.yaml
```

The checked-in official real bootstrap baseline also has a canonical overlap assembly path:

```bash
for subject in subj02 subj05 subj07; do
  python -m fmri2img.workflows.prepare_imagery_index \
    --config configs/canonical/multisubj_overlap_bootstrap.yaml \
    --override dataset.subject="\"${subject}\""
done
python -m fmri2img.workflows.prepare_overlap_bootstrap --config configs/canonical/multisubj_overlap_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets --config configs/canonical/multisubj_overlap_bootstrap.yaml
python -m fmri2img.workflows.preflight_data --config configs/canonical/multisubj_overlap_bootstrap.yaml
```

`prepare_imagery_index` now supports both:

- a subject-rooted raw imagery layout
- the split pod layout with shared GLMsingle metadata plus per-subject beta volumes

For the split layout, canonical prep produces:

- `cache/indices/imagery/{subject}.parquet`
- `outputs/canonical/prepared/imagery/{subject}.source_report.json`
- `outputs/canonical/prepared/imagery/{subject}.report.json`

This path is now covered by an end-to-end volumetric fixture test that:

- starts from a raw perception index
- builds an imagery index from raw imagery metadata + NIfTI files
- rebuilds imagery from the split metadata/beta layout that mirrors the live pod
- canonicalizes a 768-D target cache
- builds a mixed perception/imagery index
- materializes ROI features from real ROI masks
- preflights the run to `bootstrap_ready`
- trains, evaluates, and exports through the canonical workflow surface

## Test coverage added in the hardening pass

The following failure-oriented cases are now covered:

- checked-in canonical smoke config path resolution
- missing canonical artifacts in the MVP config
- malformed or missing `nsdId`
- ROI alias mismatch behavior
- pair sampler failure when no true perception/imagery pairs exist
- partial vividness label masking
- target-dimension mismatch
- checkpoint/model compatibility mismatch
- export manifest validation
- mixed-index split collisions across perception and imagery sources
- blocked vs bootstrap-ready preflight classification
- paper-ready threshold enforcement so tiny overlap runs remain classified as `bootstrap_ready`
- canonical ROI materialization from volumetric data + ROI masks
- non-canonical target-cache rejection during `prepare_targets`
- canonical overlap-bootstrap assembly across multiple subjects
- canonical imagery-index rebuild from split metadata/beta layout
- inference-device fallback and checkpoint/device alignment during eval/transfer/analysis
- truthful `train_history.json` validation cosine logging

## Real repo context validation

The local repository contains a real perception index at:

- `data/indices/nsd_index/subject=subj01/index.parquet`

That index was statically validated against the canonical normalizer:

- `nsdId` exists and is usable
- deterministic train/val/test splits can be inferred
- true cross-condition pairing is absent in the perception-only index, so transfer metrics cannot be meaningfully computed from it alone

## What could not be fully validated locally

The canonical MVP config still requires artifacts that are not present locally:

- an imagery index or mixed-condition index
- a real 768-D target cache at `outputs/targets/vit_l14_image_768.parquet`
- a reachable ROI mask root for the real subject

The local perception index points to remote `beta_path` NIfTI files on S3. Since
the workspace does not include materialized voxel arrays for those samples, the
canonical real-data path was validated statically rather than by full training on
actual subject data.

## Current trust boundary

Trusted now:

- the checked-in canonical smoke workflow
- the canonical artifact-prep workflow surface
- the canonical config loader and path semantics
- canonical target cache validation
- canonical ROI materialization from real masks into `roi_features_json`
- pair-aware batching logic
- checkpoint reload compatibility checks
- Animus export manifest validation

Not yet fully validated end-to-end on real research artifacts:

- real mixed perception/imagery training on local subject voxel data
- vividness/confidence learning on a real labeled dataset
- the target-cache build path against real NSD image access in this workspace

## Live bootstrap run status

The first real mixed-condition bootstrap run was executed on the live `orchestraiq-jupyter` pod and is now represented by the checked-in config:

- `configs/canonical/multisubj_overlap_bootstrap.yaml`

That live run used:

- subjects `subj02`, `subj05`, `subj07`
- 4 shared `nsdId` pairs
- atlas-union bootstrap ROI groups
- a real 768-D ViT-L/14 target cache
- canonical train/eval/transfer/export workflows

The eval-device workaround required during that first live run is no longer necessary; canonical eval, transfer, and analysis workflows now resolve runtime devices explicitly and move the model onto the chosen device before inference.

The remaining reproducibility gap from that live run has also been closed in code: canonical imagery prep no longer depends on stale cached parquet files that were missing `nsdId`, and can rebuild the official overlap bootstrap inputs directly from the pod-style imagery metadata/beta layout.
