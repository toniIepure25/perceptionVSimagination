# Paper 2 Public NOD Animus Lane

This document defines a separate paper-2 research lane for the fixed public
NOD shared-only slice. It is intentionally independent from the current
`full_imagery_overlap_shared_only` benchmark lane and does not change the
paper-1 evidence freeze.

## 1. Lane identity

- Exact lane name:
  `paper2_public_nod_imagenet_run10_shared_only_animus`
- Canonical config anchor:
  `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
- Why it is separate from `full_imagery_overlap_shared_only`:
  - the full-overlap lane is the current paired benchmark and threshold paper
    thread
  - this NOD lane is public-data-only and perception-only
  - it is therefore better aligned with Animus-facing reliability and
    deployment questions than with the paired threshold hypothesis
- How it helps the Animus subsystem:
  - it already exercises the shared-only decoder path on public data
  - it already has export and downstream contract surfaces
  - it is the best immediate public lane for studying reliability, trust, and
    confidence-bearing export without waiting for richer paired data
- Why it is public-data feasible now:
  - the fixed NOD slice is already prepared and target-cached
  - ROI materialization, trainer preflight, smoke train/eval/transfer/export,
    and downstream contract audits already exist on the live pod
  - no private or external paired source is required to start the lane-design
    work

## 2. Repo-grounded current state

The current lane state is grounded in:

- `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
- `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.report.json`
- `cache/indices/public_nod/imagenet_run10_target_embedding_cache.report.json`
- `cache/indices/public_nod/imagenet_run10_roi_materialized.report.json`
- `outputs/public_nod/train/trainer_preflight.json`
- `outputs/public_nod/train/imagenet_run10_shared_only_preflight/preflight_data.json`
- `outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json`
- `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`

Current fixed public contract:

- dataset: `ds004496`
- task: `imagenet`
- fixed run: `10`
- subjects: `sub-01..sub-09`
- sessions: `ses-imagenet01..04`
- adapter rows: `36`
- expected prepared / target / ROI rows: `3600`
- target space: `vit_l14_image_768`

Current live operational state already achieved:

- prepared dataset: real fixed slice, `3600` rows, split counts
  `train=2880`, `val=360`, `test=360`
- target cache: real `3600`-row cache with normalized `768`-D target metadata
- ROI artifact: real `3600`-row ROI materialization with current feature dims
  `early_visual=3`, `ventral_visual=0`, `metacognitive=3`
- trainer preflight:
  - `join_ready=true`
  - `roi_ready=true`
  - `downstream_prep_ready=true`
  - `trainer_config_ready=true`
  - `preflight_ready=true`
  - `training_ready=false`
- smoke train:
  - `smoke_ready=true`
  - one completed epoch on the fixed slice
  - still explicitly operational-only
- smoke eval / transfer / export:
  - `eval_smoke_ready=true`
  - `transfer_smoke_ready=true`
  - `export_smoke_ready=true`
- downstream contract audit:
  - `downstream_contract_ready=true`
  - normalized target metadata is cross-consistent
  - normalized condition semantics are cross-consistent
  - experiment name and benchmark role are stable across smoke export surfaces

Current condition semantics on this lane:

- `present_conditions=["perception"]`
- `missing_conditions=["imagery"]`
- `paired_metrics_available=false`
- paired metrics remain unavailable by design because this lane is
  perception-only

## 3. Honest publication assessment

What is already strong:

- the public NOD slice is reproducibly integrated into the canonical shared-only
  path
- the lane already has a real fixed-slice prep contract, target cache, ROI
  artifact, trainer preflight, smoke train/eval/transfer/export, and
  downstream-contract verdict
- the downstream export surface is already normalized enough to support a
  reliability-oriented Animus paper direction

What is still only operational:

- the current train/eval/transfer/export evidence is smoke-only
- the checked-in config still carries an operational/preflight benchmark role,
  not a paper-ready experimental role
- the current artifacts prove pipeline stability, not scientific performance
- raw smoke metrics should not be presented as publication evidence

What is missing before this becomes publishable:

- a dedicated non-smoke paper-2 baseline run with separated paper-2 outputs
- reliability or uncertainty evaluation that is stronger than raw similarity
  summaries
- ROI-group and session/subject robustness analyses
- a confidence-bearing export or audit surface that stays useful for Animus
  consumers

## 4. Novel paper framing

Selected framing:

**Reliability-aware shared-only decoder for public natural-image fMRI with
Animus-facing confidence export**

Why this is the strongest honest framing:

- a plain public shared-only baseline paper would mostly repeat the operational
  smoke story without adding a distinct scientific contribution
- the NOD lane is perception-only, so it cannot honestly stand in for the
  paired threshold paper
- the repo already has strong export and downstream contract surfaces, which
  makes reliability, confidence, and trust the most natural novel extension
- this framing is directly useful for the Animus subsystem because it asks which
  public shared-only outputs are safe to export or trust, not merely whether a
  decoder can run

Why it is better than a plain baseline paper:

- it introduces a deployment-relevant question
- it uses the existing export contract as part of the scientific story
- it stays public-data-feasible without pretending the lane answers the paired
  threshold question

## 5. Concrete research questions

1. Can a public-data shared-only decoder expose calibrated confidence signals
   that improve downstream Animus trust decisions over raw cosine alone?
2. Which ROI groups contribute most to stable decoding quality and confidence on
   the fixed public NOD slice?
3. How stable are outputs and confidence estimates across subjects and sessions
   on the fixed run-10 public slice?
4. Can uncertainty-aware scores identify low-trust outputs better than
   uncalibrated similarity metrics?
5. Can the current Animus export contract carry reliability metadata without
   breaking the existing public shared-only downstream surface?

## 6. Exact experiment ladder

Minimal serious ladder for this lane:

1. Baseline
   - create the first dedicated non-smoke paper-2 shared-only baseline on the
     fixed NOD slice
   - write train/eval/transfer/export outputs under a separate paper-2 output
     root
2. Shared-capacity ablations
   - shared-dimension down/up controls
   - reconstruction-weight control
3. Reliability additions
   - post-hoc confidence calibration from held-out validation behavior
   - compare raw cosine against calibrated risk / low-trust flags
4. ROI ablations
   - early-visual only
   - metacognitive only
   - drop-one-group controls
5. Session / subject robustness
   - per-session stability tables
   - per-subject stability summaries
   - cross-session confidence drift checks
6. Downstream export / confidence audit
   - confidence-bearing manifest metadata
   - decoder-card reliability summary
   - export-side audit confirming downstream compatibility

This ladder is realistic because it stays inside the existing public NOD slice,
reuses the current shared-only workflow family, and focuses on reliability and
robustness rather than paired-threshold claims.

## 7. Exact repo surfaces to use

Current reused configs:

- `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
- `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`

Current reused workflows:

- `fmri2img.workflows.preflight_public_nod_shared_only_trainer`
- `fmri2img.workflows.train_decoder`
- `fmri2img.workflows.eval_decoder`
- `fmri2img.workflows.eval_transfer`
- `fmri2img.workflows.export_for_animus`
- `fmri2img.workflows.report_public_nod_shared_only_smoke`
- `fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke`
- `fmri2img.workflows.audit_public_nod_shared_only_downstream_contract`

New paper-2 support workflow:

- `fmri2img.workflows.plan_public_nod_animus_paper_lane`

Current outputs that belong to this lane:

- `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`
- `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- `outputs/public_nod/train/trainer_preflight.json`
- `outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json`
- `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`

Paper-2 planning output:

- `outputs/public_nod/paper2/imagenet_run10_shared_only/paper_lane_plan.json`

Lane-specific docs:

- `docs/PAPER_2_PUBLIC_NOD_ANIMUS_LANE.md`
- `docs/NOD_PUBLIC_DATASET.md`
- `docs/ANIMUS_CORE_DECODER.md`

## 8. Immediate executable next step

The first safe step for this new lane is the lane-specific paper-support report:

```bash
./.venv/bin/python -m fmri2img.workflows.plan_public_nod_animus_paper_lane \
  --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml
```

Why this is the first step:

- it materializes a separate paper-2 planning artifact
- it freezes the new lane identity and experiment ladder without modifying the
  full-overlap paper thread
- it keeps current smoke artifacts in their correct operational-only role

## 9. Validation

This phase should validate:

- the new workflow imports and runs from the project `.venv`
- focused workflow tests pass
- if safe, the new planning artifact is written on the live pod under the new
  paper-2 output root

Current claim boundary:

- this document does not promote the public NOD smoke outputs to benchmark
  evidence
- this document does not change paper-1 claims
- this document does not claim production Animus readiness
