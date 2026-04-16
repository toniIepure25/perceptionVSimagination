# Paper 2 Public NOD Animus Lane

This document defines a separate paper-2 research lane for the fixed public
NOD shared-only slice. It is intentionally independent from the current
`full_imagery_overlap_shared_only` benchmark lane and does not rewrite,
repurpose, or weaken the paper-1 evidence freeze.

## 1. Lane identity

- Exact lane name:
  `paper2_public_nod_imagenet_run10_shared_only_animus`
- Canonical config anchor:
  `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
- Why it is separate from `full_imagery_overlap_shared_only`:
  - the full-overlap lane is the current paired benchmark and paper-1 thread
  - this NOD lane is public-data-only and perception-only
  - it therefore cannot honestly stand in for the paired threshold question
- How it helps the Animus subsystem:
  - it reuses the shared-only decoder and Animus export surface on public data
  - it is the cleanest current lane for reliability, confidence, and trust
    questions that matter to downstream Animus consumption
  - it can harden deployment-facing decoder outputs without changing the
    existing paired benchmark story
- Why it is public-data feasible now:
  - the repo already contains a checked-in fixed-slice config and a dedicated
    public-NOD workflow family
  - `docs/NOD_PUBLIC_DATASET.md` documents a real fixed `run-10` public slice
    on the live pod
  - the lane remains entirely inside the existing public NOD integration

## 2. Repo-grounded current state

This section separates three evidence levels and does not merge them:

- Checked-in lane surface in the repo:
  - `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
  - `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
  - `src/fmri2img/workflows/plan_public_nod_animus_paper_lane.py`
  - `src/fmri2img/workflows/report_public_nod_shared_only_smoke.py`
  - `src/fmri2img/workflows/report_public_nod_shared_only_eval_export_smoke.py`
  - `src/fmri2img/workflows/audit_public_nod_shared_only_downstream_contract.py`
- Documented live-pod status inside the repo:
  - `docs/NOD_PUBLIC_DATASET.md`
  - `docs/EXPERIMENT_REGISTRY.md`
  - `docs/PROJECT_MASTER_LOG.md`
- Local workspace status at the time of this phase:
  - `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`

### Fixed public contract from the canonical config

- dataset: `ds004496`
- task: `imagenet`
- fixed run: `10`
- subjects: `sub-01..sub-09`
- sessions: `ses-imagenet01..04`
- adapter rows: `36`
- expected prepared / target / ROI rows: `3600`
- target space: `vit_l14_image_768`
- current experiment role in config:
  - `benchmark_role=practical_animus_preflight_only`
  - `evidence_tier=operational`

### Prepared dataset

- Repo-defined artifact path:
  `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`
- Repo-defined report path:
  `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.report.json`
- Documented live-pod state:
  - `docs/EXPERIMENT_REGISTRY.md` and
    `src/fmri2img/workflows/plan_public_nod_animus_paper_lane.py` describe this
    artifact as a real `3600`-row fixed-slice prepared dataset
  - `docs/NOD_PUBLIC_DATASET.md` documents the upstream resolved `36`-row
    adapter surface that feeds the fixed run-10 slice
- Local workspace state now:
  - the fixed prepared dataset is now present in this workspace
  - this is sufficient for local paper-2 robustness joins against copied
    `per_trial_pairs.csv` artifacts

### Target cache

- Repo-defined artifact path:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- Repo-defined report path:
  `cache/indices/public_nod/imagenet_run10_target_embedding_cache.report.json`
- Documented live-pod state:
  - the planner workflow and registry describe a real fixed `3600`-row,
    `768`-D target cache for the public slice
- Local workspace state now:
  - the file is not present in this workspace

### ROI artifact

- Repo-defined artifact path:
  `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- Repo-defined report path:
  `cache/indices/public_nod/imagenet_run10_roi_materialized.report.json`
- Documented live-pod state:
  - the planner workflow and registry describe a real fixed `3600`-row ROI
    artifact aligned to the shared-only lane
- Local workspace state now:
  - the file is not present in this workspace

### Trainer preflight

- Repo-defined artifact path:
  `outputs/public_nod/train/trainer_preflight.json`
- Documented live-pod state:
  - `docs/EXPERIMENT_REGISTRY.md` and `docs/PROJECT_MASTER_LOG.md` describe the
    lane as `operational_ready=true` and `training_ready=false`
  - the planner workflow encodes the expected trainer-preflight checks
- Local workspace state now:
  - the file is not present in this workspace

### Smoke train / eval / transfer / export

- Repo-defined smoke paths:
  - `outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json`
  - `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
  - `outputs/public_nod/transfer/imagenet_run10_shared_only_smoke/...`
  - `outputs/public_nod/export/imagenet_run10_shared_only_smoke/...`
- Documented live-pod state:
  - the registry describes completed smoke train/eval/transfer/export coverage
    for the fixed public slice
  - the checked-in workflow family validates these surfaces as operational
    smoke-only contracts, not evidence-grade runs
- Local workspace state now:
  - the smoke report and eval/export smoke report are not present
  - local verification of smoke completeness is blocked here

### Downstream contract artifacts

- Repo-defined downstream audit path:
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`
- Local workspace state now:
  - this file exists locally
  - current payload status:
    - `downstream_contract_ready=false`
    - `eval_smoke_ready=false`
    - `transfer_smoke_ready=false`
    - `export_smoke_ready=false`
    - `training_ready=false`
  - current local blocked reason:
    - missing
      `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`
    - missing
      `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- Documented live-pod state:
  - `docs/EXPERIMENT_REGISTRY.md` and `docs/PROJECT_MASTER_LOG.md` record a
    successful paper-lane planning artifact on the live pod with
    `operational_ready=true`, `downstream_contract_ready=true`,
    `evidence_ready_candidate=false`, and `training_ready=false`

### Honest audit verdict

- Canonical lane surface: present
- Documented live-pod operational bundle: present in repo records
- Local workspace materialization: incomplete / blocked
- Publication evidence today: not established
- Repro classification for this workspace:
  `blocked by missing artifacts`

## 3. Honest publication assessment

What is already strong:

- the lane is genuinely separate from the full-overlap paper thread
- the repo already has a coherent fixed public-NOD config and workflow surface
- the documented live-pod history indicates this lane is operationally viable on
  public data
- the lane naturally fits Animus-facing reliability and export questions better
  than paired-threshold claims

What is still only operational:

- the config is still explicitly tagged as
  `practical_animus_preflight_only` / `operational`
- all reported train/eval/transfer/export surfaces are smoke-oriented, not
  publication evidence
- the local workspace does not currently contain the prepared dataset, target
  cache, ROI artifact, or smoke bundle needed to reproduce the stronger live
  operational state

What is missing before this becomes publishable:

- a dedicated non-smoke paper-2 baseline run with separated paper-2 outputs
- a paper-2 config or controlled override surface that is not framed as
  preflight-only
- reliability or uncertainty analyses that are stronger than raw cosine or
  smoke success
- ROI and subject/session robustness analyses
- a confidence-bearing downstream export and audit surface
- reproducible artifact presence for the public-NOD bundle in the active
  execution environment

Smoke artifacts are not publication evidence and should not be presented as
such.

## 4. Novel paper framing

Selected framing:

**Reliability-aware shared-only decoder for public natural-image fMRI with
Animus-facing confidence export**

Why this is the strongest honest direction:

- it is novel relative to a plain “public shared-only baseline works” paper
- it stays inside what the public NOD lane can honestly support
- it is directly useful to Animus because it focuses on when decoder outputs
  should be trusted, exported, or down-weighted
- it does not pretend this perception-only lane answers the paired
  shared-private threshold question

Why this is better than a plain baseline paper:

- a baseline-only story would mostly restate operational integration
- reliability and confidence create a real systems-and-methods contribution
- export-facing trust metadata connects the paper to a concrete subsystem need

## 5. Concrete research questions

1. Can a public-data shared-only decoder expose calibrated confidence signals
   that improve downstream Animus trust decisions over raw cosine alone?
2. Which ROI groups contribute most to stable decoding quality and confidence on
   the fixed public NOD slice?
3. How stable are outputs and confidence estimates across subjects and sessions
   on the fixed run-10 public slice?
4. Can uncertainty-aware scores identify low-trust outputs better than raw
   similarity metrics?
5. Can the current Animus export contract carry reliability metadata without
   breaking the public shared-only downstream surface?

## 6. Exact experiment ladder for this new lane

Minimal serious ladder:

1. Baseline
   - promote the lane from preflight-only framing to a dedicated paper-2
     non-smoke baseline surface
   - keep the same fixed public slice and `vit_l14_image_768` target space
   - write outputs under `outputs/public_nod/paper2/`
2. Baseline controls
   - compare the current shared-only width against one smaller and one larger
     shared bottleneck
   - add one reconstruction-weight control
3. Reliability additions
   - post-hoc cosine-to-risk calibration on validation behavior
   - residual-norm or prediction-dispersion confidence baseline
   - low-trust flagging audited against held-out performance
4. ROI ablations
   - `early_visual` only
   - `metacognitive` only
   - drop-one-group controls
5. Session / subject robustness
   - per-session tables
   - per-subject summaries
   - cross-session confidence drift checks
6. Downstream export / confidence audit
   - add reliability metadata to the manifest or decoder card
   - verify downstream compatibility using an audit surface rather than prose

This ladder is realistic because it reuses the fixed public slice and focuses
on reliability and robustness instead of over-claiming a new decoding regime.

## 7. Exact repo surfaces to use

Configs for this lane:

- `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
- `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- `configs/canonical/public_nod_imagenet_run10_shared_only_paper2_baseline.yaml`

Workflows for this lane:

- `fmri2img.workflows.acquire_public_nod`
- `fmri2img.workflows.inspect_public_nod`
- `fmri2img.workflows.prepare_public_nod_index`
- `fmri2img.workflows.prepare_public_nod_shared_only_adapter`
- `fmri2img.workflows.prepare_public_nod_target_selection`
- `fmri2img.workflows.build_public_nod_target_embedding_cache`
- `fmri2img.workflows.preflight_public_nod_shared_only_trainer`
- `fmri2img.workflows.train_decoder`
- `fmri2img.workflows.eval_decoder`
- `fmri2img.workflows.eval_transfer`
- `fmri2img.workflows.export_for_animus`
- `fmri2img.workflows.report_public_nod_shared_only_smoke`
- `fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke`
- `fmri2img.workflows.audit_public_nod_shared_only_downstream_contract`
- `fmri2img.workflows.plan_public_nod_animus_paper_lane`
- `fmri2img.workflows.report_public_nod_paper2_baseline`
- `fmri2img.workflows.compare_public_nod_paper2_runs`
- `fmri2img.workflows.diagnose_public_nod_paper2_condition_semantics`
- `fmri2img.workflows.compare_public_nod_paper2_roi_runs`
- `fmri2img.workflows.analyze_public_nod_paper2_robustness`
- `fmri2img.workflows.compare_public_nod_paper2_robustness`
- `fmri2img.workflows.analyze_public_nod_paper2_trust_signals`
- `fmri2img.workflows.compare_public_nod_paper2_trust_signals`
- `fmri2img.workflows.analyze_public_nod_paper2_risk_buckets`
- `fmri2img.workflows.compare_public_nod_paper2_risk_buckets`

Outputs that belong to this lane:

- `cache/indices/public_nod/...`
- `outputs/public_nod/train/...`
- `outputs/public_nod/eval/...`
- `outputs/public_nod/transfer/...`
- `outputs/public_nod/export/...`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/paper_lane_plan.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/paper2_baseline_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/reliability_seed_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim32/paper2_ablation_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim32/reliability_seed_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim128/condition_semantics_diagnosis.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim128/paper2_ablation_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/shared_dim128/reliability_seed_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/early_visual_only/paper2_ablation_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/early_visual_only/reliability_seed_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/metacognitive_only/paper2_ablation_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/metacognitive_only/reliability_seed_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/robustness_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/early_visual_only/robustness_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/metacognitive_only/robustness_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/trust_signal_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/early_visual_only/trust_signal_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/metacognitive_only/trust_signal_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/baseline/risk_bucket_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/early_visual_only/risk_bucket_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/metacognitive_only/risk_bucket_report.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/shared_capacity_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/reliability_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/roi_ablation_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/roi_reliability_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/robustness_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/robustness_tail_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/trust_signal_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/trust_instability_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/risk_bucket_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/risk_monotonicity_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/coarse_risk_comparison.json`
- `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/coarse_risk_monotonicity_comparison.json`

Docs that belong to this lane:

- `docs/PAPER_2_PUBLIC_NOD_ANIMUS_LANE.md`
- `docs/NOD_PUBLIC_DATASET.md`

Do not reuse the durable paper framing of `full_imagery_overlap_shared_only`
for this lane.

## 8. Immediate executable next step

The paper-2 lane now also has a real exploratory trust-signal pack:

- `baseline`, `early_visual_only`, and `metacognitive_only` each have a
  machine-readable `trust_signal_report.json`
- all three runs still join cleanly to the fixed prepared dataset on `pair_id`
- bottom-10% low-score samples are enriched in the current low-performing
  subject/session groups for all three runs, roughly `1.45x` to `1.51x` above
  the base unstable-group share
- `metacognitive_only` remains the strongest average ROI-only run, the most
  stable by subject dispersion, and the cleanest trust-tail run by the current
  bottom-10% threshold
- `baseline` ties the strongest bottom-10 instability enrichment, while
  `early_visual_only` remains the most even by session dispersion and shows the
  strongest bottom-5 instability enrichment
- this materially improves publishability because the lane now has an honest
  argument that low-score tails carry trust information beyond mean cosine
  alone, but it still does not create publication-grade evidence

The next dedicated execution step should therefore move from exploratory
tail-based trust signals into a more explicit risk analysis:

```bash
./.venv/bin/python -m fmri2img.workflows.compare_public_nod_paper2_trust_signals
```

The paper-2 lane now also has a real exploratory risk-bucket pack:

- `baseline`, `early_visual_only`, and `metacognitive_only` each now have a
  machine-readable `risk_bucket_report.json`
- all three risk reports still join cleanly to the fixed prepared dataset on
  `pair_id`
- each run shows a real lowest-bucket versus highest-bucket instability gap of
  roughly `0.31` to `0.33`, so the low-score buckets do carry more unstable
  examples than the top-score buckets
- none of the three runs satisfies a clean monotonic bucket-by-bucket risk
  rise as cosine falls under the current heuristic, so this pack does not yet
  justify a stronger graded-risk claim beyond the existing tail signals
- `metacognitive_only` remains the strongest overall paper-2 run by eval
  cosine, transfer cosine, subject-level stability, and bottom-10 trust-tail
  cleanliness
- `early_visual_only` remains the strongest by session stability and is the
  cleanest current run on the bucket-level cosine-versus-instability Spearman
  diagnostic
- this materially improves publishability because the lane now has an honest
  negative/qualified result on score-based risk stratification instead of only
  positive tail-enrichment summaries, but it still does not create
  publication-grade evidence

The next dedicated execution step should therefore move from bucketed risk
stratification into a narrower calibration-style table that tests whether
instability rates rise monotonically across coarser cosine bins or across
subject/session-defined risk groups:

```bash
./.venv/bin/python -m fmri2img.workflows.compare_public_nod_paper2_risk_buckets
```

The paper-2 lane now also has a real exploratory coarse-bin risk follow-up:

- the refreshed `risk_bucket_report.json` files for `baseline`,
  `early_visual_only`, and `metacognitive_only` now include explicit
  `deciles`, `tertiles`, `quintiles`, and low-performing-group-conditioned
  tables
- the lane now also has
  `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/coarse_risk_comparison.json`
  and
  `outputs/public_nod/paper2/imagenet_run10_shared_only/comparison/coarse_risk_monotonicity_comparison.json`
- all three updated reports still join cleanly to the fixed prepared dataset
  on `pair_id`
- tertiles strengthen the exploratory risk signal over deciles: `baseline`
  becomes monotonic by unstable-group share with tertiles, with a tertile
  cosine-versus-instability Spearman of `-1.0`
- quintiles also strengthen the correlation view over deciles for all three
  runs, but they still do not produce a clean monotonic flag in the current
  pack
- `metacognitive_only` remains the strongest overall paper-2 run by eval
  cosine, transfer cosine, subject-level stability, and bottom-10 trust-tail
  cleanliness
- `early_visual_only` remains the strongest by session stability, while
  `baseline` is currently the cleanest coarse-bin risk run because the tertile
  view is the only one that flips to a clean monotonic signal
- the low-performing-group-conditioned tables mainly confirm that the signal is
  concentrated in the globally defined low-performing groups rather than
  revealing a new within-group graded-risk story
- this materially improves publishability because the lane now has a more
  precise exploratory trust-risk story with a coarser-bin sensitivity check,
  but it still does not create publication-grade evidence

The next dedicated execution step should therefore test whether the coarse-bin
result survives a more explicit low-performing-group-conditioned risk table or
whether it collapses once the subgroup definition is tightened:

```bash
./.venv/bin/python -m fmri2img.workflows.analyze_public_nod_paper2_risk_buckets --run-root outputs/public_nod/paper2/imagenet_run10_shared_only/baseline
```

## 9. Validation

Focused validation for the baseline, comparison, robustness, and trust phase:

- run the focused canonical workflow tests for the paper-2 planner and baseline
  report surfaces plus the comparison, robustness, and trust workflow surfaces
- run pod preflight before each non-smoke ablation command that requires new
  execution
- keep `training_ready=false` and `evidence_ready_candidate=false` unless a
  later lane-specific gate is explicitly added and truly satisfied

Current comparative status from the current controlled packs:

- baseline remains the best current run by eval and transfer cosine on the
  shared-capacity pack
- `shared_dim32` has the cleanest bottom-decile cosine tail among the three
  shared-capacity runs
- `shared_dim128` now has an artifact-backed diagnosis showing the original
  export bundle was malformed rather than a full lane-level eval/transfer
  mismatch, and its refreshed export surface is now operationally consistent
- `early_visual_only` and `metacognitive_only` both retain real signal on the
  fixed public slice
- `metacognitive_only` is currently the strongest ROI-only view by eval cosine,
  transfer cosine, and the current bottom-decile low-trust heuristic
- the new bucketed risk pack shows real lowest-vs-highest bucket instability
  gaps for all three runs, but not a fully monotonic bucketed risk rise, so it
  supports exploratory risk stratification without establishing a stronger
  graded-risk claim beyond the existing trust-tail heuristics
- the new coarse-bin follow-up strengthens the exploratory risk story because
  tertiles recover a clean monotonic instability rise for `baseline` and
  quintiles improve the rank-correlation view across the three runs, but the
  resulting signal is still fragile and subgroup-conditioned summaries remain
  largely tautological rather than evidence-grade
