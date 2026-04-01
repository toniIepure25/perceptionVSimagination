# Project Master Log

## 1. Project mission

Build a professional, reproducible platform for decoding and disentangling
visual perception, mental imagery, and subjective experience from fMRI.

Immediate MVP paper direction:

- shared/private latent disentanglement for perception vs imagery
- 768-D ViT-L/14 content decoding
- ROI-aware analysis
- optional vividness/confidence prediction when real labels exist

Long-term Animus role:

- canonical fMRI preprocessing and ROI materialization backend
- exportable decoder checkpoints and metadata
- trustworthy perception-to-imagery transfer substrate
- future subjective-state and reality-monitoring extensions

## 2. Canonical architecture summary

Official model path:

- ROI-specific hierarchical encoder
- shared/private latent layer
- multitask heads

Official dataset/prep path:

- canonical perception/imagery indices
- canonical target cache: `vit_l14_image_768`
- canonical overlap assembly
- canonical ROI materialization into `roi_features_json` / `roi_values_json`

Official evaluation path:

- `train_decoder`
- `eval_decoder`
- `eval_transfer`
- `run_analysis`
- `export_for_animus`
- `run_legacy_ridge_baseline`

Multi-subject rule:

- canonical training is ROI-first
- raw full-fMRI vectors are optional auxiliary context
- incompatible raw voxel shapes are dropped at collate time instead of blocking
  ROI-materialized multi-subject runs

## 3. What has been implemented so far

Key completed milestones:

- canonical package/workflow refactor
- canonical shared/private decoder contract
- canonical ROI grouping and ROI materialization path
- canonical imagery/perception prep surface
- canonical preflight readiness workflow
- first real mixed-condition bootstrap run
- full public NSD-Imagery acquisition and canonical rebuild
- multi-subject ROI-first batching fix for unequal raw voxel dimensionality
- canonical Ridge comparison workflow
- canonical shared-only ablation mode

Important fixes:

- device-safe eval/transfer/export
- path-stable checked-in configs
- real-data preflight validation
- truthful training-history cosine logging
- split-layout imagery prep support
- expanded overlap training no longer blocked by raw shape mismatch

## 4. Experiments already run

### E0. Canonical smoke

- config: `configs/canonical/shared_private_smoke.yaml`
- dataset: checked-in fake/smoke fixture
- purpose: workflow validation only
- outcome: full train/eval/export surface works

### E1. Real bootstrap baseline

- config: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- dataset: `subj02/subj05/subj07`, `4` shared ids
- canonical result:
  - cosine `0.05325`
  - MSE `0.00247`
  - domain accuracy `0.21053`
- interpretation:
  - real pipeline success
  - still too small for scientific conclusions

### E2. Tiny Ridge comparison

- config: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- dataset: same `4`-id bootstrap overlap
- Ridge result:
  - cosine `0.42202`
  - MSE `0.001505`
- interpretation:
  - simple linear baseline strongly dominates in the tiny-data regime

### E3. Full imagery expansion audit

- config/report basis:
  - `configs/canonical/max_available_overlap.yaml`
  - `outputs/canonical/prepared/full_imagery_overlap/report.json`
- data change:
  - full public NSD-Imagery acquired for `subj01..subj08`
- outcome:
  - usable overlap expands from `4` to `5` shared ids
  - included subjects: `subj02/subj03/subj05/subj07`
  - combined rows: `94`
- interpretation:
  - data ceiling moved slightly
  - full public imagery still yields only `5` canonical overlapable ids

### E4. Expanded canonical shared-private baseline

- config basis: `configs/canonical/max_available_overlap.yaml`
- dataset: `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- checkpoint:
  - `outputs/canonical/train/full_imagery_overlap/best_decoder.pt`
- metrics:
  - val cosine `0.06319`
  - test cosine `0.06927`
  - test MSE `0.002424`
  - imagery mean `0.07151`
  - perception mean `0.05735`
  - domain accuracy `0.89474`
- interpretation:
  - canonical multi-subject path is operational
  - positive signal exists
  - still far behind Ridge

### E5. Expanded Ridge baseline refresh

- config basis: `configs/canonical/max_available_overlap.yaml`
- model:
  - `legacy_ridge_on_roi_values`
- artifacts:
  - `outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/metrics.json`
- metrics:
  - val cosine `0.47876`
  - test cosine `0.55199`
  - test MSE `0.001167`
- interpretation:
  - Ridge remains the strongest current baseline by a large margin

### E6. Shared-only ablation

- config basis: `configs/canonical/max_available_overlap.yaml`
- overrides:
  - `training.device="cpu"`
  - `model.disentanglement_mode="shared_only"`
  - `model.use_domain_head=false`
- checkpoint:
  - `outputs/canonical/train/full_imagery_overlap_shared_only/best_decoder.pt`
- metrics:
  - val cosine `0.07430`
  - test cosine `0.13596`
  - test MSE `0.002250`
  - imagery mean `0.13422`
  - perception mean `0.14527`
- interpretation:
  - shared-only clearly outperforms full shared-private on the current tiny
    overlap dataset
  - private/disentanglement structure is likely the larger drag right now

### E7. Shared-private with domain head disabled

- config basis: `configs/canonical/max_available_overlap.yaml`
- overrides:
  - `training.device="cpu"`
  - `model.use_domain_head=false`
- checkpoint:
  - `outputs/canonical/train/full_imagery_overlap_nodomain/best_decoder.pt`
- metrics:
  - val cosine `0.04813`
  - test cosine `0.05907`
  - test MSE `0.002450`
- interpretation:
  - disabling the domain head alone does not help
  - the main small-data drag is not just the domain auxiliary

### E8. Shared-private private-dimension sweep

Dataset held fixed:

- `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- `5` shared `nsdId` groups
- `94` rows

#### E8a. Reduced private capacity, `private_dim=16`

- checkpoint:
  - `outputs/canonical/train/full_imagery_overlap_priv16/best_decoder.pt`
- metrics:
  - val cosine `0.04995`
  - test cosine `0.10784`
  - test MSE `0.002323`
  - domain accuracy `0.26316`
- interpretation:
  - materially better than the original shared-private model
  - still clearly below shared-only

#### E8b. Reduced private capacity, `private_dim=8`

- checkpoint:
  - `outputs/canonical/train/full_imagery_overlap_priv8/best_decoder.pt`
- metrics:
  - val cosine `0.04076`
  - test cosine `0.09595`
  - test MSE `0.002354`
  - domain accuracy `0.89474`
- interpretation:
  - better than the original shared-private model
  - slightly worse than `private_dim=16`
  - still below shared-only

## 5. Baseline comparison summary

Current ranking on the expanded `5`-id overlap set:

1. Ridge
   - cosine `0.55199`
   - MSE `0.001167`
2. Canonical shared-only
   - cosine `0.13596`
   - MSE `0.002250`
3. Canonical shared-private, `private_dim=16`
   - cosine `0.10784`
   - MSE `0.002323`
4. Canonical shared-private, `private_dim=8`
   - cosine `0.09595`
   - MSE `0.002354`
5. Canonical shared-private
   - cosine `0.06927`
   - MSE `0.002424`
6. Canonical shared-private no-domain
   - cosine `0.05907`
   - MSE `0.002450`

What this means:

- the canonical architecture is operational
- shared-only is materially better than shared-private in the current tiny-data
  regime
- reducing private capacity helps shared-private somewhat
- `private_dim=16` is the best recovery variant tested so far
- Ridge still dominates so strongly that the paper hypothesis is not yet fairly
  testable on performance

What cannot yet be concluded:

- whether disentanglement helps once overlap scale is materially larger
- whether the shared-private model is fundamentally worse, or just too data-hungry
- whether better ROI decomposition would change the ordering

## 6. Current blockers

Data blockers:

- only `5` overlapable canonical imagery ids are currently available from the
  full public imagery source
- held-out paired evaluation remains only `1` usable pair group

Environment blockers:

- GPU availability can fluctuate on the live pod
- some pod-side runtimes still warn about optional `nibabel` absence, though the
  canonical ROI-materialized path remains usable

Modeling blockers:

- the full shared-private objective is too heavy for the current tiny overlap set
- disentanglement/private branches are likely over-parameterized for this regime

Operational blockers:

- no larger overlap-capable dataset is currently mounted or recoverable locally

## 7. Decisions made

- keep the benchmark fixed:
  - same expanded overlap dataset
  - same 768-D target space
  - same evaluation surface
- do not redesign the model yet
- do not move to new scientific tasks before resolving the data ceiling
- use CPU-safe ablation runs when the live GPU is already busy
- treat the canonical ROI-first multi-subject path as the official model-input contract

Current belief:

- the main bottleneck is still data scale
- on the present tiny dataset, shared-only is still the strongest canonical neural baseline
- if shared-private is revisited before more data arrives, smaller private capacity is the only justified direction so far

## 8. Next recommended actions

Immediate next step:

- acquire or mount a materially larger overlapable perception/imagery dataset
  while keeping the same fixed comparison:
  - Ridge
  - canonical shared-only
  - canonical shared-private

Fallback next step if data cannot move soon:

- consider a very small follow-up around `private_dim≈16` or similarly reduced
  private capacity, not a broad architecture redesign

What should wait:

- vividness expansion
- stimulus-vs-percept work
- SDXL/reconstruction expansion
- large ROI redesign

## 9. Open questions

Scientific:

- at what overlap scale, if any, does shared-private begin to justify itself?
- is the current boost from shared-only a pure capacity effect or a sign that
  private variance is genuinely unlearnable at this scale?

Engineering:

- should the next modeling pass be smaller private dimensions, weaker
  orthogonality, or fewer active losses?
- does `private_dim=16` already capture most of the recoverable gain in this regime?
- should a checked-in ablation config set be added for the shared-only and
  no-domain controls?

Risk:

- without more overlap data, it is easy to overfit engineering effort to a
  scientifically underpowered regime

## 10. Change log / session log

- 2026-04-01: canonical refactor and hardening completed; official workflow
  surface established
- 2026-04-01: first real bootstrap run completed on the live pod
- 2026-04-01: canonical imagery prep repaired for the split pod layout
- 2026-04-01: first trustworthy canonical vs Ridge baseline established on the
  `4`-id overlap set
- 2026-04-02: full public NSD-Imagery acquired; overlap expanded to `5` ids
- 2026-04-02: multi-subject ROI-first batching fix landed and enabled the
  expanded canonical rerun
- 2026-04-02: expanded canonical shared-private baseline rerun completed on the
  `5`-id overlap set
- 2026-04-02: shared-only and no-domain ablations completed; shared-only became
  the strongest canonical variant on the current dataset
- 2026-04-02: narrow private-capacity sweep completed; smaller private latents
  improved shared-private but did not beat shared-only
