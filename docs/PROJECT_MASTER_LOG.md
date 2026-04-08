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
- dedicated Animus Core Decoder workflow surface
- professional README/onboarding surface aligned to the evidence freeze

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

Evidence freeze roles:

- Ridge is the external low-data reference baseline.
- Shared-only is the best current canonical neural baseline.
- Shared-private is the main hypothesis family, not the current performance
  leader.
- Shared-private with `private_dim=16` is the best current exploratory
  disentanglement variant.
- Shared-only should now be treated as the practical Animus Core Decoder path.

What cannot yet be concluded:

- whether disentanglement helps once overlap scale is materially larger
- whether the shared-private model is fundamentally worse, or just too data-hungry
- whether better ROI decomposition would change the ordering

Current paper-grade hypotheses:

- low-overlap regimes favor Ridge and shared-only over explicit disentanglement
- shared-private may only become beneficial after a threshold of paired overlap
- ROI-first canonicalization is necessary for valid multi-subject
  perception/imagery decoding
- private-capacity scaling likely governs whether disentanglement helps or hurts
- the project's central novelty is now a threshold hypothesis, not an assumed
  disentanglement win

## 6. Current blockers

Data blockers:

- only `5` overlapable canonical imagery ids are currently available from the
  full public imagery source
- held-out paired evaluation remains only `1` usable pair group
- no richer paired source is currently accessible from the mounted environment

Environment blockers:

- GPU availability can fluctuate on the live pod
- some pod-side runtimes still warn about optional `nibabel` absence, though the
  canonical ROI-materialized path remains usable

Modeling blockers:

- the full shared-private objective is too heavy for the current tiny overlap set
- disentanglement/private branches are likely over-parameterized for this regime

Operational blockers:

- no larger overlap-capable dataset is currently mounted or recoverable locally
- the next decisive progress now depends on external paired-data acquisition or
  mount integration

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
- the repo should now communicate three evidence levels explicitly:
  - external reference baseline
  - best current neural baseline
  - exploratory hypothesis family
- the practical Animus path should now run through shared-only
- the research path should now be framed as threshold testing:
  when does shared-private overtake shared-only, if ever?
- the repo now has a dedicated shared-only workflow surface for subsystem users,
  separate from the generic threshold-testing workflows

## 8. Next recommended actions

Immediate next step:

- acquire or mount a materially larger overlapable perception/imagery dataset
  while keeping the same fixed comparison:
  - Ridge
  - canonical shared-only
  - canonical shared-private, `private_dim=16`
- execute the concrete program in:
  - `docs/DATA_ACQUISITION_PROGRAM.md`
  - `docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`

Ranked data options:

1. richer NSD-style paired imagery/perception data beyond the current public release
2. a secondary public paired imagery/perception dataset
3. large perception-only datasets for the practical Animus lane

Fallback next step if data cannot move soon:

- consider a very small follow-up around `private_dim≈16` or similarly reduced
  private capacity, not a broad architecture redesign

What should wait:

- vividness expansion
- stimulus-vs-percept work
- SDXL/reconstruction expansion
- large ROI redesign
- stronger threshold-paper claims without larger paired data

Project-elevation step:

- keep the benchmark ladder and evidence freeze explicit in every future report
- frame the paper program around threshold-style evidence rather than assumed
  disentanglement superiority
- use `animus_core_decoder.yaml` as the practical subsystem config and
  `threshold_shared_private_p16.yaml` as the primary hypothesis config
- keep Paper 1 framed as an honest benchmark/evidence paper until materially
  larger paired data changes the evidence boundary
- treat the current draft as a serious benchmark manuscript rather than a
  placeholder summary
- prepare Paper 1 for `Imaging Neuroscience` as the primary venue while keeping
  a benchmark-oriented fallback path for `NeurIPS Datasets and Benchmarks`

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
- how should the evidence freeze evolve once materially larger overlap data is
  available?

Risk:

- without more overlap data, it is easy to overfit engineering effort to a
  scientifically underpowered regime
- it is also easy to over-write the current paper as a positive disentanglement
  paper instead of the benchmark/evidence paper the data currently supports
- it is easy to underinvest in writing rigor once the codebase is strong; the
  manuscript needs the same evidence discipline as the benchmark

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
- 2026-04-02: evidence freeze, benchmark ladder, paper positioning, and
  top-level research dossier added to formalize the project as a paper program
- 2026-04-02: shared-only promoted into the Animus Core Decoder lane; threshold
  hypothesis formalized as the main research program framing
- 2026-04-02: dedicated Animus Core Decoder wrapper commands added and the
  fixed benchmark ladder docs aligned to the practical-vs-exploratory split
- 2026-04-02: external-data acquisition program formalized; public NSD-Imagery
  acquisition promoted into the canonical workflow namespace and the next
  decisive step narrowed to larger paired-data integration
- 2026-04-02: Paper 1 drafting scaffold added, including the outline, claims
  map, and figures/tables plan for the current realistic benchmark/evidence
  paper path
- 2026-04-02: first serious Paper 1 draft package written under `docs/paper1/`,
  including section drafts and a manuscript-style full draft constrained by the
  current evidence freeze
- 2026-04-02: first real Paper 1 figure/table package generated under
  `docs/paper1/figures/` and `docs/paper1/tables/`, including the benchmark
  ladder, overlap-scarcity, shared-only-vs-shared-private, and threshold
  schematic figures plus the main results / evidence-boundary / reproducibility
  tables
- 2026-04-02: Paper 1 claims-tightening and citation-planning pass completed;
  manuscript sections now reference concrete figure/table assets, stronger
  claims were softened to match the evidence freeze, and citation planning was
  added before bibliography lock
- 2026-04-02: Paper 1 bibliography lock completed with verified references,
  manuscript citations injected into the core section drafts and full draft,
  related work tightened around benchmark framing, and captions / submission /
  appendix scaffolds added to move the paper toward a submission-ready package
- 2026-04-02: Paper 1 primary venue chosen as `Imaging Neuroscience`; the full
  draft was editorially hardened for venue fit, the appendix was assembled into
  a real supplement document, the submission checklist was upgraded, and a
  submission-package plan was added without changing the evidence boundary
- 2026-04-03: root README rewritten as a professional research-platform landing
  page aligned to the evidence freeze, clarifying the practical Animus lane,
  the threshold-testing research lane, the current benchmark ordering, and the
  repository's implementation and documentation map for new researchers
- 2026-04-04: README reframed for public academic presentation under the
  project identity `perceptionVSimagination`; Animus-facing language was
  removed from the front page, the overview was tightened for a cognitive
  science audience, and the docs test was updated so `START_HERE.md` remains
  the exhaustive command surface while `README.md` stays professional and
  publication-facing
- 2026-04-04: README refined again for stronger professional presentation,
  readability, and interdisciplinary framing; section hierarchy, research
  significance, paper trajectories, and evidence-disciplined project summary
  were improved to make the repository more legible to an academic admissions
  audience without changing the frozen scientific claims
- 2026-04-04: Codex workflow setup was tightened for the current repository:
  `AGENTS.md` was made repo-specific, `PLANS.md` and `Documentation.md` were
  defined as the planning and working-log surfaces, and the repo-local skill
  docs were normalized around explicit inputs, outputs, and handoffs without
  changing the scientific evidence boundary
- 2026-04-04: workflow-review pass completed for dual-lane day-to-day use:
  the file boundary between `AGENTS.md`, `PLANS.md`, `Documentation.md`, and
  `docs/PROJECT_MASTER_LOG.md` was tightened, `.venv` was made the explicit
  canonical environment for validation and workflow commands, a lightweight
  daily loop and optional lane-separated worktree pattern were documented, and
  the five repo-local skills were given a clearer orchestration sequence for
  research design, execution, auditing, and paper handoff
- 2026-04-04: workflow hardening added a compact durable experiment ledger in
  `docs/EXPERIMENT_REGISTRY.md` and a real `.venv` guard on key canonical
  workflow entrypoints so preflight, train, eval, analysis, export, and Ridge
  baseline commands fail fast with an actionable message when run outside the
  project environment
- 2026-04-04: Animus Core export hardening added a compact decoder card
  (`decoder_card.json` and `decoder_card.md`) alongside the existing export
  manifest so the practical shared-only subsystem is easier to inspect, hand
  off, and integrate without changing the frozen benchmark ladder or evidence
  interpretation
- 2026-04-04: Animus integration guidance was tightened so the decoder card is
  now the preferred quick human-facing inspection surface, `manifest.json`
  remains the full machine-readable contract, and a tiny
  `inspect_animus_export` helper was added for fast bundle inspection and
  validation from the project `.venv`
- 2026-04-04: Paper 1 was hardened further toward an **Imaging Neuroscience**
  submission package: the full draft was tightened for flow and reduced
  redundancy, appendix/supplement boundaries were clarified around configs,
  commands, artifact paths, and export-contract details, the submission package
  plan and checklist were updated, and a final claim/style audit was completed
  without changing the frozen benchmark ordering or evidence boundary
- 2026-04-04: a disciplined public-data expansion program was added for the
  dual-lane repo: public dataset opportunities were ranked by threshold,
  practical Animus, and future-paper roles; a machine-readable catalog and
  small inspection helper were added; and the strategy docs now explicitly keep
  perception-only and non-NSD imagery datasets separate from the primary fixed
  threshold ladder unless later evidence justifies promotion
- 2026-04-04: the remote public-data execution surface was audited and
  documented as the environment of record for real acquisition/integration
  work: the live pod `orchestraiq-jupyter-75555bb5f5-hxwp5` in
  `runai-romania-dev` was verified, the repo path
  `/home/jovyan/local-data/perceptionVSimagination` and active cache/output
  roots were confirmed, about `215G` free space was measured before any new
  download, no next-step public datasets were already staged there, and the
  current blocker was recorded that the pod still lacks the repo `.venv` needed
  for canonical workflow execution
- 2026-04-04: the live pod blocker was removed by provisioning a repo-local
  `.venv` under `/home/jovyan/local-data/perceptionVSimagination`, validating
  guarded canonical module execution there, and completing the first real
  `ds004496` practical-data step as a metadata-only Git acquisition under
  `cache/public_datasets/ds004496` with a checked-in workflow wrapper and
  provenance file; this expanded the practical Animus data path without
  changing the frozen threshold ladder or evidence interpretation
- 2026-04-04: the real `ds004496` clone was inspected on the live pod and the
  first practical contract was narrowed to an honest, minimal surface:
  `imagenet` perception-only, multi-session subjects `sub-01..sub-09`, and
  visible `ciftify` beta/label derivatives. A guarded
  `fmri2img.workflows.inspect_public_nod` helper and tightened NOD docs were
  added so the repo can distinguish “inspection-ready” from “training-ready”
  without altering the primary threshold ladder
- 2026-04-04: the live pod git state was normalized back onto `origin/main`
  after earlier copy-based sync drift, with the old pod-local `517b77b` commit
  preserved on a backup branch and remote-only `.pod_deps/` artifacts excluded
  from status noise. The NOD helper was then extended to expose the first
  explicit prepared-index contract: `imagenet`, multi-session subjects
  `sub-01..sub-09`, common sessions `ses-imagenet01..04`, and expected
  common-session run counts, still without claiming shared-only training
  readiness
- 2026-04-04: the first real NOD prepared-index workflow was added for the
  practical Animus lane. `fmri2img.workflows.prepare_public_nod_index` now
  builds a row-level index for the `imagenet` multi-session common-session
  subset and records visible-versus-resolved path states for raw events,
  `fmriprep` BOLD/confounds, and `ciftify` beta/label inputs so the repo can
  track NOD readiness without inflating it into training readiness
- 2026-04-04: the first real NOD prepared index was built on the live pod for
  the `imagenet` common-session subset (`360` rows). The resulting readiness
  breakdown showed `324` `incomplete` rows, `36` `missing_payload` rows, and
  `0` rows currently usable for later shared-only prep, establishing an honest
  operational boundary between “prepared index exists” and “dataset is
  training-ready”
- 2026-04-04: the next NOD readiness step was tightened into an exact payload
  manifest rather than a broad annex pull. The repo can now name the first
  unresolved subset precisely as the `36` `run-10` rows across
  `sub-01..sub-09` and `ses-imagenet01..04`, with an estimated unresolved
  payload size of about `8.23 GiB`. A guarded
  `fmri2img.workflows.materialize_public_nod_payloads` helper now writes the
  manifest/report and refuses materialization when `git-annex` is missing,
  recording the current live-pod blocker without inflating NOD readiness
- 2026-04-04: `git-annex` was enabled directly on the live pod with a minimal
  `apt-get install --no-install-recommends git-annex` step, and the exact
  `36`-row NOD subset was retried through the checked-in materialization
  workflow. Retrieval still failed because the current GitHub metadata mirror
  clone has no usable annex source for the requested keys
  (`remote.origin.annex-ignore=true`, `git-annex whereis` reports `0 copies`),
  so the prepared-index readiness stayed unchanged at `324` `incomplete`,
  `36` `missing_payload`, and `0` usable rows
- 2026-04-04: the payload-source blocker for the first NOD subset was resolved
  by switching from annex retrieval to the official OpenNeuro public S3 path.
  The checked-in materialization workflow now supports direct OpenNeuro S3
  downloads into the existing annex-object targets, and the exact `36`
  `run-10` rows (`144` files, about `8.23 GiB`) were retrieved successfully on
  the live pod. Rerunning the prepared index moved the NOD slice to
  `324` `incomplete`, `36` `resolved`, and `36` usable rows for later
  shared-only prep, without changing the frozen threshold ladder or evidence
  interpretation
- 2026-04-05: the first downstream shared-only adapter surface for NOD was
  added on top of the resolved `36`-row slice. The checked-in
  `fmri2img.workflows.prepare_public_nod_shared_only_adapter` workflow now
  emits a stable adapter parquet and report for exactly `sub-01..sub-09`,
  `ses-imagenet01..04`, `run-10`, with `adapter_ready=true`,
  `prep_ready=true`, and `training_ready=false`. This makes the practical
  Animus lane more reusable without widening the NOD contract or changing any
  scientific claims
- 2026-04-05: the next downstream contract for the same fixed NOD slice was
  added via `fmri2img.workflows.prepare_public_nod_target_selection`. The
  workflow expands the `36` resolved adapter rows into a deterministic
  `3600`-row trial-level target-selection artifact by validating
  `events.tsv` `stim_file` basenames against `ciftify` `label.txt`. The new
  report marks the output as `target_selection_ready=true`,
  `downstream_prep_ready=true`, and `training_ready=false`, preserving the
  boundary that target embeddings, ROI materialization, and a real shared-only
  training config remain separate steps
- 2026-04-05: the fixed NOD slice now also has an explicit canonical
  target-embedding cache contract via
  `fmri2img.workflows.prepare_public_nod_target_embedding_cache`. The workflow
  consumes the deterministic `3600`-row target-selection artifact and emits a
  repo-usable target-embedding manifest for the intended `768`-D ViT-L/14
  cache surface. On the live pod, all `3600` stimulus paths are visible but
  `0` JPEG payloads are currently resolved, so the report correctly keeps
  `target_embedding_ready=false`, `downstream_prep_ready=false`, and
  `training_ready=false` without changing any evidence-facing interpretation
- 2026-04-05: the fixed NOD slice now also has a real canonical target cache.
  `fmri2img.workflows.materialize_public_nod_stimuli` retrieved the exact
  `3600` JPEG stimulus payloads referenced by the manifest from the official
  OpenNeuro public S3 path, and
  `fmri2img.workflows.build_public_nod_target_embedding_cache` then built the
  real `openai/clip-vit-large-patch14` / `clip_target_768` cache for the same
  exact slice. The resulting report now marks the slice as
  `target_embedding_ready=true`, `downstream_prep_ready=true`, and
  `training_ready=false`, preserving the boundary that ROI materialization,
  dataset-side join logic, and a checked-in shared-only train/eval config are
  still separate engineering steps
- 2026-04-05: the fixed NOD slice now also has explicit downstream join and
  ROI-contract surfaces. `fmri2img.workflows.prepare_public_nod_shared_only_join_contract`
  emits a `3600`-row `pair_id`-keyed join artifact linking the adapter,
  target-selection table, target cache, and future ROI keys for the exact
  fixed slice, and
  `fmri2img.workflows.prepare_public_nod_roi_materialization_contract`
  emits a verified `36`-row ROI-source contract that confirms beta/label
  alignment while keeping `roi_ready=false`,
  `downstream_prep_ready=false`, and `training_ready=false`. This makes the
  fixed NOD slice downstream-consumable without widening the slice or
  pretending that neural-side materialization or training is already ready
- 2026-04-06: the fixed NOD slice now has a real ROI parquet and a real
  prepared dataset artifact on the live pod. The checked-in
  `fmri2img.workflows.materialize_public_nod_roi_artifact` workflow built the
  `3600`-row `pair_id`-keyed ROI artifact for the exact slice, and
  `fmri2img.workflows.prepare_public_nod_shared_only_prepared_dataset` built
  the aligned `3600`-row prepared dataset that consumes the join contract, ROI
  artifact, and target cache end-to-end. This advances the practical Animus
  lane to `downstream_prep_ready=true` for the fixed slice while keeping
  `training_ready=false` and preserving the boundary that no evidence-facing
  interpretation changed
- 2026-04-06: the fixed NOD slice now also has a checked-in shared-only config
  and a real trainer-ingestion preflight surface. The new
  `configs/canonical/public_nod_imagenet_run10_shared_only.yaml` config points
  only to the fixed prepared dataset and target cache, and
  `fmri2img.workflows.preflight_public_nod_shared_only_trainer` now verifies
  that the canonical trainer path can load the fixed slice, align the target
  cache and ROI artifact by `pair_id`, build a real batch, and run a real
  forward packet on the live pod. The dedicated preflight marks
  `trainer_config_ready=true` and `preflight_ready=true`, while
  `training_ready` correctly remains `false`
- 2026-04-06: the fixed NOD slice now has a successful smoke-only canonical
  trainer artifact path on the live pod. The checked-in
  `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml` config
  drove a one-epoch `train_decoder` smoke run that wrote
  `best_decoder.pt`, `config_snapshot.json`, `roi_summary.json`,
  `target_summary.json`, and `train_history.json` under
  `outputs/public_nod/train/imagenet_run10_shared_only_smoke/`, and the new
  `fmri2img.workflows.report_public_nod_shared_only_smoke` workflow recorded a
  machine-readable `smoke_report.json` with `smoke_ready=true` while
  `training_ready` correctly remained `false`
- 2026-04-06: the fixed NOD slice now also has a machine-readable eval/export
  smoke status surface. The canonical `export_for_animus` path successfully
  packaged the fixed smoke checkpoint under
  `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`, while the
  canonical `eval_decoder` path remained blocked on the same fixed slice
  because `compute_pair_metrics` still assumes both `perception` and
  `imagery` conditions are present. The new
  `fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke`
  workflow records this honestly as `export_smoke_ready=true`,
  `eval_smoke_ready=false`, and `training_ready=false`
- 2026-04-06: canonical eval smoke is now perception-only-safe for the fixed
  NOD slice. The smallest guard in `fmri2img.evaluation.decoder.compute_pair_metrics`
  now preserves existing paired behavior when both conditions exist but returns
  an explicit unavailable pair-metrics block when only `perception` is
  present. With that guard in place, the live pod successfully wrote
  `metrics.json`, `roi_summary.json`, and `resolved_roi_groups.json` under
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/`, and the
  eval/export smoke report now records `eval_smoke_ready=true`,
  `export_smoke_ready=true`, and `training_ready=false`
- 2026-04-06: canonical post-train hardening for incomplete-condition public
  data now extends through transfer as well as eval. The evaluation module now
  carries an explicit reusable condition-availability contract, and the live
  pod successfully wrote `transfer_metrics.json` and `per_trial_pairs.csv`
  under `outputs/public_nod/transfer/imagenet_run10_shared_only_smoke/` for
  the fixed perception-only NOD slice. The combined smoke report now records
  `present_conditions=["perception"]`, `missing_conditions=["imagery"]`,
  `eval_smoke_ready=true`, `transfer_smoke_ready=true`,
  `export_smoke_ready=true`, and `training_ready=false`
- 2026-04-06: the fixed-slice post-train stack now preserves normalized
  condition semantics across downstream eval, transfer, and export
  consumption. The canonical export bundle for
  `outputs/public_nod/export/imagenet_run10_shared_only_smoke/` now carries
  explicit `condition_semantics` in both `manifest.json` and
  `decoder_card.json`, and the combined smoke report under
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/` now records one
  shared normalized `condition_semantics` block rather than relying on
  implicit pair-metric inference. This remains operational hardening only, and
  `training_ready` correctly remains `false`
- 2026-04-06: the fixed-slice post-train stack now also preserves normalized
  target metadata across export, report, and inspection surfaces. The live
  export bundle under
  `outputs/public_nod/export/imagenet_run10_shared_only_smoke/` now carries
  `metadata.target_spec_normalized` in `manifest.json`, the exported
  `decoder_card.json` now exposes normalized target metadata with
  `source_field_shape="target_name"`, and the combined smoke report under
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/` now records a
  top-level normalized `target_spec` block. This remains operational
  hardening only, and `training_ready` correctly remains `false`
- 2026-04-07: the fixed NOD smoke bundle now has an explicit downstream
  contract verdict. The new canonical workflow
  `fmri2img.workflows.audit_public_nod_shared_only_downstream_contract`
  writes `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`,
  which confirms that `manifest.json`, `decoder_card.json`, and
  `eval_export_smoke_report.json` agree on normalized target metadata,
  normalized condition semantics, experiment identity, benchmark role, target
  dimension, and operational-only readiness. The live audit report marks
  `downstream_contract_ready=true` while `training_ready` correctly remains
  `false`
- 2026-04-07: downstream contract auditing is now reusable beyond the fixed
  NOD smoke bundle. The repo now has an internal shared audit core plus a
  second concrete workflow,
  `fmri2img.workflows.audit_shared_private_smoke_downstream_contract`, and the
  live pod successfully wrote
  `outputs/canonical/eval/shared_private_smoke/downstream_contract_audit.json`
  after regenerating the canonical `shared_private_smoke` train/eval/transfer/
  export bundle. The same compact verdict shape now proves normalized target
  metadata, normalized condition semantics, experiment identity, target
  dimension, and operational-only readiness on both the perception-only fixed
  NOD slice and a paired non-NOD canonical smoke bundle, while
  `training_ready` remains `false`
- 2026-04-07: the repo now also has one top-level canonical dispatcher for
  downstream contract audits:
  `fmri2img.workflows.audit_downstream_contract`. The live pod proved the
  generic path on both currently supported families,
  `public_nod_imagenet_run10_shared_only_smoke` and `shared_private_smoke`,
  while preserving their concrete semantics and compact verdict shape.
  Unsupported bundle families now return an explicit blocked report instead of
  silently pretending to be generically supported. This remains operational
  hardening only, and `training_ready` remains `false`
- 2026-04-07: the generic downstream audit dispatcher no longer owns its own
  family mapping. Supported bundle-family registration now lives in
  `fmri2img.workflows._downstream_contract_registry`, close to the shared audit
  core, and the live pod proved that the registry-backed dispatcher still
  produces the same real contract verdicts for both supported families:
  `public_nod_imagenet_run10_shared_only_smoke` and `shared_private_smoke`.
  Unsupported bundle families remain explicitly blocked rather than being
  implicitly treated as supported. This remains operational hardening only, and
  `training_ready` remains `false`
- 2026-04-07: blocked generic downstream-audit verdicts now come from one
  shared helper in `fmri2img.workflows._downstream_contract_audit` instead of
  from a private constructor inside the top-level dispatcher. The live pod
  proved that both supported families still produce the same real ready
  contract verdicts through `fmri2img.workflows.audit_downstream_contract`,
  while unsupported bundle names remain truthfully blocked with
  `training_ready=false` and the same stable compact top-level shape. This
  remains operational hardening only
- 2026-04-07: the generic downstream dispatcher no longer owns even the
  blocked-path `operational_boundary` strings. Those defaults now live in the
  shared downstream-audit support layer alongside the blocked-report helper,
  and the live pod proved that both supported families still produce the same
  real ready verdicts through the generic path while unsupported bundle names
  remain truthfully blocked with unchanged top-level semantics. This remains
  operational hardening only, and `training_ready` remains `false`
- 2026-04-07: the repo now also has a first explicit readiness-promotion audit
  for the strongest currently proven paired bundle:
  `fmri2img.workflows.audit_shared_private_smoke_readiness`. The live pod wrote
  `outputs/canonical/eval/shared_private_smoke/readiness_audit.json`, which
  marks `operational_ready=true`, `downstream_contract_ready=true`,
  `evidence_ready_candidate=true`, and `training_ready=false` for
  `shared_private_smoke`. The report makes the current promotion boundary
  explicit: the bundle is a real paired evidence candidate, but it is still
  smoke-scoped and fixture-backed, so it is not yet eligible for evidence-grade
  or production-facing promotion
- 2026-04-07: the repo now also has a first non-smoke shared-only readiness
  lane for the best current canonical neural baseline. The new checked-in config
  `configs/canonical/full_imagery_overlap_shared_only.yaml` and workflow
  `fmri2img.workflows.audit_full_imagery_overlap_shared_only_readiness` now
  audit the real pod bundle under
  `outputs/canonical/{train,eval,transfer,export}/full_imagery_overlap_shared_only/`.
  After refreshing the export bundle onto the current normalized manifest/card
  contract, the live readiness artifact
  `outputs/canonical/eval/full_imagery_overlap_shared_only/readiness_audit.json`
  marks `operational_ready=true`, `downstream_contract_ready=true`,
  `evidence_ready_candidate=true`, and `training_ready=false`. The remaining
  blockers are explicit and engineering-honest: the train artifact still has
  `max_available_overlap` override provenance instead of a dedicated checked-in
  shared-only training run, and the held-out paired evaluation slice is still
  only `1/32` paired groups. This strengthens the project’s benchmark-adjacent
  and Animus-adjacent promotion path without changing the evidence freeze or
  claiming production readiness
- 2026-04-07: blocker `#1` on the shared-only non-smoke promotion lane is now
  gone. The live pod reran the canonical sequence for
  `configs/canonical/full_imagery_overlap_shared_only.yaml`, refreshed the
  train/eval/transfer/export bundle under
  `outputs/canonical/{train,eval,transfer,export}/full_imagery_overlap_shared_only/`,
  and rewrote
  `outputs/canonical/eval/full_imagery_overlap_shared_only/readiness_audit.json`.
  The refreshed train `config_snapshot.json` now records
  `experiment.name="full_imagery_overlap_shared_only"` instead of the old
  `max_available_overlap` override provenance. The readiness state remains
  honest: `operational_ready=true`, `downstream_contract_ready=true`,
  `evidence_ready_candidate=true`, and `training_ready=false`, with only one
  remaining blocker: the held-out paired evaluation slice is still only
  `1/32` paired groups
- 2026-04-07: blocker `#2` on the same shared-only non-smoke promotion lane is
  now machine-readable as a real prepared-data ceiling rather than only as a
  failed gate string. The live pod refreshed
  `outputs/canonical/eval/full_imagery_overlap_shared_only/readiness_audit.json`
  so it now includes a `heldout_support` section showing `94` prepared rows,
  `5` paired groups total, split paired-group counts `{train: 3, val: 1,
  test: 1}`, and
  `current_dataset_can_meet_training_pair_threshold=false`. A temporary
  exact-config rebuild under
  `outputs/canonical/prepared/full_imagery_overlap_phase3_audit/` reproduced
  the same `5` overlap ids and the same `3/1/1` split, confirming that the
  current mounted benchmark cannot honestly satisfy the unchanged `32`-pair
  `training_ready` gate for this lane. The readiness state remains
  `operational_ready=true`, `downstream_contract_ready=true`,
  `evidence_ready_candidate=true`, and `training_ready=false`
- 2026-04-08: the repo now also has a machine-readable promotion-path audit for
  the current best non-smoke canonical lane:
  `fmri2img.workflows.audit_full_imagery_overlap_promotion_path`. The live pod
  wrote
  `outputs/canonical/eval/full_imagery_overlap_shared_only/promotion_path_audit.json`,
  which compares `full_imagery_overlap_shared_only` against the nearest
  checked-in canonical alternatives
  (`animus_core_decoder`, `threshold_shared_private_p16`,
  `max_available_overlap`, and `multisubj_overlap_bootstrap`). The report
  proves that no mounted canonical lane currently improves paired support
  beyond `5` total / `1` held-out pair while also exposing a stronger real
  post-train bundle, so the current main promotion lane remains unchanged and
  the next honest move is explicit: paired-data expansion, not lane switching,
  gate weakening, or benchmark inflation
- 2026-04-08: the repo now also has a machine-readable mounted-source ceiling
  audit for that same main lane:
  `fmri2img.workflows.audit_full_imagery_overlap_data_expansion`. The live pod
  wrote
  `outputs/canonical/eval/full_imagery_overlap_shared_only/data_expansion_audit.json`,
  which confirms that the full canonical imagery indices already exist for
  `subj01..subj08` and that only `subj02`, `subj03`, `subj05`, and `subj07`
  overlap the mounted perception indices at all. The current lane therefore
  already exhausts the environment’s honest paired support at `5` total / `1`
  held-out pair group, and no prepared mixed index under
  `outputs/canonical/prepared/` exceeds that ceiling. The next honest move is
  now explicit and machine-readable: external paired-data expansion, not more
  overlap hardening on the currently mounted public source
