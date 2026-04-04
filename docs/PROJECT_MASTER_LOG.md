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
