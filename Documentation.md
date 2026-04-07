# Workflow Notes

Use this file as the working journal for milestones, decisions, and follow-up
actions that arise during active repository work. It complements, but does not
replace, `docs/PROJECT_MASTER_LOG.md`.

## File boundary

- `Documentation.md`: active working memory for the current repo state
- `docs/EXPERIMENT_REGISTRY.md`: durable compact ledger of important runs
- `docs/PROJECT_MASTER_LOG.md`: durable project history and official milestones

Promotion rule:

- if it only matters for the current pass, keep it here
- if it is a durable run worth finding later, add a compact entry to
  `docs/EXPERIMENT_REGISTRY.md`
- if it changes the enduring repo story, add a concise milestone to
  `docs/PROJECT_MASTER_LOG.md`

## How this file should be used

- Capture local working decisions before they become permanent docs changes.
- Track what changed, why it changed, and what still needs follow-up.
- Record doc-routing decisions so engineering changes do not accidentally turn
  into scientific claims.
- Summarize the next concrete action for the user or the next Codex session.

## Daily note rhythm

For a normal pass:

1. record the scope and target lane
2. note the exact validation command run from `.venv`
3. state the decision made
4. record the next concrete follow-up

Keep this file short enough that the current operational state is easy to scan.

## Logging rules

- Keep entries short and date-stamped.
- Prefer decisions and follow-ups over narrative transcripts.
- If a change becomes a real milestone, append a concise version to
  `docs/PROJECT_MASTER_LOG.md`.
- If the change affects evidence, benchmark ordering, or paper framing, cite
  the governing docs explicitly.

## Entry template

```md
## YYYY-MM-DD - <short title>

- Scope: engineering | experiment | paper | reproducibility
- Status: in_progress | completed | blocked
- Surfaces touched: <files, configs, workflows, docs>
- Validation: <command run or reason no validation applied>
- Decision: <what was decided and why>
- Claim boundary: <why this does or does not change scientific interpretation>
- Follow-up: <next concrete action>
```

## Decision routing

Use this routing table when deciding where a result belongs:

- Workflow surface or command contract:
  `START_HERE.md`, `docs/REPRODUCIBILITY.md`, `docs/VALIDATION.md`
- Architecture or data contract:
  `docs/ARCHITECTURE.md`, `docs/CURRENT_STATE.md`
- Experiment result or rerun:
  run report and `docs/EXPERIMENT_REGISTRY.md` first, then
  `docs/REPRODUCIBILITY.md`
- Evidence boundary or benchmark ordering:
  `docs/CURRENT_EVIDENCE_FREEZE.md`, `docs/BENCHMARK_LADDER.md`,
  `docs/PAPER_1_CLAIMS_MAP.md`
- Manuscript framing:
  `docs/paper1/`, related `docs/PAPER_1_*`, then `README.md`

Paper handoff rule:

- run report or audit first
- evidence-facing docs second, but only if the trust boundary truly changed
- manuscript and README updates last

## Current standing reminders

- Ridge remains the strongest fixed-benchmark baseline.
- Shared-only remains the best current canonical neural baseline.
- Shared-private remains exploratory on the current paired benchmark.
- Animus-oriented subsystem work should improve the shared-only practical lane
  without overstating scientific conclusions.
- Canonical workflow validation should be run from the project `.venv`.
- Key canonical workflow entrypoints now fail fast outside the project `.venv`.

## Latest entry

## 2026-04-05 - NOD target-selection contract built for the fixed adapter slice

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/prepare_public_nod_target_selection.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused tests via
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q -k 'prepare_public_nod_target_selection_builds_deterministic_trial_table or prepare_public_nod_target_selection_rejects_label_mismatch or prepare_public_nod_shared_only_adapter_requires_full_fixed_slice or public_dataset_program_docs_and_catalog_exist'`;
  remote build via
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_selection`
  and remote focused tests from the pod `.venv`
- Decision: added the smallest canonical target-selection surface over the
  fixed NOD adapter slice by expanding the `36` adapter rows into a
  deterministic `3600`-row trial-level target table using matching
  `events.tsv` `stim_file` and `label.txt` basenames
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  still a narrow practical Animus-lane target-selection artifact only
- Follow-up: build the smallest canonical target embedding cache over this
  target-selection artifact before considering any real shared-only training
  config

## 2026-04-05 - NOD shared-only adapter built for the fixed resolved slice

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/prepare_public_nod_shared_only_adapter.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused tests via
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q -k 'prepare_public_nod_shared_only_adapter_keeps_only_fixed_resolved_subset or prepare_public_nod_shared_only_adapter_requires_full_fixed_slice or materialize_public_nod_payloads_uses_direct_openneuro_s3_by_default or public_dataset_program_docs_and_catalog_exist'`;
  remote build via
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_adapter`
  and remote focused tests from the pod `.venv`
- Decision: added the smallest downstream shared-only adapter surface over the
  already resolved NOD `run-10` subset and kept the state boundary explicit:
  adapter-ready, prep-ready, not training-ready
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  still a narrow practical Animus-lane prep artifact only
- Follow-up: define the smallest target-selection and ROI-materialization
  contracts needed before any real shared-only training config can point to
  this adapter output

## 2026-04-04 - NOD direct official payload retrieval succeeded for the exact subset

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/materialize_public_nod_payloads.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: audited the official OpenNeuro public S3 bucket, confirmed
  dataset-relative object paths under
  `https://s3.amazonaws.com/openneuro.org/ds004496/`, ran
  `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize --strategy direct_openneuro_s3`
  on pod `orchestraiq-jupyter-75555bb5f5-hxwp5`, and reran
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_index`
- Decision: the fixed `36`-row `run-10` subset is now genuinely resolved via
  official direct download, yielding `36` usable rows for later shared-only
  prep without widening the NOD contract
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  still a practical Animus-lane data-readiness improvement only
- Follow-up: implement the smallest shared-only prep adapter over the now
  resolved `36`-row subset before considering any wider NOD expansion

## 2026-04-04 - NOD annex enabled but upstream payload source still missing

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/materialize_public_nod_payloads.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: enabled `git-annex` on pod
  `orchestraiq-jupyter-75555bb5f5-hxwp5` via `apt-get install -y --no-install-recommends git-annex`,
  initialized annex state inside the `ds004496` clone, attempted
  `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize`,
  and reran `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_index`
- Decision: the local tooling blocker is removed, but the current GitHub-based
  metadata mirror clone still has no usable annex source for the target keys,
  so the exact `36`-row subset remains unresolved and the prepared index is
  unchanged
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  still a practical Animus-lane operational pass only
- Follow-up: identify the real annex-capable upstream for `ds004496`, then
  rerun the existing exact-subset materialization workflow before considering
  any later shared-only prep step

## 2026-04-04 - NOD exact payload manifest and annex-tooling blocker

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/materialize_public_nod_payloads.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: remote audit on pod `orchestraiq-jupyter-75555bb5f5-hxwp5`
  confirmed `216G` free space, identified the exact `36` `missing_payload`
  rows, and measured their estimated unresolved payload size at about
  `8.23 GiB`; remote helper execution wrote the exact manifest/report, refused
  `--materialize` because `git-annex` is absent, and rerunning the prepared
  index left readiness unchanged; local focused tests cover the new manifest helper
- Decision: added a tight manifest/report workflow for the first unresolved NOD
  payload subset and kept the materialization boundary honest by refusing
  `--materialize` when `git-annex` is absent on the live pod
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  still a practical Animus-lane readiness pass only
- Follow-up: add `git-annex` to the live pod image or runtime, then rerun
  `fmri2img.workflows.materialize_public_nod_payloads --materialize` and
  rebuild the prepared index before considering any later shared-only prep step

## 2026-04-04 - NOD prepared index built on live pod

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `docs/NOD_PUBLIC_DATASET.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: remote build via
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_index` and
  remote focused checks from pod `.venv`
- Decision: materialized the first real NOD prepared index for the `imagenet`
  multi-session common-session subset and kept row-level readiness honest with
  `incomplete` versus `missing_payload` rather than inflating the clone into
  training readiness
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is
  a practical Animus-lane preparation artifact only
- Follow-up: determine whether the next move should be selective annex
  materialization for the exact `fmriprep`/`ciftify` inputs named by the index,
  or a narrower adapter pass over the currently visible raw-event subset

## 2026-04-04 - NOD prepared-index adapter

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/prepare_public_nod_index.py`,
  `src/fmri2img/workflows/inspect_public_nod.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`, `docs/ANIMUS_CORE_DECODER.md`,
  `START_HERE.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused tests via
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q -k 'prepare_public_nod_index_marks_resolved_and_missing_payload or inspect_public_nod_summarizes_minimal_layout or public_dataset_program_docs_and_catalog_exist or docs_reference_canonical_workflows'`
- Decision: added the first real prepared-index workflow for the NOD
  `imagenet` multi-session common-session subset, with row-level visibility and
  payload-resolution flags so the repo can distinguish `resolved` rows from
  `missing_payload` rows
- Claim boundary: no threshold-benchmark or evidence-boundary change; this is a
  practical Animus-lane indexing surface only
- Follow-up: build the first remote prepared index on the live pod and decide
  whether enough rows are actually `resolved` to justify a later prepared-index
  adapter into the shared-only train/eval path

## 2026-04-04 - Remote git normalization and NOD prepared-index contract

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/inspect_public_nod.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: precise remote git audit on pod
  `orchestraiq-jupyter-75555bb5f5-hxwp5`; local focused tests via
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q -k 'inspect_public_nod_summarizes_minimal_layout or public_dataset_program_docs_and_catalog_exist or acquire_public_nod_wrapper_invokes_official_script or docs_reference_canonical_workflows'`
- Decision: normalized the live pod checkout back to `origin/main` while
  preserving the old pod-local commit on a backup branch, and tightened the
  first NOD prepared-index contract to `imagenet`, `sub-01..sub-09`, common
  sessions `ses-imagenet01..04`, and `ciftify` beta/label derivatives
- Claim boundary: no benchmark ordering or evidence interpretation changed; the
  new contract is still inspection-ready and contract-ready, not training-ready
- Follow-up: implement the first prepared-index adapter for the `imagenet`
  multi-session subset using the explicit file-pattern contract now captured in
  `inspect_public_nod`

## 2026-04-04 - NOD inspection contract

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/inspect_public_nod.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `START_HERE.md`,
  `tests/test_canonical_workflows.py`, `Documentation.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: inspected the real `ds004496` clone on pod
  `orchestraiq-jupyter-75555bb5f5-hxwp5`; local focused tests via
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q -k 'inspect_public_nod_summarizes_minimal_layout or acquire_public_nod_wrapper_invokes_official_script or public_dataset_program_docs_and_catalog_exist or docs_reference_canonical_workflows'`
- Decision: defined the smallest honest NOD contract as `imagenet` plus the
  multi-session cohort `sub-01..sub-09` with visible `ciftify`
  `*_beta.dscalar.nii` and `*_label.txt` derivatives, then added a guarded
  inspection helper rather than pretending the dataset is already training-ready
- Claim boundary: no benchmark ordering, evidence interpretation, or threshold
  claim changed; NOD remains a perception-only practical Animus dataset
- Follow-up: turn the inspection contract into the first prepared-index adapter
  for the `imagenet` multi-session subset before adding any shared-only train
  config

## 2026-04-04 - Remote NOD acquisition bootstrap

- Scope: engineering, data acquisition, reproducibility
- Status: completed
- Surfaces touched: `scripts/download_nod_metadata.py`,
  `src/fmri2img/workflows/acquire_public_nod.py`,
  `docs/NOD_PUBLIC_DATASET.md`, `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/DATA_ACQUISITION_PROGRAM.md`, `docs/ANIMUS_CORE_DECODER.md`,
  `configs/public_datasets/catalog.json`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: remote `.venv` provisioned on
  `/home/jovyan/local-data/perceptionVSimagination`; verified
  `.venv/bin/python`, `.venv/bin/pip`, guarded module execution, and ran
  `./.venv/bin/python -m fmri2img.workflows.acquire_public_nod --output cache/public_datasets/ds004496`
  on pod `orchestraiq-jupyter-75555bb5f5-hxwp5`
- Decision: created the smallest real ds004496 integration path as a
  metadata-only Git acquisition plus provenance record, explicitly scoped to
  the practical Animus lane
- Claim boundary: no benchmark ordering, evidence interpretation, or threshold
  claim changed; ds004496 remains a perception-only practical dataset and not a
  replacement for the primary paired ladder
- Follow-up: inspect the cloned NOD layout and derivatives, then decide the
  smallest perception-only shared-only adapter contract before any annexed
  download

## 2026-04-04 - Remote public-data execution audit

- Scope: experiment, reproducibility, strategy
- Status: completed
- Surfaces touched: `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/DATA_ACQUISITION_PROGRAM.md`, `Documentation.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: live remote audit via `kubectl` against namespace
  `runai-romania-dev` and pod `orchestraiq-jupyter-75555bb5f5-hxwp5`;
  verified repo/data/cache/output paths and free space before any download
- Decision: treat the live Jupyter pod as the execution surface of record for
  real dataset acquisition and canonical prep; record the actual remote repo
  path and storage roots; do not start remote canonical acquisition/prep until
  the repo `.venv` exists there
- Claim boundary: no benchmark ordering, evidence interpretation, or dataset
  suitability claim changed; this pass only hardened the operational contract
  around where public-data work should actually happen
- Follow-up: provision the project `.venv` on the live pod, then implement the
  first remote acquisition surface for `ds004496` or the first remote metadata
  inspection pass for `ds000203`

## 2026-04-04 - Animus export decoder card

- Scope: engineering, reproducibility
- Status: completed
- Surfaces touched: `src/fmri2img/export/animus.py`,
  `tests/test_canonical_trainer_and_export.py`, `docs/ANIMUS_CORE_DECODER.md`,
  `Documentation.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: `./.venv/bin/pytest tests/test_canonical_trainer_and_export.py -q`
- Decision: added a compact `decoder_card.json` and `decoder_card.md` to the
  Animus export bundle so the shared-only practical subsystem is easier to
  inspect and hand off without parsing the full manifest
- Claim boundary: no benchmark ordering or scientific interpretation changed;
  this is an Animus-lane export usability improvement only
- Follow-up: if downstream integration work starts consuming export bundles,
  decide whether the decoder card should become the preferred human-facing
  inspection surface in `docs/ANIMUS_INTEGRATION.md`

## 2026-04-04 - Animus export inspection surface

- Scope: engineering, reproducibility
- Status: completed
- Surfaces touched: `docs/ANIMUS_INTEGRATION.md`,
  `src/fmri2img/workflows/inspect_animus_export.py`,
  `tests/test_canonical_trainer_and_export.py`, `Documentation.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: `./.venv/bin/pytest tests/test_canonical_trainer_and_export.py -q`
- Decision: promoted `decoder_card.json` / `decoder_card.md` to the preferred
  quick human-facing inspection surface, kept `manifest.json` as the full
  machine-readable contract, and added a tiny inspection helper for export bundles
- Claim boundary: no scientific claims changed; this is an Animus integration
  usability improvement only
- Follow-up: if downstream consumers stabilize, decide whether the helper output
  should become the standard quick-check step in deployment or handoff notes

## 2026-04-04 - Paper 1 submission hardening

- Scope: paper
- Status: completed
- Surfaces touched: `docs/paper1/PAPER_1_FULL_DRAFT.md`,
  `docs/paper1/PAPER_1_APPENDIX.md`,
  `docs/paper1/PAPER_1_SUBMISSION_PACKAGE_PLAN.md`,
  `docs/paper1/PAPER_1_SUBMISSION_CHECKLIST.md`,
  `docs/PAPER_POSITIONING.md`, `Documentation.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: manual claims/style audit against
  `docs/CURRENT_EVIDENCE_FREEZE.md`, `docs/PAPER_1_CLAIMS_MAP.md`,
  `docs/BENCHMARK_LADDER.md`, and `docs/TOP_LEVEL_RESEARCH_DOSSIER.md`
- Decision: kept **Imaging Neuroscience** as the primary venue and tightened
  the manuscript package toward that style by reducing repo-internal tone,
  clarifying supplement boundaries, and making figure/table/appendix support
  more submission-like
- Claim boundary: no benchmark ordering or evidence interpretation changed; the
  paper remains an honest benchmark/evidence paper under overlap scarcity
- Follow-up: convert the markdown manuscript and appendix into the target venue
  template and complete the final bibliography/caption style pass

## 2026-04-04 - Public dataset expansion program

- Scope: experiment, reproducibility, strategy
- Status: completed
- Surfaces touched: `docs/PUBLIC_DATASET_OPPORTUNITY_MAP.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `configs/public_datasets/catalog.json`,
  `src/fmri2img/workflows/show_public_dataset_options.py`,
  `docs/TOP_LEVEL_RESEARCH_DOSSIER.md`, `Documentation.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: repo-context audit plus public-source verification against the
  relevant dataset records/papers; local helper smoke check from `.venv`
- Decision: chose a hybrid public-data pass that ranks real public datasets by
  threshold relevance, practical Animus value, and future-paper value, while
  keeping the primary fixed ladder unchanged
- Claim boundary: no evidence-freeze or benchmark-ordering changes; perception-only
  public datasets remain Animus/support assets, and non-NSD imagery datasets remain
  secondary-benchmark or future-paper assets unless later validated otherwise
- Follow-up: implement the first concrete public-data acquisition surface for
  `ds004496` (NOD) as the best immediate practical Animus target

## 2026-04-04 - Workflow setup refinement

- Scope: engineering, experiment, paper, reproducibility
- Status: completed
- Surfaces touched: `AGENTS.md`, `PLANS.md`, `Documentation.md`,
  `.agents/skills/*/SKILL.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: canonical workflow validation should be run from the project
  environment; `.venv` includes `torchvision`, and
  `./.venv/bin/pytest tests/test_canonical_workflows.py -q` passes there
- Decision: tightened the Codex workflow contract around the actual canonical
  configs, lanes, docs, and readiness labels already used by the repository
- Claim boundary: no evidence-facing claims changed; this was an operational
  workflow cleanup only
- Follow-up: add repo-specific checks for skill-doc frontmatter or workflow-doc
  consistency, and keep future canonical workflow validation anchored to the
  project `.venv`

## 2026-04-05 - NOD target-embedding cache contract

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/prepare_public_nod_target_embedding_cache.py`,
  `tests/test_canonical_workflows.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  plus `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_embedding_cache`
  and focused remote pytest on the live pod
- Decision: added the smallest honest canonical target-embedding surface over
  the fixed `3600`-row NOD target-selection artifact by emitting a manifest for
  the intended `768`-D ViT-L/14 cache contract instead of faking embeddings
- Claim boundary: no benchmark, evidence-freeze, or paper-claim changes; the
  live NOD slice remains a narrow practical Animus-lane prep path only, and
  the current target JPEG payloads are still unresolved on the pod
- Follow-up: materialize the exact `3600` NOD stimulus JPEG payloads, then
  compute the real `clip_target_768` cache before considering any shared-only
  train/eval config

## 2026-04-05 - NOD real target cache built for the fixed slice

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched: `src/fmri2img/workflows/materialize_public_nod_stimuli.py`,
  `src/fmri2img/workflows/build_public_nod_target_embedding_cache.py`,
  `tests/test_canonical_workflows.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote exact JPEG
  materialization on the live pod; remote rerun of
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_embedding_cache`;
  remote real cache build via
  `./.venv/bin/python -m fmri2img.workflows.build_public_nod_target_embedding_cache`;
  focused remote pytest on the pod
- Decision: converted the fixed NOD target-embedding contract into a real
  canonical `clip_target_768` cache for the exact `3600`-row slice by
  materializing only the referenced JPEGs from the official OpenNeuro public
  S3 path and then embedding them with `openai/clip-vit-large-patch14`
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; this remains a narrow practical Animus-lane prep result only
- Follow-up: define the dataset-side join contract plus ROI materialization
  contract needed before any honest shared-only train/eval config can consume
  the fixed NOD slice

## 2026-04-05 - NOD join contract and ROI contract added for the fixed slice

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/prepare_public_nod_shared_only_join_contract.py`,
  `src/fmri2img/workflows/prepare_public_nod_roi_materialization_contract.py`,
  `tests/test_canonical_workflows.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `docs/PUBLIC_DATASET_INTEGRATION_PLAN.md`,
  `docs/ANIMUS_CORE_DECODER.md`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; remote runs of
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_join_contract`
  and
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_roi_materialization_contract`;
  focused remote pytest on the pod
- Decision: added the smallest machine-readable dataset-side join contract and
  the smallest verified ROI materialization contract for the exact fixed NOD
  slice. The join contract is `join_ready=true`, while the ROI surface remains
  contract-only and keeps `roi_ready=false`,
  `downstream_prep_ready=false`, and `training_ready=false`
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; this remains a narrow practical Animus-lane downstream-prep result
  only
- Follow-up: materialize the actual ROI-side artifact keyed by `pair_id`, then
  add the dataset-side loader path that consumes the NOD join contract, ROI
  artifact, and target cache without widening the slice

## 2026-04-06 - NOD ROI artifact and prepared dataset built on the live pod

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/materialize_public_nod_roi_artifact.py`,
  `src/fmri2img/workflows/prepare_public_nod_shared_only_prepared_dataset.py`,
  `tests/test_canonical_workflows.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; remote runs of
  `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_roi_artifact`
  and
  `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_prepared_dataset`;
  focused remote pytest on the pod
- Decision: materialized the real `pair_id`-keyed ROI parquet for the exact
  fixed NOD slice and built the first real prepared dataset artifact that
  consumes the join contract, ROI artifact, and target cache end-to-end
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; this remains a narrow practical Animus-lane downstream-prep result
  only
- Detail: the live ROI artifact now uses only the subject-universal atlas
  sources across the fixed slice (`BA_exvivo` and `aparc`), leaving
  subject-specific `floc-faces` / `floc-places` masks out of the fixed-slice
  materialized feature set rather than faking them
- Follow-up: define the smallest checked-in shared-only preflight/train config
  that points to the fixed prepared dataset and target cache, while keeping
  `training_ready=false` until canonical trainer validation is complete

## 2026-04-06 - NOD trainer-preflight config and canonical ingestion validation

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`,
  `src/fmri2img/workflows/preflight_public_nod_shared_only_trainer.py`,
  `tests/test_canonical_workflows.py`, `docs/NOD_PUBLIC_DATASET.md`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; remote runs of
  `./.venv/bin/python -m fmri2img.workflows.preflight_public_nod_shared_only_trainer --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml`
  and
  `./.venv/bin/python -m fmri2img.workflows.preflight_data --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml --output outputs/public_nod/train/imagenet_run10_shared_only_preflight/preflight_data.json`;
  focused remote pytest on the pod
- Decision: added the smallest checked-in shared-only config for the fixed NOD
  slice and validated that the canonical trainer path can load the prepared
  dataset, align the target cache and ROI artifact by `pair_id`, build a real
  batch, and run a real forward packet without widening the slice
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; this remains a narrow practical Animus-lane preflight result only
- Detail: the dedicated trainer preflight marks
  `trainer_config_ready=true`, `preflight_ready=true`, and
  `training_ready=false`; the generic canonical preflight report classifies the
  same config as `bootstrap_ready`
- Follow-up: run the smallest controlled `train_decoder` smoke on this exact
  config to validate trainer output-path artifacts without treating it as a
  benchmark run

## 2026-04-06 - NOD shared-only trainer smoke artifacts and smoke report

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`,
  `src/fmri2img/workflows/report_public_nod_shared_only_smoke.py`,
  `START_HERE.md`, `docs/NOD_PUBLIC_DATASET.md`,
  `tests/test_canonical_workflows.py`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; remote run of
  `./.venv/bin/python -m fmri2img.workflows.train_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  remote run of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: added the smallest checked-in smoke-only trainer config and a
  machine-readable smoke report workflow, then confirmed on the live pod that
  the canonical trainer can write real smoke artifacts for the exact fixed NOD
  slice without widening scope
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; smoke losses and checkpoint outputs are operational only and are
  not benchmark evidence
- Detail: the live smoke output path now contains `best_decoder.pt`,
  `config_snapshot.json`, `roi_summary.json`, `target_summary.json`,
  `train_history.json`, and `smoke_report.json`; the smoke report marks
  `trainer_config_ready=true`, `preflight_ready=true`, `smoke_ready=true`,
  and `training_ready=false`
- Follow-up: keep the fixed slice unchanged and, if needed later, use this
  smoke path only as an operational gate before any separately-scoped eval or
  benchmark work

## 2026-04-06 - NOD eval/export smoke and honest blocked-state report

- Scope: engineering, data acquisition
- Status: completed with partial operational success
- Surfaces touched:
  `src/fmri2img/workflows/report_public_nod_shared_only_eval_export_smoke.py`,
  `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`,
  `START_HERE.md`, `docs/NOD_PUBLIC_DATASET.md`,
  `tests/test_canonical_workflows.py`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; real remote run of
  `./.venv/bin/python -m fmri2img.workflows.eval_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  real remote run of
  `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  remote run of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: reused the fixed smoke config and smoke checkpoint, then added the
  smallest eval/export smoke report workflow so the repo records export success
  and eval blockage explicitly instead of inferring readiness from missing
  files
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; export artifacts and the eval blocker are operational only and are
  not evidence-facing results
- Detail: the live pod export smoke produced `manifest.json`,
  `decoder_card.json`, `decoder_card.md`, `config_snapshot.json`, and a copied
  `best_decoder.pt` under
  `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`; the canonical
  eval smoke did not write `metrics.json`, `roi_summary.json`, or
  `resolved_roi_groups.json` because `compute_pair_metrics` currently assumes
  both `perception` and `imagery` conditions are present
- Readiness: the eval/export smoke report marks `eval_smoke_ready=false`,
  `export_smoke_ready=true`, and `training_ready=false`
- Follow-up: add the smallest perception-only-safe eval smoke path or guard in
  the canonical evaluation surface before treating eval smoke as operationally
  ready for this fixed NOD slice

## 2026-04-06 - NOD perception-only eval smoke guard and successful live eval smoke

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/evaluation/decoder.py`,
  `tests/test_canonical_workflows.py`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; real remote run of
  `./.venv/bin/python -m fmri2img.workflows.eval_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  real remote run of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: added the smallest canonical perception-only guard in
  `compute_pair_metrics` so paired metrics remain unchanged when both
  conditions exist, but perception-only eval now returns an explicit
  unavailable pair-metrics block instead of crashing
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; the resulting evaluation metrics remain operational smoke outputs
  only and are not benchmark evidence
- Detail: the live pod now writes `metrics.json`, `roi_summary.json`, and
  `resolved_roi_groups.json` under
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/`; the machine-
  readable eval/export smoke report now marks `eval_smoke_ready=true`,
  `export_smoke_ready=true`, and `training_ready=false`
- Follow-up: keep using the explicit `pair_metrics.available=false` contract
  for perception-only slices unless a later task intentionally scopes a more
  general evaluation API refinement

## 2026-04-06 - Condition-availability hardening for canonical eval and transfer

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/evaluation/decoder.py`,
  `src/fmri2img/workflows/report_public_nod_shared_only_eval_export_smoke.py`,
  `tests/test_canonical_workflows.py`, `Documentation.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest from `.venv`; remote `git pull --rebase`
  on the live pod; real remote run of
  `./.venv/bin/python -m fmri2img.workflows.eval_transfer --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  remote regeneration of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: generalized the perception-only-safe condition contract into the
  canonical evaluation module so eval and transfer share the same machine-
  readable availability semantics instead of relying on scattered ad hoc checks
- Claim boundary: no threshold-benchmark, evidence-freeze, or paper-claim
  changes; eval/transfer outputs remain operational hardening artifacts only
- Detail: `compute_pair_metrics` now depends on an explicit reusable
  condition-availability description; the combined eval/export smoke report now
  tracks transfer smoke separately; the live pod now writes
  `transfer_metrics.json` and `per_trial_pairs.csv` for the fixed NOD slice
  with `present_conditions=["perception"]`,
  `missing_conditions=["imagery"]`, and
  `pair_metrics_require_both_perception_and_imagery`
- Readiness: the live combined report marks `eval_smoke_ready=true`,
  `transfer_smoke_ready=true`, `export_smoke_ready=true`, and
  `training_ready=false`
- Follow-up: if a later public-data slice is imagery-only or partially paired,
  reuse the same canonical condition-availability contract instead of adding
  dataset-specific eval guards

## 2026-04-06 - Downstream condition-semantics normalization for eval, transfer, and export

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/evaluation/decoder.py`,
  `src/fmri2img/workflows/report_public_nod_shared_only_eval_export_smoke.py`,
  `src/fmri2img/workflows/export_for_animus.py`,
  `src/fmri2img/export/animus.py`,
  `src/fmri2img/workflows/inspect_animus_export.py`,
  `tests/test_canonical_workflows.py`,
  `tests/test_canonical_trainer_and_export.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest and py_compile from `.venv`; remote
  `git pull --rebase` on the live pod; real remote rerun of
  `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  remote regeneration of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: downstream post-train consumers now normalize condition semantics
  from eval/transfer payloads instead of inferring paired availability from the
  presence or absence of pair metrics
- Claim boundary: operational hardening only; no benchmark progress, no
  evidence-freeze change, and `training_ready` remains `false`
- Detail: the live export bundle now preserves normalized
  `condition_semantics` in `manifest.json` and `decoder_card.json`; the
  combined eval/transfer/export smoke report now exposes one shared normalized
  `condition_semantics` block plus per-surface normalized views for the fixed
  perception-only NOD slice
- Readiness: the live combined report still marks `eval_smoke_ready=true`,
  `transfer_smoke_ready=true`, `export_smoke_ready=true`, and
  `training_ready=false`, while explicitly recording
  `present_conditions=["perception"]`, `missing_conditions=["imagery"]`, and
  `paired_metrics_available=false`
- Follow-up: teach any later downstream consumer to read the normalized
  `condition_semantics` block first instead of guessing from `pair_metrics`
  presence

## 2026-04-06 - Downstream target-spec normalization for eval, export, and inspection

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/export/animus.py`,
  `src/fmri2img/workflows/report_public_nod_shared_only_eval_export_smoke.py`,
  `src/fmri2img/workflows/inspect_animus_export.py`,
  `tests/test_canonical_workflows.py`,
  `tests/test_canonical_trainer_and_export.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest and py_compile from `.venv`; remote
  `git pull --rebase` on the live pod; real remote rerun of
  `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`;
  remote regeneration of
  `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  remote validation with
  `./.venv/bin/python -m fmri2img.workflows.inspect_animus_export outputs/public_nod/export/imagenet_run10_shared_only_smoke --validate`;
  focused remote pytest on the pod
- Decision: downstream post-train consumers now normalize target metadata from
  `target_spec.name` and `target_spec.target_name` into one explicit canonical
  packet instead of depending on one legacy field shape
- Claim boundary: operational hardening only; no benchmark progress, no
  evidence-freeze change, and `training_ready` remains `false`
- Detail: the live export bundle now preserves
  `metadata.target_spec_normalized` in `manifest.json`; `decoder_card.json`
  and `inspect_animus_export` now expose normalized target metadata; the live
  combined smoke report now exposes one shared normalized `target_spec` block
  and uses it for `manifest_target_name`
- Readiness: the live combined report still marks `eval_smoke_ready=true`,
  `transfer_smoke_ready=true`, `export_smoke_ready=true`, and
  `training_ready=false`, while the normalized target block now records
  `target_name_normalized="vit_l14_image_768"`,
  `target_dimension_normalized=768`, and
  `source_field_shape="target_name"`
- Follow-up: if later downstream consumers need target metadata, read
  `metadata.target_spec_normalized` or the combined report `target_spec` block
  first instead of branching on `name` versus `target_name`

## 2026-04-07 - Downstream contract audit for the fixed NOD smoke bundle

- Scope: engineering, data acquisition
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/audit_public_nod_shared_only_downstream_contract.py`,
  `src/fmri2img/evaluation/decoder.py`,
  `docs/NOD_PUBLIC_DATASET.md`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused pytest and py_compile from `.venv`; remote
  `git pull --rebase` on the live pod; real remote run of
  `./.venv/bin/python -m fmri2img.workflows.audit_public_nod_shared_only_downstream_contract --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`;
  focused remote pytest on the pod
- Decision: the fixed NOD smoke bundle now has one explicit machine-readable
  downstream contract verdict instead of relying on manual inspection across
  `manifest.json`, `decoder_card.json`, and
  `eval_export_smoke_report.json`
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the live audit report at
  `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/downstream_contract_audit.json`
  verifies normalized target metadata, normalized condition semantics,
  experiment identity, benchmark role, target dimension, source-field-shape
  preservation, and the expected operational-only readiness state
- Readiness: the live audit report marks `downstream_contract_ready=true`,
  `eval_smoke_ready=true`, `transfer_smoke_ready=true`,
  `export_smoke_ready=true`, and `training_ready=false`
- Follow-up: if any later post-train consumer needs the fixed smoke bundle
  contract, read the audit verdict first instead of re-implementing surface-by-
  surface consistency checks

## 2026-04-07 - Reusable downstream contract audit proved on fixed NOD plus shared_private_smoke

- Scope: engineering, validation
- Status: completed
- Surfaces touched:
  `src/fmri2img/evaluation/decoder.py`,
  `src/fmri2img/workflows/_downstream_contract_audit.py`,
  `src/fmri2img/workflows/audit_public_nod_shared_only_downstream_contract.py`,
  `src/fmri2img/workflows/audit_shared_private_smoke_downstream_contract.py`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused `py_compile`, focused `.venv` pytest, local real
  `export_for_animus` plus real
  `audit_shared_private_smoke_downstream_contract` against the existing
  `outputs/canonical/hardening_smoke/` bundle; remote `git pull --rebase`,
  real rerun of the fixed NOD audit, real pod regeneration of the canonical
  `shared_private_smoke` train/eval/transfer/export bundle, real pod run of
  `audit_shared_private_smoke_downstream_contract`, and focused remote pytest
- Decision: downstream contract auditing is no longer trapped in the NOD-only
  workflow. The repo now has one reusable compact audit core plus two concrete
  consumers:
  `fmri2img.workflows.audit_public_nod_shared_only_downstream_contract` and
  `fmri2img.workflows.audit_shared_private_smoke_downstream_contract`
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the normalization helper now also understands legacy eval/transfer
  payloads that only expose `by_condition` plus `pair_metrics.n_pairs`, which
  lets the canonical shared-private smoke bundle emit a truthful normalized
  condition contract without changing paired semantics
- Readiness: the fixed NOD audit remains
  `downstream_contract_ready=true` with perception-only unavailable paired
  metrics explicit, and the new canonical shared-private smoke audit at
  `outputs/canonical/eval/shared_private_smoke/downstream_contract_audit.json`
  also marks `downstream_contract_ready=true` while `training_ready=false`
- Follow-up: if another post-train bundle needs contract freezing, reuse the
  internal audit core and keep the verdict shape compact instead of creating a
  bundle-specific schema

## 2026-04-07 - Generic downstream contract audit dispatcher proved on both supported bundle families

- Scope: engineering, validation
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/audit_downstream_contract.py`,
  `src/fmri2img/workflows/audit_public_nod_shared_only_downstream_contract.py`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused `py_compile`, focused `.venv` pytest, local real
  generic audit invocation against the existing `outputs/canonical/hardening_smoke/`
  bundle, remote `git pull --rebase`, real pod rerun of
  `./.venv/bin/python -m fmri2img.workflows.audit_downstream_contract --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`,
  real pod rerun of
  `./.venv/bin/python -m fmri2img.workflows.audit_downstream_contract --config configs/canonical/shared_private_smoke.yaml`,
  and focused remote pytest
- Decision: the repo now has one top-level canonical downstream audit entrypoint
  that dispatches by `experiment.name` to the supported concrete bundle families
  without changing their verdict shape or readiness semantics
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the dispatcher currently supports exactly
  `public_nod_imagenet_run10_shared_only_smoke` and `shared_private_smoke`.
  Unsupported configs now produce a truthful blocked report instead of an
  implicit or fake generic success
- Readiness: both supported live bundle families still mark
  `downstream_contract_ready=true` through the generic path; fixed NOD remains
  perception-only with explicit unavailable paired metrics, and
  `shared_private_smoke` remains paired with
  `paired_metrics_available=true`
- Follow-up: if another bundle family is added later, register it explicitly in
  the dispatcher only after a real concrete audit path exists

## 2026-04-07 - Generic downstream audit dispatcher now reads from an explicit registry

- Scope: engineering, validation
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/_downstream_contract_registry.py`,
  `src/fmri2img/workflows/audit_downstream_contract.py`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused `py_compile`, focused `.venv` pytest, local real
  generic audit invocation against the existing
  `outputs/canonical/hardening_smoke/` bundle, remote `git pull --rebase`,
  real pod rerun of the generic audit on both supported families, and focused
  remote pytest
- Decision: the top-level dispatcher no longer owns an ad hoc family mapping.
  Supported bundle-family registration now lives in one checked-in registry
  module close to the audit core, and the dispatcher resolves strictly through
  that registry
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the registry currently contains exactly two proven families:
  `public_nod_imagenet_run10_shared_only_smoke` and `shared_private_smoke`.
  Unsupported bundle names still return a truthful blocked report with the same
  stable top-level verdict shape
- Readiness: both supported live bundle families remain
  `downstream_contract_ready=true` through the registry-backed generic path;
  fixed NOD remains perception-only with explicit unavailable paired metrics,
  while `shared_private_smoke` remains paired with
  `paired_metrics_available=true`
- Follow-up: if another family is proven later, add one explicit registry entry
  only after its concrete audit path and real artifact proof exist

## 2026-04-07 - Generic downstream audit blocked reports now come from one shared helper

- Scope: engineering, validation
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/_downstream_contract_audit.py`,
  `src/fmri2img/workflows/audit_downstream_contract.py`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused `py_compile`, focused `.venv` pytest, local real
  generic audit invocation against the existing
  `outputs/canonical/hardening_smoke/` bundle, remote `git pull --rebase`,
  real pod rerun of the generic audit on both supported families, and focused
  remote pytest
- Decision: the generic dispatcher no longer owns a private blocked-report
  constructor. Unsupported-family and blocked generic-audit payloads now come
  from one shared helper in the downstream audit support layer
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the shared helper preserves the stable compact blocked verdict shape:
  empty `artifact_paths`, `target_spec`, `condition_semantics`, `identity`,
  and `consistency` blocks; state defaults of
  `downstream_contract_ready=false`, `eval_smoke_ready=false`,
  `transfer_smoke_ready=false`, `export_smoke_ready=false`, and
  `training_ready=false`; plus explicit `blocked_reasons` and
  `operational_boundary`
- Readiness: both supported live bundle families still mark
  `downstream_contract_ready=true` through the generic path; unsupported
  families remain truthfully blocked with the same top-level contract shape
- Follow-up: if concrete bundle-specific blocked paths need the same top-level
  payload later, route them through the shared helper instead of adding another
  local constructor

## 2026-04-07 - Generic downstream blocked-path boundary defaults now live in the shared audit support layer

- Scope: engineering, validation
- Status: completed
- Surfaces touched:
  `src/fmri2img/workflows/_downstream_contract_audit.py`,
  `src/fmri2img/workflows/audit_downstream_contract.py`,
  `tests/test_canonical_workflows.py`,
  `Documentation.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/PROJECT_MASTER_LOG.md`
- Validation: local focused `py_compile`, focused `.venv` pytest, local real
  generic audit invocation against the existing
  `outputs/canonical/hardening_smoke/` bundle, remote `git pull --rebase`,
  real pod rerun of the generic audit on both supported families, and focused
  remote pytest
- Decision: the shared blocked-report helper now owns the generic blocked-path
  `operational_boundary` defaults. The top-level dispatcher no longer embeds
  those strings directly
- Claim boundary: operational contract hardening only; no benchmark progress,
  no evidence-freeze change, and `training_ready` remains `false`
- Detail: the shared support layer now exposes one explicit default generic
  blocked boundary describing registered-family-only support, truthful blocked
  reporting for unsupported configs, and preserved `training_ready=false`
  semantics. The stable compact blocked-report top-level shape is unchanged
- Readiness: both supported live bundle families still mark
  `downstream_contract_ready=true` through the generic path, while unsupported
  bundle names remain truthfully blocked with the same compact verdict shape
- Follow-up: if the concrete bundle-specific blocked paths ever need to align
  with the same top-level shape, route them through the same shared helper
  rather than duplicating one more constructor
