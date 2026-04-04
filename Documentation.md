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
