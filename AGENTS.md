# Repository Workflow Contract

This file is the Codex operating contract for `perceptionVSimagination`.
Optimize for minimal diffs, strong reproducibility, and clear separation
between reusable engineering and scientific interpretation.

## What This File Is For

- `AGENTS.md`: the default operating contract for how work should be done here
- `PLANS.md`: multi-session strategy, experiment programs, and active run
  matrices
- `Documentation.md`: current working notes, decisions, and follow-up items
- `docs/EXPERIMENT_REGISTRY.md`: compact durable ledger of important runs
- `docs/PROJECT_MASTER_LOG.md`: durable project memory and official milestones

If two files seem to overlap, prefer the narrower one. Do not turn
`Documentation.md` into the long-term archive, and do not turn
`docs/PROJECT_MASTER_LOG.md` into a session transcript.

## Working agreements

- Plan first for complex tasks or anything that changes experiments, docs
  structure, or workflow surfaces.
- Prefer the smallest viable edit. Reuse the canonical workflow/config surface
  instead of introducing new entrypoints.
- Do not add dependencies unless the current stack cannot support the change.
- After code changes, run the smallest relevant validation command.
- Be explicit about assumptions, risks, and unresolved issues.
- Keep scientific claims separate from engineering changes and doc updates.

## Repository map

- `src/fmri2img/` is the official implementation surface.
- `src/fmri2img/workflows/` is the canonical CLI/workflow namespace.
- `configs/canonical/` is the checked-in benchmark and subsystem config set.
- `tests/` covers canonical workflows, prep, model behavior, and docs surface.
- `docs/` is the source of truth for architecture, validation,
  reproducibility, evidence boundaries, benchmark status, and paper drafting.
- `legacy/` and older `scripts/` surfaces are historical unless a task
  explicitly asks for reproduction of older behavior.

## Default lanes

Classify work into one primary lane before editing:

- `Animus subsystem engineering`
  Shared-only practical decoder, export, preflight, ROI materialization,
  downstream-ready metadata, and stable command surfaces.
- `Threshold research`
  Controlled shared-private experiments against the fixed ladder on paired
  perception/imagery data.
- `Data acquisition / benchmark expansion`
  Larger overlapable paired data, new rebuildable sources, and benchmark-scope
  expansion.
- `Paper writing`
  Manuscript-facing text that must stay inside the current evidence freeze.

Do not merge these lanes into one narrative. Engineering hardening does not
justify new empirical claims.

## Daily Loop

Use this default operating rhythm unless the task clearly needs something else:

1. Classify the lane and identify the canonical baseline config or doc surface.
2. If the work spans sessions or multiple runs, create or update an entry in
   `PLANS.md`.
3. Do the smallest useful edit or run.
4. Validate from the project `.venv`, not from system Python.
5. Record the local outcome and next action in `Documentation.md`.
6. If a run or experiment is durable enough to matter later, append it to
   `docs/EXPERIMENT_REGISTRY.md`.
7. If the change is a real milestone or durable repo-state change, append a
   concise entry to `docs/PROJECT_MASTER_LOG.md`.

Default command style:

- `source .venv/bin/activate`
- or call tools explicitly through `./.venv/bin/python`, `./.venv/bin/pytest`,
  `./.venv/bin/ruff`

## Canonical commands

Prefer these commands and configs:

- Smoke sanity:
  `python -m fmri2img.workflows.train_decoder --config configs/canonical/shared_private_smoke.yaml`
- Practical Animus lane:
  `python -m fmri2img.workflows.preflight_animus_core_decoder`
  `python -m fmri2img.workflows.train_animus_core_decoder`
  `python -m fmri2img.workflows.eval_animus_core_decoder --checkpoint ...`
  `python -m fmri2img.workflows.export_animus_core_decoder --checkpoint ...`
- Bootstrap paired baseline:
  `python -m fmri2img.workflows.prepare_overlap_bootstrap --config configs/canonical/multisubj_overlap_bootstrap.yaml`
  `python -m fmri2img.workflows.preflight_data --config configs/canonical/multisubj_overlap_bootstrap.yaml`
  `python -m fmri2img.workflows.train_decoder --config configs/canonical/multisubj_overlap_bootstrap.yaml`
- Fixed ladder / threshold lane:
  `python -m fmri2img.workflows.run_legacy_ridge_baseline --config configs/canonical/max_available_overlap.yaml`
  `python -m fmri2img.workflows.train_decoder --config configs/canonical/threshold_shared_private_p16.yaml`
- Common eval/export:
  `python -m fmri2img.workflows.eval_decoder --config ... --checkpoint ...`
  `python -m fmri2img.workflows.eval_transfer --config ... --checkpoint ...`
  `python -m fmri2img.workflows.run_analysis --config ... --checkpoint ...`
  `python -m fmri2img.workflows.export_for_animus --config ... --checkpoint ...`

Use `--override` for small controlled changes. Avoid copying configs unless the
variant needs a stable checked-in contract.

Run canonical validation and workflows from the project `.venv`.

## Default baselines

- Practical subsystem work: `configs/canonical/animus_core_decoder.yaml`
- Smoke-only validation: `configs/canonical/shared_private_smoke.yaml`
- Real bootstrap baseline: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- Current fixed ladder dataset: `configs/canonical/max_available_overlap.yaml`
- Primary threshold hypothesis run: `configs/canonical/threshold_shared_private_p16.yaml`

Keep `vit_l14_image_768` fixed unless the task explicitly studies target-space
changes.

## Evidence and claim boundary

Read these before experiment design, paper drafting, or benchmark reframing:

- `docs/CURRENT_EVIDENCE_FREEZE.md`
- `docs/BENCHMARK_LADDER.md`
- `docs/PAPER_1_CLAIMS_MAP.md`
- `docs/REPRODUCIBILITY.md`
- `docs/VALIDATION.md`

Current frozen interpretation:

- Ridge is the strongest overall baseline on the fixed paired benchmark.
- Shared-only is the best current canonical neural baseline.
- Shared-private is the hypothesis family, not the current performance leader.
- `private_dim=16` is the strongest current shared-private variant, but remains
  exploratory.

Do not let README, comments, config names, or paper text outrun those facts.

## Planning and logging

- Use `PLANS.md` for long-horizon experiment/program planning.
- Use `Documentation.md` for milestone logging, decisions, and follow-up
  actions during active work.
- Use `docs/EXPERIMENT_REGISTRY.md` for compact durable experiment/run entries.
- Always append meaningful milestones to `docs/PROJECT_MASTER_LOG.md`.

Separation rule:

- `PLANS.md` answers: what are we trying to do over the next several steps?
- `Documentation.md` answers: what happened in this working pass and what is
  next?
- `docs/EXPERIMENT_REGISTRY.md` answers: which durable runs matter and where
  their artifacts live?
- `docs/PROJECT_MASTER_LOG.md` answers: what durable repo-level milestone just
  became true?

Update doc surfaces by change type:

- Workflow or command changes:
  `START_HERE.md`, `docs/REPRODUCIBILITY.md`, `docs/VALIDATION.md`,
  `tests/test_canonical_workflows.py`
- Architecture or data-contract changes:
  `docs/ARCHITECTURE.md`, `docs/CURRENT_STATE.md`, `docs/VALIDATION.md`
- New experiment results:
  run report and `docs/EXPERIMENT_REGISTRY.md` first, then
  `docs/REPRODUCIBILITY.md`, then only update
  `docs/CURRENT_EVIDENCE_FREEZE.md`, `docs/BENCHMARK_LADDER.md`, and
  `docs/PAPER_1_CLAIMS_MAP.md` if the evidence boundary genuinely changed
- Manuscript/public narrative changes:
  `docs/paper1/`, `docs/PAPER_1_*`, and finally `README.md` after the evidence
  docs are aligned

## Reproducibility contract

- Canonical commands, tests, and validations must run from the project `.venv`.
- Canonical workflow entrypoints fail fast outside the project `.venv` with an
  actionable message.
- Run `preflight_data` before real training on canonical data paths.
- Respect readiness labels: `smoke_only`, `bootstrap_ready`, `paper_ready`,
  `blocked`.
- Preserve dataset scope, seed policy, target space, and evaluation surface
  unless the task is explicitly about changing them.
- Expected real-run artifacts include source reports, prep reports,
  `preflight.json`, `config_snapshot.json`, `roi_summary.json`,
  `target_summary.json`, `train_history.json`, `best_decoder.pt`, eval metrics,
  and export manifests.
- Do not mix 512-D and 768-D target caches.
- Do not present `bootstrap_ready` output as paper-confirmatory evidence.

## Skills

Repo-specific skills live under `.agents/skills/`:

- `research-scout`: decide whether an idea belongs to Animus engineering,
  threshold research, or data acquisition
- `experiment-design`: turn a scoped question into an executable plan
- `ablation-runner`: execute controlled variants against the fixed ladder
- `repro-auditor`: verify whether runs, artifacts, or claims are genuinely
  rerunnable
- `paper-drafter`: write manuscript-facing text constrained by the evidence
  freeze

Use the skill that matches the task. Chain them in that order when the job
spans strategy, execution, auditing, and paper updates.

Default orchestration:

- idea triage: `research-scout`
- executable plan: `experiment-design`
- controlled run: `ablation-runner`
- artifact/claim audit: `repro-auditor`
- manuscript-facing update: `paper-drafter`

Skip or reorder only when the task is already downstream:

- skip `research-scout` when the question is already fixed
- skip `experiment-design` for a trivial one-file engineering change
- go straight to `repro-auditor` when the issue is rerunnability or claim safety
- go straight to `paper-drafter` only when no new evidence or rerun decision is needed

The normal handoff is narrow and directional: do not use `paper-drafter` to
invent evidence, and do not use `ablation-runner` to redesign the research question.

## Optional Worktrees

If parallel lane work starts causing branch clutter, use separate git worktrees
by lane. Keep this optional and lightweight:

- `animus-core`
- `threshold-research`
- `paper-lane`
- `data-acquisition`

Use worktrees only when they reduce collisions between engineering, runs, and
paper edits. Do not create process overhead for single-lane work.

## Validation expectations

Use existing targets first:

- `make lint`
- `make test`
- focused checks such as `pytest tests/test_canonical_workflows.py -v`

For docs-only edits, run the smallest test or check that validates the command
surface or referenced workflow behavior when such a test exists.
