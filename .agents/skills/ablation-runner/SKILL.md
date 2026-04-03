---
name: ablation-runner
description: "Use when executing controlled model or workflow variants in this repository, especially canonical shared-only/shared-private comparisons, private-capacity sweeps, domain-head diagnostics, or prep/eval reruns that must stay anchored to the fixed benchmark ladder and the repo’s reproducibility artifact contract."
---

# Ablation Runner

Use this skill for execution, not brainstorming. The goal is to run a controlled variant and report what changed without causing benchmark drift.

## Inputs

- a concrete experiment question
- one checked-in baseline config
- the minimal override set
- the expected output directory or artifact surface

## Read First

1. `docs/BENCHMARK_LADDER.md`
2. `docs/CURRENT_EVIDENCE_FREEZE.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/VALIDATION.md`
5. The chosen canonical config in `configs/canonical/`

## Canonical Execution Surface

Prefer these commands:

- `python -m fmri2img.workflows.preflight_data --config ...`
- `python -m fmri2img.workflows.train_decoder --config ... --override ...`
- `python -m fmri2img.workflows.eval_decoder --config ... --checkpoint ...`
- `python -m fmri2img.workflows.eval_transfer --config ... --checkpoint ...`
- `python -m fmri2img.workflows.run_analysis --config ... --checkpoint ...`
- `python -m fmri2img.workflows.export_for_animus --config ... --checkpoint ...`
- `python -m fmri2img.workflows.run_legacy_ridge_baseline --config ...`

Use `train_animus_core_decoder` only for the practical shared-only lane.

## Baseline Choices

- Practical subsystem ablations: start from `configs/canonical/animus_core_decoder.yaml`
- Threshold ablations: start from `configs/canonical/threshold_shared_private_p16.yaml`
- Full current-ladder comparisons: start from `configs/canonical/max_available_overlap.yaml`
- Bootstrap reproducibility work: start from `configs/canonical/multisubj_overlap_bootstrap.yaml`

## Execution Rules

- Change one variable at a time.
- Keep dataset scope, target space, and evaluation surface fixed unless the run is explicitly a new benchmark-construction task.
- Use `--override` for small controlled changes instead of cloning configs prematurely.
- Always capture the resulting `config_snapshot.json`, `train_history.json`, checkpoint, and evaluation metrics.
- If `preflight_data` says `blocked`, stop and report missing artifacts instead of improvising.
- If the run is only `bootstrap_ready`, report it as operational or exploratory, not paper-confirmatory.

## Legacy Boundary

`src/fmri2img/eval/ablation_driver.py` and `configs/experiments/ablation.yaml` belong to the older two-stage/feature-training surface. Use them only when the user explicitly asks for that legacy path. They are not the default for the current canonical ladder.

## Result Summary Format

For each run, report:

- baseline config
- overrides
- output directory
- key metrics versus baseline
- whether the run is `validated`, `exploratory`, or `diagnostic`
- which docs need updates if the result matters
- whether `docs/PROJECT_MASTER_LOG.md` needs a new milestone entry

## Handoff

- hand off to `repro-auditor` if artifact or readiness status is uncertain
- hand off to `paper-drafter` only after results are logged and claim-safe

## Boundaries

- Do not rewrite the evidence freeze during execution.
- Do not call a diagnostic sweep a new benchmark rung.
- Do not collapse the practical Animus lane and the shared-private research lane into one narrative.
