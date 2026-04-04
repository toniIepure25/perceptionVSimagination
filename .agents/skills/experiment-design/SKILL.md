---
name: experiment-design
description: "Use when converting a repository-specific question into an executable plan for the canonical perception-vs-imagery platform, including which configs to use, which controls belong in the fixed ladder, what artifacts must be produced, and how to keep Animus subsystem work separate from paper-claim work."
---

# Experiment Design

Use this skill after a question has been scoped and before code changes or long runs begin.

## Inputs

- the exact question being asked
- the primary lane
- any fixed constraints on data, targets, compute, or claims
- the canonical config most likely to anchor the plan

## Read First

1. `START_HERE.md`
2. `docs/ARCHITECTURE.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/VALIDATION.md`
5. `docs/CURRENT_EVIDENCE_FREEZE.md`
6. `docs/PAPER_1_CLAIMS_MAP.md`
7. `docs/BENCHMARK_LADDER.md`
8. Relevant config(s) under `configs/canonical/`

## Planning Workflow

1. Declare the lane:
   - `Animus subsystem engineering`
   - `Threshold research`
   - `Data acquisition / benchmark expansion`
2. Fix the comparison surface before proposing changes.
3. Name the exact config baseline.
4. Specify the minimal controlled overrides.
5. Define required artifacts, validation commands from the project `.venv`, and docs updates.

## Default Baselines

- Practical Animus lane: `configs/canonical/animus_core_decoder.yaml`
- Real bootstrap baseline: `configs/canonical/multisubj_overlap_bootstrap.yaml`
- Current fixed ladder dataset: `configs/canonical/max_available_overlap.yaml`
- Primary exploratory threshold model: `configs/canonical/threshold_shared_private_p16.yaml`
- Workflow sanity only: `configs/canonical/shared_private_smoke.yaml`

## Design Rules

- Keep `vit_l14_image_768` fixed unless the task is explicitly about target-space changes.
- For threshold work, preserve the ladder order and compare against:
  1. Ridge
  2. shared-only
  3. shared-private `private_dim=16`
- For Animus work, optimize for stable content decoding, export, and subsystem clarity; do not smuggle in new scientific claims.
- Change one scientific degree of freedom at a time.
- If the proposal only diagnoses underperformance, label it `diagnostic`, not `headline benchmark`.
- If a proposed success condition would exceed `docs/PAPER_1_CLAIMS_MAP.md`, restate it as future evidence required rather than as a current claim.

## What To Output

Produce a compact plan containing:

- question and hypothesis
- primary config and exact command set
- controls/baselines
- expected output paths
- pass/fail criteria
- readiness expectation: `smoke_only`, `bootstrap_ready`, `paper_ready`, or `blocked`
- required tests or smoke checks
- docs to update after the milestone
- whether the outcome would affect `docs/CURRENT_EVIDENCE_FREEZE.md` or only a narrower run report

## Handoff

- hand off to `ablation-runner` for execution
- hand off to `paper-drafter` only after real results exist
- hand off to `repro-auditor` if the main question is rerunnability rather than design

Normal sequence: `research-scout` -> `experiment-design` -> `ablation-runner`.

## Boundaries

- Do not run the plan here.
- Do not use legacy `scripts/` surfaces unless the user explicitly wants historical reproduction.
- Do not promote shared-private over shared-only unless the design includes a fair larger-overlap test against Ridge.
- Do not write manuscript conclusions here; hand off to `paper-drafter` after results exist.
