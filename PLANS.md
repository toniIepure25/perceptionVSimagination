# Long-Horizon Plans

Use this file for multi-step work that spans more than one session, more than
one run, or more than one lane. It is the forward plan, not the evidence log.

## What Does Not Belong Here

Do not use this file for:

- raw session notes or command transcripts
- permanent milestone history
- paper prose drafts
- evidence-freeze updates written before results exist

Execution outcomes can be linked here, but the detailed local note belongs in
`Documentation.md`, the compact durable run record belongs in
`docs/EXPERIMENT_REGISTRY.md`, and the durable milestone belongs in
`docs/PROJECT_MASTER_LOG.md`.

## When to use this file

Create or refresh a plan when work involves:

- a new experiment matrix or ablation campaign
- a benchmark-ladder rerun or benchmark-expansion effort
- a subsystem refactor that changes workflow contracts or artifacts
- paper package work that depends on upstream runs, figures, or evidence
  reviews

If the task is single-session and local, update `Documentation.md` instead.

## Planning rules

- Pick one primary lane:
  `Animus subsystem engineering`, `Threshold research`,
  `Data acquisition / benchmark expansion`, or `Paper writing`
- Anchor every plan to an existing checked-in config or doc surface.
- Record dependencies before proposed execution.
- Separate engineering deliverables from empirical claims.
- State the decision gate that would justify updating evidence-facing docs.

## Active plan template

Copy this block for each active initiative:

```md
## Plan: <short title>

- Lane: <one primary lane>
- Status: proposed | active | blocked | completed
- Owner: Codex / user / shared
- Baseline surface: <config, workflow, or doc set>
- Why now: <one short paragraph>

### Question

<the exact question this plan answers>

### Success condition

- <engineering outcome or empirical decision criterion>

### Constraints

- <fixed benchmark, target space, subject scope, claim boundary, compute limit>

### Steps

1. <smallest reversible step>
2. <next dependency-aware step>
3. <validation or audit step>
4. <doc update step>

### Required artifacts

- <output path or report>
- <checkpoint / metrics / figure / memo>

### Run ledger

- YYYY-MM-DD | <config> | <override summary> | <output path> | planned | running | blocked | done

### Validation

- <smallest test or command>

### Docs to touch

- <specific docs if the plan succeeds>

### Decision gate

- Evidence docs move only if <explicit condition>
- Otherwise update only <run report / Documentation.md / master log>

### Risks / blockers

- <specific blocker or misinterpretation risk>
```

## Standard planning lanes

### Animus subsystem engineering

Use when the goal is a better practical shared-only subsystem, clearer export,
or more stable prep/eval behavior.

Default baseline:
- `configs/canonical/animus_core_decoder.yaml`

Expected outputs:
- workflow/test changes
- validated command surface
- export or metadata improvements

Do not turn this lane into paper-claim inflation.

### Threshold research

Use when testing whether shared-private improves under a fairer comparison.

Default baselines:
- `configs/canonical/max_available_overlap.yaml`
- `configs/canonical/threshold_shared_private_p16.yaml`

Required controls:
1. Ridge
2. shared-only
3. shared-private `private_dim=16`

Evidence rule:
- If the result is still tiny-overlap or `bootstrap_ready`, keep it out of the
  evidence freeze unless it changes an already documented operational fact.

### Data acquisition / benchmark expansion

Use when the real bottleneck is paired-data scale rather than model design.

Start from:
- `docs/DATA_ACQUISITION_PROGRAM.md`
- `docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`

Required outputs:
- source audit
- rebuildability path
- expected benchmark delta

### Paper writing

Use when drafting text, figures, or tables that depend on the frozen benchmark
state.

Start from:
- `docs/CURRENT_EVIDENCE_FREEZE.md`
- `docs/PAPER_1_CLAIMS_MAP.md`
- `docs/paper1/`

Do not open a paper plan until the upstream experiment or audit inputs are
already identified.

## Current repo planning heuristics

- Prefer explaining underperformance before proposing new model families.
- Prefer benchmark expansion over repeated micro-optimizations on `94` rows /
  `5` shared paired `nsdId`s / `1` held-out paired group.
- Prefer shared-only improvements for Animus-facing subsystem work.
- Treat shared-private wins as unproven until they beat shared-only on a fair,
  fixed comparison surface.
- Keep canonical commands and validation anchored to the project `.venv`.
