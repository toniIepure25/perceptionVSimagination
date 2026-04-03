# Benchmark Ladder

This document defines the official benchmark ladder for the current
perception/imagery decoding program.

## Purpose

The ladder separates:

- external reference baselines
- current canonical neural baselines
- exploratory hypothesis models
- diagnostic controls

The goal is to prevent benchmark drift and to make the current evidence state
immediately legible to new researchers.

## Official dataset for the current ladder

Current fixed dataset:

- config base: [`configs/canonical/max_available_overlap.yaml`](../configs/canonical/max_available_overlap.yaml)
- prepared mixed index:
  `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired groups: `1`

## Ladder rungs

### R0. Workflow sanity

Purpose:

- verify the platform, not model quality

Command:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
```

Status:

- operational validation only

### R1. External low-data reference baseline

Model:

- Ridge on canonical ROI values

Command:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
```

Role:

- strongest current external low-data reference baseline

Current result:

- cosine `0.55199`
- MSE `0.001167`

### R2. Best current canonical neural baseline

Model:

- Animus Core Decoder, implemented as canonical shared-only

Command:

```bash
python -m fmri2img.workflows.train_animus_core_decoder
```

Role:

- default practical neural subsystem and best current canonical neural baseline

Official config:

- `configs/canonical/animus_core_decoder.yaml`

Current result:

- cosine `0.13596`
- MSE `0.002250`

### R3. Best current shared-private exploratory variant

Model:

- threshold-testing shared-private, reduced private capacity

Command:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

Role:

- strongest shared-private variant tested so far

Official config:

- `configs/canonical/threshold_shared_private_p16.yaml`

Current result:

- cosine `0.10784`
- MSE `0.002323`

### R4. Canonical shared-private baseline

Model:

- canonical shared-private

Command:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/max_available_overlap.yaml
```

Role:

- main research-hypothesis family baseline

Current result:

- cosine `0.06927`
- MSE `0.002424`

### R5. Diagnostic controls

Examples:

- shared-private with domain head disabled
- narrow private-capacity sweep around the best exploratory range

Role:

- diagnose why shared-private underperforms
- not the main benchmark target

## Current benchmark ordering

1. Ridge
2. Shared-only
3. Shared-private, `private_dim=16`
4. Shared-private, `private_dim=8`
5. Shared-private
6. Shared-private, no domain head

## Decision policy

Use the ladder this way:

- Compare new neural models to Ridge first.
- Compare any shared-private variant to the Animus Core Decoder second.
- Treat shared-private as exploratory until it beats shared-only on a materially
  larger overlap benchmark.
- Do not introduce new benchmark rungs unless they answer a sharply defined
  scientific question.

## Promotion rules

Promote a model to a higher rung only if:

- it improves content cosine and MSE on the same fixed benchmark
- the gain is reproducible across reruns or larger overlap datasets
- the result changes the scientific interpretation rather than only a single
  training trace

## Rerun rule when new data arrives

When a materially larger paired source is acquired, do not change the ladder.
Rebuild the prepared overlap dataset and rerun:

1. `configs/canonical/max_available_overlap.yaml` for Ridge and shared data prep
2. `configs/canonical/animus_core_decoder.yaml` for the practical subsystem lane
3. `configs/canonical/threshold_shared_private_p16.yaml` for the exploratory threshold lane

## Related documents

- [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md)
- [EXPANDED_OVERLAP_COMPARISON.md](EXPANDED_OVERLAP_COMPARISON.md)
- [TOP_LEVEL_RESEARCH_DOSSIER.md](TOP_LEVEL_RESEARCH_DOSSIER.md)
