# Paper 1 Figures And Tables

This document plans the core visuals for the current realistic paper path.

## Figure principles

- every figure should reinforce the frozen benchmark ladder
- no figure should imply that shared-private already wins
- the visuals should make the overlap-scarcity regime legible
- the paper should separate:
  - external reference baseline
  - practical Animus lane
  - exploratory threshold lane

## Figure 1. Platform and benchmark overview

### Purpose

Show the canonical ROI-first pipeline and the three benchmark roles:

- Ridge
- shared-only Animus Core Decoder
- shared-private `private_dim=16`

### Panels

- data flow: perception index + imagery index -> overlap assembly -> ROI
  materialization -> fixed target space
- model ladder: Ridge vs shared-only vs shared-private
- output paths: eval / transfer / export

### Status

- ready to draft conceptually now

## Figure 2. Data-regime / overlap-ceiling figure

### Purpose

Make the current scarcity regime visually undeniable.

### Content

- public NSD-Imagery acquisition -> canonical overlap ceiling
- subjects contributing overlap: `subj02`, `subj03`, `subj05`, `subj07`
- benchmark size:
  - `94` rows
  - `5` shared paired ids
  - `1` held-out paired group

### Status

- ready now from current reports

## Figure 3. Benchmark ladder results

### Purpose

Show the frozen ordering on the current benchmark.

### Content

- bar plot or dot plot for content cosine
- secondary panel for MSE
- order:
  - Ridge
  - shared-only
  - shared-private `private_dim=16`
  - optional diagnostic controls in lighter style

### Status

- ready now

## Figure 4. Shared-only vs shared-private family comparison

### Purpose

Show that:

- shared-only is the best canonical neural baseline
- private-capacity reduction helps shared-private somewhat
- but does not beat shared-only

### Content

- shared-only
- shared-private
- shared-private `private_dim=16`
- shared-private `private_dim=8`
- shared-private no-domain

### Status

- ready now

## Figure 5. Threshold hypothesis conceptual figure

### Purpose

Bridge the current paper to the next stronger paper path.

### Content

- x-axis: paired overlap scale
- y-axis: expected relative benefit of shared-private over shared-only
- low-overlap region:
  - Ridge / shared-only favored
- transition region:
  - hypothesis zone
- larger-overlap region:
  - potential shared-private benefit

### Status

- conceptual only for Paper 1
- should be clearly labeled as a hypothesis figure, not an empirical curve

## Figure 6. Animus subsystem role figure

### Purpose

Clarify the repo’s dual identity.

### Content

- shared-only -> Animus Core Decoder lane
- shared-private `private_dim=16` -> threshold-testing research lane
- Ridge -> external scientific reference

### Status

- optional
- include only if it improves clarity rather than adding presentation overhead

## Table 1. Frozen benchmark ladder

### Columns

- rung
- model
- role
- config
- status
- current metric summary

### Status

- ready now

## Table 2. Main benchmark results

### Columns

- model
- cosine
- MSE
- imagery mean
- perception mean
- paired eval groups
- notes

### Status

- ready now

## Table 3. Evidence boundary table

### Columns

- question
- answer supported now?
- evidence level
- what is missing

### Purpose

Translate the claims map into a paper-friendly summary.

### Status

- ready now

## Table 4. Reproducibility / artifact contract table

### Columns

- workflow stage
- canonical command
- main config
- main artifacts

### Purpose

Support the benchmark/resource angle of the paper.

### Status

- ready now

## Appendix figures

Possible appendix visuals:

- train-history traces for shared-only vs shared-private
- per-condition cosine breakdown
- example export-manifest schema for the Animus Core Decoder
- preflight/readiness summary

## Data sources for figures and tables

- [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md)
- [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md)
- [EXPANDED_OVERLAP_COMPARISON.md](EXPANDED_OVERLAP_COMPARISON.md)
- [PROJECT_MASTER_LOG.md](PROJECT_MASTER_LOG.md)
- `outputs/canonical/*`
- `outputs/animus/core_decoder/*`
- `outputs/research/threshold_shared_private_p16/*`
