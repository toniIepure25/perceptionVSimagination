# Paper 1 Outline

## Working paper identity

Current realistic paper path:

- an honest benchmark and evidence paper under overlap scarcity
- fixed ladder:
  - Ridge
  - shared-only
  - shared-private `private_dim=16`
- ROI-first canonical multi-subject platform
- threshold hypothesis motivated, but not yet confirmed

This paper should be written as a rigorous benchmark/resource/evidence paper,
not as a “shared-private wins” paper.

## Title candidates

1. `Perception-to-Imagery Decoding Under Overlap Scarcity: A Reproducible ROI-First Benchmark`
2. `Benchmarking Perception and Mental Imagery Decoding Under Extreme Paired-Data Scarcity`
3. `Shared-Only Beats Disentanglement Under Low Overlap: An ROI-First Benchmark for Perception-to-Imagery fMRI Decoding`
4. `A Reproducible Benchmark Ladder for Perception-to-Imagery fMRI Decoding Under Overlap Scarcity`
5. `From Practical Decoder to Threshold Hypothesis: A Benchmark for Perception-to-Imagery fMRI Decoding`

Recommended working title:

- `Perception-to-Imagery Decoding Under Overlap Scarcity: A Reproducible ROI-First Benchmark`

## One-sentence thesis

When paired perception/imagery overlap is extremely limited, a disciplined
ROI-first benchmark shows that simpler decoders outperform explicit
shared/private disentanglement, motivating a threshold hypothesis rather than
an assumed disentanglement win.

## Contribution statement

The paper should claim four contributions:

1. A reproducible ROI-first platform for multi-subject perception/imagery
   decoding with canonical preparation, preflight, training, evaluation, and
   export.
2. An honest fixed benchmark ladder separating a linear baseline, a practical
   shared-only neural decoder, and exploratory shared-private variants.
3. An empirical low-overlap result: Ridge wins strongly and shared-only is the
   best current canonical neural model.
4. A threshold-style research hypothesis: explicit disentanglement may require
   materially larger paired overlap before it helps.

## Abstract skeleton

### Sentence plan

1. Problem:
   Perception-to-imagery decoding is scientifically important, but paired
   imagery/perception benchmarks are often constrained by scarce overlap.
2. Gap:
   Existing work tends to assume richer paired structure or emphasizes
   reconstruction quality without clearly separating baseline strength from
   disentanglement claims.
3. Method/platform:
   We present a reproducible ROI-first multi-subject platform for canonical
   preparation, benchmarking, transfer evaluation, and export.
4. Benchmark:
   We evaluate a fixed ladder consisting of Ridge, a shared-only neural
   decoder, and shared-private variants on the current max-available real
   overlap dataset.
5. Main finding:
   In this extreme low-overlap regime, Ridge is strongest and shared-only
   outperforms all tested shared-private variants.
6. Interpretation:
   These results support an overlap-scarcity benchmark framing and motivate a
   threshold hypothesis for when explicit disentanglement may begin to help.
7. Significance:
   The platform serves both as a practical shared-only decoder for Animus and
   as a disciplined research testbed for future threshold studies.

### Abstract fill-in slots

- dataset size:
  - `94` rows
  - `5` shared paired `nsdId`s
  - subjects: `subj02`, `subj03`, `subj05`, `subj07`
- current ordering:
  - Ridge
  - shared-only
  - shared-private `private_dim=16`
- caution:
  - held-out paired evaluation remains small

## Section-by-section outline

## 1. Introduction

### Goal

Frame the problem as a benchmarked scientific question:

- how should we decode shared content between perception and imagery under real
  paired-data scarcity?
- when, if ever, is explicit shared/private disentanglement justified?

### Subsections

1. Why perception vs imagery matters
2. Why low-overlap paired data is a central bottleneck
3. Why benchmark discipline matters more than ad hoc model complexity here
4. Our approach and paper scope

### Key introduction claims

- current paired-data scarcity is not a nuisance detail; it changes which model
  classes are justified
- practical and exploratory lanes should be separated
- ROI-first canonicalization is necessary for multi-subject validity

## 2. Related Work

### Structure

1. fMRI decoding and reconstruction baselines
2. Mental imagery and perception/imagery transfer
3. Disentanglement and domain-separated neural modeling
4. Multi-subject fMRI normalization / ROI aggregation
5. Benchmarking and negative-result rigor in neuro-AI

### Writing stance

- do not frame this as “we beat the field”
- frame it as “we build the missing benchmark structure and identify a
  scientifically important low-overlap regime”

## 3. Platform And Methods

### 3.1 Canonical platform

- canonical preparation surface
- canonical preflight
- canonical train/eval/transfer/export workflows
- ROI-first multi-subject batching contract

### 3.2 Model ladder

- Ridge as external low-data reference
- shared-only as best current canonical neural baseline
- shared-private family as threshold-testing hypothesis family
- `private_dim=16` as the best current exploratory shared-private variant

### 3.3 Input and target representation

- ROI-group materialization
- target space: `vit_l14_image_768`
- why this target space is fixed across the ladder

### 3.4 Evaluation surface

- content cosine
- content MSE
- imagery/perception means
- transfer evaluation
- paired-group counts
- domain accuracy as secondary diagnostic only

## 4. Dataset And Benchmark Construction

### 4.1 Data sources

- NSD perception indices
- full public NSD-Imagery source
- canonical overlap assembly

### 4.2 Real benchmark ceiling

- public source yields only `5` shared paired ids in the current accessible
  regime
- explain why that ceiling matters scientifically

### 4.3 Trust and reproducibility

- artifact provenance
- preflight status
- benchmark ladder freeze

## 5. Experiments

### 5.1 Primary benchmark

- Ridge
- shared-only
- shared-private `private_dim=16`

### 5.2 Diagnostic controls

- shared-private baseline
- shared-private `private_dim=8`
- shared-private no-domain

These should be framed as interpretive controls, not headline models.

### 5.3 Implementation notes

- CPU-safe or device-safe execution details
- fixed benchmark settings
- what was kept constant

## 6. Results

### 6.1 Main ladder result

Report the frozen ordering:

1. Ridge
2. shared-only
3. shared-private `private_dim=16`

### 6.2 Shared-private recovery evidence

- reduced private capacity helps relative to the default shared-private model
- but still does not beat shared-only

### 6.3 Interpretation under overlap scarcity

- current data favors simpler decoders
- present evidence supports the threshold hypothesis as a motivation, not a
  confirmed finding

## 7. Discussion

### Core discussion points

- what the paper really shows
- why negative or non-confirmatory evidence matters here
- why shared-only can already serve as the practical Animus Core Decoder
- why the threshold hypothesis remains worth testing

### Discussion boundaries

- do not overinterpret domain accuracy
- do not claim neuroscientific meaning for private latents yet
- do not imply vividness/reality-monitoring findings

## 8. Limitations

- only `5` shared paired ids
- only `1` held-out paired evaluation group
- public imagery source already exhausted for the current benchmark
- threshold hypothesis is motivated, not confirmed
- current ROI story is operationally sufficient but not yet a full
  neuroscientific localization result

## 9. Future Work

### Near-term

- acquire larger paired data and rerun the fixed ladder

### Mid-term

- overlap-threshold study
- ROI decomposition study after larger overlap exists

### Longer-term

- Animus-facing source/confidence interfaces
- vividness / confidence / subjective reality

## 10. Appendix plan

- exact configs
- preflight outputs
- artifact contract
- extra diagnostic results
- canonical command list

## Writing order recommendation

Write in this order:

1. Methods and benchmark construction
2. Results
3. Claims/evidence table
4. Introduction
5. Related work
6. Discussion and limitations
7. Abstract

## Required source artifacts for drafting

- [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md)
- [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md)
- [EXPANDED_OVERLAP_COMPARISON.md](EXPANDED_OVERLAP_COMPARISON.md)
- [REAL_BOOTSTRAP_REPORT.md](REAL_BOOTSTRAP_REPORT.md)
- [PROJECT_MASTER_LOG.md](PROJECT_MASTER_LOG.md)
