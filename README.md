# perceptionVSimagination

**perceptionVSimagination** is an interdisciplinary research project at the
intersection of **cognitive neuroscience**, **machine learning**, and
**computational modeling of mental imagery**.

The project asks a focused scientific question:

> **How much of visual perception and mental imagery can be decoded through a
> shared neural representation, and when does explicit shared/private
> disentanglement become useful?**

This repository is not just a model implementation. It is a **reproducible
research platform** for building, validating, and benchmarking
perception-to-imagery fMRI decoders under realistic data constraints.

The internal Python package namespace remains `fmri2img` for historical
compatibility, but the public research identity of the project is
**perceptionVSimagination**.

## Contents

- [Project Summary](#project-summary)
- [Why This Research Matters](#why-this-research-matters)
- [Interdisciplinary Significance](#interdisciplinary-significance)
- [Current Evidence State](#current-evidence-state)
- [Core Research Questions](#core-research-questions)
- [What The Repository Contributes](#what-the-repository-contributes)
- [How The Project Evolved](#how-the-project-evolved)
- [Current Results And Their Meaning](#current-results-and-their-meaning)
- [Possible Paper Trajectories](#possible-paper-trajectories)
- [Current Limitation And Next Step](#current-limitation-and-next-step)
- [Documentation Map](#documentation-map)
- [Technical Note](#technical-note)

## Project Summary

Perception and mental imagery are closely related cognitive phenomena, but they
are not identical. A central challenge in cognitive science is to understand:

- what neural structure is shared between seeing and imagining
- what remains domain-specific
- how strongly current data support one modeling assumption over another

This repository approaches that problem through an **ROI-first,
multi-subject decoding framework**. It provides a stable empirical surface for:

- canonical data preparation
- overlap-aware benchmark construction
- multi-subject ROI materialization
- reproducible training and evaluation
- comparison between simple baselines and more structured neural models

The project is therefore both:

- a **scientific benchmark** for testing hypotheses about perception and
  imagery
- a **systems platform** for conducting disciplined, reproducible research

## Why This Research Matters

Perception-to-imagery decoding matters because it sits at the boundary between
representation, cognition, and conscious experience.

From a cognitive-science perspective, the project is relevant to:

- the overlap and distinction between perception and mental imagery
- internal versus externally driven visual representations
- multi-subject representation structure in the human brain
- future questions about vividness, confidence, and subjective reality once
  richer data become available

From a machine-learning perspective, it addresses a different but equally
important issue:

- when do more structured models genuinely help, and when do they merely add
  complexity in low-data settings?

That tension is at the heart of the current research program.

## Interdisciplinary Significance

One reason this project is interesting academically is that it genuinely spans
multiple disciplines rather than sitting neatly inside one.

### Cognitive neuroscience

The project studies the neural relationship between perception and imagery, a
core problem in the science of mental representation.

### Machine learning

It compares simple and structured decoders under a fixed benchmark ladder,
testing whether shared/private modeling is justified by evidence rather than by
architectural intuition.

### Computational modeling

It treats perception and imagery as related but potentially separable latent
domains, making the project relevant to broader questions about representation,
generalization, and domain structure.

### Research engineering

It emphasizes reproducibility, benchmark discipline, and explicit evidence
boundaries, which are essential when working with small and fragile
neuroimaging datasets.

## Current Evidence State

This README follows
[`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md) as a hard
constraint. The paper and the repository should become stronger by becoming
**clearer and more disciplined**, not by becoming more aggressive.

### Current benchmark at a glance

| Property                          | Current value                          |
| --------------------------------- | -------------------------------------- |
| Subjects                          | `subj02`, `subj03`, `subj05`, `subj07` |
| Total rows                        | `94`                                   |
| Shared paired `nsdId`s            | `5`                                    |
| Held-out paired evaluation groups | `1`                                    |
| Target space                      | `vit_l14_image_768`                    |

This is a **real benchmark**, but still a **very small paired-overlap regime**.

### Frozen benchmark ladder

| Rank | Model                           | Scientific role                                 | Test cosine |   Test MSE |
| ---- | ------------------------------- | ----------------------------------------------- | ----------: | ---------: |
| 1    | Ridge                           | External low-data reference baseline            |   `0.55199` | `0.001167` |
| 2    | Shared-only                     | Best current canonical neural baseline          |   `0.13596` | `0.002250` |
| 3    | Shared-private `private_dim=16` | Best current exploratory shared-private variant |   `0.10784` | `0.002323` |
| 4    | Shared-private `private_dim=8`  | Exploratory recovery variant                    |   `0.09595` | `0.002354` |
| 5    | Shared-private                  | Canonical hypothesis-family baseline            |   `0.06927` | `0.002424` |
| 6    | Shared-private no-domain        | Diagnostic control                              |   `0.05907` | `0.002450` |

### What is justified now

- the ROI-first multi-subject platform is operational on real data
- the benchmark ladder is meaningful and reproducible
- Ridge is the strongest current overall reference baseline
- shared-only is the strongest current canonical neural baseline
- reduced private capacity improves shared-private somewhat, but does not make
  it the best model

### What is not justified yet

- that shared-private disentanglement already improves decoding performance
- that the threshold hypothesis has been confirmed
- that private latents already support strong neuroscientific interpretation
- that the current benchmark can support strong claims about vividness,
  confidence, or other subjective-state variables

## Core Research Questions

The repository is currently organized around the following research questions:

1. **Why do simple decoders dominate in extreme low-overlap regimes?**
2. **Is there a threshold of paired overlap after which disentanglement begins
   to help?**
3. **How important is private-latent capacity in determining whether
   shared/private modeling helps or hurts?**
4. **What is the most reliable neural baseline for current real-data
   experimentation?**
5. **What evidence would be needed before stronger claims about disentanglement
   become scientifically defensible?**

These questions are intended to be answered incrementally, with the benchmark
ladder held fixed wherever possible.

## What The Repository Contributes

### 1. A reproducible canonical workflow

The repository provides an end-to-end canonical pipeline for:

- data acquisition
- imagery/perception preparation
- overlap assembly
- ROI materialization
- preflight validation
- model training
- evaluation and transfer analysis
- export and artifact packaging

### 2. ROI-first multi-subject decoding

The current multi-subject path is intentionally **ROI-first** rather than based
on naive full-voxel stacking. This matters because:

- raw fMRI geometry differs across subjects
- ROI materialization provides a cleaner and more defensible comparison surface
- unequal raw dimensionality no longer blocks valid multi-subject training

### 3. A disciplined benchmark ladder

The project explicitly separates:

- a strong simple reference baseline
- the best current canonical neural baseline
- an exploratory shared-private hypothesis family

That separation is scientifically valuable because it prevents overclaiming and
keeps the benchmark interpretable.

## How The Project Evolved

The repository evolved through several stages:

1. **Initial exploratory pipelines**
   Early work focused on perception-oriented and generation-oriented decoding
   paths.
2. **Canonical workflow refactor**
   The project was reorganized around a unified preparation, training,
   evaluation, and export surface.
3. **Real-data validation**
   The platform was exercised on a true paired perception/imagery benchmark.
4. **ROI-first multi-subject hardening**
   Multi-subject training was repaired by promoting ROI-materialized features to
   the official input contract.
5. **Evidence freeze and benchmark formalization**
   The repository moved from loosely exploratory experimentation to a frozen
   benchmark ladder and explicit evidence boundaries.
6. **Paper-production phase**
   The codebase was extended into a paper-ready research platform with a full
   benchmark/evidence manuscript package.

This evolution is important because it shows the project is not just a single
experiment, but a maturing research program.

## Current Results And Their Meaning

The current results support a clear but modest conclusion:

- the platform works
- the benchmark is meaningful
- the present data regime strongly favors simpler models
- shared-only is the best current neural baseline
- shared-private remains an open hypothesis rather than a supported winner

Scientifically, that is still valuable.

It tells us that the important unresolved question is no longer:

> “Can we build a shared/private model?”

but rather:

> “Under what empirical conditions does shared/private structure actually begin
> to help?”

That is a much sharper and more publishable scientific question.

## Possible Paper Trajectories

This repository is intended to support more than one paper over time.

### Paper path 1. Honest benchmark/evidence paper

The current realistic paper path is:

> **Perception-to-Imagery Decoding Under Overlap Scarcity: A Reproducible
> ROI-First Benchmark**

This paper is intentionally modest and rigorous. It is a benchmark/evidence
paper, not a positive disentanglement-win paper.

### Paper path 2. Threshold paper

If materially larger paired data become available, the next stronger paper
would ask:

> **When does explicit shared/private disentanglement begin to help
> perception-to-imagery decoding?**

### Paper path 3. Richer subjective-state work

If future datasets include vividness, confidence, or reality-monitoring labels,
the project could grow toward a broader cognitive-neuroscience paper on
internally generated experience and decoded subjective state.

## Current Limitation And Next Step

The current bottleneck is **paired-data scale**.

Even after integrating the full public NSD-Imagery source available in the
current environment, the benchmark still tops out at:

- `94` rows
- `5` shared paired `nsdId`s
- `1` held-out paired evaluation group

So the next decisive scientific step is straightforward:

> acquire or integrate a materially larger paired perception/imagery dataset
> and rerun the fixed benchmark ladder unchanged.

That is the cleanest way to test whether the current shared-private
underperformance is a true modeling limitation or primarily a data-regime
effect.

## Documentation Map

For orientation:

- [`START_HERE.md`](START_HERE.md)
- [`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md)
- [`docs/BENCHMARK_LADDER.md`](docs/BENCHMARK_LADDER.md)
- [`docs/TOP_LEVEL_RESEARCH_DOSSIER.md`](docs/TOP_LEVEL_RESEARCH_DOSSIER.md)
- [`docs/PROJECT_MASTER_LOG.md`](docs/PROJECT_MASTER_LOG.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

For the current paper package:

- [`docs/paper1/PAPER_1_FULL_DRAFT.md`](docs/paper1/PAPER_1_FULL_DRAFT.md)
- [`docs/paper1/PAPER_1_APPENDIX.md`](docs/paper1/PAPER_1_APPENDIX.md)
- [`docs/paper1/PAPER_1_TARGET_VENUE.md`](docs/paper1/PAPER_1_TARGET_VENUE.md)

For broader research framing:

- [`docs/PAPER_POSITIONING.md`](docs/PAPER_POSITIONING.md)
- [`docs/THRESHOLD_HYPOTHESIS.md`](docs/THRESHOLD_HYPOTHESIS.md)
- [`docs/DATA_ACQUISITION_PROGRAM.md`](docs/DATA_ACQUISITION_PROGRAM.md)
- [`docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`](docs/EXTERNAL_DATA_INTEGRATION_PLAN.md)

## Technical Note

The package and workflow namespace remain `fmri2img`, so commands and imports
still use that name. For the full command surface, environment variables, and
reproducibility details, see:

- [`START_HERE.md`](START_HERE.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

## Legacy Note

Historical perception-only, feature-first, and generation-first paths remain in
the repository for comparison and reproducibility, but they are not the current
source of truth.
