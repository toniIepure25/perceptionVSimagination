# fmri2img

`fmri2img` is a research-grade platform for **perception-to-imagery fMRI
decoding** built around a reproducible, ROI-first canonical workflow.

The repository now has two official roles:

1. **Practical subsystem role**
   The current **Animus Core Decoder** lane, implemented as a stable
   shared-only neural decoder for content prediction and export.
2. **Research-program role**
   A disciplined benchmark and hypothesis-testing program for the question:
   **when, if ever, does explicit shared/private disentanglement begin to help
   perception-to-imagery decoding?**

This README is intentionally grounded in
[`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md). It is not
written as a positive disentanglement story, because the current data do not
support that claim.

## Executive summary

### What the repository currently demonstrates

- a reproducible ROI-first platform for multi-subject perception/imagery
  decoding
- canonical preparation, preflight, training, evaluation, transfer analysis,
  and export on real data
- a frozen benchmark ladder with clear role separation between:
  - an external scientific reference baseline
  - the best current canonical neural baseline
  - an exploratory shared-private hypothesis family
- a practical shared-only decoder lane that is already usable as the current
  **Animus Core Decoder**

### Current frozen benchmark state

- dataset: current max-available canonical paired benchmark
- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- total rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired evaluation groups: `1`
- target space: `vit_l14_image_768`

### Current evidence-based ordering

| Rank | Model | Role | Test cosine | Test MSE |
| --- | --- | --- | ---: | ---: |
| 1 | Ridge | External low-data reference baseline | `0.55199` | `0.001167` |
| 2 | Shared-only | Best current canonical neural baseline / Animus Core Decoder | `0.13596` | `0.002250` |
| 3 | Shared-private `private_dim=16` | Best current exploratory shared-private variant | `0.10784` | `0.002323` |
| 4 | Shared-private `private_dim=8` | Exploratory recovery variant | `0.09595` | `0.002354` |
| 5 | Shared-private | Canonical hypothesis-family baseline | `0.06927` | `0.002424` |
| 6 | Shared-private no-domain | Diagnostic control | `0.05907` | `0.002450` |

### What this means

- **Ridge** is still the strongest overall reference model in the current
  low-overlap regime.
- **Shared-only** is the best current canonical neural model and the correct
  practical subsystem lane for Animus.
- **Shared-private** remains scientifically interesting, but it is still a
  hypothesis family rather than the current winner.
- The main bottleneck is still **paired-data scale**, not basic infrastructure.

## Why this project exists

Perception-to-imagery decoding sits at an important intersection of machine
learning, cognitive neuroscience, and future brain-computer interface design.
The scientific question is not only whether visual content can be decoded from
brain activity, but whether externally driven perception and internally
generated imagery should be modeled as:

- sharing a common representational core
- expressing domain-specific private variance
- or remaining better served by simpler baselines under scarce paired data

In practice, this question is hard to answer fairly because real paired
perception/imagery datasets are often small, heterogeneous across subjects, and
easy to benchmark inconsistently. This repository exists to solve that systems
problem first: to provide a stable benchmark surface, a reproducible canonical
pipeline, and a research program that makes the current evidence boundary
explicit.

## Research program

The repository is now organized around five top-level scientific questions:

1. In extreme low-overlap regimes, why do Ridge and shared-only outperform
   explicit disentanglement?
2. Is there a paired-overlap threshold after which shared-private modeling
   becomes helpful?
3. Is private-capacity scaling one of the main control knobs determining
   whether disentanglement helps or hurts?
4. Can a shared-only decoder serve as a robust and useful practical Animus
   subsystem even before richer subjective-state labels exist?
5. What evidence is required before promoting shared-private from exploratory
   hypothesis model to official canonical winner?

The current paper path is intentionally modest and rigorous:

> **Perception-to-Imagery Decoding Under Overlap Scarcity: A Reproducible
> ROI-First Benchmark**

That paper is a benchmark/evidence paper, not a state-of-the-art disentangling
paper.

## How the project evolved

This repository did not begin in its current form. It evolved through several
stages:

1. **Legacy experimentation**
   Earlier perception-only, generation-first, and mixed script surfaces
   accumulated valuable ideas but were harder to compare cleanly.
2. **Canonical platform refactor**
   The project was reorganized around a canonical workflow surface for prep,
   preflight, training, evaluation, transfer analysis, and export.
3. **Real-data validation**
   The platform was exercised on a true perception/imagery overlap benchmark,
   exposing both data limitations and infrastructure issues.
4. **ROI-first multi-subject hardening**
   The training path was made robust to unequal raw fMRI dimensionality across
   subjects by promoting ROI-materialized features to the official multi-subject
   input contract.
5. **Evidence freeze and benchmark ladder**
   The repository moved from ad hoc experimentation to a disciplined ladder:
   Ridge, shared-only, shared-private `private_dim=16`, and diagnostic controls.
6. **Paper-production and subsystem formalization**
   Shared-only was promoted into the **Animus Core Decoder** lane, while
   shared-private `private_dim=16` became the primary threshold-testing
   hypothesis model.

## Implementation overview

### 1. Canonical data and preparation layer

The canonical prep path constructs reproducible benchmark artifacts from:

- perception indices
- imagery indices
- overlap assembly under shared stimulus identifiers
- ROI materialization into `roi_features_json` / `roi_values_json`
- a fixed `vit_l14_image_768` target cache

This layer is responsible for:

- provenance-aware preparation
- split validation
- preflight classification
- keeping benchmark semantics stable across runs

### 2. ROI-first multi-subject input contract

The current multi-subject training path is **ROI-first**:

- ROI groups are the official multi-subject input representation
- raw full-fMRI vectors are optional auxiliary context
- incompatible raw voxel shapes are not allowed to block ROI-materialized
  multi-subject training

This is not only an engineering convenience. It is part of the platform’s
scientific validity, because naive same-shape voxel stacking across subjects is
not a reasonable default assumption.

### 3. Model lanes

The repository now distinguishes three roles clearly:

| Lane | Model | Purpose |
| --- | --- | --- |
| External baseline | Ridge | Low-data scientific reference |
| Practical subsystem | Shared-only | Current Animus Core Decoder |
| Exploratory research | Shared-private `private_dim=16` | Threshold-testing hypothesis model |

Additional shared-private variants remain useful as diagnostics, but they are
not the headline benchmark entries.

### 4. Workflow surface

The canonical workflow surface covers:

- acquisition
- preparation
- preflight
- training
- evaluation
- transfer analysis
- ROI analysis
- export

The practical shared-only lane also has dedicated wrapper commands so subsystem
users do not need to think in terms of experimental overrides.

### 5. Export and Animus integration

The practical shared-only export path records:

- target-space metadata
- benchmark role / evidence tier
- Animus subproject identity
- decoder role
- stability tier
- interface readiness for:
  - content
  - source
  - confidence

Current state:

- `content`: active
- `source`: scaffolded
- `confidence`: scaffolded

This keeps the practical subsystem honest without pretending that richer
subjective-state decoding is already validated.

## Current results and interpretation

The current benchmark supports a clear but narrow conclusion:

- the platform is operational
- the benchmark ladder is meaningful
- simple baselines dominate in the current low-overlap regime
- shared-only is the correct neural baseline today
- shared-private remains unproven on present data

The current results **do not** justify claiming that:

- explicit disentanglement improves content decoding
- the threshold hypothesis has already been confirmed
- private latents already support strong neuroscientific interpretation
- vividness, confidence, or subjective reality are part of the present empirical
  contribution

Those boundaries are not a weakness in the repo; they are one of its strengths.
The repository is explicitly designed to make these boundaries legible rather
than hiding them behind moving-target experimentation.

## Practical Animus lane vs research lane

### Practical Animus Core Decoder

Use the shared-only lane when the priority is:

- dependable content decoding
- stable export
- clean subsystem reuse
- minimal scientific ambiguity

Primary config:

- `configs/canonical/animus_core_decoder.yaml`

Primary commands:

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.eval_animus_core_decoder --checkpoint ...
python -m fmri2img.workflows.export_animus_core_decoder --checkpoint ...
```

### Threshold-testing research lane

Use the research lane when the priority is answering the scientific question:

> when does explicit shared/private disentanglement begin to help?

Primary configs:

- `configs/canonical/max_available_overlap.yaml`
- `configs/canonical/threshold_shared_private_p16.yaml`

Primary commands:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

## Quick start

### Installation

```bash
pip install -e ".[all]"
```

Recommended environment:

- Python `>=3.10`
- see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for env vars and
  real-data paths

### Fastest smoke check

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
```

### Practical subsystem path

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
python -m fmri2img.workflows.train_animus_core_decoder
```

### External paired-data acquisition

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

### Fixed benchmark ladder

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

### Generic canonical bootstrap / subject-local prep surface

The repo also keeps a generic canonical workflow surface for smoke tests,
bootstrap-scale experiments, and subject-local preparation:

```bash
python -m fmri2img.workflows.prepare_perception_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_mixed_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_roi_features \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
python -m fmri2img.workflows.run_analysis \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

## Data status and current bottleneck

The public NSD-Imagery source is already integrated and canonicalized. That was
an important systems milestone, but it does **not** solve the main empirical
problem. In the currently accessible environment, the benchmark still tops out
at:

- `94` rows
- `5` shared paired `nsdId`s
- `1` held-out paired evaluation group

So the next decisive progress depends on **larger paired data**, not on
claim-inflating model churn on the same tiny benchmark.

## Documentation map

If you are new to the repository, start here:

- [`START_HERE.md`](START_HERE.md)
- [`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md)
- [`docs/BENCHMARK_LADDER.md`](docs/BENCHMARK_LADDER.md)
- [`docs/ANIMUS_CORE_DECODER.md`](docs/ANIMUS_CORE_DECODER.md)
- [`docs/TOP_LEVEL_RESEARCH_DOSSIER.md`](docs/TOP_LEVEL_RESEARCH_DOSSIER.md)
- [`docs/THRESHOLD_HYPOTHESIS.md`](docs/THRESHOLD_HYPOTHESIS.md)
- [`docs/PROJECT_MASTER_LOG.md`](docs/PROJECT_MASTER_LOG.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

Paper 1 package:

- [`docs/paper1/PAPER_1_FULL_DRAFT.md`](docs/paper1/PAPER_1_FULL_DRAFT.md)
- [`docs/paper1/PAPER_1_APPENDIX.md`](docs/paper1/PAPER_1_APPENDIX.md)
- [`docs/paper1/PAPER_1_TARGET_VENUE.md`](docs/paper1/PAPER_1_TARGET_VENUE.md)

Data acquisition and next-step planning:

- [`docs/DATA_ACQUISITION_PROGRAM.md`](docs/DATA_ACQUISITION_PROGRAM.md)
- [`docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`](docs/EXTERNAL_DATA_INTEGRATION_PLAN.md)

## What this repository is not

At its current evidence state, this repository is **not**:

- a positive shared-private win paper
- a vividness/confidence decoding paper
- a stimulus-vs-percept paper
- a generic image reconstruction leaderboard

It is a disciplined benchmark-and-subsystem platform that already has real
practical value while still being honest about what the current data can and
cannot prove.

## Legacy note

Historical perception-only, feature-first, and generation-first paths remain in
the repository for comparison and reproducibility, but they are no longer the
canonical source of truth.
