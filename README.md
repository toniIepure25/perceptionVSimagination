# perceptionVSimagination

**perceptionVSimagination** is a research project on **decoding visual
perception and mental imagery from fMRI** under realistic paired-data
constraints.

The repository has evolved into a disciplined **ROI-first benchmark platform**
for studying when simple baselines, shared neural representations, and explicit
shared/private disentanglement are justified in perception-to-imagery decoding.

The internal Python package namespace remains `fmri2img` for historical
compatibility, but the project identity presented publicly is
**perceptionVSimagination**.

## Project overview

The scientific problem behind this repository is straightforward to state but
hard to test fairly:

> When brain activity is recorded during both visual perception and mental
> imagery, what should a decoder assume is shared across the two domains, and
> what should remain domain-specific?

That question matters for:

- cognitive neuroscience of perception and imagery
- multi-subject neural representation analysis
- the design of stable brain-to-latent decoding systems
- future work on subjective-state decoding, once richer data become available

In practice, however, the field faces a real bottleneck: **paired
perception/imagery overlap is often very small**. This makes it easy to
over-interpret complex models and under-appreciate strong simple baselines.

This project is therefore designed not just to train models, but to provide a
**reproducible empirical framework** for understanding what current data
actually support.

## Current research focus

The repository is currently organized around five central questions:

1. In extreme low-overlap regimes, why do linear and shared-only decoders
   outperform explicit disentanglement?
2. Is there a paired-overlap threshold after which shared/private structure
   begins to help?
3. Is private-capacity scaling one of the main factors governing whether
   disentanglement helps or hurts?
4. What is the most reliable neural baseline for current real-data MVP
   experimentation?
5. What evidence is required before stronger claims about disentanglement
   become scientifically justified?

## Current evidence state

This README follows
[`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md) as a hard
boundary.

### Current max-available paired benchmark

- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- total rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired evaluation groups: `1`
- target space: `vit_l14_image_768`

This is a **real** benchmark, but it is still too small to support strong
positive claims about disentanglement.

### Frozen benchmark ordering

| Rank | Model | Role | Test cosine | Test MSE |
| --- | --- | --- | ---: | ---: |
| 1 | Ridge | External low-data reference baseline | `0.55199` | `0.001167` |
| 2 | Shared-only | Best current canonical neural baseline | `0.13596` | `0.002250` |
| 3 | Shared-private `private_dim=16` | Best current exploratory shared-private variant | `0.10784` | `0.002323` |
| 4 | Shared-private `private_dim=8` | Exploratory recovery variant | `0.09595` | `0.002354` |
| 5 | Shared-private | Canonical hypothesis-family baseline | `0.06927` | `0.002424` |
| 6 | Shared-private no-domain | Diagnostic control | `0.05907` | `0.002450` |

### Interpretation

What the current benchmark supports:

- the platform is operational on real multi-subject data
- the ROI-first benchmarking surface is meaningful and reproducible
- Ridge is the strongest current overall reference baseline
- shared-only is the strongest current canonical neural baseline
- reduced private capacity helps shared-private somewhat, but not enough

What the current benchmark does **not** support:

- the claim that shared-private disentanglement already improves decoding
- the claim that the threshold hypothesis has been confirmed
- strong neuroscientific interpretation of private latents
- strong claims about subjective-state, vividness, or confidence decoding

## What this repository contributes

### 1. A reproducible canonical workflow

The repository provides an end-to-end workflow for:

- data acquisition
- canonical preparation
- overlap assembly
- ROI materialization
- preflight validation
- model training
- evaluation and transfer analysis
- export

### 2. ROI-first multi-subject modeling

The current multi-subject path is intentionally **ROI-first** rather than based
on naive full-voxel stacking across subjects.

This matters because:

- raw fMRI geometry differs across subjects
- ROI-materialized features provide a cleaner and more defensible comparison
  surface
- unequal raw-dimensionality no longer blocks valid multi-subject training

### 3. A disciplined benchmark ladder

The project no longer treats every model as equally provisional. It now
distinguishes clearly between:

- an external reference baseline
- the best current neural baseline
- an exploratory shared-private hypothesis family

That distinction is important scientifically because it prevents the repository
from overstating what current data can support.

## How the project evolved

The repository evolved through several stages:

1. early exploratory pipelines for perception-focused and generation-oriented
   decoding
2. canonical workflow refactoring around preparation, training, evaluation, and
   export
3. real-data validation on a paired perception/imagery benchmark
4. repair of multi-subject training through ROI-first batching
5. evidence-freeze and benchmark-ladder formalization
6. paper-production and research-program structuring

In other words, the project has moved from “trying model ideas” to operating as
a **research platform with explicit evidence boundaries**.

## Implementation summary

### Data layer

The canonical prep path builds reproducible benchmark artifacts from:

- perception indices
- imagery indices
- overlap assembly under shared stimulus identifiers
- ROI materialization into `roi_features_json` / `roi_values_json`
- a fixed `vit_l14_image_768` target cache

### Modeling layer

The current official model family is based on:

- ROI-specific branch encoders
- a configurable shared-only or shared-private latent structure
- a fixed image-embedding target space
- optional auxiliary components used only when scientifically justified

### Evaluation layer

The main comparison surface uses:

- content cosine similarity
- content MSE
- imagery/perception mean summaries where available
- paired-group counts
- transfer analysis and diagnostic quantities as secondary evidence

## Current bottleneck

The main bottleneck is **not** basic infrastructure anymore.

The main bottleneck is **paired-data scale**.

Even after integrating the full public NSD-Imagery source available in the
current environment, the benchmark still tops out at:

- `94` rows
- `5` shared paired `nsdId`s
- `1` held-out paired evaluation group

The next decisive scientific step therefore requires a **larger paired
perception/imagery dataset**, not more aggressive claims on the current one.

## Why the current results are still valuable

The project is already scientifically useful because it establishes an honest
evidence state:

- complex models can be implemented and tested fairly
- strong simple baselines are taken seriously
- low-overlap scarcity is treated as an empirical condition, not hidden
- the current strongest neural baseline is identified clearly
- the next research question is now sharper: *when does disentanglement begin
  to help, if ever?*

That is a valuable contribution even without a positive shared-private result.

## Documentation map

Start here:

- [`START_HERE.md`](START_HERE.md)
- [`docs/CURRENT_EVIDENCE_FREEZE.md`](docs/CURRENT_EVIDENCE_FREEZE.md)
- [`docs/BENCHMARK_LADDER.md`](docs/BENCHMARK_LADDER.md)
- [`docs/TOP_LEVEL_RESEARCH_DOSSIER.md`](docs/TOP_LEVEL_RESEARCH_DOSSIER.md)
- [`docs/PROJECT_MASTER_LOG.md`](docs/PROJECT_MASTER_LOG.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

For the current paper path:

- [`docs/paper1/PAPER_1_FULL_DRAFT.md`](docs/paper1/PAPER_1_FULL_DRAFT.md)
- [`docs/paper1/PAPER_1_APPENDIX.md`](docs/paper1/PAPER_1_APPENDIX.md)
- [`docs/paper1/PAPER_1_TARGET_VENUE.md`](docs/paper1/PAPER_1_TARGET_VENUE.md)

For the broader research program:

- [`docs/PAPER_POSITIONING.md`](docs/PAPER_POSITIONING.md)
- [`docs/THRESHOLD_HYPOTHESIS.md`](docs/THRESHOLD_HYPOTHESIS.md)
- [`docs/DATA_ACQUISITION_PROGRAM.md`](docs/DATA_ACQUISITION_PROGRAM.md)
- [`docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`](docs/EXTERNAL_DATA_INTEGRATION_PLAN.md)

## Quick technical note

The repository package and workflow namespace remain `fmri2img`, so commands
and imports use that name. For the full command surface, environment variables,
and reproducibility details, see:

- [`START_HERE.md`](START_HERE.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

## Legacy note

Historical perception-only, feature-first, and generation-first paths remain in
the repository for comparison and reproducibility, but they are not the current
source of truth.
