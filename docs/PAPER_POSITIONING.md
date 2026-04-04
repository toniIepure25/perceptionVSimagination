# Paper Positioning

This document defines the honest paper positioning for the current state of the
repository.

## Current stance

The project is no longer just an `fMRI -> image reconstruction` codebase.

Its emerging paper direction is:

`Disentangling shared and private neural structure between perception and imagery, under real-data constraints, with ROI-first multi-subject decoding.`

That statement is aspirational, but the current evidence does not yet support a
claim that explicit disentanglement wins.

## What the project can already claim

- it provides a reproducible ROI-first platform for perception/imagery decoding
- it supports canonical multi-subject preparation, preflight, training,
  evaluation, and export
- it establishes an honest benchmark ladder on real overlapable data
- it shows that in the current low-overlap regime, Ridge and shared-only are
  stronger than the tested shared-private variants
- it identifies private-capacity scaling as a plausible reason shared-private
  underperforms in tiny-data conditions

## What the project cannot yet claim

- that shared-private disentanglement improves performance
- that private latents are neuroscientifically meaningful on the current data
- that the current overlap set is large enough to validate the intended paper
  hypothesis
- that vividness or subjective-state decoding is yet part of the main empirical
  story

## What is novel already

Novelty already supported:

- ROI-first canonicalization for unequal-shape multi-subject
  perception/imagery decoding
- a disciplined benchmark ladder that separates linear, shared-only, and
  shared-private regimes
- an empirical finding that shared-only currently beats explicit disentanglement
  on the available low-overlap paired dataset

This is different from a generic reconstruction paper because the core question
is not just whether images can be reconstructed, but when shared content,
private variance, and transfer structure should be modeled separately.

## What would make the eventual paper publishable

The strongest paper version would show:

- a materially larger overlapable dataset
- a threshold regime where shared-private becomes competitive or superior
- clear comparison against Ridge and shared-only
- ROI-resolved evidence for where shared versus private information lives
- honest limits around subjective-state claims

## Current realistic paper path

If writing from the current evidence alone, the best framing is:

`A reproducible benchmark and evidence freeze for perception/imagery decoding under extreme overlap scarcity.`

That is more modest than the long-term paper goal, but it is scientifically
honest.

The paper-writing scaffold for that path is now:

- [PAPER_1_OUTLINE.md](PAPER_1_OUTLINE.md)
- [PAPER_1_CLAIMS_MAP.md](PAPER_1_CLAIMS_MAP.md)
- [PAPER_1_FIGURES_AND_TABLES.md](PAPER_1_FIGURES_AND_TABLES.md)

The first draft package now also includes:

- `docs/paper1/PAPER_1_ABSTRACT.md`
- `docs/paper1/PAPER_1_INTRODUCTION.md`
- `docs/paper1/PAPER_1_RELATED_WORK.md`
- `docs/paper1/PAPER_1_METHODS.md`
- `docs/paper1/PAPER_1_BENCHMARK_SETUP.md`
- `docs/paper1/PAPER_1_RESULTS.md`
- `docs/paper1/PAPER_1_DISCUSSION.md`
- `docs/paper1/PAPER_1_LIMITATIONS.md`
- `docs/paper1/PAPER_1_CONCLUSION.md`
- `docs/paper1/PAPER_1_FULL_DRAFT.md`

## Stronger future paper path

If overlap grows meaningfully, the paper can become:

`When does explicit shared-private disentanglement help perception-to-imagery decoding?`

That would be a stronger and more novel contribution because it turns the
current negative/ambiguous result into a threshold-style scientific question.

## Current acquisition bottleneck

The public NSD-Imagery source has already been integrated and only expands the
current paired benchmark to `5` shared ids. The next paper-strength change in
evidence now depends on a larger paired source, not more ad hoc model churn on
the same benchmark.

## Animus-facing future paper path

If richer subjective labels and reality-monitoring paradigms become available,
the longer-term paper trajectory becomes:

`From content decoding to decoded experience: shared content, private generation, and subjective reality strength.`

That is future-phase and should not be implied by current results.

## Official language to use now

Prefer:

- shared-private is the main hypothesis family
- shared-only is the best current canonical neural baseline and current Animus
  Core Decoder
- shared-private `private_dim=16` is the primary current threshold-testing model
- Ridge is the external low-data reference baseline
- the present benchmark is operationally validated but scientifically underpowered

Avoid:

- shared-private is state of the art here
- disentanglement has been shown to help
- the current benchmark already supports a full paper claim

## Current writing rule

Paper 1 should be written as:

- a benchmark/resource/evidence paper
- a disciplined low-overlap finding
- a threshold hypothesis motivator

It should not be written as:

- a positive disentanglement paper
- a vividness paper
- a decoded-experience paper

## Current drafting status

Paper 1 has now advanced from:

- outline and claims scaffold

to:

- section-level first draft text
- a manuscript-style full-draft assembly
- a first figure/table package
- a bibliography lock with verified references
- a caption package and submission scaffold
- a chosen primary target venue (`Imaging Neuroscience`)
- an assembled appendix/supplement package
- a submission-package plan for template conversion

Still required before submission quality:

- final bibliography style conversion for the target venue
- final caption polishing and table layout cleanup
- tighter cross-section editing for style and redundancy
- venue-specific formatting
- final proofread and last claim audit against the frozen evidence
- optional decision on whether to add a direct dataset-host citation for the
  public imagery release

The current submission-hardening target is now explicitly:

- primary venue: **Imaging Neuroscience**
- main text: compact scientific narrative with explicit figure/table support
- appendix/supplement: reproducibility, configs, commands, artifact paths, and
  export contract summary

That means the paper should read less like an internal repo dossier and more
like a careful neuroscience submission with a strong supplement.

Paper 1 now also has:

- a first real figure package under `docs/paper1/figures/`
- a first real table package under `docs/paper1/tables/`
- a claims-tightening pass aligned to the evidence freeze
- a citation plan, bibliography lock, and bibliography TODO package under
  `docs/paper1/`
- a caption package, target-venue note, submission checklist, appendix, and
  submission-package plan under `docs/paper1/`
