# Top-Level Research Dossier

## 1. Research mission

Build a research-grade platform for decoding, separating, and eventually
quantifying:

- shared content between visual perception and mental imagery
- private variance specific to perception versus imagery
- future subjective-state signals such as vividness, confidence, and reality strength

This matters beyond `fMRI -> image reconstruction` because the scientific target
is not only what content is represented, but how externally driven and
internally generated experience overlap and diverge.

Animus connection:

- this repository is the scientific backend for later brain-to-latent and
  brain-to-experience systems
- the current platform should mature into the trusted decoder, transfer, ROI,
  and export subsystem for Animus

## 2. Current empirical findings

Current fixed real benchmark:

- max-available canonical overlap set
- `94` rows
- `5` shared paired `nsdId`s
- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- held-out paired evaluation groups: `1`

Current ranking:

1. Ridge
2. Shared-only
3. Shared-private, `private_dim=16`
4. Shared-private, `private_dim=8`
5. Shared-private
6. Shared-private, no domain head

What these findings imply:

- the platform is operational and scientifically usable
- ROI-first multi-subject canonicalization is validated
- the present overlap regime strongly favors simpler models
- shared-only is now strong enough to serve as the practical Animus Core
  Decoder lane
- the full shared-private hypothesis is still exploratory, not confirmed
- private-capacity scaling matters, but does not yet rescue shared-private

## 3. Core paper hypotheses

### H1. Low-overlap regimes favor simpler decoders

In extreme overlap scarcity, Ridge and shared-only models should outperform
explicit shared-private disentanglement because private branches add burden
faster than they add signal.

### H2. Disentanglement has a data threshold

Shared-private modeling may become beneficial only after a threshold of paired
perception/imagery overlap is reached.

### H3. ROI-first canonicalization is necessary for valid multi-subject decoding

Without ROI-first materialization, multi-subject perception/imagery benchmarks
are fragile or invalid because raw full-fMRI dimensionality differs across
subjects.

### H4. Private-capacity scaling governs whether disentanglement helps or hurts

If shared-private becomes competitive later, private latent capacity and related
regularization will likely determine whether disentanglement captures real
structure or only overfits low-data variance.

## 4. Novelty positioning

What is genuinely novel:

- framing the problem as a benchmarked transition from shared-only to
  shared-private decoding rather than assuming disentanglement helps by default
- treating overlap scale as a scientific variable, not only a nuisance
- integrating ROI-first multi-subject preprocessing with a hypothesis-driven
  perception/imagery transfer framework

What is mainly engineering:

- workflow canonicalization
- export scaffolding
- preflight and artifact preparation

What would make the eventual paper publishable:

- a clear threshold-style result showing when shared-private overtakes
  shared-only or Ridge
- or a strong negative result showing that explicit disentanglement is not
  justified in realistic low-overlap regimes

Both could be publishable if framed rigorously.

## 5. Benchmark ladder

Official ladder:

- Ridge: external low-data reference baseline
- Shared-only: best current canonical neural baseline and practical Animus Core
  Decoder
- Shared-private, `private_dim=16`: best current exploratory disentanglement variant
- Shared-private: canonical hypothesis-family baseline

Diagnostic-only controls:

- shared-private, `private_dim=8`
- shared-private, no domain head

See [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md) for commands and promotion rules.

## 6. Experiment roadmap

### Immediate next experiments

- acquire or mount materially larger overlapable perception/imagery data
- rerun the fixed ladder unchanged:
  - Ridge
  - Shared-only
  - Shared-private, `private_dim=16`

Current ranked data options:

1. richer NSD-style paired imagery/perception data beyond the current public release
2. a secondary public paired imagery/perception dataset as a separate benchmark
   (best current public candidate: `ds000203`)
3. large perception-only datasets for the practical Animus lane
   (best current public candidate: `ds004496`, with `ds004192` and `ds001499`
   as follow-on public options)

### Medium-term experiments

- overlap-threshold study across multiple dataset scales
- ROI decomposition study after overlap becomes informative
- narrow private-capacity and regularization study on the larger overlap set

### Future-phase experiments

- vividness/confidence modeling with real labels
- stimulus-vs-percept or reality-monitoring datasets
- richer Animus-facing subjective-state decoding

## 7. Decision rules

Keep shared-private as the main hypothesis family if:

- overlap expands materially
- shared-private narrows the gap to shared-only
- or shared-private begins to provide interpretable ROI/private-structure gains

Demote shared-private to a secondary branch if:

- larger overlap still leaves it consistently below shared-only
- improvements remain fragile and only arise from diagnostics

Promote shared-only as the default paper model if:

- the data ceiling persists
- shared-only keeps winning on expanded overlap benchmarks
- and no clear disentanglement advantage appears

Only claim disentanglement helps if:

- it beats shared-only on a materially larger overlap benchmark
- the gain is repeated across reruns or subject subsets
- and the result does not disappear under a fair Ridge comparison

## 8. Paper path options

### Path A. Honest paper now

`A reproducible benchmark and evidence freeze for perception/imagery decoding under overlap scarcity`

Strength:

- publishable as a rigorous benchmark/resource/negative-result package

Limitation:

- weaker novelty than the long-term vision

Current writing scaffold:

- [PAPER_1_OUTLINE.md](PAPER_1_OUTLINE.md)
- [PAPER_1_CLAIMS_MAP.md](PAPER_1_CLAIMS_MAP.md)
- [PAPER_1_FIGURES_AND_TABLES.md](PAPER_1_FIGURES_AND_TABLES.md)

### Path B. Stronger paper after overlap expansion

`When does shared-private disentanglement help perception-to-imagery decoding?`

Strength:

- strongest scientific trajectory from the current evidence

Requirement:

- materially larger overlapable dataset

### Path C. Ambitious Animus-facing paper

`Toward decoded experience: shared content, private generation, and subjective reality strength`

Strength:

- highest conceptual ambition

Requirement:

- richer paired data, subjective labels, and future reality-monitoring tasks

## 9. Risks

Scientific risks:

- overinterpreting tiny held-out paired evaluation
- mistaking low-data underperformance for a permanent indictment of disentanglement

Dataset risks:

- overlap remains too small for decisive conclusions
- paired evaluation stays statistically weak

Interpretation risks:

- overclaiming ROI meaning from underpowered benchmarks
- reading domain accuracy as substantive evidence on a tiny split

Operational risks:

- future work drifts into feature expansion before data expansion
- benchmark discipline erodes if too many variants are introduced

## 10. Immediate next action

The single best next action is:

- obtain a materially larger overlapable perception/imagery dataset and rerun
  the fixed benchmark ladder without changing the model family or target space

See also:

- [DATA_ACQUISITION_PROGRAM.md](DATA_ACQUISITION_PROGRAM.md)
- [EXTERNAL_DATA_INTEGRATION_PLAN.md](EXTERNAL_DATA_INTEGRATION_PLAN.md)
- [PUBLIC_DATASET_OPPORTUNITY_MAP.md](PUBLIC_DATASET_OPPORTUNITY_MAP.md)
- [PUBLIC_DATASET_INTEGRATION_PLAN.md](PUBLIC_DATASET_INTEGRATION_PLAN.md)

That is the shortest path to a stronger paper and the clearest way to decide
whether shared-private is a viable central claim or only a long-term hypothesis.

## 11. Current paper-writing status

Paper 1 is now scaffolded as:

- an honest benchmark and evidence-freeze paper
- built around the frozen ladder:
  - Ridge
  - shared-only
  - shared-private `private_dim=16`
- explicitly limited to what current low-overlap evidence supports

The next larger paired dataset should extend this scaffold naturally into the
stronger threshold paper rather than replace it wholesale.
