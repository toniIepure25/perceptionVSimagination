# Current Evidence Freeze

As of `2026-04-02`, this document freezes the current empirical state of the
canonical perception/imagery decoding program.

## Scope of the freeze

Dataset scope:

- canonical max-available overlap dataset
- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- total rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired evaluation groups: `1`
- target space: `vit_l14_image_768`

This is a real-data MVP benchmark, but it is still too small to support strong
claims about the full shared-private paper hypothesis.

## Frozen benchmark ladder

| Rank | Model | Role | Test cosine | Test MSE | Status |
| --- | --- | --- | ---: | ---: | --- |
| 1 | Ridge | External low-data reference baseline | `0.55199` | `0.001167` | validated |
| 2 | Shared-only | Best current canonical neural baseline | `0.13596` | `0.002250` | validated |
| 3 | Shared-private, `private_dim=16` | Best current shared-private exploratory variant | `0.10784` | `0.002323` | exploratory |
| 4 | Shared-private, `private_dim=8` | Exploratory recovery variant | `0.09595` | `0.002354` | exploratory |
| 5 | Shared-private | Canonical hypothesis family baseline | `0.06927` | `0.002424` | exploratory |
| 6 | Shared-private, no domain head | Diagnostic control | `0.05907` | `0.002450` | exploratory |

## What has been validated

Infrastructure and workflow:

- canonical ROI-first multi-subject training works on unequal raw voxel
  dimensionality
- canonical prep, preflight, train, eval, transfer, analysis, and export paths
  are operational on real data
- canonical imagery rebuild from full public NSD-Imagery source is working
- the fixed Ridge comparison workflow is reproducible

Empirical findings:

- the current max-available overlap set supports a meaningful benchmark ladder,
  but not a fair test of the full shared-private hypothesis
- shared-only is the strongest canonical neural model on the present dataset
- reducing private capacity helps shared-private relative to its default form
- disabling the domain head alone does not rescue shared-private

## What remains exploratory

- the shared-private family itself
- the claim that explicit disentanglement improves perception-to-imagery
  decoding
- the interpretation of domain accuracy on this tiny held-out paired set
- any ROI-level neuroscientific conclusion beyond basic operational sanity

## Claims that are currently justified

- ROI-first canonicalization is necessary and sufficient for valid multi-subject
  runs in the present platform
- in the current low-overlap regime, simple linear decoding is much stronger
  than the shared-private neural family
- in the current low-overlap regime, a shared-only neural model is a better
  canonical neural baseline than explicit shared-private disentanglement
- private-latent capacity matters: smaller private capacity reduces, but does
  not eliminate, the shared-private penalty

## Claims that are not yet justified

- that shared-private disentanglement improves content decoding
- that shared-private disentanglement improves transfer generalization
- that the private branches capture meaningful perception-private or
  imagery-private structure rather than low-data burden
- that the current benchmark says anything decisive about vividness or
  subjective-state decoding
- that the current dataset is large enough to validate or reject the intended
  paper hypothesis

## Official interpretation

The current repository should be read in this order:

- Ridge is the external low-data reference baseline.
- Shared-only is the best current canonical neural baseline and the practical
  Animus Core Decoder lane.
- Shared-private is the main research hypothesis family, not the current
  performance leader.
- Shared-private with `private_dim=16` is the strongest shared-private variant
  tested so far, but it remains below shared-only.

## What evidence is still missing

- a larger paired source than the already integrated public NSD-Imagery release
- materially larger overlapable perception/imagery data
- more than one meaningful held-out paired group
- repeated superiority of shared-private over shared-only on a fixed benchmark
- stronger ROI-resolved evidence that disentanglement helps interpretation
  rather than only adding capacity

## Related reports

- [PROJECT_MASTER_LOG.md](PROJECT_MASTER_LOG.md)
- [EXPANDED_OVERLAP_COMPARISON.md](EXPANDED_OVERLAP_COMPARISON.md)
- [TOP_LEVEL_RESEARCH_DOSSIER.md](TOP_LEVEL_RESEARCH_DOSSIER.md)
