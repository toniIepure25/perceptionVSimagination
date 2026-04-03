# Paper 1 Claims Map

This document maps the realistic paper’s claims to the current evidence.

## Claim status legend

- `Supported`: directly backed by current results or validated platform behavior
- `Partially supported`: motivated by current evidence, but not yet strong
  enough for a full claim
- `Not justified yet`: should not appear as a main paper claim

## Supported claims

| Claim | Status | Current evidence | Source |
| --- | --- | --- | --- |
| The repository provides a reproducible ROI-first platform for multi-subject perception/imagery decoding. | Supported | canonical prep, preflight, train, eval, transfer, analysis, and export are all validated | [VALIDATION.md](VALIDATION.md) |
| ROI-first batching is necessary for valid unequal-shape multi-subject runs in the present benchmark. | Supported | expanded multi-subject run only became valid after the ROI-first batching fix | [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md) |
| Under the current low-overlap benchmark, Ridge is the strongest model tested. | Supported | frozen benchmark ladder | [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md) |
| Under the current low-overlap benchmark, shared-only is the strongest canonical neural baseline. | Supported | shared-only beats all shared-private variants tested so far | [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md) |
| The current public-data benchmark is scientifically underpowered for a strong disentanglement claim. | Supported | only `94` rows, `5` shared ids, `1` held-out paired group | [CURRENT_EVIDENCE_FREEZE.md](CURRENT_EVIDENCE_FREEZE.md) |
| Shared-private `private_dim=16` is the best shared-private variant tested so far. | Supported | benchmark ladder ordering | [BENCHMARK_LADDER.md](BENCHMARK_LADDER.md) |

## Partially supported claims

| Claim | Status | Why partial | What would strengthen it |
| --- | --- | --- | --- |
| Low-overlap regimes favor simpler decoders over explicit disentanglement. | Partially supported | strongly suggested by current results, but only on one tiny benchmark regime | repeated evidence on larger paired datasets or controlled overlap sweeps |
| Private-capacity scaling is a meaningful control knob in low-data regimes. | Partially supported | `private_dim=16` improves over default shared-private, but only modestly and on a tiny dataset | more scales, repeated reruns, larger paired data |
| The current benchmark is useful as an evidence-freeze paper even without a positive disentanglement result. | Partially supported | conceptually strong and evidence-backed, but publication value depends on final writing and positioning | polished paper package, stronger related-work framing, clearer figures |
| Shared-only can already serve as a practical Animus Core Decoder. | Partially supported | subsystem role is justified operationally, but external deployment utility still depends on future data and downstream integration | broader data validation, more robust export and downstream usage examples |

## Not yet justified claims

| Claim | Status | Why not justified | Evidence required |
| --- | --- | --- | --- |
| Shared-private disentanglement improves content decoding. | Not justified yet | current results show the opposite ordering | materially larger paired benchmark where shared-private beats shared-only |
| Shared-private disentanglement improves transfer generalization. | Not justified yet | held-out paired evaluation is too small and does not support the claim | more paired groups and repeated superiority in transfer metrics |
| Private latents are already neuroscientifically meaningful. | Not justified yet | current benchmark is too small and underpowered for strong interpretability claims | larger paired data, robust ROI analyses, stability across reruns |
| The threshold hypothesis has already been confirmed. | Not justified yet | current evidence only motivates it | overlap-expansion study showing performance transition as data grows |
| The repo already supports a paper on vividness, confidence, or subjective reality strength. | Not justified yet | those labels/tasks are not part of the main current evidence | real labeled data and dedicated benchmark results |

## Headline claims safe to use now

These are the strongest statements that can safely appear in the paper:

1. We provide a reproducible ROI-first benchmark platform for multi-subject
   perception/imagery decoding.
2. On the current max-available low-overlap paired benchmark, Ridge is the
   strongest overall baseline and shared-only is the strongest canonical neural
   model.
3. Explicit shared/private disentanglement is not yet supported as the best
   model in this regime.
4. The results motivate a threshold hypothesis for when disentanglement may
   begin to help, rather than confirming that it already does.

## Claims to reserve for the stronger future threshold paper

These should be postponed until larger paired data exists:

- a true crossover point where shared-private overtakes shared-only
- a data-scale law or threshold curve for disentanglement benefit
- robust ROI-resolved private/shared interpretation
- stronger Animus-facing source/confidence conclusions

## Drafting rule

If a sentence sounds like:

- “shared-private helps”
- “disentanglement wins”
- “private latents capture meaningful structure”

it should be checked against this file before being included in the paper.
