# Threshold Hypothesis

This document formalizes the central research question for the current paper
program:

`When does explicit shared/private disentanglement begin to help in perception-to-imagery fMRI decoding?`

## Why this is the right question now

Current evidence already shows:

- the platform is operational
- shared-only is the strongest canonical neural baseline
- shared-private improves when private capacity is reduced
- Ridge still dominates the low-overlap regime

So the right research move is not to assume disentanglement should already win.
It is to test whether disentanglement has a threshold regime.

## Current threshold benchmark ladder

External reference:

- Ridge

Current canonical neural baseline:

- shared-only

Primary threshold-hypothesis model:

- shared-private, `private_dim=16`

Secondary diagnostics:

- shared-private, `private_dim=8`
- shared-private, no domain head

## Threshold claim to test

Working hypothesis:

- below some paired-overlap scale, shared-only or linear decoding will dominate
- above that scale, shared-private may begin to help if private variance becomes
  learnable rather than mostly burden

This is a threshold claim, not a universal superiority claim.

## Evidence required to promote shared-private

Shared-private should only be promoted over shared-only if all of the following
 become true on a materially larger overlap benchmark:

- it beats shared-only on content cosine
- it matches or improves MSE
- the gain is not dependent on a single lucky rerun
- the result survives fair comparison against Ridge

Additional supportive evidence:

- clearer ROI-resolved structure
- better transfer behavior
- stable behavior across subject subsets

## Evidence that would demote shared-private

If larger overlap data arrives and shared-private still remains clearly below
shared-only, then the right move is:

- demote shared-private from main model claim
- keep it as a secondary explanatory branch
- promote shared-only as the main paper model

## Official config for the threshold model

- [`configs/canonical/threshold_shared_private_p16.yaml`](../configs/canonical/threshold_shared_private_p16.yaml)

This is the primary current hypothesis config because it is the strongest
shared-private variant tested so far.

## Immediate experiment rule

When new overlapable data becomes available, rerun this fixed ladder before
changing the architecture:

1. Ridge
2. Shared-only
3. Shared-private, `private_dim=16`

Only after that result should broader tuning or architectural changes be
considered.
