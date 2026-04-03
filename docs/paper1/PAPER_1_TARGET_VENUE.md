# Paper 1 Target Venue

## Primary target venue

**Imaging Neuroscience**

Why this is the best fit:

- the paper is fundamentally a neuroscience-motivated benchmark/evidence paper,
  not a large benchmark-track resource pitch
- the main contribution is an honest empirical clarification under overlap
  scarcity, which fits a venue that values careful imaging methodology,
  interpretation, and explicit limits
- the manuscript already centers multi-subject fMRI decoding, mental imagery,
  ROI-first canonicalization, and a disciplined evidence boundary rather than a
  leaderboard-style performance story

Why it matches the current evidence strength:

- the paper's central result is negative/disciplinary rather than a
  state-of-the-art win
- the current paired benchmark is real but scientifically small, so a venue
  that tolerates careful empirical modesty is a better fit than one expecting a
  broad benchmark resource or dramatic scale contribution
- the practical `Animus Core Decoder` lane is relevant, but the manuscript's
  main value is still the clarified scientific benchmark state

Primary manuscript implications:

- keep the neuroscience motivation and perception-versus-imagery framing in the
  foreground
- retain the reproducibility/resource contribution, but do not oversell Paper 1
  as a massive public benchmark release
- keep limitations explicit and reviewer-friendly
- make the appendix/supplement strong on configs, commands, artifact paths, and
  benchmark provenance

Official venue references:

- Imaging Neuroscience journal page: <https://direct.mit.edu/imag>

## Secondary fallback venue

**NeurIPS Datasets and Benchmarks**

Why it remains a plausible fallback:

- the repository now has a strong reproducibility surface
- the benchmark ladder is frozen and the artifact contract is explicit
- the paper has value as a disciplined benchmark/evidence contribution even
  without a positive shared-private result

Why it is not the primary target for this pass:

- the present benchmark is still small for a benchmark-track flagship framing
- the paper reads more naturally as a careful imaging-neuroscience benchmark
  paper than as a broad dataset/resource release
- the strongest differentiator right now is scientific discipline under
  overlap scarcity, not benchmark scale

Official venue references:

- NeurIPS Datasets and Benchmarks call: <https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks>

## Practical decision

For this pass, Paper 1 should be edited as an **Imaging Neuroscience**
submission package while keeping the figures, tables, appendix, and
reproducibility surface sufficiently benchmark-oriented that a Datasets and
Benchmarks fallback remains viable.

## What this choice changes

The venue choice implies these near-term adjustments:

- reduce repo-internal linking noise inside the manuscript body
- tighten transitions so the paper reads like a conventional scientific
  manuscript rather than a stitched internal report
- keep the appendix comprehensive and reproducibility-focused
- maintain very explicit evidence boundaries around the shared-private story

## What this choice does not change

- the benchmark ladder
- the evidence freeze
- the practical role split:
  - Ridge as external reference baseline
  - shared-only as practical `Animus Core Decoder`
  - shared-private `private_dim=16` as the primary exploratory threshold model
