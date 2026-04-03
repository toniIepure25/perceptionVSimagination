# Paper 1 Captions

These are the current submission-oriented captions for the main figures and
tables. They are written to stay inside the frozen evidence boundary and to
support an Imaging Neuroscience-style submission package while remaining
compatible with a benchmark-oriented fallback venue.

## Figures

### Figure 1. Frozen benchmark ladder on the current low-overlap dataset

Benchmark ordering on the fixed `94`-row / `5`-shared-id paired dataset.
Horizontal bars show test cosine similarity on the same evaluation surface for
all models. Ridge remains the strongest overall reference baseline. Shared-only
is the strongest canonical neural baseline. Shared-private `private_dim=16` is
the strongest exploratory shared-private variant tested so far, but remains
below shared-only. Higher is better.

Assets:
- [SVG](figures/figure1_benchmark_ladder.svg)
- [PNG](figures/figure1_benchmark_ladder.png)

### Figure 2. The current benchmark is constrained by severe overlap scarcity

Summary of the present empirical regime. The benchmark contains `94` mixed rows,
`5` shared paired stimulus IDs, and only `1` held-out paired evaluation group
across `subj02`, `subj03`, `subj05`, and `subj07`. This figure explains why the
paper must be interpreted as an evidence-freeze benchmark paper rather than a
strong positive disentanglement paper. It provides the key context for the
manuscript’s central claim that overlap scarcity is the main current limit on
interpretation.

Assets:
- [SVG](figures/figure2_overlap_scarcity.svg)
- [PNG](figures/figure2_overlap_scarcity.png)

### Figure 3. Shared-only currently beats every shared-private variant tested

Within-family comparison among the canonical neural models. Panel A shows test
cosine similarity and Panel B shows test mean squared error. Reduced private
capacity improves the shared-private family relative to the default
shared-private model, but even the best exploratory variant (`private_dim=16`)
remains below shared-only on the frozen benchmark. This figure supports the
claim that shared-private remains exploratory rather than currently preferred.

Assets:
- [SVG](figures/figure3_shared_only_vs_shared_private.svg)
- [PNG](figures/figure3_shared_only_vs_shared_private.png)

### Figure 4. Threshold hypothesis schematic for future paired-data expansion

Conceptual illustration of the research program hypothesis, not empirical
evidence from the current paper. The figure visualizes the idea that explicit
shared/private structure may become helpful only after materially larger paired
overlap is available. The current benchmark point lies firmly in the low-overlap
scarcity regime, so the figure should be read as a roadmap for future testing,
not as confirmation of a crossover.

Assets:
- [SVG](figures/figure4_threshold_hypothesis.svg)
- [PNG](figures/figure4_threshold_hypothesis.png)

## Tables

### Table 1. Main benchmark results

Primary benchmark metrics for the frozen ladder. The table reports each model’s
role in the program, test cosine similarity, test mean squared error, condition
means where available, and paired evaluation group count. It is the main results
table for Paper 1 and should appear in the main text.

Suggested placement:
- main text

Asset:
- [TABLE_1_MAIN_RESULTS.md](tables/TABLE_1_MAIN_RESULTS.md)

### Table 2. Claims and evidence boundary

Current claim boundary for Paper 1. The table distinguishes which claims are
already supported, which are only partially supported, and which remain outside
the justified evidence boundary until materially larger paired data are
available. It is especially useful for reviewers because it makes the paper’s
negative-result discipline explicit.

Suggested placement:
- main text

Asset:
- [TABLE_2_CLAIMS_EVIDENCE_BOUNDARY.md](tables/TABLE_2_CLAIMS_EVIDENCE_BOUNDARY.md)

### Table 3. Reproducibility and artifact contract

The official workflow/config/artifact contract for the benchmark. This table
summarizes the canonical acquisition, preparation, benchmark, and subsystem
surfaces that make the paper reproducible and that separate the practical
Animus lane from the exploratory threshold-testing lane. This table is a good
candidate for appendix or supplement placement if main-text space is tight.

Suggested placement:
- appendix or supplement if main-text space is tight

Asset:
- [TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md](tables/TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md)
