# Paper 1 Benchmark Setup

## Draft

### Benchmark identity

Paper 1 is built around the current max-available canonical paired benchmark.
The goal of this benchmark is not to demonstrate a state-of-the-art win, but to
establish a controlled and honest evidence regime for
perception-to-imagery decoding under overlap scarcity.

### Data regime

The fixed dataset currently contains:

- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- total rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired evaluation groups: `1`

The overlap set is derived from canonical perception indices and the full public
NSD-Imagery release after canonical filtering and overlap assembly. The public
imagery acquisition path is reproducible, but in the present accessible
environment it still yields only five overlapable paired IDs after canonical
filtering. This fact is not a minor implementation inconvenience; it is the
central empirical constraint under which the benchmark must be interpreted.
Within the broader benchmark landscape, this makes Paper 1 closer to a paired
overlap study built on top of NSD-style public resources than to a generic
large-scale visual dataset paper (Allen et al., 2022; Chang et al., 2019;
Hebart et al., 2023).

Figure 2 visualizes this scarcity regime. Table 3 and Appendix B summarize the
official workflow/config/artifact contract that keeps the benchmark
reproducible despite its small empirical scale.

### Benchmark roles

The ladder is frozen as follows:

1. `Ridge`
   External low-data reference baseline.
2. `Shared-only`
   Best current canonical neural baseline and practical `Animus Core Decoder`.
3. `Shared-private private_dim=16`
   Best current shared-private exploratory hypothesis model.

Additional shared-private variants are treated as diagnostic controls rather
than headline benchmark entries.

### Preparation and readiness

The canonical preparation surface includes:

- imagery-index preparation
- overlap assembly
- target-cache preparation
- ROI materialization
- preflight classification

The current benchmark is considered real and operationally validated, but only
`bootstrap_ready`, not `paper_ready`, because the paired overlap remains too
small for a strong paper-scale scientific claim about disentanglement benefit.

### Controlled comparison surface

The benchmark keeps the following fixed across the main comparison:

- target space: `vit_l14_image_768`
- ROI-first input contract
- mixed perception/imagery prepared index
- evaluation metrics
- split semantics

This is important because it prevents the comparison from drifting into a
moving-target contest between unrelated preprocessing choices, target spaces, or
loss setups.

### Why this benchmark is still valuable

Although the dataset is underpowered for a strong positive disentanglement
claim, it is large enough to answer a narrower and still important question:
what is currently justified to say about decoder structure under real paired
overlap scarcity? In that sense, the benchmark serves as both a resource and an
evidence freeze. It tells us which model is strongest now, which ideas remain
exploratory, and what additional data is needed before the main threshold
hypothesis can be tested fairly.
