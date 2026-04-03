# Paper 1 Methods

## Draft

### Overview

The current benchmark is built around a canonical ROI-first workflow for
perception/imagery decoding. The workflow surface covers preparation, preflight,
training, evaluation, transfer analysis, ROI analysis, and export, and is
designed to keep dataset assumptions, artifact provenance, and evaluation
semantics explicit. The key methodological goal is not to maximize flexibility,
but to make the benchmark reproducible and interpretable under a scarce paired
data regime.

### Canonical data preparation

The preparation pipeline constructs a mixed perception/imagery index from
canonical perception indices, canonical imagery indices, and a fixed target
cache in the `vit_l14_image_768` space. Imagery preparation supports both
subject-rooted layouts and split metadata/beta layouts. Canonical overlap
assembly then identifies perception/imagery overlap under a shared stimulus
identifier, propagates valid splits, and materializes ROI features into
serialized branch-ready fields. A preflight stage classifies whether a run is
blocked, smoke-only, bootstrap-ready, or paper-ready, and surfaces missing
artifacts before training begins. In the current paper, this preparation surface
is instantiated on the NSD-style public benchmark resources integrated through
the repo (Allen et al., 2022).

### ROI-first multi-subject input contract

The benchmark uses three ROI groups:

- early visual
- ventral visual
- metacognitive

These are represented as branch-ready feature tensors rather than requiring a
single shared raw voxel dimensionality across subjects. This design is critical
for valid multi-subject training because the underlying raw full-fMRI vectors
can differ in shape across subjects. In the current canonical contract, raw
full-fMRI tensors are optional auxiliary context, while ROI-materialized
features are the official multi-subject input path. This choice is aligned with
the broader motivation for multi-subject representational alignment strategies
that avoid naive voxel-wise identity matching across subjects (Haxby et al.,
2011).

### Fixed target space

All models in the benchmark predict a shared target representation:
`vit_l14_image_768`. Fixing the target space prevents benchmark drift and keeps
the comparison focused on decoder structure rather than downstream embedding
choice. The same target cache is used across the ladder so that performance
differences reflect the model family rather than the supervision target. The
target-space choice itself follows modern visual representation learning based
on CLIP embeddings (Radford et al., 2021).

### Benchmark ladder

The current ladder is intentionally small and role-separable.

`Ridge` is the external low-data reference baseline. It is not the practical
Animus subsystem, but it is the strongest current scientific reference under the
present low-overlap regime.

`Shared-only` is the best current canonical neural baseline. It is also the
current `Animus Core Decoder`, meaning it serves as the practical neural
subsystem for export and downstream reuse. Importantly, shared-only is not
framed as a claim that disentanglement is unnecessary in principle; it is the
best supported neural model on current evidence.

`Shared-private private_dim=16` is the primary threshold-testing hypothesis
model. It represents the strongest shared-private variant tested so far, but
remains explicitly exploratory. Additional shared-private controls are used only
to diagnose failure modes and capacity sensitivity.

### Model families

The canonical neural model family uses ROI-specific branch encoders and a latent
structure that can be configured as either shared-only or shared-private. In
shared-only mode, the model collapses to a practical content decoder that
removes domain and vividness heads. In shared-private mode, it includes shared
and private latent structure plus optional domain-related components. The
present paper does not propose a new architecture. Instead, it evaluates how
these already-defined model families behave under a fixed low-overlap benchmark.

### Evaluation

The primary evaluation metrics are:

- content cosine similarity
- content mean squared error

Secondary summaries include:

- imagery mean cosine
- perception mean cosine
- paired evaluation group counts
- transfer evaluation on the held-out paired subset
- domain accuracy as a diagnostic-only quantity for shared-private models

Because the held-out paired set is extremely small in the current benchmark,
domain accuracy and transfer summaries are interpreted cautiously and never used
as the main evidence for a scientific claim.

### Practical subsystem and export

The shared-only path is additionally treated as the practical `Animus Core
Decoder`. Its export manifest records target-space metadata, ROI grouping,
checkpoint provenance, stability tier, and interface readiness for future
content, source, and confidence interfaces. This practical subsystem role is
important for downstream reuse, but it remains distinct from the research claim
about whether shared-private disentanglement is scientifically or empirically
superior. Appendix B and Appendix E summarize the exact workflow commands and
the practical export contract used in Paper 1.
