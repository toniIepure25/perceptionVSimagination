# Paper 1 Related Work

## Draft

### fMRI decoding and reconstruction

A large body of work has shown that visual information can be decoded from fMRI
into semantic, embedding, or image-like outputs (Naselaris et al., 2011;
Allen et al., 2022; Radford et al., 2021). Large-scale visual fMRI resources
such as NSD, BOLD5000, and THINGS-data have also improved the benchmark
landscape for studying learned visual representations (Allen et al., 2022;
Chang et al., 2019; Hebart et al., 2023). Much of this
literature emphasizes reconstruction quality, retrieval accuracy, or transfer
into large pretrained representation spaces. That tradition is essential for
showing that fMRI contains decodable visual structure, but it does not by
itself resolve how internally generated imagery should be benchmarked against
perception, especially when paired overlap is limited. Our work is aligned with
this literature in target choice and decoder evaluation, but differs in placing
benchmark discipline and data-regime interpretation at the center of the
contribution.

### Perception versus imagery transfer

Mental imagery has long been studied as a partially overlapping but not
identical counterpart to perception (Pearson, 2019; Horikawa & Kamitani,
2017). Neuroimaging work has reported
shared recruitment of visual regions as well as domain-dependent differences in
signal strength, reliability, and top-down modulation (Pearson, 2019;
Dijkstra et al., 2018). Recent
machine-learning-oriented work has started to ask whether decoders trained on
perception generalize to imagery or whether explicit cross-domain modeling is
needed (Horikawa & Kamitani, 2017). The challenge is that many candidate
datasets remain
overlap-poor, making it hard to separate a true failure of shared
representations from a simple lack of paired evidence. Our benchmark is aimed
at exactly that ambiguity.

### Simple versus complex decoders in low-data regimes

Simple linear decoding remains a standard and often strong point of comparison
in fMRI research (Naselaris et al., 2011). That makes strong simple baselines
especially important in paired-overlap-scarce settings, where model capacity
can easily outrun the available supervision. Our current results place that
simple-versus-complex comparison directly inside a disciplined benchmark ladder
rather than treating it as an incidental implementation detail.

### Shared/private and disentanglement-based models

Shared/private or domain-separated architectures are a natural conceptual fit
for perception and imagery because the scientific problem itself suggests both
common content and domain-specific variance. Similar ideas appear in domain
separation and representation-learning work more broadly (Bousmalis et al.,
2016). However, conceptual fit does not imply empirical benefit in a
given regime. In low-overlap settings, private branches may simply add burden
and instability. Our work therefore does not present shared-private modeling as
already established. Instead, it treats the shared-private family as a
hypothesis family whose benefits must be measured against both linear and
shared-only baselines.

### Multi-subject alignment and ROI-based canonicalization

A recurring challenge in multi-subject fMRI work is that raw voxel spaces are
not directly comparable across subjects, and representational alignment often
requires subject-aware abstractions rather than naive voxel identity matching
(Haxby et al., 2011). Many pipelines handle this
with subject-specific models, strong anatomical restrictions, or feature spaces
that abstract away from raw voxel identity. Our ROI-first canonicalization
belongs to this broader tradition: rather than forcing multi-subject benchmark
validity through equal raw dimensionality, it materializes branch-ready ROI
features and treats raw full-fMRI vectors as optional auxiliary context. In the
present work, that design is not just an engineering convenience; it is part of
the scientific validity of the multi-subject benchmark.

### Stable subsystems and brain-to-latent interfaces

Practical brain-to-latent systems need more than an aspirational research model;
they need stable, exportable, and interpretable subsystem boundaries.
This matters for eventual downstream integration, including systems like Animus
that may later incorporate source routing, confidence estimation, or subjective
state modeling. Our contribution here is to separate the current practical lane
from the exploratory research lane. Shared-only serves as the present practical
neural subsystem, while shared-private remains the threshold-testing hypothesis
model. That separation makes the system useful now without overstating what has
been validated scientifically.
