# Perception-to-Imagery Decoding Under Overlap Scarcity: A Reproducible ROI-First Benchmark

## Abstract

Perception-to-imagery decoding is a scientifically important but empirically
constrained problem: the key question is not only whether visual content can be
decoded from brain activity, but whether externally driven perception and
internally generated imagery should be modeled with shared and private
structure. In practice, that question is often confounded by limited paired
perception/imagery overlap, heterogeneous subject geometry, and benchmark drift
between simple baselines and higher-capacity neural models. We present a
reproducible ROI-first multi-subject benchmark platform for canonical
preparation, benchmarking, transfer analysis, and export. We evaluate a fixed
ladder consisting of Ridge, a shared-only neural decoder, and shared-private
variants on the current max-available real overlap dataset, comprising 94 rows,
5 shared paired stimulus IDs, and 1 held-out paired evaluation group across
subjects `subj02`, `subj03`, `subj05`, and `subj07`. In this extreme
low-overlap regime, Ridge is strongest overall and shared-only outperforms all
tested shared-private variants. The best current shared-private model, using
reduced private capacity (`private_dim=16`), improves over the default
shared-private model but remains below shared-only. These results do not
support a claim that explicit disentanglement already improves
perception-to-imagery decoding performance. Instead, they motivate a threshold
hypothesis: shared-private structure may require materially larger paired
overlap before it becomes beneficial. The platform therefore serves as a
disciplined benchmark and reproducibility surface for future threshold studies,
while also identifying shared-only as the current practical neural subsystem.

## 1. Introduction

Decoding visual content from fMRI has become a central problem in neuro-AI,
spanning reconstruction, representation prediction, and retrieval-oriented
mapping from brain activity to learned embedding spaces (e.g., Allen et al.,
2022; Chang et al., 2019; Hebart et al., 2023; Radford et al., 2021). Within that
broader landscape, perception-to-imagery decoding introduces a particularly
interesting challenge. The scientific question is not only whether externally
driven visual content can be decoded, but how it relates to internally
generated imagery, and whether the two should be modeled as sharing neural
content structure while retaining domain-specific variance (Pearson, 2019;
Horikawa & Kamitani, 2017).

That question is harder to answer than it first appears. Real paired
perception/imagery datasets often provide only a small overlap between the
stimuli observed during perception and those available during imagery.
Multi-subject integration further complicates the picture because raw voxel
spaces differ across subjects, ROI resources are heterogeneous, and evaluation
protocols can drift across model families. Under such conditions, it is easy to
mistake architectural ambition for scientific progress.

We argue that overlap scarcity must be treated as a first-class empirical
regime. Rather than assuming that explicit shared/private disentanglement should
help by default, we build a reproducible ROI-first benchmark that makes the
current evidence boundary explicit. Our platform canonicalizes preparation,
preflight, overlap assembly, ROI materialization, training, evaluation,
transfer analysis, and export. It also formalizes a fixed benchmark ladder that
separates an external linear reference baseline, a practical shared-only neural
decoder, and exploratory shared-private variants.

The resulting benchmark is real but small. On the largest paired dataset
currently recoverable in the accessible environment, we obtain only 94 rows, 5
shared paired stimulus IDs, and 1 held-out paired evaluation group across four
subjects. On this benchmark, Ridge is the strongest model overall. Shared-only
is the strongest canonical neural baseline. Shared-private variants improve
modestly under reduced private capacity, but do not overtake shared-only. These
results are scientifically useful because they clarify what can be claimed
today: the platform is operational and trustworthy, but explicit disentanglement
is not yet supported as the best model family in this low-overlap regime.
Figure 2 and Table 2 summarize the scarcity boundary and the resulting claim
discipline that make those conclusions necessarily modest.

This leads to a threshold-style research question. The current paper does not
claim that shared-private structure wins. Instead, it establishes a benchmark
and evidence freeze that motivate the next decisive experiment: rerunning the
same ladder on materially larger paired data without changing the target space
or benchmark meaning.

The remainder of the paper positions the benchmark relative to prior decoding
and imagery work, details the ROI-first pipeline, reports the frozen benchmark
ordering, and clarifies what the present evidence does and does not support.

## 2. Related Work

Visual fMRI decoding has advanced rapidly through reconstruction pipelines,
retrieval-style models, and embedding-prediction approaches (Naselaris et al.,
2011; Allen et al., 2022; Radford et al., 2021). Large-scale visual fMRI
resources such as NSD, BOLD5000, and THINGS-data have also helped stabilize the
benchmark landscape (Allen et al., 2022; Chang et al., 2019; Hebart et al.,
2023). That literature shows that visual information is decodable from brain
activity, but it often emphasizes reconstruction quality or large-model
interfacing rather than the paired perception/imagery question.

Mental imagery work, by contrast, has long studied overlap and dissociation
between perception and internally generated visual states (Pearson, 2019;
Horikawa & Kamitani, 2017). Recent results also highlight domain-dependent
temporal differences between imagery and perception rather than simple identity
(Dijkstra et al., 2018). This makes perception-to-imagery transfer and
cross-domain decoding a meaningful question rather than a trivial extension of
perception-only decoding.

Simple linear decoding remains a standard and often strong reference point in
fMRI research (Naselaris et al., 2011). That makes strong simple baselines
especially important in overlap-scarce paired settings, where model capacity
can easily outrun the available supervision. Shared/private or domain-separated
architectures are conceptually attractive because the problem itself suggests
both common content and domain-specific variance. Similar ideas appear in
domain separation and representation-learning work more broadly (Bousmalis et
al., 2016). However, conceptual fit does not guarantee empirical benefit in a
small paired-data regime, which is why we treat shared-private modeling as a
hypothesis family rather than a winner by default.

Finally, our benchmark sits within a tradition of multi-subject fMRI analysis
that avoids naive raw-voxel comparability assumptions. Multi-subject alignment
often requires subject-aware abstractions rather than direct voxel identity
matching (Haxby et al., 2011). The ROI-first canonicalization used here belongs
to that broader line of work, while also supporting a practical downstream
subsystem boundary for future Animus-facing brain-to-latent applications.

## 3. Methods

The benchmark is built around a canonical ROI-first workflow for
perception/imagery decoding. The workflow surface includes preparation,
preflight, training, evaluation, transfer analysis, ROI analysis, and export.
Canonical preparation constructs a mixed perception/imagery index from
perception indices, imagery indices, and a fixed target cache in the
`vit_l14_image_768` space. Canonical overlap assembly identifies shared
perception/imagery stimulus coverage, propagates valid splits, and materializes
ROI features into branch-ready serialized fields. In the current paper, this
preparation surface is instantiated on the NSD-style public benchmark resources
integrated through the repo (Allen et al., 2022).

The benchmark uses three ROI groups: early visual, ventral visual, and
metacognitive. These ROI-materialized features form the official multi-subject
input path. Raw full-fMRI vectors are treated as optional auxiliary context,
which avoids invalidly forcing equal raw dimensionality across subjects. This
ROI-first design is not only an engineering choice but part of the benchmark’s
scientific validity and is aligned with the broader motivation for
representation-based multi-subject alignment (Haxby et al., 2011).

All models predict the same target representation, `vit_l14_image_768`, so that
the comparison stays focused on decoder structure rather than target-space
drift (Radford et al., 2021). The benchmark ladder is intentionally small.
Ridge serves as the
external low-data reference baseline. Shared-only serves as the best current
canonical neural baseline and the practical `Animus Core Decoder`. The
shared-private family remains exploratory, with `private_dim=16` representing
the strongest shared-private variant tested so far.

The evaluation surface includes content cosine similarity and content MSE as the
primary metrics, along with imagery/perception mean cosine summaries,
transfer-focused evaluation on held-out paired subsets, and paired-group counts.
Domain accuracy is recorded only as a secondary diagnostic for shared-private
models and is not treated as a headline result in the current paper. Appendix B
and Appendix E summarize the exact workflow commands, artifact roots, and the
practical export contract used in Paper 1.

## 4. Benchmark Setup

Paper 1 uses the current max-available canonical paired benchmark. The prepared
mixed index contains 94 rows, 5 shared paired `nsdId`s, and 1 held-out paired
evaluation group across subjects `subj02`, `subj03`, `subj05`, and `subj07`.
This benchmark is derived from canonical perception indices and the full public
NSD-Imagery release after canonical filtering and overlap assembly.

The benchmark remains scientifically small even though it is operationally
real. The public imagery source has already been integrated and still yields
only five shared paired IDs in the current accessible environment. This makes
the present benchmark suitable for an evidence-freeze paper, but not yet for a
strong positive claim about when explicit shared/private disentanglement should
help. Figure 2 visualizes this data ceiling, and Table 3 summarizes the
official workflows, configs, and benchmark artifacts that define the paper's
reproducible comparison surface. Appendix A and Appendix B collect the exact
overlap details, commands, and artifact roots. Within the larger benchmark
landscape, this places Paper 1 closer to a constrained paired-overlap study
than to a general large-scale visual resource paper (Allen et al., 2022; Chang
et al., 2019; Hebart et al., 2023).

![Figure 2. The current benchmark is constrained by severe overlap scarcity](figures/figure2_overlap_scarcity.png)

The comparison surface is frozen. All main models use the same target space,
the same ROI-first input contract, the same prepared mixed index, and the same
evaluation surface. This allows the benchmark ordering to be interpreted as a
meaningful empirical result rather than an artifact of configuration drift.

## 5. Results

The current benchmark yields a clear ordering. Ridge is the strongest model
overall, with test cosine `0.55199` and test MSE `0.001167`. Shared-only is the
strongest canonical neural baseline, with test cosine `0.13596` and test MSE
`0.002250`. The strongest shared-private variant tested so far, shared-private
with `private_dim=16`, achieves test cosine `0.10784` and test MSE `0.002323`,
improving over the default shared-private model but remaining below shared-only.
Figure 1 and Table 1 summarize the frozen ordering and the full metric bundle.

![Figure 1. Frozen benchmark ladder on the current low-overlap dataset](figures/figure1_benchmark_ladder.png)

The default shared-private model reaches test cosine `0.06927` and test MSE
`0.002424`. Additional controls show that reducing private capacity helps
modestly, while disabling the domain head alone does not rescue the model.
Figure 3 isolates the within-family comparison. These findings are more
consistent with a low-overlap mismatch than with a broken-infrastructure
explanation.

![Figure 3. Shared-only currently beats every shared-private variant tested](figures/figure3_shared_only_vs_shared_private.png)

The main empirical message is therefore not a positive shared-private result.
Instead, it is a disciplined low-overlap finding: simple models dominate under
the present data ceiling, and shared-only is currently the correct canonical
neural baseline. The shared-private family remains an important hypothesis
family, but not the current winner. Table 1 should therefore be read as a
benchmark-ordering result, not as evidence that the shared-private question has
already been settled in general.

## 6. Discussion

These results matter because they convert an ambiguous modeling problem into a
disciplined evidence state. The current benchmark supports the narrower
conclusion that shared-private disentanglement is not yet justified as the best
model family in this regime. That is a scientifically useful outcome, even
though it is not the headline many disentanglement papers would prefer. Table 2
summarizes what is supported, what is only partially supported, and what still
requires larger paired data.

The results also sharpen the next research question. We do not need to ask
whether shared-private models can be built or run on real data; that has already
been demonstrated. The real question is whether there exists a paired-overlap
threshold beyond which shared-private structure begins to outperform shared-only
and linear baselines. The current paper motivates that threshold hypothesis but
does not claim to have confirmed it. Figure 4 is therefore presented as a
conceptual roadmap rather than as empirical evidence.

![Figure 4. Threshold hypothesis schematic for future paired-data expansion](figures/figure4_threshold_hypothesis.png)

At the same time, the benchmark clarifies the practical subsystem story.
Shared-only is already strong enough to serve as the current `Animus Core
Decoder` lane. This separation between practical and exploratory lanes is
important for a repository that aims to be both a real subsystem and a research
program. It also remains compatible with the broader imagery literature, which
supports meaningful overlap between imagery and perception without implying that
the same decoder structure should already dominate in a tiny paired benchmark
(Pearson, 2019; Dijkstra et al., 2018).

For an Imaging Neuroscience audience, the main value of that practical lane is
not that it changes the scientific ranking, but that it demonstrates how a
reproducible benchmark can still yield a usable subsystem while preserving
strict empirical modesty.

## 7. Limitations

The benchmark is limited by extreme paired-data scarcity. It contains only 94
rows, 5 shared paired IDs, and 1 held-out paired evaluation group. As a result,
the benchmark is not large enough to support strong claims about transfer
generalization or about the neuroscientific meaning of private latents.

The current public paired-data source is also already exhausted for the present
benchmark. Accordingly, the main next bottleneck is external paired-data
acquisition rather than additional tuning on the same public benchmark (Allen
et al., 2022).

Finally, the paper does not attempt to make vividness, confidence, subjective
reality, or stimulus-vs-percept claims. Those questions remain future-phase and
require richer data.

## 8. Conclusion

We introduced a reproducible ROI-first benchmark platform for multi-subject
perception-to-imagery fMRI decoding and used it to establish an honest evidence
state under overlap scarcity. Ridge is the strongest current baseline,
shared-only is the best current canonical neural model and practical Animus Core
Decoder, and shared-private `private_dim=16` is the strongest exploratory
disentanglement variant tested so far but does not overtake shared-only.

The next decisive experiment is straightforward: acquire a materially larger
paired perception/imagery dataset and rerun the same ladder unchanged. If
shared-private improves under that larger regime, the threshold hypothesis gains
support. If it does not, the field gains an equally valuable result about when
simpler models remain the correct default for perception-to-imagery decoding.
Appendix A through Appendix E summarize the frozen benchmark assets, commands,
artifact paths, and export contract needed to carry that next-stage rerun
forward without changing the comparison surface.
