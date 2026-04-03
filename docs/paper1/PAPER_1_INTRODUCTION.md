# Paper 1 Introduction

## Draft

Decoding visual content from fMRI has become a central problem in
neuro-AI, spanning image reconstruction, semantic embedding prediction, and
cross-modal transfer from brain activity to learned representation spaces
(e.g., Allen et al., 2022; Chang et al., 2019; Hebart et al., 2023;
Radford et al., 2021).
Within that broader landscape, perception-to-imagery decoding poses a
distinct scientific challenge. The question is not only whether stimulus
content can be decoded from externally driven visual responses, but how that
content overlaps with internally generated mental imagery, and when the two
domains should be modeled as sharing structure versus expressing domain-specific
variance. This distinction matters for cognitive neuroscience, for practical
brain-to-latent systems, and for future efforts to decode richer subjective
experience (Pearson, 2019; Horikawa & Kamitani, 2017).

Despite that importance, paired perception/imagery benchmarking remains
fragile. Real datasets often provide only a small overlap between perception
trials and imagery trials for the same stimuli. Multi-subject integration is
made harder by unequal voxel dimensionality, inconsistent ROI availability, and
dataset-specific preprocessing assumptions. As a result, the empirical question
of whether explicit shared/private disentanglement helps perception-to-imagery
decoding is easy to ask and hard to test fairly. In low-overlap settings,
higher-capacity neural models can appear scientifically appealing while
remaining poorly calibrated against strong simple baselines.

We address that gap by treating overlap scarcity as a first-class scientific
condition rather than a nuisance to be hand-waved away. Our contribution is not
to claim that a more complex decoder already wins. Instead, we build a
reproducible ROI-first benchmark platform that makes the evidence boundary
explicit. The platform canonicalizes data preparation, target construction,
multi-subject batching, readiness checks, benchmark execution, and export. It
supports a fixed benchmark ladder consisting of an external linear reference
baseline, a shared-only neural decoder, and exploratory shared-private
variants. This separation is important: it allows the current best practical
neural subsystem to remain useful even while the more ambitious disentanglement
hypothesis remains under test.

The resulting benchmark is real but extremely small. On the largest paired
dataset currently recoverable in the accessible environment, we obtain a
max-available overlap set of 94 rows, 5 shared paired stimulus IDs, and only 1
held-out paired evaluation group across four subjects. On this benchmark, Ridge
is the strongest overall baseline. Among neural models, a shared-only decoder
outperforms all tested shared-private variants, while reduced private capacity
improves shared-private modestly without overturning the ordering. These
results are scientifically useful precisely because they are not a disguised
positive result: they suggest that in extreme low-overlap regimes, benchmark
discipline and simple baselines matter more than architectural ambition. Figure
2 visualizes the current scarcity regime directly.

This framing leads naturally to a threshold hypothesis. Rather than assuming
that explicit shared/private disentanglement should help by default, we argue
that its utility is likely contingent on paired overlap scale. The present paper
does not confirm that threshold. Instead, it establishes the benchmark,
documents the current evidence freeze, and motivates the next decisive
experiment: rerunning the same ladder on materially larger paired data without
changing the target space, task definition, or evaluation surface.

The paper makes four concrete contributions. First, we provide a reproducible
ROI-first platform for multi-subject perception/imagery decoding with canonical
preparation, preflight, training, evaluation, transfer analysis, and export.
Second, we formalize a fixed benchmark ladder that clearly separates a linear
reference baseline, a practical shared-only neural decoder, and exploratory
shared-private variants. Third, we report an honest low-overlap empirical
result: Ridge is strongest overall and shared-only is the best current
canonical neural baseline. Fourth, we show that these findings motivate a
threshold-style research program for disentanglement rather than a premature
claim that disentanglement already helps.

The remainder of the paper positions the benchmark relative to prior decoding
and imagery work, details the ROI-first methodology, reports the frozen
benchmark ordering, and closes by clarifying the evidence boundary that the
current dataset can support.
