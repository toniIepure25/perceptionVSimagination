# Paper 1 Abstract

## Draft

Perception-to-imagery decoding is a scientifically important but empirically
constrained problem: the key question is not only whether visual content can be
decoded from brain activity, but whether externally driven perception and
internally generated imagery should be modeled with shared and private
structure. In practice, that question is often confounded by limited paired
perception/imagery overlap, heterogeneous subject geometry, and benchmark drift
between simple baselines and higher-capacity neural models. We present a
reproducible ROI-first benchmark platform for multi-subject
perception-to-imagery fMRI decoding that canonicalizes preparation, preflight,
training, evaluation, transfer analysis, and export within a single workflow
surface. We evaluate a fixed benchmark ladder on the current max-available
real-data overlap set, consisting of 94 rows, 5 shared paired stimulus IDs,
and 1 held-out paired evaluation group across subjects `subj02`, `subj03`,
`subj05`, and `subj07`. The ladder includes a Ridge baseline, a shared-only
neural decoder, and a family of shared-private models. In this extreme
low-overlap regime, Ridge is the strongest overall model, while the shared-only
decoder is the strongest canonical neural baseline. The best current
shared-private variant, using reduced private capacity (`private_dim=16`),
improves over the default shared-private model but remains below shared-only.
These results do not support a claim that explicit disentanglement currently
improves perception-to-imagery decoding performance. Instead, they motivate a
threshold hypothesis: shared-private structure may require materially larger
paired overlap before it becomes beneficial. Beyond the benchmark contribution,
the same platform yields a practical shared-only decoder lane for downstream
Animus integration and a disciplined research testbed for future threshold
studies under richer paired data.
