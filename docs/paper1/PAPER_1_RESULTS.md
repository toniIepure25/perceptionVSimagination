# Paper 1 Results

## Draft

### Frozen benchmark ordering

The current max-available paired benchmark yields a clear and evidence-based
ordering:

1. `Ridge`
2. `Shared-only`
3. `Shared-private private_dim=16`
4. Other shared-private variants

This ordering should be treated as the central empirical result of Paper 1.
The benchmark is real, reproducible, and operationally validated, but it is
also scientifically small. The key value of the result lies in clarifying what
can and cannot be claimed under a scarce paired-data regime.

Figure 1 and Table 1 summarize the frozen ladder directly. Figure 2 makes the
current evidence ceiling explicit: `94` rows, `5` shared paired
`nsdId`s, and only `1` held-out paired evaluation group.

![Figure 1. Frozen benchmark ladder on the current low-overlap dataset](figures/figure1_benchmark_ladder.png)

### Main benchmark metrics

`Ridge`, the external low-data reference baseline, is the strongest model
tested on the current benchmark, achieving a test cosine of `0.55199` and a
test MSE of `0.001167`.

`Shared-only`, the best current canonical neural baseline and practical Animus
Core Decoder, achieves a test cosine of `0.13596` and a test MSE of `0.002250`.
Although substantially below Ridge, it clearly outperforms the shared-private
family under the current benchmark conditions.

`Shared-private private_dim=16`, the best current exploratory disentanglement
variant, achieves a test cosine of `0.10784` and a test MSE of `0.002323`. This
variant improves over the default shared-private model, but it remains below
shared-only and therefore does not justify promoting explicit disentanglement as
the best current canonical model.

Table 1 also records the currently available imagery/perception mean cosine
summaries and paired-group counts. Those additional fields reinforce the same
interpretation: the benchmark is operational, but the held-out paired evidence
is still too small to justify a stronger claim than the current frozen ordering.

For completeness, the default `Shared-private` model achieves a test cosine of
`0.06927` and a test MSE of `0.002424`, while the `private_dim=8` recovery
variant reaches a test cosine of `0.09595` and a test MSE of `0.002354`. A
no-domain control performs worse still, indicating that simply disabling the
domain head is not sufficient to rescue shared-private in the present regime.

### Shared-only versus shared-private

The most important within-family comparison is between shared-only and
shared-private. Figure 3 isolates that comparison. Shared-only remains clearly
ahead of every tested
shared-private variant on the current benchmark. This result matters because it
separates two questions that are often conflated. The first question is whether
the canonical neural platform is operational; the answer is yes. The second
question is whether explicit shared/private structure already improves decoding
on the available paired data; the answer is no.

![Figure 3. Shared-only currently beats every shared-private variant tested](figures/figure3_shared_only_vs_shared_private.png)

This distinction is scientifically valuable. It means that the current platform
is not failing because of broken infrastructure, but because the present data
regime appears to favor simpler inductive bias. In other words, the benchmark
does not support the strong claim that explicit disentanglement already helps
here, even if it is not yet large enough to decide whether disentanglement
could help later.

### Shared-private recovery evidence

The private-capacity sweep provides a limited but useful signal. Reducing
private capacity improves the shared-private family relative to the default
shared-private model, with `private_dim=16` outperforming both the default and
the `private_dim=8` variant. This suggests that private-latent capacity is a
meaningful control knob under low-data conditions. However, the improvement is
not large enough to overturn the benchmark ordering, and it should therefore be
interpreted as diagnostic evidence rather than as support for the claim that
shared-private is already the preferred model.

### Transfer and domain diagnostics

Transfer and domain-related quantities remain secondary in the current paper.
The held-out paired evaluation set contains only one usable paired group, so
these summaries are too statistically weak to support a major scientific claim.
Accordingly, the present paper treats domain accuracy and transfer behavior as
operational diagnostics rather than as headline evidence for or against
disentanglement.

### What the results support

The current results support three conclusions. First, an ROI-first canonical
benchmark for multi-subject perception/imagery decoding is feasible and
reproducible on real data. Second, in the present low-overlap regime, simple
models are favored: Ridge is strongest overall and shared-only is the strongest
canonical neural model. Third, the results motivate a threshold hypothesis for
disentanglement rather than confirming it. The current data justify a
disciplined benchmark/evidence paper, but not a positive shared-private win
paper. The supported, partially supported, and not-yet-justified claim
boundaries are summarized in Table 2.
