# Paper 1 Limitations

## Draft

The present benchmark has several important limitations, and they should be
stated explicitly.

First, the paired overlap is extremely small. The current max-available real
benchmark contains only `94` rows and `5` shared paired `nsdId`s, with only `1`
held-out paired evaluation group. This means that the benchmark is large enough
to establish ordering and evidence boundaries, but not large enough to support
strong positive claims about the conditions under which shared-private
disentanglement should win.

Second, the current public paired-data source is already exhausted for the
benchmark as currently defined. The full public NSD-Imagery release has been
integrated canonically, and after overlap assembly it still yields only the
current five paired IDs. As a result, further progress on the main threshold
question now depends on acquiring a larger paired source rather than on further
massaging the same public data. In practical terms, the main public NSD-style
resource path is already integrated, and the remaining bottleneck is larger
paired overlap rather than missing benchmark plumbing (Allen et al., 2022).

Third, transfer and domain-related diagnostics remain underpowered. Because the
held-out paired set is so small, domain accuracy and transfer summaries cannot
support a strong scientific interpretation. They are useful operational
diagnostics, but they should not be elevated to primary evidence in the current
paper.

Fourth, the present results do not justify a strong neuroscientific
interpretation of private latents. The shared-private family remains
exploratory, and the current dataset is too small to distinguish meaningful
private structure from low-data burden or excess capacity. Accordingly, the
paper should avoid claims that the private branches have already isolated
neuroscientifically interpretable perception-private or imagery-private
representations.

Fifth, the current paper does not yet address vividness, confidence, subjective
reality strength, or stimulus-vs-percept ambiguity as main empirical questions.
Those are important future directions, but they require richer paired data and,
in several cases, richer labels. They should therefore remain explicitly
outside the main claim boundary of Paper 1.

Taken together, these limitations do not make the current benchmark
uninteresting. Instead, they define what kind of paper this can honestly be: a
benchmark/evidence paper under overlap scarcity, not a full positive
disentanglement paper.
