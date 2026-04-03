# Paper 1 Discussion

## Draft

The main value of the current benchmark is that it makes a hard scientific
situation legible. Perception-to-imagery decoding is often discussed as though
the main open question were architectural sophistication. Our results suggest a
different framing: under extreme paired-data scarcity, the more urgent question
is which claims are justified at all. The current benchmark shows that a strong
linear baseline remains difficult to beat, and that a shared-only neural model
is presently a better canonical baseline than explicit shared/private
disentanglement. Figure 2 and Table 2 make that limited evidence regime
explicit.

This negative or non-winning result should not be read as a failure of the
platform. On the contrary, it is evidence that the platform is now strong
enough to falsify an overoptimistic story. The canonical workflows, the
ROI-first multi-subject input contract, the public imagery acquisition path, and
the fixed-ladder comparison all worked well enough to reveal that shared-private
structure is not yet the best-supported modeling choice in the current regime.
That is a useful scientific outcome, especially in a literature where more
complex models can be attractive conceptually but difficult to evaluate fairly
when paired overlap is limited. It is also compatible with a broader
cognitive-neuroscience view in which imagery and perception overlap
meaningfully without being simply identical neural states (Pearson, 2019;
Dijkstra et al., 2018).

The current results also make the threshold hypothesis more precise. The
question is no longer “can we build a shared-private model?” or even “can we run
it on real data?” Both answers are already yes. The question is now “under what
paired overlap regime, if any, does explicit shared/private structure begin to
help more than it hurts?” That is a stronger and more disciplined scientific
question than the original broad ambition, because it turns an aspirational
architecture into a testable empirical program. Figure 4 is included for
exactly that purpose: it is a conceptual map of the research program, not an
empirical claim that the threshold has already been observed.

![Figure 4. Threshold hypothesis schematic for future paired-data expansion](figures/figure4_threshold_hypothesis.png)

The distinction between the practical and exploratory lanes is also important.
Shared-only is currently the strongest canonical neural baseline, and it is
therefore the right practical `Animus Core Decoder` lane. This does not mean
that shared-only is the final scientific answer. It means that the repository
can already provide a stable, exportable, ROI-first neural content decoder while
the shared-private family remains under investigation. In research programs that
aim to become real subsystems, that separation is a strength rather than a
compromise.

At the same time, the paper should be careful not to overinterpret the current
ordering. It would be premature to conclude that explicit disentanglement is
fundamentally the wrong modeling direction for perception-to-imagery decoding.
The current benchmark is too small for such a claim. What we can say is
narrower and stronger: in the present low-overlap regime, disentanglement is not
yet justified as the best-performing model family, and reduced private capacity
helps only modestly. Larger paired data are required before the threshold
hypothesis can be tested fairly.

That framing is what makes the current paper valuable. Rather than pretending to
have already solved the harder problem, it establishes a trustworthy baseline
state for the field and for the repository itself. It tells future work exactly
what must happen next: preserve the ladder, expand the paired data, and rerun
the benchmark without changing the target space or benchmark meaning.
