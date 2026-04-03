# Paper 1 Conclusion

## Draft

We presented a reproducible ROI-first benchmark platform for multi-subject
perception-to-imagery fMRI decoding and used it to evaluate a fixed benchmark
ladder under the current max-available low-overlap paired regime. The main
result is clear: Ridge is the strongest overall baseline, shared-only is the
best current canonical neural model and practical Animus Core Decoder, and
shared-private `private_dim=16` is the strongest exploratory disentanglement
variant tested so far but does not overtake shared-only.

These results matter because they convert an ambitious but underconstrained
modeling problem into a disciplined evidence state. The platform is operational,
the benchmark ladder is frozen, the practical and exploratory lanes are clearly
separated, and the main open scientific question is now sharply defined. Rather
than claiming that explicit disentanglement already helps, the present paper
shows that under overlap scarcity it is not yet supported as the best model
family, and that larger paired data are required to test the threshold
hypothesis fairly. That evidence boundary is part of the contribution, not an
embarrassing caveat.

The next decisive experiment is therefore straightforward: acquire a materially
larger paired perception/imagery dataset and rerun the same ladder unchanged:
Ridge, shared-only, and shared-private `private_dim=16`. If shared-private
begins to close the gap under that larger regime, the threshold hypothesis gains
support. If it does not, the field gains an equally valuable result: a clearer
understanding of when simpler models remain the right default for
perception-to-imagery decoding. Appendix A and Appendix B summarize the exact
frozen benchmark assets and commands needed to carry that next-stage rerun
forward without changing the comparison surface.
