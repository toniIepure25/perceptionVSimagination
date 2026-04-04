# Paper 1 Submission Package Plan

This document maps the current repository manuscript assets onto a submission
package for the primary target venue, **Imaging Neuroscience**.

## Venue alignment

Paper 1 should now be treated as an **Imaging Neuroscience** submission unless
the venue decision is explicitly reopened. The package should therefore read as:

- a careful imaging-neuroscience benchmark/evidence paper
- explicit about overlap scarcity and empirical limits
- strong on supplement/reproducibility support
- disciplined about keeping the shared-private threshold hypothesis motivated,
  not confirmed

## Main manuscript

- primary manuscript source:
  [`PAPER_1_FULL_DRAFT.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_FULL_DRAFT.md)
- section-level source files:
  - [`PAPER_1_ABSTRACT.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_ABSTRACT.md)
  - [`PAPER_1_INTRODUCTION.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_INTRODUCTION.md)
  - [`PAPER_1_RELATED_WORK.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_RELATED_WORK.md)
  - [`PAPER_1_METHODS.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_METHODS.md)
  - [`PAPER_1_BENCHMARK_SETUP.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_BENCHMARK_SETUP.md)
  - [`PAPER_1_RESULTS.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_RESULTS.md)
  - [`PAPER_1_DISCUSSION.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_DISCUSSION.md)
  - [`PAPER_1_LIMITATIONS.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_LIMITATIONS.md)
  - [`PAPER_1_CONCLUSION.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_CONCLUSION.md)

## Appendix / supplement

- appendix manuscript:
  [`PAPER_1_APPENDIX.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_APPENDIX.md)
- appendix planning source:
  [`PAPER_1_APPENDIX_PLAN.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_APPENDIX_PLAN.md)

Recommended supplement emphasis for this venue:

- frozen overlap details
- exact configs and commands
- artifact roots and provenance
- export contract summary
- figure/table provenance

## Figures and tables

- figures:
  [`docs/paper1/figures/`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/figures)
- tables:
  [`docs/paper1/tables/`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/tables)
- captions:
  [`PAPER_1_CAPTIONS.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_CAPTIONS.md)

## References

- bibliography source:
  [`PAPER_1_REFERENCES.bib`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_REFERENCES.bib)
- human-readable reference lock:
  [`PAPER_1_REFERENCES.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/PAPER_1_REFERENCES.md)

## Reproducibility support

- frozen evidence state:
  [`CURRENT_EVIDENCE_FREEZE.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/CURRENT_EVIDENCE_FREEZE.md)
- benchmark ladder:
  [`BENCHMARK_LADDER.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/BENCHMARK_LADDER.md)
- reproducibility:
  [`REPRODUCIBILITY.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/REPRODUCIBILITY.md)
- Animus subsystem surface:
  [`ANIMUS_CORE_DECODER.md`](/home/tonystark/Desktop/perceptionVSimagination/docs/ANIMUS_CORE_DECODER.md)

## Conversion steps still required

1. Move the manuscript and appendix into the chosen venue template.
2. Convert markdown figure/table placement into venue-specific floats.
3. Apply venue-specific bibliography style.
4. Finalize acknowledgments, author metadata, and ethics/disclosure sections as
   required by the target venue.

## Near-submission package checklist for this pass

- main manuscript prose tightened for flow and reduced repo-internal tone
- figure and table references made explicit in the main draft
- appendix strengthened as a real supplement boundary rather than a loose dump
- claims re-audited against the frozen evidence docs
- Imaging Neuroscience positioning kept primary while preserving a benchmark-oriented fallback

## Package boundary

This plan is intentionally content-first rather than template-first. The goal
of the current pass is to make the manuscript package scientifically ready for
conversion without changing the evidence boundary.

The intended near-submission bundle is:

- `PAPER_1_FULL_DRAFT.md` as the main scientific narrative
- `PAPER_1_APPENDIX.md` as the supplement-ready reproducibility package
- `PAPER_1_CAPTIONS.md` plus `docs/paper1/tables/` and `docs/paper1/figures/`
  as the figure/table package
- `PAPER_1_REFERENCES.bib` as the bibliography source for venue conversion
