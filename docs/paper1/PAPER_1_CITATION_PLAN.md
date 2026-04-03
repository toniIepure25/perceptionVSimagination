# Paper 1 Citation Plan

This document maps the current Paper 1 draft to the references it should cite.
It is intentionally conservative: confirmed anchor references are listed
directly, while uncertain or still-missing references are left as explicit
placeholders rather than being guessed.

The current bibliography lock now lives in:

- [PAPER_1_REFERENCES.md](PAPER_1_REFERENCES.md)
- [PAPER_1_REFERENCES.bib](PAPER_1_REFERENCES.bib)

## Confirmed anchor references

1. `Allen et al., 2022`
   `A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence.`
   Use for:
   - Natural Scenes Dataset context
   - large-scale visual fMRI benchmark framing
   - motivation for natural-image representation decoding

2. `Radford et al., 2021`
   `Learning Transferable Visual Models From Natural Language Supervision.`
   Use for:
   - CLIP / `vit_l14_image_768` target-space motivation
   - representation-learning context for embedding prediction

3. `Horikawa and Kamitani, 2017`
   `Generic decoding of seen and imagined objects using hierarchical visual features.`
   Use for:
   - perception-versus-imagery decoding precedent
   - hierarchical-feature decoding framing

4. `Pearson, 2019`
   `The human imagination: the cognitive neuroscience of visual mental imagery.`
   Use for:
   - mental imagery overview
   - introduction/discussion framing for imagery as a scientific target

## Candidate references that still need bibliographic verification or are still optional

- `Low-data decoder comparison literature`
  Candidate use:
  - support for the claim that simple/linear models can remain strong in low-data regimes
  Status:
  - now optional; the manuscript no longer depends on a separate generic
    low-data citation beyond the core decoding literature

- direct dataset-host citation for the public imagery release
  Candidate use:
  - benchmark acquisition-source citation if a final venue expects it
  Status:
  - still optional and unresolved

## Section-by-section mapping

### Abstract

External citations are optional in the abstract unless the target venue expects
them. Keep the abstract mostly citation-light and self-contained.

### Introduction

Essential:
- `Allen et al., 2022`
- `Radford et al., 2021`
- `Pearson, 2019`
- `Horikawa and Kamitani, 2017`
- `Chang et al., 2019`
- `Hebart et al., 2023`

Claims needing citation:
- visual content can be decoded into learned representation spaces
- imagery is scientifically important and partially overlapping with perception
- target-space choice is grounded in modern representation learning

### Related Work

Essential:
- `Allen et al., 2022`
- `Radford et al., 2021`
- `Pearson, 2019`
- `Horikawa and Kamitani, 2017`
- `Dijkstra et al., 2018`
- `Naselaris et al., 2011`
- `Bousmalis et al., 2016`
- `Haxby et al., 2011`
- `Chang et al., 2019`
- `Hebart et al., 2023`

Still missing and should be filled before submission:
- only optional extras if the target venue needs them; the manuscript now has a
  sufficient verified literature backbone for a citation-grounded draft

### Methods

Essential:
- `Radford et al., 2021`
- `Allen et al., 2022`
- `Haxby et al., 2011`

Claims needing citation:
- the target space derives from CLIP-style embedding learning
- the benchmark derives from the current public NSD-style imagery/perception resources
- the ROI-first contract is motivated by multi-subject comparability constraints

### Benchmark Setup

Essential:
- `Allen et al., 2022`
- `Chang et al., 2019`
- `Hebart et al., 2023`

Claims needing citation:
- source-dataset provenance
- benchmark acquisition / public release context

### Results

Mostly internal evidence; external citations are minimal here.

Use:
- internal figure/table references
- no literature citations unless comparing directly against prior reported
  perception/imagery benchmark structure

### Discussion

Essential:
- `Pearson, 2019` is useful if the discussion briefly reconnects to imagery as a
  cognitive-neuroscience target

Optional:
- a verified imagery-versus-perception overlap/dynamics citation

Do not cite speculative sources to support unsupported claims. Discussion
citations should contextualize the open question, not overstate the current
result.

## Draft sections already using anchor citations

- [PAPER_1_INTRODUCTION.md](PAPER_1_INTRODUCTION.md)
- [PAPER_1_RELATED_WORK.md](PAPER_1_RELATED_WORK.md)
- [PAPER_1_METHODS.md](PAPER_1_METHODS.md)
- [PAPER_1_BENCHMARK_SETUP.md](PAPER_1_BENCHMARK_SETUP.md)
- [PAPER_1_FULL_DRAFT.md](PAPER_1_FULL_DRAFT.md)

## Citation hygiene rules for Paper 1

- Prefer a small number of strong, relevant references over a broad list.
- Keep Paper 1 as a benchmark/evidence paper, not a literature-maximal review.
- Do not cite future Animus goals as if they are already empirically supported.
- If a claim is currently unsupported, keep it labeled as future work rather
  than trying to rescue it with citation volume.
