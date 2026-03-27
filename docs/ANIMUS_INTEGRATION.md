# Animus Integration

This repository is now the scientific backend for future Animus-style brain-to-latent systems.

## Exported artifact bundle

`python -m fmri2img.workflows.export_for_animus ...` writes:

- decoder checkpoint
- manifest with artifact version
- target latent spec
- preprocessing spec
- ROI group spec
- workflow metadata

## Supported downstream assumptions

Animus consumers can assume:

- content target space is explicitly declared
- ROI grouping is serialized
- artifact versioning is explicit
- checkpoint path is stable within the export bundle

## Non-goals for this phase

- no SDXL/IP-Adapter export contract yet
- no claim of full subjective-state modeling without labels
- no claim of stimulus-vs-percept ambiguity support without additional datasets

## Recommended handoff contract

Use the manifest as the source of truth and do not infer:

- target dimension
- ROI groups
- calibration semantics
- checkpoint provenance

from file names alone.
