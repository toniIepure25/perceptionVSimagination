# Animus Integration

This repository is now the scientific backend for future Animus-style brain-to-latent systems.

## Current practical subsystem

The current practical subsystem for Animus is now the shared-only
`Animus Core Decoder`.

See:

- [ANIMUS_CORE_DECODER.md](ANIMUS_CORE_DECODER.md)

This is intentionally distinct from the shared-private threshold-testing
research track.

## Exported artifact bundle

`python -m fmri2img.workflows.export_for_animus ...` writes:

- decoder checkpoint
- manifest with artifact version
- target latent spec
- preprocessing spec
- ROI group spec
- workflow metadata

The export manifest now also carries:

- experiment identity and benchmark role
- Animus subproject metadata
- interface readiness for content, source/domain, and confidence

## Supported downstream assumptions

Animus consumers can assume:

- content target space is explicitly declared
- ROI grouping is serialized
- artifact versioning is explicit
- checkpoint path is stable within the export bundle
- shared-only exports can be treated as the current stable neural decoder lane

## Non-goals for this phase

- no SDXL/IP-Adapter export contract yet
- no claim of full subjective-state modeling without labels
- no claim of stimulus-vs-percept ambiguity support without additional datasets
- no claim that shared-private is already the current production decoder path

## Recommended handoff contract

Use the manifest as the source of truth and do not infer:

- target dimension
- ROI groups
- calibration semantics
- checkpoint provenance

from file names alone.
