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
- `decoder_card.json`
- `decoder_card.md`
- `manifest.json`
- target latent spec
- preprocessing spec
- ROI group spec
- workflow metadata

Surface roles:

- `decoder_card.json` / `decoder_card.md`
  Preferred quick human-facing inspection surface for downstream integration.
- `manifest.json`
  Full machine-readable contract for automation, validation, and exact export metadata.

The full manifest carries:

- experiment identity and benchmark role
- Animus subproject metadata
- interface readiness for content, source/domain, and confidence

The decoder card exposes the same practical summary in a smaller, easier-to-scan form.

## Quick inspection

Preferred quick path:

```bash
source .venv/bin/activate
python -m fmri2img.workflows.inspect_animus_export \
  outputs/animus/core_decoder/export/full_imagery_overlap_shared_only \
  --validate
```

Use this when you want a fast summary of:

- what decoder this is
- what benchmark role it has
- what evidence tier it has
- which interfaces are active vs scaffolded
- where the checkpoint and config snapshot live inside the bundle

It prints the decoder-card summary and, with `--validate`, checks that the
expected bundle files are actually present.

Manual inspection order:

1. open `decoder_card.md` for the fastest read
2. use `decoder_card.json` for structured quick inspection
3. use `manifest.json` when exact machine-readable metadata is needed

## Supported downstream assumptions

Animus consumers can assume:

- content target space is explicitly declared
- ROI grouping is serialized
- artifact versioning is explicit
- checkpoint path is stable within the export bundle
- shared-only exports can be treated as the current stable neural decoder lane

From the decoder card, a consumer should be able to tell immediately:

- decoder identity:
  `experiment.name`, `animus.subproject`, `animus.decoder_role`
- benchmark role:
  `experiment.benchmark_role`
- evidence tier:
  `experiment.evidence_tier`
- interface readiness:
  `interfaces.content`, `interfaces.source`, `interfaces.confidence`
- artifact locations:
  `artifacts.checkpoint`, `artifacts.config_snapshot`

## Non-goals for this phase

- no SDXL/IP-Adapter export contract yet
- no claim of full subjective-state modeling without labels
- no claim of stimulus-vs-percept ambiguity support without additional datasets
- no claim that shared-private is already the current production decoder path

## Recommended handoff contract

Use the decoder card for the first-pass human read. Use the manifest as the
full source of truth and do not infer:

- target dimension
- ROI groups
- calibration semantics
- checkpoint provenance

from file names alone.
