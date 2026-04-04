# Animus Core Decoder

This document formalizes the current practical decoder subsystem for Animus.

## Purpose

The `Animus Core Decoder` is the current reliable neural decoding engine for:

- content decoding into `vit_l14_image_768`
- ROI-first multi-subject inference
- stable train/eval/export workflows
- clean handoff into future Animus downstream systems

It is intentionally narrower than the full research program.

## Current model identity

Current practical model:

- shared-only canonical decoder

Why shared-only:

- it is the strongest current canonical neural baseline
- it is materially stronger than all tested shared-private variants on the
  current overlap benchmark
- it keeps the current subsystem honest while shared-private remains a
  threshold-testing research track

## What the Core Decoder is for

Use the Core Decoder when the priority is:

- dependable content decoding
- stable export for downstream systems
- minimal conceptual burden
- current-best neural behavior inside the canonical platform

Do not use the Core Decoder to claim that disentanglement has been validated.

## Official config

- [`configs/canonical/animus_core_decoder.yaml`](../configs/canonical/animus_core_decoder.yaml)

This config defines:

- `disentanglement_mode=shared_only`
- `use_domain_head=false`
- `use_vividness_head=false`
- Animus-facing export metadata
- dedicated train/eval/export output roots under `outputs/animus/core_decoder`

## Official commands

Train:

```bash
python -m fmri2img.workflows.train_animus_core_decoder
```

Preflight:

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
```

Evaluate:

```bash
python -m fmri2img.workflows.eval_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

Transfer:

```bash
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/animus_core_decoder.yaml \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

Export:

```bash
python -m fmri2img.workflows.export_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

The dedicated wrappers default to `configs/canonical/animus_core_decoder.yaml`
but still accept `--config` and `--override` when a controlled variant or smoke
fixture is needed.

## Artifact contract

The export manifest now records:

- experiment name and benchmark role
- Animus subproject identity
- decoder role
- stability tier
- interface readiness for:
  - content
  - source/domain
  - confidence

Current interface state:

- content: active
- source/domain: scaffolded
- confidence: scaffolded

The export bundle also writes a compact decoder card:

- `decoder_card.json`
- `decoder_card.md`

Use the decoder card for fast subsystem inspection. Use `manifest.json` as the
full machine-readable contract.

That means the export surface is ready for future source/confidence extensions
without pretending those heads are currently validated for deployment.

## Relation to the external baseline

Ridge remains the strongest external low-data reference baseline.

The Core Decoder is not intended to replace Ridge as a scientific reference. It
is the best current canonical neural subsystem for Animus-facing reuse.

## Relation to the threshold research track

The threshold research question is separate:

- when does shared-private disentanglement begin to beat shared-only?

That question remains open and belongs to the research lane, not the practical
Animus Core Decoder lane.

## Relation to external data acquisition

The next larger paired dataset should be integrated for the Core Decoder
without changing its role:

- shared-only remains the practical Animus lane
- Ridge remains the external scientific reference
- shared-private `private_dim=16` remains the primary exploratory threshold
  model

Larger perception-only datasets may later strengthen the Animus lane, but they
should not be confused with the paired threshold benchmark.

For the current external-data program, see:

- `docs/DATA_ACQUISITION_PROGRAM.md`
- `docs/EXTERNAL_DATA_INTEGRATION_PLAN.md`
- `docs/NOD_PUBLIC_DATASET.md`

The first checked-in public strengthening path for the practical lane is now
`ds004496` metadata acquisition through:

```bash
./.venv/bin/python -m fmri2img.workflows.acquire_public_nod \
  --output cache/public_datasets/ds004496
```

That surface is intentionally metadata-first. It establishes public provenance
and remote path discipline for the Animus lane without pretending that NOD
already belongs to the paired threshold ladder.

Current remote status:

- metadata clone completed on the live pod at
  `/home/jovyan/local-data/perceptionVSimagination/cache/public_datasets/ds004496`
- full annexed imaging content is still pending a separate storage-aware
  acquisition step
- the smallest viable NOD contract is now inspection-only:
  `imagenet`, multi-session subjects `sub-01..sub-09`, and visible
  `ciftify` beta/label derivatives for a future practical shared-only adapter

## Current limits

- current overlap benchmark remains extremely small
- source/domain output is not yet the practical decoder interface
- confidence/vividness is scaffolded only
- the Core Decoder should be treated as a stable content-decoding subsystem, not
  a full decoded-experience system
