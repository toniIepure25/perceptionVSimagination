# Quick Start

This repository now has one canonical path: the shared/private perception-imagery decoder platform.

Run canonical commands and validation from the project environment:

```bash
source .venv/bin/activate
```

or call tools explicitly through `./.venv/bin/python` and `./.venv/bin/pytest`.

## Read first

- [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- [docs/CURRENT_EVIDENCE_FREEZE.md](docs/CURRENT_EVIDENCE_FREEZE.md)
- [docs/BENCHMARK_LADDER.md](docs/BENCHMARK_LADDER.md)
- [docs/TOP_LEVEL_RESEARCH_DOSSIER.md](docs/TOP_LEVEL_RESEARCH_DOSSIER.md)
- [docs/ANIMUS_CORE_DECODER.md](docs/ANIMUS_CORE_DECODER.md)
- [docs/THRESHOLD_HYPOTHESIS.md](docs/THRESHOLD_HYPOTHESIS.md)
- [docs/PAPER_1_OUTLINE.md](docs/PAPER_1_OUTLINE.md)
- [docs/PAPER_1_CLAIMS_MAP.md](docs/PAPER_1_CLAIMS_MAP.md)
- [docs/PAPER_1_FIGURES_AND_TABLES.md](docs/PAPER_1_FIGURES_AND_TABLES.md)
- [docs/DATA_ACQUISITION_PROGRAM.md](docs/DATA_ACQUISITION_PROGRAM.md)
- [docs/EXTERNAL_DATA_INTEGRATION_PLAN.md](docs/EXTERNAL_DATA_INTEGRATION_PLAN.md)
- [docs/PROJECT_MASTER_LOG.md](docs/PROJECT_MASTER_LOG.md)
- [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [docs/VALIDATION.md](docs/VALIDATION.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Canonical commands

Practical Animus Core Decoder:

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.eval_animus_core_decoder --checkpoint ...
python -m fmri2img.workflows.export_animus_core_decoder --checkpoint ...
```

External paired-data acquisition:

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

Practical public-data acquisition:

```bash
python -m fmri2img.workflows.acquire_public_nod \
  --output cache/public_datasets/ds004496
```

Practical public-data inspection:

```bash
python -m fmri2img.workflows.inspect_public_nod
```

Practical public-data prepared index:

```bash
python -m fmri2img.workflows.prepare_public_nod_index
```

Primary threshold-testing research model:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
```

Smoke check:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
python -m fmri2img.workflows.report_public_nod_shared_only_smoke \
  --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml
```

Bootstrap prep:

```bash
python -m fmri2img.workflows.prepare_perception_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_imagery_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_mixed_index \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_roi_features \
  --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/shared_private_bootstrap.yaml
```

Train:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml
```

Evaluate:

```bash
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

Transfer evaluation:

```bash
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

ROI analysis:

```bash
python -m fmri2img.workflows.run_analysis \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

Animus export:

```bash
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/shared_private_bootstrap.yaml \
  --checkpoint outputs/canonical/train/shared_private_bootstrap/best_decoder.pt
```

## Legacy note

The older `scripts/` surface is still present for historical comparison, but it is no longer the official onboarding path.
