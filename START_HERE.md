# Quick Start

This repository now has one canonical path: the shared/private perception-imagery decoder platform.

## Read first

- [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [docs/VALIDATION.md](docs/VALIDATION.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Canonical commands

Smoke check:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/shared_private_smoke.yaml
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
