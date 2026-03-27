# fmri2img

`fmri2img` is a research-grade platform for disentangling perception, imagery, and subjective experience from fMRI.

The repository keeps the historical package name for compatibility, but the canonical project direction is now:

- shared-vs-private latent disentanglement for perception vs imagery
- 768-D ViT-L/14 content decoding
- optional vividness/confidence prediction when real labels exist
- ROI-resolved analysis
- reproducible train/eval/transfer/export workflows
- future Animus integration

## Canonical docs

- [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/ANIMUS_INTEGRATION.md](docs/ANIMUS_INTEGRATION.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/VALIDATION.md](docs/VALIDATION.md)
- [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Official workflow surface

Smoke-test the checked-in canonical fixture:

```bash
python -m fmri2img.workflows.train_decoder --config configs/canonical/shared_private_smoke.yaml
```

Prepare a real bootstrap run:

```bash
python -m fmri2img.workflows.prepare_perception_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_imagery_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_targets --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_mixed_index --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.prepare_roi_features --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.preflight_data --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.train_decoder --config configs/canonical/shared_private_bootstrap.yaml
python -m fmri2img.workflows.eval_decoder --config configs/canonical/shared_private_bootstrap.yaml --checkpoint ...
python -m fmri2img.workflows.eval_transfer --config configs/canonical/shared_private_bootstrap.yaml --checkpoint ...
python -m fmri2img.workflows.run_analysis --config configs/canonical/shared_private_bootstrap.yaml --checkpoint ...
python -m fmri2img.workflows.export_for_animus --config configs/canonical/shared_private_bootstrap.yaml --checkpoint ...
```

## Canonical modeling direction

The official model family is the shared-private multitask decoder:

- ROI-specific branch encoders
- `z_shared`, `z_perception_private`, `z_imagery_private`
- content head in `vit_l14_image_768`
- optional domain head
- optional vividness/confidence head

## Legacy note

Historical perception-only, feature-first, and generation-first paths remain in the repository for comparison and reproduction, but they are not the current source of truth.
