# Table 3. Reproducibility and artifact contract

| Benchmark stage | Official config | Official command | Primary artifact or output |
| --- | --- | --- | --- |
| Public imagery acquisition | n/a | python -m fmri2img.workflows.acquire_public_nsd_imagery --subjects all --skip-stimuli --output cache/nsd_imagery_full_all | cache/nsd_imagery_full_all/{metadata,betas,stimuli?} |
| Benchmark overlap assembly | configs/canonical/max_available_overlap.yaml | python -m fmri2img.workflows.prepare_overlap_bootstrap --config configs/canonical/max_available_overlap.yaml --overwrite-existing | outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet |
| Target cache | configs/canonical/max_available_overlap.yaml | python -m fmri2img.workflows.prepare_targets --config configs/canonical/max_available_overlap.yaml | outputs/targets/full_imagery_overlap_vit_l14_image_768.parquet |
| Preflight | configs/canonical/max_available_overlap.yaml | python -m fmri2img.workflows.preflight_data --config configs/canonical/max_available_overlap.yaml | outputs/canonical/prepared/full_imagery_overlap/preflight.json |
| External baseline | configs/canonical/max_available_overlap.yaml | python -m fmri2img.workflows.run_legacy_ridge_baseline --config configs/canonical/max_available_overlap.yaml | outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/metrics.json |
| Animus Core Decoder | configs/canonical/animus_core_decoder.yaml | python -m fmri2img.workflows.train_animus_core_decoder | outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt |
| Threshold-testing model | configs/canonical/threshold_shared_private_p16.yaml | python -m fmri2img.workflows.train_decoder --config configs/canonical/threshold_shared_private_p16.yaml | outputs/research/threshold_shared_private_p16/train/full_imagery_overlap/best_decoder.pt |
