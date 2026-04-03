# Paper 1 Appendix

This appendix assembles the reproducibility and benchmark details that support
the main manuscript, *Perception-to-Imagery Decoding Under Overlap Scarcity: A
Reproducible ROI-First Benchmark*. It is written to stay within the frozen
evidence boundary documented in
[CURRENT_EVIDENCE_FREEZE.md](/home/tonystark/Desktop/perceptionVSimagination/docs/CURRENT_EVIDENCE_FREEZE.md).

## Appendix A. Frozen dataset and overlap details

Paper 1 uses the current max-available canonical paired benchmark:

- subjects: `subj02`, `subj03`, `subj05`, `subj07`
- total mixed rows: `94`
- shared paired `nsdId`s: `5`
- held-out paired evaluation groups: `1`
- target space: `vit_l14_image_768`

The subject-level overlap breakdown is:

| Subject | Shared paired IDs | Mixed rows |
| --- | ---: | ---: |
| `subj02` | 2 | 38 |
| `subj03` | 1 | 18 |
| `subj05` | 1 | 19 |
| `subj07` | 1 | 19 |

The benchmark is operationally real but still classified as `bootstrap_ready`,
not `paper_ready`, because the preflight threshold for paper-scale overlap is
not met. In the checked-in config, that threshold remains
`preparation.preflight.paper_pair_threshold: 32`.

The canonical prepared artifacts for the frozen benchmark are:

- mixed index:
  `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- overlap report:
  `outputs/canonical/prepared/full_imagery_overlap/report.json`
- overlap ID list:
  `outputs/canonical/prepared/full_imagery_overlap/overlap_nsd_ids.json`
- preflight report:
  `outputs/canonical/prepared/full_imagery_overlap/preflight.json`
- target cache:
  `outputs/targets/full_imagery_overlap_vit_l14_image_768.parquet`

## Appendix B. Official configs and workflow commands

### B.1 Frozen benchmark configs

Paper 1 is defined by three checked-in configs:

- external baseline:
  [`configs/canonical/max_available_overlap.yaml`](/home/tonystark/Desktop/perceptionVSimagination/configs/canonical/max_available_overlap.yaml)
- practical shared-only subsystem:
  [`configs/canonical/animus_core_decoder.yaml`](/home/tonystark/Desktop/perceptionVSimagination/configs/canonical/animus_core_decoder.yaml)
- primary exploratory threshold model:
  [`configs/canonical/threshold_shared_private_p16.yaml`](/home/tonystark/Desktop/perceptionVSimagination/configs/canonical/threshold_shared_private_p16.yaml)

### B.2 Canonical acquisition and preparation

Public imagery acquisition:

```bash
python -m fmri2img.workflows.acquire_public_nsd_imagery \
  --subjects all \
  --skip-stimuli \
  --output cache/nsd_imagery_full_all
```

Overlap assembly:

```bash
python -m fmri2img.workflows.prepare_overlap_bootstrap \
  --config configs/canonical/max_available_overlap.yaml \
  --overwrite-existing
```

Target preparation:

```bash
python -m fmri2img.workflows.prepare_targets \
  --config configs/canonical/max_available_overlap.yaml
```

Preflight:

```bash
python -m fmri2img.workflows.preflight_data \
  --config configs/canonical/max_available_overlap.yaml
```

### B.3 Frozen benchmark ladder commands

External Ridge baseline:

```bash
python -m fmri2img.workflows.run_legacy_ridge_baseline \
  --config configs/canonical/max_available_overlap.yaml
```

Animus Core Decoder:

```bash
python -m fmri2img.workflows.preflight_animus_core_decoder
python -m fmri2img.workflows.train_animus_core_decoder
python -m fmri2img.workflows.eval_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
python -m fmri2img.workflows.export_animus_core_decoder \
  --checkpoint outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/best_decoder.pt
```

Primary threshold-testing model:

```bash
python -m fmri2img.workflows.train_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --checkpoint outputs/research/threshold_shared_private_p16/train/full_imagery_overlap/best_decoder.pt
python -m fmri2img.workflows.export_for_animus \
  --config configs/canonical/threshold_shared_private_p16.yaml \
  --checkpoint outputs/research/threshold_shared_private_p16/train/full_imagery_overlap/best_decoder.pt
```

## Appendix C. Artifact bundle structure

The main benchmark artifacts used or produced by Paper 1 are:

| Stage | Primary path |
| --- | --- |
| Public imagery source | `cache/nsd_imagery_full_all/` |
| Canonical imagery indices | `cache/indices/imagery_full_all/{subject}.parquet` |
| Mixed overlap dataset | `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet` |
| Target cache | `outputs/targets/full_imagery_overlap_vit_l14_image_768.parquet` |
| Ridge metrics | `outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/metrics.json` |
| Shared-only train root | `outputs/animus/core_decoder/train/full_imagery_overlap_shared_only/` |
| Shared-only eval root | `outputs/animus/core_decoder/eval/full_imagery_overlap_shared_only/` |
| Shared-only export root | `outputs/animus/core_decoder/export/full_imagery_overlap_shared_only/` |
| Shared-private p16 train root | `outputs/research/threshold_shared_private_p16/train/full_imagery_overlap/` |
| Shared-private p16 eval root | `outputs/research/threshold_shared_private_p16/eval/full_imagery_overlap/` |
| Shared-private p16 export root | `outputs/research/threshold_shared_private_p16/export/full_imagery_overlap/` |

## Appendix D. Additional benchmark variants and controls

The main manuscript focuses on the frozen ladder. Additional shared-private
variants were retained as diagnostic controls:

| Variant | Role | Test cosine | Test MSE |
| --- | --- | ---: | ---: |
| Shared-private | Canonical hypothesis-family baseline | 0.06927 | 0.002424 |
| Shared-private `private_dim=8` | Exploratory recovery variant | 0.09595 | 0.002354 |
| Shared-private no-domain | Diagnostic control | 0.05907 | 0.002450 |

These controls are useful because they show that:

- the shared-private family is operational, not broken
- reducing private capacity helps modestly
- disabling the domain head alone does not rescue the model

They do **not** justify promoting shared-private over shared-only on the current
benchmark.

## Appendix E. Animus Core Decoder export contract

The practical shared-only lane is treated as the current `Animus Core Decoder`.
Its export manifest is expected to record:

- `subproject = animus_core_decoder`
- `decoder_role = practical_content_decoder`
- `stability_tier = current_default`
- `disentanglement_mode = shared_only`
- `source_interface_status = scaffolded`
- `confidence_interface_status = scaffolded`

The intended interface interpretation is:

- `content`: active
- `source`: scaffolded
- `confidence`: scaffolded

This practical export contract is deliberately narrower than the threshold
research story. Paper 1 does **not** claim that source or confidence decoding
is already validated for deployment.

## Appendix F. Evidence boundary summary

Supported now:

- reproducible ROI-first multi-subject benchmarking on real paired data
- Ridge as the strongest current overall reference baseline
- shared-only as the strongest current canonical neural baseline
- reduced private capacity as a meaningful but incomplete recovery direction

Partially supported:

- the idea that low-overlap regimes favor simpler decoders over explicit
  disentanglement
- the idea that private-capacity scaling is an important control knob

Not yet justified:

- that explicit shared-private disentanglement improves content decoding
- that the threshold hypothesis has been empirically confirmed
- that private latents already support strong neuroscientific interpretation
- that Paper 1 demonstrates vividness, confidence, or subjective-state decoding

Accordingly, Paper 1 should be read as an honest benchmark/evidence paper under
overlap scarcity, not as a positive disentanglement paper.

## Appendix G. Figure and table provenance

The Paper 1 figure assets are generated from frozen benchmark evidence by:

- [`scripts/paper1/build_paper1_assets.py`](/home/tonystark/Desktop/perceptionVSimagination/scripts/paper1/build_paper1_assets.py)

The source data used by that script is:

- [`docs/paper1/assets/paper1_source_data.json`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/assets/paper1_source_data.json)

The table assets are maintained as versioned markdown files under:

- [`docs/paper1/tables/`](/home/tonystark/Desktop/perceptionVSimagination/docs/paper1/tables)

This keeps the paper package reproducible without pretending that the present
benchmark is larger or more decisive than it is.

## Appendix H. Next decisive experiment

The next decisive experiment does not require a new benchmark definition. It
requires a larger paired dataset. Once such data are available, the fixed
comparison should be rerun unchanged:

1. Ridge
2. shared-only `Animus Core Decoder`
3. shared-private `private_dim=16`

That rerun is the natural bridge from Paper 1 to the stronger future threshold
paper.
