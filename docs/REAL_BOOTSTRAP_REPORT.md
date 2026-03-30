# Real Bootstrap Report

## Summary

This document records the first successful real mixed-condition bootstrap run of the canonical shared-private decoder on the live `orchestraiq-jupyter` pod.

It is an operational and scientific bootstrap, not a paper-scale result.

The checked-in canonical config for this path is:

- `configs/canonical/multisubj_overlap_bootstrap.yaml`

## Run Context

- Environment: live `orchestraiq-jupyter` pod with 1x H100
- Subjects: `subj02`, `subj05`, `subj07`
- Data type: real perception + imagery overlap only
- Shared pairing basis: `nsdId`
- Shared overlap size: 4 paired `nsdId`s across the three subjects
- Target space: 768-D ViT-L/14 image embeddings
- ROI path: canonical ROI materialization, using atlas-union bootstrap groups rather than the final paper-grade ROI decomposition
- Vividness/confidence: unavailable in this dataset, so the vividness head was disabled for the effective run

## Artifacts Used

Prepared artifacts for the live bootstrap included:

- overlap mixed index with ROI features
- overlap-specific target cache at `outputs/targets/overlap_multisubj_vit_l14_image_768.parquet`
- preflight report reaching `bootstrap_ready`
- canonical training checkpoint
- canonical eval and transfer metrics
- Animus export bundle

The live pod output locations were:

- training checkpoint: `outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt`
- eval metrics: `outputs/canonical/eval/multisubj_overlap_bootstrap/metrics.json`
- transfer metrics: `outputs/canonical/transfer/multisubj_overlap_bootstrap/transfer_metrics.json`
- export manifest: `outputs/canonical/export/multisubj_overlap_bootstrap/manifest.json`

As of the current repository state, the bootstrap prep path has been hardened so this run no longer depends on previously prepared overlap-era imagery parquet files. The official `prepare_imagery_index` workflow can now rebuild the subject imagery indices from the live pod's split metadata/beta layout and record source provenance alongside the canonical filtered outputs.

## Observed Metrics

Observed metrics from the live bootstrap run:

- content cosine: `0.0348`
- content cosine std: `0.0236`
- content MSE: `0.00251`
- imagery cosine mean: `0.0350` over `16` samples
- perception cosine mean: `0.0338` over `3` samples
- paired transfer groups in the eval split: `1`
- mean imagery-minus-perception cosine gap: `0.00128`
- domain accuracy: `0.3158`

## Train-History Note

The original live run was completed before the training-history cosine fix in this repository pass.

That means the stored `train_history.json` field named `val_content_cosine` from the original live run should be interpreted as `cosine - 1`, not raw cosine. Checkpoint selection was still correct, because maximizing `cosine - 1` is equivalent to maximizing cosine, but the recorded history field was misleading.

Future reruns with the checked-in code now record the true validation cosine directly.

## Scientific Interpretation

The current bootstrap run is useful as a systems validation result, not as an efficacy result.

What the run shows:

- the canonical shared-private model can train on real mixed-condition data
- canonical ROI materialization can feed the real model path
- the 768-D target cache path works on real stimuli
- canonical preflight, train, eval, transfer, and export complete successfully on real data

What the run does not show:

- meaningful content decoding performance
- meaningful perception-to-imagery transfer generalization
- meaningful domain discrimination
- any vividness/confidence prediction result
- any conclusion about fine-grained ROI contributions

## Sanity Audit

### Content metrics

The positive but near-zero cosine is plausible for an extremely small bootstrap run with only 4 paired stimuli. It is weak signal, but not obviously pathological. Nothing in the metrics suggests suspiciously high performance or obvious leakage.

### Domain accuracy

`0.3158` domain accuracy is poor and should not be interpreted as a meaningful finding. The held-out split was tiny and imbalanced, with `16` imagery samples and `3` perception samples. At this scale, domain accuracy is mostly a stability diagnostic. The result says the domain head is not useful yet, not that the shared/private architecture is invalid.

### Transfer metrics

Transfer metrics were nearly identical to the base eval metrics because only one paired test group survived into the held-out split. That is a limitation of the dataset size, not evidence that transfer evaluation is broken.

### Leakage / degenerate-training check

The run does not show signs of obvious leakage:

- content metrics remain near chance rather than spuriously high
- the paired eval count is tiny but honest
- vividness was not silently faked
- ROI grouping was explicit and recorded

The run also does not show catastrophic collapse. Training loss decreased across epochs, although the validation signal remained extremely weak and noisy.

## Why This Is Still Only A Bootstrap

- only 4 shared `nsdId` pairs were available
- atlas-union ROI groups were used as a bootstrap fallback, not the final paper-grade ROI decomposition
- vividness/confidence labels were absent
- the transfer split contained only one paired group
- cross-subject overlap was sufficient for a systems test, but not for a research claim

## Recommended Next Step

The single best next experiment is:

- expand the overlap mixed-condition dataset beyond the current 4 shared pairs while keeping the same canonical shared-private model and workflow surface

Why this is the next best step:

- it directly improves statistical usefulness without changing the architecture
- it will immediately make transfer metrics and domain behavior more interpretable
- it avoids conflating data-scale issues with model-design issues

Only after overlap scale improves should the next major effort focus on replacing the atlas-union bootstrap ROI groups with the final paper-grade ROI decomposition.
