# Real Bootstrap Report

## Summary

This document records two related milestones for the canonical shared-private decoder on the live `orchestraiq-jupyter` pod:

- the first successful live mixed-condition bootstrap run
- the first fully fresh rerun trained from the rebuilt canonical preparation artifacts

The current checked-in reference config for this path is:

- `configs/canonical/multisubj_overlap_bootstrap.yaml`

This is an operational and scientific bootstrap, not a paper-scale result.

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

Prepared artifacts for the bootstrap path included:

- overlap mixed index with ROI features
- overlap-specific target cache at `outputs/targets/overlap_multisubj_vit_l14_image_768.parquet`
- preflight report reaching `bootstrap_ready`
- canonical training checkpoint
- canonical eval and transfer metrics
- Animus export bundle

The official output locations are:

- training checkpoint: `outputs/canonical/train/multisubj_overlap_bootstrap/best_decoder.pt`
- eval metrics: `outputs/canonical/eval/multisubj_overlap_bootstrap/metrics.json`
- transfer metrics: `outputs/canonical/transfer/multisubj_overlap_bootstrap/transfer_metrics.json`
- analysis summary: `outputs/canonical/analysis/multisubj_overlap_bootstrap/roi_resolved_summary.json`
- export manifest: `outputs/canonical/export/multisubj_overlap_bootstrap/manifest.json`
- comparison baseline: `outputs/canonical/baselines/multisubj_overlap_ridge_legacy/metrics.json`

As of the current repository state, the bootstrap prep path has been hardened so this run no longer depends on previously prepared overlap-era imagery parquet files. The official `prepare_imagery_index` workflow can now rebuild the subject imagery indices from the live pod's split metadata/beta layout and record source provenance alongside the canonical filtered outputs.

## Original Live Run

The first successful live bootstrap run validated the platform end to end on real data before the preparation path was fully rebuilt from canonical artifacts.

Observed metrics from that original run were:

- content cosine: `0.0348`
- content cosine std: `0.0236`
- content MSE: `0.00251`
- imagery cosine mean: `0.0350` over `16` samples
- perception cosine mean: `0.0338` over `3` samples
- paired transfer groups in the eval split: `1`
- mean imagery-minus-perception cosine gap: `0.00128`
- domain accuracy: `0.3158`

That run established operational viability, but it was not yet the official fresh canonical baseline because the overlap artifacts were still inherited from an earlier prepared state.

## Fresh Canonical Baseline Rerun

The current official fresh baseline was retrained from the rebuilt canonical prep artifacts.

Important operational note:

- the H100 was occupied by another high-memory process during the fresh rerun
- to avoid interference, the canonical retrain was executed in CPU-safe mode with the same checked-in config and rebuilt artifacts

Fresh canonical metrics:

- content cosine: `0.00685`
- content cosine std: `0.01977`
- content MSE: `0.002586`
- imagery cosine mean: `0.00574` over `16` samples
- perception cosine mean: `0.01275` over `3` samples
- paired transfer groups in the eval split: `1`
- mean imagery-minus-perception cosine gap: `-0.00700`
- domain accuracy: `0.5263`

Fresh canonical training history:

- epoch 1: train loss `25.37`, validation cosine `-0.0247`
- epoch 2: train loss `23.07`, validation cosine `-0.0278`
- epoch 3: train loss `22.42`, validation cosine `-0.0253`
- epoch 4: train loss `20.59`, validation cosine `-0.0243`
- epoch 5: train loss `20.08`, validation cosine `-0.0235`

The negative validation cosine on this fresh rerun is weak but believable at this scale. It is more consistent with an underdetermined tiny-data regime than with a broken execution path.

## Train-History Note

The original live run was completed before the training-history cosine fix in this repository pass.

That means the stored `train_history.json` field named `val_content_cosine` from the original live run should be interpreted as `cosine - 1`, not raw cosine. Checkpoint selection was still correct, because maximizing `cosine - 1` is equivalent to maximizing cosine, but the recorded history field was misleading.

The fresh canonical rerun reported above was executed after that fix and records the true validation cosine directly.

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

The original run's small positive cosine and the fresh rerun's near-zero cosine are both plausible for an extremely small bootstrap run with only 4 paired stimuli. The signal is weak and unstable, but not obviously pathological. Nothing in the metrics suggests suspiciously high performance or obvious leakage.

### Domain accuracy

`0.3158` on the original run and `0.5263` on the fresh rerun should not be interpreted as meaningful findings. The held-out split was tiny and imbalanced, with `16` imagery samples and `3` perception samples. At this scale, domain accuracy is mostly a stability diagnostic. These numbers say the domain head is not interpretable yet, not that the shared/private architecture is invalid.

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

For a concrete comparison against a simple legacy baseline on the same data, see:

- `docs/TINY_OVERLAP_BASELINE_COMPARISON.md`

Only after overlap scale improves should the next major effort focus on replacing the atlas-union bootstrap ROI groups with the final paper-grade ROI decomposition.
