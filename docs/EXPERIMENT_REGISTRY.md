# Experiment Registry

This file is the compact durable ledger of important experiments and runs.
Append short entries here when a run matters beyond the current working pass.

Use it for:

- benchmark-ladder runs
- Animus Core Decoder runs that matter operationally
- threshold-research runs
- data-acquisition runs that materially affect the benchmark or evidence state
- paper-relevant reruns worth citing later

How it relates to the other workflow surfaces:

- `Documentation.md`: current-pass notes, immediate decisions, and follow-up
- `PLANS.md`: multi-step strategy and active run programs
- `docs/EXPERIMENT_REGISTRY.md`: compact durable run ledger
- `docs/PROJECT_MASTER_LOG.md`: milestone-level repo memory and official state shifts

Logging rule:

- append an entry when the run is durable enough that you may need to find it
  later without reading session notes
- keep entries short and factual
- promote to `docs/PROJECT_MASTER_LOG.md` only when the run changes the durable
  repo story

## Registry format

Copy this block:

```md
## <experiment id>

- Date: YYYY-MM-DD
- Lane: Animus subsystem engineering | Threshold research | Data acquisition | Paper support
- Benchmark rung / role: <shared-only baseline | ridge reference | private-dim sweep | data expansion audit | etc.>
- Config: <checked-in config path plus key overrides if any>
- Dataset / prepared artifacts: <mixed index, target cache, or source report basis>
- Output / artifact path: <primary output directory or report>
- Status: planned | running | blocked | done | promoted
- Result summary: <1 short line of metrics or concrete outcome>
- Interpretation summary: <1 short line, evidence-disciplined>
- Promoted to evidence?: yes | no
```

## Recent entries

## EXP-2026-04-02-RIDGE-MAX-OVERLAP

- Date: 2026-04-02
- Lane: Threshold research
- Benchmark rung / role: Ridge reference baseline on the fixed max-available overlap benchmark
- Config: `configs/canonical/max_available_overlap.yaml`
- Dataset / prepared artifacts: `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- Output / artifact path: `outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/`
- Status: promoted
- Result summary: test cosine `0.55199`, test MSE `0.001167`
- Interpretation summary: Ridge remains the strongest current baseline on the frozen benchmark
- Promoted to evidence?: yes

## EXP-2026-04-02-SHARED-ONLY-MAX-OVERLAP

- Date: 2026-04-02
- Lane: Animus subsystem engineering
- Benchmark rung / role: shared-only canonical neural baseline on the fixed max-available overlap benchmark
- Config: `configs/canonical/max_available_overlap.yaml` with `model.disentanglement_mode=\"shared_only\"`, `model.use_domain_head=false`
- Dataset / prepared artifacts: `outputs/canonical/prepared/full_imagery_overlap/full_imagery_overlap_mixed_with_roi.parquet`
- Output / artifact path: `outputs/canonical/train/full_imagery_overlap_shared_only/`
- Status: promoted
- Result summary: test cosine `0.13596`, test MSE `0.002250`
- Interpretation summary: shared-only is the best current canonical neural baseline and supports the practical Animus lane
- Promoted to evidence?: yes

## EXP-2026-04-04-NOD-METADATA-ACQUISITION

- Date: 2026-04-04
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane public dataset bootstrap for `ds004496`
- Config: n/a; canonical acquisition command `./.venv/bin/python -m fmri2img.workflows.acquire_public_nod --output cache/public_datasets/ds004496`
- Dataset / prepared artifacts: OpenNeuro `ds004496` metadata mirror only
- Output / artifact path: `cache/public_datasets/ds004496/`
- Status: done
- Result summary: metadata-only Git clone completed on the live pod and wrote `acquisition_provenance.json`
- Interpretation summary: practical public-data path is now real and reproducible, but annexed imaging content and any training adapter remain future work
- Promoted to evidence?: no

## EXP-2026-04-04-NOD-PREPARED-INDEX

- Date: 2026-04-04
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane prepared-index build for the NOD `imagenet` common-session subset
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_index`
- Dataset / prepared artifacts: OpenNeuro `ds004496` metadata clone at `cache/public_datasets/ds004496/`
- Output / artifact path: `cache/indices/public_nod/imagenet_multisession_common_sessions.parquet`
- Status: done
- Result summary: built a `360`-row prepared index with status counts `{'incomplete': 324, 'missing_payload': 36}` and `0` usable rows
- Interpretation summary: the subset contract is now materialized, but payload resolution still blocks shared-only preparation
- Promoted to evidence?: no

## EXP-2026-04-04-NOD-MISSING-PAYLOAD-MANIFEST

- Date: 2026-04-04
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane exact payload manifest for the first unresolved NOD subset
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads`
- Dataset / prepared artifacts: OpenNeuro `ds004496` metadata clone plus the prepared index at `cache/indices/public_nod/imagenet_multisession_common_sessions.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_missing_payload_manifest.json`
- Status: blocked
- Result summary: built a `36`-row manifest for the `run-10` rows across `sub-01..sub-09` and `ses-imagenet01..04`, estimated about `8.23 GiB` of missing payloads, and confirmed that `git-annex` is missing on the live pod so materialization could not proceed; rerunning the prepared index left readiness unchanged at `{'incomplete': 324, 'missing_payload': 36}` with `0` usable rows
- Interpretation summary: the first exact NOD payload target is now explicit and bounded, but the pod image still blocks real payload retrieval and shared-only preparation remains unavailable
- Promoted to evidence?: no

## EXP-2026-04-04-NOD-ANNEX-ENABLEMENT-AND-RETRIEVAL-ATTEMPT

- Date: 2026-04-04
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane exact-subset annex enablement and retrieval attempt for the first unresolved NOD payload subset
- Config: n/a; pod enablement via `apt-get install -y --no-install-recommends git-annex`, canonical workflow `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize`
- Dataset / prepared artifacts: OpenNeuro `ds004496` metadata mirror clone plus the exact manifest at `cache/indices/public_nod/imagenet_missing_payload_manifest.json`
- Output / artifact path: `cache/indices/public_nod/imagenet_missing_payload_report.json`
- Status: blocked
- Result summary: enabled `git-annex` on the live pod, initialized annex state in the `ds004496` clone, and attempted retrieval for the exact `36`-row `run-10` subset; retrieval still failed because no annex remote is known to contain the requested keys, leaving the prepared index unchanged at `{'incomplete': 324, 'missing_payload': 36}` with `0` usable rows
- Interpretation summary: the local tooling blocker is resolved, but the current GitHub metadata mirror remains insufficient as a payload source; the next blocker is upstream annex availability, not local execution
- Promoted to evidence?: no

## EXP-2026-04-04-NOD-DIRECT-S3-EXACT-SUBSET-RETRIEVAL

- Date: 2026-04-04
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane exact-subset payload retrieval for the first usable NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_payloads --materialize --strategy direct_openneuro_s3`
- Dataset / prepared artifacts: OpenNeuro `ds004496` exact `36`-row manifest plus official public S3 source under `https://s3.amazonaws.com/openneuro.org/ds004496/`
- Output / artifact path: `cache/indices/public_nod/imagenet_missing_payload_retrieval_report.json`
- Status: done
- Result summary: downloaded `144` exact files for the fixed `run-10` subset (`36` BOLD, `36` confounds, `36` beta, `36` label) totaling about `8.23 GiB`; rerunning the prepared index changed readiness to `{'incomplete': 324, 'resolved': 36}` with `36` usable rows
- Interpretation summary: the first NOD subset is now genuinely payload-ready for later shared-only prep, but this remains a narrow practical Animus-lane data-readiness result and not threshold evidence
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-SHARED-ONLY-ADAPTER

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane downstream shared-only adapter over the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_adapter`
- Dataset / prepared artifacts: prepared index `cache/indices/public_nod/imagenet_multisession_common_sessions.parquet` with the resolved `run-10` subset
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet`
- Status: done
- Result summary: built a `36`-row adapter artifact for `sub-01..sub-09`, `ses-imagenet01..04`, `run-10`, with `all_payloads_resolved=true`, `adapter_ready=true`, `prep_ready=true`, and `training_ready=false`
- Interpretation summary: the first NOD slice is now packaged as a stable downstream shared-only prep artifact, but target-selection and ROI-materialization contracts still block honest model training
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-TARGET-SELECTION

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane canonical target-selection contract over the fixed resolved NOD adapter slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_selection`
- Dataset / prepared artifacts: adapter artifact `cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_target_selection.parquet`
- Status: done
- Result summary: built a deterministic `3600`-row trial-level target-selection artifact from the fixed `36` adapter rows, with `100` target rows per run, `3600` unique target identifiers, and `target_selection_ready=true`, `downstream_prep_ready=true`, `training_ready=false`
- Interpretation summary: the fixed NOD slice now has a stable target-selection contract for downstream prep, but a canonical target embedding cache, ROI-materialization contract, and shared-only train/eval config are still required before honest model training
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-TARGET-EMBEDDING-CONTRACT

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane canonical target-embedding cache contract over the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_target_embedding_cache`
- Dataset / prepared artifacts: target-selection artifact `cache/indices/public_nod/imagenet_run10_target_selection.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_target_embedding_manifest.parquet`
- Status: done
- Result summary: built a `3600`-row target-embedding manifest for the fixed `run-10` NOD slice with `3600` visible stimulus paths, `0` resolved JPEG payloads, `target_embedding_ready=false`, `downstream_prep_ready=false`, and `training_ready=false`
- Interpretation summary: the canonical NOD target-cache contract is now explicit and repo-usable, but the current live-pod stimulus JPEGs are still unresolved annex paths so real embeddings and honest shared-only training remain blocked
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-TARGET-CACHE

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane real canonical target cache for the fixed resolved NOD slice
- Config: n/a; canonical workflows `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_stimuli --materialize` and `./.venv/bin/python -m fmri2img.workflows.build_public_nod_target_embedding_cache`
- Dataset / prepared artifacts: target-embedding manifest `cache/indices/public_nod/imagenet_run10_target_embedding_manifest.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- Status: done
- Result summary: materialized all `3600` fixed-slice stimulus JPEGs from the official OpenNeuro public S3 path (`103592285` bytes, about `0.096 GiB`) and built a real `3600`-row `clip_target_768` cache with `target_embedding_ready=true`, `downstream_prep_ready=true`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice now has a real canonical target cache for later shared-only prep, but ROI materialization, dataset-side join logic, and a checked-in shared-only train/eval config are still required before honest training
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-JOIN-CONTRACT

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane dataset-side join contract for the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_join_contract`
- Dataset / prepared artifacts: shared-only adapter `cache/indices/public_nod/imagenet_run10_shared_only_adapter.parquet`, target-selection artifact `cache/indices/public_nod/imagenet_run10_target_selection.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet`
- Status: done
- Result summary: built a `3600`-row machine-readable join contract keyed by `pair_id` for the exact fixed NOD slice, with `join_ready=true`, `roi_ready=false`, `downstream_prep_ready=false`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice now has an explicit downstream join surface linking adapter rows, trial-level target selection, target cache rows, and future ROI keys without widening the slice or implying that neural-side materialization already exists
- Promoted to evidence?: no

## EXP-2026-04-05-NOD-ROI-CONTRACT

- Date: 2026-04-05
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane ROI materialization contract for the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_roi_materialization_contract`
- Dataset / prepared artifacts: join contract `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_roi_materialization_contract.parquet`
- Status: done
- Result summary: built a `36`-row verified ROI contract over the exact fixed NOD slice, confirming beta/label alignment across `3600` join rows while keeping `roi_ready=false`, `downstream_prep_ready=false`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice now has a precise neural-side contract for the required future ROI artifact, but the ROI-side materialized parquet and loader integration still do not exist and training remains honestly blocked
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-ROI-ARTIFACT

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane real ROI artifact for the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.materialize_public_nod_roi_artifact`
- Dataset / prepared artifacts: ROI contract `cache/indices/public_nod/imagenet_run10_roi_materialization_contract.parquet`, join contract `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- Status: done
- Result summary: built a real `3600`-row ROI parquet keyed by `pair_id` for the exact fixed NOD slice, with `3600` unique `pair_id`s, `roi_ready=true`, `downstream_prep_ready=false`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice now has a real neural-side artifact aligned to the join contract, using only the subject-universal atlas sources available across the slice rather than faking subject-specific floc masks
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-PREPARED-DATASET

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane prepared dataset artifact for the fixed resolved NOD slice
- Config: n/a; canonical workflow `./.venv/bin/python -m fmri2img.workflows.prepare_public_nod_shared_only_prepared_dataset`
- Dataset / prepared artifacts: join contract `cache/indices/public_nod/imagenet_run10_shared_only_join_contract.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`
- Output / artifact path: `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`
- Status: done
- Result summary: built a real `3600`-row prepared dataset keyed by `pair_id` for the exact fixed NOD slice, with full join/ROI/target-cache alignment, `downstream_prep_ready=true`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice is now machine-consumable end-to-end for downstream shared-only prep without widening the slice or implying that shared-only training has been validated yet
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-TRAINER-PREFLIGHT

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane trainer-ingestion preflight for the fixed resolved NOD slice
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.preflight_public_nod_shared_only_trainer --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml` and `./.venv/bin/python -m fmri2img.workflows.preflight_data --config configs/canonical/public_nod_imagenet_run10_shared_only.yaml --output outputs/public_nod/train/imagenet_run10_shared_only_preflight/preflight_data.json`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- Output / artifact path: `outputs/public_nod/train/trainer_preflight.json`; canonical preflight report `outputs/public_nod/train/imagenet_run10_shared_only_preflight/preflight_data.json`
- Status: done
- Result summary: validated that the canonical trainer path can ingest the exact fixed NOD slice end-to-end without widening scope. The live pod built a real `16`-sample trainer batch with ROI feature dims `{early_visual: 3, ventral_visual: 0, metacognitive: 3}`, produced a real `16 x 768` content-prediction forward packet, and marked `trainer_config_ready=true`, `preflight_ready=true`, and `training_ready=false`
- Interpretation summary: the fixed NOD slice now has a checked-in shared-only config and a real trainer-ingestion preflight surface, but this is still an operational readiness result only. No benchmark run, no leaderboard claim, and no evidence-facing interpretation changed
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-TRAINER-SMOKE

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane operational smoke validation for the fixed resolved NOD slice
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.train_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml` and `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`
- Output / artifact path: `outputs/public_nod/train/imagenet_run10_shared_only_smoke/`; smoke report `outputs/public_nod/train/imagenet_run10_shared_only_smoke/smoke_report.json`
- Status: done
- Result summary: the live pod completed a one-epoch fixed-slice `train_decoder` smoke run and wrote the canonical trainer artifacts `best_decoder.pt`, `config_snapshot.json`, `roi_summary.json`, `target_summary.json`, and `train_history.json`, plus a machine-readable `smoke_report.json` that marks `trainer_config_ready=true`, `preflight_ready=true`, `smoke_ready=true`, and `training_ready=false`
- Interpretation summary: this proves the canonical trainer path can create real output artifacts for the exact fixed NOD slice without widening scope, but the run remains operational smoke only. Loss values and checkpoints from this run are not benchmark evidence and do not change any evidence-facing interpretation
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-EVAL-EXPORT-SMOKE

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane post-train smoke validation for the fixed resolved NOD slice
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.eval_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`, `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`, and `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, smoke checkpoint `outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`
- Output / artifact path: export bundle `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`; eval/export smoke report `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- Status: partial
- Result summary: the live pod export smoke succeeded and wrote `best_decoder.pt`, `config_snapshot.json`, `manifest.json`, `decoder_card.json`, and `decoder_card.md` under the fixed smoke export path. The live pod eval smoke remained blocked and did not write `metrics.json`, `roi_summary.json`, or `resolved_roi_groups.json`; the machine-readable report marks `eval_smoke_ready=false`, `export_smoke_ready=true`, and `training_ready=false`
- Interpretation summary: this validates that the canonical export path can package the fixed NOD smoke checkpoint without widening scope, while also showing that the current canonical eval path still assumes paired perception/imagery conditions and is not yet operationally ready for this perception-only fixed slice. No benchmark claim or evidence-facing interpretation changed
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-EVAL-SMOKE-GUARD

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane canonical eval smoke completion for the fixed resolved NOD slice
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.eval_decoder --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt` and `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, smoke checkpoint `outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`
- Output / artifact path: eval bundle `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/`; eval/export smoke report `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- Status: done
- Result summary: after adding a perception-only-safe guard in canonical pair-metric computation, the live pod eval smoke wrote `metrics.json`, `roi_summary.json`, and `resolved_roi_groups.json` for the fixed NOD slice. The regenerated eval/export smoke report marks `eval_smoke_ready=true`, `export_smoke_ready=true`, and `training_ready=false`, with `pair_metrics.available=false`, `present_conditions=["perception"]`, and `missing_conditions=["imagery"]`
- Interpretation summary: this makes canonical eval smoke operationally safe for the fixed perception-only NOD slice without inventing imagery rows or changing paired-slice semantics. The eval outputs remain smoke-only operational artifacts and do not change any evidence-facing interpretation
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-EVAL-TRANSFER-HARDENING

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane canonical eval/transfer hardening for public-data slices with incomplete condition coverage
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.eval_transfer --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt` and `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, smoke checkpoint `outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`
- Output / artifact path: transfer bundle `outputs/public_nod/transfer/imagenet_run10_shared_only_smoke/`; combined smoke report `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- Status: done
- Result summary: generalized the canonical condition-availability contract so transfer smoke now completes on the fixed perception-only NOD slice and writes `transfer_metrics.json` plus `per_trial_pairs.csv`. The live combined report records `present_conditions=["perception"]`, `missing_conditions=["imagery"]`, `pair_metrics.available=false`, `eval_smoke_ready=true`, `transfer_smoke_ready=true`, `export_smoke_ready=true`, and `training_ready=false`
- Interpretation summary: this hardens canonical post-train evaluation reuse for perception-only public-data slices without inventing imagery rows or changing any benchmark semantics. The resulting eval/transfer outputs remain operational artifacts only and do not change any evidence-facing interpretation
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-DOWNSTREAM-CONDITION-NORMALIZATION

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane downstream-consumption hardening for incomplete-condition public-data smoke artifacts
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt` and `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, smoke checkpoint `outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`
- Output / artifact path: combined smoke report `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`; export bundle `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`
- Status: done
- Result summary: normalized downstream condition semantics are now preserved across eval, transfer, and export consumption. The live combined report carries a shared `condition_semantics` block with `present_conditions=["perception"]`, `missing_conditions=["imagery"]`, `paired_metrics_available=false`, and `pair_metrics_available_from_payload=false`; the regenerated export `manifest.json` and `decoder_card.json` now preserve the same explicit condition contract
- Interpretation summary: this is operational hardening only. It makes downstream post-train consumers safer for perception-only public-data slices without changing paired-slice semantics, benchmark status, or any evidence-facing interpretation
- Promoted to evidence?: no

## EXP-2026-04-06-NOD-DOWNSTREAM-TARGET-NORMALIZATION

- Date: 2026-04-06
- Lane: Data acquisition
- Benchmark rung / role: practical Animus-lane downstream target-metadata hardening for fixed-slice smoke artifacts
- Config: `configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`; canonical workflows `./.venv/bin/python -m fmri2img.workflows.export_for_animus --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml --checkpoint outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`, `./.venv/bin/python -m fmri2img.workflows.report_public_nod_shared_only_eval_export_smoke --config configs/canonical/public_nod_imagenet_run10_shared_only_smoke.yaml`, and `./.venv/bin/python -m fmri2img.workflows.inspect_animus_export outputs/public_nod/export/imagenet_run10_shared_only_smoke --validate`
- Dataset / prepared artifacts: prepared dataset `cache/indices/public_nod/imagenet_run10_shared_only_prepared_dataset.parquet`, target cache `cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet`, ROI artifact `cache/indices/public_nod/imagenet_run10_roi_materialized.parquet`, smoke checkpoint `outputs/public_nod/train/imagenet_run10_shared_only_smoke/best_decoder.pt`
- Output / artifact path: export bundle `outputs/public_nod/export/imagenet_run10_shared_only_smoke/`; combined smoke report `outputs/public_nod/eval/imagenet_run10_shared_only_smoke/eval_export_smoke_report.json`
- Status: done
- Result summary: downstream post-train consumers now normalize target metadata from either `target_spec.name` or `target_spec.target_name`. The live export `manifest.json` now preserves `metadata.target_spec_normalized`, `decoder_card.json` now exposes normalized target metadata with `source_field_shape="target_name"`, and the live combined smoke report now carries a normalized top-level `target_spec` block and a normalized `export_smoke.normalized_target_spec`
- Interpretation summary: this is operational hardening only. It removes a downstream artifact-shape inconsistency without changing dataset scope, benchmark semantics, or any evidence-facing interpretation
- Promoted to evidence?: no
