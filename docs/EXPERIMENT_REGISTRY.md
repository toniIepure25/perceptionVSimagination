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
