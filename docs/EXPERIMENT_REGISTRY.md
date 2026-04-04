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
