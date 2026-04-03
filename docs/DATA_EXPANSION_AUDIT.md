# Data Expansion Audit

> Historical note: this document captures the step-by-step audit that led from
> the original `4`-pair ceiling to the later `5`-pair public-data ceiling.
> For the current actionable external-data program, use
> [DATA_ACQUISITION_PROGRAM.md](DATA_ACQUISITION_PROGRAM.md) and
> [EXTERNAL_DATA_INTEGRATION_PLAN.md](EXTERNAL_DATA_INTEGRATION_PLAN.md).

## Update After Full Acquisition

This audit was written before the full public NSD-Imagery source was actually downloaded into the live pod.

That later acquisition changed one major conclusion:

- the mounted imagery source was incomplete in the pod
- but the full public source still yields the same `720` source trials and the same `5` canonical `setB` `nsdId`s per subject

So the earlier “truncated source” hypothesis should now be read as historical, not final.

The current final state is:

- full public imagery metadata and beta bundles for `subj01..subj08` are now present under `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery_full_all`
- the overlap ceiling moved from `4` to `5` by adding `subj03`
- the new primary blocker is no longer imagery acquisition
- the new primary blocker is canonical multi-subject training, which still assumes one shared raw full-fMRI dimensionality across subjects

## Scope And Method

This audit focused on one operational question:

- what exact mounted or recoverable data is preventing the canonical overlap dataset from growing beyond the current `4` shared perception/imagery `nsdId`s?

The audit used:

- live read-only inspection of the `orchestraiq-jupyter` pod
- repo-local canonical prep reports and overlap artifacts
- mounted cache and preserved-cache directories under `/home/jovyan/local-data` and `/home/jovyan/work`
- old scripts and docs that reference expected imagery and NSD mount locations

No models, losses, or benchmark definitions were changed in this pass.

## Verified Inventory

### Mounted, Usable Sources

Perception-side sources:

- canonical perception indices in this repo for `subj01`, `subj02`, `subj05`, `subj07`
- perception indices in the sibling repo `/home/jovyan/work/FMRI2images/data/indices/nsd_index/subject=subj01..subj08`
- local preextracted perception cache for `subj02`, `subj05`, `subj07`:
  - `/home/jovyan/local-data/fmri2images_cache/preextracted/subject=subj02`
  - `/home/jovyan/local-data/fmri2images_cache/preextracted/subject=subj05`
  - `/home/jovyan/local-data/fmri2images_cache/preextracted/subject=subj07`
- preserved preextracted perception cache for `subj01`, `subj03`, `subj04`, `subj06`, `subj08`:
  - `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted/subject=subj01`
  - `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted/subject=subj03`
  - `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted/subject=subj04`
  - `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted/subject=subj06`
  - `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted/subject=subj08`

Imagery-side sources:

- shared imagery metadata roots:
  - `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery/metadata`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/metadata`
- currently mounted imagery beta bundles:
  - `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery/betas/subj01/betas_nsdimagery.nii.gz`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/betas/subj02/betas_nsdimagery.nii.gz`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/betas/subj05/betas_nsdimagery.nii.gz`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/betas/subj07/betas_nsdimagery.nii.gz`
- current imagery indices:
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/subj01.parquet`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/subj02.parquet`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/subj05.parquet`
  - `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery/subj07.parquet`

Stimulus/target support:

- NSD stimuli HDF5:
  - `/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`

### Verified Partial Or Legacy Sources

- stale `subj01` imagery parquet is partial and not canonical enough to use directly:
  - `720` rows total
  - only `160` rows with valid `nsdId`
  - mixes `imagery`, `attention`, and `perception`
- old imagery evaluation outputs exist under:
  - `/home/jovyan/local-data/perceptionVSimagination/outputs/reports/imagery/`
- these are useful provenance clues, but not official canonical prep inputs

### Missing-But-Expected Sources

- no imagery beta bundles were found anywhere on the pod for:
  - `subj03`
  - `subj04`
  - `subj06`
  - `subj08`
- no raw NSD perception `ppdata` tree was found for `subj01` under the mounted shared NSD root
- mounted raw perception `ppdata` for `subj02`, `subj05`, `subj07` is sparse and not obviously complete

## Why The Current Imagery Source Looks Truncated

Two pieces of evidence strongly suggest the currently mounted imagery source is a reduced bootstrap bundle, not a full NSD-Imagery release:

1. Canonical source reports for `subj02`, `subj05`, and `subj07` show only `720` source rows per subject before filtering.
2. The mounted imagery beta bundles are much smaller than the repo’s own download script expects:
   - mounted sizes: about `359M` to `449M` per subject
   - documented full-subject expectation in `scripts/download_nsd_imagery.py`: about `1.5 GB` per subject

The current canonical prep reports show:

- `288` imagery rows
- `288` attention rows
- `144` perception rows
- `240` rows per stimulus set `A`, `B`, `C`

After canonical filtering to `condition=imagery` and `stimulus_set=B`, each subject contributes only:

- `80` rows
- `5` unique `nsdId`s

So the present ceiling is not just “few mounted subjects.” The mounted imagery source itself is tiny.

## Subject-By-Subject Overlap Feasibility

| Subject | Perception Source Present | Imagery Beta Present | Canonical Imagery Rebuild Status | Overlap With Current Tiny Imagery IDs | Practical Status |
|---|---|---|---|---:|---|
| `subj01` | Yes | Yes | Likely rebuildable with the split-layout wrapper, but not yet normalized in the current canonical path | `0` | Does not break the current ceiling |
| `subj02` | Yes | Yes | Canonically rebuildable now | `2` | Already in the official overlap set |
| `subj03` | Yes | No | Blocked on missing imagery beta package | `1` if a matching tiny imagery package existed | Best small expansion candidate if only a tiny package can be acquired |
| `subj04` | Yes | No | Blocked on missing imagery beta package | `0` for the current tiny imagery ids | Only interesting if a fuller imagery package is available |
| `subj05` | Yes | Yes | Canonically rebuildable now | `1` | Already in the official overlap set |
| `subj06` | Yes | No | Blocked on missing imagery beta package | `0` for the current tiny imagery ids | Only interesting if a fuller imagery package is available |
| `subj07` | Yes | Yes | Canonically rebuildable now | `1` | Already in the official overlap set |
| `subj08` | Yes | No | Blocked on missing imagery beta package | `0` for the current tiny imagery ids | Only interesting if a fuller imagery package is available |

### Notes On The Overlap Column

The “overlap with current tiny imagery ids” column was computed against the only five `nsdId`s presently recoverable from the mounted imagery source:

- `28752`
- `30857`
- `53882`
- `61178`
- `65873`

Perception overlap with those five ids is:

- `subj01`: `0`
- `subj02`: `2`
- `subj03`: `1`
- `subj04`: `0`
- `subj05`: `1`
- `subj06`: `0`
- `subj07`: `1`
- `subj08`: `0`

This explains the current `4`-pair ceiling across `subj02/subj05/subj07`.

## Verified Blockers

1. The mounted imagery source is extremely small.
2. Only four imagery beta bundles exist on the pod at all.
3. `subj01` imagery does exist, but it does not overlap the current perception ids, so fixing it does not increase the current ceiling.
4. Subjects `subj03`, `subj04`, `subj06`, and `subj08` have perception-side support but no mounted imagery beta bundles.
5. The raw shared NSD `ppdata` mount is incomplete enough that it is not the fastest route to overlap expansion.

## Ranked Expansion Paths

### 1. Acquire Full Imagery Bundles For The Existing Imagery Subjects

Status:

- best practical next move

Why:

- the current imagery source itself is tiny
- expanding from the current mini bundles to full subject bundles should increase overlap far more than another small subject repair
- canonical prep already supports these subjects

Exact paths to acquire or mount:

- full imagery metadata:
  - `nsddata/experiments/nsdimagery/`
- full imagery betas:
  - `nsddata_betas/ppdata/subj01/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj02/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj05/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`
  - `nsddata_betas/ppdata/subj07/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/`

Expected payoff:

- likely the largest increase for the least engineering work
- likely moves the overlap set from `4` shared pairs to something genuinely informative

Effort:

- moderate

### 2. Acquire `subj03` Imagery As The Smallest Subject Expansion

Status:

- best small-step expansion if only one additional subject can be added quickly

Why:

- `subj03` already has perception-side support
- under the current tiny imagery-id regime it would likely add `1` more shared pair

Expected payoff:

- low if only another tiny imagery bundle is mounted
- potentially much higher if a full `subj03` imagery package exists

Effort:

- moderate

### 3. Acquire Imagery For `subj03/04/06/08` As A Higher-Upside Subject Expansion

Status:

- best high-upside move if the full NSD-Imagery release actually covers all `8` subjects

Evidence:

- `scripts/download_nsd_imagery.py` is written for `subj01..subj08`
- the sibling `FMRI2images` repo already has perception indices for all `8` subjects

Uncertainty:

- not verified against the remote source in this audit
- the pod currently mounts no imagery beta bundles for `subj03/04/06/08`

Expected payoff:

- potentially substantial
- could move the project from bootstrap-scale into a much more meaningful overlap regime

Effort:

- moderate to high

### 4. Canonicalize `subj01` Imagery From The Already Mounted Beta Bundle

Status:

- useful cleanup, not the best way to break the ceiling

Why:

- `subj01` does have a mounted imagery beta bundle
- but the currently recoverable imagery ids do not overlap its perception index

Expected payoff:

- near-zero for overlap size
- modest for coverage/accounting hygiene

Effort:

- trivial to moderate

### 5. Improve Or Complete Raw Perception `ppdata` Mounts

Status:

- secondary

Why:

- perception-side completeness matters for richer downstream experiments
- but it is not the main reason the current overlap ceiling is stuck at `4`

Expected payoff:

- low for immediate overlap expansion

Effort:

- moderate

## Storage And Effort Notes

Current relevant storage footprints on the pod:

- `/home/jovyan/local-data/perceptionVSimagination/cache/nsd_imagery`: about `1.5G`
- `/home/jovyan/local-data/perceptionVSimagination/cache/indices/imagery`: about `4.0G`
- `/home/jovyan/local-data/fmri2images_cache/preextracted`: about `4.5G`
- `/home/jovyan/work/preserved_localdata/fmri2images_moved/cache/preextracted`: about `7.2G`

Available free space is not the limiting factor:

- local NVMe under `/home/jovyan/local-data`: about `263G` free
- shared storage under `/home/jovyan/work`: about `184T` free

So the limiting factor is data presence and mount normalization, not disk space.

## Bottom Line

The single best next action is:

- replace or augment the current tiny imagery bundles with full NSD-Imagery subject packages, starting with the already-supported imagery subjects `subj01`, `subj02`, `subj05`, and `subj07`

If that is not immediately possible, the next-best smaller move is:

- acquire `subj03` imagery, because it is the only non-mounted subject that appears likely to add even a small amount of overlap under the current tiny-id regime
