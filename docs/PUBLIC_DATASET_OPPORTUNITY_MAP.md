# Public Dataset Opportunity Map

This document is the repo-specific map of public dataset opportunities after
the current NSD-Imagery public path has been exhausted for the primary fixed
benchmark.

Use this file to answer:

- which public datasets matter now
- which lane they serve
- whether they preserve the meaning of the current fixed ladder
- which datasets should remain secondary benchmarks or future-paper assets

This document does **not** change the current frozen benchmark. It classifies
public datasets by how they fit the repo.

## Role definitions

- `Role A`: primary threshold-benchmark expansion
  Public paired perception/imagery data that could directly strengthen the
  current threshold question.
- `Role B`: practical Animus lane strengthening
  Public perception-oriented data that can strengthen the shared-only practical
  subsystem without answering the paired threshold question directly.
- `Role C`: future novel paper paths
  Public datasets that support later work on imagery, internally generated
  state, source routing, working memory, or consciousness-style decoding.

## Decision rule

Do not silently fold a non-NSD or perception-only dataset into the primary
threshold ladder.

If a dataset does not preserve the current ladder semantics, classify it
explicitly as one of:

- secondary benchmark
- Animus-only support dataset
- future-paper dataset

## Ranked candidates

| Rank | Dataset | Public ID / host | Task type | Role | Classification | Acquisition difficulty | Normalization difficulty | Likely payoff | Repo verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NSD-Imagery public release | OpenNeuro `ds004937` | paired perception/imagery extension with NSD-style identifiers | A | current canonical public source | low | low to moderate | already realized | keep as the canonical public source, but not the next decisive expansion |
| 2 | Visual imagery and false memory for pictures | legacy OpenfMRI / OpenNeuro `ds000203` | imagery and reality-monitoring task | A/C | secondary paired-imagery benchmark candidate | low to moderate | moderate to high | moderate | best immediate public paired-imagery candidate, but not a direct replacement for the NSD-style threshold ladder |
| 3 | Natural Object Dataset (NOD) | OpenNeuro `ds004496` | perception-only natural-image fMRI | B | Animus robustness dataset | moderate | moderate | high | best immediate public dataset for strengthening the practical Animus lane |
| 4 | THINGS-fMRI | OpenNeuro `ds004192` | perception-only object fMRI | B/C | Animus robustness dataset and future object-semantic benchmark | moderate | moderate | high | best medium-effort public object benchmark for the practical lane and future work |
| 5 | BOLD5000 | OpenNeuro `ds001499` | perception-only large-scale visual-image fMRI | B | Animus robustness dataset | moderate | moderate | medium to high | historically important and operationally useful, but lower priority than NOD for immediate strengthening |
| 6 | Covert consciousness under anesthesia | OpenNeuro `ds006623` | mental imagery under graded propofol sedation | C | future-paper dataset | moderate | high | high for a novel paper path | strong for future source-routing / internally generated state work, not for the current threshold benchmark |

## Candidate notes

### 1. NSD-Imagery public release (`ds004937`)

Source:
- OpenNeuro `ds004937`
- canonical repo wrapper already present via `fmri2img.workflows.acquire_public_nsd_imagery`

Why it matters:
- it is the only public source currently aligned to the current benchmark semantics
- it preserves NSD-style identifiers and the current ladder meaning

Why it is not enough:
- the public release is already integrated and still yields only `5` shared
  paired `nsdId`s on the fixed benchmark

Verdict:
- keep as the canonical public baseline source
- do not treat it as the next decisive expansion

### 2. Visual imagery and false memory for pictures (`ds000203`)

Verified task:
- mental imagery / reality-monitoring task with pictures and words in `26`
  healthy participants

Why it matters:
- it is genuinely imagery-related public fMRI
- it supports a secondary paired-imagery benchmark or a future reality-monitoring paper path

Why it does **not** replace the primary ladder:
- it is not the same natural-image paired NSD-style setting
- its stimulus structure and task semantics differ materially from the current
  threshold benchmark

Verdict:
- best immediate public paired-imagery candidate
- classify as a **secondary benchmark**, not as the primary ladder replacement

### 3. Natural Object Dataset (`ds004496`)

Verified task:
- large-scale perception-only natural-image fMRI with `57,120` images across
  `30` participants

Why it matters:
- strongest immediate public option for making the shared-only practical lane
  more robust
- large enough to support perception-side subsystem strengthening and
  representation learning

Why it does **not** answer the threshold question directly:
- it is perception-only

Verdict:
- best immediate public dataset for the practical Animus lane
- first safe remote step now completed as a metadata-only clone under
  `cache/public_datasets/ds004496` on the live pod

### 4. THINGS-fMRI (`ds004192`)

Verified task:
- large-scale object-focused perception-only fMRI, `3` subjects, `12`
  sessions, `8,740` unique images from `720` object concepts

Why it matters:
- excellent semantic/object coverage
- useful for perception-side robustness, object semantics, and future transfer analyses

Why it is not first for Animus:
- fewer subjects than NOD

Verdict:
- best medium-effort object benchmark for Animus robustness and future object-semantic work

### 5. BOLD5000 (`ds001499`)

Verified task:
- large-scale perception-only visual-image fMRI with nearly `5,000` distinct
  images

Why it matters:
- overlaps standard computer-vision image ecosystems
- natural bridge for perception-side model strengthening

Why it ranks below NOD:
- smaller and less subject-rich for immediate subsystem strengthening

Verdict:
- strong supporting perception dataset, but not the first public strengthening target

### 6. Covert consciousness under anesthesia (`ds006623`)

Verified task:
- fMRI during tennis, navigation, and squeeze imagery plus motor response under
  graded propofol sedation in `26` healthy volunteers

Why it matters:
- unusually strong public dataset for internally generated state, imagery under
  altered consciousness, and future source-routing papers

Why it is not part of the current ladder:
- task semantics are fundamentally different from the current natural-image
  perception/imagery threshold benchmark

Verdict:
- high-value **future-paper dataset**

## Best picks by role

### Best immediate public dataset for the threshold research program

- `ds000203` as a **secondary benchmark**, not as a replacement for the main
  NSD-style ladder

### Best immediate public dataset for the practical Animus lane

- `ds004496` (NOD)

### Best medium-effort public dataset for a second benchmark

- `ds004192` (THINGS-fMRI)

### Best future public dataset for a novel follow-on paper

- `ds006623` (covert consciousness under anesthesia)

## Machine-readable catalog

A compact machine-readable version of this map is checked in at:

- `configs/public_datasets/catalog.json`

Quick inspection command:

```bash
source .venv/bin/activate
python -m fmri2img.workflows.show_public_dataset_options
python -m fmri2img.workflows.show_public_dataset_options --role B
```
