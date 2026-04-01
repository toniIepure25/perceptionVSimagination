# Architecture

## Platform layers

- `fmri2img.data`
  canonical mixed-condition dataset and pairing
- `fmri2img.preprocessing`
  preprocessing metadata and artifact description
- `fmri2img.roi`
  ROI-group resolution and validation
- `fmri2img.targets`
  canonical 768-D target-space loader
- `fmri2img.models`
  shared/private decoder contract and modules
- `fmri2img.training`
  multitask loss and trainer
- `fmri2img.evaluation`
  canonical metrics and transfer summaries
- `fmri2img.export`
  Animus-facing manifest export
- `fmri2img.workflows`
  official CLI entrypoints

## Canonical decoder

### 1. ROI-specific hierarchical encoder

Three branch encoders consume ROI-group inputs:

- `early_visual`
- `ventral_visual`
- `metacognitive`

Each branch uses a shallow regularized PyTorch encoder for low-SNR robustness.
For real runs, those branch inputs now come from canonical ROI-materialized
dataset columns such as `roi_features_json` or `roi_values_json`. The raw full
fMRI vector is treated as optional auxiliary context, not as the official
multi-subject model input.

### 2. Shared-private latent split

The concatenated visual branches feed a disentanglement layer that outputs:

- `z_shared`
- `z_perception_private`
- `z_imagery_private`

Auxiliary reconstruction is included for latent-to-visual consistency.

### 3. Multitask heads

- `ContentHead` predicts the canonical 768-D target
- `DomainHead` predicts perception vs imagery
- `VividnessHead` predicts vividness/confidence when labels exist

The content path uses `z_shared` only.
The canonical workflow now disables the vividness/confidence head when the dataset
lacks vividness and confidence supervision.

For controlled canonical ablations, the decoder also supports:

- `disentanglement_mode=shared_private`
- `disentanglement_mode=shared_only`

In `shared_only`, private latents are held inactive and the domain head is
disabled automatically so the comparison stays conceptually clean.

## Data contract

Canonical sample fields:

- `subject`
- `condition`
- `nsd_id`
- `pair_id`
- `split`
- `fmri`
- `clip_target_768`
- ROI-group branch inputs
- optional `vividness`
- optional `confidence`

In canonical multi-subject training:

- ROI-group branch inputs are the authoritative model input
- raw `fmri` may be absent at batch time even when referenced in the index
- if serialized ROI features are already present, the dataset does not force a
  raw voxel load before training
- if raw fMRI vectors have unequal subject-specific dimensionality, the batcher
  preserves ROI features and drops the incompatible stacked raw tensor

## Validation notes

- Pair-aware batching only activates when the dataset contains real cross-condition
  `pair_id` groups with both perception and imagery samples.
- Missing `nsdId` values are treated as fatal because canonical target lookup and
  shared/private pairing both depend on them.
- Checked-in canonical configs use project-root-relative paths and are validated
  before dataset construction.
- The shared/private decoder and current canonical loss terms operate on
  ROI-derived branch inputs and do not require a same-shape raw full-fMRI tensor.

## Legacy note

The historical repo still contains older encoder families and generation code, but the architecture above is now the official design center for the project.
