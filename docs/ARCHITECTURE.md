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
Today, those branch inputs must come from precomputed ROI features in the dataset
contract or from an explicit smoke-test fallback. The legacy ROI mask loader is
not yet wired into the canonical workflow entrypoints.

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

## Validation notes

- Pair-aware batching only activates when the dataset contains real cross-condition
  `pair_id` groups with both perception and imagery samples.
- Missing `nsdId` values are treated as fatal because canonical target lookup and
  shared/private pairing both depend on them.
- Checked-in canonical configs use project-root-relative paths and are validated
  before dataset construction.

## Legacy note

The historical repo still contains older encoder families and generation code, but the architecture above is now the official design center for the project.
