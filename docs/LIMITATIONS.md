# Limitations

## Real-data limitations

- The repository does not currently ship a real mixed perception/imagery index.
- The canonical MVP config does not currently have a real 768-D target cache checked in.
- The local real perception index points to remote `beta_path` files rather than materialized local voxel arrays.

## ROI limitations

- The canonical workflow now builds ROI branch inputs directly from ROI masks, but it still depends on real volumetric inputs or prepared ROI features. Flat 1D voxel arrays cannot be retrofitted with volumetric ROI masks.
- ROI branch analysis currently summarizes branch embedding norms, not full causal ROI ablations.
- The old `ROIPooler` remains in the codebase as a legacy compatibility path, but the official ROI materialization path is now `fmri2img.roi.materialize`.

## Subjective-state limitations

- Vividness/confidence prediction is only trustworthy when real labels exist.
- The canonical workflow now disables the vividness/confidence head when neither vividness nor confidence labels are available.
- The current evidential-style head is a lightweight uncertainty mechanism, not a full Bayesian calibration system.

## Scientific-scope limitations

- True stimulus-vs-percept or reality-monitoring claims still require different datasets or future experiments.
- SDXL/IP-Adapter reconstruction is still a future downstream integration step, not the canonical backbone.
- Cross-subject training and stronger uncertainty calibration remain future work.
