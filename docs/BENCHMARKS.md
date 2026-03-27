# Benchmarks

## Canonical benchmark questions

1. How well does `z_shared` decode 768-D visual content?
2. How much does performance drop from perception to imagery?
3. How much paired transfer survives for shared `nsdId` stimuli?
4. Which ROI streams contribute strongest branch signal?
5. When labels exist, how well do vividness/confidence heads predict subjective ratings?

## Canonical metrics

- Content cosine similarity
- Content MSE
- Content InfoNCE loss
- Mean perception cosine
- Mean imagery cosine
- Mean imagery-minus-perception pair gap
- ROI branch norm summary
- Optional vividness MSE
- Optional confidence summary

## Official benchmark commands

```bash
python -m fmri2img.workflows.eval_decoder \
  --config configs/canonical/shared_private_mvp.yaml \
  --checkpoint outputs/canonical/train/shared_private_mvp/best_decoder.pt
```

```bash
python -m fmri2img.workflows.eval_transfer \
  --config configs/canonical/shared_private_mvp.yaml \
  --checkpoint outputs/canonical/train/shared_private_mvp/best_decoder.pt
```

```bash
python -m fmri2img.workflows.run_analysis \
  --config configs/canonical/shared_private_mvp.yaml \
  --checkpoint outputs/canonical/train/shared_private_mvp/best_decoder.pt
```

## What is not canonical

- legacy shared1000 benchmark scripts
- old feature-extraction-only comparisons that do not use the shared/private model contract
- generation-first metrics as the main headline numbers
