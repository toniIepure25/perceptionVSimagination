# Experiment Configurations

Research experiment configs for the perception-vs-imagination study.

---

## Available Configs

| Config | Purpose |
|--------|---------|
| `ablation.yaml` | Systematic component ablation (loss, architecture, preprocessing) |
| `novel_analyses.yaml` | Six novel neuroscience analysis directions |
| `perception_to_imagery_eval.yaml` | Cross-domain transfer evaluation (H1--H3) |
| `reproducibility.yaml` | Full experiment protocol (seeds, splits, metrics) |

---

## Novel Analyses

`novel_analyses.yaml` configures all six research directions:

1. **Dimensionality gap** -- PCA participation ratio, intrinsic dimensionality
2. **Uncertainty as vividness** -- MC Dropout variance as imagery quality proxy
3. **Semantic survival** -- per-concept preservation across perception→imagery
4. **Topological RSA** -- persistent homology of representational geometry
5. **Cross-subject fingerprints** -- individual imagery style signatures
6. **Semantic-structural dissociation** -- CLIP vs. SD-latent transfer ratio

```bash
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml
python scripts/run_novel_analyses.py --config configs/experiments/novel_analyses.yaml --dry-run
```

---

## Ablation Studies

`ablation.yaml` inherits from `../training/two_stage_sota.yaml` and defines five ablation dimensions:

1. **Loss functions** -- MSE-only, cosine-only, InfoNCE-only, combinations
2. **Architecture** -- residual blocks, latent dimensionality
3. **Multi-layer supervision** -- enabled/disabled, layer weight distribution
4. **Preprocessing** -- PCA dimensionality, reliability threshold
5. **Regularization** -- dropout rates, batch normalization

```bash
python -m fmri2img.eval.ablation_driver --base-config configs/experiments/ablation.yaml
```

---

## Cross-Domain Evaluation

`perception_to_imagery_eval.yaml` evaluates perception-trained models on NSD-Imagery data:

```bash
python scripts/eval_perception_to_imagery_transfer.py \
    --config configs/experiments/perception_to_imagery_eval.yaml
```

---

## Reproducibility Protocol

`reproducibility.yaml` is a reference document (not a runtime config) that specifies:

- Global random seed (42)
- Hardware requirements and expected runtimes
- Data versioning (NSD v1.0, NSD-Imagery beta)
- Checkpoint naming conventions
- Evaluation protocol (metrics, galleries, statistical tests)
- Parameters for each novel analysis direction

See also: [docs/research/PAPER_DRAFT_OUTLINE.md](../../docs/research/PAPER_DRAFT_OUTLINE.md)
