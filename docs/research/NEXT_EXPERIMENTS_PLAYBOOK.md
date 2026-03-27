# Next Experiments Playbook (Novel / Innovative)

> Date: 2026-03-16
> Goal: continue from current findings with high-value, publication-grade experiments.

---

## A. Attention-as-Bridge (H6)

### Why
Determine whether attention is an intermediate representational regime between perception and imagery.

### Commands (cluster)

```bash
cd /home/jovyan/local-data/perceptionVSimagination
export PATH=/home/jovyan/local-data/venv/bin:$PATH
export VIRTUAL_ENV=/home/jovyan/local-data/venv
export PYTHONPATH=/home/jovyan/local-data/perceptionVSimagination/src:$PYTHONPATH

# Ridge
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint checkpoints/ridge_baseline/subj01/ridge.pkl \
  --mode attention \
  --model-type ridge \
  --output-dir outputs/attention_eval/subj01/ridge \
  --data-root /home/jovyan/work/data/nsd/nsdimagery \
  --split test \
  --device cuda

# FMRI2images V30e
python scripts/eval_perception_to_imagery_transfer.py \
  --index cache/indices/imagery/subj01.parquet \
  --checkpoint /home/jovyan/work/data/FMRI2images/experimental_results/V30e_rerank_head_2048/subj01/checkpoint_best.pt \
  --mode attention \
  --model-type external_fmri2images \
  --output-dir outputs/attention_eval/subj01/v30e \
  --data-root /home/jovyan/work/data/nsd/nsdimagery \
  --nsd-root /home/jovyan/work/data/nsd \
  --split test \
  --device cuda
```

### Success Criteria
- Condition ordering test: perception ≥ attention ≥ imagery OR perception ≈ attention > imagery
- Report cosine + R@K and confidence intervals.

---

## B. Retrieval Collapse Mechanism (H4)

### Why
We repeatedly observe small cosine gap but larger retrieval drop on imagery.

### Minimal analysis script logic
1. Load `per_trial.csv` + predictions for perception/imagery.
2. Compute off-diagonal cosine mean within each condition.
3. Compute local neighborhood density (kNN radius in embedding space).
4. Test whether imagery has higher density (harder retrieval despite similar mean alignment).

### Outputs
- `outputs/retrieval_geometry/subj01/summary.json`
- `outputs/retrieval_geometry/subj01/density_plot.png`

---

## C. Shared-Stimulus Pair Analysis (Set B)

### Why
Use matched stimuli (same `nsd_id`) to isolate within-item perception→imagery shift.

### Procedure
- Restrict to set B IDs: `28752, 30857, 53882, 61178, 65873`
- Pair perception and imagery trials by `nsd_id`
- Compute per-id Δcosine and paired bootstrap CI

### Outputs
- `outputs/shared_stimuli/subj01/paired_transfer.json`
- `outputs/shared_stimuli/subj01/paired_transfer_plot.png`

---

## D. Cross-Capacity Consistency

### Why
Separate neural effect from model artifact.

### Procedure
- Build `EmbeddingBundle` for Ridge and V30e on same subj01 data.
- Run same analysis subset: dimensionality, topology, reality_monitor, reality_confusion, compositional.
- Use `src/fmri2img/analysis/cross_capacity.py` to compare effect signs and rank consistency.

### Primary metric
- Sign agreement rate across shared metrics.

---

## E. Publishing-Ready Novelty Claims (if confirmed)

1. **Three-condition trajectory** (perception-attention-imagery) in a shared CLIP space.
2. **Alignment vs discrimination dissociation** quantified geometrically.
3. **Cross-capacity replication** of imagery transfer trend.
4. **Within-item transfer** on known shared NSD stimuli.

---

## F. Priority Order (1-week sprint)

1. Run Attention-as-Bridge (A)
2. Run Shared-Stimulus pairs (C)
3. Run Retrieval geometry (B)
4. Run Cross-capacity consistency (D)
5. Integrate into paper draft + figures
