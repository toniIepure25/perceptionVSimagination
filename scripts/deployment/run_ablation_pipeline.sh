#!/bin/bash
# =============================================================================
# Ablation Grid + Multi-Layer Training + Evaluation Pipeline
# =============================================================================
# Phase A: 4 novel MLP ablations + 4 novel TwoStage ablations
# Phase B: 2 multi-layer TwoStage (baseline + novel)
# Phase C: Test-split eval + Shared-1000 benchmark
#
# Usage (in tmux on cluster):
#   source /home/jovyan/local-data/activate.sh
#   cd /home/jovyan/local-data/perceptionVSimagination
#   bash scripts/deployment/run_ablation_pipeline.sh 2>&1 | tee outputs/ablation.log
# =============================================================================

set -e

PYTHON="/home/jovyan/local-data/venv/bin/python"
SUBJECT="subj01"
CLIP_CACHE="outputs/clip_cache/clip.parquet"
CLIP_MULTI="outputs/clip_cache/clip_multilayer.parquet"
SCRIPT="scripts/train_from_features_v2.py"

NOVEL_FEAT="outputs/features/novel/${SUBJECT}"
BASE_FEAT="outputs/features/baseline/${SUBJECT}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ABLATION GRID + MULTI-LAYER + EVALUATION PIPELINE              ║"
echo "║  Subject: ${SUBJECT}                                            ║"
echo "║  Started: $(date)                                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Verify prerequisites
$PYTHON -c "
import fmri2img; print(f'fmri2img {fmri2img.__version__}')
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy as np
for cfg in ['baseline', 'novel']:
    X = np.load(f'outputs/features/{cfg}/${SUBJECT}/X.npy')
    print(f'{cfg} features: {X.shape}')
print('All prerequisites OK')
"
echo ""

# ============================================================================
# PHASE A: MLP Ablation Grid (4 configs, novel features)
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE A: MLP Ablation Grid (Novel Features)                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# A1: novel_cosine — cosine-only (isolate soft-reliability effect)
echo ""
echo "=== A1: novel_cosine (cosine-only) ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model mlp --config-name novel_cosine \
    --checkpoint-dir checkpoints/mlp_novel_cosine \
    --report-dir outputs/reports/ablation \
    --cosine-weight 1.0 --mse-weight 0.0 --infonce-weight 0.0 \
    --hidden 1024 --dropout 0.1 --lr 1e-3 --wd 1e-4 \
    --epochs 50 --patience 7 --batch-size 256

# A2: novel_cosine_mse — cosine + MSE
echo ""
echo "=== A2: novel_cosine_mse ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model mlp --config-name novel_cosine_mse \
    --checkpoint-dir checkpoints/mlp_novel_cosine_mse \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.5 --mse-weight 0.5 --infonce-weight 0.0 \
    --hidden 1024 --dropout 0.1 --lr 1e-3 --wd 1e-4 \
    --epochs 50 --patience 7 --batch-size 256

# A3: novel_light_infonce — gentle contrastive
echo ""
echo "=== A3: novel_light_infonce ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model mlp --config-name novel_light_infonce \
    --checkpoint-dir checkpoints/mlp_novel_light_infonce \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.8 --mse-weight 0.1 --infonce-weight 0.1 \
    --temperature 0.1 \
    --hidden 1024 --dropout 0.1 --lr 1e-3 --wd 1e-4 \
    --epochs 50 --patience 7 --batch-size 256

# A4: novel_strong_infonce — symmetric InfoNCE, higher temp
echo ""
echo "=== A4: novel_strong_infonce ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model mlp --config-name novel_strong_infonce \
    --checkpoint-dir checkpoints/mlp_novel_strong_infonce \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.5 --mse-weight 0.1 --infonce-weight 0.4 \
    --temperature 0.07 \
    --hidden 1024 --dropout 0.1 --lr 1e-3 --wd 1e-4 \
    --epochs 50 --patience 7 --batch-size 256

echo ""
echo "✓ Phase A (MLP ablation) complete"

# ============================================================================
# PHASE A': TwoStage Ablation Grid (4 configs, novel features)
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE A': TwoStage Ablation Grid (Novel Features)              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# A5: novel_ts_cosine — cosine-only
echo ""
echo "=== A5: novel_ts_cosine (TwoStage cosine-only) ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model two_stage --config-name novel_ts_cosine \
    --checkpoint-dir checkpoints/ts_novel_cosine \
    --report-dir outputs/reports/ablation \
    --cosine-weight 1.0 --mse-weight 0.0 --infonce-weight 0.0 \
    --latent-dim 768 --n-blocks 4 --head-hidden 512 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 12 --batch-size 256

# A6: novel_ts_cosine_mse
echo ""
echo "=== A6: novel_ts_cosine_mse ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model two_stage --config-name novel_ts_cosine_mse \
    --checkpoint-dir checkpoints/ts_novel_cosine_mse \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.5 --mse-weight 0.5 --infonce-weight 0.0 \
    --latent-dim 768 --n-blocks 4 --head-hidden 512 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 12 --batch-size 256

# A7: novel_ts_light_infonce
echo ""
echo "=== A7: novel_ts_light_infonce ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model two_stage --config-name novel_ts_light_infonce \
    --checkpoint-dir checkpoints/ts_novel_light_infonce \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.8 --mse-weight 0.1 --infonce-weight 0.1 \
    --temperature 0.1 \
    --latent-dim 768 --n-blocks 4 --head-hidden 512 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 12 --batch-size 256

# A8: novel_ts_strong_infonce
echo ""
echo "=== A8: novel_ts_strong_infonce ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --model two_stage --config-name novel_ts_strong_infonce \
    --checkpoint-dir checkpoints/ts_novel_strong_infonce \
    --report-dir outputs/reports/ablation \
    --cosine-weight 0.5 --mse-weight 0.1 --infonce-weight 0.4 \
    --temperature 0.07 \
    --latent-dim 768 --n-blocks 4 --head-hidden 512 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 12 --batch-size 256

echo ""
echo "✓ Phase A' (TwoStage ablation) complete"

# ============================================================================
# PHASE B: Multi-Layer TwoStage (baseline + novel)
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE B: Multi-Layer TwoStage Training                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# B1: multilayer_baseline — learnable weights, no InfoNCE
echo ""
echo "=== B1: multilayer_baseline ==="
$PYTHON $SCRIPT \
    --features-dir "$BASE_FEAT" --clip-cache "$CLIP_CACHE" \
    --multilayer-cache "$CLIP_MULTI" \
    --model multilayer --config-name multilayer_baseline \
    --checkpoint-dir checkpoints/multilayer_baseline \
    --report-dir outputs/reports/multilayer \
    --learnable-weights \
    --cosine-weight 1.0 --mse-weight 0.0 --infonce-weight 0.0 \
    --latent-dim 768 --n-blocks 4 --head-hidden 1024 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 15 --batch-size 128

# B2: multilayer_novel — learnable weights + multi-layer InfoNCE
echo ""
echo "=== B2: multilayer_novel ==="
$PYTHON $SCRIPT \
    --features-dir "$NOVEL_FEAT" --clip-cache "$CLIP_CACHE" \
    --multilayer-cache "$CLIP_MULTI" \
    --model multilayer --config-name multilayer_novel \
    --checkpoint-dir checkpoints/multilayer_novel \
    --report-dir outputs/reports/multilayer \
    --learnable-weights --multilayer-infonce \
    --infonce-combination weighted_pool \
    --cosine-weight 0.8 --mse-weight 0.0 --infonce-weight 0.2 \
    --temperature 0.07 \
    --latent-dim 768 --n-blocks 4 --head-hidden 1024 \
    --lr 5e-4 --wd 1e-4 --epochs 80 --patience 15 --batch-size 128

echo ""
echo "✓ Phase B (Multi-layer) complete"

# ============================================================================
# PHASE C: Evaluation
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE C: Comprehensive Evaluation                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# C1: Test-split evaluation (fast — uses pre-extracted features)
echo ""
echo "=== C1: Test-split evaluation ==="
$PYTHON scripts/eval_all_models.py \
    --subject "$SUBJECT" \
    --features-dirs "$BASE_FEAT" "$NOVEL_FEAT" \
    --clip-cache "$CLIP_CACHE" \
    --checkpoints-root checkpoints \
    --output-dir outputs/eval/test_split

# C2: Shared-1000 benchmark (slower — loads NIfTI)
echo ""
echo "=== C2: Shared-1000 benchmark ==="

BETA_ROOT="/home/jovyan/work/data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"

# Build checkpoint args dynamically
CKPT_ARGS=""
for d in checkpoints/*/; do
    dname=$(basename "$d")
    subj_dir="$d${SUBJECT}"
    if [ -d "$subj_dir" ]; then
        if [ -f "$subj_dir/ridge.pkl" ]; then
            CKPT_ARGS="$CKPT_ARGS ${dname}=ridge:${subj_dir}/ridge.pkl"
        fi
        if [ -f "$subj_dir/mlp.pt" ]; then
            CKPT_ARGS="$CKPT_ARGS ${dname}=mlp:${subj_dir}/mlp.pt"
        fi
        if [ -f "$subj_dir/two_stage_best.pt" ]; then
            CKPT_ARGS="$CKPT_ARGS ${dname}=two_stage:${subj_dir}/two_stage_best.pt"
        fi
        if [ -f "$subj_dir/multilayer_best.pt" ]; then
            CKPT_ARGS="$CKPT_ARGS ${dname}=multilayer:${subj_dir}/multilayer_best.pt"
        fi
    fi
done

$PYTHON scripts/eval_shared1000_benchmark.py \
    --subject "$SUBJECT" \
    --stim-info cache/nsd_stim_info_merged.csv \
    --preproc-dirs outputs/preproc/baseline/${SUBJECT} outputs/preproc/novel/${SUBJECT} \
    --beta-root "$BETA_ROOT" \
    --clip-cache "$CLIP_CACHE" \
    --checkpoints $CKPT_ARGS \
    --output-dir outputs/eval/shared1000

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ALL PHASES COMPLETE                                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  Test-split:  outputs/eval/test_split/${SUBJECT}/"
echo "  Shared-1000: outputs/eval/shared1000/${SUBJECT}/"
echo "  Reports:     outputs/reports/ablation/${SUBJECT}/"
echo "               outputs/reports/multilayer/${SUBJECT}/"
echo ""
echo "Completed: $(date)"
