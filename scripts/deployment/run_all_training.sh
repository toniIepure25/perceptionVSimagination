#!/bin/bash
# =============================================================================
# Master Training Pipeline for H100 Cluster
# =============================================================================
# Runs full suite: Ridge (baseline+novel), MLP, TwoStage (baseline+novel)
#
# Memory-safe: processes data in chunks to avoid OOM (pod limit ~64GB)
# Persistent: uses venv on local-data, survives pod restarts
#
# Usage (in tmux on cluster):
#   source /home/jovyan/local-data/activate.sh
#   cd /home/jovyan/local-data/perceptionVSimagination
#   bash scripts/deployment/run_all_training.sh 2>&1 | tee outputs/training.log
# =============================================================================

set -e

VENV_PYTHON="/home/jovyan/local-data/venv/bin/python"
SUBJECT="subj01"
CLIP_CACHE="outputs/clip_cache/clip.parquet"
CLIP_MULTI="outputs/clip_cache/clip_multilayer.parquet"
INDEX_ROOT="data/indices/nsd_index"

# Timestamp for logs
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="outputs/logs/${TS}"
mkdir -p "$LOGDIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PERCEPTION vs IMAGINATION — Full Training Pipeline             ║"
echo "║  Subject: ${SUBJECT}                                            ║"
echo "║  Timestamp: ${TS}                                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Verify prerequisites
echo "=== Checking prerequisites ==="
$VENV_PYTHON -c "
import fmri2img; print(f'fmri2img {fmri2img.__version__}')
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import pandas as pd
idx = pd.read_parquet('data/indices/nsd_index/subject=subj01/index.parquet')
print(f'Index: {len(idx)} trials')
clip = pd.read_parquet('outputs/clip_cache/clip.parquet')
print(f'CLIP cache: {len(clip)} embeddings')
import numpy as np
for cfg in ['baseline', 'novel']:
    d = f'outputs/preproc/{cfg}/subj01'
    pca = np.load(f'{d}/pca_components.npy')
    print(f'Preproc {cfg}: PCA {pca.shape[0]} components')
print('All prerequisites OK')
"
echo ""

# ============================================================================
# STEP 1: Ridge Regression (Baseline)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 1/6: Ridge Regression — Baseline Preprocessing            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/ridge/${SUBJECT}/ridge_baseline.pkl" ]; then
    echo "  ✓ Ridge baseline already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_ridge.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/baseline" \
        --clip-cache "$CLIP_CACHE" \
        --alpha-grid "0.1,1,3,10,30,100,300,1000" \
        --checkpoint-dir "checkpoints/ridge_baseline" \
        --report-dir "outputs/reports/baseline" \
        2>&1 | tee "${LOGDIR}/ridge_baseline.log"
    
    echo ""
    echo "  ✓ Ridge baseline complete"
fi
echo ""

# ============================================================================
# STEP 2: Ridge Regression (Novel — soft reliability)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 2/6: Ridge Regression — Novel Preprocessing               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/ridge/${SUBJECT}/ridge_novel.pkl" ]; then
    echo "  ✓ Ridge novel already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_ridge.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/novel" \
        --clip-cache "$CLIP_CACHE" \
        --alpha-grid "0.1,1,3,10,30,100,300,1000" \
        --checkpoint-dir "checkpoints/ridge_novel" \
        --report-dir "outputs/reports/novel" \
        2>&1 | tee "${LOGDIR}/ridge_novel.log"
    
    echo ""
    echo "  ✓ Ridge novel complete"
fi
echo ""

# ============================================================================
# STEP 3: MLP Encoder (Baseline)  
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 3/6: MLP Encoder — Baseline Preprocessing                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/mlp_baseline/${SUBJECT}/mlp.pt" ]; then
    echo "  ✓ MLP baseline already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_mlp.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/baseline" \
        --clip-cache "$CLIP_CACHE" \
        --hidden 1024 --dropout 0.1 \
        --lr 1e-3 --wd 1e-4 \
        --epochs 50 --patience 7 \
        --batch-size 256 \
        --cosine-weight 1.0 --mse-weight 0.0 --infonce-weight 0.0 \
        --checkpoint-dir "checkpoints/mlp_baseline" \
        --report-dir "outputs/reports/baseline" \
        2>&1 | tee "${LOGDIR}/mlp_baseline.log"
    
    echo ""
    echo "  ✓ MLP baseline complete"
fi
echo ""

# ============================================================================
# STEP 4: MLP Encoder (Novel — with InfoNCE)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 4/6: MLP Encoder — Novel Preprocessing + InfoNCE          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/mlp_novel/${SUBJECT}/mlp.pt" ]; then
    echo "  ✓ MLP novel already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_mlp.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/novel" \
        --clip-cache "$CLIP_CACHE" \
        --hidden 1024 --dropout 0.1 \
        --lr 1e-3 --wd 1e-4 \
        --epochs 50 --patience 7 \
        --batch-size 256 \
        --cosine-weight 0.6 --mse-weight 0.0 --infonce-weight 0.4 \
        --checkpoint-dir "checkpoints/mlp_novel" \
        --report-dir "outputs/reports/novel" \
        2>&1 | tee "${LOGDIR}/mlp_novel.log"
    
    echo ""
    echo "  ✓ MLP novel complete"
fi
echo ""

# ============================================================================
# STEP 5: Two-Stage Encoder (Baseline)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 5/6: Two-Stage Encoder — Baseline                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/two_stage_baseline/${SUBJECT}/two_stage_best.pt" ]; then
    echo "  ✓ TwoStage baseline already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_two_stage.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/baseline" \
        --clip-cache "$CLIP_CACHE" \
        --latent-dim 768 --n-blocks 4 \
        --head-type mlp --head-hidden 512 \
        --mse-weight 0.3 --cosine-weight 0.3 --infonce-weight 0.4 \
        --temperature 0.05 \
        --lr 5e-5 --wd 1e-4 \
        --batch-size 48 --epochs 150 --patience 20 \
        --device cuda \
        --checkpoint-dir "checkpoints/two_stage_baseline" \
        --report-dir "outputs/reports/baseline" \
        2>&1 | tee "${LOGDIR}/two_stage_baseline.log"
    
    echo ""
    echo "  ✓ TwoStage baseline complete"
fi
echo ""

# ============================================================================
# STEP 6: Two-Stage Encoder (Novel — multi-layer + soft reliability)
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STEP 6/6: Two-Stage Encoder — Novel (Multi-Layer + Soft)        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "checkpoints/two_stage_novel/${SUBJECT}/two_stage_best.pt" ]; then
    echo "  ✓ TwoStage novel already trained, skipping"
else
    $VENV_PYTHON src/fmri2img/training/train_two_stage.py \
        --subject "$SUBJECT" \
        --index-root "$INDEX_ROOT" \
        --use-preproc \
        --preproc-dir "outputs/preproc/novel" \
        --clip-cache "$CLIP_CACHE" \
        --multilayer-cache "$CLIP_MULTI" \
        --latent-dim 768 --n-blocks 4 \
        --head-type mlp --head-hidden 512 \
        --mse-weight 0.3 --cosine-weight 0.3 --infonce-weight 0.4 \
        --temperature 0.05 \
        --lr 5e-5 --wd 1e-4 \
        --batch-size 48 --epochs 150 --patience 20 \
        --device cuda \
        --checkpoint-dir "checkpoints/two_stage_novel" \
        --report-dir "outputs/reports/novel" \
        2>&1 | tee "${LOGDIR}/two_stage_novel.log"
    
    echo ""
    echo "  ✓ TwoStage novel complete"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ALL TRAINING COMPLETE                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Checkpoints:"
echo "  Ridge baseline:    checkpoints/ridge_baseline/${SUBJECT}/"
echo "  Ridge novel:       checkpoints/ridge_novel/${SUBJECT}/"
echo "  MLP baseline:      checkpoints/mlp_baseline/${SUBJECT}/"
echo "  MLP novel:         checkpoints/mlp_novel/${SUBJECT}/"
echo "  TwoStage baseline: checkpoints/two_stage_baseline/${SUBJECT}/"
echo "  TwoStage novel:    checkpoints/two_stage_novel/${SUBJECT}/"
echo ""
echo "Reports in: outputs/reports/{baseline,novel}/${SUBJECT}/"
echo "Logs in:    ${LOGDIR}/"
echo ""
echo "Next: Run evaluation with eval_shared1000_full.py"
