#!/bin/bash
# =============================================================================
# Perception vs. Imagination — Full Experiment Setup for H100 80GB
# =============================================================================
#
# This script sets up the complete experiment environment on the remote H100
# machine (orchestraiq cluster), reusing existing NSD data and compatible
# caches from the FMRI2images project.
#
# Remote Environment (auto-detected):
#   - GPU: NVIDIA H100 80GB HBM3, CUDA 12.8
#   - Python: 3.13.12 via /opt/conda
#   - PyTorch: 2.10.0+cu128
#   - NSD data: /home/jovyan/work/data/nsd/ (subj01,02,05,07)
#   - Old project: /home/jovyan/work/FMRI2images/ (caches reusable)
#
# Usage:
#   bash scripts/setup_h100_experiment.sh [--subject SUBJ] [--skip-training] [--dry-run]
#
# Resumable: tracks progress in .setup_progress file; re-run to resume.
# =============================================================================

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
SUBJECT="${1:-subj01}"                                    # Default subject
PROJECT_DIR="/home/jovyan/work/perceptionVSimagination"   # New project location
OLD_PROJECT="/home/jovyan/work/FMRI2images"               # Existing project
NSD_ROOT="/home/jovyan/work/data/nsd"                     # NSD data root
LOCAL_DATA="/home/jovyan/local-data"                      # Fast local storage
CONDA_PYTHON="/opt/conda/bin/python"                      # Conda python
CONDA_PIP="/opt/conda/bin/pip"                            # Conda pip
PROGRESS_FILE="${PROJECT_DIR}/.setup_progress"
LOG_DIR="${PROJECT_DIR}/outputs/logs/setup"
REPO_URL="https://github.com/toniIepure25/perceptionVSimagination.git"

# H100-specific training settings
BATCH_SIZE_CLIP=512       # CLIP cache building (H100 can handle large batches)
BATCH_SIZE_TRAIN=128      # Model training
BATCH_SIZE_ADAPTER=64     # Adapter training
NUM_WORKERS=8             # Data loading workers
TRAINING_EPOCHS=200       # Two-stage training epochs
ADAPTER_EPOCHS=50         # Adapter fine-tuning epochs

# Parse optional flags
SKIP_TRAINING=false
DRY_RUN=false
SKIP_INSTALL=false
for arg in "$@"; do
    case $arg in
        --skip-training) SKIP_TRAINING=true ;;
        --dry-run)       DRY_RUN=true ;;
        --skip-install)  SKIP_INSTALL=true ;;
        subj*)           SUBJECT="$arg" ;;
    esac
done

# ==============================================================================
# Utilities
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_step()    { echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BOLD}${BLUE}  STEP: $1${NC}"; echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
log_ok()      { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
log_err()     { echo -e "  ${RED}✗${NC} $1"; }
log_info()    { echo -e "  ${CYAN}→${NC} $1"; }
log_skip()    { echo -e "  ${YELLOW}⏭${NC} $1 (already done)"; }

elapsed_time() {
    local start=$1
    local end=$(date +%s)
    local diff=$((end - start))
    echo "$((diff / 60))m $((diff % 60))s"
}

# Progress tracking — lets us resume from where we left off
mark_done() {
    mkdir -p "$(dirname "$PROGRESS_FILE")"
    echo "$1" >> "$PROGRESS_FILE"
}

is_done() {
    [ -f "$PROGRESS_FILE" ] && grep -qxF "$1" "$PROGRESS_FILE"
}

# Dry run wrapper
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $*"
    else
        eval "$@"
    fi
}

# ==============================================================================
# Banner
# ==============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Perception vs. Imagination — H100 Experiment Setup           ║${NC}"
echo -e "${BOLD}║   Subject: ${CYAN}${SUBJECT}${NC}${BOLD}    GPU: H100 80GB    CUDA: 12.8              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Project dir : ${PROJECT_DIR}"
echo -e "  NSD data    : ${NSD_ROOT}"
echo -e "  Old project : ${OLD_PROJECT}"
echo -e "  Dry run     : ${DRY_RUN}"
echo -e "  Skip train  : ${SKIP_TRAINING}"
echo ""

SETUP_START=$(date +%s)

# ==============================================================================
# STEP 0: Validate Prerequisites
# ==============================================================================
log_step "0/12  Validate Prerequisites"

# Check GPU
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log_ok "GPU: ${GPU_NAME} (${GPU_MEM})"
else
    log_err "No GPU detected! nvidia-smi failed."
    exit 1
fi

# Check CUDA via PyTorch
if $CONDA_PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$($CONDA_PYTHON -c "import torch; print(torch.__version__)")
    CUDA_VER=$($CONDA_PYTHON -c "import torch; print(torch.version.cuda)")
    log_ok "PyTorch ${TORCH_VER} with CUDA ${CUDA_VER}"
else
    log_err "PyTorch CUDA not available"
    exit 1
fi

# Check NSD data
if [ -f "${NSD_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5" ]; then
    HDF5_SIZE=$(du -h "${NSD_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5" | cut -f1)
    log_ok "NSD stimuli HDF5: ${HDF5_SIZE}"
else
    log_err "NSD stimuli HDF5 not found at ${NSD_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    exit 1
fi

# Check NSD betas
BETA_DIR="${NSD_ROOT}/nsddata_betas/ppdata/${SUBJECT}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
if [ -d "${BETA_DIR}" ]; then
    BETA_COUNT=$(find "${BETA_DIR}" -name "betas_session*.nii.gz" 2>/dev/null | wc -l)
    log_ok "NSD betas for ${SUBJECT}: ${BETA_COUNT} session files"
else
    log_err "NSD betas not found for ${SUBJECT} at ${BETA_DIR}"
    exit 1
fi

# Check NSD experiment info
if [ -f "${NSD_ROOT}/nsddata/experiments/nsd/nsd_stim_info_merged.csv" ]; then
    log_ok "NSD stim_info_merged.csv found"
else
    log_err "NSD stim_info_merged.csv not found"
    exit 1
fi

# Check old project for reusable caches
if [ -d "${OLD_PROJECT}" ]; then
    log_ok "Old FMRI2images project found — will reuse compatible caches"
else
    log_warn "Old project not found — will build everything from scratch"
fi

# ==============================================================================
# STEP 1: Clone / Update Repository
# ==============================================================================
log_step "1/12  Clone Repository"

if is_done "clone_repo"; then
    log_skip "Repository already cloned"
else
    if [ -d "${PROJECT_DIR}/.git" ]; then
        log_info "Project dir exists, pulling latest..."
        run_cmd "cd ${PROJECT_DIR} && git pull origin main 2>/dev/null || true"
    else
        log_info "Cloning repository..."
        run_cmd "git clone ${REPO_URL} ${PROJECT_DIR} 2>/dev/null || true"
        if [ ! -d "${PROJECT_DIR}" ]; then
            log_warn "Git clone failed (private repo?). Creating directory structure..."
            run_cmd "mkdir -p ${PROJECT_DIR}"
        fi
    fi
    mark_done "clone_repo"
    log_ok "Repository ready at ${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

# ==============================================================================
# STEP 2: Install Dependencies
# ==============================================================================
log_step "2/12  Install Dependencies"

if is_done "install_deps" || [ "$SKIP_INSTALL" = true ]; then
    log_skip "Dependencies already installed"
else
    STEP_START=$(date +%s)

    log_info "Installing missing Python packages into /opt/conda..."

    # Core packages not yet installed
    PACKAGES=(
        "open_clip_torch>=2.20.0"
        "diffusers>=0.18.0"
        "transformers>=4.30.0"
        "accelerate>=0.20.0"
        "safetensors>=0.3.0"
        "nibabel"
        "s3fs>=2023.5.0"
        "fsspec"
        "einops>=0.6.0"
        "faiss-cpu"
        "joblib"
        "requests"
        "ruff"
        "pytest"
    )

    run_cmd "$CONDA_PIP install --quiet ${PACKAGES[*]} 2>&1 | tail -5"

    # Install the project package in editable mode
    if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
        log_info "Installing fmri2img package (editable)..."
        run_cmd "$CONDA_PIP install -e '${PROJECT_DIR}' --quiet 2>&1 | tail -3"
    fi

    # Verify key imports
    $CONDA_PYTHON -c "
import torch, numpy, pandas, scipy, sklearn, h5py, nibabel, pyarrow
import open_clip, diffusers, transformers, accelerate, einops
print('All imports successful')
" && log_ok "All key packages verified" || log_err "Some imports failed"

    mark_done "install_deps"
    log_ok "Dependencies installed ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 3: Set Up H100 Optimizations
# ==============================================================================
log_step "3/12  Configure H100 Optimizations"

# Create H100-optimized environment settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
export TORCH_BACKENDS_CUDNN_ALLOW_TF32=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Write env file for reuse
cat > "${PROJECT_DIR}/.env_h100" << 'ENVEOF'
# H100 80GB Optimizations — source this before running experiments
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
export TORCH_BACKENDS_CUDNN_ALLOW_TF32=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Enable TF32 for matmuls (2x speedup on H100, negligible precision loss)
export NVIDIA_TF32_OVERRIDE=1
ENVEOF

log_ok "H100 env config written to .env_h100"

# Write H100-specific YAML config overlay
mkdir -p "${PROJECT_DIR}/configs/system"
cat > "${PROJECT_DIR}/configs/system/h100_80gb.yaml" << 'YAMLEOF'
# H100 80GB System-Specific Configuration Overlay
# Source: auto-generated by setup_h100_experiment.sh
#
# Apply on top of base.yaml:
#   python script.py --config configs/base.yaml --override-config configs/system/h100_80gb.yaml

system:
  gpu: H100-80GB-HBM3
  cuda_version: "12.8"
  driver_version: "570.211.01"
  pytorch_version: "2.10.0+cu128"

dataset:
  num_workers: 8
  pin_memory: true
  prefetch_factor: 4       # H100 memory bandwidth allows aggressive prefetch

preprocessing:
  pca_k: 512

clip:
  batch_size: 512           # H100 80GB can handle large CLIP batches
  device: cuda

training:
  device: cuda
  mixed_precision: true     # AMP with bf16 on H100
  gradient_clip: 1.0
  batch_size: 128           # Larger batches utilise H100 better

  # H100 specific
  compile_model: false      # torch.compile — enable if model is compatible
  tf32_matmul: true         # TensorFloat-32 for 2x matmul speedup
  bf16_training: true       # BFloat16 native on H100 (better than fp16)

  # Cosine annealing
  lr_scheduler: cosine
  warmup_epochs: 5
  min_lr: 1.0e-6

adapter:
  batch_size: 64
  epochs: 50
  learning_rate: 1.0e-3
YAMLEOF

log_ok "H100 config overlay at configs/system/h100_80gb.yaml"

# ==============================================================================
# STEP 4: Link NSD Data & Reuse Caches
# ==============================================================================
log_step "4/12  Link NSD Data & Reuse Caches"

if is_done "link_data"; then
    log_skip "Data already linked"
else
    STEP_START=$(date +%s)

    # Create directory structure
    mkdir -p "${PROJECT_DIR}/cache/nsd_hdf5"
    mkdir -p "${PROJECT_DIR}/cache/s3_cache"
    mkdir -p "${PROJECT_DIR}/cache/indices/imagery"
    mkdir -p "${PROJECT_DIR}/cache/preproc"
    mkdir -p "${PROJECT_DIR}/cache/preextracted"
    mkdir -p "${PROJECT_DIR}/data/indices/nsd_index"
    mkdir -p "${PROJECT_DIR}/data/nsd_imagery"
    mkdir -p "${PROJECT_DIR}/outputs/clip_cache"
    mkdir -p "${PROJECT_DIR}/outputs/reports/imagery"
    mkdir -p "${PROJECT_DIR}/outputs/imagery_ablations"
    mkdir -p "${PROJECT_DIR}/outputs/novel_analyses"
    mkdir -p "${PROJECT_DIR}/checkpoints/ridge/${SUBJECT}"
    mkdir -p "${PROJECT_DIR}/checkpoints/mlp/${SUBJECT}"
    mkdir -p "${PROJECT_DIR}/checkpoints/two_stage/${SUBJECT}"
    mkdir -p "${PROJECT_DIR}/checkpoints/clip_adapter/${SUBJECT}"

    # --- Symlink NSD stimuli HDF5 ---
    STIM_HDF5="${NSD_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    TARGET="${PROJECT_DIR}/cache/nsd_hdf5/nsd_stimuli.hdf5"
    if [ ! -L "$TARGET" ] && [ ! -f "$TARGET" ]; then
        run_cmd "ln -sf ${STIM_HDF5} ${TARGET}"
        log_ok "Linked NSD stimuli HDF5 (37GB)"
    else
        log_skip "NSD stimuli HDF5 already linked"
    fi

    # --- Symlink NSD stim info ---
    STIM_INFO="${NSD_ROOT}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    TARGET="${PROJECT_DIR}/cache/nsd_stim_info_merged.csv"
    if [ ! -L "$TARGET" ] && [ ! -f "$TARGET" ]; then
        run_cmd "ln -sf ${STIM_INFO} ${TARGET}"
        log_ok "Linked nsd_stim_info_merged.csv"
    else
        log_skip "nsd_stim_info_merged.csv already linked"
    fi

    # --- Reuse CLIP cache from old project ---
    OLD_CLIP="${OLD_PROJECT}/outputs/clip_cache/clip.parquet"
    NEW_CLIP="${PROJECT_DIR}/outputs/clip_cache/clip.parquet"
    if [ -f "${OLD_CLIP}" ] && [ ! -f "${NEW_CLIP}" ]; then
        run_cmd "cp ${OLD_CLIP} ${NEW_CLIP}"
        SIZE=$(du -h "${NEW_CLIP}" | cut -f1)
        log_ok "Copied CLIP cache from old project (${SIZE})"
    elif [ -f "${NEW_CLIP}" ]; then
        log_skip "CLIP cache already exists"
    else
        log_warn "No existing CLIP cache found — will build from scratch"
    fi

    # --- Reuse NSD indices from old project ---
    for subj in subj01 subj02 subj05 subj07; do
        OLD_IDX="${OLD_PROJECT}/data/indices/nsd_index/subject=${subj}/index.parquet"
        NEW_IDX="${PROJECT_DIR}/data/indices/nsd_index/subject=${subj}"
        if [ -f "${OLD_IDX}" ] && [ ! -f "${NEW_IDX}/index.parquet" ]; then
            mkdir -p "${NEW_IDX}"
            run_cmd "cp ${OLD_IDX} ${NEW_IDX}/index.parquet"
            log_ok "Copied NSD index for ${subj}"
        elif [ -f "${NEW_IDX}/index.parquet" ]; then
            log_skip "NSD index for ${subj} already exists"
        fi
    done

    # --- Reuse pre-extracted fMRI features ---
    for subj in subj01 subj02 subj05 subj07; do
        OLD_FEAT="${OLD_PROJECT}/cache/preextracted/subject=${subj}"
        NEW_FEAT="${PROJECT_DIR}/cache/preextracted/subject=${subj}"
        if [ -d "${OLD_FEAT}" ] && [ ! -d "${NEW_FEAT}" ]; then
            run_cmd "ln -sf ${OLD_FEAT} ${NEW_FEAT}"
            log_ok "Linked pre-extracted features for ${subj}"
        elif [ -d "${NEW_FEAT}" ]; then
            log_skip "Pre-extracted features for ${subj} already exist"
        fi
    done

    # --- Reuse preprocessing artifacts ---
    for subj in subj01 subj02 subj05 subj07; do
        OLD_PREPROC="${OLD_PROJECT}/cache/preproc/subject=${subj}"
        NEW_PREPROC="${PROJECT_DIR}/cache/preproc/subject=${subj}"
        if [ -d "${OLD_PREPROC}" ] && [ ! -d "${NEW_PREPROC}" ]; then
            run_cmd "ln -sf ${OLD_PREPROC} ${NEW_PREPROC}"
            log_ok "Linked preprocessing artifacts for ${subj}"
        elif [ -d "${NEW_PREPROC}" ]; then
            log_skip "Preprocessing for ${subj} already exists"
        fi
    done

    mark_done "link_data"
    log_ok "Data linking complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 5: Check/Download NSD-Imagery Data
# ==============================================================================
log_step "5/12  NSD-Imagery Data"

if is_done "imagery_data"; then
    log_skip "NSD-Imagery data already set up"
else
    STEP_START=$(date +%s)

    # NSD-Imagery is available from OpenNeuro (ds004937) or NSD S3 bucket
    # The imagery betas are typically in a separate directory structure
    IMAGERY_DIR="${PROJECT_DIR}/data/nsd_imagery"

    # Check if imagery data already exists
    IMAGERY_BETAS="${IMAGERY_DIR}/${SUBJECT}"
    if [ -d "${IMAGERY_BETAS}" ] && [ "$(find "${IMAGERY_BETAS}" -name "*.nii.gz" 2>/dev/null | wc -l)" -gt 0 ]; then
        IMAGERY_COUNT=$(find "${IMAGERY_BETAS}" -name "*.nii.gz" 2>/dev/null | wc -l)
        log_ok "NSD-Imagery betas found for ${SUBJECT}: ${IMAGERY_COUNT} files"
    else
        log_info "NSD-Imagery data not found locally."
        log_info "Attempting to download from NSD S3 bucket (public, no credentials needed)..."

        # The NSD-Imagery extension data is in the NSD S3 bucket
        # Bucket: natural-scenes-dataset (public access)
        # Path: nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ (imagery sessions)
        #
        # Alternative: OpenNeuro ds004937
        # URL: https://openneuro.org/datasets/ds004937

        mkdir -p "${IMAGERY_DIR}"

        # Try downloading via the NSD-Imagery OpenNeuro dataset
        # This uses aws s3 sync with no authentication
        if command -v aws &>/dev/null; then
            log_info "Using AWS CLI to download imagery data..."
            run_cmd "aws s3 sync --no-sign-request \
                s3://openneuro.org/ds004937/ \
                ${IMAGERY_DIR}/ \
                --exclude '*' \
                --include '${SUBJECT}/*' \
                2>&1 | tail -5" || true
        else
            log_info "AWS CLI not available. Installing..."
            run_cmd "$CONDA_PIP install --quiet awscli 2>&1 | tail -2"
            # Try via Python/boto3 as fallback
            $CONDA_PYTHON << PYEOF || true
import subprocess, sys, os

imagery_dir = "${IMAGERY_DIR}"
subject = "${SUBJECT}"

# Try OpenNeuro download
print(f"  Downloading NSD-Imagery for {subject} from OpenNeuro ds004937...")
try:
    result = subprocess.run([
        "aws", "s3", "sync", "--no-sign-request",
        f"s3://openneuro.org/ds004937/",
        imagery_dir,
        "--exclude", "*",
        "--include", f"*{subject}*",
    ], capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        print(f"  Download complete for {subject}")
    else:
        print(f"  S3 download returned code {result.returncode}: {result.stderr[:200]}")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  You may need to manually download from:")
    print("  https://openneuro.org/datasets/ds004937")
    print(f"  Place data in: {imagery_dir}/{subject}/")
PYEOF
        fi

        # Verify download
        if [ -d "${IMAGERY_BETAS}" ] && [ "$(find "${IMAGERY_BETAS}" -name "*.nii.gz" 2>/dev/null | wc -l)" -gt 0 ]; then
            IMAGERY_COUNT=$(find "${IMAGERY_BETAS}" -name "*.nii.gz" 2>/dev/null | wc -l)
            log_ok "NSD-Imagery downloaded: ${IMAGERY_COUNT} files for ${SUBJECT}"
        else
            log_warn "NSD-Imagery download may have failed or data not yet available."
            log_warn "Manual download required from: https://openneuro.org/datasets/ds004937"
            log_warn "Place files in: ${IMAGERY_DIR}/${SUBJECT}/"
            log_warn "Continuing setup — imagery-specific steps will be skipped."
        fi
    fi

    mark_done "imagery_data"
    log_ok "NSD-Imagery step complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 6: Build/Verify NSD Perception Index
# ==============================================================================
log_step "6/12  Build NSD Perception Index"

if is_done "perception_index"; then
    log_skip "Perception index already built"
else
    STEP_START=$(date +%s)
    INDEX_FILE="${PROJECT_DIR}/data/indices/nsd_index/subject=${SUBJECT}/index.parquet"

    if [ -f "${INDEX_FILE}" ]; then
        ROW_COUNT=$($CONDA_PYTHON -c "import pandas as pd; print(len(pd.read_parquet('${INDEX_FILE}')))" 2>/dev/null || echo "0")
        log_ok "Perception index exists: ${ROW_COUNT} rows"
    else
        log_info "Building perception index for ${SUBJECT}..."
        if [ -f "${PROJECT_DIR}/scripts/build_full_index.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/build_full_index.py \
                --cache-root ${PROJECT_DIR}/cache \
                --subject ${SUBJECT} \
                --output ${PROJECT_DIR}/data/indices/nsd_index/ \
                2>&1 | tee ${LOG_DIR}/build_index_${SUBJECT}.log | tail -10"
        else
            log_warn "build_full_index.py not found — using index from old project"
        fi
    fi

    mark_done "perception_index"
    log_ok "Perception index ready ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 7: Build/Verify CLIP Embedding Cache
# ==============================================================================
log_step "7/12  Build CLIP Embedding Cache"

if is_done "clip_cache"; then
    log_skip "CLIP cache already built"
else
    STEP_START=$(date +%s)
    CLIP_FILE="${PROJECT_DIR}/outputs/clip_cache/clip.parquet"

    if [ -f "${CLIP_FILE}" ]; then
        CLIP_SIZE=$(du -h "${CLIP_FILE}" | cut -f1)
        log_ok "CLIP cache exists (${CLIP_SIZE})"
    else
        log_info "Building CLIP embedding cache (batch_size=${BATCH_SIZE_CLIP})..."
        log_info "This pre-computes CLIP ViT-L/14 embeddings for all ~73K NSD stimuli."
        log_info "Expected time on H100: ~30-60 minutes"

        if [ -f "${PROJECT_DIR}/scripts/build_clip_cache.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/build_clip_cache.py \
                --cache-root ${PROJECT_DIR}/cache \
                --output ${CLIP_FILE} \
                --batch-size ${BATCH_SIZE_CLIP} \
                2>&1 | tee ${LOG_DIR}/build_clip_cache.log | tail -10"
        else
            log_warn "build_clip_cache.py not found — CLIP cache must be built manually"
        fi
    fi

    mark_done "clip_cache"
    log_ok "CLIP cache ready ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 8: Preprocessing (3-stage: z-score → standardize → PCA)
# ==============================================================================
log_step "8/12  Preprocessing Pipeline"

if is_done "preprocessing"; then
    log_skip "Preprocessing already complete"
else
    STEP_START=$(date +%s)

    # Check if preprocessing from old project is usable
    PREPROC_DIR="${PROJECT_DIR}/cache/preproc/subject=${SUBJECT}"
    if [ -d "${PREPROC_DIR}" ]; then
        log_ok "Preprocessing artifacts found (linked from old project)"
        log_info "Checking compatibility..."
        if [ -f "${PREPROC_DIR}/${SUBJECT}/meta.json" ]; then
            PCA_K=$($CONDA_PYTHON -c "import json; print(json.load(open('${PREPROC_DIR}/${SUBJECT}/meta.json'))['pca_components'])" 2>/dev/null || echo "unknown")
            log_info "Old preprocessing PCA components: ${PCA_K}"
            if [ "$PCA_K" != "512" ]; then
                log_warn "Old preprocessing uses PCA k=${PCA_K}, new project expects k=512"
                log_info "Will re-run preprocessing with k=512..."
                NEEDS_REPREPROC=true
            else
                NEEDS_REPREPROC=false
            fi
        else
            NEEDS_REPREPROC=true
        fi
    else
        NEEDS_REPREPROC=true
    fi

    if [ "$NEEDS_REPREPROC" = true ]; then
        log_info "Running 3-stage preprocessing for ${SUBJECT}..."
        log_info "T0: per-volume z-score → T1: reliability masking → T2: PCA (k=512)"

        if [ -f "${PROJECT_DIR}/scripts/fit_preprocessing.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/fit_preprocessing.py \
                --subject ${SUBJECT} \
                2>&1 | tee ${LOG_DIR}/preprocessing_${SUBJECT}.log | tail -10"
        else
            log_warn "fit_preprocessing.py not found — preprocessing must be run manually"
        fi
    else
        log_ok "Preprocessing compatible — reusing existing artifacts"
    fi

    mark_done "preprocessing"
    log_ok "Preprocessing complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 9: Build NSD-Imagery Index
# ==============================================================================
log_step "9/12  Build NSD-Imagery Index"

if is_done "imagery_index"; then
    log_skip "Imagery index already built"
else
    STEP_START=$(date +%s)
    IMAGERY_INDEX="${PROJECT_DIR}/cache/indices/imagery/${SUBJECT}.parquet"
    IMAGERY_DIR="${PROJECT_DIR}/data/nsd_imagery"

    if [ -f "${IMAGERY_INDEX}" ]; then
        log_ok "Imagery index already exists"
    elif [ -d "${IMAGERY_DIR}/${SUBJECT}" ] || [ -d "${IMAGERY_DIR}" ]; then
        log_info "Building NSD-Imagery index for ${SUBJECT}..."

        if [ -f "${PROJECT_DIR}/scripts/build_nsd_imagery_index.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/build_nsd_imagery_index.py \
                --subject ${SUBJECT} \
                --data-root ${IMAGERY_DIR} \
                --cache-root ${PROJECT_DIR}/cache/ \
                --output ${IMAGERY_INDEX} \
                --verbose \
                2>&1 | tee ${LOG_DIR}/build_imagery_index_${SUBJECT}.log | tail -10"
        fi
    else
        log_warn "No NSD-Imagery data found — skipping imagery index"
        log_warn "Download from https://openneuro.org/datasets/ds004937"
    fi

    mark_done "imagery_index"
    log_ok "Imagery index step complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 10: Train Models (Ridge → MLP → Two-Stage)
# ==============================================================================
log_step "10/12  Train Perception Models"

if [ "$SKIP_TRAINING" = true ]; then
    log_skip "Training skipped (--skip-training flag)"
elif is_done "train_models"; then
    log_skip "Models already trained"
else
    STEP_START=$(date +%s)

    # Source H100 optimizations
    source "${PROJECT_DIR}/.env_h100"

    # --- Ridge Regression (fast baseline) ---
    RIDGE_CKPT="${PROJECT_DIR}/checkpoints/ridge/${SUBJECT}/best.pt"
    if [ -f "${RIDGE_CKPT}" ]; then
        log_skip "Ridge model already trained"
    else
        log_info "Training Ridge regression baseline (~5 min on H100)..."
        if [ -f "${PROJECT_DIR}/scripts/run_full_pipeline.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/run_full_pipeline.py \
                --subject ${SUBJECT} --mode baseline \
                --resume-from train \
                2>&1 | tee ${LOG_DIR}/train_ridge_${SUBJECT}.log | tail -15"
        else
            log_warn "run_full_pipeline.py not found — train Ridge manually"
        fi
        log_ok "Ridge training complete"
    fi

    # --- MLP Encoder ---
    MLP_CKPT="${PROJECT_DIR}/checkpoints/mlp/${SUBJECT}/best.pt"
    if [ -f "${MLP_CKPT}" ]; then
        log_skip "MLP model already trained"
    else
        log_info "Training MLP encoder (~30 min on H100)..."

        # Try dedicated MLP training script first, fall back to pipeline
        if [ -f "${PROJECT_DIR}/scripts/train_mlp.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/train_mlp.py \
                --subject ${SUBJECT} \
                --output-dir ${PROJECT_DIR}/checkpoints/mlp/${SUBJECT} \
                --batch-size ${BATCH_SIZE_TRAIN} \
                2>&1 | tee ${LOG_DIR}/train_mlp_${SUBJECT}.log | tail -15"
        fi
        log_ok "MLP training complete"
    fi

    # --- Two-Stage Encoder (main model) ---
    TWOSTAGE_CKPT="${PROJECT_DIR}/checkpoints/two_stage/${SUBJECT}/best.pt"
    # Also check alternative checkpoint names
    TWOSTAGE_CKPT_ALT="${PROJECT_DIR}/checkpoints/two_stage/${SUBJECT}/two_stage_best.pt"
    if [ -f "${TWOSTAGE_CKPT}" ] || [ -f "${TWOSTAGE_CKPT_ALT}" ]; then
        log_skip "Two-Stage model already trained"
    else
        log_info "Training Two-Stage encoder with novel contributions (~3-4 hours on H100)..."
        log_info "Features: soft reliability weighting + InfoNCE loss + multi-layer CLIP"

        if [ -f "${PROJECT_DIR}/scripts/run_full_pipeline.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/run_full_pipeline.py \
                --subject ${SUBJECT} --mode novel \
                2>&1 | tee ${LOG_DIR}/train_two_stage_${SUBJECT}.log | tail -20"
        fi
        log_ok "Two-Stage training complete"
    fi

    mark_done "train_models"
    log_ok "Model training complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 11: Cross-Domain Evaluation & Adapters
# ==============================================================================
log_step "11/12  Cross-Domain Evaluation & Adapter Training"

if [ "$SKIP_TRAINING" = true ]; then
    log_skip "Evaluation skipped (--skip-training flag)"
elif is_done "eval_and_adapters"; then
    log_skip "Evaluation & adapters already complete"
else
    STEP_START=$(date +%s)

    # Source H100 optimizations
    source "${PROJECT_DIR}/.env_h100"

    # Find the best checkpoint
    BEST_CKPT=""
    for ckpt_path in \
        "${PROJECT_DIR}/checkpoints/two_stage/${SUBJECT}/best.pt" \
        "${PROJECT_DIR}/checkpoints/two_stage/${SUBJECT}/two_stage_best.pt" \
        "${PROJECT_DIR}/checkpoints/mlp/${SUBJECT}/best.pt" \
        "${PROJECT_DIR}/checkpoints/ridge/${SUBJECT}/best.pt"; do
        if [ -f "$ckpt_path" ]; then
            BEST_CKPT="$ckpt_path"
            break
        fi
    done

    if [ -z "$BEST_CKPT" ]; then
        log_warn "No trained checkpoint found — skipping evaluation"
    else
        log_ok "Using checkpoint: $(basename $(dirname $BEST_CKPT))/$(basename $BEST_CKPT)"

        # Determine model type from checkpoint path
        MODEL_TYPE="two_stage"
        echo "$BEST_CKPT" | grep -q "mlp" && MODEL_TYPE="mlp"
        echo "$BEST_CKPT" | grep -q "ridge" && MODEL_TYPE="ridge"

        # --- Perception baseline evaluation ---
        log_info "Evaluating on perception test set (within-domain)..."
        IMAGERY_INDEX="${PROJECT_DIR}/cache/indices/imagery/${SUBJECT}.parquet"

        if [ -f "${IMAGERY_INDEX}" ] && [ -f "${PROJECT_DIR}/scripts/eval_perception_to_imagery_transfer.py" ]; then
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/eval_perception_to_imagery_transfer.py \
                --index ${IMAGERY_INDEX} \
                --checkpoint ${BEST_CKPT} \
                --model-type ${MODEL_TYPE} \
                --mode perception --split test \
                --output-dir ${PROJECT_DIR}/outputs/reports/imagery/perception_baseline \
                2>&1 | tee ${LOG_DIR}/eval_perception_${SUBJECT}.log | tail -10"
            log_ok "Perception evaluation complete"

            # --- Cross-domain evaluation (perception → imagery) ---
            log_info "Evaluating on imagery test set (cross-domain H1)..."
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/eval_perception_to_imagery_transfer.py \
                --index ${IMAGERY_INDEX} \
                --checkpoint ${BEST_CKPT} \
                --model-type ${MODEL_TYPE} \
                --mode imagery --split test \
                --output-dir ${PROJECT_DIR}/outputs/reports/imagery/cross_domain \
                2>&1 | tee ${LOG_DIR}/eval_imagery_${SUBJECT}.log | tail -10"
            log_ok "Cross-domain evaluation complete"

            # --- Adapter ablation suite (H3) ---
            log_info "Running adapter ablation suite (Linear + MLP + MLP+Condition)..."
            log_info "This trains 3 adapter variants and evaluates each (~1 hour total on H100)"

            if [ -f "${PROJECT_DIR}/scripts/run_imagery_ablations.py" ]; then
                run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/run_imagery_ablations.py \
                    --index ${IMAGERY_INDEX} \
                    --checkpoint ${BEST_CKPT} \
                    --model-type ${MODEL_TYPE} \
                    --output-dir ${PROJECT_DIR}/outputs/imagery_ablations/${SUBJECT} \
                    --epochs ${ADAPTER_EPOCHS} \
                    --with-condition \
                    --device cuda \
                    2>&1 | tee ${LOG_DIR}/ablations_${SUBJECT}.log | tail -20"
                log_ok "Adapter ablation suite complete"
            fi
        else
            log_warn "Imagery index not found or eval script missing — skipping cross-domain evaluation"
            log_warn "This is expected if NSD-Imagery data was not downloaded"
        fi

        # --- Shared-1000 evaluation (within-domain comprehensive) ---
        if [ -f "${PROJECT_DIR}/scripts/eval_shared1000_full.py" ]; then
            log_info "Running Shared-1000 evaluation..."
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/eval_shared1000_full.py \
                --subject ${SUBJECT} \
                --checkpoint ${BEST_CKPT} \
                --model-type ${MODEL_TYPE} \
                --output-dir ${PROJECT_DIR}/outputs/reports/shared1000/${SUBJECT} \
                2>&1 | tee ${LOG_DIR}/eval_shared1000_${SUBJECT}.log | tail -10" || true
        fi
    fi

    mark_done "eval_and_adapters"
    log_ok "Evaluation & adapters complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# STEP 12: Novel Analyses & Figures
# ==============================================================================
log_step "12/12  Novel Analyses & Publication Figures"

if [ "$SKIP_TRAINING" = true ]; then
    log_skip "Analyses skipped (--skip-training flag)"
elif is_done "novel_analyses"; then
    log_skip "Novel analyses already complete"
else
    STEP_START=$(date +%s)

    # Source H100 optimizations
    source "${PROJECT_DIR}/.env_h100"

    # --- Run all 15 novel analysis directions ---
    if [ -f "${PROJECT_DIR}/scripts/run_novel_analyses.py" ] && \
       [ -f "${PROJECT_DIR}/configs/experiments/novel_analyses.yaml" ]; then
        log_info "Running 15 novel analysis directions..."
        log_info "Dimensionality · Uncertainty · Semantic Survival · Topological RSA · ..."

        run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/run_novel_analyses.py \
            --config ${PROJECT_DIR}/configs/experiments/novel_analyses.yaml \
            2>&1 | tee ${LOG_DIR}/novel_analyses.log | tail -20" || true
        log_ok "Novel analyses complete"
    fi

    # --- Generate paper figures ---
    if [ -f "${PROJECT_DIR}/scripts/make_paper_figures.py" ]; then
        ABLATION_DIR="${PROJECT_DIR}/outputs/imagery_ablations/${SUBJECT}"
        if [ -d "${ABLATION_DIR}" ]; then
            log_info "Generating paper-ready figures..."
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/make_paper_figures.py \
                --ablation-dir ${ABLATION_DIR} \
                --output-dir ${ABLATION_DIR}/figures \
                2>&1 | tee ${LOG_DIR}/paper_figures.log | tail -10" || true
            log_ok "Paper figures generated"
        fi
    fi

    # --- Generate novel analysis figures ---
    if [ -f "${PROJECT_DIR}/scripts/make_novel_figures.py" ]; then
        NOVEL_DIR="${PROJECT_DIR}/outputs/novel_analyses"
        if [ -d "${NOVEL_DIR}" ]; then
            log_info "Generating novel analysis figures..."
            run_cmd "$CONDA_PYTHON ${PROJECT_DIR}/scripts/make_novel_figures.py \
                --results-dir ${NOVEL_DIR} \
                2>&1 | tee ${LOG_DIR}/novel_figures.log | tail -10" || true
            log_ok "Novel analysis figures generated"
        fi
    fi

    mark_done "novel_analyses"
    log_ok "Analyses & figures complete ($(elapsed_time $STEP_START))"
fi

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                     Setup Complete!                             ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total time: ${BOLD}$(elapsed_time $SETUP_START)${NC}"
echo -e "  Subject:    ${BOLD}${SUBJECT}${NC}"
echo -e "  Project:    ${PROJECT_DIR}"
echo ""

# Show what's available
echo -e "${BOLD}  Artifacts:${NC}"
[ -f "${PROJECT_DIR}/data/indices/nsd_index/subject=${SUBJECT}/index.parquet" ] && \
    echo -e "    ${GREEN}✓${NC} Perception index"
[ -f "${PROJECT_DIR}/cache/indices/imagery/${SUBJECT}.parquet" ] && \
    echo -e "    ${GREEN}✓${NC} Imagery index"
[ -f "${PROJECT_DIR}/outputs/clip_cache/clip.parquet" ] && \
    echo -e "    ${GREEN}✓${NC} CLIP embedding cache"
[ -d "${PROJECT_DIR}/cache/preproc/subject=${SUBJECT}" ] && \
    echo -e "    ${GREEN}✓${NC} Preprocessing artifacts"
[ -d "${PROJECT_DIR}/cache/preextracted/subject=${SUBJECT}" ] && \
    echo -e "    ${GREEN}✓${NC} Pre-extracted fMRI features"

for model in ridge mlp two_stage; do
    ckpt=$(find "${PROJECT_DIR}/checkpoints/${model}/${SUBJECT}/" -name "*.pt" 2>/dev/null | head -1)
    [ -n "$ckpt" ] && echo -e "    ${GREEN}✓${NC} ${model} checkpoint"
done

[ -f "${PROJECT_DIR}/outputs/reports/imagery/perception_baseline/metrics.json" ] && \
    echo -e "    ${GREEN}✓${NC} Perception baseline evaluation"
[ -f "${PROJECT_DIR}/outputs/reports/imagery/cross_domain/metrics.json" ] && \
    echo -e "    ${GREEN}✓${NC} Cross-domain evaluation"
[ -f "${PROJECT_DIR}/outputs/imagery_ablations/${SUBJECT}/results_table.csv" ] && \
    echo -e "    ${GREEN}✓${NC} Adapter ablation results"

echo ""
echo -e "${BOLD}  Next Steps:${NC}"
echo -e "    # Re-run to resume from where you left off:"
echo -e "    bash scripts/setup_h100_experiment.sh ${SUBJECT}"
echo ""
echo -e "    # Run for additional subjects:"
echo -e "    bash scripts/setup_h100_experiment.sh subj02"
echo -e "    bash scripts/setup_h100_experiment.sh subj05"
echo -e "    bash scripts/setup_h100_experiment.sh subj07"
echo ""
echo -e "    # Run tests:"
echo -e "    cd ${PROJECT_DIR} && $CONDA_PYTHON -m pytest tests/ -v"
echo ""
echo -e "    # Check project status:"
echo -e "    bash ${PROJECT_DIR}/scripts/check_setup.sh ${SUBJECT}"
echo ""
echo -e "    # View results:"
echo -e "    cat ${PROJECT_DIR}/outputs/imagery_ablations/${SUBJECT}/results_table.md"
echo ""
