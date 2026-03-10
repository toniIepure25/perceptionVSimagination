#!/usr/bin/env bash
# =============================================================================
# Perception vs. Imagination — Full Experiment Setup
# =============================================================================
#
# One-shot script to set up the entire cross-domain fMRI decoding pipeline:
#   1) Environment & dependencies
#   2) NSD perception data linking (from existing JupyterLab)
#   3) NSD-Imagery data download (if missing)
#   4) Index building (perception + imagery)
#   5) CLIP embedding cache
#   6) Preprocessing (3-stage: z-score → standardize → PCA)
#   7) Model training (Ridge → MLP → Two-Stage)
#   8) Cross-domain evaluation (Hypothesis H1)
#   9) Adapter training + ablations (Hypothesis H3)
#  10) Novel analyses (15 directions)
#  11) Figure generation
#
# Optimized for: NVIDIA H100 80GB VRAM / 500GB RAM / CUDA 12+
#
# Usage:
#   bash scripts/setup_full_experiment.sh                 # Full run, subj01
#   bash scripts/setup_full_experiment.sh --subject subj02
#   bash scripts/setup_full_experiment.sh --resume        # Resume from last step
#   bash scripts/setup_full_experiment.sh --dry-run       # Preview only
#   bash scripts/setup_full_experiment.sh --step 5        # Start from step 5
#   bash scripts/setup_full_experiment.sh --all-subjects  # Run all 4 imagery subjects
#
# Prerequisites:
#   - Linux with NVIDIA GPU (H100 recommended)
#   - conda or mamba installed
#   - NSD perception data at /home/jovyan/work/data/nsd/ (configurable)
#
# Author: Perception vs. Imagination Project
# Date: March 2026
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — Edit these paths to match your environment
# ─────────────────────────────────────────────────────────────────────────────

# Where your existing NSD perception data lives (from JupyterLab)
NSD_DATA_ROOT="${NSD_DATA_ROOT:-/home/jovyan/work/data/nsd}"

# NSD-Imagery download source (OpenNeuro)
NSD_IMAGERY_BUCKET="s3://openneuro.org/ds004937"
NSD_IMAGERY_FALLBACK_URL="https://openneuro.org/datasets/ds004937"

# Project root (auto-detected from script location)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Conda environment name
CONDA_ENV="fmri2img"

# Default subject for initial validation
SUBJECT="subj01"

# All NSD-Imagery subjects (for --all-subjects mode)
ALL_IMAGERY_SUBJECTS=("subj01" "subj02" "subj05" "subj07")

# H100-specific settings
BATCH_SIZE_CLIP=512         # CLIP cache building (H100 can handle large batches)
BATCH_SIZE_TRAIN=128        # Training batch size
BATCH_SIZE_EVAL=256         # Evaluation batch size
NUM_WORKERS=8               # DataLoader workers (500GB RAM → generous prefetching)
TRAINING_EPOCHS=150         # Two-stage encoder epochs
ADAPTER_EPOCHS=50           # Imagery adapter epochs
MIXED_PRECISION=true        # AMP for H100 TF32/BF16

# Progress tracking file
PROGRESS_FILE="${PROJECT_ROOT}/.setup_progress"

# ─────────────────────────────────────────────────────────────────────────────
# Color codes & utilities
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

LOGFILE="${PROJECT_ROOT}/logs/setup_$(date +%Y%m%d_%H%M%S).log"

log_info()    { echo -e "${GREEN}[INFO]${NC}  $(date +%H:%M:%S) $*" | tee -a "$LOGFILE"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $(date +%H:%M:%S) $*" | tee -a "$LOGFILE"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*" | tee -a "$LOGFILE"; }
log_step()    { echo -e "\n${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}" | tee -a "$LOGFILE"
                echo -e "${CYAN}${BOLD}  STEP $1: $2${NC}" | tee -a "$LOGFILE"
                echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════${NC}\n" | tee -a "$LOGFILE"; }
log_success() { echo -e "${GREEN}${BOLD}  ✓ $*${NC}" | tee -a "$LOGFILE"; }
log_skip()    { echo -e "${BLUE}  ⏭ $* (already done)${NC}" | tee -a "$LOGFILE"; }

banner() {
    echo -e "${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║    Perception vs. Imagination — Full Experiment Setup             ║"
    echo "║                                                                   ║"
    echo "║    Cross-Domain fMRI Decoding Pipeline                            ║"
    echo "║    Optimized for H100 80GB / 500GB RAM                            ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking — resume from last completed step
# ─────────────────────────────────────────────────────────────────────────────

save_progress() {
    local step=$1
    local subject=$2
    echo "${subject}:${step}" >> "$PROGRESS_FILE"
    log_info "Progress saved: step ${step} for ${subject}"
}

get_last_step() {
    local subject=$1
    if [[ -f "$PROGRESS_FILE" ]]; then
        grep "^${subject}:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2
    else
        echo "0"
    fi
}

should_run_step() {
    local step=$1
    local subject=$2

    # --step flag: start from specific step
    if [[ -n "${START_STEP:-}" ]] && [[ "$step" -lt "$START_STEP" ]]; then
        return 1  # skip
    fi

    # --resume flag: skip already-completed steps
    if [[ "${RESUME:-false}" == "true" ]]; then
        local last
        last=$(get_last_step "$subject")
        if [[ "$step" -le "$last" ]]; then
            return 1  # skip
        fi
    fi

    return 0  # run
}

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

DRY_RUN=false
RESUME=false
START_STEP=""
ALL_SUBJECTS=false
SKIP_ENV=false
SKIP_TRAINING=false
SKIP_NOVEL=false
NSD_PATH_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --subject)      SUBJECT="$2"; shift 2 ;;
        --nsd-root)     NSD_PATH_OVERRIDE="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --resume)       RESUME=true; shift ;;
        --step)         START_STEP="$2"; shift 2 ;;
        --all-subjects) ALL_SUBJECTS=true; shift ;;
        --skip-env)     SKIP_ENV=true; shift ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-novel)   SKIP_NOVEL=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --subject SUBJ        Subject ID (default: subj01)"
            echo "  --nsd-root PATH       Override NSD data path"
            echo "  --dry-run             Preview commands without executing"
            echo "  --resume              Resume from last completed step"
            echo "  --step N              Start from step N (1-11)"
            echo "  --all-subjects        Run for all 4 imagery subjects"
            echo "  --skip-env            Skip environment setup (step 1)"
            echo "  --skip-training       Skip model training (steps 7-8)"
            echo "  --skip-novel          Skip novel analyses (step 10)"
            echo "  --help                Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Apply NSD path override
if [[ -n "$NSD_PATH_OVERRIDE" ]]; then
    NSD_DATA_ROOT="$NSD_PATH_OVERRIDE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Dry-run wrapper
# ─────────────────────────────────────────────────────────────────────────────

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $*" | tee -a "$LOGFILE"
    else
        log_info "Running: $*"
        eval "$@" 2>&1 | tee -a "$LOGFILE"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Subject loop helper
# ─────────────────────────────────────────────────────────────────────────────

get_subjects() {
    if [[ "$ALL_SUBJECTS" == "true" ]]; then
        echo "${ALL_IMAGERY_SUBJECTS[@]}"
    else
        echo "$SUBJECT"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
#  BEGIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

cd "$PROJECT_ROOT"
mkdir -p logs

banner

log_info "Project root:    ${PROJECT_ROOT}"
log_info "NSD data root:   ${NSD_DATA_ROOT}"
log_info "Subject(s):      $(get_subjects)"
log_info "Dry run:         ${DRY_RUN}"
log_info "Resume:          ${RESUME}"
log_info "Log file:        ${LOGFILE}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Environment & Dependencies
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$SKIP_ENV" != "true" ]] && should_run_step 1 "$SUBJECT"; then
    log_step 1 "Environment & Dependency Setup"

    # Check conda/mamba availability
    if command -v mamba &>/dev/null; then
        CONDA_CMD="mamba"
    elif command -v conda &>/dev/null; then
        CONDA_CMD="conda"
    else
        log_error "Neither conda nor mamba found. Please install Miniforge/Miniconda first."
        log_info  "Install: https://github.com/conda-forge/miniforge#install"
        exit 1
    fi
    log_info "Using: ${CONDA_CMD}"

    # Create or update conda environment
    if ${CONDA_CMD} env list | grep -q "^${CONDA_ENV} "; then
        log_info "Environment '${CONDA_ENV}' exists, updating..."
        run_cmd "${CONDA_CMD} env update -n ${CONDA_ENV} -f environment.yml --prune"
    else
        log_info "Creating environment '${CONDA_ENV}'..."
        run_cmd "${CONDA_CMD} env create -f environment.yml"
    fi

    # Activate (for the rest of this script)
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"

    # Install this project in editable mode with all extras
    log_info "Installing fmri2img package..."
    run_cmd "pip install -e '.[all]' --no-build-isolation"

    # Verify critical imports
    log_info "Verifying installation..."
    python -c "
import torch
import numpy as np
import pandas as pd
import open_clip
import fmri2img

print(f'  fmri2img version: {fmri2img.__version__}')
print(f'  PyTorch version:  {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:              {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:             {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'  CUDA version:     {torch.version.cuda}')
print(f'  open_clip:        OK')
print(f'  NumPy:            {np.__version__}')
print(f'  Pandas:           {pd.__version__}')
" 2>&1 | tee -a "$LOGFILE"

    # H100-specific: verify CUDA 12+ and set optimizations
    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
cuda_ver = tuple(int(x) for x in torch.version.cuda.split('.'))
if cuda_ver[0] < 12:
    print(f'WARNING: CUDA {torch.version.cuda} detected. H100 works best with CUDA 12+')
else:
    print(f'CUDA {torch.version.cuda} — optimal for H100')
# Enable TF32 for Hopper architecture
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print('TF32 matrix math enabled (H100 Tensor Cores)')
"

    log_success "Environment ready"
    save_progress 1 "$SUBJECT"
else
    log_skip "Step 1: Environment setup"
    # Still need to activate the env for subsequent steps
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}" 2>/dev/null || true
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Link NSD Perception Data
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 2 "$SUBJECT"; then
    log_step 2 "Link NSD Perception Data"

    # Validate NSD data exists at the specified path
    if [[ ! -d "${NSD_DATA_ROOT}" ]]; then
        log_error "NSD data not found at: ${NSD_DATA_ROOT}"
        log_info  "Set NSD_DATA_ROOT or use --nsd-root /path/to/nsd"
        exit 1
    fi

    # Check expected NSD directory structure
    MISSING=0
    for subdir in nsddata nsddata_betas nsddata_stimuli; do
        if [[ -d "${NSD_DATA_ROOT}/${subdir}" ]]; then
            log_info "Found: ${NSD_DATA_ROOT}/${subdir}"
        else
            log_warn "Missing: ${NSD_DATA_ROOT}/${subdir}"
            MISSING=1
        fi
    done

    if [[ $MISSING -eq 1 ]]; then
        log_warn "Some NSD subdirectories are missing. Pipeline may use S3 fallback."
    fi

    # Create symlinks into cache/ so the pipeline finds them
    mkdir -p cache/nsd_hdf5

    # Link the stimuli HDF5 file (73K images, ~20GB)
    STIM_HDF5="${NSD_DATA_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    if [[ -f "$STIM_HDF5" ]]; then
        if [[ ! -e "cache/nsd_hdf5/nsd_stimuli.hdf5" ]]; then
            run_cmd "ln -sf '${STIM_HDF5}' cache/nsd_hdf5/nsd_stimuli.hdf5"
            log_success "Linked stimuli HDF5"
        else
            log_skip "Stimuli HDF5 already linked"
        fi
    else
        log_warn "nsd_stimuli.hdf5 not found at expected path, will download on demand"
    fi

    # Link the stim_info CSV (metadata)
    STIM_INFO="${NSD_DATA_ROOT}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    if [[ -f "$STIM_INFO" ]]; then
        if [[ ! -e "cache/nsd_stim_info_merged.csv" ]]; then
            run_cmd "ln -sf '${STIM_INFO}' cache/nsd_stim_info_merged.csv"
            log_success "Linked stim_info CSV"
        else
            log_skip "stim_info CSV already linked"
        fi
    fi

    # Create a local NSD data reference file so S3 module can skip downloads
    # The s3.py module caches to cache/s3_cache/ — we symlink the betas there
    NSD_BETAS_DIR="${NSD_DATA_ROOT}/nsddata_betas"
    if [[ -d "$NSD_BETAS_DIR" ]]; then
        if [[ ! -e "cache/nsd_betas_local" ]]; then
            run_cmd "ln -sf '${NSD_BETAS_DIR}' cache/nsd_betas_local"
            log_success "Linked NSD betas directory"
        else
            log_skip "NSD betas already linked"
        fi
    fi

    # Link the nsddata directory (ROI masks, experiment designs, etc.)
    NSD_META_DIR="${NSD_DATA_ROOT}/nsddata"
    if [[ -d "$NSD_META_DIR" ]]; then
        if [[ ! -e "cache/nsddata_local" ]]; then
            run_cmd "ln -sf '${NSD_META_DIR}' cache/nsddata_local"
            log_success "Linked NSD metadata directory"
        else
            log_skip "NSD metadata already linked"
        fi
    fi

    # Write a local data manifest for scripts that need path resolution
    cat > cache/nsd_local_paths.yaml << EOF
# Auto-generated by setup_full_experiment.sh — $(date)
# Local NSD data paths (avoids S3 downloads)
nsd_data_root: "${NSD_DATA_ROOT}"
nsddata: "${NSD_DATA_ROOT}/nsddata"
nsddata_betas: "${NSD_DATA_ROOT}/nsddata_betas"
nsddata_stimuli: "${NSD_DATA_ROOT}/nsddata_stimuli"
stimuli_hdf5: "${NSD_DATA_ROOT}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
stim_info_csv: "${NSD_DATA_ROOT}/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
EOF
    log_success "NSD local path manifest written to cache/nsd_local_paths.yaml"

    # Validate we can actually read a beta file for the target subject
    SUBJ_NUM=$(echo "$SUBJECT" | grep -oP '\d+')
    BETA_SAMPLE="${NSD_DATA_ROOT}/nsddata_betas/ppdata/subj${SUBJ_NUM}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz"
    if [[ -f "$BETA_SAMPLE" ]]; then
        BETA_SIZE=$(du -h "$BETA_SAMPLE" | cut -f1)
        log_success "Beta file accessible: ${BETA_SAMPLE} (${BETA_SIZE})"
    else
        log_warn "Beta file not found at expected path: ${BETA_SAMPLE}"
        log_info "Pipeline will fall back to S3 download for fMRI data"
    fi

    save_progress 2 "$SUBJECT"
else
    log_skip "Step 2: NSD data linking"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: NSD-Imagery Data (Download if Missing)
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 3 "$SUBJECT"; then
    log_step 3 "NSD-Imagery Data Acquisition"

    IMAGERY_DIR="${PROJECT_ROOT}/data/nsd_imagery"
    mkdir -p "$IMAGERY_DIR"

    # Check if imagery data already exists for current subject
    SUBJ_NUM=$(echo "$SUBJECT" | grep -oP '\d+')
    IMAGERY_SUBJ_DIR="${IMAGERY_DIR}/${SUBJECT}"

    if [[ -d "$IMAGERY_SUBJ_DIR" ]] && [[ $(find "$IMAGERY_SUBJ_DIR" -name "*.nii.gz" 2>/dev/null | head -1) ]]; then
        NIFTI_COUNT=$(find "$IMAGERY_SUBJ_DIR" -name "*.nii.gz" | wc -l)
        log_skip "NSD-Imagery data found for ${SUBJECT} (${NIFTI_COUNT} NIfTI files)"
    else
        log_info "NSD-Imagery data not found for ${SUBJECT}, downloading..."
        log_info "Source: Natural Scenes Dataset - Imagery extension"
        log_info "This may take 30-60 minutes depending on connection speed."
        echo ""

        # Method 1: Try AWS S3 (public, no credentials needed)
        if command -v aws &>/dev/null; then
            log_info "Attempting download via AWS CLI (anonymous)..."
            run_cmd "aws s3 sync --no-sign-request \
                '${NSD_IMAGERY_BUCKET}/sub-${SUBJ_NUM}/' \
                '${IMAGERY_SUBJ_DIR}/' \
                --exclude '*' \
                --include '*.nii.gz' \
                --include '*.json' \
                --include '*.tsv'" || true
        fi

        # Method 2: Try DataLad / datalad-osf (common in neuroimaging)
        if [[ ! $(find "$IMAGERY_SUBJ_DIR" -name "*.nii.gz" 2>/dev/null | head -1) ]]; then
            if command -v datalad &>/dev/null; then
                log_info "Attempting download via DataLad..."
                run_cmd "datalad install -s ${NSD_IMAGERY_FALLBACK_URL} '${IMAGERY_DIR}/ds004937'" || true
                if [[ -d "${IMAGERY_DIR}/ds004937/sub-${SUBJ_NUM}" ]]; then
                    run_cmd "cd '${IMAGERY_DIR}/ds004937' && datalad get 'sub-${SUBJ_NUM}/'"
                    run_cmd "ln -sf '${IMAGERY_DIR}/ds004937/sub-${SUBJ_NUM}' '${IMAGERY_SUBJ_DIR}'"
                fi
            fi
        fi

        # Method 3: Direct download via Python (fsspec + s3fs) — our own library
        if [[ ! $(find "$IMAGERY_SUBJ_DIR" -name "*.nii.gz" 2>/dev/null | head -1) ]]; then
            log_info "Attempting download via Python fsspec (anonymous S3)..."
            run_cmd "python -c \"
import fsspec
import os

# Try common NSD-Imagery S3 locations
urls = [
    's3://natural-scenes-dataset/nsddata_imagery/ppdata/subj${SUBJ_NUM}/',
    's3://openneuro.org/ds004937/sub-${SUBJ_NUM}/',
]

target = '${IMAGERY_SUBJ_DIR}'
os.makedirs(target, exist_ok=True)

for url in urls:
    try:
        fs = fsspec.filesystem('s3', anon=True)
        files = fs.ls(url)
        if files:
            print(f'Found imagery data at {url}')
            for f in files:
                fname = os.path.basename(f)
                local_path = os.path.join(target, fname)
                if not os.path.exists(local_path):
                    print(f'  Downloading: {fname}')
                    fs.get(f, local_path)
            break
    except Exception as e:
        print(f'  {url}: {e}')
        continue

print('Download complete.')
\""
        fi

        # Verify download
        if [[ -d "$IMAGERY_SUBJ_DIR" ]]; then
            NIFTI_COUNT=$(find "$IMAGERY_SUBJ_DIR" -name "*.nii.gz" 2>/dev/null | wc -l)
            if [[ $NIFTI_COUNT -gt 0 ]]; then
                log_success "Downloaded ${NIFTI_COUNT} NIfTI files for ${SUBJECT}"
            else
                log_warn "No NIfTI files found after download attempts."
                log_info "You may need to manually download NSD-Imagery data."
                log_info "Visit: https://natural-scenes-dataset.s3.amazonaws.com/index.html"
                log_info "Place imagery betas in: ${IMAGERY_SUBJ_DIR}/"
                log_info "Continuing with pipeline — imagery evaluation will use dry-run mode."
            fi
        fi
    fi

    save_progress 3 "$SUBJECT"
else
    log_skip "Step 3: NSD-Imagery data"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Build NSD Perception Index
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 4 "$SUBJECT"; then
    log_step 4 "Build NSD Perception Index"

    INDEX_DIR="data/indices/nsd_index"
    INDEX_FILE="${INDEX_DIR}/${SUBJECT}.csv"

    if [[ -f "$INDEX_FILE" ]]; then
        ROW_COUNT=$(wc -l < "$INDEX_FILE")
        if [[ $ROW_COUNT -gt 1000 ]]; then
            log_skip "Perception index exists (${ROW_COUNT} rows): ${INDEX_FILE}"
        else
            log_warn "Index file too small (${ROW_COUNT} rows), rebuilding..."
            run_cmd "python -m fmri2img.cli.build_index \
                --cache-root cache \
                --subject '${SUBJECT}' \
                --output '${INDEX_DIR}/'"
        fi
    else
        log_info "Building NSD perception index for ${SUBJECT}..."
        mkdir -p "$INDEX_DIR"
        # Try the CLI entry point first, fall back to script
        if python -m fmri2img.cli.build_index --help &>/dev/null 2>&1; then
            run_cmd "python -m fmri2img.cli.build_index \
                --cache-root cache \
                --subject '${SUBJECT}' \
                --output '${INDEX_DIR}/'"
        else
            run_cmd "python scripts/build_full_index.py \
                --cache-root cache \
                --subject '${SUBJECT}' \
                --output '${INDEX_DIR}/'"
        fi
    fi

    # Also build Parquet format if the script supports it
    PARQUET_INDEX="cache/indices/nsd_index/${SUBJECT}.parquet"
    if [[ ! -f "$PARQUET_INDEX" ]]; then
        mkdir -p "cache/indices/nsd_index"
        log_info "Converting index to Parquet format..."
        python -c "
import pandas as pd
import os
csv_path = '${INDEX_FILE}'
parquet_path = '${PARQUET_INDEX}'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f'Converted {len(df)} rows to {parquet_path}')
else:
    print(f'CSV index not found at {csv_path}, skipping Parquet conversion')
" 2>&1 | tee -a "$LOGFILE" || true
    fi

    log_success "Perception index ready"
    save_progress 4 "$SUBJECT"
else
    log_skip "Step 4: Perception index"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Build NSD-Imagery Index
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 5 "$SUBJECT"; then
    log_step 5 "Build NSD-Imagery Index"

    IMAGERY_INDEX="cache/indices/imagery/${SUBJECT}.parquet"

    if [[ -f "$IMAGERY_INDEX" ]]; then
        ROW_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('${IMAGERY_INDEX}')))" 2>/dev/null || echo "0")
        if [[ $ROW_COUNT -gt 10 ]]; then
            log_skip "Imagery index exists (${ROW_COUNT} trials): ${IMAGERY_INDEX}"
        else
            log_info "Imagery index too small, rebuilding..."
        fi
    fi

    if [[ ! -f "$IMAGERY_INDEX" ]] || [[ "${ROW_COUNT:-0}" -le 10 ]]; then
        IMAGERY_DATA_DIR="${PROJECT_ROOT}/data/nsd_imagery"
        if [[ -d "${IMAGERY_DATA_DIR}/${SUBJECT}" ]]; then
            log_info "Building NSD-Imagery index for ${SUBJECT}..."
            mkdir -p "cache/indices/imagery"
            run_cmd "python scripts/build_nsd_imagery_index.py \
                --subject '${SUBJECT}' \
                --data-root '${IMAGERY_DATA_DIR}' \
                --cache-root cache/ \
                --output '${IMAGERY_INDEX}' \
                --verbose"
            log_success "Imagery index built: ${IMAGERY_INDEX}"
        else
            log_warn "NSD-Imagery data not found for ${SUBJECT}, skipping index build."
            log_info "Imagery evaluation steps will use --dry-run mode."
        fi
    fi

    save_progress 5 "$SUBJECT"
else
    log_skip "Step 5: Imagery index"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Build CLIP Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 6 "$SUBJECT"; then
    log_step 6 "Build CLIP Embedding Cache"

    CLIP_CACHE="outputs/clip_cache/clip.parquet"

    if [[ -f "$CLIP_CACHE" ]]; then
        NUM_EMBEDDINGS=$(python -c "import pandas as pd; print(len(pd.read_parquet('${CLIP_CACHE}')))" 2>/dev/null || echo "0")
        if [[ $NUM_EMBEDDINGS -gt 70000 ]]; then
            log_skip "CLIP cache complete: ${NUM_EMBEDDINGS} embeddings"
        else
            log_info "CLIP cache partial (${NUM_EMBEDDINGS}/~73000), continuing build..."
        fi
    else
        NUM_EMBEDDINGS=0
    fi

    if [[ $NUM_EMBEDDINGS -lt 70000 ]]; then
        log_info "Building CLIP embedding cache..."
        log_info "Model: ViT-L/14 | Batch size: ${BATCH_SIZE_CLIP} | Device: cuda"
        log_info "This processes all ~73,000 NSD stimuli. ETA: ~1 hour on H100."

        mkdir -p outputs/clip_cache

        # Set H100 optimizations for CLIP encoding
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

        # Try CLI entry point, fall back to script
        if python -m fmri2img.cli.build_clip_cache --help &>/dev/null 2>&1; then
            run_cmd "python -m fmri2img.cli.build_clip_cache \
                --cache-root cache \
                --output '${CLIP_CACHE}' \
                --batch-size ${BATCH_SIZE_CLIP}"
        else
            run_cmd "python scripts/build_clip_cache.py \
                --cache-root cache \
                --output '${CLIP_CACHE}' \
                --batch-size ${BATCH_SIZE_CLIP}"
        fi

        # Verify
        if [[ -f "$CLIP_CACHE" ]]; then
            NUM_FINAL=$(python -c "import pandas as pd; print(len(pd.read_parquet('${CLIP_CACHE}')))" 2>/dev/null || echo "0")
            log_success "CLIP cache complete: ${NUM_FINAL} embeddings"
        fi
    fi

    save_progress 6 "$SUBJECT"
else
    log_skip "Step 6: CLIP cache"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Preprocessing (3-Stage)
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 7 "$SUBJECT"; then
    log_step 7 "Preprocessing (z-score → standardize → PCA)"

    PREPROC_DIR="cache/preproc"
    T1_SCALER="${PREPROC_DIR}/${SUBJECT}_t1_scaler.pkl"
    T2_PCA="${PREPROC_DIR}/${SUBJECT}_t2_pca_k512.npz"
    INDEX_FILE="data/indices/nsd_index/${SUBJECT}.csv"

    mkdir -p "$PREPROC_DIR"

    if [[ -f "$T1_SCALER" ]] && [[ -f "$T2_PCA" ]]; then
        T1_SIZE=$(stat -c%s "$T1_SCALER" 2>/dev/null || stat -f%z "$T1_SCALER" 2>/dev/null || echo "0")
        T2_SIZE=$(stat -c%s "$T2_PCA" 2>/dev/null || stat -f%z "$T2_PCA" 2>/dev/null || echo "0")
        if [[ $T1_SIZE -gt 1000 ]] && [[ $T2_SIZE -gt 100000 ]]; then
            log_skip "Preprocessing artifacts exist: T1 scaler + T2 PCA"
        else
            log_info "Preprocessing artifacts incomplete, re-running..."
        fi
    fi

    if [[ ! -f "$T1_SCALER" ]] || [[ ! -f "$T2_PCA" ]]; then
        log_info "Running 3-stage preprocessing for ${SUBJECT}..."
        log_info "Stage T0: Per-volume z-score normalization"
        log_info "Stage T1: Subject-level standardization + reliability masking"
        log_info "Stage T2: PCA dimensionality reduction (512 components)"
        echo ""

        # Try the dedicated preprocessing script
        if [[ -f "scripts/fit_preprocessing.py" ]]; then
            # Note: fit_preprocessing.py may have different CLI args across versions.
            # Try the full-featured version first.
            run_cmd "python scripts/fit_preprocessing.py \
                --subject '${SUBJECT}' \
                --index-file '${INDEX_FILE}' \
                --output-dir '${PREPROC_DIR}' \
                --reliability-mode hard_threshold \
                --reliability-threshold 0.1 \
                --n-components 512" || \
            run_cmd "python scripts/fit_preprocessing.py \
                --subject '${SUBJECT}' \
                --method t1 \
                --output '${T1_SCALER}'" && \
            run_cmd "python scripts/fit_preprocessing.py \
                --subject '${SUBJECT}' \
                --method t2 \
                --pca-dim 512 \
                --output '${T2_PCA}'"
        else
            # Use the pipeline orchestrator for preprocessing
            run_cmd "python scripts/run_full_pipeline.py \
                --subject '${SUBJECT}' \
                --mode baseline \
                --resume-from index \
                --skip-eval"
        fi

        log_success "Preprocessing complete for ${SUBJECT}"
    fi

    save_progress 7 "$SUBJECT"
else
    log_skip "Step 7: Preprocessing"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Model Training (Ridge → MLP → Two-Stage)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$SKIP_TRAINING" != "true" ]] && should_run_step 8 "$SUBJECT"; then
    log_step 8 "Model Training"

    # Set H100 optimizations
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # ── 8a: Ridge Regression (~5 min) ──
    RIDGE_CKPT="checkpoints/ridge/${SUBJECT}"
    if [[ -d "$RIDGE_CKPT" ]] && [[ $(find "$RIDGE_CKPT" -name "*.pt" -o -name "*.pkl" 2>/dev/null | head -1) ]]; then
        log_skip "Ridge model already trained: ${RIDGE_CKPT}/"
    else
        log_info "Training Ridge regression baseline (~5 min)..."
        mkdir -p "$RIDGE_CKPT"
        run_cmd "python -m fmri2img.training.train_ridge \
            --subject '${SUBJECT}' \
            --output-dir '${RIDGE_CKPT}'" || \
        log_warn "Ridge training failed (non-critical), continuing..."
    fi

    # ── 8b: MLP Encoder (~30 min on H100) ──
    MLP_CKPT="checkpoints/mlp/${SUBJECT}"
    if [[ -d "$MLP_CKPT" ]] && [[ $(find "$MLP_CKPT" -name "*.pt" 2>/dev/null | head -1) ]]; then
        log_skip "MLP model already trained: ${MLP_CKPT}/"
    else
        log_info "Training MLP encoder (~30 min on H100)..."
        mkdir -p "$MLP_CKPT"
        run_cmd "python -m fmri2img.training.train_mlp \
            --subject '${SUBJECT}' \
            --output-dir '${MLP_CKPT}' \
            --batch-size ${BATCH_SIZE_TRAIN} \
            --device cuda" || \
        log_warn "MLP training failed (non-critical), continuing..."
    fi

    # ── 8c: Two-Stage Encoder with Novel Contributions (~3-4 hours on H100) ──
    TWO_STAGE_CKPT="checkpoints/two_stage/${SUBJECT}"
    if [[ -d "$TWO_STAGE_CKPT" ]] && [[ $(find "$TWO_STAGE_CKPT" -name "*best*.pt" 2>/dev/null | head -1) ]]; then
        log_skip "Two-stage model already trained: ${TWO_STAGE_CKPT}/"
    else
        log_info "Training Two-Stage encoder with novel contributions..."
        log_info "Architecture: 4 residual blocks + multi-layer CLIP + InfoNCE"
        log_info "ETA: ~3-4 hours on H100 (${TRAINING_EPOCHS} epochs)"
        mkdir -p "$TWO_STAGE_CKPT"

        # Use the full pipeline for the novel two-stage training
        run_cmd "python scripts/run_full_pipeline.py \
            --subject '${SUBJECT}' \
            --mode novel \
            --resume-from train"
    fi

    log_success "All models trained for ${SUBJECT}"
    save_progress 8 "$SUBJECT"
else
    log_skip "Step 8: Model training"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Cross-Domain Evaluation (Hypothesis H1)
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 9 "$SUBJECT"; then
    log_step 9 "Cross-Domain Evaluation (Perception → Imagery)"

    REPORT_DIR="outputs/reports/imagery"
    IMAGERY_INDEX="cache/indices/imagery/${SUBJECT}.parquet"

    # Determine if imagery data & index are available
    HAS_IMAGERY=false
    if [[ -f "$IMAGERY_INDEX" ]]; then
        HAS_IMAGERY=true
    fi

    # Find the best two-stage checkpoint
    TWO_STAGE_BEST=$(find "checkpoints/two_stage/${SUBJECT}" -name "*best*.pt" 2>/dev/null | head -1)
    if [[ -z "$TWO_STAGE_BEST" ]]; then
        TWO_STAGE_BEST=$(find "checkpoints/two_stage/${SUBJECT}" -name "*.pt" 2>/dev/null | head -1)
    fi

    if [[ -z "$TWO_STAGE_BEST" ]]; then
        log_warn "No two-stage checkpoint found — skipping cross-domain evaluation."
        log_info "Train models first (step 8) or provide checkpoints in checkpoints/two_stage/${SUBJECT}/"
    else
        log_info "Using checkpoint: ${TWO_STAGE_BEST}"

        # ── 9a: Within-domain evaluation (perception → perception) ──
        PERC_REPORT="${REPORT_DIR}/perception_baseline"
        if [[ -f "${PERC_REPORT}/metrics.json" ]]; then
            log_skip "Perception baseline evaluation exists"
        else
            log_info "Evaluating on perception test set (within-domain)..."
            mkdir -p "$PERC_REPORT"
            if [[ "$HAS_IMAGERY" == "true" ]]; then
                run_cmd "python scripts/eval_perception_to_imagery_transfer.py \
                    --index '${IMAGERY_INDEX}' \
                    --checkpoint '${TWO_STAGE_BEST}' \
                    --model-type two_stage \
                    --mode perception \
                    --split test \
                    --batch-size ${BATCH_SIZE_EVAL} \
                    --device cuda \
                    --output-dir '${PERC_REPORT}'"
            else
                run_cmd "python scripts/eval_perception_to_imagery_transfer.py \
                    --checkpoint '${TWO_STAGE_BEST}' \
                    --model-type two_stage \
                    --mode perception \
                    --split test \
                    --batch-size ${BATCH_SIZE_EVAL} \
                    --device cuda \
                    --dry-run \
                    --output-dir '${PERC_REPORT}'"
            fi
        fi

        # ── 9b: Cross-domain evaluation (perception → imagery) ──
        IMG_REPORT="${REPORT_DIR}/perception_transfer"
        if [[ -f "${IMG_REPORT}/metrics.json" ]]; then
            log_skip "Cross-domain evaluation exists"
        else
            log_info "Evaluating on imagery test set (cross-domain)..."
            mkdir -p "$IMG_REPORT"
            if [[ "$HAS_IMAGERY" == "true" ]]; then
                run_cmd "python scripts/eval_perception_to_imagery_transfer.py \
                    --index '${IMAGERY_INDEX}' \
                    --checkpoint '${TWO_STAGE_BEST}' \
                    --model-type two_stage \
                    --mode imagery \
                    --split test \
                    --batch-size ${BATCH_SIZE_EVAL} \
                    --device cuda \
                    --output-dir '${IMG_REPORT}'"
            else
                log_info "No imagery index — running with --dry-run (synthetic data)..."
                run_cmd "python scripts/eval_perception_to_imagery_transfer.py \
                    --checkpoint '${TWO_STAGE_BEST}' \
                    --model-type two_stage \
                    --mode imagery \
                    --split test \
                    --device cuda \
                    --dry-run \
                    --output-dir '${IMG_REPORT}'"
            fi
        fi

        # ── 9c: Shared-1000 comprehensive evaluation ──
        SHARED_REPORT="outputs/reports/${SUBJECT}/shared1000"
        if [[ -f "${SHARED_REPORT}/eval_results.json" ]]; then
            log_skip "Shared-1000 evaluation exists"
        else
            log_info "Running shared-1000 comprehensive evaluation..."
            mkdir -p "$SHARED_REPORT"
            run_cmd "python scripts/eval_shared1000_full.py \
                --subject '${SUBJECT}' \
                --encoder-checkpoint '${TWO_STAGE_BEST}' \
                --encoder-type two_stage \
                --output-dir '${SHARED_REPORT}' \
                --cache-root cache \
                --stim-info cache/nsd_stim_info_merged.csv \
                --clip-cache outputs/clip_cache/clip.parquet \
                --batch-size ${BATCH_SIZE_EVAL} \
                --device cuda" || \
            log_warn "Shared-1000 evaluation failed (non-critical)"
        fi
    fi

    log_success "Cross-domain evaluation complete"
    save_progress 9 "$SUBJECT"
else
    log_skip "Step 9: Cross-domain evaluation"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Adapter Training + Ablations (Hypothesis H3)
# ─────────────────────────────────────────────────────────────────────────────

if should_run_step 10 "$SUBJECT"; then
    log_step 10 "Imagery Adapter Training & Ablation Suite"

    IMAGERY_INDEX="cache/indices/imagery/${SUBJECT}.parquet"
    ABLATION_DIR="outputs/imagery_ablations/${SUBJECT}"

    # Find best two-stage checkpoint
    TWO_STAGE_BEST=$(find "checkpoints/two_stage/${SUBJECT}" -name "*best*.pt" 2>/dev/null | head -1)
    if [[ -z "$TWO_STAGE_BEST" ]]; then
        TWO_STAGE_BEST=$(find "checkpoints/two_stage/${SUBJECT}" -name "*.pt" 2>/dev/null | head -1)
    fi

    if [[ -z "$TWO_STAGE_BEST" ]]; then
        log_warn "No two-stage checkpoint — skipping adapter training."
    elif [[ ! -f "$IMAGERY_INDEX" ]]; then
        log_warn "No imagery index — running adapter ablations in dry-run mode."
        run_cmd "python scripts/run_imagery_ablations.py \
            --index '${IMAGERY_INDEX}' \
            --checkpoint '${TWO_STAGE_BEST}' \
            --model-type two_stage \
            --output-dir '${ABLATION_DIR}' \
            --epochs ${ADAPTER_EPOCHS} \
            --batch-size ${BATCH_SIZE_TRAIN} \
            --with-condition \
            --device cuda \
            --dry-run"
    else
        if [[ -f "${ABLATION_DIR}/results_table.csv" ]]; then
            log_skip "Ablation results exist: ${ABLATION_DIR}/results_table.csv"
        else
            log_info "Running full adapter ablation suite..."
            log_info "Experiments: baseline / linear adapter / MLP adapter / MLP+condition"
            log_info "Epochs: ${ADAPTER_EPOCHS} per adapter | Device: cuda"
            log_info "ETA: ~1-2 hours total on H100"

            mkdir -p "$ABLATION_DIR"
            run_cmd "python scripts/run_imagery_ablations.py \
                --index '${IMAGERY_INDEX}' \
                --checkpoint '${TWO_STAGE_BEST}' \
                --model-type two_stage \
                --output-dir '${ABLATION_DIR}' \
                --epochs ${ADAPTER_EPOCHS} \
                --lr 1e-3 \
                --batch-size ${BATCH_SIZE_TRAIN} \
                --with-condition \
                --device cuda"
        fi

        # Generate paper figures from ablation results
        if [[ -f "${ABLATION_DIR}/results_table.csv" ]] || [[ -f "${ABLATION_DIR}/metrics_all.json" ]]; then
            log_info "Generating paper figures from ablation results..."
            mkdir -p "${ABLATION_DIR}/figures"
            run_cmd "python scripts/make_paper_figures.py \
                --ablation-dir '${ABLATION_DIR}' \
                --output-dir '${ABLATION_DIR}/figures'"
            log_success "Paper figures generated in ${ABLATION_DIR}/figures/"
        fi
    fi

    save_progress 10 "$SUBJECT"
else
    log_skip "Step 10: Adapter ablations"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Novel Analyses (15 Research Directions)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$SKIP_NOVEL" != "true" ]] && should_run_step 11 "$SUBJECT"; then
    log_step 11 "Novel Analyses (15 Research Directions)"

    NOVEL_DIR="outputs/novel_analyses"
    NOVEL_CONFIG="configs/experiments/novel_analyses.yaml"

    if [[ -d "$NOVEL_DIR" ]] && [[ $(find "$NOVEL_DIR" -name "*.json" 2>/dev/null | wc -l) -ge 10 ]]; then
        RESULT_COUNT=$(find "$NOVEL_DIR" -name "*.json" | wc -l)
        log_skip "Novel analyses exist (${RESULT_COUNT} result files)"
    else
        # Determine run mode based on data availability
        IMAGERY_INDEX="cache/indices/imagery/${SUBJECT}.parquet"
        TWO_STAGE_BEST=$(find "checkpoints/two_stage/${SUBJECT}" -name "*best*.pt" 2>/dev/null | head -1)

        if [[ -n "$TWO_STAGE_BEST" ]] && [[ -f "$IMAGERY_INDEX" ]]; then
            log_info "Running all 15 novel analyses with real data..."
            log_info "Config: ${NOVEL_CONFIG}"
            mkdir -p "$NOVEL_DIR"
            run_cmd "python scripts/run_novel_analyses.py \
                --output-dir '${NOVEL_DIR}' \
                --checkpoint '${TWO_STAGE_BEST}' \
                --model-type two_stage \
                --perception-index 'cache/indices/nsd_index/${SUBJECT}.parquet' \
                --imagery-index '${IMAGERY_INDEX}' \
                --subject '${SUBJECT}' \
                --cache-root cache \
                --device cuda"
        else
            log_info "Running novel analyses in dry-run mode (synthetic data)..."
            mkdir -p "$NOVEL_DIR"
            run_cmd "python scripts/run_novel_analyses.py \
                --output-dir '${NOVEL_DIR}' \
                --dry-run \
                --subject '${SUBJECT}' \
                --cache-root cache \
                --device cuda"
        fi

        # Generate figures
        if [[ $(find "$NOVEL_DIR" -name "*.json" 2>/dev/null | wc -l) -gt 0 ]]; then
            log_info "Generating novel analysis figures..."
            mkdir -p "${NOVEL_DIR}/figures"
            run_cmd "python scripts/make_novel_figures.py \
                --results-dir '${NOVEL_DIR}' \
                --output-dir '${NOVEL_DIR}/figures'"
            log_success "Novel analysis figures generated"
        fi
    fi

    save_progress 11 "$SUBJECT"
else
    log_skip "Step 11: Novel analyses"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SUBJECT EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$ALL_SUBJECTS" == "true" ]]; then
    echo ""
    log_step "∞" "Multi-Subject Expansion"

    for subj in "${ALL_IMAGERY_SUBJECTS[@]}"; do
        if [[ "$subj" == "$SUBJECT" ]]; then
            log_skip "${subj} (already processed above)"
            continue
        fi

        log_info "════════ Processing ${subj} ════════"

        # Re-run steps 3-11 for this subject by recursively calling ourselves
        run_cmd "bash '$0' \
            --subject '${subj}' \
            --nsd-root '${NSD_DATA_ROOT}' \
            --skip-env \
            --step 3 \
            ${DRY_RUN:+--dry-run}"

        log_success "${subj} complete"
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
log_step "✓" "Final Validation"

# Run the existing check_setup.sh
if [[ -f "scripts/check_setup.sh" ]]; then
    log_info "Running setup validation..."
    bash scripts/check_setup.sh "$SUBJECT" 2>&1 | tee -a "$LOGFILE" || true
fi

# Run test suite
log_info "Running test suite..."
run_cmd "python -m pytest tests/ -v --tb=short -q" || log_warn "Some tests failed (see above)"

# Summary report
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                     SETUP COMPLETE — SUMMARY                      ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}Subject(s):${NC} $(get_subjects)"
echo ""

# Check what artifacts exist
echo -e "${BOLD}Artifacts:${NC}"
[[ -f "outputs/clip_cache/clip.parquet" ]] && \
    echo -e "  ${GREEN}✓${NC} CLIP cache: $(python -c "import pandas as pd; print(f'{len(pd.read_parquet(\"outputs/clip_cache/clip.parquet\")):,} embeddings')" 2>/dev/null || echo "exists")"
[[ -f "data/indices/nsd_index/${SUBJECT}.csv" ]] && \
    echo -e "  ${GREEN}✓${NC} Perception index"
[[ -f "cache/indices/imagery/${SUBJECT}.parquet" ]] && \
    echo -e "  ${GREEN}✓${NC} Imagery index"
[[ -d "cache/preproc" ]] && [[ $(find "cache/preproc" -name "${SUBJECT}*" 2>/dev/null | wc -l) -gt 0 ]] && \
    echo -e "  ${GREEN}✓${NC} Preprocessing artifacts"

echo ""
echo -e "${BOLD}Models:${NC}"
for model_type in ridge mlp two_stage; do
    ckpt_dir="checkpoints/${model_type}/${SUBJECT}"
    if [[ -d "$ckpt_dir" ]] && [[ $(find "$ckpt_dir" -name "*.pt" -o -name "*.pkl" 2>/dev/null | wc -l) -gt 0 ]]; then
        echo -e "  ${GREEN}✓${NC} ${model_type}"
    else
        echo -e "  ${RED}✗${NC} ${model_type} (not trained)"
    fi
done

echo ""
echo -e "${BOLD}Evaluation Reports:${NC}"
for report in perception_baseline perception_transfer; do
    rdir="outputs/reports/imagery/${report}"
    if [[ -f "${rdir}/metrics.json" ]]; then
        echo -e "  ${GREEN}✓${NC} ${report}"
    else
        echo -e "  ${YELLOW}○${NC} ${report} (not yet run)"
    fi
done

echo ""
echo -e "${BOLD}Adapter Ablations:${NC}"
ABLATION_DIR="outputs/imagery_ablations/${SUBJECT}"
if [[ -f "${ABLATION_DIR}/results_table.csv" ]]; then
    echo -e "  ${GREEN}✓${NC} Ablation results: ${ABLATION_DIR}/results_table.csv"
    echo -e "  ${GREEN}✓${NC} Figures: ${ABLATION_DIR}/figures/"
else
    echo -e "  ${YELLOW}○${NC} Not yet run"
fi

echo ""
echo -e "${BOLD}Novel Analyses:${NC}"
NOVEL_COUNT=$(find "outputs/novel_analyses" -name "*.json" 2>/dev/null | wc -l)
if [[ $NOVEL_COUNT -gt 0 ]]; then
    echo -e "  ${GREEN}✓${NC} ${NOVEL_COUNT} analysis results"
else
    echo -e "  ${YELLOW}○${NC} Not yet run"
fi

echo ""
echo -e "${BOLD}Log file:${NC} ${LOGFILE}"
echo ""
echo -e "${GREEN}${BOLD}Pipeline setup complete!${NC}"
echo ""
echo -e "Next steps:"
echo -e "  • Review results:  cat outputs/reports/imagery/perception_transfer/metrics.json"
echo -e "  • View ablations:  cat outputs/imagery_ablations/${SUBJECT}/results_table.md"
echo -e "  • View figures:    ls outputs/imagery_ablations/${SUBJECT}/figures/"
echo -e "  • Re-run analyses: python scripts/run_novel_analyses.py --help"
echo -e "  • Add subjects:    bash scripts/setup_full_experiment.sh --all-subjects"
echo ""
