#!/bin/bash
#SBATCH --job-name=warp_ts_ctf           # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=256G                       # Memory per node (256GB recommended)
#SBATCH --gres=gpu:4                     # Number of GPUs
#SBATCH --time=12:00:00                  # Time limit (12 hours)
#SBATCH --output=logs/warp_ts_ctf_%j.out      # Standard output log
#SBATCH --error=logs/warp_ts_ctf_%j.err       # Standard error log

# =============================================================================
# WARP Tilt Series Defocus Handedness + CTF Estimation Script
# Steps: Check Defocus Handedness → Set Handedness → CTF Estimation
# =============================================================================

echo "============================================================"
echo "WARP Tilt Series CTF Estimation Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf defaults)
PERDEVICE=1                              # Processes per GPU (reduce if OOM)

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo ""

echo "Setting up environment..."

source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited

export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"

# Verify GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi

# Verify WARP
echo ""
echo "Verifying WARP installation..."
"${WARP_DIR}/WarpTools"

# Check settings file exists
if [ ! -f "$TILTSERIES_SETTINGS" ]; then
    echo "ERROR: Settings file not found: $TILTSERIES_SETTINGS"
    exit 1
fi

# =============================================================================
# Step 1: Check Defocus Handedness
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Check Defocus Handedness"
echo "============================================================"

CHECK_OUTPUT=$("${WARP_DIR}/WarpTools" ts_defocus_hand \
--settings "$TILTSERIES_SETTINGS" \
--check 2>&1)

echo "$CHECK_OUTPUT"

# Determine if we need to flip based on correlation sign
if echo "$CHECK_OUTPUT" | grep -q "average correlation is negative"; then
    echo ""
    echo "Negative correlation detected -> Setting handedness to FLIP"
    FLIP_NEEDED=1
elif echo "$CHECK_OUTPUT" | grep -q "average correlation is positive"; then
    echo ""
    echo "Positive correlation detected -> Setting handedness to NO FLIP"
    FLIP_NEEDED=0
else
    echo ""
    echo "WARNING: Could not determine correlation sign from output"
    echo "Assuming FLIP is needed (common for cryo-ET)"
    FLIP_NEEDED=1
fi

# =============================================================================
# Step 2: Set Correct Defocus Handedness
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Set Defocus Handedness"
echo "============================================================"

if [ $FLIP_NEEDED -eq 1 ]; then
    echo "Setting defocus handedness to: FLIP"
    "${WARP_DIR}/WarpTools" ts_defocus_hand \
    --settings "$TILTSERIES_SETTINGS" \
    --set_flip
else
    echo "Setting defocus handedness to: NO FLIP"
    "${WARP_DIR}/WarpTools" ts_defocus_hand \
    --settings "$TILTSERIES_SETTINGS" \
    --set_noflip
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to set defocus handedness"
    exit 1
fi

# =============================================================================
# Step 3: CTF Estimation on Tilt Series
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: CTF Estimation on Tilt Series"
echo "============================================================"

echo ""
echo "CTF parameters:"
echo "  - Window size: $CTF_WINDOW"
echo "  - Max resolution: $CTF_RANGE_MAX A"
echo "  - Max defocus: $CTF_DEFOCUS_MAX um"
echo "  - Processes per GPU: $PERDEVICE"

"${WARP_DIR}/WarpTools" ts_ctf \
--settings "$TILTSERIES_SETTINGS" \
--window "$CTF_WINDOW" \
--range_high "$CTF_RANGE_MAX" \
--defocus_max "$CTF_DEFOCUS_MAX" \
--perdevice "$PERDEVICE"

CTF_EXIT_CODE=$?

if [ $CTF_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: CTF estimation failed (Exit code: $CTF_EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - Frame series CTF not completed first"
    echo "  - Images not linked in warp_frameseries/average/"
    echo "  - Out of memory (try reducing --perdevice)"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Tilt Series CTF Estimation Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings applied:"
if [ $FLIP_NEEDED -eq 1 ]; then
    echo "  - Defocus handedness: FLIP"
else
    echo "  - Defocus handedness: NO FLIP"
fi
echo ""
echo "CTF parameters:"
echo "  - Window: $CTF_WINDOW"
echo "  - Max resolution: $CTF_RANGE_MAX A"
echo "  - Max defocus: $CTF_DEFOCUS_MAX um"
echo ""
echo "Output locations:"
echo "  - CTF XML files: warp_tiltseries/*.xml"
echo "  - Power spectra: warp_tiltseries/powerspectrum/"
echo ""
echo "Next step:"
echo "  Reconstruct tomograms:"
echo "  sbatch scripts/warp_ts_reconstruct.slurm"
echo "============================================================"
