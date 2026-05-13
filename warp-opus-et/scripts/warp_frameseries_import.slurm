#!/bin/bash
#SBATCH --job-name=warp_frameseries      # Job name
#SBATCH --partition=normal               # GPU partition (adjust as needed)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=128G                       # Memory per node (128GB sufficient for CTF)
#SBATCH --gres=gpu:4                     # Number of GPUs (adjust as needed)
#SBATCH --time=12:00:00                  # Time limit (12 hours)
#SBATCH --output=logs/warp_frameseries_%j.out # Standard output log
#SBATCH --error=logs/warp_frameseries_%j.err  # Standard error log

# =============================================================================
# WARP Frame Series Import and CTF Estimation Script
# Steps: Create Settings -> Link Images -> CTF Estimation
# =============================================================================

echo "============================================================"
echo "WARP Frame Series Processing Started"
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
GAIN_FLIP_X=0                            # set 1 only if known
GAIN_FLIP_Y=0                            # set 1 only if known
GAIN_TRANSPOSE=0                         # set 1 only if known

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo ""

echo "Setting up environment..."

# Load conda and activate environment
source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited

export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"

# Change to working directory
cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"

# Verify GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi

# Verify WARP tools
echo ""
echo "Verifying WARP installation..."
"${WARP_DIR}/WarpTools"

# =============================================================================
# Step 1: Create Frame Series Settings
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Create Frame Series Settings"
echo "============================================================"

echo ""
echo "Parameters:"
echo "  - Input data: $FRAME_DIR"
echo "  - Output directory: $FRAMESERIES_DIR"
echo "  - File extension: $FILE_EXTENSION"
echo "  - Frame mode: $FRAME_MODE"
echo "  - Pixel size: $ANGPIX A"
echo "  - Exposure: $EXPOSURE e-/A^2"

CREATE_SETTINGS_ARGS=(
create_settings
--folder_data "$FRAME_DIR"
--folder_processing "$FRAMESERIES_DIR"
--extension "$FILE_EXTENSION"
--angpix "$ANGPIX"
--exposure "$EXPOSURE"
--output "${FRAMESERIES_DIR}.settings"
)

case "$FRAME_MODE" in
single)
;;
movie|eer)
CREATE_SETTINGS_ARGS+=(--bin 1)
if [ -n "$GAIN_FILE" ]; then
    CREATE_SETTINGS_ARGS+=(--gain_path "$GAIN_FILE")
fi
if [ -n "$DEFECTS_FILE" ]; then
    CREATE_SETTINGS_ARGS+=(--defects_path "$DEFECTS_FILE")
fi
if [ "$GAIN_FLIP_X" = "1" ]; then
    CREATE_SETTINGS_ARGS+=(--gain_flip_x)
fi
if [ "$GAIN_FLIP_Y" = "1" ]; then
    CREATE_SETTINGS_ARGS+=(--gain_flip_y)
fi
if [ "$GAIN_TRANSPOSE" = "1" ]; then
    CREATE_SETTINGS_ARGS+=(--gain_transpose)
fi
if [ "$FRAME_MODE" = "eer" ]; then
    if [ -z "$EER_NGROUPS" ]; then
        echo "ERROR: EER_NGROUPS must be set when FRAME_MODE=eer"
        exit 1
    fi
    CREATE_SETTINGS_ARGS+=(--eer_ngroups "$EER_NGROUPS")
fi
;;
*)
echo "ERROR: FRAME_MODE must be one of: single, movie, eer"
exit 1
;;
esac

"${WARP_DIR}/WarpTools" "${CREATE_SETTINGS_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create frame series settings"
    exit 1
fi

echo "Settings file created: ${FRAMESERIES_DIR}.settings"

# =============================================================================
# Step 2: Process Frame Series
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Process Frame Series"
echo "============================================================"

# Count files to link
NUM_FILES=$(ls $FRAME_DIR/$FILE_EXTENSION 2>/dev/null | wc -l)
echo "Found $NUM_FILES input files"

if [ $NUM_FILES -eq 0 ]; then
    echo "ERROR: No files found in $FRAME_DIR matching $FILE_EXTENSION"
    exit 1
fi

PROCESSED_FILES=$NUM_FILES

# =============================================================================
# Step 3: CTF Estimation on Frame Series
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: CTF Estimation on Frame Series"
echo "============================================================"

echo ""
echo "CTF parameters:"
echo "  - Grid: $CTF_GRID"
echo "  - Max resolution: $CTF_RANGE_MAX A"
echo "  - Max defocus: $CTF_DEFOCUS_MAX um"

if [ "$FRAME_MODE" = "single" ]; then
    # Single-frame MRC: no motion correction. WARP expects averaged images in average/.
    mkdir -p "${FRAMESERIES_DIR}/average"

    echo "Creating symbolic links for single-frame images..."
    for f in "$FRAME_DIR"/$FILE_EXTENSION; do
        if [ -f "$f" ]; then
            ln -sf "$(realpath "$f")" "${FRAMESERIES_DIR}/average/$(basename "$f")"
        fi
    done

    PROCESSED_FILES=$(ls "${FRAMESERIES_DIR}/average/" 2>/dev/null | wc -l)
    echo "Successfully linked $PROCESSED_FILES files to ${FRAMESERIES_DIR}/average/"

    "${WARP_DIR}/WarpTools" fs_ctf \
    --settings "${FRAMESERIES_DIR}.settings" \
    --grid "$CTF_GRID" \
    --range_max "$CTF_RANGE_MAX" \
    --defocus_max "$CTF_DEFOCUS_MAX" \
    --perdevice 1
else
    # Fractionated MRC/TIFF/EER: motion correction and CTF estimation.
    export WARP_FORCE_MRC_FLOAT32=1

    "${WARP_DIR}/WarpTools" fs_motion_and_ctf \
    --settings "${FRAMESERIES_DIR}.settings" \
    --m_grid "$MOTION_GRID" \
    --c_grid "$CTF_GRID" \
    --c_range_max "$CTF_RANGE_MAX" \
    --c_defocus_max "$CTF_DEFOCUS_MAX" \
    --c_use_sum \
    --out_averages \
    --out_average_halves \
    --perdevice 1
fi

CTF_EXIT_CODE=$?

if [ $CTF_EXIT_CODE -ne 0 ]; then
    echo "WARNING: CTF estimation may have encountered errors (Exit code: $CTF_EXIT_CODE)"
else
    echo ""
    echo "CTF estimation completed successfully"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Frame Series Processing Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings:"
echo "  - Pixel size: $ANGPIX A"
echo "  - Exposure: $EXPOSURE e-/A^2"
echo "  - Frame mode: $FRAME_MODE"
echo "  - Files processed: $PROCESSED_FILES"
echo ""
echo "Output locations:"
echo "  - Settings: ${FRAMESERIES_DIR}.settings"
echo "  - Averages: ${FRAMESERIES_DIR}/average/"
echo "  - CTF XML files: ${FRAMESERIES_DIR}/*.xml"
echo "  - Power spectra: ${FRAMESERIES_DIR}/powerspectrum/"
echo ""
echo "Next steps:"
echo "  1. Create tilt series settings and import tilt series from MDOC:"
echo "     sbatch scripts/warp_tiltseries_setup.slurm"
echo "============================================================"
