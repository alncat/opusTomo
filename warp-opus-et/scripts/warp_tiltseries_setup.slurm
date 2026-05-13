#!/bin/bash
#SBATCH --job-name=warp_tiltseries       # Job name
#SBATCH --partition=normal               # CPU partition (adjust as needed)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=128G                       # Memory per node
#SBATCH --time=12:00:00                  # Time limit (12 hours)
#SBATCH --output=logs/warp_tiltseries_%j.out  # Standard output log
#SBATCH --error=logs/warp_tiltseries_%j.err   # Standard error log

# =============================================================================
# WARP Tilt Series Setup Script
# Steps: Create Settings -> Import from MDOC
# =============================================================================

echo "============================================================"
echo "WARP Tilt Series Setup Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf defaults)
AUTO_INSTALL_MRCFILE=1                   # 1 = install mrcfile into WARP_ENV if missing

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo ""

echo "Setting up environment..."

# Load conda and activate environment
source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited

# Change to working directory
cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"

export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"

# Verify WARP tools
echo ""
echo "Verifying WARP installation..."
"${WARP_DIR}/WarpTools"

# =============================================================================
# Step 1: Create Tilt Series Settings
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Create Tilt Series Settings"
echo "============================================================"

# Create tomostar output directory
mkdir -p "$TOMOSTAR_DIR"

if [ -z "$TOMO_DIM_X" ] || [ -z "$TOMO_DIM_Y" ]; then
    if [ -z "$SAMPLE_FRAME" ]; then
        SAMPLE_FRAME=$(find "$FRAMESERIES_DIR/average" -maxdepth 1 -type f \( -name "*.mrc" -o -name "*.mrcs" \) 2>/dev/null | head -1)
    fi
    if [ -z "$SAMPLE_FRAME" ] || [ ! -f "$SAMPLE_FRAME" ]; then
        echo "ERROR: TOMO_DIM_X/Y are blank and no SAMPLE_FRAME MRC was found."
        echo "       Set SAMPLE_FRAME to one unbinned frame/average MRC, or set TOMO_DIM_X and TOMO_DIM_Y manually."
        exit 1
    fi

    echo "Reading unbinned X/Y dimensions from sample frame: $SAMPLE_FRAME"
    echo "Using mrcfile from WARP_ENV: $WARP_ENV"
    if ! python -c "import mrcfile" >/dev/null 2>&1; then
        if [ "$AUTO_INSTALL_MRCFILE" = "1" ]; then
            echo "mrcfile not found in WARP_ENV=$WARP_ENV; installing with python -m pip install mrcfile"
            if ! python -m pip install mrcfile; then
                echo "ERROR: Failed to install mrcfile in WARP_ENV=$WARP_ENV"
                echo "       Install it manually, or set TOMO_DIM_X/Y manually."
                exit 1
            fi
        else
            echo "ERROR: Python package mrcfile not found in WARP_ENV=$WARP_ENV"
            echo "       Set AUTO_INSTALL_MRCFILE=1, install it manually, or set TOMO_DIM_X/Y manually."
            exit 1
        fi
    fi
    if ! python -c "import mrcfile" >/dev/null 2>&1; then
        echo "ERROR: mrcfile is still unavailable in WARP_ENV=$WARP_ENV after installation attempt."
        echo "       Set TOMO_DIM_X/Y manually, or fix the WARP_ENV Python/pip installation."
        exit 1
    fi
    frame_dims=$(python - "$SAMPLE_FRAME" <<'PY'
import sys
import mrcfile
with mrcfile.open(sys.argv[1], permissive=True) as m:
    nx = int(m.header.nx)
    ny = int(m.header.ny)
    print(f"{nx} {ny}")
PY
    )
    if [ $? -ne 0 ] || [ -z "$frame_dims" ]; then
        echo "ERROR: Failed to read dimensions from SAMPLE_FRAME=$SAMPLE_FRAME"
        exit 1
    fi
    [ -z "$TOMO_DIM_X" ] && TOMO_DIM_X=$(echo "$frame_dims" | awk '{print $1}')
    [ -z "$TOMO_DIM_Y" ] && TOMO_DIM_Y=$(echo "$frame_dims" | awk '{print $2}')
fi

if [ -z "$TOMO_DIM_X" ] || [ -z "$TOMO_DIM_Y" ] || [ -z "$TOMO_DIM_Z" ]; then
    echo "ERROR: TOMO_DIM_X, TOMO_DIM_Y, and TOMO_DIM_Z must be set before create_settings."
    exit 1
fi

echo ""
echo "Parameters:"
echo "  - Input tomostar: $TOMOSTAR_DIR"
echo "  - Output directory: $PROCESSING_DIR"
echo "  - Pixel size: $ANGPIX A"
echo "  - Exposure: $EXPOSURE e-/A^2"
echo "  - Tomogram dimensions: ${TOMO_DIM_X}x${TOMO_DIM_Y}x${TOMO_DIM_Z} (unbinned, approximate)"

"${WARP_DIR}/WarpTools" create_settings \
--folder_data "$TOMOSTAR_DIR" \
--folder_processing "$PROCESSING_DIR" \
--extension "*.tomostar" \
--angpix "$ANGPIX" \
--exposure "$EXPOSURE" \
--tomo_dimensions "${TOMO_DIM_X}x${TOMO_DIM_Y}x${TOMO_DIM_Z}" \
--output "${PROCESSING_DIR}.settings"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create tilt series settings"
    exit 1
fi

echo "Settings file created: ${PROCESSING_DIR}.settings"

# =============================================================================
# Step 2: Import Tilt Series from MDOC
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Import Tilt Series from MDOC"
echo "============================================================"

# Check if MDOC directory exists
if [ ! -d "$MDOC_DIR" ]; then
    echo "ERROR: MDOC directory not found: $MDOC_DIR"
    echo "Please ensure MDOC files are in the specified directory"
    exit 1
fi

# Count MDOC files
NUM_MDOC=$(ls "$MDOC_DIR"/*.mdoc 2>/dev/null | wc -l)
echo "Found $NUM_MDOC MDOC files in $MDOC_DIR"

if [ $NUM_MDOC -eq 0 ]; then
    echo "ERROR: No MDOC files found in $MDOC_DIR"
    exit 1
fi

echo ""
echo "Import parameters:"
echo "  - MDOC directory: $MDOC_DIR"
echo "  - Frame series: $FRAMESERIES_DIR"
echo "  - Tilt exposure: $EXPOSURE e-/A^2"
echo "  - Min intensity: $MIN_INTENSITY"
echo "  - Output: $TOMOSTAR_DIR"

"${WARP_DIR}/WarpTools" ts_import \
--mdocs "$MDOC_DIR" \
--frameseries "$FRAMESERIES_DIR" \
--tilt_exposure "$EXPOSURE" \
--min_intensity "$MIN_INTENSITY" \
--output "$TOMOSTAR_DIR"

IMPORT_EXIT_CODE=$?

if [ $IMPORT_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Import may have encountered errors (Exit code: $IMPORT_EXIT_CODE)"
else
    echo ""
    echo "Import completed successfully"
fi

# Count generated tomostar files
NUM_TOMOSTAR=$(ls "$TOMOSTAR_DIR"/*.tomostar 2>/dev/null | wc -l)
echo "Generated $NUM_TOMOSTAR tomostar files"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Tilt Series Setup Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings:"
echo "  - Pixel size: $ANGPIX A"
echo "  - Exposure: $EXPOSURE e-/A^2"
echo "  - Tomogram dimensions: ${TOMO_DIM_X}x${TOMO_DIM_Y}x${TOMO_DIM_Z}"
echo ""
echo "Output locations:"
echo "  - Settings: ${PROCESSING_DIR}.settings"
echo "  - Tomostar files: $TOMOSTAR_DIR/"
echo "  - Tilt series metadata: $PROCESSING_DIR/"
echo ""
echo "Next steps:"
echo "  1. Export tilt stacks for alignment:"
echo "     sbatch scripts/warp_export_stacks.slurm"
echo ""
echo "  2. Run AreTomo2 alignment + angle negation:"
echo "     sbatch scripts/warp_aretomo_align_negate.slurm"
echo ""
echo "  3. Update WARP settings with dimensions from AreTomo output:"
echo "     sbatch scripts/warp_update_tomo_dims.slurm"
echo "============================================================"
