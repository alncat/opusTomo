#!/bin/bash
#SBATCH --job-name=warp_export           # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --time=06:00:00                  # Time limit (6 hours)
#SBATCH --output=logs/warp_export_%j.out      # Standard output log
#SBATCH --error=logs/warp_export_%j.err       # Standard error log

# =============================================================================
# WARP Tilt Stack Export Script
# Exports tilt stacks from WARP for external alignment (AreTomo2, IMOD, etc.)
# =============================================================================

echo "============================================================"
echo "WARP Tilt Stack Export Started"
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

# Verify WARP
echo ""
echo "Verifying WARP installation..."
"${WARP_DIR}/WarpTools"

# Check settings file exists
if [ ! -f "$TILTSERIES_SETTINGS" ]; then
    echo "ERROR: Settings file not found: $TILTSERIES_SETTINGS"
    exit 1
fi

# Extract original pixel size from settings, e.g.
# <Param Name="PixelSize" Value="1.96" />
ANGPIX=$(sed -n 's/.*Name="PixelSize".*Value="\([0-9.][0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
if [ -z "$ANGPIX" ]; then
    echo "ERROR: Could not extract PixelSize from $TILTSERIES_SETTINGS. Expected a line like:"
    echo '       <Param Name="PixelSize" Value="1.96" />'
    exit 1
fi

# Calculate binned pixel size at full precision — WARP derives the binning
# ratio as requested_angpix / settings_angpix, so rounding here would leave
# the ts_stack output at a slightly off binning factor.
ALIGN_ANGPIX=$(awk "BEGIN {printf \"%.6f\", $ANGPIX * $BINNING_FACTOR}")

# Display configuration
echo ""
echo "============================================================"
echo "Export Configuration"
echo "============================================================"
echo "Original pixel size: $ANGPIX A"
echo "Binning factor: ${BINNING_FACTOR}x"
echo "Target pixel size: $ALIGN_ANGPIX A"

# =============================================================================
# Export Tilt Stacks
# =============================================================================
echo ""
echo "============================================================"
echo "Exporting Tilt Stacks"
echo "============================================================"

echo ""
echo "Command: WarpTools ts_stack --settings $TILTSERIES_SETTINGS --angpix $ALIGN_ANGPIX"

"${WARP_DIR}/WarpTools" ts_stack \
--settings "$TILTSERIES_SETTINGS" \
--angpix "$ALIGN_ANGPIX"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Export failed (exit code: $EXIT_CODE)"
    exit 1
fi

# Count exported stacks
NUM_STACKS=$(find "$TILTSTACK_DIR" -name "*.st" 2>/dev/null | wc -l)
echo ""
echo "Exported $NUM_STACKS tilt stacks"

if [ $NUM_STACKS -eq 0 ]; then
    echo "WARNING: No tilt stacks found in $TILTSTACK_DIR"
    exit 1
fi

# List exported files
echo ""
echo "Exported files:"
find "$TILTSTACK_DIR" -name "*.st" -exec basename {} \; 2>/dev/null | head -20

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Export Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings:"
echo "  - Settings file: $TILTSERIES_SETTINGS"
echo "  - Original pixel size: $ANGPIX A"
echo "  - Binning factor: ${BINNING_FACTOR}x"
echo "  - Target pixel size: $ALIGN_ANGPIX A"
echo ""
echo "Output locations:"
echo "  - Tilt stacks: $TILTSTACK_DIR/*/*.st"
echo "  - Raw tilt angles: $TILTSTACK_DIR/*/*.rawtlt"
echo ""
echo "Next steps:"
echo "  1. Run AreTomo2 alignment + angle negation:"
echo "     sbatch scripts/warp_aretomo_align_negate.slurm"
echo ""
echo "  2. Update WARP settings with dimensions from AreTomo output:"
echo "     sbatch scripts/warp_update_tomo_dims.slurm"
echo ""
echo "  3. Import alignments into WARP:"
echo "     sbatch scripts/warp_import_alignments.slurm"
echo "============================================================"
