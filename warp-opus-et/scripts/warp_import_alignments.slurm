#!/bin/bash
#SBATCH --job-name=warp_import_align     # Job name
#SBATCH --partition=normal               # CPU partition (no GPU needed)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=02:00:00                  # Time limit (2 hours)
#SBATCH --output=logs/warp_import_%j.out      # Standard output log
#SBATCH --error=logs/warp_import_%j.err       # Standard error log

# =============================================================================
# WARP Import Alignments Script
# Imports AreTomo2 alignment parameters (.xf, .tlt) into WARP
# =============================================================================

echo "============================================================"
echo "WARP Import Alignments Started"
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
"${WARP_DIR}/WarpTools" --help > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: WARP tools not found or not working"
    exit 1
fi
echo "WARP: OK"

# Check settings file exists
if [ ! -f "$TILTSERIES_SETTINGS" ]; then
    echo "ERROR: Settings file not found: $TILTSERIES_SETTINGS"
    exit 1
fi

if [ -z "$ANGPIX" ]; then
    ANGPIX=$(sed -n 's/.*Name="PixelSize".*Value="\([0-9.][0-9.]*\)".*/\1/p' "$TILTSERIES_SETTINGS" | head -1)
fi
if [[ "$ANGPIX" == \<* ]] || [ -z "$ANGPIX" ]; then
    echo "ERROR: Could not determine ANGPIX from $TILTSERIES_SETTINGS. Expected a line like:"
    echo '       <Param Name="PixelSize" Value="1.96" />'
    echo "       Set ANGPIX manually if the settings format differs."
    exit 1
fi
if [[ "$BINNING_FACTOR" == \<* ]] || [ -z "$BINNING_FACTOR" ]; then
    echo "ERROR: Set BINNING_FACTOR to the same binning used by warp_export_stacks.slurm / AreTomo."
    exit 1
fi
ALIGN_ANGPIX=$(awk "BEGIN {printf \"%.6f\", $ANGPIX * $BINNING_FACTOR}")

# Check alignments directory exists
if [ ! -d "$TILTSTACK_DIR" ]; then
    echo "ERROR: Alignments directory not found: $TILTSTACK_DIR"
    exit 1
fi

# Count alignment files
NUM_TLT=$(find "$TILTSTACK_DIR" -name "*.tlt" 2>/dev/null | wc -l)
NUM_XF=$(find "$TILTSTACK_DIR" -name "*.xf" 2>/dev/null | wc -l)

echo ""
echo "Found alignment files:"
echo "  - .tlt files: $NUM_TLT"
echo "  - .xf files: $NUM_XF"

if [ $NUM_TLT -eq 0 ] || [ $NUM_XF -eq 0 ]; then
    echo "ERROR: No alignment files found in $TILTSTACK_DIR"
    echo "Make sure AreTomo2 alignment completed successfully"
    exit 1
fi

# Display configuration
echo ""
echo "============================================================"
echo "Import Configuration"
echo "============================================================"
echo "Settings file: $TILTSERIES_SETTINGS"
echo "Alignments directory: $TILTSTACK_DIR"
echo ""
echo "Pixel sizes:"
echo "  - Original (unbinned): $ANGPIX A"
echo "  - Binned (for alignment): $ALIGN_ANGPIX A"
echo ""
echo "IMPORTANT: Using binned pixel size ($ALIGN_ANGPIX A)"
echo "           This must match the pixel size used in AreTomo2 alignment"

# =============================================================================
# Import Alignments
# =============================================================================
echo ""
echo "============================================================"
echo "Importing Alignments"
echo "============================================================"

echo ""
echo "Command:"
echo "  WarpTools ts_import_alignments \\"
echo "      --settings $TILTSERIES_SETTINGS \\"
echo "      --alignments $TILTSTACK_DIR \\"
echo "      --alignment_angpix $ALIGN_ANGPIX"

"${WARP_DIR}/WarpTools" ts_import_alignments \
--settings "$TILTSERIES_SETTINGS" \
--alignments "$TILTSTACK_DIR" \
--alignment_angpix "$ALIGN_ANGPIX"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Import alignments failed (exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - .xf or .tlt files missing"
    echo "  - Wrong alignment_angpix (should match AreTomo pixel size)"
    echo "  - Settings file doesn't match alignment files"
    exit 1
fi

# =============================================================================
# Verify Import
# =============================================================================
echo ""
echo "============================================================"
echo "Verifying Import"
echo "============================================================"

# Count how many tilt series now have alignment data
ALIGNED_COUNT=$(find "warp_tiltseries" -name "*.xml" -exec grep -l "Alignment" {} \; 2>/dev/null | wc -l)
echo "Tilt series with alignment: $ALIGNED_COUNT"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Import Alignments Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings:"
echo "  - Settings: $TILTSERIES_SETTINGS"
echo "  - Alignments: $TILTSTACK_DIR"
echo "  - Alignment pixel size: $ALIGN_ANGPIX A (matches AreTomo)"
echo ""
echo "Output:"
echo "  - Alignment data imported into: warp_tiltseries/*.xml"
echo ""
echo "Next steps:"
echo "  1. Run defocus handedness check + tilt-series CTF estimation:"
echo "     sbatch scripts/warp_ts_ctf.slurm"
echo ""
echo "  2. Reconstruct tomograms:"
echo "     sbatch scripts/warp_ts_reconstruct.slurm"
echo "============================================================"
