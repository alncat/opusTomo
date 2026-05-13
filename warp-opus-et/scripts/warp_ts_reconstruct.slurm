#!/bin/bash
#SBATCH --job-name=warp_recon            # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=256G                       # Memory per node (256GB recommended)
#SBATCH --gres=gpu:4                     # Number of GPUs
#SBATCH --time=24:00:00                  # Time limit (24 hours)
#SBATCH --output=logs/warp_recon_%j.out       # Standard output log
#SBATCH --error=logs/warp_recon_%j.err        # Standard error log

# =============================================================================
# WARP Tomogram Reconstruction Script
# Reconstructs tomograms from aligned and CTF-estimated tilt series
# =============================================================================

echo "============================================================"
echo "WARP Tomogram Reconstruction Started"
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

# Output format
export WARP_FORCE_MRC_FLOAT32=1          # Force 32-bit float output

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

# Display configuration
echo ""
echo "============================================================"
echo "Reconstruction Configuration"
echo "============================================================"
echo "Settings file: $TILTSERIES_SETTINGS"
echo ""
echo "Reconstruction parameters:"
echo "  - Original pixel size: $ANGPIX A"
echo "  - Binning factor: ${BINNING_FACTOR}x"
echo "  - Pixel size: $ALIGN_ANGPIX A"
echo "  - Output format: 32-bit float MRC"
echo "  - Processes per GPU: $PERDEVICE"

# =============================================================================
# Run Reconstruction
# =============================================================================
echo ""
echo "============================================================"
echo "Running Tomogram Reconstruction"
echo "============================================================"

echo ""
echo "Command:"
echo "  export WARP_FORCE_MRC_FLOAT32=1"
echo "  WarpTools ts_reconstruct \\"
echo "      --settings $TILTSERIES_SETTINGS \\"
echo "      --angpix $ALIGN_ANGPIX \\"
echo "      --perdevice $PERDEVICE"

"${WARP_DIR}/WarpTools" ts_reconstruct \
--settings "$TILTSERIES_SETTINGS" \
--angpix "$ALIGN_ANGPIX" \
--perdevice "$PERDEVICE"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Reconstruction failed (Exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - CTF estimation not completed"
    echo "  - Alignments not imported"
    echo "  - Out of memory (try reducing --perdevice or using 8x binning)"
    exit 1
fi

# =============================================================================
# Verify Reconstruction
# =============================================================================
echo ""
echo "============================================================"
echo "Verifying Reconstruction"
echo "============================================================"

# Count reconstructed tomograms
RECON_COUNT=$(find "warp_tiltseries/reconstruction" -name "*.mrc" 2>/dev/null | wc -l)
echo "Reconstructed tomograms: $RECON_COUNT"

# List output files
echo ""
echo "Output files:"
find "warp_tiltseries/reconstruction" -name "*.mrc" 2>/dev/null | head -10

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Tomogram Reconstruction Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Settings:"
echo "  - Settings: $TILTSERIES_SETTINGS"
echo "  - Pixel size: $ALIGN_ANGPIX A"
echo "  - Output format: 32-bit float MRC"
echo ""
echo "Output:"
echo "  - Tomograms: warp_tiltseries/reconstruction/*.mrc"
echo ""
echo "Next steps:"
echo "  1. Visualize tomograms in IMOD, ChimeraX, or other software"
echo ""
echo "  2. Run template matching / particle picking:"
echo "     sbatch scripts/gen_template_from_mrc.slurm"
echo "     sbatch scripts/gen_sphere_mask.slurm"
echo "     sbatch scripts/gen_tm_jobs_aretomo.slurm"
echo "     sbatch scripts/run_tm_sequential.slurm"
echo "     sbatch scripts/extract_tm_candidates_parallel.slurm"
echo ""
echo "  3. Convert PyTOM XML picks to WARP STAR files:"
echo "     sbatch scripts/convert_to_star.slurm"
echo "     sbatch scripts/convert_pytom_to_warp.slurm"
echo ""
echo "  4. Export subtomograms:"
echo "     sbatch scripts/warp_export_particles.slurm"
echo "============================================================"
