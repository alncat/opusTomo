#!/bin/bash
#SBATCH --job-name=warp_export_parts     # Job name
#SBATCH --partition=normal               # GPU partition
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --mem=128G                       # Memory
#SBATCH --gres=gpu:4                     # 4 GPUs
#SBATCH --time=12:00:00                  # Time limit (12 hours)
#SBATCH --output=logs/warp_export_%j.out      # Standard output
#SBATCH --error=logs/warp_export_%j.err       # Standard error

# =============================================================================
# WARP Export Particles for OPUS-ET
# Exports subtomograms from aligned tilt series using particle positions
# =============================================================================

echo "============================================================"
echo "WARP Export Particles for OPUS-ET"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
INPUT_PATTERN="*_warp.star"              # Pattern to match STAR files
OUTPUT_PREFIX="particles"                # Prefix for output files

# -----------------------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited

# WARP library path
export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

# Verify WARP
echo "Checking WARP installation..."
if [ ! -x "${WARP_DIR}/WarpTools" ]; then
    echo "ERROR: WarpTools not found at ${WARP_DIR}/WarpTools"
    exit 1
fi
echo "WarpTools: OK"
echo ""

# Verify settings file
if [ ! -f "$TILTSERIES_SETTINGS" ]; then
    echo "ERROR: Settings file not found: $TILTSERIES_SETTINGS"
    exit 1
fi

for pixel_value in "$OUTPUT_ANGPIX" "$COORDS_ANGPIX"; do
    if [[ "$pixel_value" == \<* ]] || [ -z "$pixel_value" ] || ! [[ "$pixel_value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: OUTPUT_ANGPIX and COORDS_ANGPIX must be set explicitly to numeric pixel sizes."
        echo "       OUTPUT_ANGPIX is the exported subtomogram pixel size, often binned for OPUS-ET."
        echo "       COORDS_ANGPIX is the coordinate pixel size in the input STAR, usually unbinned ANGPIX."
        exit 1
    fi
done

# Check input directory
if [ ! -d "$TM_WARP_DIR" ]; then
    echo "ERROR: Input directory not found: $TM_WARP_DIR"
    echo "Please run convert_pytom_to_warp.slurm first"
    exit 1
fi

# Count input files
NUM_INPUT=$(find "$TM_WARP_DIR" -name "$INPUT_PATTERN" -type f 2>/dev/null | wc -l)
if [ $NUM_INPUT -eq 0 ]; then
    echo "ERROR: No STAR files matching '$INPUT_PATTERN' in $TM_WARP_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$EXPORT_STAR")"

echo "Configuration:"
echo "  Settings: $TILTSERIES_SETTINGS"
echo "  Input directory: $TM_WARP_DIR"
echo "  Input pattern: $INPUT_PATTERN"
echo "  Input files: $NUM_INPUT"
echo "  Output STAR: $EXPORT_STAR"
echo "  Subtomogram box size: $SUBTOMO_BOX_SIZE pixels"
echo "  Diameter: $DIAMETER A"
echo "  Output pixel size: $OUTPUT_ANGPIX A"
echo "  Coordinate pixel size: $COORDS_ANGPIX A"
echo ""

# -----------------------------------------------------------------------------
# Run Export Particles
# -----------------------------------------------------------------------------

echo "============================================================"
echo "Running WARP ts_export_particles"
echo "============================================================"
echo ""

echo "Command:"
echo "  ${WARP_DIR}/WarpTools ts_export_particles \\"
echo "      --settings $TILTSERIES_SETTINGS \\"
echo "      --input_directory $TM_WARP_DIR \\"
echo "      --input_pattern \"$INPUT_PATTERN\" \\"
echo "      --output_star $EXPORT_STAR \\"
echo "      --output_angpix $OUTPUT_ANGPIX \\"
echo "      --box $SUBTOMO_BOX_SIZE \\"
echo "      --diameter $DIAMETER \\"
echo "      --relative_output_paths \\"
echo "      --3d \\"
echo "      --coords_angpix $COORDS_ANGPIX \\"
echo "      --output_ctf_csv \\"
echo "      --dont_correct_ctf_3d"
echo ""

"${WARP_DIR}/WarpTools" ts_export_particles \
--settings "$TILTSERIES_SETTINGS" \
--input_directory "$TM_WARP_DIR" \
--input_pattern "$INPUT_PATTERN" \
--output_star "$EXPORT_STAR" \
--output_angpix "$OUTPUT_ANGPIX" \
--box "$SUBTOMO_BOX_SIZE" \
--diameter "$DIAMETER" \
--relative_output_paths \
--3d \
--coords_angpix "$COORDS_ANGPIX" \
--output_ctf_csv \
--dont_correct_ctf_3d

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Export failed (exit code: $EXIT_CODE)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Export Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - STAR file: $EXPORT_STAR"
echo "  - Subtomograms: warp_tiltseries/subtomo/TS_XXX/*.mrc"
echo "  - CTF CSV: next to the exported subtomograms"
echo ""
echo "Next steps:"
echo "  1. Start OPUS-ET training:"
echo "     sbatch scripts/train_opuset.slurm"
echo "     If MASK_MRC is missing, train_opuset.slurm can create a default"
echo "     spherical mask from an exported subtomogram with dsdsh create_mask."
echo ""
echo "  2. Or manually create pose pickle:"
echo "     dsd parse_pose_star $EXPORT_STAR -D $SUBTOMO_BOX_SIZE --Apix $OUTPUT_ANGPIX -o particles_pose.pkl"
echo ""
echo "  3. For a custom mask, run scripts/gen_training_mask.slurm or set MASK_MRC."
echo "============================================================"
