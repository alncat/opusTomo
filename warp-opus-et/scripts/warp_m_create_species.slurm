#!/bin/bash
#SBATCH --job-name=warp_m_species         # Job name
#SBATCH --partition=normal                # CPU partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=64G                         # Memory per node
#SBATCH --time=02:00:00                   # Time limit (2 hours)
#SBATCH --output=logs/warp_m_species_%j.out    # Standard output log
#SBATCH --error=logs/warp_m_species_%j.err     # Standard error log

# =============================================================================
# Step 2 of M refinement: create_species.
# Run once per species you want to add to the population. Multiple species can
# coexist in one population — re-run this script with a different SPECIES_BASE.
# =============================================================================

# Do not use set -e here; input checks below report explicit failures.

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific values come from species.conf; override with environment vars if needed.
HALF1="${M_HALF1:-}"
HALF2="${M_HALF2:-}"
MASK="${M_MASK:-}"
PARTICLES_STAR="${M_PARTICLES_STAR:-}"
M_DIR="m"
TEMPORAL_SAMPLES=1

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source ~/.bashrc
conda activate "$WARP_ENV"
ulimit -v unlimited
export LD_LIBRARY_PATH="${CONDA_LIB}:${WARP_DIR}:$LD_LIBRARY_PATH"
export PATH="${WARP_DIR}:$PATH"
export WARP_FORCE_MRC_FLOAT32=1

cd "$WORK_DIR" || exit 1

POPULATION_FILE="${M_DIR}/${POPULATION_NAME}.population"
if [ ! -f "$POPULATION_FILE" ]; then
    echo "ERROR: population not found: $POPULATION_FILE"
    echo "       Run scripts/warp_m_setup.slurm first."
    exit 1
fi

for label in HALF1 HALF2 MASK PARTICLES_STAR; do
    value="${!label}"
    if [ -z "$value" ]; then
        echo "ERROR: $label is empty. Fill M_${label} in $SPECIES_CONF."
        exit 1
    fi
done

for f in "$HALF1" "$HALF2" "$MASK" "$PARTICLES_STAR"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done

if ! [[ "$ANGPIX_RESAMPLE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: ANGPIX_RESAMPLE must be a number (got: $ANGPIX_RESAMPLE)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Create species
# -----------------------------------------------------------------------------
echo "Creating species: $SPECIES_BASE"
echo "  diameter:  $DIAMETER A"
echo "  symmetry:  $SYM"
echo "  lowpass:   $LOWPASS A"
echo "  resample:  $ANGPIX_RESAMPLE A"

MTools create_species \
--population "$POPULATION_FILE" \
--name "$SPECIES_BASE" \
--diameter "$DIAMETER" \
--sym "$SYM" \
--temporal_samples "$TEMPORAL_SAMPLES" \
--half1 "$HALF1" \
--half2 "$HALF2" \
--mask "$MASK" \
--particles_relion "$PARTICLES_STAR" \
--angpix_resample "$ANGPIX_RESAMPLE" \
--lowpass "$LOWPASS"

# Resolve the WARP-appended hash suffix
SPECIES_DIR=$(find "${M_DIR}/species" -maxdepth 1 -type d -name "${SPECIES_BASE}_*" 2>/dev/null | head -1)
if [ -n "$SPECIES_DIR" ]; then
    echo ""
    echo "Species created at: $SPECIES_DIR"
    echo "  full species name: $(basename "$SPECIES_DIR")"
fi

echo ""
echo "Done. Next: sbatch scripts/warp_m_refine.slurm (with SPECIES_BASE=$SPECIES_BASE)."
