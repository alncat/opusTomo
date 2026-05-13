#!/bin/bash
#SBATCH --job-name=warp_m_setup           # Job name
#SBATCH --partition=normal                # CPU partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --mem=16G                         # Memory per node
#SBATCH --time=00:30:00                   # Time limit (30 minutes)
#SBATCH --output=logs/warp_m_setup_%j.out      # Standard output log
#SBATCH --error=logs/warp_m_setup_%j.err       # Standard error log

# =============================================================================
# Step 1 of M refinement: create population + create source.
# One-shot per project (run once before any species are added).
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific overrides (uncomment to override pipeline.conf/species.conf defaults)
M_DIR="m"                                # MTools output dir

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
mkdir -p "$M_DIR"

POPULATION_FILE="${M_DIR}/${POPULATION_NAME}.population"

# -----------------------------------------------------------------------------
# 1. Create population (skip if already exists)
# -----------------------------------------------------------------------------
if [ -f "$POPULATION_FILE" ]; then
    echo "Population already exists: $POPULATION_FILE — skipping create_population."
else
    echo "Creating population: $POPULATION_NAME"
    MTools create_population \
    --directory "$M_DIR" \
    --name "$POPULATION_NAME"
fi

# -----------------------------------------------------------------------------
# 2. Create data source
# -----------------------------------------------------------------------------
echo "Creating source: $SOURCE_NAME (linked to $POPULATION_FILE)"
MTools create_source \
--name "$SOURCE_NAME" \
--population "$POPULATION_FILE" \
--processing_settings "$TILTSERIES_SETTINGS"

echo ""
echo "Done. Next: sbatch scripts/warp_m_create_species.slurm (per species)."
