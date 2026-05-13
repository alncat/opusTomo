#!/bin/bash
#SBATCH --job-name=warp_m_mask            # Job name
#SBATCH --partition=normal                # CPU partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=00:30:00                   # Time limit (30 minutes)
#SBATCH --output=logs/warp_m_mask_%j.out       # Standard output log
#SBATCH --error=logs/warp_m_mask_%j.err        # Standard error log

# =============================================================================
# update_mask: regenerate the soft mask from the current half-map filtered
# reconstruction. Run ONLY BETWEEN refinement passes — it reads
# <SPECIES_BASE>_filt.mrc, which is produced by MCore, so at least one
# warp_m_refine.slurm pass must have completed first. Running it after the
# final pass has no effect since no further refinement consumes the new mask.
# After running this, submit scripts/warp_m_refine.slurm for the next pass.
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION — source shared pipeline.conf + species.conf, then override per-script values
# -----------------------------------------------------------------------------
SCRIPT_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/pipeline.conf" 2>/dev/null || { echo "ERROR: pipeline.conf not found. Set SKILL_DIR or copy pipeline.example.conf -> pipeline.conf and edit."; exit 1; }
SPECIES_CONF="${SPECIES_CONF:-${SCRIPT_DIR}/species.conf}"
source "$SPECIES_CONF" 2>/dev/null || { echo "ERROR: $SPECIES_CONF not found. Set SKILL_DIR or copy species.example.conf -> species.conf and edit."; exit 1; }

# Script-specific values come from species.conf; override with environment vars if needed.
THRESHOLD="${M_MASK_THRESHOLD:-}"        # density threshold (inspect _filt.mrc)
DILATE="2"                               # dilation in pixels (soft-edge padding)
M_DIR="m"
MAP_OVERRIDE=""                          # absolute path; "" = use resolved species prefix

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
if [ -d "${M_DIR}/species/${SPECIES_BASE}" ]; then
    SPECIES_DIRS=("${M_DIR}/species/${SPECIES_BASE}")
else
    mapfile -t SPECIES_DIRS < <(find "${M_DIR}/species" -maxdepth 1 -type d -name "${SPECIES_BASE}_*" 2>/dev/null | sort)
fi

if [ ${#SPECIES_DIRS[@]} -eq 0 ]; then
    echo "ERROR: species '$SPECIES_BASE' not found under ${M_DIR}/species/"
    echo "       Run scripts/warp_m_create_species.slurm first."
    exit 1
fi
if [ ${#SPECIES_DIRS[@]} -gt 1 ]; then
    echo "ERROR: SPECIES_BASE='$SPECIES_BASE' matched multiple species directories:"
    printf '  %s\n' "${SPECIES_DIRS[@]}"
    echo "       Set SPECIES_BASE to the full hashed directory name, e.g. SPECIES_BASE='ribosome_1199b7f2'."
    exit 1
fi

SPECIES_DIR="${SPECIES_DIRS[0]}"
SPECIES_NAME=$(basename "$SPECIES_DIR")
SPECIES_FILE="${SPECIES_DIR}/${SPECIES_NAME}.species"
if [ ! -f "$SPECIES_FILE" ]; then
    SPECIES_FILE="${SPECIES_DIR}/${SPECIES_BASE}.species"
fi
if [ -n "$MAP_OVERRIDE" ]; then
    MAP="$MAP_OVERRIDE"
else
    MAP="${SPECIES_DIR}/${SPECIES_NAME}_filt.mrc"
    if [ ! -f "$MAP" ]; then
        MAP="${SPECIES_DIR}/${SPECIES_BASE}_filt.mrc"
    fi
fi

if [ ! -f "$MAP" ]; then
    echo "ERROR: input map not found: $MAP"
    echo "       Either run a refinement pass first, or set MAP_OVERRIDE."
    exit 1
fi
if [ -z "$THRESHOLD" ]; then
    echo "ERROR: M_MASK_THRESHOLD is empty in $SPECIES_CONF."
    echo "       Inspect the filtered map and set a density threshold before updating the mask."
    exit 1
fi

echo "Updating mask for species: $(basename "$SPECIES_DIR")"
echo "  map:       $MAP"
echo "  threshold: $THRESHOLD"
echo "  dilate:    $DILATE"

MTools update_mask \
--population "$POPULATION_FILE" \
--species "$SPECIES_FILE" \
--map "$MAP" \
--threshold "$THRESHOLD" \
--dilate "$DILATE"

echo ""
echo "Done. Next: re-run scripts/warp_m_refine.slurm to refine with the new mask."
