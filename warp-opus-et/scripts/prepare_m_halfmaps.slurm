#!/bin/bash
#SBATCH --job-name=prep_halfmaps          # Job name
#SBATCH --partition=normal               # CPU partition
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=2                # CPU cores
#SBATCH --mem=16G                        # Memory
#SBATCH --time=00:30:00                  # Time limit
#SBATCH --output=logs/prep_halfmaps_%j.out
#SBATCH --error=logs/prep_halfmaps_%j.err

# =============================================================================
# Phase 8d: Prepare Half-Maps for M Refinement
# =============================================================================
# 1. Finds half-maps from fixed-mode training (fixed_*_subset*/tmp0.mrc)
# 2. Links them as opuset/TM_LABEL/half1.mrc and half2.mrc
# 3. Creates a mask from half1 using a user-specified density threshold
# =============================================================================

echo "============================================================"
echo "Prepare Half-Maps for M Refinement"
echo "Job ID: $SLURM_JOB_ID"
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

# Script-specific overrides
M_HALFMAP_THRESHOLD="${M_HALFMAP_THRESHOLD:-}"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

echo "Setting up environment..."

source ~/.bashrc
conda activate "$OPUSET_ENV"

cd "$WORK_DIR" || exit 1
echo "Working directory: $(pwd)"
echo ""

if ! command -v dsdsh >/dev/null 2>&1; then
    echo "ERROR: dsdsh not found after activating $OPUSET_ENV"
    exit 1
fi

# -----------------------------------------------------------------------------
# Find half-maps
# -----------------------------------------------------------------------------

HALFMAP_BASE="$WORK_DIR/opuset/$TM_LABEL"

echo "Searching for half-maps under: $HALFMAP_BASE"
echo ""

# Sort to get subset1 before subset2 (lexical ordering of fixed_sel*_subset*)
mapfile -t HALFMAPS < <(find "$HALFMAP_BASE" -path "*/fixed_*subset*/tmp0.mrc" -type f 2>/dev/null | sort)

if [ ${#HALFMAPS[@]} -lt 2 ]; then
    echo "ERROR: Found ${#HALFMAPS[@]} half-map(s), need at least 2."
    echo "  Searched pattern: */fixed_*subset*/tmp0.mrc under $HALFMAP_BASE"
    echo ""
    echo "  Available MRC files:"
    find "$HALFMAP_BASE" -name "tmp0.mrc" -o -name "*.mrc" 2>/dev/null | head -20
    exit 1
fi

echo "Found ${#HALFMAPS[@]} half-map(s):"
for i in "${!HALFMAPS[@]}"; do
    echo "  $((i+1)). ${HALFMAPS[$i]}"
done

# Use first two sorted results as half1 and half2
HALF1="${HALFMAPS[0]}"
HALF2="${HALFMAPS[1]}"

# -----------------------------------------------------------------------------
# Link half-maps
# -----------------------------------------------------------------------------

echo ""
echo "Linking half-maps to standardized names..."

ln -sf "$(realpath "$HALF1")" "$HALFMAP_BASE/half1.mrc"
ln -sf "$(realpath "$HALF2")" "$HALFMAP_BASE/half2.mrc"

echo "  half1.mrc → $HALF1"
echo "  half2.mrc → $HALF2"

# -----------------------------------------------------------------------------
# Create mask (optional — only if threshold is set)
# -----------------------------------------------------------------------------

if [ -n "$M_HALFMAP_THRESHOLD" ]; then
    MASK_OUT="$HALFMAP_BASE/halfmap_mask.mrc"

    echo ""
    echo "Creating mask from half1 with threshold=$M_HALFMAP_THRESHOLD..."

    dsdsh create_mask "$HALF1" \
        -o "$MASK_OUT" \
        --threshold "$M_HALFMAP_THRESHOLD" \
        


    if [ $? -ne 0 ] || [ ! -f "$MASK_OUT" ]; then
        echo "ERROR: Failed to create mask: $MASK_OUT"
        exit 1
    fi

    echo "  Mask: $MASK_OUT"
    ls -lh "$MASK_OUT"
else
    echo ""
    echo "WARNING: M_HALFMAP_THRESHOLD not set — skipping mask creation."
    echo "  Set it in species.conf and re-run, or create the mask manually:"
    echo "  dsdsh create_mask $HALFMAP_BASE/half1.mrc -o $HALFMAP_BASE/halfmap_mask.mrc --threshold <value> 
"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Half-Map Preparation Complete"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  half1.mrc: $HALFMAP_BASE/half1.mrc"
echo "  half2.mrc: $HALFMAP_BASE/half2.mrc"
if [ -n "$M_HALFMAP_THRESHOLD" ]; then
    echo "  mask:      $MASK_OUT"
fi
echo ""
echo "M_HALF1, M_HALF2, and M_MASK in species.conf default to these paths."
echo "  → Phase M is ready to submit:"
echo "     sbatch --export=ALL,SKILL_DIR=\"\$(pwd)/scripts\" scripts/warp_m_create_species.slurm"
echo "============================================================"
